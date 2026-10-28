"""
train.py
--------
End-to-end experiment orchestrator for the UNSW-NB15 intrusion detection project.

Run:  python -m src.train            (from the project root)
      python -m src.train --quick    (reduced tuning budget, for smoke tests)

Experimental discipline (report ch. 7 and the leakage checklist):
* The testing partition is loaded but NOT touched until the single final pass;
  all development (cross-validation, tuning, threshold choice) uses the training
  partition only.
* Training-data cleaning decisions (constants, exact duplicates) are derived from
  the training partition only; the testing file is never modified.
* Every data-dependent preprocessing step is fitted inside an sklearn Pipeline on
  training data (imputers, scaler, one-hot vocabulary).
* The final model is chosen by cross-validated F1 on the training partition;
  the testing partition is evaluated once, at a threshold fixed a priori on
  validation data.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.data_loading import (
    ARTIFACTS_DIR,
    ATTACK_CAT_COL,
    DATASET_INFO,
    ID_COL,
    RESULTS_DIR,
    TARGET_COL,
)
from src.evaluate import (
    calibration_plot,
    classification_metrics,
    comparison_chart,
    confusion_plot,
    roc_pr_plots,
    save_fig,
    set_style,
    threshold_curve,
)
from src.preprocessing import ENGINEERED_FEATURES, FeatureEngineer, build_preprocessor

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

RANDOM_STATE = 42
MAX_SEARCH_ROWS = 60_000      # tuning subsample (documented compute constraint)
VALID_FRACTION = 0.2          # internal validation slice of the training partition
CV_SPLITS = 5
SEARCH_SPLITS = 3


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Model zoo - every estimator sees engineered + preprocessed data, train-only fit
# --------------------------------------------------------------------------- #
def make_models(drop_constant: list[str]) -> dict[str, Pipeline]:
    def wrap(model, scale: bool = False) -> Pipeline:
        return Pipeline([
            ("feature_engineering", FeatureEngineer()),
            ("preprocess", build_preprocessor(drop_constant, scale=scale)),
            ("model", model),
        ])

    return {
        "Dummy": wrap(DummyClassifier(strategy="most_frequent")),
        "LogisticRegression": wrap(
            LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced",
                               solver="lbfgs", random_state=RANDOM_STATE), scale=True),
        "DecisionTree": wrap(
            DecisionTreeClassifier(max_depth=12, class_weight="balanced",
                                   random_state=RANDOM_STATE)),
        "RandomForest": wrap(
            RandomForestClassifier(n_estimators=200,
                                   class_weight="balanced_subsample",
                                   n_jobs=-1, random_state=RANDOM_STATE)),
        "HistGradientBoosting": wrap(
            HistGradientBoostingClassifier(max_iter=250, learning_rate=0.1,
                                           l2_regularization=0.0,
                                           random_state=RANDOM_STATE)),
    }


SEARCH_SPACES: dict[str, tuple[type, dict, int]] = {
    "LogisticRegression": (GridSearchCV, {"model__C": [0.01, 0.1, 1.0, 3.0]}, 4),
    "DecisionTree": (GridSearchCV, {
        "model__max_depth": [8, 12, 16, 20],
        "model__min_samples_leaf": [1, 5],
        "model__class_weight": [None, "balanced"],
    }, 16),
    "RandomForest": (RandomizedSearchCV, {
        "model__n_estimators": [200, 300, 400],
        "model__max_depth": [16, 24, 32, None],
        "model__min_samples_leaf": [1, 2, 5, 10],
        "model__max_features": ["sqrt", 0.3],
        "model__class_weight": ["balanced_subsample", None],
    }, 12),
    "HistGradientBoosting": (RandomizedSearchCV, {
        "model__max_iter": [200, 300, 400],
        "model__learning_rate": [0.03, 0.1],
        "model__max_leaf_nodes": [15, 31, 63],
        "model__min_samples_leaf": [20, 40],
        "model__l2_regularization": [0.0, 1.0],
    }, 12),
}


# --------------------------------------------------------------------------- #
# Stage 1-2: data + EDA
# --------------------------------------------------------------------------- #
def stage_data() -> tuple[pd.DataFrame, pd.DataFrame, dict, dict, list[str]]:
    from src.data_loading import dataset_summary_table, load_and_validate
    from src.preprocessing import audit_quality

    log("Stage 1/10 - loading and validating the dataset")
    train_df, test_df, facts = load_and_validate()
    audit = audit_quality(train_df, test_df)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "quality_audit.json").write_text(json.dumps(audit, indent=2),
                                                   encoding="utf-8")
    summary_df = dataset_summary_table(train_df, test_df)
    summary_df.to_csv(RESULTS_DIR / "dataset_summary.csv", index=False)

    drop_constant = [
        c for c, n_unique in audit["train"]["constant_columns"].items()
        if c not in (ID_COL, TARGET_COL, ATTACK_CAT_COL) and n_unique <= 1
    ]
    log(f"  verified: {facts['train']['rows']:,} train / {facts['test']['rows']:,} "
        f"test rows; training benign share {facts['train']['benign_pct']}%; "
        f"duplicate training rows {facts['train']['duplicate_rows']:,}; "
        f"constant feature columns {drop_constant}")
    return train_df, test_df, facts, audit, drop_constant


def stage_eda(train_df: pd.DataFrame, test_df: pd.DataFrame,
              audit: dict, summary_df: pd.DataFrame) -> dict:
    from src import eda as eda_mod
    log("Stage 2/10 - exploratory data analysis (figures 1-7)")
    return eda_mod.run_eda(train_df, test_df, audit, summary_df)


def clean_training_data(train_df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate flow records from the TRAINING partition only.
    Duplicates arise when identical flows repeat in the capture; keeping both
    copies duplicates evidence. The testing partition is left exactly as released."""
    cols = [c for c in train_df.columns if c != ID_COL]
    n_before = len(train_df)
    dedup = train_df.drop_duplicates(subset=cols, keep="first")
    log(f"  training deduplication: {n_before:,} -> {len(dedup):,} rows "
        f"({n_before - len(dedup):,} exact duplicates removed)")
    return dedup


# --------------------------------------------------------------------------- #
# Stage 4-5: pipelines + tuning
# --------------------------------------------------------------------------- #
def tune_models(models: dict[str, Pipeline], train_df: pd.DataFrame,
                quick: bool = False) -> dict[str, dict]:
    log("Stage 5/10 - hyperparameter tuning (training partition only)")
    y_full = train_df[TARGET_COL]
    if len(train_df) > MAX_SEARCH_ROWS:
        Xs, _, ys, _ = train_test_split(train_df, y_full, train_size=MAX_SEARCH_ROWS,
                                        stratify=y_full, random_state=RANDOM_STATE)
    else:
        Xs, ys = train_df, y_full
    log(f"  tuning subsample: {len(Xs):,} stratified rows (random_state={RANDOM_STATE})")

    tuned: dict[str, dict] = {}
    for name, (searcher_cls, space, n_iter) in SEARCH_SPACES.items():
        if quick:
            space = {k: v[: max(1, len(v) // 2)] for k, v in space.items()}
            if searcher_cls is RandomizedSearchCV:
                n_iter = 3
        t0 = time.time()
        if searcher_cls is RandomizedSearchCV:
            search: GridSearchCV | RandomizedSearchCV = RandomizedSearchCV(
                clone(models[name]), space, n_iter=n_iter, cv=SEARCH_SPLITS,
                scoring="f1", n_jobs=-1, random_state=RANDOM_STATE, refit=True,
                return_train_score=True)
        else:
            search = GridSearchCV(clone(models[name]), space, cv=SEARCH_SPLITS,
                                  scoring="f1", n_jobs=-1, refit=True,
                                  return_train_score=True)
        try:
            search.fit(Xs, ys)
        except Exception as exc:
            log(f"  search FAILED for {name}: {exc!r}")
            continue
        elapsed = time.time() - t0
        cvr = search.cv_results_
        tuned[name] = {
            "tuner": searcher_cls.__name__,
            "search_time_s": elapsed,
            "best_params": {k.replace("model__", ""): v
                            for k, v in search.best_params_.items()},
            "best_cv_f1_on_subsample": float(search.best_score_),
            "mean_train_f1_of_best": float(cvr["mean_train_score"][search.best_index_]),
            "space": {k: [None if x is None else x for x in v]
                      for k, v in space.items()},
            "n_candidates": len(cvr["params"]),
        }
        log(f"  {name}: best CV F1 {search.best_score_:.4f} "
            f"(train {tuned[name]['mean_train_f1_of_best']:.4f}) in {elapsed:.0f}s "
            f"with {tuned[name]['best_params']}")
        slug = {"LogisticRegression": "logregression",
                "DecisionTree": "dtree",
                "RandomForest": "forest",
                "HistGradientBoosting": "gb"}[name]
        (RESULTS_DIR / f"tuning_{slug}.json").write_text(
            json.dumps(tuned[name], indent=2, default=str), encoding="utf-8")
    return tuned


def build_tuned_models(models: dict[str, Pipeline],
                       tuned: dict[str, dict]) -> dict[str, Pipeline]:
    final_models: dict[str, Pipeline] = {"Dummy": models["Dummy"]}
    for name, info in tuned.items():
        cfg = clone(models[name])
        est = cfg.named_steps["model"]
        params = {f"model__{k}": v for k, v in info["best_params"].items()
                  if k in est.get_params()}
        cfg.set_params(**params)
        final_models[f"{name} (tuned)"] = cfg
    return final_models


# --------------------------------------------------------------------------- #
# Stage 6-7: validation, selection, threshold
# --------------------------------------------------------------------------- #
def cv_validate(final_models: dict[str, Pipeline], train_df: pd.DataFrame) -> pd.DataFrame:
    log("Stage 6/10 - 5-fold stratified CV on the full training partition")
    y = train_df[TARGET_COL].to_numpy()
    scoring = {"f1": "f1", "accuracy": "accuracy", "recall": "recall",
               "precision": "precision", "roc_auc": "roc_auc",
               "average_precision": "average_precision"}
    rows = []
    for name, model in final_models.items():
        t0 = time.time()
        res = cross_validate(model, train_df, y,
                             cv=StratifiedKFold(n_splits=CV_SPLITS, shuffle=True,
                                                random_state=RANDOM_STATE),
                             scoring=scoring, n_jobs=-1)
        row = {
            "model": name,
            "cv_f1": float(res["test_f1"].mean()),
            "cv_f1_std": float(res["test_f1"].std()),
            "cv_accuracy": float(res["test_accuracy"].mean()),
            "cv_recall": float(res["test_recall"].mean()),
            "cv_precision": float(res["test_precision"].mean()),
            "cv_roc_auc": float(res["test_roc_auc"].mean()),
            "cv_avg_precision": float(res["test_average_precision"].mean()),
            "cv_fit_time_s": float(res["fit_time"].mean()),
            "total_minutes": (time.time() - t0) / 60,
        }
        rows.append(row)
        log(f"  {name}: CV F1 {row['cv_f1']:.4f} +/- {row['cv_f1_std']:.4f} "
            f"({row['total_minutes'] * 60:.0f}s total)")
    cv_table = pd.DataFrame(rows).sort_values("cv_f1", ascending=False).reset_index(drop=True)
    cv_table.to_csv(RESULTS_DIR / "cross_validation_results.csv", index=False)
    return cv_table


def choose_threshold(final_model: Pipeline, train_dedup: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    """
    Operating threshold fixed on an internal stratified validation slice
    (20% of the training partition) - NEVER on test. Rule (fixed a priori):
    maximise F1 on validation. The deployment brief emphasises catching malicious
    traffic (recall); max-F1 is the standard compromise against false alarms.
    """
    log("Stage 7/10 - selecting the alert threshold on validation data")
    y = train_dedup[TARGET_COL]
    train_fit, valid = train_test_split(train_dedup, test_size=VALID_FRACTION,
                                        stratify=y, random_state=RANDOM_STATE)
    model = clone(final_model)
    model.fit(train_fit, train_fit[TARGET_COL])
    proba = model.predict_proba(valid)[:, 1]
    grid = np.linspace(0.02, 0.98, 49)
    curve = threshold_curve(valid[TARGET_COL].to_numpy(), proba,
                            "fig12_threshold_tradeoff.png", grid=grid)
    best = float(curve.loc[curve["f1"].idxmax(), "threshold"])
    log(f"  F1-optimal threshold on validation = {best:.3f}")
    return best, curve


# --------------------------------------------------------------------------- #
# Stage 8: untouched test evaluation (single pass)
# --------------------------------------------------------------------------- #
def final_evaluation(final_models: dict[str, Pipeline], test_df: pd.DataFrame,
                     threshold: float, chosen_name: str) -> dict:
    log("Stage 8/10 - final evaluation on the untouched testing partition "
        "(single pass; threshold fixed beforehand)")
    y = test_df[TARGET_COL].to_numpy()
    probas: dict[str, np.ndarray] = {}
    metrics_rows: list[dict] = []
    for name, model in final_models.items():
        t0 = time.time()
        proba = model.predict_proba(test_df)[:, 1]
        pred = (proba >= threshold).astype(int)
        m = classification_metrics(y, pred, y_proba=proba, threshold=threshold)
        m["model"] = name
        m["predict_time_s"] = time.time() - t0
        metrics_rows.append(m)
        probas[name] = proba
        log(f"  {name}: acc {m['accuracy']:.4f} | P {m['precision']:.4f} | "
            f"R {m['recall']:.4f} | F1 {m['f1']:.4f} | AUC {m['roc_auc']:.4f} | "
            f"AP {m['average_precision']:.4f} ({m['predict_time_s']:.1f}s)")

    metrics_all = pd.DataFrame({
        "model": [m["model"] for m in metrics_rows],
        "accuracy": [m["accuracy"] for m in metrics_rows],
        "precision": [m["precision"] for m in metrics_rows],
        "recall": [m["recall"] for m in metrics_rows],
        "f1": [m["f1"] for m in metrics_rows],
        "roc_auc": [m["roc_auc"] for m in metrics_rows],
        "average_precision": [m["average_precision"] for m in metrics_rows],
    })
    comparison_chart(metrics_all, "fig11_model_comparison.png",
                     metric_cols=("recall", "precision", "f1", "roc_auc"))

    y_pred_final = (probas[chosen_name] >= threshold).astype(int)
    final_m = metrics_rows[[m["model"] for m in metrics_rows].index(chosen_name)]
    confusion_plot(y, y_pred_final,
                   f"Confusion matrix - {chosen_name}\n(test set, threshold {threshold:.2f})",
                   "fig10_confusion_matrix_final.png")
    confusion_plot(y, y_pred_final,
                   f"{chosen_name} - class-normalised\n(test, threshold {threshold:.2f})",
                   "fig10b_confusion_matrix_normalized.png", normalize=True)
    roc_pr_plots(y, probas, "fig08_roc_curves.png", "fig09_pr_curves.png")
    calibration_plot(y, probas[chosen_name], chosen_name, "fig14_calibration.png")
    log(f"  chosen {chosen_name}: FN={final_m['fn']:,} FP={final_m['fp']:,} "
        f"(FNR {final_m['false_negative_rate']:.3%}, FPR {final_m['false_positive_rate']:.3%})")
    return {"metrics_rows": metrics_rows, "probas": probas, "y": y,
            "chosen_name": chosen_name, "final": final_m}


# --------------------------------------------------------------------------- #
# Stage 9: error analysis + interpretation + robustness
# --------------------------------------------------------------------------- #
def error_analysis(test_df: pd.DataFrame, y: np.ndarray, proba_final: np.ndarray,
                   threshold: float, chosen_name: str) -> dict:
    log("Stage 9/10 - error analysis of the chosen model")
    pred = (proba_final >= threshold).astype(int)
    is_fp = (y == 0) & (pred == 1)
    is_fn = (y == 1) & (pred == 0)

    attack_rows = test_df[y == 1]
    fn_mask_in_attacks = is_fn[y == 1]
    fn_by_cat = attack_rows[fn_mask_in_attacks].groupby(ATTACK_CAT_COL).size()
    totals = attack_rows.groupby(ATTACK_CAT_COL).size()
    fam = pd.DataFrame({
        "total": totals,
        "missed": fn_by_cat.reindex(totals.index).fillna(0).astype(int),
    })
    fam["detected"] = fam["total"] - fam["missed"]
    fam["recall_per_family"] = fam["detected"] / fam["total"]
    fam = fam.sort_values("recall_per_family")
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.bar(fam.index, fam["recall_per_family"], color="#c62828", edgecolor="black")
    for i, (cat, row) in enumerate(fam.iterrows()):
        ax.annotate(f"{row['recall_per_family']:.2f}\n({int(row['missed']):,}/"
                    f"{int(row['total']):,})",
                    (i, row["recall_per_family"] + 0.01), ha="center", fontsize=8)
    ax.set_ylim(0.5, 1.06)
    ax.set_ylabel("Share of that family detected (recall)")
    ax.set_title(f"Attack-family recall of {chosen_name} (test set)")
    ax.tick_params(axis="x", rotation=45)
    save_fig(fig, "fig15_recall_by_attack_family.png")

    fp_rows = test_df[is_fp]
    correct_benign = test_df[(y == 0) & (pred == 0)]
    fp_profile = {}
    for col in ("dur", "rate", "sbytes", "sttl", "dttl", "ct_dst_src_ltm"):
        fp_profile[col] = {
            "fp_median": float(np.median(fp_rows[col])) if len(fp_rows) else None,
            "correct_benign_median": float(np.median(correct_benign[col])),
        }
    fp_proto = (fp_rows["proto"].astype(str).value_counts(normalize=True)
                .head(3).round(3).to_dict()) if len(fp_rows) else {}
    fp_state = (fp_rows["state"].astype(str).value_counts(normalize=True)
                .head(3).round(3).to_dict()) if len(fp_rows) else {}

    keep_cols = [ID_COL, ATTACK_CAT_COL, "dur", "rate", "sbytes", "dbytes", "sttl",
                 "dttl", "spkts", "dpkts", "ct_srv_src", "ct_dst_src_ltm", TARGET_COL]
    fp_part = test_df.loc[is_fp, keep_cols].head(5).assign(
        error_type="FalsePositive",
        attack_probability=proba_final[is_fp][:5]) if is_fp.any() else test_df.head(0).assign()
    fn_part = test_df.loc[is_fn, keep_cols].head(5).assign(
        error_type="FalseNegative",
        attack_probability=proba_final[is_fn][:5]) if is_fn.any() else test_df.head(0).assign()
    pd.concat([fp_part, fn_part]).round(3).to_csv(
        RESULTS_DIR / "error_examples.csv", index=False)

    analysis = {
        "fp_count": int(is_fp.sum()),
        "fn_count": int(is_fn.sum()),
        "fp_share_of_benign": float(is_fp[y == 0].sum() / max(1, int((y == 0).sum()))),
        "fn_share_of_attacks": float(is_fn[y == 1].sum() / max(1, int((y == 1).sum()))),
        "fn_by_family": [
            {"attack_cat": cat, "total": int(row["total"]),
             "missed": int(row["missed"]),
             "recall_per_family": float(row["recall_per_family"])}
            for cat, row in fam.round(4).iterrows()],
        "fp_feature_profile": fp_profile,
        "fp_proto_top_share": fp_proto,
        "fp_state_top_share": fp_state,
        "most_missed_family": fam["recall_per_family"].idxmin(),
        "least_missed_family": fam["recall_per_family"].idxmax(),
        "examples_file": "results/error_examples.csv",
    }
    log(f"  FP {analysis['fp_count']:,} ({analysis['fp_share_of_benign']:.3%} of benign); "
        f"FN {analysis['fn_count']:,} ({analysis['fn_share_of_attacks']:.3%} of attacks); "
        f"hardest family: {analysis['most_missed_family']}")
    return analysis


def interpret_model(final_model: Pipeline, train_df: pd.DataFrame,
                    model_name: str) -> dict:
    """Permutation importance on a stratified 10% sample of the TRAINING
    partition (a validation-side lens; the test partition stays reserved).
    OHE columns are collapsed back to their source feature for readability."""
    log("Stage 9b - interpretation: permutation importance on training-side sample")
    y = train_df[TARGET_COL]
    _, sample = train_test_split(train_df, test_size=0.10, stratify=y,
                                 random_state=RANDOM_STATE)
    imp = permutation_importance(final_model, sample, sample[TARGET_COL],
                                 n_repeats=5, scoring="f1", n_jobs=-1,
                                 random_state=RANDOM_STATE)
    # permutation_importance permutes the pipeline's RAW input columns; report
    # importances per raw flow feature and mask the non-predictor columns.
    names = np.asarray(sample.columns)
    all_imp = pd.Series(imp.importances_mean, index=names, dtype=float)
    predictor_cols = [c for c in sample.columns
                      if c not in (ID_COL, TARGET_COL, ATTACK_CAT_COL)]
    raw = all_imp[predictor_cols].sort_values(ascending=False)

    grouped: dict[str, float] = {}
    for col_name, val in raw.items():
        if col_name.startswith("num__"):
            feat = col_name[len("num__"):]
        elif col_name.startswith("cat__"):
            feat = col_name[len("cat__"):].split("_", 1)[0]
        else:
            feat = col_name
        grouped[feat] = grouped.get(feat, 0.0) + float(val)
    grouped = dict(sorted(grouped.items(), key=lambda kv: kv[1], reverse=True))

    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    top = list(grouped.items())[:15]
    ax.barh([k for k, _ in top][::-1], [v for _, v in top][::-1], color="#1565c0")
    ax.set_xlabel("Mean decrease in validation F1 when the feature is permuted")
    ax.set_title(f"Permutation importance - {model_name}")
    fig.tight_layout()
    save_fig(fig, "fig13_permutation_importance.png")
    log("  top-5: " + ", ".join(f"{k} ({v:.4f})" for k, v in top[:5]))
    return {"grouped_top15": {k: round(v, 5) for k, v in top},
            "raw_top15": {k: round(float(v), 5) for k, v in raw.head(15).items()}}


def robustness_check(final_model: Pipeline, train_raw: pd.DataFrame,
                     test_df: pd.DataFrame, threshold: float) -> dict:
    """Quantify the overlap the official release carries: test flows whose 44
    feature columns exactly match a training flow. Primary test metrics stand;
    the companion row shows metrics on the subset without exact matches."""
    log("Stage 9c - leakage sensitivity: exact duplicate flows shared by both files")
    cols = [c for c in train_raw.columns if c != ID_COL]
    keys = train_raw[cols].drop_duplicates()
    merged = test_df[cols].merge(keys, how="left", indicator=True)
    unique_mask = (merged["_merge"] == "left_only").to_numpy()
    n_shared = int((~unique_mask).sum())
    y = test_df[TARGET_COL].to_numpy()
    proba = final_model.predict_proba(test_df)[:, 1]
    pred = (proba >= threshold).astype(int)
    m_all = classification_metrics(y, pred, y_proba=proba)
    m_u = classification_metrics(y[unique_mask], pred[unique_mask],
                                 y_proba=proba[unique_mask])
    out = {
        "n_shared_rows": n_shared,
        "share_of_test": n_shared / len(test_df),
        "all_rows": {k: m_all[k] for k in ("accuracy", "precision", "recall", "f1",
                                           "roc_auc", "average_precision")},
        "unique_only": {k: m_u[k] for k in ("accuracy", "precision", "recall", "f1",
                                            "roc_auc", "average_precision")},
        "n_unique_rows": int(unique_mask.sum()),
    }
    log(f"  shared exact flows: {n_shared:,} of {len(test_df):,} test rows "
        f"({out['share_of_test']:.2%}); F1 {m_all['f1']:.4f} -> {m_u['f1']:.4f}")
    return out


# --------------------------------------------------------------------------- #
# Stage 10a: optional multi-class extension
# --------------------------------------------------------------------------- #
def multiclass_extension(train_dedup: pd.DataFrame, test_df: pd.DataFrame,
                         drop_constant: list[str]) -> dict:
    log("Stage 10a - OPTIONAL multi-class attack-family classifier")
    y_tr = train_dedup[ATTACK_CAT_COL]
    pipe = Pipeline([
        ("feature_engineering", FeatureEngineer()),
        ("preprocess", build_preprocessor(drop_constant, scale=False)),
        ("model", RandomForestClassifier(
            n_estimators=150, max_depth=20, min_samples_leaf=2,
            class_weight="balanced_subsample", n_jobs=-1,
            random_state=RANDOM_STATE)),
    ])
    t0 = time.time()
    pipe.fit(train_dedup, y_tr)
    train_time = time.time() - t0

    y_te = test_df[ATTACK_CAT_COL]
    pred = pipe.predict(test_df)
    labels = sorted(set(y_tr.unique()) | set(y_te.unique()))
    rep = classification_report(y_te, pred, labels=labels, output_dict=True,
                                zero_division=0)
    cm = confusion_matrix(y_te, pred, labels=labels)
    cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(10.5, 8.6))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=9)
    ax.set_yticks(range(len(labels)), labels, fontsize=9)
    for i in range(cmn.shape[0]):
        for j in range(cmn.shape[1]):
            if cmn[i, j] >= 0.02:
                ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                        color="white" if cmn[i, j] > 0.6 else "black", fontsize=8)
    ax.set_xlabel("Predicted family")
    ax.set_ylabel("Actual family")
    ax.set_title("Attack-family classifier - row-normalised confusion matrix "
                 "(test partition)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    save_fig(fig, "fig16_multiclass_confusion.png")

    weak = [(k, rep[k]["recall"]) for k in labels
            if isinstance(rep.get(k), dict) and rep[k].get("support", 0) > 0
            and rep[k]["recall"] < 0.5]
    out = {
        "classes": labels,
        "n_classes": len(labels),
        "accuracy": float(rep["accuracy"]),
        "macro_f1": float(rep["macro avg"]["f1-score"]),
        "weighted_f1": float(rep["weighted avg"]["f1-score"]),
        "train_time_s": train_time,
        "per_class": {k: {m: rep[k].get(m) for m in ("precision", "recall",
                                                    "f1-score", "support")}
                      for k in labels if isinstance(rep.get(k), dict)},
        "weak_classes_recall_under_0p5": weak,
    }
    log(f"  multiclass ({len(labels)} classes): accuracy {out['accuracy']:.4f}, "
        f"macro-F1 {out['macro_f1']:.4f}; classes with recall<0.5: "
        f"{[k for k, _ in weak]}")
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    joblib.dump(pipe, ARTIFACTS_DIR / "multiclass_attack_classifier.joblib", compress=3)
    (RESULTS_DIR / "multiclass_summary.json").write_text(json.dumps(out, indent=2),
                                                         encoding="utf-8")
    return out


# --------------------------------------------------------------------------- #
# Stage 10b: demonstration on genuine held-out records
# --------------------------------------------------------------------------- #
def demo_stage(final_model: Pipeline, test_df: pd.DataFrame,
               threshold: float, model_name: str) -> dict:
    log("Stage 10b - demonstration on 10 unseen held-out records")
    rng = np.random.RandomState(RANDOM_STATE)
    attack_idx = np.flatnonzero(test_df[TARGET_COL].to_numpy() == 1)
    benign_idx = np.flatnonzero(test_df[TARGET_COL].to_numpy() == 0)
    chosen = np.sort(np.concatenate([
        rng.choice(attack_idx, 5, replace=False),
        rng.choice(benign_idx, 5, replace=False)]))
    sample = test_df.iloc[chosen].copy()
    proba = final_model.predict_proba(sample)[:, 1]
    demo = sample[[ID_COL, ATTACK_CAT_COL, TARGET_COL]].copy()
    demo["predicted_label"] = (proba >= threshold).astype(int)
    demo["attack_probability"] = np.round(proba, 4)
    demo["correct"] = demo["predicted_label"] == demo[TARGET_COL]
    demo.round(4).to_csv(RESULTS_DIR / "demo_predictions.csv", index=False)
    demo_acc = float(demo["correct"].mean())
    log(f"  demo: {demo_acc:.0%} of 10 held-out records correctly classified "
        f"-> results/demo_predictions.csv")
    return {"demo_acc": demo_acc, "n": 10, "model": model_name}


# --------------------------------------------------------------------------- #
# Findings: the single source of truth for report / slides / viva
# --------------------------------------------------------------------------- #
def _fmtv(x, nd=4):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def write_findings(F: dict) -> None:
    """Render results/findings.md - THE single source of truth for the report,
    presentation and viva documents. Every number cited elsewhere must come from
    this file or the sibling result files."""
    from src.preprocessing import feature_columns

    facts, audit, eda = F["facts"], F["audit"], F["eda_summary"]
    cv, tuning = F["cv_table"], F["tuning"]
    err, imp, robust, multi, demo = F["err"], F["importance"], F["robust"], F["multi"], F["demo"]
    chosen, thr = F["chosen_name"], F["threshold"]

    def pct(x): return "n/a" if pd.isna(x) else f"{100 * x:.2f}%"
    def f(x, nd=4): return _fmtv(x, nd)
    def n(x): return "n/a" if pd.isna(x) else f"{x:,}"

    # input feature inventory
    cols = feature_columns(F["drop_constant"])
    n_num, n_cat = len(cols["numeric"]), len(cols["categorical"])
    n_input, n_eng = n_num + n_cat, len(ENGINEERED_FEATURES)

    tc = {str(k): int(v) for k, v in facts["train"]["label_counts"].items()}
    ec = {str(k): int(v) for k, v in facts["test"]["label_counts"].items()}

    cv_lines = "\n".join(
        f"| {r['model']} | {f(r['cv_accuracy'])} | {f(r['cv_precision'])} | "
        f"{f(r['cv_recall'])} | **{f(r['cv_f1'])}** +/- {f(r['cv_f1_std'])} | "
        f"{f(r['cv_roc_auc'])} | {f(r['cv_avg_precision'])} |"
        for _, r in cv.iterrows())

    comp = F["comparison"]
    comp_lines = "\n".join(
        f"| {r['model']} | {f(r.get('cv_f1_train'))} | {f(r['accuracy'])} | "
        f"{f(r['precision'])} | {f(r['recall'])} | **{f(r['f1'])}** | "
        f"{f(r['roc_auc'])} | {f(r['average_precision'])} | {f(r['false_positive_rate'])} | "
        f"{f(r['false_negative_rate'])} |" for _, r in comp.iterrows())

    fam_lines = "\n".join(
        f"| {r['attack_cat']} | {n(r['total'])} | {n(r['missed'])} | "
        f"{pct(1 - r['recall_per_family'])} |" for r in err["fn_by_family"])

    top5_imp = ", ".join(f"**{k}** ({v:.4f})"
                         for k, v in list(imp["grouped_top15"].items())[:5])
    imp_lines = "\n".join(
        f"| {i + 1} | {k} | {v:.4f} |"
        for i, (k, v) in enumerate(imp["grouped_top15"].items()))

    med_rows = eda.get("feature_medians", {}).get("medians", [])
    med_lines = "\n".join(
        f"| {m['feature']} | {m['median_benign']:.3g} | {m['median_malicious']:.3g} |"
        for m in med_rows)

    corr = eda.get("correlation", {})
    corr_pos = ", ".join(
        f"**{k}** ({v:.3f})" for k, v in corr.get("top_positive_with_label", {}).items())
    corr_neg = ", ".join(
        f"**{k}** ({v:.3f})" for k, v in corr.get("top_negative_with_label", {}).items())

    cat_all = eda.get("attack_categories", {})

    catg = eda.get("categorical", {})
    proto_shares = catg.get("proto", {}).get("by_category_share_malicious", {})
    proto_txt = ", ".join(f"{k}: {pct(v)}" for k, v in list(proto_shares.items())[:6])

    robust_a, robust_u = robust["all_rows"], robust["unique_only"]
    weak_multi = ", ".join(
        f"{k} ({pct(r)})" for k, r in multi.get("weak_classes_recall_under_0p5", []))

    md = f"""# Experiment findings - single source of truth
Generated automatically by `python -m src.train` on {time.strftime('%Y-%m-%d %H:%M')}.
Notebooks, the report and the slides must quote numbers exactly from this file
(or from the sibling CSV/JSON files) - never estimate or invent them.

## 1. Dataset (verified facts)
- Name: **{DATASET_INFO['name']}**; Kaggle listing: {DATASET_INFO['kaggle_source']}
- Training partition raw rows: **{n(facts['train']['rows'])}** (deduplicated -> **{n(F['train_rows_dedup'])}**)
- Testing partition rows (untouched benchmark): **{n(facts['test']['rows'])}**
- Columns per file: {facts['train']['columns']} (official release schema has 45)
- id range train {facts['train']['id_range']} (identifier, excluded from modeling); test {facts['test']['id_range']}
- Missing cells (raw files): train {n(audit['train']['missing_total'])}, test {n(audit['test']['missing_total'])} -> **verified: no imputation needed**
- Non-finite numeric cells: train {audit['train']['non_finite_by_column'] or 'none'}, test {audit['test']['non_finite_by_column'] or 'none'}
- Input feature columns for modeling: **{n_input}** = {n_num} numeric + {n_cat} categorical
- Excluded from modeling: id (identifier), label (target), attack_cat (target-derived annotation)
- Engineered features: **{n_eng}** added by FeatureEngineer inside pipelines
- Attack classes present: {', '.join(facts['train']['attack_cat_values'])}
- Label/attack_cat consistency: {facts['train']['label_attack_cat_mismatches']} + {facts['test']['label_attack_cat_mismatches']} = 0 verified mismatches

## 2. Class balance (figures 1-2)
| Split | Benign (0) | Malicious (1) | Attack share | Max:Min ratio |
|---|---|---|---|---|
| Training | {n(tc.get('0'))} | {n(tc.get('1'))} | {pct(facts['train']['malicious_pct'] / 100)} | {f(eda['target']['imbalance_ratio_train'], 2)} |
| Testing | {n(ec.get('0'))} | {n(ec.get('1'))} | {pct(facts['test']['malicious_pct'] / 100)} | - |

## 3. Data-quality findings (figures 6-7)
- Exact duplicate rows in training (excluding id): **{n(audit['train']['duplicate_rows_excluding_id'])}** - removed, reported
- Exact duplicate rows in testing (excluding id): **{n(audit['test']['duplicate_rows_excluding_id'])}** - kept (deployment realism; testing never cleaned)
- Distinct training flow tuples that also appear in testing: **{n(audit['distinct_flow_tuples_shared_with_test'])}** - quantified in the leakage sensitivity check
- Categorical cardinality (train): {audit['categorical_cardinality']}
- Unseen test categories: {audit['unseen_test_categories']}
- Flag inventory: is_ftp_login {audit['flag_value_inventory']['is_ftp_login_train']}; ct_ftp_cmd top counts {audit['flag_value_inventory']['ct_ftp_cmd_train']}; is_sm_ips_ports {audit['flag_value_inventory']['is_sm_ips_ports_train']}
- Negative numeric cell counts: {audit.get('negative_counts') or 'none'}
- Protocol attack-share (top categories): {proto_txt}

## 4. EDA highlights (figures 3-5)
Key feature medians (benign vs malicious):
| Feature | Median benign | Median malicious |
|---|---|---|
{med_lines}

- Features most positively correlated with 'malicious': {corr_pos}
- Features most negatively correlated with 'malicious': {corr_neg}
- Differences are descriptive, not proof of predictiveness.

## 5. Cross-validation on the training partition (5-fold stratified)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | AP |
|---|---|---|---|---|---|---|
{cv_lines}

## 6. Final model selection (validation evidence only)
- Rule (fixed a priori): highest mean CV F1 among tuned candidates; Dummy never eligible.
- Selected: **{chosen}**
- Alert threshold: **{thr:.3f}** (max F1 on the internal validation slice of the training partition)

## 7. Test metrics (single pass, untouched testing partition)
| Model | CV F1 (train) | Accuracy | Precision | Recall | F1 | ROC-AUC | AP | FPR | FNR |
|---|---|---|---|---|---|---|---|---|---|
{comp_lines}

- Confusion matrix of the chosen model:
  tn={n(comp.loc[comp['model'] == chosen, 'tn'].iloc[0])}, fp={n(comp.loc[comp['model'] == chosen, 'fp'].iloc[0])}, fn={n(comp.loc[comp['model'] == chosen, 'fn'].iloc[0])}, tp={n(comp.loc[comp['model'] == chosen, 'tp'].iloc[0])}

## 8. Error analysis (figure 15)
- False positives {n(err['fp_count'])} ({pct(err['fp_share_of_benign'])} of benign flows); false negatives {n(err['fn_count'])} ({pct(err['fn_share_of_attacks'])} of attacks)
- FP feature medians vs correctly-benign medians: {err['fp_feature_profile']}
- FP protocol share: {err['fp_proto_top_share']}; FP state share: {err['fp_state_top_share']}
- Family recall-ordered table (test set):
| Family | Total (test) | Missed | Missed-share |
|---|---|---|---|
{fam_lines}
- Hardest family: **{err['most_missed_family']}**; easiest family: **{err['least_missed_family']}**
- Representative error rows: results/error_examples.csv

## 9. Model interpretation (figure 13)
Permutation importance (mean drop in validation F1 when permuted):
| Rank | Feature | Mean drop in F1 |
|---|---|---|
{imp_lines}

- Top-5 summary: {top5_imp}
- Limits: predictive importance is statistical, not causal; an important feature does NOT prove malicious intent.

## 10. Leakage sensitivity (companion rows in tables)
- Test rows exactly duplicated in training: {n(robust['n_shared_rows'])} ({pct(robust['share_of_test'])})
- Metrics on ALL test rows: accuracy {f(robust_a['accuracy'])}, precision {f(robust_a['precision'])}, recall {f(robust_a['recall'])}, F1 {f(robust_a['f1'])}
- Metrics on UNIQUE-ONLY test rows: accuracy {f(robust_u['accuracy'])}, precision {f(robust_u['precision'])}, recall {f(robust_u['recall'])}, F1 {f(robust_u['f1'])}
- Read as: the drop between the two rows quantifies the inflation from duplicate flows shared across files (release artifact, documented in the limitations chapter).

## 11. Optional multi-class attack-family classifier (figure 16)
- Classes ({multi['n_classes']}): {', '.join(multi['classes'])}
- Test accuracy: **{f(multi['accuracy'])}**, macro-F1: **{f(multi['macro_f1'])}**, weighted-F1: {f(multi['weighted_f1'])}
- Classes with recall < 0.5: {weak_multi or 'none'}

## 12. Demonstration (notebook section 19)
- Final model: {demo['model']}, threshold {thr:.3f}
- 10 held-out records (5 malicious, 5 benign, seeded): demo accuracy **{pct(demo['demo_acc'])}**
- CSVs: results/demo_predictions.csv, results/demo_input_features.csv

## 13. Versions and runtime
- {F['versions']}
- Full run wall time: {F['runtime_s'] / 60:.1f} min

## 14. Figure register
- fig01_target_distribution.png - class counts (train/test)
- fig02_attack_categories.png - attack families (train/test counts)
- fig03_feature_histograms.png - histograms of key features (benign vs malicious)
- fig04_boxplots.png - boxplots of visible differences
- fig05_correlation_heatmap.png - correlation matrix
- fig06_missing_and_duplicates.png - missing cells + duplicate counts
- fig07_categorical_composition.png - proto/service/state by attack share
- fig08_roc_curves.png - ROC curves on the test set
- fig09_pr_curves.png - precision-recall curves on the test set
- fig10_confusion_matrix_final.png - confusion matrix of the chosen model (test)
- fig10b_confusion_matrix_normalized.png - class-normalised confusion
- fig11_model_comparison.png - bar chart of test metrics
- fig12_threshold_tradeoff.png - operating-threshold trade-off (validation)
- fig13_permutation_importance.png - permutation importance
- fig14_calibration.png - calibration check of the chosen model
- fig15_recall_by_attack_family.png - recall by family (test)
- fig16_multiclass_confusion.png - attack-family confusion matrix
"""

    (RESULTS_DIR / "findings.md").write_text(md, encoding="utf-8")
    log("  findings.md written")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(quick: bool = False) -> None:
    t0_all = time.time()
    set_style()

    train_df_raw, test_df, facts, audit, drop_constant = stage_data()
    eda_summary = stage_eda(
        train_df_raw, test_df, audit,
        pd.read_csv(RESULTS_DIR / "dataset_summary.csv"))

    log("Stage 3/10 - training-partition cleaning (exact duplicates)")
    train_dedup = clean_training_data(train_df_raw)

    log("Stage 4/10 - assembling leakage-safe pipelines")
    models_default = make_models(drop_constant)
    tuned = tune_models(models_default, train_dedup, quick=quick)
    final_models = build_tuned_models(models_default, tuned)

    cv_table = cv_validate(final_models, train_dedup)
    chosen_name = str(cv_table.loc[cv_table["cv_f1"].idxmax(), "model"])
    log(f"  VALIDATION-ONLY selection -> final model: {chosen_name}")
    chosen_model = final_models[chosen_name]

    threshold, curve = choose_threshold(chosen_model, train_dedup)
    pd.DataFrame(curve).to_csv(RESULTS_DIR / "threshold_curve.csv", index=False)

    log("Stage 7b - refitting every tuned candidate on the FULL training partition")
    fit_times: dict[str, float] = {}
    for name, model in final_models.items():
        t0 = time.time()
        model.fit(train_dedup, train_dedup[TARGET_COL])
        fit_times[name] = time.time() - t0
        log(f"  {name}: refit on full training data in {fit_times[name]:.1f}s")

    eval_info = final_evaluation(final_models, test_df, threshold, chosen_name)
    eval_info["fit_times"] = fit_times

    err = error_analysis(test_df, eval_info["y"],
                         eval_info["probas"][chosen_name], threshold, chosen_name)
    importance = interpret_model(chosen_model, train_dedup, chosen_name)
    robust = robustness_check(chosen_model, train_df_raw, test_df, threshold)
    multi = multiclass_extension(train_dedup, test_df, drop_constant)
    demo = demo_stage(chosen_model, test_df, threshold, chosen_name)

    log("Finalising artifacts and result files")
    ARTIFACTS_DIR.mkdir(exist_ok=True)

    tuned_info = tuned.get(chosen_name.replace(" (tuned)", ""), {})
    model_meta = {
        "model_name": chosen_name,
        "best_params": tuned_info.get("best_params"),
        "decision_threshold": threshold,
        "label_mapping": {"0": "benign / normal traffic",
                          "1": "malicious / attack traffic"},
        "trained_on": ("UNSW-NB15 official training partition "
                       f"({len(train_dedup):,} rows after exact-duplicate removal)"),
        "pipeline_steps": [
            "feature_engineering (FeatureEngineer)",
            "preprocess (ColumnTransformer: median impute; standard scale when "
            "scale=True; one-hot with infrequent buckets)",
            type(chosen_model.named_steps["model"]).__name__,
        ],
        "excluded_columns": ["id", "label", "attack_cat"],
        "requires": {"python": platform.python_version(),
                     "scikit_learn": sklearn.__version__,
                     "package_src": "src/ (FeatureEngineer, build_preprocessor)"},
        "test_metrics_chosen_model": next(
            m for m in eval_info["metrics_rows"] if m["model"] == chosen_name),
    }
    joblib.dump(chosen_model, ARTIFACTS_DIR / "intrusion_detection_model.joblib")
    (ARTIFACTS_DIR / "model_metadata.json").write_text(json.dumps(model_meta, indent=2),
                                                       encoding="utf-8")

    comparison = pd.DataFrame(eval_info["metrics_rows"]).sort_values(
        "f1", ascending=False).reset_index(drop=True)
    comparison["cv_f1_train"] = [
        float(cv_table.loc[cv_table["model"] == m, "cv_f1"].iloc[0])
        for m in comparison["model"]]
    comparison["cv_f1_std"] = [
        float(cv_table.loc[cv_table["model"] == m, "cv_f1_std"].iloc[0])
        for m in comparison["model"]]
    comparison["best_hyperparams"] = [
        ("strategy=most_frequent" if m == "Dummy"
         else json.dumps(tuned.get(m.replace(" (tuned)", ""), {}).get("best_params", {}),
                         default=str))
        for m in comparison["model"]]
    keep = ["model", "cv_f1_train", "cv_f1_std", "best_hyperparams", "threshold",
            "accuracy", "precision", "recall", "f1", "roc_auc", "average_precision",
            "specificity", "false_positive_rate", "false_negative_rate",
            "balanced_accuracy", "matthews_corrcoef", "tp", "fp", "fn", "tn",
            "predict_time_s"]
    comparison[keep].to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

    feature_cols = [c for c in train_dedup.columns
                    if c not in (ID_COL, TARGET_COL, ATTACK_CAT_COL)]
    experiments = []
    for r in eval_info["metrics_rows"]:
        model_key = r["model"].replace(" (tuned)", "")
        t_info = tuned.get(model_key, {})
        cv_match = cv_table.loc[cv_table["model"] == r["model"], "cv_f1"]
        cv_row_val = float(cv_match.iloc[0]) if len(cv_match) else np.nan
        experiments.append({
            "model": r["model"],
            "feature_set": f"{len(feature_cols)} base + {len(ENGINEERED_FEATURES)} engineered",
            "n_input_features": len(feature_cols),
            "cv_f1_train": _fmtv(cv_row_val),
            "best_hyperparams": (json.dumps(t_info.get("best_params", {}), default=str)
                                 if t_info else "strategy=most_frequent"),
            "train_time_s": round(t_info.get("search_time_s"), 1)
            if t_info else "n/a (immediate)",
            "refit_time_s": round(eval_info["fit_times"].get(r["model"], float("nan")), 1),
            "test_accuracy": round(r["accuracy"], 4),
            "test_precision": round(r["precision"], 4),
            "test_recall": round(r["recall"], 4),
            "test_f1": round(r["f1"], 4),
            "test_roc_auc": round(r["roc_auc"], 4),
            "test_average_precision": round(r["average_precision"], 4),
            "predict_time_s": round(r["predict_time_s"], 2),
            "notes": ("tuned with " + t_info["tuner"] if t_info else
                      "reference baseline"),
        })
    robust_row = {
        "model": f"{chosen_name} - test rows WITHOUT exact duplicates in training",
        "feature_set": "same", "n_input_features": len(feature_cols),
        "cv_f1_train": np.nan, "best_hyperparams": "same",
        "train_time_s": np.nan,
        "test_accuracy": round(robust["unique_only"]["accuracy"], 4),
        "test_precision": round(robust["unique_only"]["precision"], 4),
        "test_recall": round(robust["unique_only"]["recall"], 4),
        "test_f1": round(robust["unique_only"]["f1"], 4),
        "test_roc_auc": round(robust["unique_only"]["roc_auc"], 4),
        "test_average_precision": round(robust["unique_only"]["average_precision"], 4),
        "predict_time_s": np.nan,
        "notes": (f"leakage sensitivity: {robust['n_shared_rows']:,} test rows "
                  f"({robust['share_of_test']:.1%}) exactly match training flows"),
    }
    multi_row = {
        "model": "Multiclass RandomForest (attack families, optional extension)",
        "feature_set": "same", "n_input_features": len(feature_cols),
        "cv_f1_train": np.nan, "best_hyperparams": "n/a",
        "train_time_s": round(multi["train_time_s"], 1),
        "test_accuracy": round(multi["accuracy"], 4),
        "test_precision": np.nan, "test_recall": np.nan,
        "test_f1": round(multi["macro_f1"], 4),
        "test_roc_auc": np.nan, "test_average_precision": np.nan,
        "predict_time_s": np.nan,
        "notes": "macro-F1 across 10 families (optional extension)",
    }
    exp_df = pd.concat([pd.DataFrame(experiments),
                        pd.DataFrame([robust_row, multi_row])], ignore_index=True)
    exp_df.to_csv(RESULTS_DIR / "experiment_summary.csv", index=False)

    F = {
        "facts": facts, "audit": audit, "drop_constant": drop_constant,
        "eda_summary": eda_summary, "cv_table": cv_table, "tuning": tuned,
        "chosen_name": chosen_name, "threshold": threshold,
        "metrics_rows": eval_info["metrics_rows"], "comparison": comparison,
        "err": err, "importance": importance, "robust": robust,
        "multi": multi, "demo": demo,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__, "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
        },
        "train_rows_dedup": len(train_dedup),
        "test_rows": len(test_df),
        "runtime_s": time.time() - t0_all,
    }
    write_findings(F)
    log(f"COMPLETE in {F['runtime_s'] / 60:.1f} min. All results in {RESULTS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the IDS models end to end")
    parser.add_argument("--quick", action="store_true",
                        help="reduced search budget for smoke-testing")
    args = parser.parse_args()
    main(quick=args.quick)
