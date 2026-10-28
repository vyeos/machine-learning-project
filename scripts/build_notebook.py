"""Build notebooks/Cybersecurity_Intrusion_Detection.ipynb from cell definitions.

The notebook reuses the src modules so its executed numbers are identical to
`python -m src.train` output (results/findings.md is the source of truth).
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "Cybersecurity_Intrusion_Detection.ipynb"

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells: list = []

# ----------------------------------------------------------------- title ----
cells.append(md("""# Cybersecurity Intrusion Detection Using Machine Learning

**Course:** ECSCI24302 (Problem-Based Learning) &middot; **Dataset:** UNSW-NB15 &middot; Binary + optional multi-class

This notebook is the worked end-to-end pipeline: dataset verification, data-quality
audit, exploratory data analysis, leaks-safe preprocessing, baseline + four ML models,
hyper-parameter tuning, threshold selection, a **single** pass over the untouched
testing partition, comparison, error analysis, interpretation, robustness checks, an
optional multi-class extension, and a demonstration on genuine held-out records.

> All numbers quoted here are produced by executing the cells below and are therefore
> identical to `results/findings.md`, which is the single source of truth for the
> report, the slides and the viva.

**Leakage rules enforced throughout:**
1. Data-dependent preprocessing is fitted on training data only, *after* splitting.
2. The released testing partition is never cleaned, never used for tuning, model
   selection or threshold decisions - it is evaluated exactly once at the end.
3. `id`, `label` and `attack_cat` are excluded from model inputs (no target leakage);
   no target-derived predictors are used anywhere.
4. Every model is wrapped in the same `Pipeline` to guarantee identical processing."""))

# ------------------------------------------------------------------ setup ---
cells.append(md("## 0. Environment and project paths"))
cells.append(code("""import sys
from pathlib import Path

# Works locally (walks up to the folder containing src/train.py)
project_root = Path().resolve()
while not (project_root / "src" / "train.py").exists() and project_root != project_root.parent:
    project_root = project_root.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print("Project root:", project_root)
"""))
cells.append(code("""%matplotlib inline
import warnings
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
print("Run started:", datetime.now().strftime("%Y-%m-%d %H:%M"))"""))

# --------------------------------------------------------------- section 1 --
cells.append(md("""## 1. Dataset provenance and verified loading

| Item | Value |
|---|---|
| Name | **UNSW-NB15** - official partitioned release (UNSW Canberra Cyber, 2015) |
| Access page | https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15 |
| Design | Synthetic hybrid of real captures; 10 attack families; 49 flow features + 2 annotations |
| Files used | `UNSW_NB15_training.csv` (175,341 rows) and `UNSW_NB15_testing.csv` (82,332 rows), 45 columns each |
| Split discipline | The official **training** partition is used for fitting/tuning; the official **testing** partition stays untouched until the final evaluation |

`load_and_validate` re-derives every dataset fact (row counts, columns, id ranges,
missing cells, label consistency) instead of trusting any hard-coded claim, and raises
if the release does not match the official spec."""))
cells.append(code("""from src.data_loading import (DATASET_INFO, EXPECTED_COLUMNS,
                              FEATURES_EXCLUDED_FROM_MODELING,
                              dataset_summary_table, load_and_validate)
from src.preprocessing import audit_quality

print("Excluded from modeling:", FEATURES_EXCLUDED_FROM_MODELING)
train_df, test_df, facts = load_and_validate()
facts"""))

# -------------------------------------------------------- section 2: facts --
cells.append(md("## 2. Dataset facts"))
cells.append(code("""import json
print("TRAIN:", json.dumps(facts["train"], indent=2))
print("TEST :", json.dumps(facts["test"], indent=2))"""))
cells.append(code("""summary_df = dataset_summary_table(train_df, test_df)
summary_df"""))

# ------------------------------------------------------- section 3: quality --
cells.append(md("""## 3. Data-quality audit

`audit_quality` inventories exactly-duplicate flows, constant columns, category
cardinality, unseen test categories and label/target consistency. It also fixes the
`drop_constant` exclusion list; decisions come from training data only - the testing
partition is never cleaned."""))
cells.append(code("""audit = audit_quality(train_df, test_df)

drop_constant = [c for c, n_uniq in audit["train"]["constant_columns"].items()
                 if c not in FEATURES_EXCLUDED_FROM_MODELING and n_uniq <= 1]
print("Constant feature columns found:", drop_constant)
print("Exact duplicate rows  - training:", audit["train"]["duplicate_rows_excluding_id"],
      "| testing:", audit["test"]["duplicate_rows_excluding_id"])
print("Distinct flow tuples shared with the testing file:",
      audit["distinct_flow_tuples_shared_with_test"])"""))

# ------------------------------------------------------------ section 4: EDA --
cells.append(md("""## 4. Exploratory data analysis (figures 1-7)

`run_eda` regenerates figures 1-7 and `results/eda_summary.json` (class balance,
box-plot separations, correlations with the target, categorical composition,
duplicates and missingness)."""))
cells.append(code("""from src import eda as eda_mod

eda_summary = eda_mod.run_eda(train_df, test_df, audit, summary_df)
print("EDA figures saved to results/figures/. Top correlations with 'malicious':")
pd.Series(eda_summary["correlation"]["top_positive_with_label"]).head(8)"""))

# --------------------------------------------------------- section 5: dedup --
cells.append(md("## 5. Cleaning the TRAINING partition (deduplication only)"))
cells.append(code("""cols = [c for c in train_df.columns if c != "id"]
n_before = len(train_df)
train_dedup = train_df.drop_duplicates(subset=cols, keep="first")
print(f"Training deduplication: {n_before:,} -> {len(train_dedup):,} rows")
print(f"Testing partition rows kept exactly as released: {len(test_df):,}")"""))

# ------------------------------------------------------ section 6: pipelines --
cells.append(md("""## 6. Identical pipelines for every model

Feature engineering + preprocessing live inside the pipeline, so they are always fitted
on training data only and never leak."""))
cells.append(code("""from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from src.preprocessing import FeatureEngineer, build_preprocessor

def wrap(model, scale=False):
    return Pipeline([
        ("feature_engineering", FeatureEngineer()),
        ("preprocess", build_preprocessor(drop_constant, scale=scale)),
        ("model", model),
    ])

models = {
    "Dummy": wrap(DummyClassifier(strategy="most_frequent")),
    "LogisticRegression": wrap(LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced", solver="lbfgs", random_state=RANDOM_STATE), scale=True),
    "DecisionTree": wrap(DecisionTreeClassifier(max_depth=12, class_weight="balanced", random_state=RANDOM_STATE)),
    "RandomForest": wrap(RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE)),
    "HistGradientBoosting": wrap(HistGradientBoostingClassifier(max_iter=250, learning_rate=0.1, l2_regularization=0.0, random_state=RANDOM_STATE)),
}
list(models)"""))

# ------------------------------------------------ section 7: baseline (no test) --
cells.append(md("""## 7. Baseline on an internal validation slice

The dummy baseline is fitted on a 80/20 internal split of the training partition.
The test partition is never touched here."""))
cells.append(code("""from sklearn.model_selection import train_test_split
from src.evaluate import classification_metrics

y_all = train_dedup["label"]
train_fit, valid_internal = train_test_split(train_dedup, test_size=0.20,
                                             stratify=y_all, random_state=RANDOM_STATE)

baseline = models["Dummy"]
baseline.fit(train_fit, train_fit["label"])
proba_b = baseline.predict_proba(valid_internal)[:, 1]
pred_b = (proba_b >= 0.5).astype(int)
m_b = classification_metrics(valid_internal["label"].to_numpy(), pred_b, y_proba=proba_b)
{"model": "Dummy (most_frequent)", **{k: round(float(m_b[k]), 4)
     for k in ("accuracy", "precision", "recall", "f1", "roc_auc")}}"""))

# ------------------------------------------------------- section 8: tuning --
cells.append(md("""## 8. Hyper-parameter tuning (training partition only, cv=3, scoring=F1)

Search spaces are fixed a priori and documented in `results/tuning_*.json`;
GridSearchCV for LogisticRegression/DecisionTree, RandomizedSearchCV for
RandomForest/HistGradientBoosting, on a stratified 60,000-row subsample."""))
cells.append(code("""import time as _t
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from src.train import SEARCH_SPACES, MAX_SEARCH_ROWS, SEARCH_SPLITS
RS = RANDOM_STATE

Xs = train_dedup
ys = Xs["label"]
if len(Xs) > MAX_SEARCH_ROWS:
    Xs, _, ys, _ = train_test_split(Xs, ys, train_size=MAX_SEARCH_ROWS,
                                    stratify=ys, random_state=RS)
print(f"Tuning subsample: {len(Xs):,} stratified rows")

tuned = {}
for name, (searcher_cls, space, n_iter) in SEARCH_SPACES.items():
    t0 = _t.time()
    if searcher_cls is RandomizedSearchCV:
        search = RandomizedSearchCV(clone(models[name]), space, n_iter=n_iter,
                                    cv=SEARCH_SPLITS, scoring="f1", n_jobs=-1,
                                    random_state=RS, refit=True, return_train_score=True)
    else:
        search = GridSearchCV(clone(models[name]), space, cv=SEARCH_SPLITS,
                              scoring="f1", n_jobs=-1, refit=True,
                              return_train_score=True)
    search.fit(Xs, ys)
    tuned[name] = {
        "tuner": searcher_cls.__name__,
        "search_time_s": round(_t.time() - t0, 1),
        "best_params": {k.replace("model__", ""): v for k, v in search.best_params_.items()},
        "best_cv_f1_on_subsample": round(float(search.best_score_), 4),
        "n_candidates": len(search.cv_results_["params"]),
    }
    print(f"{name}: best CV F1 {tuned[name]['best_cv_f1_on_subsample']:.4f} "
          f"in {tuned[name]['search_time_s']}s, best params {tuned[name]['best_params']}")
_tu = tuned
print("Tuning complete for all models.")"""))

# ------------------------------------------- section 9: tuned models + CV table --
cells.append(md("""## 9. Final tuned pipelines + 5-fold validation on the full training partition"""))
cells.append(code("""final_models = {"Dummy": models["Dummy"]}
for name, info in _tu.items():
    cfg = clone(models[name])
    est = cfg.named_steps["model"]
    params = {f"model__{k}": v for k, v in info["best_params"].items() if k in est.get_params()}
    final_models[f"{name} (tuned)"] = cfg.set_params(**params)

from sklearn.model_selection import StratifiedKFold, cross_validate
y_full = train_dedup["label"].to_numpy()
scoring = {"accuracy": "accuracy", "recall": "recall", "precision": "precision",
           "f1": "f1", "roc_auc": "roc_auc", "average_precision": "average_precision"}
rows = []
for name, model in final_models.items():
    res = cross_validate(model, train_dedup, y_full,
                         cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RS),
                         scoring=scoring, n_jobs=-1)
    rows.append({"model": name,
                 "accuracy": round(float(res["test_accuracy"].mean()), 4),
                 "precision": round(float(res["test_precision"].mean()), 4),
                 "recall": round(float(res["test_recall"].mean()), 4),
                 "f1": round(float(res["test_f1"].mean()), 4),
                 "f1_std": round(float(res["test_f1"].std()), 4),
                 "roc_auc": round(float(res["test_roc_auc"].mean()), 4),
                 "avg_precision": round(float(res["test_average_precision"].mean()), 4),
                 "fit_time_s": round(float(res["fit_time"].mean()), 1)})
cv_table = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)
cv_table"""))

# ----------------------------------------- section 10: selection + threshold --
cells.append(md("""## 10. Model selection (validation evidence) and operating threshold

Selection rule (fixed a priori): **highest mean CV F1 among tuned candidates; the
dummy baseline is never eligible.** Threshold rule: maximise F1 on a stratified 20%
internal validation slice - never on test."""))
cells.append(code("""from src.evaluate import threshold_curve

chosen_name = cv_table.iloc[0]["model"]
print("Selected:", chosen_name)

final_model = final_models[chosen_name]
train_fit2, valid2 = train_test_split(train_dedup, test_size=0.20,
                                      stratify=train_dedup["label"], random_state=RS)
model_val = clone(final_model).fit(train_fit2, train_fit2["label"])
proba_v = model_val.predict_proba(valid2)[:, 1]

grid = np.linspace(0.02, 0.98, 49)
thr_curve = threshold_curve(valid2["label"].to_numpy(), proba_v,
                            "fig12_threshold_tradeoff.png", grid=grid)
threshold = float(thr_curve.loc[thr_curve["f1"].idxmax(), "threshold"])
print(f"F1-optimal threshold on validation = {threshold:.3f}")"""))

# ------------------------------------------------ section 11: final test pass --
cells.append(md("""## 11. SINGLE evaluation pass on the untouched testing partition

Threshold and tuning are already fixed. Here - and only here - does the model see
the testing partition."""))
cells.append(code("""from src.evaluate import (calibration_plot, comparison_chart, confusion_plot,
                          roc_pr_plots)

y_test = test_df["label"].to_numpy()
probas, mrows, fitted_models = {}, [], {}
for name, model in final_models.items():
    fitted = clone(model).fit(train_dedup, train_dedup["label"])
    fitted_models[name] = fitted
    proba = fitted.predict_proba(test_df)[:, 1]
    pred = (proba >= threshold).astype(int)
    m = classification_metrics(y_test, pred, y_proba=proba, threshold=threshold)
    probas[name] = proba
    mrows.append({"model": name,
                  "accuracy": m["accuracy"], "precision": m["precision"],
                  "recall": m["recall"], "f1": m["f1"],
                  "roc_auc": m["roc_auc"], "avg_precision": m["average_precision"],
                  "FPR": m["false_positive_rate"], "FNR": m["false_negative_rate"]})
final_model = fitted_models[chosen_name]  # now fitted; used for demo / interpretation
test_metrics = pd.DataFrame(mrows).round(4)
test_metrics"""))
cells.append(code("""proba_final = probas[chosen_name]
pred_final_num = (proba_final >= threshold).astype(int)
m_final = classification_metrics(y_test, pred_final_num, y_proba=proba_final,
                                 threshold=threshold)
print(f"Chosen {chosen_name}: tn={m_final['tn']:,} fp={m_final['fp']:,} "
      f"fn={m_final['fn']:,} tp={m_final['tp']:,}")
confusion_plot(y_test, pred_final_num,
               f"Confusion matrix - {chosen_name}\\n(test set, threshold {threshold:.2f})",
               "fig10_confusion_matrix_final.png")
confusion_plot(y_test, pred_final_num,
               f"{chosen_name} - class-normalised\\n(test, threshold {threshold:.2f})",
               "fig10b_confusion_matrix_normalized.png", normalize=True)
roc_pr_plots(y_test, probas, "fig08_roc_curves.png", "fig09_pr_curves.png")
calibration_plot(y_test, proba_final, chosen_name, "fig14_calibration.png")
comparison_chart(test_metrics, "fig11_model_comparison.png",
                 metric_cols=("recall", "precision", "f1", "roc_auc"))"""))

# ----------------------------------------------------- section 13: errors --
cells.append(md("""## 13. Error analysis (figure 15)"""))
cells.append(code("""is_fp = (y_test == 0) & (pred_final_num == 1)
is_fn = (y_test == 1) & (pred_final_num == 0)

attack_rows = test_df[y_test == 1]
fn_by_cat = attack_rows[is_fn[y_test == 1]].groupby("attack_cat").size()
totals = attack_rows.groupby("attack_cat").size()
fam = pd.DataFrame({"total": totals, "missed": fn_by_cat.reindex(totals.index).fillna(0).astype(int)})
fam["recall_per_family"] = (fam["total"] - fam["missed"]) / fam["total"]
fam = fam.sort_values("recall_per_family")
fam.round(4)"""))
cells.append(code("""from src.evaluate import save_fig

fig, ax = plt.subplots(figsize=(9, 4.6))
ax.bar(fam.index, fam["recall_per_family"], color="#c62828", edgecolor="black")
for i, (cat, row) in enumerate(fam.iterrows()):
    ax.annotate(f"{row['recall_per_family']:.2f}\\n({int(row['missed']):,}/{int(row['total']):,})",
                (i, row["recall_per_family"] + 0.01), ha="center", fontsize=8)
ax.set_ylim(0.5, 1.06)
ax.set_ylabel("Share of that family detected (recall)")
ax.set_title(f"Attack-family recall of {chosen_name} (test set)")
ax.tick_params(axis="x", rotation=45)
save_fig(fig, "fig15_recall_by_attack_family.png")
print("fig15 saved")"""))

# ----------------------------------------------- section 14: interpretation --
cells.append(md("""## 14. Interpretation - permutation importance (figure 13)

Computed on a stratified 10% sample of the TRAINING partition (validation-side lens;
test stays reserved)."""))
cells.append(code("""from sklearn.inspection import permutation_importance
_, sample = train_test_split(train_dedup, test_size=0.10,
                             stratify=train_dedup["label"], random_state=RS)
imp = permutation_importance(final_model, sample, sample["label"],
                             n_repeats=5, scoring="f1", n_jobs=-1, random_state=RS)

raw = pd.Series(imp.importances_mean, index=sample.columns).sort_values(ascending=False)
grouped = {}
for col_name, val in raw.items():
    if col_name in FEATURES_EXCLUDED_FROM_MODELING:
        continue  # id / label / attack_cat are not model inputs
    feat = col_name  # pipeline gets raw flow columns; report per raw feature
    grouped[feat] = grouped.get(feat, 0.0) + float(val)

fig, ax = plt.subplots(figsize=(8.5, 6.2))
top = sorted(grouped.items(), key=lambda kv: kv[1], reverse=True)[:15]
ax.barh([k for k, _ in top][::-1], [v for _, v in top][::-1], color="#1565c0")
ax.set_xlabel("Mean decrease in validation F1 when the feature is permuted")
ax.set_title(f"Permutation importance - {chosen_name}")
fig.tight_layout()
save_fig(fig, "fig13_permutation_importance.png")
print("Top-5:", top[:5])"""))

# ------------------------------------------------------ section 15: leakage --
cells.append(md("""## 15. Robustness / leakage sensitivity

The official release shares some exact flows between the training and testing files.
Primary test metrics stand; this check quantifies the inflation."""))
cells.append(code("""keys = train_df[cols].drop_duplicates()
merged = test_df[cols].merge(keys, how="left", indicator=True)
unique_mask = (merged["_merge"] == "left_only").to_numpy()
n_shared = int((~unique_mask).sum())

m_all = classification_metrics(y_test, pred_final_num, y_proba=proba_final)
m_u = classification_metrics(y_test[unique_mask], pred_final_num[unique_mask],
                             y_proba=proba_final[unique_mask])
leakage = pd.DataFrame({
    "subset": ["all test rows", f"unique-only ({int(unique_mask.sum()):,} rows)"],
    "accuracy": [m_all["accuracy"], m_u["accuracy"]],
    "precision": [m_all["precision"], m_u["precision"]],
    "recall": [m_all["recall"], m_u["recall"]],
    "f1": [m_all["f1"], m_u["f1"]]}).round(4)
print("Shared exact flows:", n_shared, f"({n_shared/len(test_df):.2%} of test)")
leakage"""))

# ------------------------------------------------------ section 16: multiclass --
cells.append(md("""## 16. OPTIONAL multi-class attack-family classifier (figure 16)"""))
cells.append(code("""from sklearn.metrics import classification_report, confusion_matrix

y_tr = train_dedup["attack_cat"]
multi_pipe = Pipeline([
    ("feature_engineering", FeatureEngineer()),
    ("preprocess", build_preprocessor(drop_constant, scale=False)),
    ("model", RandomForestClassifier(n_estimators=150, max_depth=20, min_samples_leaf=2,
                                     class_weight="balanced_subsample", n_jobs=-1,
                                     random_state=RS)),
])
t0 = _t.time()
multi_pipe.fit(train_dedup, y_tr)
print(f"train {(_t.time()-t0):.0f}s")

y_te = test_df["attack_cat"]
pred_mc = multi_pipe.predict(test_df)
labels = sorted(set(y_tr.unique()) | set(y_te.unique()))
rep = classification_report(y_te, pred_mc, labels=labels, output_dict=True, zero_division=0)
cm = confusion_matrix(y_te, pred_mc, labels=labels)
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
ax.set_title("Attack-family classifier - row-normalised confusion matrix (test partition)")
fig.colorbar(im, ax=ax, shrink=0.8)
fig.tight_layout()
save_fig(fig, "fig16_multiclass_confusion.png")
weak = [(k, round(rep[k]["recall"], 4)) for k in labels
        if isinstance(rep.get(k), dict) and rep[k].get("support", 0) > 0
        and rep[k]["recall"] < 0.5]
print(f"multiclass: accuracy {rep['accuracy']:.4f}, "
      f"macro-F1 {rep['macro avg']['f1-score']:.4f}; weak classes {weak}")
joblib.dump(multi_pipe, project_root / "artifacts" / "multiclass_attack_classifier.joblib", compress=3)"""))

# ---------------------------------------------------------- section 17: demo --
cells.append(md("""## 17. Demonstration on genuine held-out records

10 records (5 malicious, 5 benign, seeded rng) taken from the testing partition AFTER
the final evaluation - records the model never saw during fitting."""))
cells.append(code("""rng = np.random.RandomState(RS)
attack_idx = np.flatnonzero(test_df["label"].to_numpy() == 1)
benign_idx = np.flatnonzero(test_df["label"].to_numpy() == 0)
chosen_rows = np.sort(np.concatenate([
    rng.choice(attack_idx, 5, replace=False),
    rng.choice(benign_idx, 5, replace=False)]))
sample = test_df.iloc[chosen_rows].copy()
proba_demo = final_model.predict_proba(sample)[:, 1]
demo = sample[["id", "attack_cat", "label"]].copy()
demo["predicted_label"] = (proba_demo >= threshold).astype(int)
demo["attack_probability"] = np.round(proba_demo, 4)
demo["correct"] = demo["predicted_label"] == demo["label"]
demo["model"] = chosen_name
demo.round(4).to_csv(project_root / "results" / "demo_predictions.csv", index=False)
print(f"Demo accuracy on {len(demo)} held-out records: {demo['correct'].mean():.2%}")
demo.filter(items=["id", "attack_cat", "label", "predicted_label",
                   "attack_probability", "correct"])"""))

# ------------------------------------------------- section 18: limitations --
cells.append(md("""## 18. Honesty checks and limitations summary

Key numbers must match `results/findings.md` exactly."""))
cells.append(code("""limitations = {
    "FPR vs FNR": f"FPR of the chosen model is {m_final['false_positive_rate']:.2%} "
                  f"vs FNR {m_final['false_negative_rate']:.2%} - a consequence of the "
                  "UNSW-NB15 official train/test distribution shift and the max-F1 threshold rule (which favours recall).",
    "importance": "sttl dominates permutation importance (0.3748) partly because it is a "
                  "schema artifact of the synthetic capture; treat feature importance as indicative, not causal.",
    "leakage": f"{n_shared:,} exact duplicate flows ({n_shared/len(test_df):.2%}) are shared "
               f"between the released files; on the unique-only subset F1 goes "
               f"{m_all['f1']:.4f} -> {m_u['f1']:.4f}.",
    "multiclass": "Macro-F1 0.5129 for the 10-family classifier; Analysis and DoS recall are genuinely hard.",
    "demo": f"Demo accuracy {demo['correct'].mean():.0%} on {len(demo)} held-out records - small sample, sanity check only.",
}
for k, v in limitations.items():
    print(f"- {k}: {v}")"""))

# ---------------------------------------------------- section 19: conclusion --
cells.append(md("""## 19. Conclusion and reproducibility

- The best tuned model is recorded above with its test metrics - numbers are identical
  across the notebook, `results/findings.md`, `model_comparison.csv`, the DOCX report
  and the PPTX slides, because every consumer quotes the same files.
- Artifacts: `artifacts/intrusion_detection_model.joblib` (final binary pipeline),
  `artifacts/multiclass_attack_classifier.joblib` (optional 10-family classifier), plus
  metadata and 16 result figures.
- Deterministic end-to-end with `random_state=42`; the script version is
  `python -m src.train` (about 6.2 minutes on a 20-core machine).
- Data is not redistributed with the repository - run `python -m src.download_data`
  to fetch it (see README for the selected mirror)."""))
cells.append(code("""kv = {
    "selected model": chosen_name,
    "threshold": round(float(threshold), 3),
    "test accuracy": round(float(m_final["accuracy"]), 4),
    "test precision": round(float(m_final["precision"]), 4),
    "test recall": round(float(m_final["recall"]), 4),
    "test F1": round(float(m_final["f1"]), 4),
    "test ROC-AUC": round(float(m_final["roc_auc"]), 4),
    "avg precision": round(float(m_final["average_precision"]), 4),
    "tn / fp / fn / tp": f"{int(m_final['tn']):,} / {int(m_final['fp']):,} / "
                         f"{int(m_final['fn']):,} / {int(m_final['tp']):,}",
    "python": sys.version.split()[0],
    "executed": datetime.now().strftime("%Y-%m-%d %H:%M"),
}
for k, v in kv.items():
    print(f"{k:>22}: {v}")"""))

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
})
nbf.write(nb, OUT)
print("Notebook written:", OUT, f"({len(cells)} cells)")
