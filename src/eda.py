"""
eda.py
------
Exploratory data analysis for UNSW-NB15: moves from verified facts to plots,
each saved to results/figures and paired with a machine-readable summary in
results/eda_summary.json so that notebook / report / slides all use one truth.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data_loading import (
    ATTACK_CAT_COL,
    ID_COL,
    TARGET_COL,
    FIGURES_DIR,
    RESULTS_DIR,
)
from src.evaluate import save_fig, set_style
from src.preprocessing import SKEWED_NUMERIC

PALETTE = {0: "#2e7d32", 1: "#c62828"}  # 0 benign, 1 malicious
LABEL_NAMES = {0: "Benign (0)", 1: "Malicious (1)"}


# --------------------------------------------------------------------------- #
# 1. Target distribution + class imbalance
# --------------------------------------------------------------------------- #
def plot_target_distribution(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    for ax, df, name in zip(axes, (train_df, test_df), ("Training set", "Testing set")):
        counts = df[TARGET_COL].value_counts().reindex([0, 1])
        bars = ax.bar([LABEL_NAMES[0], LABEL_NAMES[1]], counts.values,
                      color=[PALETTE[0], PALETTE[1]], edgecolor="black")
        for b, v in zip(bars, counts.values):
            ax.annotate(f"{v:,}\n({v / counts.sum():.1%})", (b.get_x() + b.get_width() / 2, v),
                        ha="center", va="bottom", fontsize=9)
        ax.set_title(f"{name} - class counts")
        ax.set_ylabel("Number of flow records")
        ax.set_ylim(0, counts.max() * 1.18)
        ax.set_xlim(-0.55, 1.55)
    fig.suptitle("Benign vs malicious records (target = label)", y=1.03, fontsize=13)
    save_fig(fig, "fig01_target_distribution.png")

    summary = {
        "train_counts": {int(k): int(v) for k, v in train_df[TARGET_COL].value_counts().items()},
        "test_counts": {int(k): int(v) for k, v in test_df[TARGET_COL].value_counts().items()},
        "train_attack_share": float(train_df[TARGET_COL].mean()),
        "test_attack_share": float(test_df[TARGET_COL].mean()),
        "imbalance_ratio_train": float(
            train_df[TARGET_COL].value_counts().max() / train_df[TARGET_COL].value_counts().min()
        ),
    }
    return summary


# --------------------------------------------------------------------------- #
# 2. Attack-category distribution (multi-class preview)
# --------------------------------------------------------------------------- #
def plot_attack_categories(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    order = (
        train_df[ATTACK_CAT_COL].value_counts().index.tolist()
    )
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ct = pd.crosstab(train_df[ATTACK_CAT_COL], train_df[TARGET_COL]).loc[order]
    ct.columns = ["Benign", "Malicious"]
    ct.plot(kind="bar", stacked=True, color=[PALETTE[0], PALETTE[1]], ax=ax,
            edgecolor="white", width=0.8)
    ax.set_title("Attack-category composition (training set)")
    ax.set_xlabel("")
    ax.set_ylabel("Records")
    ax.legend()
    test_counts = test_df[ATTACK_CAT_COL].value_counts()
    summary = {
        "train_counts": {k: int(v) for k, v in train_df[ATTACK_CAT_COL].value_counts().items()},
        "test_counts": {k: int(v) for k, v in test_counts.items()},
        "train_share_normal": float((train_df[ATTACK_CAT_COL].str.lower() == "normal").mean()),
    }
    save_fig(fig, "fig02_attack_categories.png")
    return summary


# --------------------------------------------------------------------------- #
# 3. Numerical distributions (histogram grid of key features)
# --------------------------------------------------------------------------- #
KEY_FEATURES = ["dur", "rate", "sbytes", "dbytes", "sttl", "dttl", "sload", "dload",
                "swin", "dwin", "tcprtt", "ct_dst_src_ltm", "ct_srv_src", "ct_state_ttl"]


def plot_feature_histograms(df: pd.DataFrame, log_scale_cols=("dur", "sbytes", "dbytes",
                                                              "sload", "dload", "rate")) -> dict:
    n = len(KEY_FEATURES)
    fig, axes = plt.subplots(n // 2 if n >= 8 else 1, 2, figsize=(11.5, 4.2 * (n // 6)))
    axes = np.atleast_1d(axes).ravel()
    for ax, col in zip(axes, KEY_FEATURES):
        sub = df[[col, TARGET_COL]]
        if col in log_scale_cols:
            x = np.log1p(sub[col].clip(lower=0))
            ax.set_xlabel(f"log1p({col})")
        else:
            x = sub[col]
            ax.set_xlabel(col)
        ax.hist(x[sub[TARGET_COL] == 0], bins=45, alpha=0.65,
                color=PALETTE[0], density=True, label="Benign")
        ax.hist(x[sub[TARGET_COL] == 1], bins=45, alpha=0.55,
                color=PALETTE[1], density=True, label="Malicious")
        med_b = float(sub.loc[sub[TARGET_COL] == 0, col].median())
        med_m = float(sub.loc[sub[TARGET_COL] == 1, col].median())
        ax.set_title(f"{col}  (median benign={med_b:.3g} | malicious={med_m:.3g})", fontsize=10)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=PALETTE[0]), plt.Rectangle((0, 0), 1, 1, fc=PALETTE[1])]
    axes[0].legend(handles, ["Benign", "Malicious"], fontsize=8, loc="best")
    fig.suptitle("Feature distributions: benign vs malicious traffic", y=1.02, fontsize=13)
    fig.tight_layout()
    save_fig(fig, "fig03_feature_histograms.png")

    # quick median table for the report (rounded, readable)
    rows = []
    for col in KEY_FEATURES:
        rows.append({
            "feature": col,
            "median_benign": float(df.loc[df[TARGET_COL] == 0, col].median()),
            "median_malicious": float(df.loc[df[TARGET_COL] == 1, col].median()),
        })
    return {"medians": rows}


# --------------------------------------------------------------------------- #
# 4. Boxplots for a handful of features (visible with normal axes)
# --------------------------------------------------------------------------- #
BOX_FEATURES = ["dur", "rate", "sbytes", "sttl", "dttl", "swin", "ct_dst_src_ltm", "ct_srv_src"]


def plot_box(df: pd.DataFrame) -> dict:
    long = df.melt(id_vars=TARGET_COL, value_vars=BOX_FEATURES,
                   var_name="feature", value_name="value")
    fig, axes = plt.subplots(2, 4, figsize=(12.5, 6.4))
    for ax, col in zip(axes.ravel(), BOX_FEATURES):
        sub = long[long["feature"] == col]
        q1, q99 = sub["value"].quantile([0.01, 0.99])
        sns.boxplot(data=sub, x=TARGET_COL, y="value", hue=TARGET_COL, ax=ax,
                    palette=PALETTE, legend=False, showfliers=False, width=0.55)
        ax.set(ylim=(q1 - 0.05 * (q99 - q1 + 1e-9), q99 * 1.05))
        ax.set_xlabel("")
        ax.set_xticks([0, 1], ["Benign", "Malicious"])
        ax.set_title(col, fontsize=10)
    fig.suptitle("Boxplots (1st-99th percentile visible): benign vs malicious", y=1.02, fontsize=13)
    fig.tight_layout()
    save_fig(fig, "fig04_boxplots.png")
    return {"features": BOX_FEATURES}


# --------------------------------------------------------------------------- #
# 5. Correlation heatmap (numeric subset incl. engineered features)
# --------------------------------------------------------------------------- #
def correlation_with_target(df: pd.DataFrame) -> pd.Series:
    num = df.select_dtypes(include=[np.number]).drop(columns=[TARGET_COL, ID_COL],
                                                    errors="ignore")
    corr = num.corr(numeric_only=True)
    return corr[TARGET_COL].sort_values(key=abs, ascending=False)


def plot_correlation(df: pd.DataFrame) -> dict:
    corr = df.select_dtypes(include=[np.number]).corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(15.5, 12.5))
    sns.heatmap(corr, mask=mask, ax=ax, cmap="RdBu_r", vmin=-1, vmax=1,
                square=True, linewidths=0.4,
                cbar_kws={"shrink": 0.75, "label": "Pearson correlation"},
                annot=corr.shape[0] <= 30,
                fmt=".1f", annot_kws={"size": 6.5})
    ax.set_title("Correlation of numerical features (Pearson)")
    fig.tight_layout()
    save_fig(fig, "fig05_correlation_heatmap.png")

    labels = df[TARGET_COL]
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                if c not in (TARGET_COL, ID_COL)]
    association = {}
    for c in num_cols:
        values = df[c].to_numpy(dtype=float)
        ok = np.isfinite(values)
        if ok.sum() > 2 and 1 < df[c].nunique() <= target_unique_limit(c, df):
            association[c] = float(np.corrcoef(values[ok], labels.to_numpy()[ok])[0, 1])
    top = sorted(association.items(), key=lambda kv: abs(kv[1]), reverse=True)[:15]
    return {
        "top_positive_with_label": dict([t for t in top if t[1] > 0][:8]),
        "top_negative_with_label": dict([t for t in top if t[1] < 0][:8]),
        "top_15_abs": {k: round(v, 4) for k, v in top},
    }


def target_unique_limit(col: str, df: pd.DataFrame) -> int:
    """Treat high-cardinality numeric columns as continuous for the corrinsight scan."""
    return int(max(20, 0.5 * len(df)))


# --------------------------------------------------------------------------- #
# 6. Missing values + duplicates visual summary
# --------------------------------------------------------------------------- #
def plot_missing_and_duplicates(summary_df: pd.DataFrame, audit: dict) -> dict:
    """Audit picture: missing cells per column (both files) + duplicate counts."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))

    missing = summary_df[["column", "non_missing_train", "non_missing_test"]].set_index("column")
    miss_train = len(summary_df) * [0]
    rows = summary_df["column"].tolist()
    miss_train_counts = summary_df["non_missing_train"] - audit["train"]["rows"]
    miss_test_counts = summary_df["non_missing_test"] - audit["test"]["rows"]
    x = np.arange(len(rows))
    axes[0].bar(x - 0.2, miss_train_counts, width=0.4, label="Training", color="#1565c0")
    axes[0].bar(x + 0.2, miss_test_counts, width=0.4, label="Testing", color="#ef6c00")
    axes[0].set_xticks(x, rows, rotation=90, fontsize=7)
    axes[0].set_ylabel("Missing cells")
    axes[0].set_title("Missing values per column (verified, raw files)")
    axes[0].legend(fontsize=8)
    ymax = max(1, int(max(miss_train_counts.max(), miss_test_counts.max())))
    axes[0].set_ylim(0, ymax * 1.4)

    dups = {
        "Exact duplicates\n(training)": audit["train"]["duplicate_rows_excluding_id"],
        "Exact duplicates\n(testing)": audit["test"]["duplicate_rows_excluding_id"],
        "Train rows that also\nappear in testing": audit["distinct_flow_tuples_shared_with_test"],
    }
    colors = ["#ef6c00", "#6a1b9a", "#c62828"]
    bars = axes[1].bar(dups.keys(), dups.values(), color=colors, edgecolor="black")
    for b, v in zip(bars, dups.values()):
        axes[1].annotate(f"{v:,}", (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom")
    axes[1].set_ylabel("Duplicate flow records (excluding id)")
    axes[1].set_title("Duplicate analysis")
    fig.tight_layout()
    save_fig(fig, "fig06_missing_and_duplicates.png")
    return {
        "missing_cells_total": int(miss_train_counts.sum() + miss_test_counts.sum()),
        "duplicates": dups,
    }


# --------------------------------------------------------------------------- #
# 7. Protocol / service / state composition by class
# --------------------------------------------------------------------------- #
def plot_categorical_composition(train_df: pd.DataFrame) -> dict:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
    insights = {}
    for ax, col in zip(axes, ("proto", "service", "state")):
        top = train_df[col].value_counts().head(8).index.tolist()
        ct = train_df.loc[train_df[col].isin(top)].groupby(col)[TARGET_COL].agg(
            ["count", "mean"])
        ct = ct.sort_values("count", ascending=False)
        bars = ax.bar(ct.index, ct["count"], color="#1565c0", edgecolor="black")
        ax2 = ax.twinx()
        ax2.plot(ct.index, ct["mean"], "o--", color="#c62828", label="share malicious")
        ax2.set_ylim(0, 1)
        if ax is axes[2]:
            ax2.set_ylabel("Share malicious", color="#c62828")
            ax2.legend(fontsize=8, loc="upper center")
        ax.set_title(f"{col} (top-8 categories)", fontsize=10)
        ax.set_ylabel("Records")
        ax.tick_params(axis="x", rotation=45)
        insights[col] = {
            "by_category_share_malicious": {k: round(float(v), 4)
                                            for k, v in ct["mean"].items()},
            "category_value_counts": {k: int(v) for k, v in ct["count"].items()},
        }
    fig.suptitle("Categorical composition and attack share (training set)", y=1.03)
    fig.tight_layout()
    save_fig(fig, "fig07_categorical_composition.png")
    return insights


# --------------------------------------------------------------------------- #
# Run everything
# --------------------------------------------------------------------------- #
def run_eda(train_df: pd.DataFrame, test_df: pd.DataFrame,
            audit: dict, summary_df: pd.DataFrame) -> dict:
    set_style()
    out = {}
    out["target"] = plot_target_distribution(train_df, test_df)
    out["attack_categories"] = plot_attack_categories(train_df, test_df)
    out["feature_medians"] = plot_feature_histograms(train_df)
    out["boxplots"] = plot_box(train_df)
    out["correlation"] = plot_correlation(train_df)
    out["quality_visual_summary"] = plot_missing_and_duplicates(summary_df, audit)
    out["categorical"] = plot_categorical_composition(train_df)
    (RESULTS_DIR / "eda_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out
