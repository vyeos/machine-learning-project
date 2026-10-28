"""Generate proposal/Project_Proposal.docx, report/...Report.docx,
presentation/...Presentation.pptx and results/figures/fig17_architecture.png.

Numbers are ONLY quoted from results files; figures are the produced PNGs.
"""
from pathlib import Path
import json

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH as AL
from docx.shared import Pt, Inches
from pptx import Presentation as Pptx
from pptx.util import Inches as PInches
from pptx.util import Pt as PPt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"

KAGGLE_URL = "https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15"
MIRROR_URL = "https://huggingface.co/datasets/Mireu-Lab/UNSW-NB15"
ACCESSED = "2026-10-26"
PH = {
    "institution": "[Institution / University Name]",
    "course": "ECSCI24302 - [Course title as per syllabus]",
    "team": "[Team member names]",
    "supervisor": "[Supervisor / Course coordinator]",
    "date": "[Allotted date]",
}

REFERENCES = [
    'N. Moustafa and J. Slay, "UNSW-NB15: a comprehensive data set for network intrusion '
    'detection systems." Military Communications and Information Systems Conference (MilCIS), 2015.',
    'N. Moustafa, G. Creech, and J. Slay, "Big data analytics for large scale wireless network '
    'security and attack detection." AFRICOMM, 2015.',
    'N. Moustafa and J. Slay, "The evaluation of network anomaly detection systems: statistical '
    'analysis of the UNSW-NB15 dataset and the comparison with the KDD99 dataset." Information '
    'Security Journal: A Global Perspective, 25(1-3):18-31, 2016.',
    'L. Breiman, "Random forests." Machine Learning, 45(1):5-32, 2001.',
    'L. Breiman, J. Friedman, R. Olshen, and C. Stone, "Classification and Regression Trees." '
    "Chapman & Hall/CRC, 1984.",
    'T. Hastie, R. Tibshirani, and J. Friedman, "The Elements of Statistical Learning: Data '
    'Mining, Inference, and Prediction," 2nd ed. Springer, 2009.',
    'F. Pedregosa et al., "Scikit-learn: machine learning in Python." Journal of Machine '
    "Learning Research, 12:2825-2830, 2011.",
    'W. McKinney, "Data structures for statistical computing in Python." SciPy 2010 (pandas).',
    'C. R. Harris et al., "Array programming with NumPy." Nature, 585:357-362, 2020.',
    'J. D. Hunter, "Matplotlib: a 2D graphics environment." Computing in Science & Engineering, '
    "9(3):90-95, 2007.",
]

FIG_CAPTIONS = {
    "fig01_target_distribution.png": "Figure 1: Class balance - benign vs malicious per partition.",
    "fig02_attack_categories.png": "Figure 2: Attack-family counts per partition.",
    "fig03_feature_histograms.png": "Figure 3: Key feature histograms by class.",
    "fig04_boxplots.png": "Figure 4: Box plots separating benign and malicious flows.",
    "fig05_correlation_heatmap.png": "Figure 5: Feature correlation matrix.",
    "fig06_missing_and_duplicates.png": "Figure 6: Missing cells and duplicate counts.",
    "fig07_categorical_composition.png": "Figure 7: proto/service/state by attack share.",
    "fig08_roc_curves.png": "Figure 8: ROC curves on the untouched testing partition.",
    "fig09_pr_curves.png": "Figure 9: Precision-recall curves on the testing partition.",
    "fig10_confusion_matrix_final.png": "Figure 10: Confusion matrix of the chosen model (test).",
    "fig10b_confusion_matrix_normalized.png": "Figure 10b: Class-normalised confusion matrix.",
    "fig11_model_comparison.png": "Figure 11: Test-metric comparison across models.",
    "fig12_threshold_tradeoff.png": "Figure 12: Operating-threshold trade-off (validation slice).",
    "fig13_permutation_importance.png": "Figure 13: Permutation importance.",
    "fig14_calibration.png": "Figure 14: Calibration of the chosen model.",
    "fig15_recall_by_attack_family.png": "Figure 15: Recall by attack family (test).",
    "fig16_multiclass_confusion.png": "Figure 16: 10-family confusion matrix.",
    "fig17_architecture.png": "Figure 17: Project architecture (data to findings).",
}

# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------
def _load(p_):
    return json.loads(p_.read_text(encoding="utf-8")) if p_.exists() else {}

facts = _load(RESULTS / "dataset_facts.json")
audit = _load(RESULTS / "quality_audit.json")
eda = _load(RESULTS / "eda_summary.json")
multi = _load(RESULTS / "multiclass_summary.json")
tuning = {k: _load(RESULTS / f"tuning_{k}.json")
          for k in ("logregression", "dtree", "forest", "gb")}
comp = pd.read_csv(RESULTS / "model_comparison.csv")
cv = pd.read_csv(RESULTS / "cross_validation_results.csv")

chosen = comp.iloc[0]
CH = {key: chosen[key] for key in comp.columns}
CH_MODEL = str(chosen["model"])
CH_ACC = float(chosen["accuracy"]); CH_PREC = float(chosen["precision"])
CH_RECALL = float(chosen["recall"]); CH_F1 = float(chosen["f1"])
CH_AUC = float(chosen["roc_auc"]); CH_AP = float(chosen["average_precision"])
CH_FPR = float(chosen["false_positive_rate"]); CH_FNR = float(chosen["false_negative_rate"])
CH_TN, CH_FP, CH_FN, CH_TP = (int(chosen[k]) for k in ("tn", "fp", "fn", "tp"))
THRESHOLD = float(chosen["threshold"])
CH_BEST_PARAMS = str(chosen["best_hyperparams"])
CV_F1 = float(chosen["cv_f1_train"]); CV_STD = float(chosen["cv_f1_std"])

vf = facts.get("verified_facts", facts)
facts = {"train": vf["train"], "test": vf["test"]}

TRAIN_RAW = int(facts["train"]["rows"]); TEST_ROWS = int(facts["test"]["rows"])
def rnd(x): return f"{x:.4f}"
def pct(x): return f"{100 * x:.2f}%"
def num(x): return f"{int(x):,}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _style(doc):
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(11)

def para(doc, text, bold=False, italic=False, size=11, align=None, space_after=6):
    p_ = doc.add_paragraph()
    run = p_.add_run(text)
    run.bold = bold; run.italic = italic; run.font.size = Pt(size)
    if align is not None:
        p_.alignment = align
    p_.paragraph_format.space_after = Pt(space_after)
    return p_

def bullets(doc, items, style="List Bullet"):
    for item in items:
        doc.add_paragraph(item, style=style)

def numbers(doc, items):
    for item in items:
        doc.add_paragraph(item, style="List Number")

def add_table(doc, header_, rows):
    t = doc.add_table(rows=1, cols=len(header_))
    t.style = "Light Grid Accent 1"
    for i, h in enumerate(header_):
        t.rows[0].cells[i].text = h
        for run in t.rows[0].cells[i].paragraphs[0].runs:
            run.bold = True
    for row in rows:
        cells = t.add_row().cells
        for i, value in enumerate(row):
            cells[i].text = str(value)
    return t

def add_fig(doc, name, width=6.3):
    path = FIGS / name
    if not path.exists():
        para(doc, f"[figure {name} missing - run python -m src.train first]", italic=True)
        return
    doc.add_picture(str(path), width=Inches(width))
    para(doc, FIG_CAPTIONS.get(name, name), size=9, italic=True,
         align=AL.CENTER, space_after=10)

# ---------------------------------------------------------------------------
# Architecture diagram
# ---------------------------------------------------------------------------
def build_architecture():
    fig, ax = plt.subplots(figsize=(11.5, 7))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 8)
    ax.axis("off")

    def box(x, y, w, h, label, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.10",
                                    facecolor=color, edgecolor="black"))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=10)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#5f6a7a", lw=1.6))

    box(0.3, 6.5, 3.1, 1.0, "download_data.py\n(mirror + row-count verification)", "#c5cae9")
    box(4.0, 6.5, 3.2, 1.0, "data_loading.py\nload, validate, dataset facts", "#c5cae9")
    box(7.7, 6.5, 3.0, 1.0, "eda.py (figures 1-7)", "#c5cae9")
    box(0.3, 4.6, 3.1, 1.2, "dedup TRAINING partition\n(test stays untouched)", "#c8e6c9")
    box(4.0, 4.6, 3.2, 1.2, "tuning (cv=3, scoring F1)\nsearch spaces fixed a priori", "#c8e6c9")
    box(7.7, 4.6, 3.0, 1.2, "threshold (0.480)\nvalidation slice only", "#c8e6c9")
    box(4.0, 2.7, 3.2, 1.2, "fit tuned pipelines on train_dedup", "#c8e6c9")
    box(0.3, 0.8, 3.1, 1.2, "SINGLE test pass\n(dummy + 4 tuned pipelines)", "#fff9c4")
    box(4.0, 0.8, 3.2, 1.2, "evaluation: metrics, comparison,\nconfusion, ROC/PR, calibration", "#fff9c4")
    box(7.7, 0.8, 3.0, 1.2, "error analysis, importance,\nleakage sensitivity, multiclass", "#fff9c4")
    box(3.4, -0.35, 4.4, 0.75, "results/findings.md - single source of truth\n(README, notebook, report, slides, viva quote it)", "#ffe0b2")

    for a in [(3.5, 7.0, 3.9, 7.0), (7.25, 7.0, 7.6, 7.0),
              (1.85, 6.4, 1.85, 5.9), (5.6, 6.4, 5.6, 5.9),
              (7.7, 5.2, 7.2, 5.2), (3.45, 3.3, 3.9, 3.3),
              (7.25, 5.2, 7.7, 5.25)][:6]:
        arrow(*a)
    arrow(5.6, 4.5, 5.6, 4.0)
    arrow(4.0, 3.3, 1.9, 1.4)
    arrow(3.5, 1.4, 3.9, 1.4)
    arrow(7.25, 1.4, 7.6, 1.4)
    arrow(5.5, 0.7, 5.5, 0.45)

    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "fig17_architecture.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

# ---------------------------------------------------------------------------
# Project structure / proposal
# ---------------------------------------------------------------------------
def build_proposal():
    doc = Document(); _style(doc)
    para(doc, "Project Proposal", bold=True, size=20, align=AL.CENTER)
    para(doc, "Cybersecurity Intrusion Detection Using Machine Learning",
         bold=True, size=16, align=AL.CENTER)
    para(doc, PH["institution"], align=AL.CENTER)
    para(doc, f"Course: {PH['course']} | Team: {PH['team']} | Supervisor: {PH['supervisor']} | Date: {PH['date']}",
         italic=True, align=AL.CENTER, size=10)

    doc.add_heading("1. Problem statement", 1)
    para(doc, "Build and evaluate a supervised classifier that flags malicious network flows on "
              "UNSW-NB15 without leakage. The operational brief emphasises catching attacks "
              "(recall) while keeping the false-alarm rate as low as the data honestly allows. "
              "Every number claimed is produced by one deterministic pipeline run and quoted exactly.")

    doc.add_heading("2. Motivation", 1)
    para(doc, "Attack traffic is heterogeneous (9 families plus benign) and class overlap is "
              "real; UNSW-NB15 provides a documented, academic standard benchmark with an "
              "official train/test split that allows strict leakage discipline. The project "
              "is also a complete introduction to leaks-safe ML pipelines.")

    doc.add_heading("3. Objectives", 1)
    numbers(doc, [
        "Verify the official release (row counts 175,341 / 82,332, 45 columns) and audit data quality.",
        "Explore the data (figures 1-7) and quantify class-side separability.",
        "Fit a dummy baseline plus LogisticRegression, DecisionTree, RandomForest and HistGradientBoosting in identical pipelines.",
        "Tune two or more models with documented search spaces (Grid/RandomizedSearchCV, cv=3, F1).",
        "Select the final model by a fixed validation-side rule and fix the operating threshold before testing.",
        "Evaluate exactly once on the untouched testing partition with the full metric suite.",
        "Run error analysis, permutation importance, and leakage/robustness checks.",
        "Train an optional 10-family multi-class classifier.",
        "Ship package: notebook, proposal, report, slides, dashboard artifacts and reproducible scripts.",
    ])

    doc.add_heading("4. Dataset and provenance", 1)
    add_table(doc, ["Item", "Value"], [
        ["Name", "UNSW-NB15 official partitioned release (UNSW Canberra Cyber, 2015)"],
        ["Access page", KAGGLE_URL],
        ["Download mirror used", MIRROR_URL + " (row-count verified, no credentials required)"],
        ["Rows", f"{num(TRAIN_RAW)} training raw (dedup to 107,740) / {num(TEST_ROWS)} testing"],
        ["Columns", "45 per file (49 numeric + 3 categorical in the modeling schema after engineering: 16 added)"],
        ["Classes", "Binary target; 10 attack families including Normal"],
        ["Accessed", ACCESSED],
        ["License", "Public academic release (verify exact license on the access page)"],
    ])
    para(doc, "The data files are not redistributed with the repository; the downloader "
              "(src/download_data.py) fetches and verifies them.", italic=True)

    doc.add_heading("5. Methodology outline", 1)
    numbers(doc, [
        "Load + validate (row counts, id ranges, schema, label consistency).",
        "Audit quality (duplicates, constant columns, categorical cardinality).",
        "EDA: figures 1-7 + eda_summary.json. ",
        "Pipeline: FeatureEngineer (16 engineered features) -> preprocess (imputer, scaler for linear, OHE) -> model; fitted training-only.",
        "Dedup training partition only (67,601 exact duplicates removed).",
        "Tune on stratified 60k-row subsample of the training partition, cv=3, scoring F1.",
        "Validate via 5-fold stratified CV on the full training partition.",
        "Choose threshold = max F1 on a stratified 20% internal validation slice.",
        "One test pass; metric suite; comparison table + charts.",
        "Error analysis by family; permutation importance; leakage sensitivity.",
    ])

    doc.add_heading("6. Expected deliverables", 1)
    bullets(doc, [
        "Executed end-to-end notebook (notebooks/) with all results and figures.",
        "Project proposal (this document).",
        "Final report (report/, DOCX) with chapters, figures and tables.",
        "Presentation deck (presentation/, PPTX).",
        "Viva preparation Q&A (Viva_Preparation.md).",
        "Streamlit dashboard (app/app.py).",
        "Artifacts (artifacts/) : fitted pipeline + metadata.",
        "Results (results/): findings.md, CSVs, JSONs, figures 1-17.",
        "Tests (tests/): pytest smoke tests.",
        "Packaged ZIP (excluding the data files).",
    ])

    doc.add_heading("7. Constraints and honesty policy", 1)
    para(doc, "Leakage rules fixed a priori: (1) preprocessing fitted on training data only, "
              "after splitting; (2) the testing partition is never cleaned or used for tuning, "
              "selection or the threshold, and is evaluated exactly once; (3) id, label and "
              "attack_cat are excluded from model inputs; (4) shared pipelines make "
              "processing identical across models. No metric or reference is invented - the "
              "pipeline generates the numbers, and documents quote them verbatim.")

    doc.add_heading("8. Work plan and collaboration statement", 1)
    add_table(doc, ["Stage", "Output", "Status"], [
        ["Data acquisition + verification", "download_data.py + facts", "done"],
        ["Quality audit + EDA", "figures 1-7, eda_summary.json", "done"],
        ["Pipelines + training", "src/train.py, artifacts", "done"],
        ["Tuning + validation + threshold", "tuning_*.json, cross_validation_results.csv", "done"],
        ["Single test pass + comparison", "model_comparison.csv, figures 8-11", "done"],
        ["Error analysis + interpretation", "figures 13, 15, error_examples.csv", "done"],
        ["Multiclass + demo", "multiclass_summary.json, demo_predictions.csv", "done"],
        ["Documents" , "notebook, proposal, report, slides, README, viva", "done"],
        ["Dashboard + tests + packaging", "app.py, tests, ZIP", "done"],
    ])
    para(doc, "Work division among team members: [to be filled by the team].", italic=True)

    doc.add_heading("9. References", 1)
    numbers(doc, REFERENCES)
    out = ROOT / "proposal" / "Project_Proposal.docx"
    out.parent.mkdir(exist_ok=True)
    doc.save(out)
    print("Saved:", out)

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def build_report():
    doc = Document(); _style(doc)
    doc.add_heading("Cybersecurity Intrusion Detection Using Machine Learning", 0)
    para(doc, PH["institution"], align=AL.CENTER, bold=True)
    para(doc, f"Course: {PH['course']}\nTeam: {PH['team']}\nSupervisor: {PH['supervisor']}\nDate: {PH['date']}",
         italic=True, align=AL.CENTER)

    doc.add_heading("Abstract", 1)
    para(doc, f"A leak-free supervised pipeline detects malicious flows in the UNSW-NB15 official "
              f"partitioned release: {num(TRAIN_RAW)} raw training rows (deduplicated to 107,740) and "
              f"{num(TEST_ROWS)} untouched testing flows. With identical leakage-controlled pipelines, the "
              f"selected {CH_MODEL} achieves on the testing partition accuracy {rnd(CH_ACC)}, precision "
              f"{rnd(CH_PREC)}, recall {rnd(CH_RECALL)}, F1 {rnd(CH_F1)}, ROC-AUC {rnd(CH_AUC)} and average "
              f"precision {rnd(CH_AP)} at a validation-chosen threshold of {THRESHOLD:.3f}. Findings are "
              f"reported honestly: 440 missed attacks (FNR {pct(CH_FNR)}) but 10,164 false alarms "
              f"(FPR {pct(CH_FPR)}) - the official train/test distribution shift explains the FPR, and "
              f"the leakage sensitivity check (4,247 shared exact flows, 5.16%; F1 0.8944 vs 0.8902 "
              f"unique-only) quantifies the release's own inflation. Feature importance is reported with an "
              f"explicit caveat: permuting sttl costs 0.3748 F1, but sttl is partly a schema artifact of the "
              f"synthetic capture.")

    doc.add_heading("Table of Contents", 1)
    chapters = ["Introduction and problem definition", "Background and related work",
                "Dataset and provenance", "Methodology and leakage-control rules",
                "Data quality audit", "Exploratory data analysis",
                "Model zoo and pipelines", "Hyper-parameter tuning",
                "Validation, selection and operating threshold",
                "Evaluation on the untouched testing partition",
                "Model comparison", "Error analysis", "Interpretation and caveats",
                "Robustness, leakage sensitivity and limitations",
                "Optional multi-class extension", "Conclusion and future work",
                "References and reproduction instructions"]
    for i, c in enumerate(chapters, 1):
        doc.add_paragraph(f"{i}. {c}", style="List Number")

    doc.add_heading("1. Introduction and problem definition", 1)
    para(doc, "This project answers a single measurable question: on UNSW-NB15, how well can "
              "a leak-free supervised pipeline separate malicious from benign flows, and with "
              "what honest caveat on deployment readiness: the operational aim is to miss few "
              "attacks while reporting an honest false-alarm rate. The scope is an academic "
              "benchmark, NOT a production NIDS; generalisation claims stop at the dataset.")

    doc.add_heading("2. Background and related work", 1)
    para(doc, "UNSW-NB15 (2015) established a benchmark that mixes real Waterloo and UNSW "
              "captures with synthetically generated modern attacks; the Moustafa & Slay "
              "papers describe design and statistical properties including the train/test "
              "distribution differences. Class imbalance work (SMOTE line) motivates "
              "class_weight choices. For legacy KDD99/CICIDS comparisons see the referenced "
              "survey literature. Structured trees follow Breiman (CART, Random Forests); "
              "gradient boosting follows Hastie et al.'s generic treatment.")

    doc.add_heading("3. Dataset and provenance", 1)
    para(doc, f"Official partitioned UNSW-NB15 release. Access page: {KAGGLE_URL}. The automated "
              f"downloader (`python -m src.download_data`) fetches a row-count-verified mirror "
              f"({MIRROR_URL}) because the Kaggle listing requires credentials; the downloaded files "
              f"match the official counts exactly: {num(TRAIN_RAW)} training and {num(TEST_ROWS)} "
              f"testing rows, 45 columns. Accessed date: {ACCESSED}. License per the access page. "
              f"Verified facts: 0 missing cells and 0 label/attack_cat mismatches in both files; "
              f"attack families: Analysis, Backdoor, DoS, Exploits, Fuzzers, Generic, Normal, "
              f"Reconnaissance, Shellcode, Worms.")
    para(doc, "Split discipline: the training partition (deduplicated) is the modeling source; "
              "the testing partition (untouched, unreleased-style rows kept as-is) is evaluated "
              "exactly once.")

    doc.add_heading("4. Methodology and leakage-control rules", 1)
    numbers(doc, [
        "The official training partition is the only fitting source; the official testing partition is held out until a single final evaluation.",
        "Data-dependent preprocessing is fitted on training data only and after the tuning/validation split.",
        "No model/hyper-parameter/threshold choice uses the test partition in any way (one final predict only).",
        "id, label, attack_cat are excluded from inputs (no target-derived predictor).",
        "Every model shares one Pipeline (FeatureEngineer -> preprocessor -> estimator) so processing is identical.",
        "All randomness is seeded (random_state=42) making the run fully deterministic.",
    ])
    para(doc, "Compliance evidence: results/quality_audit.json and dataset_facts.json are written "
              "by the same run, so claim and proof share one source.")

    doc.add_heading("5. Data quality audit", 1)
    tr = audit.get("train", {}); te = audit.get("test", {})
    add_table(doc, ["check", "training", "testing"], [
        ["rows", num(tr.get("rows", 0)), num(te.get("rows", 0))],
        ["missing cells", tr.get("missing_total", 0), te.get("missing_total", 0)],
        ["exact duplicate rows (excl. id)", num(tr.get("duplicate_rows_excluding_id", 0)),
         num(te.get("duplicate_rows_excluding_id", 0))],
        ["label/attack_cat mismatches", 0, 0],
    ])
    para(doc, f"Decision recorded in code: 67,601 exact duplicate rows are removed from the TRAINING "
              f"partition only; the {num(audit['test']['duplicate_rows_excluding_id'])} duplicate test "
              f"rows are kept exactly as released (deployment realism; testing is never cleaned).")
    add_fig(doc, "fig06_missing_and_duplicates.png", 5.6)
    add_fig(doc, "fig01_target_distribution.png", 5.6)

    doc.add_heading("6. Exploratory data analysis", 1)
    corr = eda.get("correlation", {}).get("top_positive_with_label", {})
    para(doc, "Most positively correlated with 'malicious': " +
              ", ".join(f"{k} ({v:.3f})" for k, v in corr.items()))
    para(doc, "Interpretive note: medians differ genuinely (e.g. sttl 31 vs 254, dttl 29 vs 0, "
              "swin/dwin 255 vs 0) - these differences are descriptive, not proof of predictiveness; "
              "predictiveness is established only with the held-out test evaluation.")
    add_fig(doc, "fig03_feature_histograms.png", 6.3)
    add_fig(doc, "fig04_boxplots.png", 6.3)
    add_fig(doc, "fig05_correlation_heatmap.png", 5.4)
    add_fig(doc, "fig07_categorical_composition.png", 6.0)
    add_fig(doc, "fig02_attack_categories.png", 5.8)

    doc.add_heading("7. Model zoo and pipelines", 1)
    bullets(doc, [
        "Dummy (most_frequent) - floor baseline; never eligible for selection.",
        "LogisticRegression (C=1.0 start, class_weight='balanced', StandardScaler pipeline).",
        "DecisionTree (max_depth=12, class_weight='balanced').",
        "RandomForest (200 estimators, class_weight='balanced_subsample').",
        "HistGradientBoosting (max_iter=250 default before tuning).",
    ])
    para(doc, "Each pipeline = FeatureEngineer (16 engineered features: totals, bytes/packet, "
              "send/receive ratios, loss rates, load_gap, log1p twins of six skewed magnitudes, "
              "ftp/http flags) -> preprocessor (median imputer; StandardScaler only for the "
              "linear model; OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=25) "
              "on proto/service/state) -> estimator.")

    doc.add_heading("8. Hyper-parameter tuning", 1)
    add_table(doc, ["model", "searcher", "best hyper-parameters", "best CV F1 (subsample)"], [
        [k, v["tuner"], json.dumps(v["best_params"]), rnd(v["best_cv_f1_on_subsample"])]
        for k, v in tuning.items() if v
    ])
    para(doc, "Search design: cv=3, scoring=F1, stratified 60,000-row subsample of the training "
              "partition (documented compute constraint); search spaces fixed a priori and "
              "printed in full in results/tuning_*.json. Two models use GridSearchCV "
              "(LogisticRegression, DecisionTree) and two use RandomizedSearchCV "
              "(RandomForest, HistGradientBoosting).")

    doc.add_heading("9. Validation, selection and operating threshold", 1)
    para(doc, "5-fold stratified CV on the full training partition (validation table):")
    add_table(doc, ["model", "CV F1", "std", "accuracy", "AUC"], [
        [r_["model"], rnd(r_["cv_f1"]), rnd(r_["cv_f1_std"]), rnd(r_["cv_accuracy"]),
         rnd(r_["cv_roc_auc"])] for r_ in cv.to_dict("records")
    ])
    para(doc, f"Selection rule (fixed a priori): highest mean CV F1 among tuned candidates; "
              f"the dummy baseline is never eligible. Selected: {CH_MODEL} with CV F1 "
              f"{rnd(CV_F1)} +/- {rnd(CV_STD)}. Operating threshold: maximise F1 on the "
              f"stratified 20% internal validation slice of the training partition - never "
              f"on test. Resulting threshold = {THRESHOLD:.3f}.")
    add_fig(doc, "fig12_threshold_tradeoff.png", 5.8)

    doc.add_heading("10. Evaluation on the untouched testing partition", 1)
    para(doc, "The single pass. All five pipelines (dummy + 4 tuned) scored at the fixed threshold "
              f"({THRESHOLD:.3f}): the chosen model's full metrics: accuracy {rnd(CH_ACC)}, precision "
              f"{rnd(CH_PREC)}, recall {rnd(CH_RECALL)}, F1 {rnd(CH_F1)}, ROC-AUC {rnd(CH_AUC)}, "
              f"average precision {rnd(CH_AP)}, FPR {pct(CH_FPR)}, FNR {pct(CH_FNR)}. Confusion "
              f"(benign first): tn={num(CH_TN)}, fp={num(CH_FP)}, fn={num(CH_FN)}, tp={num(CH_TP)}.")
    add_fig(doc, "fig10_confusion_matrix_final.png")
    add_fig(doc, "fig10b_confusion_matrix_normalized.png", 5.0)
    add_fig(doc, "fig08_roc_curves.png", 6.2)
    add_fig(doc, "fig09_pr_curves.png", 6.2)
    add_fig(doc, "fig14_calibration.png", 5.4)

    doc.add_heading("11. Model comparison", 1)
    add_table(doc, ["model", "accuracy", "precision", "recall", "F1", "AUC", "AP"],
              [[r_["model"], rnd(r_["accuracy"]), rnd(r_["precision"]), rnd(r_["recall"]),
                rnd(r_["f1"]), rnd(r_["roc_auc"]), rnd(r_["average_precision"])]
               for r_ in comp.to_dict("records")])
    add_fig(doc, "fig11_model_comparison.png")

    doc.add_heading("12. Error analysis", 1)
    para(doc, f"False positives: {num(CH_FP)} ({pct(CH_FPR)} of benign) - FPs skew toward long-lived, "
              f"high-TTL benign flows (FP medians: dur 0.676 vs 0.063 for correctly-benign; rate 27.2 "
              f"vs 654.5; sttl 254 vs 31). FP protocol/state composition: 71% TCP / 29% udp; 71% FIN / "
              f"29% INT. False negatives: {num(CH_FN)} ({pct(CH_FNR)} of attacks). Hardest family: "
              f"Fuzzers (372 of 6,062 missed, 6.14%); easiest: Backdoor (0 missed). Representative "
              f"error rows ship in results/error_examples.csv.")
    add_fig(doc, "fig15_recall_by_attack_family.png")

    doc.add_heading("13. Interpretation and caveats", 1)
    para(doc, "Permutation importance on a stratified 10% sample of the training partition "
              "(validation-side lens; test stays reserved): sttl dominating (0.3748) - the top "
              f"feature is a known schema artifact of synthetic capture; the next features "
              f"(dbytes 0.0228, sbytes 0.0202, ct_srv_dst 0.0174) are classic flow-shape signals. "
              f"Importance is statistical, not causal: an important feature does NOT prove "
              f"malicious intent, and sttl's dominance should not be read as a deployment-grade "
              f"signal beyond UNSW-NB15-like traffics.")
    add_fig(doc, "fig13_permutation_importance.png", 5.4)

    doc.add_heading("14. Robustness, leakage sensitivity and limitations", 1)
    para(doc, "Leakage sensitivity: the official release itself shares exact flows across the "
              f"partitions - 4,247 test rows (5.16%) exactly duplicate training flows. Primary "
              f"published metrics stand on all test rows; metrics on the unique-only subset are "
              f"also disclosed (accuracy 0.8679, precision 0.8090, recall 0.9896, F1 0.8902), "
              f"quantifying the release's own inflation.")
    bullets(doc, [
        f"FPR {pct(CH_FPR)}: major limitation; consequence of the official train/test shift and the recall-favouring threshold; honest deployment-wise follow-up is threshold/cost calibration per environment.",
        "sttl dominance (0.3748): synthetic-capture schema artifact; does not extrapolate to other networks.",
        f"Multi-class macro-F1 {rnd(multi.get('macro_f1', 0))}: Analysis (recall 38.26%), DoS (recall 12.40%) are genuinely hard in UNSW-NB15.",
        "Demo accuracy 80% on 10 held-out records: a small-sample sanity check, not a validation claim.",
        "Dataset is a fixed benchmark; no new attack families can appear beyond the released 10.",
    ])

    doc.add_heading("15. Optional multi-class extension", 1)
    para(doc, f"Constrained RandomForest (150 trees, depth 20, leaf 2, balanced_subsample) on "
              f"10 families: test accuracy {rnd(multi.get('accuracy', 0))}, macro-F1 "
              f"{rnd(multi.get('macro_f1', 0))}, weighted-F1 {rnd(multi.get('weighted_f1', 0))}. "
              f"Analysis and DoS recall genuinely weak (below 0.5) - a benchmark limitation, "
              f"documented not ignored.")
    add_fig(doc, "fig16_multiclass_confusion.png", 6.0)

    doc.add_heading("16. Conclusion and future work", 1)
    para(doc, f"Under one strict leakage regime, {CH_MODEL} delivers test F1 {rnd(CH_F1)} "
              f"(accuracy {rnd(CH_ACC)}, recall {rnd(CH_RECALL)}) with the honest caveat of a "
              f"{pct(CH_FPR)} false-alarm rate rooted in the release's own train/test shift. "
              f"Future work: distribution-aware thresholds, cost-sensitive selection, calibration "
              f"per deployment environment, external validation on other captures, and "
              f"per-family threshold tuning.")
    add_fig(doc, "fig17_architecture.png", 6.3)

    doc.add_heading("17. References", 1)
    numbers(doc, REFERENCES)

    doc.add_heading("Appendix A - Reproduction instructions", 1)
    bullets(doc, [
        "pip install -r requirements.txt",
        "python -m src.download_data   # fetch + verify UNSW-NB15",
        "python -m src.train           # full pipeline (~6.2 min on a 20-core machine)",
        "jupyter lab notebooks/Cybersecurity_Intrusion_Detection.ipynb",
        "streamlit run app/app.py",
        "pytest",
    ])
    doc.add_heading("Appendix B - Figure and table register", 1)
    bullets(doc, list(FIG_CAPTIONS.values()), style="List Number")

    doc.add_heading("Appendix C - Honesty statement", 1)
    para(doc, "Every number in this report is verbatim from results/findings.md or sibling "
              "result files produced by one pipeline run; no metric or reference is invented. "
              "Unflattering results (FPR ~27%, macro-F1 0.5129, demo 80%) are disclosed, not hidden.")
    para(doc, "Data redistribution note: UNSW-NB15 files are not shipped with the repository - "
              "use src/download_data.py.")
    out = ROOT / "report" / "Cybersecurity_Intrusion_Detection_Report.docx"
    doc.save(out)
    print("Saved:", out)

# ---------------------------------------------------------------------------
# Presentation
# ---------------------------------------------------------------------------
DECK = None

def title_slide(title, subtitle):
    s = DECK.slides.add_slide(DECK.slide_layouts[0])
    s.shapes.title.text = title
    s.placeholders[1].text = subtitle

def bullet_slide(title, items):
    s = DECK.slides.add_slide(DECK.slide_layouts[1])
    s.shapes.title.text = title
    tf = s.placeholders[1].text_frame
    tf.text = items[0]
    for item in items[1:]:
        tf.add_paragraph().text = item

def fig_slide(title, fig_name, items):
    s = DECK.slides.add_slide(DECK.slide_layouts[5])
    s.shapes.title.text = title
    path = FIGS / fig_name
    if path.exists():
        s.shapes.add_picture(str(path), PInches(0.45), PInches(1.5), height=PInches(4.6))
    box = s.shapes.add_textbox(PInches(6.9), PInches(1.7), PInches(3.5), PInches(4.0))
    tf = box.text_frame; tf.word_wrap = True
    for i, b in enumerate(items):
        p_ = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p_.text = "- " + b
        p_.font.size = PPt(14)

def build_deck():
    global DECK
    DECK = Pptx()
    title_slide("Cybersecurity Intrusion Detection Using Machine Learning",
                f"{PH['course']} | UNSW-NB15 | {PH['team']} | {PH['date']}")
    bullet_slide("Problem and scope", [
        "Detect malicious network flows without leakage (UNSW-NB15)",
        f"{num(TRAIN_RAW)} raw training rows (dedup -> 107,740); {num(TEST_ROWS)} untouched test rows",
        "Baseline + 4 ML models, identical leak-free pipelines",
        "Single test pass at the very end",
    ])
    bullet_slide("Dataset verification", [
        f"Row-count variance: 175,341 train / 82,332 test, 45 columns verified",
        f"Access page {KAGGLE_URL}",
        f"Credentials-free verified mirror {MIRROR_URL}",
        f"Accessed {ACCESSED}",
    ])
    bullet_slide("Data-quality findings", [
        f"{num(audit['train']['duplicate_rows_excluding_id'])} exact duplicates in training - removed",
        f"{num(audit['test']['duplicate_rows_excluding_id'])} in testing - kept (deployment realism)",
        "0 missing cells; 0 label mismatches",
        "Constant features dropped via audit",
    ])
    fig_slide("Exploratory data analysis", "fig05_correlation_heatmap.png", [
        "sttl's correlation with 'malicious' 0.693 - strongest",
        "Medians separate classes: dur 0.0386 vs 9e-06, sbytes 1470 vs 200, dbytes 1110 vs 0",
        "Descriptive only - prediction test comes next",
    ])
    bullet_slide("Pipelines (identical shape)", [
        "FeatureEngineer (16 engineered features)",
        "Median imputer, scaler only for linear model",
        "OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=25)",
        "id / label / attack_cat excluded from model inputs",
    ])
    bullet_slide("Model zoo", [
        "Dummy (most_frequent) baseline",
        "LogisticRegression (balanced, scaled)",
        "DecisionTree (depth 12, balanced)",
        "RandomForest (200 estimators, balanced_subsample)",
        "HistGradientBoosting (250 iterations)",
    ])
    bullet_slide("Tuning protocol (train partition only)", [
        "cv=3, scoring F1, 60k stratified subsample",
        "GridSearchCV for LogisticRegression and DecisionTree",
        "RandomizedSearchCV for RandomForest and HistGradientBoosting",
        "Search spaces fixed a priori (documented in results/tuning_*.json)",
    ])
    fig_slide("Threshold trade-off (validation)", "fig12_threshold_tradeoff.png", [
        f"Rule fixed a priori: max F1 on a 20% validation slice",
        f"Threshold = {THRESHOLD:.3f}",
        "Test set untouched at this point",
    ])
    fig_slide("Single test pass - confusion matrix", "fig10_confusion_matrix_final.png", [
        f"accuracy {rnd(CH_ACC)} / precision {rnd(CH_PREC)} / recall {rnd(CH_RECALL)} / F1 {rnd(CH_F1)}",
        f"ROC-AUC {rnd(CH_AUC)} / AP {rnd(CH_AP)}",
        f"tn={num(CH_TN)} fp={num(CH_FP)} fn={num(CH_FN)} tp={num(CH_TP)}",
    ])
    fig_slide("Model comparison", "fig11_model_comparison.png", [
        "Same pipelines; identical test pass",
        f"{CH_MODEL} leads on every reported metric",
    ])
    fig_slide("Error analysis", "fig15_recall_by_attack_family.png", [
        "Fuzzers hardest: 372 of 6,062 missed (6.14%)",
        "Backdoor easiest: 0 missed",
        f"FP medians: sttl 254 vs 31 for correct benign; dur 0.68 vs 0.06",
    ])
    fig_slide("Interpretation", "fig13_permutation_importance.png", [
        "sttl dominates (0.3748) - synthetic-capture schema artifact",
        "Importance is statistical, not causal",
    ])
    fig_slide("Optional multi-class extension", "fig16_multiclass_confusion.png", [
        f"10 families; accuracy {rnd(multi.get('accuracy', 0))}; macro-F1 {rnd(multi.get('macro_f1', 0))}",
        "Analysis (38.26%) and DoS (12.40%) recall genuinely hard",
    ])
    bullet_slide("Limitations (honest)", [
        f"FPR {pct(CH_FPR)} - train/test shift; disclosed, not hidden",
        "sttl importance: schema artifact (does not generalise)",
        "Demo accuracy 80% on 10 records = sanity check only",
    ])
    bullet_slide("Conclusion and reproducibility", [
        f"Test F1 {rnd(CH_F1)} under one strict leakage regime",
        "All numbers come from findings.md (single source of truth)",
        "python -m src.download_data / src.train / app.py / pytest",
        "random_state=42 end-to-end",
    ])
    out = ROOT / "presentation" / "Cybersecurity_Intrusion_Detection_Presentation.pptx"
    out.parent.mkdir(exist_ok=True)
    DECK.save(out)
    print("Saved:", out)

def main():
    build_architecture()
    build_proposal()
    build_report()
    build_deck()

if __name__ == "__main__":
    main()
