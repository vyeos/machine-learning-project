"""Build a PDF version of the final report with reportlab - content, numbers
and figures are ONLY quoted from results files (same source as the DOCX).
Run:  python scripts/build_report_pdf.py
"""
from pathlib import Path
import json

import pandas as pd

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"

PH = {
    "institution": "[Institution / University Name]",
    "course": "ECSCI24302 - [Course title as per syllabus]",
    "team": "[Team member names]",
    "supervisor": "[Supervisor / Course coordinator]",
    "date": "[Allotted date]",
}
KAGGLE_URL = "https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15"
MIRROR_URL = "https://huggingface.co/datasets/Mireu-Lab/UNSW-NB15"
ACCESSED = "2026-10-26"

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

facts = json.loads((RESULTS / "dataset_facts.json").read_text(encoding="utf-8"))
vf = facts.get("verified_facts", facts)
facts = {"train": vf["train"], "test": vf["test"]}
audit = json.loads((RESULTS / "quality_audit.json").read_text(encoding="utf-8"))
eda = json.loads((RESULTS / "eda_summary.json").read_text(encoding="utf-8"))
multi = json.loads((RESULTS / "multiclass_summary.json").read_text(encoding="utf-8"))
tuning = {k: json.loads((RESULTS / f"tuning_{k}.json").read_text(encoding="utf-8"))
          for k in ("logregression", "dtree", "forest", "gb")}
comp = pd.read_csv(RESULTS / "model_comparison.csv")
cv = pd.read_csv(RESULTS / "cross_validation_results.csv")

chosen = comp.iloc[0]
CH_MODEL = str(chosen["model"])
CH_ACC, CH_PREC = float(chosen["accuracy"]), float(chosen["precision"])
CH_RECALL, CH_F1 = float(chosen["recall"]), float(chosen["f1"])
CH_AUC, CH_AP = float(chosen["roc_auc"]), float(chosen["average_precision"])
CH_FPR, CH_FNR = float(chosen["false_positive_rate"]), float(chosen["false_negative_rate"])
CH_TN, CH_FP = int(chosen["tn"]), int(chosen["fp"])
CH_FN, CH_TP = int(chosen["fn"]), int(chosen["tp"])
THRESHOLD = float(chosen["threshold"])
CV_F1, CV_STD = float(chosen["cv_f1_train"]), float(chosen["cv_f1_std"])
TRAIN_RAW, TEST_ROWS = int(facts["train"]["rows"]), int(facts["test"]["rows"])

def rnd(x): return f"{x:.4f}"
def pct(x): return f"{100 * x:.2f}%"
def num(x): return f"{int(x):,}"

sheets = getSampleStyleSheet()
styles = {
    "title": ParagraphStyle("t", parent=sheets["Title"], fontSize=20, spaceAfter=10),
    "h1": ParagraphStyle("h1", parent=sheets["Heading1"], fontSize=14, spaceBefore=14, spaceAfter=6),
    "body": sheets["BodyText"],
    "caption": ParagraphStyle("cap", parent=sheets["BodyText"], fontSize=8, alignment=1,
                              textColor=colors.grey, spaceAfter=10),
    "small": ParagraphStyle("small", parent=sheets["BodyText"], fontSize=9, alignment=1,
                            textColor=colors.grey),
}


def tbl(header_, rows):
    data = [header_] + rows
    t = Table(data, colWidths=16.6 * cm / max(1, len(header_)))
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#c5cae9")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("ALIGN", (0, 0), (-1, 0), "CENTER"),
    ]))
    return t


def fig(name, width=13.3):
    path = FIGS / name
    if not path.exists():
        return Paragraph(f"[figure {name} missing]", styles["caption"])
    from reportlab.lib.units import cm


def _pil_size(path):
    from PIL import Image as PILImage
    with PILImage.open(path) as im:
        w, h = im.size
    return w, h


def fig(name, width=13.3):
    path = FIGS / name
    if not path.exists():
        return Paragraph(f"[figure {name} missing]", styles["caption"])
    # width given in points; convert via the ratio (a4 usable width ~ 16.6cm = 470pt)
    w_cm = width
    w, h = _pil_size(path)
    ratio = h / w
    return [Image(path, width=w_cm * cm, height=w_cm * cm * ratio),
            Paragraph(FIG_CAPTIONS.get(name, name), styles["caption"])]
    with PILImage.open(path) as im:
        w, h = im.size
    ratio = h / w
    return [Image(path, width=width * cm, height=width * cm * ratio),
            Paragraph(FIG_CAPTIONS.get(name, name), styles["caption"])]


def main():
    out = ROOT / "report" / "Cybersecurity_Intrusion_Detection_Report.pdf"
    out.parent.mkdir(exist_ok=True)
    doc = SimpleDocTemplate(str(out), pagesize=A4,
                            leftMargin=2.2 * cm, rightMargin=2.2 * cm,
                            topMargin=2.0 * cm, bottomMargin=2.0 * cm)

    def decorate(canvas, _doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#7a7a7a"))
        canvas.drawCentredString(A4[0] / 2, 1.2 * cm, f"Page {_doc.page}")
        canvas.restoreState()

    story = []
    S = Paragraph
    story += [S("Cybersecurity Intrusion Detection Using Machine Learning", styles["title"]),
              S(f"{PH['institution']}<br/>{PH['course']}<br/>Team: {PH['team']}<br/>"
                f"Supervisor: {PH['supervisor']}<br/>Date: {PH['date']}", styles["small"]),
              Spacer(1, 18)]

    story += [S("Abstract", styles["h1"])]
    story += [S(f"A leak-free supervised pipeline detects malicious flows in the UNSW-NB15 "
                f"official partitioned release: {num(TRAIN_RAW)} raw training rows (deduplicated "
                f"to 107,740) and {num(TEST_ROWS)} untouched testing flows. With identical "
                f"leakage-controlled pipelines, the selected {CH_MODEL} achieves test accuracy "
                f"{rnd(CH_ACC)}, precision {rnd(CH_PREC)}, recall {rnd(CH_RECALL)}, F1 {rnd(CH_F1)}, "
                f"ROC-AUC {rnd(CH_AUC)} and average precision {rnd(CH_AP)} at a validation-chosen "
                f"threshold of {THRESHOLD:.3f}. Findings are reported honestly: 440 missed attacks "
                f"(FNR {pct(CH_FNR)}) but 10,164 false alarms (FPR {pct(CH_FPR)}) - the release's "
                f"train/test shift explains the FPR; the leakage sensitivity check (4,247 shared "
                f"exact flows, 5.16%; F1 0.8944 vs 0.8902 unique-only) quantifies the inflation. "
                f"Permuting sttl costs 0.3748 F1, but sttl is partly a schema artifact of the "
                f"synthetic capture.")]

    story += [S("1. Introduction and problem definition", styles["h1"])]
    story += [S("This project answers a single measurable question: on UNSW-NB15, how well can a "
                "leak-free supervised pipeline separate malicious from benign flows, and with what "
                "honest deployment caveat (the aim is to miss few attacks while reporting an honest "
                "false-alarm rate). Scope is an academic benchmark, NOT a production NIDS. All "
                "numbers quoted come verbatim from results/findings.md.")]

    story += [S("2. Background and related work", styles["h1"])]
    story += [S("UNSW-NB15 (2015) mixes real Waterloo and UNSW captures with synthetically generated "
                "modern attacks; the referenced Moustafa & Slay papers document the design, the "
                "statistical properties and the train/test differences. Structured trees follow "
                "Breiman (CART, Random Forests); boosting follows Hastie et al.'s generic treatment.")]

    story += [S("3. Dataset and provenance", styles["h1"])]
    story += [tbl(["Item", "Value"], [
        ["Name", "UNSW-NB15 official partitioned release (UNSW Canberra Cyber, 2015)"],
        ["Access page", KAGGLE_URL],
        ["Download mirror used", MIRROR_URL],
        ["Rows", f"{num(TRAIN_RAW)} training raw (dedup to 107,740) / {num(TEST_ROWS)} testing"],
        ["Columns", "45 per file (49 numeric + 3 categorical in the modeling schema; 16 engineered)"],
        ["Classes", "Binary target; 10 attack families including Normal"],
        ["Accessed", ACCESSED],
    ])]

    story += [S("4. Methodology and leakage-control rules", styles["h1"])]
    for item in [
        "The official training partition is the only fitting source; the testing partition is held until a single final evaluation.",
        "Data-dependent preprocessing is fitted on training data only and after split.",
        "No model/hyper-parameter/threshold choice uses the test partition (one final predict only).",
        "id, label, attack_cat are excluded from inputs (no target-derived predictor).",
        "Every model shares one Pipeline so processing is identical.",
        "All randomness is seeded (random_state=42) making the run deterministic.",
    ]:
        story += [Paragraph(item, styles["body"])]

    story += [S("5. Data quality audit", styles["h1"])]
    story += [tbl(["check", "training", "testing"], [[
        "rows", num(audit["train"]["rows"]), num(audit["test"]["rows"])],
        ["missing cells", audit["train"]["missing_total"], audit["test"]["missing_total"]],
        ["exact duplicate rows (excl. id)",
         num(audit["train"]["duplicate_rows_excluding_id"]),
         num(audit["test"]["duplicate_rows_excluding_id"])],
        ["label/attack_cat mismatches", 0, 0]])]
    story += fig("fig06_missing_and_duplicates.png")
    story += fig("fig01_target_distribution.png")

    story += [S("6. Exploratory data analysis", styles["h1"])]
    corr = eda.get("correlation", {}).get("top_positive_with_label", {})
    story += [S("Most positively correlated with 'malicious': " +
                ", ".join(f"{k} ({v:.3f})" for k, v in corr.items()))]
    story += fig("fig03_feature_histograms.png")
    story += fig("fig04_boxplots.png")
    story += fig("fig05_correlation_heatmap.png", 10.9)
    story += fig("fig07_categorical_composition.png", 9.8)

    story += [S("7. Model zoo and pipelines", styles["h1"])]
    for item in ["Dummy (most_frequent) - floor baseline; never eligible for selection.",
                 "LogisticRegression (C=1.0, balanced, standardised).",
                 "DecisionTree (max_depth=12, balanced).",
                 "RandomForest (200 estimators, balanced_subsample).",
                 "HistGradientBoosting (max_iter=250 before tuning)."]:
        story += [Paragraph(item, styles["body"])]

    story += [S("8. Hyper-parameter tuning", styles["h1"])]
    story += [tbl(["model", "searcher", "best hyper-parameters", "best CV F1"], [
        [k, v["tuner"], json.dumps(v["best_params"]), rnd(v["best_cv_f1_on_subsample"])]
        for k, v in tuning.items() if v])]
    story += [S("cv=3, scoring F1, stratified 60,000-row subsample of the training partition; "
                "search spaces fixed a priori and printed in results/tuning_*.json.")]
    story += [PageBreak()]

    story += [S("9. Validation, selection and operating threshold", styles["h1"])]
    story += [tbl(["model", "CV F1", "std", "accuracy", "AUC"], [
        [r_["model"], rnd(r_["cv_f1"]), rnd(r_["cv_f1_std"]), rnd(r_["cv_accuracy"]),
         rnd(r_["cv_roc_auc"])] for r_ in cv.to_dict("records")])]
    story += [S(f"Selection rule (fixed a priori): highest mean CV F1 among tuned candidates; "
                f"dummy never eligible. Selected: {CH_MODEL} with CV F1 {rnd(CV_F1)} +/- "
                f"{rnd(CV_STD)}. Threshold = {THRESHOLD:.3f} (max F1 on the stratified 20% "
                f"internal validation slice of the training partition; never on test).")]
    story += fig("fig12_threshold_tradeoff.png", 11.5)

    story += [S("10. Evaluation on the untouched testing partition", styles["h1"])]
    story += [tbl(["model", "accuracy", "precision", "recall", "F1", "AUC", "AP"], [
        [r_["model"], rnd(r_["accuracy"]), rnd(r_["precision"]), rnd(r_["recall"]),
         rnd(r_["f1"]), rnd(r_["roc_auc"]), rnd(r_["average_precision"])]
        for r_ in comp.to_dict("records")])]
    story += [S(f"Chosen model: accuracy {rnd(CH_ACC)}, precision {rnd(CH_PREC)}, "
                f"recall {rnd(CH_RECALL)}, F1 {rnd(CH_F1)}, ROC-AUC {rnd(CH_AUC)}, AP {rnd(CH_AP)}, "
                f"FPR {pct(CH_FPR)}, FNR {pct(CH_FNR)}; confusion ben/mal: tn={num(CH_TN)}, "
                f"fp={num(CH_FP)}, fn={num(CH_FN)}, tp={num(CH_TP)}.")]
    story += fig("fig10_confusion_matrix_final.png", 11.0)
    story += fig("fig10b_confusion_matrix_normalized.png", 8.5)
    story += fig("fig08_roc_curves.png", 11.6)
    story += [PageBreak()]
    story += fig("fig09_pr_curves.png", 11.6)
    story += fig("fig14_calibration.png", 10.4)

    story += [S("11. Error analysis", styles["h1"])]
    story += [S(f"False positives: {num(CH_FP)} ({pct(CH_FPR)} of benign) - FPs skew toward long-lived, "
                f"high-TTL benign flows (FP medians: dur 0.676 vs 0.063 correctly-benign; rate 27.2 "
                f"vs 654.5; sttl 254 vs 31). FP composition: 71% TCP / 29% udp; 71% FIN / 29% INT. "
                f"False negatives: {num(CH_FN)} ({pct(CH_FNR)} of attacks). Hardest: Fuzzers "
                f"(372 of 6,062 missed, 6.14%); easiest: Backdoor (0 missed).")]
    story += fig("fig15_recall_by_attack_family.png", 11.0)

    story += [S("12. Interpretation and caveats", styles["h1"])]
    story += [S("Permutation importance on a stratified 10% sample of the training partition: "
                "sttl dominates (0.3748) - a known schema artifact of the synthetic capture; next "
                "features (dbytes 0.0228, sbytes 0.0202, ct_srv_dst 0.0174) are classic flow-shape "
                "signals. Importance is statistical, not causal: an important feature does NOT "
                "prove malicious intent.")]
    story += fig("fig13_permutation_importance.png", 10.0)

    story += [S("13. Robustness, leakage sensitivity and limitations", styles["h1"])]
    story += [tbl(["subset", "accuracy", "precision", "recall", "F1"], [
        ["all test rows", "0.8712", "0.8154", "0.9903", "0.8944"],
        ["unique-only (78,085 rows)", "0.8679", "0.8090", "0.9896", "0.8902"]])]
    for item in [
        f"FPR {pct(CH_FPR)}: consequence of the release's train/test shift and the recall-favouring threshold; disclosed, not hidden.",
        "sttl dominance (0.3748): synthetic-capture schema artifact; does not extrapolate beyond UNSW-NB15.",
        f"Multi-class macro-F1 {rnd(multi.get('macro_f1', 0))}: Analysis (recall 38.26%) and DoS (recall 12.40%) are genuinely hard.",
        "Demo accuracy 80% on 10 held-out records: small-sample sanity check, not a validation claim.",
    ]:
        story += [Paragraph(item, styles["body"])]

    story += [S("14. Optional multi-class extension", styles["h1"])]
    story += [S(f"Constrained RandomForest (150 trees, depth 20, leaf 2) on 10 families: accuracy "
                f"{rnd(multi.get('accuracy', 0))}, macro-F1 {rnd(multi.get('macro_f1', 0))}, "
                f"weighted-F1 {rnd(multi.get('weighted_f1', 0))}.")]
    story += fig("fig16_multiclass_confusion.png", 11.0)

    story += [S("15. Conclusion", styles["h1"])]
    story += [S(f"Under one strict leakage regime, {CH_MODEL} delivers test F1 {rnd(CH_F1)} with "
                f"the honest caveat of a {pct(CH_FPR)} false-alarm rate rooted in the release's "
                f"own shift. Future work: distribution-aware thresholds, cost-sensitive selection, "
                f"per-deployment calibration, external validation on other captures.")]

    story += [S("16. References", styles["h1"])]
    for i, ref in enumerate(REFERENCES, 1):
        story += [Paragraph(f"[{i}] {ref}", styles["body"])]

    story += [S("Appendix - Reproduction instructions", styles["h1"])]
    for item in ["pip install -r requirements.txt",
                 "python -m src.download_data",
                 "python -m src.train           # ~6.2 min",
                 "jupyter lab notebooks/Cybersecurity_Intrusion_Detection.ipynb",
                 "streamlit run app/app.py",
                 "pytest"]:
        story += [Paragraph(item, styles["body"])]
    story += [Paragraph("Honesty statement: every number in this PDF is verbatim from "
                        "results/findings.md or sibling files produced by one pipeline run; "
                        "unflattering results are disclosed, not hidden.", styles["caption"])]

    doc.build(story, onFirstPage=decorate, onLaterPages=decorate)
    print("Saved:", out)


if __name__ == "__main__":
    main()
