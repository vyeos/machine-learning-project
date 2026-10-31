"""Streamlit dashboard for the intrusion-detection project (4 pages).

Run:  streamlit run app/app.py
Uses ONLY the produced artifacts/results - no re-fitting here.
"""
from pathlib import Path

import json
import sys

import pandas as pd
import streamlit as st
import joblib

ROOT = Path(__file__).resolve().parents[1]
# Streamlit Community Cloud runs this file with its own sys.path setup; make
# sure the repo root is importable so joblib can resolve src.preprocessing
# (the FeatureEngineer class stored inside the model artifact).
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"
ARTIFACTS = ROOT / "artifacts"
FIGS = RESULTS / "figures"

st.set_page_config(page_title="Intrusion Detection Dashboard", page_icon="🛡️", layout="wide")


@st.cache_resource
def load_model():
    path = ARTIFACTS / "intrusion_detection_model.joblib"
    return joblib.load(path) if path.exists() else None


@st.cache_data
def load_json(name: str):
    path = RESULTS / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


@st.cache_data
def load_csv(name: str):
    path = RESULTS / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def fig(name: str, caption: str):
    p = FIGS / name
    if p.exists():
        st.image(str(p), caption=caption, width="stretch")
    else:
        st.info(f"Figure missing: {name} (run `python -m src.train` first)")


def page_overview():
    st.title("🛡️ Project overview - UNSW-NB15 intrusion detection")
    facts = load_json("dataset_facts.json").get("verified_facts", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Training rows (raw)", f"{facts.get('train', {}).get('rows', 0):,}")
    c2.metric("Test rows (untouched)", f"{facts.get('test', {}).get('rows', 0):,}")
    c3.metric("Attack share (train)", f"{facts.get('train', {}).get('malicious_pct', 0):.2f}%")
    c4.metric("Deduplicated train", "107,740")
    fig("fig01_target_distribution.png", "Class balance (benign vs malicious)")
    fig("fig02_attack_categories.png", "Attack families in both partitions")


def page_eda():
    st.title("🔍 Exploratory data analysis")
    fig("fig03_feature_histograms.png", "Key feature histograms")
    fig("fig04_boxplots.png", "Differences by class")
    fig("fig05_correlation_heatmap.png", "Correlation matrix")
    fig("fig07_categorical_composition.png", "Protocol / service / state attack-share")


def page_model():
    st.title("🤖 Model performance (single pass on the untouched test partition)")
    comp = load_csv("model_comparison.csv")
    if len(comp):
        st.subheader("Test-metric comparison")
        st.dataframe(comp.round(4))
    fig("fig11_model_comparison.png", "Recall / precision / F1 / AUC by model")
    fig("fig10_confusion_matrix_final.png", "Confusion matrix of the chosen model")
    fig("fig08_roc_curves.png", "ROC curves")
    fig("fig09_pr_curves.png", "Precision-recall curves")
    fig("fig15_recall_by_attack_family.png", "Recall by attack family")
    findings = RESULTS / "findings.md"
    if findings.exists():
        with st.expander("Full findings document (numbers verbatim)"):
            st.markdown(findings.read_text(encoding="utf-8"))


def page_predict():
    st.title("⚡ Live prediction demo")
    model = load_model()
    if model is None:
        st.error("Artifact missing - run `python -m src.train` first.")
        return
    demo = load_csv("demo_predictions.csv")
    st.write("Uploading UNSW-NB15-shaped rows predicts malicious probability for each.")
    uploaded = st.file_uploader("Upload CSV with the 42 UNSW-NB15 flow features",
                               type=("csv",))
    if uploaded is not None:
        try:
            X = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"Could not read CSV: {exc}")
            return
        drop = [c for c in ("id", "label", "attack_cat") if c in X.columns]
        X = X.drop(columns=drop)
        try:
            proba = model.predict_proba(X)[:, 1]
        except Exception as exc:
            st.error(f"Model could not score these rows: {exc}")
            return
        out = X.iloc[:, :4].copy()
        out["attack_probability"] = proba.round(4)
        out["alert"] = (proba >= 0.480).astype(int)
        st.dataframe(out)
        st.download_button("Download predictions", out.to_csv(index=False),
                           "predictions.csv", "text/csv")
    elif len(demo):
        st.subheader("Built-in held-out demo (from results/demo_predictions.csv)")
        st.dataframe(demo)
        st.metric("Demo accuracy (10 records)", "80%")
    fig("fig16_multiclass_confusion.png", "Optional multi-class extension")


PAGES = {"Overview": page_overview, "EDA": page_eda,
         "Model performance": page_model, "Live prediction": page_predict}

choice = st.sidebar.radio("Page", list(PAGES))
findings = RESULTS / "findings.md"
if findings.exists():
    st.sidebar.success("Results loaded: findings.md present")
else:
    st.sidebar.warning("Run `python -m src.train` to generate results.")
PAGES[choice]()
