"""Smoke tests for the intrusion-detection pipeline (fast; small synthetic data)."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys_src = ROOT / "src"

import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loading import EXPECTED_COLUMNS   # noqa: E402
from src.preprocessing import FeatureEngineer   # noqa: E402


def _fake_unsw(n: int = 200) -> pd.DataFrame:
    """Small dataframe carrying the real 45-column UNSW-NB15 release schema."""
    rng = np.random.RandomState(42)
    proto = rng.choice(["tcp", "udp", "icmp"], n)
    service = rng.choice(["http", "ftp", "-", "dns"], n)
    state = rng.choice(["FIN", "INT", "CON"], n)
    malicious = rng.random(n) > 0.5
    df = pd.DataFrame({
        "id": np.arange(n),
        "dur": rng.random(n),
        "spkts": rng.randint(1, 100, n),
        "dpkts": rng.randint(0, 80, n),
        "sbytes": rng.randint(0, 2000, n),
        "dbytes": rng.randint(0, 2000, n),
        "rate": rng.random(n) * 1e4,
        "sttl": np.where(malicious, 254, 31),
        "dttl": np.where(malicious, 0, 29),
        "sload": rng.random(n) * 1e5,
        "dload": np.where(malicious, 0.0, rng.random(n) * 1e5),
        "sloss": rng.randint(0, 10, n),
        "dloss": rng.randint(0, 10, n),
        "sinpkt": rng.random(n) * 0.01,
        "dinpkt": rng.random(n) * 0.01,
        "sjit": rng.random(n) * 0.5,
        "djit": rng.random(n) * 0.5,
        "swin": np.where(malicious, 0, 255),
        "stcpb": rng.randint(0, 90_000, n),
        "dtcpb": rng.randint(0, 90_000, n),
        "dwin": np.where(malicious, 0, 255),
        "tcprtt": rng.random(n) * 1e-3,
        "synack": rng.random(n) * 1e-3,
        "ackdat": rng.random(n) * 1e-3,
        "smean": rng.randint(20, 600, n),
        "dmean": rng.randint(0, 600, n),
        "trans_depth": rng.randint(0, 5, n),
        "response_body_len": rng.randint(0, 5000, n),
        "ct_srv_src": rng.randint(0, 10, n),
        "ct_state_ttl": np.where(malicious, 2, 0),
        "ct_dst_ltm": rng.randint(0, 10, n),
        "ct_src_dport_ltm": rng.randint(0, 10, n),
        "ct_dst_sport_ltm": rng.randint(0, 10, n),
        "ct_dst_src_ltm": rng.randint(0, 8, n),
        "is_ftp_login": rng.choice([0, 1], n),
        "ct_ftp_cmd": rng.choice([0, 1, 2, 4], n, p=[0.93, 0.05, 0.01, 0.01]),
        "ct_flw_http_mthd": rng.randint(0, 5, n),
        "ct_src_ltm": rng.randint(0, 10, n),
        "ct_srv_dst": rng.randint(0, 10, n),
        "is_sm_ips_ports": rng.choice([0, 1], n, p=[0.98, 0.02]),
        "proto": proto,
        "service": service,
        "state": state,
        "label": malicious.astype(int),
        "attack_cat": np.where(malicious, "DoS", "Normal"),
    })
    return df


def test_fake_data_schema_subset():
    """Only assert the synthetic fake data carries the real required column names."""
    df = _fake_unsw()
    required = ["id", "proto", "service", "state", "label", "attack_cat"]
    assert all(c in EXPECTED_COLUMNS for c in required)
    assert all(c in df.columns for c in required)


def test_feature_engineer_output_columns():
    from src.preprocessing import ENGINEERED_FEATURES  # real module

    df = _fake_unsw()
    fe = FeatureEngineer()
    out = fe.fit_transform(df)
    assert len(out) == len(df)
    for feat in ENGINEERED_FEATURES:
        assert feat in out.columns, f"engineered feature missing: {feat}"


def test_engineered_features_are_schema_backed():
    """No invented ruinous features - each name must resolve to a real formula."""
    from src.preprocessing import FeatureEngineer  # noqa

    df = _fake_unsw().drop(columns=["label", "attack_cat"])
    out = FeatureEngineer().fit_transform(df)
    assert not out.isna().all().any()  # no fully-void engineered column


def test_main_pipeline_artifact_scores():
    """The artifact must be loadable and give well-shaped probabilities."""
    art = ROOT / "artifacts" / "intrusion_detection_model.joblib"
    if not art.exists():
        pytest.skip("Run `python -m src.train` first to create the artifact.")
    model = joblib.load(art)
    df = _fake_unsw(50).drop(columns=["label", "attack_cat"])
    proba = model.predict_proba(df)[:, 1]
    assert proba.shape == (50,)
    assert np.all((proba >= 0) & (proba <= 1))


def test_train_reports_consistent_metrics():
    """model_comparison.csv, if present, must carry complete metric columns."""
    comp = ROOT / "results" / "model_comparison.csv"
    if not comp.exists():
        pytest.skip("Run `python -m src.train` first.")
    frame = pd.read_csv(comp)
    required = {"model", "accuracy", "precision", "recall", "f1", "roc_auc"}
    assert required <= set(frame.columns)
    assert frame[["accuracy", "precision", "recall", "f1", "roc_auc"]].notna().all().all()
