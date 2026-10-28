"""
predict.py
----------
Prediction utilities + command-line demonstration for the final intrusion
detection model. Loads the serialized sklearn Pipeline (feature engineering +
preprocessing + classifier, fit on the UNSW-NB15 training partition) and the
fixed decision threshold chosen on validation data.

Required input schema (raw flow record):
    the UNSW-NB15 flow feature columns: dur, proto, service, state, spkts, dpkts,
    sbytes, dbytes, rate, sttl, dttl, sload, dload, sloss, dloss, sinpkt, dinpkt,
    sjit, djit, swin, stcpb, dtcpb, dwin, tcprtt, synack, ackdat, smean, dmean,
    trans_depth, response_body_len, ct_srv_src, ct_state_ttl, ct_dst_ltm,
    ct_src_dport_ltm, ct_dst_sport_ltm, ct_dst_src_ltm, is_ftp_login, ct_ftp_cmd,
    ct_flw_http_mthd, ct_src_ltm, ct_srv_dst, is_sm_ips_ports

Label mapping: 0 = benign / normal traffic, 1 = malicious / attack traffic.

CLI:
    python -m src.predict --input path/to/flows.csv --output predictions.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loading import ARTIFACTS_DIR

MODEL_PATH = ARTIFACTS_DIR / "intrusion_detection_model.joblib"
META_PATH = ARTIFACTS_DIR / "model_metadata.json"

REQUIRED_FEATURES = [
    "dur", "proto", "service", "state", "spkts", "dpkts", "sbytes", "dbytes",
    "rate", "sttl", "dttl", "sload", "dload", "sloss", "dloss", "sinpkt",
    "dinpkt", "sjit", "djit", "swin", "stcpb", "dtcpb", "dwin", "tcprtt",
    "synack", "ackdat", "smean", "dmean", "trans_depth", "response_body_len",
    "ct_srv_src", "ct_state_ttl", "ct_dst_ltm", "ct_src_dport_ltm",
    "ct_dst_sport_ltm", "ct_dst_src_ltm", "is_ftp_login", "ct_ftp_cmd",
    "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst", "is_sm_ips_ports",
]
NUMERIC_FEATURES = [c for c in REQUIRED_FEATURES if c not in ("proto", "service", "state")]
TARGET_LIKE_COLUMNS = ("label", "attack_cat")


class IntrusionPredictor:
    """Loads the fitted pipeline + validation-derived threshold and predicts."""

    def __init__(self, model_path: Path | str = MODEL_PATH,
                 meta_path: Path | str = META_PATH):
        self.pipeline = _load_model(model_path)
        self.metadata = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        self.threshold = float(self.metadata.get("decision_threshold", 0.5))

    def validate_batch_schema(self, df: pd.DataFrame) -> dict:
        """Strict validation for uploaded batches (used by CLI and dashboard)."""
        missing = [c for c in REQUIRED_FEATURES if c not in df.columns]
        if missing:
            raise ValueError(
                f"Batch schema does not match the required feature list: missing {missing}")

        unexpected_targets = [c for c in TARGET_LIKE_COLUMNS if c in df.columns]
        numeric_bad: dict[str, int] = {}
        for col in NUMERIC_FEATURES:
            coerced = pd.to_numeric(df[col].astype(str).str.strip(), errors="coerce")
            bad = int(coerced.isna().sum() - df[col].isna().sum())
            if bad:
                numeric_bad[col] = bad
        if numeric_bad:
            raise ValueError(f"Non-numeric values present in numeric columns: {numeric_bad}")

        return {
            "rows": int(len(df)),
            "missing_columns": missing,
            "warnings": ([f"target-like columns present and ignored: {unexpected_targets}"]
                         if unexpected_targets else []),
        }

    def predict_frame(self, df: pd.DataFrame, feature_frame: pd.DataFrame | None = None) -> pd.DataFrame:
        """
        Predict for a batch. All 42 feature columns must be present (they are).
        Returns a frame with predicted_label and attack_probability.
        """
        data = df if feature_frame is None else feature_frame
        self.validate_batch_schema(data)
        proba = self.pipeline.predict_proba(data[REQUIRED_FEATURES])[:, 1]
        out = pd.DataFrame({
            "predicted_label": (proba >= self.threshold).astype(int),
            "attack_probability": np.round(proba, 4),
        }).reset_index(drop=True)
        return out

    def predict_record(self, record: dict) -> dict:
        row = pd.DataFrame([record], columns=REQUIRED_FEATURES, dtype=object)
        proba = self.pipeline.predict_proba(row[REQUIRED_FEATURES])[:, 1][0]
        label = int(proba >= self.threshold)
        return {"predicted_label": label,
                "predicted_class": ("malicious / attack traffic" if label else
                                    "benign / normal traffic"),
                "attack_probability": round(float(proba), 4),
                "threshold": self.threshold,
                "disclaimer": ("experimental prediction on captured UNSW-NB15-style flow "
                               "record; not a definitive security verdict")}


def _load_model(path: Path | str):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Serialized model not found: {path}\n"
            "Run 'python -m src.train' first, then retry.")
    return joblib_compat_load(path)


def joblib_compat_load(path: Path):
    import joblib
    return joblib.load(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict with the final IDS pipeline")
    parser.add_argument("--input", required=True, help="CSV of flow records")
    parser.add_argument("--output", default="predictions.csv")
    parser.add_argument("--show", type=int, default=10,
                        help="rows displayed in the CLI demo")
    args = parser.parse_args()

    predictor = IntrusionPredictor()
    df = pd.read_csv(args.input, low_memory=False)
    info = predictor.validate_batch_schema(df)
    print("Batch schema:", info)
    pred = predictor.predict_frame(df)
    print(f"Predicted {len(pred):,} records "
        f"({int(pred['predicted_label'].sum()):,} flagged malicious)")
    demo = df[["id", "dur", "proto", "service", "state", "sbytes", "dbytes"]].copy()
    demo["attack_probability"] = pred["attack_probability"]
    demo["predicted_label"] = pred["predicted_label"]
    columns = ["id", "dur", "proto", "service", "state", "sbytes", "dbytes",
               "attack_probability", "predicted_label"]
    print(demo[columns].head(args.show).to_string(index=False))
    out = demo.merge(pred, left_index=True, right_index=True)
    out.to_csv(args.output, index=False)
    print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()
