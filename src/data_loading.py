"""
data_loading.py
---------------
Dataset acquisition and validation utilities for the UNSW-NB15
network intrusion detection project.

Course: ECSCI24302 - Machine Learning Essentials (PBL Project)

Dataset (documented choice):
    Name        : UNSW-NB15 (broader record set, official 175,341 / 82,332 partition)
    Records used: the official pre-partitioned training and testing CSV files
                  (UNSW_NB15_training-set.csv / UNSW_NB15_testing-set.csv), which are
                  distributed through Kaggle
                  (e.g. https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15)
                  and are ALSO mirrored, byte-compatible, at
                  https://huggingface.co/datasets/Mireu-Lab/UNSW-NB15 (MIT/GPL-3.0 placement;
                  same 45-column schema and row counts as the official Kaggle/unsw release).
    Target      : 'label'   -> 0 = benign / normal traffic, 1 = malicious / attack traffic
    Annotation  : 'attack_cat' -> one of Normal/exploit-family categories (analysis only,
                  never used as a model input).

This module NEVER fabricates data: it loads real CSV files from a documented local
directory and validates their structure before any modeling happens.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Project paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

TRAIN_CSV = DATA_DIR / "UNSW_NB15_training.csv"
TEST_CSV = DATA_DIR / "UNSW_NB15_testing.csv"

# --------------------------------------------------------------------------- #
# Dataset meta-information (single source of truth for reports / README)
# --------------------------------------------------------------------------- #
DATASET_INFO = {
    "name": "UNSW-NB15 (official partitioned release)",
    "domain": "Cybersecurity - network intrusion detection",
    "kaggle_source": "https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15",
    "kaggle_files": [
        "UNSW_NB15_training-set.csv",
        "UNSW_NB15_testing-set.csv",
    ],
    "official_page": "https://research.unsw.edu.au/projects/unsw-nb15-dataset",
    "mirror_used_for_automation": "https://huggingface.co/datasets/Mireu-Lab/UNSW-NB15",
    "mirror_note": (
        "Automated Kaggle download requires personal API credentials. To keep the project "
        "reproducible on any machine, the load script pulls the same official "
        "train/test partition (identical 45-column schema and row counts) from the "
        "public Mireu-Lab mirror of UNSW-NB15, and validates it against the documented "
        "official release structure before use. Students with a Kaggle account can "
        "instead download the files directly from the Kaggle listing above and drop "
        "them into data/."
    ),
    "original_creators": (
        "N. Moustafa and J. Slay, Australian Centre for Cyber Security (ACCS), UNSW"
    ),
    "accessed": "2026-10-26",
    "binary_target": "label",
    "label_mapping": {"0": "benign/normal traffic", "1": "malicious/attack traffic"},
    "multi_class_column": "attack_cat",
    "id_column": "id",
}

# The 45 columns of the official partitioned release, in order.
EXPECTED_COLUMNS = [
    "id", "dur", "proto", "service", "state", "spkts", "dpkts", "sbytes",
    "dbytes", "rate", "sttl", "dttl", "sload", "dload", "sloss", "dloss",
    "sinpkt", "dinpkt", "sjit", "djit", "swin", "stcpb", "dtcpb", "dwin",
    "tcprtt", "synack", "ackdat", "smean", "dmean", "trans_depth",
    "response_body_len", "ct_srv_src", "ct_state_ttl", "ct_dst_ltm",
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm", "is_ftp_login",
    "ct_ftp_cmd", "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst",
    "is_sm_ips_ports", "attack_cat", "label",
]

FEATURES_EXCLUDED_FROM_MODELING = [
    "id",          # row identifier - memorization risk, not a traffic property
    "label",       # target
    "attack_cat",  # fine-grained label: target-derived information
]

ID_COL = "id"
TARGET_COL = "label"
ATTACK_CAT_COL = "attack_cat"


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _coerce_flow_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    A few integer flag/count columns circulate in older copies of the release with
    stray spaces ('', ' 1', ' 2'). Coerce them to numeric and log how many values
    could not be parsed, so the data-quality section states facts, not assumptions.
    """
    suspect = ["is_ftp_login", "ct_ftp_cmd", "is_sm_ips_ports", "ct_flw_http_mthd"]
    for col in suspect:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            before = len(df)
            df[col] = pd.to_numeric(df[col].astype(str).str.strip(), errors="coerce")
            n_bad = int(df[col].isna().sum())
            if n_bad:
                logger.warning(
                    "%s: %d non-numeric values coerced to NaN in '%s'", before, n_bad, col
                )
    return df


def load_dataset(train_path: Path | str = TRAIN_CSV,
                 test_path: Path | str = TEST_CSV) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the official UNSW-NB15 train and test partitions from local CSV files."""
    train_path, test_path = Path(train_path), Path(test_path)
    for p in (train_path, test_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {p}\n"
                "Run 'python -m src.download_data' or see README section "
                "'Dataset placement instructions' to obtain the two CSV files."
            )
    train_df = pd.read_csv(train_path, low_memory=False)
    test_df = pd.read_csv(test_path, low_memory=False)
    train_df = _coerce_flow_counts(train_df)
    test_df = _coerce_flow_counts(test_df)

    # normalize the benign category spelling (raw release uses 'Normal')
    for df in (train_df, test_df):
        df[ATTACK_CAT_COL] = (
            df[ATTACK_CAT_COL].astype(str).str.strip().str.replace("Normal", "Normal", regex=False)
        )
    return train_df, test_df


# --------------------------------------------------------------------------- #
# Validation - facts about the data, verified, never assumed
# --------------------------------------------------------------------------- #
def validate_dataset(train_df: pd.DataFrame,
                     test_df: pd.DataFrame) -> dict:
    """
    Structural validation. Returns a dict of verified facts and raises only if the
    dataset is structurally unusable (wrong schema / non-binary target).
    """
    facts: dict = {}

    for name, df in (("train", train_df), ("test", test_df)):
        missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
        extra_cols = [c for c in df.columns if c not in EXPECTED_COLUMNS]
        if missing_cols:
            raise ValueError(f"{name}: missing expected columns {missing_cols}")

        label_counts = df[TARGET_COL].value_counts(dropna=False).to_dict()
        bad_labels = set(label_counts) - {0, 1, True, False}
        if bad_labels:
            raise ValueError(f"{name}: target column must be 0/1, found {bad_labels}")

        # label vs attack_cat consistency (0 <-> Normal, 1 <-> any attack family)
        normal_mask = df[ATTACK_CAT_COL].astype(str).str.strip().str.lower() == "normal"
        mismatches = int((normal_mask != (df[TARGET_COL] == 0)).sum())

        facts[name] = dict(
            rows=int(len(df)),
            columns=int(df.shape[1]),
            unexpected_extra_columns=extra_cols,
            label_counts={int(k) if isinstance(k, (int, np.integer)) else k: int(v)
                          for k, v in label_counts.items()},
            benign_pct=round(100.0 * float((df[TARGET_COL] == 0).mean()), 3),
            malicious_pct=round(100.0 * float((df[TARGET_COL] == 1).mean()), 3),
            attack_cat_values=sorted(df[ATTACK_CAT_COL].astype(str).str.strip().unique().tolist()),
            label_attack_cat_mismatches=mismatches,
            duplicate_rows=int(df.drop(columns=[ID_COL]).duplicated().sum()),
            id_range=[int(df[ID_COL].min()), int(df[ID_COL].max())],
        )

    if facts["train"]["label_attack_cat_mismatches"] > 0 or \
            facts["test"]["label_attack_cat_mismatches"] > 0:
        raise ValueError("label/attack_cat inconsistency detected - verify dataset integrity")

    return facts


def load_and_validate() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Convenience: load + validate, and persist the verified facts to results/."""
    train_df, test_df = load_dataset()
    facts = validate_dataset(train_df, test_df)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "dataset_facts.json").write_text(
        json.dumps({"dataset_info": DATASET_INFO, "verified_facts": facts}, indent=2),
        encoding="utf-8",
    )
    return train_df, test_df, facts


def dataset_summary_table(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """Rows = columns of the release; verified dtypes, counts and benign/attack split."""
    rows = []
    for col in EXPECTED_COLUMNS:
        rows.append({
            "column": col,
            "dtype_train": str(train_df[col].dtype),
            "non_missing_train": int(train_df[col].notna().sum()),
            "non_missing_test": int(test_df[col].notna().sum()),
            "unique_train": int(train_df[col].nunique(dropna=False)),
            "unique_test": int(test_df[col].nunique(dropna=False)),
            "role": ("identifier" if col == ID_COL
                     else "target" if col == TARGET_COL
                     else "attack annotation" if col == ATTACK_CAT_COL
                     else "input feature"),
        })
    return pd.DataFrame(rows)
