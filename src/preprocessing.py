"""
preprocessing.py
----------------
Data-quality auditing, schema-supported feature engineering and leakage-safe
preprocessing pipelines for the UNSW-NB15 intrusion detection project.

Key design decisions (all documented in the notebook / report):
* 'id' (identifier) and 'attack_cat' (target-derived annotation) are never used
  as model inputs.
* All data-DEPENDENT preprocessing (imputation, scaling, rare-category grouping,
  oversampling) is fitted on training data only, inside sklearn Pipelines.
* Feature engineering adds only features derivable from the real schema
  (no invented columns) and uses row-guarded arithmetic to avoid infinities.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_loading import (
    ATTACK_CAT_COL,
    ID_COL,
    TARGET_COL,
    PROJECT_ROOT,
    RESULTS_DIR,
)

EPS = 1e-6

# The six most skewed traffic-volume/rate magnitudes - log1p work in a scaled space.
SKEWED_NUMERIC = ["dur", "sbytes", "dbytes", "sload", "dload", "rate"]


# --------------------------------------------------------------------------- #
# Data-quality audit
# --------------------------------------------------------------------------- #
def audit_quality(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """Objective, verified data-quality report (facts only, no treatment here)."""
    audit: dict = {}

    def _split_quality(df: pd.DataFrame, name: str) -> dict:
        finite_only = df.select_dtypes(include=[np.number])
        non_finite = finite_only.apply(
            lambda s: (~np.isfinite(s.replace([np.inf, -np.inf], np.nan))).sum()
        )
        return {
            "rows": int(len(df)),
            "missing_total": int(df.isna().sum().sum()),
            "missing_by_column": {
                c: int(n) for c, n in df.isna().sum().items() if n > 0
            },
            "non_finite_by_column": {
                c: int(n) for c, n in non_finite.items() if n > 0
            },
            "duplicate_rows_excluding_id": int(
                df.drop(columns=[ID_COL]).duplicated().sum()
            ),
            "constant_columns": {
                c: int(df[c].nunique(dropna=False))
                for c in df.columns
                if df[c].nunique(dropna=False) <= 1
            },
            "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        }

    audit["train"] = _split_quality(train_df, "train")
    audit["test"] = _split_quality(test_df, "test")

    # ---- cross-file duplicate flows (data-integrity question raised in review) ----
    cols = [c for c in train_df.columns if c != ID_COL]
    shared = (
        train_df[cols].drop_duplicates()
        .merge(test_df[cols].drop_duplicates(), how="inner")
    )
    audit["distinct_flow_tuples_shared_with_test"] = int(len(shared))

    # ---- suspicious / inconsistent categorical values ----
    audit["categorical_cardinality"] = {
        "proto": {"train": int(train_df["proto"].nunique()),
                  "test": int(test_df["proto"].nunique())},
        "service": {"train": int(train_df["service"].nunique()),
                    "test": int(test_df["service"].nunique())},
        "state": {"train": int(train_df["state"].nunique()),
                  "test": int(test_df["state"].nunique())},
    }
    audit["value_inventory"] = {
        "proto_train": sorted(train_df["proto"].astype(str).unique().tolist()),
        "proto_test": sorted(test_df["proto"].astype(str).unique().tolist()),
        "service_train": sorted(train_df["service"].astype(str).unique().tolist()),
        "state_train": sorted(train_df["state"].astype(str).unique().tolist()),
        "state_test": sorted(test_df["state"].astype(str).unique().tolist()),
    }
    audit["flag_value_inventory"] = {
        "is_ftp_login_train": {str(k): int(v) for k, v in
                               train_df["is_ftp_login"].value_counts().sort_index().items()},
        "ct_ftp_cmd_train": {str(k): int(v) for k, v in
                             train_df["ct_ftp_cmd"].value_counts().sort_index().head(12).items()},
        "is_sm_ips_ports_train": {str(k): int(v) for k, v in
                                  train_df["is_sm_ips_ports"].value_counts().sort_index().items()},
    }

    # ---- categories seen in test but not train (schema-shift exposure) ----
    unseen = {}
    for col in ("proto", "service", "state"):
        tr = set(train_df[col].astype(str).unique())
        te = set(test_df[col].astype(str).unique())
        unseen[col] = sorted(te - tr)
    audit["unseen_test_categories"] = unseen

    # ---- negative/zero-detected numeric bounds worth reporting ----
    audit["negative_counts"] = {
        c: int(n) for c, n in
        train_df.select_dtypes(include=[np.number]).lt(0).sum().items() if n > 0
    }
    return audit


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Adds schema-supported derived traffic features. Purely row-wise (no fitting,
    no target access) -> safe to use inside pipelines for training and inference.

    Engineered features (all computable from one flow record, so available at
    prediction time; no target leakage):
      total_pkts, total_bytes        : flow-size summaries
      bytes_per_pkt                  : flow-level mean packet size (differs from smean/dmean,
                                       which are direction-wise)
      s_d_bytes_ratio, s_d_pkts_ratio: forward/backward asymmetry (strong signal for
                                       one-way probe/DoS flows)
      loss_rate_s, loss_rate_d       : share of lost / retransmitted packets
      load_gap                       : directional load difference (b/s)
      log_* duplicates of dur, sbytes, dbytes, sload, dload, rate : compress extreme
                                       heavy-tail values so linear models are not dominated
                                       by a few enormous flows
      flag_ftp_login, flag_http_activity : "activity present" indicators (robust to the
                                       invalid counts {2,4} found in audit)
    """

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        num = lambda c: df[c].astype(float)  # noqa: E731

        df["total_pkts"] = num("spkts") + num("dpkts")
        df["total_bytes"] = num("sbytes") + num("dbytes")
        df["bytes_per_pkt"] = df["total_bytes"] / (df["total_pkts"] + EPS)
        df["s_d_bytes_ratio"] = (num("sbytes") + EPS) / (num("dbytes") + EPS)
        df["s_d_pkts_ratio"] = (num("spkts") + EPS) / (num("dpkts") + EPS)
        df["loss_rate_s"] = num("sloss") / (num("spkts") + EPS)
        df["loss_rate_d"] = num("dloss") / (num("dpkts") + EPS)
        df["load_gap"] = num("sload") - num("dload")

        for c in SKEWED_NUMERIC:
            df[f"log_{c}"] = np.log1p(df[c].clip(lower=0))

        df["flag_ftp_login"] = (num("is_ftp_login") > 0).astype(int)
        df["flag_http_activity"] = (num("ct_flw_http_mthd") > 0).astype(int)

        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna({f"log_{c}": 0.0 for c in SKEWED_NUMERIC})
        return df

    def get_feature_names_out(self, input_features=None):
        base = list(input_features) if input_features is not None else []
        added = [
            "total_pkts", "total_bytes", "bytes_per_pkt", "s_d_bytes_ratio",
            "s_d_pkts_ratio", "loss_rate_s", "loss_rate_d", "load_gap",
            *(f"log_{c}" for c in SKEWED_NUMERIC),
            "flag_ftp_login", "flag_http_activity",
        ]
        return np.asarray([*(c for c in base if c not in added), *added])


ENGINEERED_FEATURES = [
    "total_pkts", "total_bytes", "bytes_per_pkt", "s_d_bytes_ratio",
    "s_d_pkts_ratio", "loss_rate_s", "loss_rate_d", "load_gap",
    "log_dur", "log_sbytes", "log_dbytes", "log_sload", "log_dload", "log_rate",
    "flag_ftp_login", "flag_http_activity",
]


def feature_columns(drop_constant: list[str] | None = None) -> dict:
    """
    Model input feature lists. Excludes id / label / attack_cat by construction.
    `drop_constant` gathers columns identified from TRAINING data only
    (a column that is constant in training carries no information and cannot be
    scaled meaningfully).
    """
    drop_constant = drop_constant or []
    raw_numeric = [
        "dur", "spkts", "dpkts", "sbytes", "dbytes", "rate", "sttl", "dttl",
        "sload", "dload", "sloss", "dloss", "sinpkt", "dinpkt", "sjit", "djit",
        "swin", "stcpb", "dtcpb", "dwin", "tcprtt", "synack", "ackdat", "smean",
        "dmean", "trans_depth", "response_body_len", "ct_srv_src", "ct_state_ttl",
        "ct_dst_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
        "is_ftp_login", "ct_ftp_cmd", "ct_flw_http_mthd", "ct_src_ltm",
        "ct_srv_dst", "is_sm_ips_ports",
    ]
    categorical = ["proto", "service", "state"]

    log_twins = [f"log_{c}" for c in SKEWED_NUMERIC]
    numeric = [c for c in raw_numeric if c not in SKEWED_NUMERIC and c not in drop_constant]
    numeric += log_twins + ENGINEERED_FEATURES[:8] + ["flag_ftp_login", "flag_http_activity"]
    numeric = [c for c in numeric if c not in drop_constant]

    return {"numeric": numeric, "categorical": [c for c in categorical if c not in drop_constant]}


# --------------------------------------------------------------------------- #
# Preprocessing pipelines (fitted on training data only)
# --------------------------------------------------------------------------- #
def build_preprocessor(drop_constant: list[str] | None = None,
                       scale: bool = False) -> ColumnTransformer:
    """
    Leakage-safe column transformer.
      * numeric : median imputer  (+ optional StandardScaler for linear models)
      * categorical : constant imputer -> one-hot with infrequent-bucket handling
        (rare categories learned on TRAIN only; unseen test categories fall into
        the 'infrequent / unknown' buckets instead of crashing)
    Fitting this object calls SimpleImputer/OneHotEncoder.statistics_ etc. only on
    the data passed to fit().
    """
    cols = feature_columns(drop_constant)
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    numeric = Pipeline(steps)
    categorical = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                 min_frequency=25, sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric, cols["numeric"]),
            ("cat", categorical, cols["categorical"]),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
