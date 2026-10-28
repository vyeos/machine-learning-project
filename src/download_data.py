"""
download_data.py
----------------
Fetches the OFFICIAL UNSW-NB15 partitioned release (train/test CSVs).

Primary path for students with a Kaggle account (recommended; the course lists
Kaggle):
    kaggle datasets download -d mrwellsdavid/unsw-nb15 -p data --unzip
(files: UNSW_NB15_training-set.csv / UNSW_NB15_testing-set.csv, rename to
 UNSW_NB15_training.csv / UNSW_NB15_testing.csv or run this module)

Automation path (used in this development environment, no credentials needed):
    python -m src.download_data
downloads the same official train/test partition from the public mirror
https://huggingface.co/datasets/Mireu-Lab/UNSW-NB15 (MIT/GNU placement of the
mirror; the dataset itself is the UNSW-NB15 release by Moustafa & Slay, ACCS,
UNSW, licensed free for academic use with citation). The module verifies the
downloaded record counts against the documented official structure
(175,341 vs 82,332 rows) and corrects the train/test file naming automatically,
because public mirrors index the two partitions in reversed order.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from src.data_loading import DATA_DIR

BASE_URL = "https://huggingface.co/datasets/Mireu-Lab/UNSW-NB15/resolve/main/"
TRAIN_OUT = DATA_DIR / "UNSW_NB15_training.csv"
TEST_OUT = DATA_DIR / "UNSW_NB15_testing.csv"
EXPECTED_TRAIN_ROWS = 175_341   # official partitioned training set
EXPECTED_TEST_ROWS = 82_332     # official partitioned testing set


def _read_rows(url: str) -> pd.DataFrame:
    """Download one CSV into memory (small files; keep code dependency-free)."""
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": "unsw-nb15-pbl/1.0"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = resp.read()
    return pd.read_csv(io.BytesIO(data), low_memory=False)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if TRAIN_OUT.exists() and TEST_OUT.exists():
        print("Dataset already present:", TRAIN_OUT.parent)
        return

    print(f"Downloading UNSW-NB15 official partition from {BASE_URL} ...")
    first = _read_rows(BASE_URL + "train.csv")
    second = _read_rows(BASE_URL + "test.csv")

    # The mirror indexes the partitions in reversed order; name them by
    # verified row count, never by downloaded file name.
    if len(first) == EXPECTED_TRAIN_ROWS and len(second) == EXPECTED_TEST_ROWS:
        train_df, test_df = first, second
    elif len(first) == EXPECTED_TEST_ROWS and len(second) == EXPECTED_TRAIN_ROWS:
        train_df, test_df = second, first
    else:
        raise RuntimeError(
            f"Verified row counts do not match the official release "
            f"({EXPECTED_TRAIN_ROWS:,}/{EXPECTED_TEST_ROWS:,}); got {len(first):,} and "
            f"{len(second):,}. Refuse to save unverified data.")

    train_df.to_csv(TRAIN_OUT, index=False)
    test_df.to_csv(TEST_OUT, index=False)
    print(f"Saved: {TRAIN_OUT.name} ({len(train_df):,} rows), "
          f"{TEST_OUT.name} ({len(test_df):,} rows)")
    print("Remember to cite the original UNSW-NB15 authors in your report.")


if __name__ == "__main__":
    main()
