"""Dataset loading.

UCI Adult (Becker & Kohavi, 1996; CC BY 4.0): 48,842 rows, 6 numeric and
8 categorical columns plus the income label. Downloaded on first use and
cached in data/.
"""

from __future__ import annotations

import io
import urllib.request
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"

ADULT_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "income",
]
ADULT_NUMERIC = ["age", "fnlwgt", "education_num", "capital_gain",
                 "capital_loss", "hours_per_week"]
ADULT_CATEGORICAL = ["workclass", "education", "marital_status", "occupation",
                     "relationship", "race", "sex", "native_country", "income"]

_ADULT_URLS = [
    # Official UCI files (train + test splits)
    ("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", 0),
    ("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test", 1),
]
_ADULT_MIRROR = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/adult-all.csv"


def _fetch(url: str) -> str:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read().decode("utf-8")


def _download_adult() -> pd.DataFrame:
    try:
        parts = []
        for url, skip in _ADULT_URLS:
            parts.append(pd.read_csv(io.StringIO(_fetch(url)), header=None,
                                     names=ADULT_COLUMNS, skiprows=skip,
                                     skipinitialspace=True))
        df = pd.concat(parts, ignore_index=True)
    except Exception:
        df = pd.read_csv(io.StringIO(_fetch(_ADULT_MIRROR)), header=None,
                         names=ADULT_COLUMNS, skipinitialspace=True)
    # adult.test labels end with a period ("<=50K.")
    df["income"] = df["income"].astype(str).str.rstrip(".")
    return df.dropna(how="all")


def load_adult(max_rows: int | None = None, seed: int = 0):
    """Returns (df, numeric_cols, categorical_cols). Missing values ('?') are
    kept as their own category, which is how CTGAN usually sees this data."""
    DATA_DIR.mkdir(exist_ok=True)
    cache = DATA_DIR / "adult.csv"
    if cache.exists():
        df = pd.read_csv(cache)
    else:
        df = _download_adult()
        df.to_csv(cache, index=False)

    for c in ADULT_CATEGORICAL:
        df[c] = df[c].astype(str).str.strip()
    for c in ADULT_NUMERIC:
        df[c] = pd.to_numeric(df[c])

    if max_rows and len(df) > max_rows:
        df = df.sample(max_rows, random_state=seed).reset_index(drop=True)
    return df, list(ADULT_NUMERIC), list(ADULT_CATEGORICAL)
