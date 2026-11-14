"""Dataset loading.

UCI Adult (Becker & Kohavi, 1996; CC BY 4.0): 48,842 rows, 6 numeric and
8 categorical columns plus the income label. Downloaded on first use and
cached in data/.

Two more UCI datasets, read from files you put in data/ yourself (not downloaded):
  credit    Default of Credit Card Clients (Yeh, 2009; CC BY 4.0), 30,000 rows.
            archive.ics.uci.edu/dataset/350 -> save the sheet as data/credit_default.csv
            (header row = the row starting with "ID").
  covertype Covertype (Blackard, 1998; CC BY 4.0), 581,012 rows.
            archive.ics.uci.edu/dataset/31 -> data/covtype.data.gz
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


CREDIT_NUMERIC = (["LIMIT_BAL", "AGE"] + [f"BILL_AMT{i}" for i in range(1, 7)]
                  + [f"PAY_AMT{i}" for i in range(1, 7)])
CREDIT_CATEGORICAL = (["SEX", "EDUCATION", "MARRIAGE"]
                      + ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"] + ["default"])


def load_credit(max_rows: int | None = None, seed: int = 0):
    """Credit-card default: 14 numeric and 9 categorical columns plus the label
    `default`. The repayment-status columns PAY_* (-2..8) are treated as categories."""
    df = pd.read_csv(DATA_DIR / "credit_default.csv")
    df = df.drop(columns=["ID"]).rename(columns={"default payment next month": "default"})
    for c in CREDIT_CATEGORICAL:
        df[c] = df[c].astype(str)
    if max_rows and len(df) > max_rows:
        df = df.sample(max_rows, random_state=seed).reset_index(drop=True)
    return df[CREDIT_NUMERIC + CREDIT_CATEGORICAL], list(CREDIT_NUMERIC), list(CREDIT_CATEGORICAL)


COVER_NUMERIC = ["Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology",
                 "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways",
                 "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
                 "Horizontal_Distance_To_Fire_Points"]
COVER_CATEGORICAL = ["Wilderness_Area", "Soil_Type", "Cover_Type"]


def load_covertype(max_rows: int | None = None, seed: int = 0):
    """Covertype: 10 numeric columns; the 4 wilderness-area and 40 soil-type 0/1
    columns are folded back into one categorical column each; label Cover_Type (7 classes)."""
    raw = pd.read_csv(DATA_DIR / "covtype.data.gz", header=None)
    if max_rows and len(raw) > max_rows:
        raw = raw.sample(max_rows, random_state=seed).reset_index(drop=True)
    df = raw.iloc[:, :10].copy()
    df.columns = COVER_NUMERIC
    df["Wilderness_Area"] = (raw.iloc[:, 10:14].to_numpy().argmax(1) + 1).astype(str)
    df["Soil_Type"] = (raw.iloc[:, 14:54].to_numpy().argmax(1) + 1).astype(str)
    df["Cover_Type"] = raw.iloc[:, 54].astype(str)
    return df, list(COVER_NUMERIC), list(COVER_CATEGORICAL)


DATASETS = {"adult": (load_adult, "income"), "credit": (load_credit, "default"),
            "covertype": (load_covertype, "Cover_Type")}


def load_dataset(name: str, max_rows: int | None = None, seed: int = 0):
    """Returns (df, numeric_cols, categorical_cols, target)."""
    loader, target = DATASETS[name]
    df, num, cat = loader(max_rows, seed)
    return df, num, cat, target
