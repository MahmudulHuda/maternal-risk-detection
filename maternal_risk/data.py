from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import StratifiedGroupKFold

FEATURES = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]
TARGET = "RiskLevel"
LABELS = ["low", "mid", "high"]  # index in this list is the encoded class

# The file contains a heart rate of 7 bpm, which is a recording error.
# Anything under 30 is not a plausible resting rate for a conscious adult.
MIN_HEART_RATE = 30

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path=None):
    with open(path or PROJECT_ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_raw(path):
    df = pd.read_csv(resolve(path), encoding="utf-8-sig")
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return df[FEATURES + [TARGET]]


def encode_target(labels):
    names = labels.str.replace(" risk", "", regex=False).str.strip().str.lower()
    unknown = set(names) - set(LABELS)
    if unknown:
        raise ValueError(f"unexpected risk levels: {sorted(unknown)}")
    return names.map(LABELS.index).astype(int)


def clean(df, drop_conflicts=True):
    """Drop impossible heart rates, exact duplicates and contradictory records.

    Some identical sets of readings appear with different risk labels. With
    drop_conflicts every copy of such a reading is removed, since there is no
    way to tell which label is the right one.
    """
    df = df[df["HeartRate"] >= MIN_HEART_RATE]
    df = df.drop_duplicates()
    if drop_conflicts:
        labels_per_vector = df.groupby(FEATURES)[TARGET].transform("nunique")
        df = df[labels_per_vector == 1]
    df = df.reset_index(drop=True)
    df[TARGET] = encode_target(df[TARGET])
    return df


def feature_groups(df):
    """One id per distinct feature vector.

    Grouping on the features keeps every copy of a reading on the same side
    of a split, so a model is never scored on a vector it saw in training.
    After clean() with drop_conflicts each vector is a single row and this is
    a no-op; it matters when conflicting records are kept.
    """
    return df.groupby(FEATURES, sort=False).ngroup()


def train_test_split_grouped(df, test_fraction, seed):
    n_splits = round(1 / test_fraction)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, test_idx = next(splitter.split(df[FEATURES], df[TARGET], feature_groups(df)))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)
