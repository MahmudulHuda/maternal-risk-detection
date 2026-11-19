import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer, recall_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_validate

from maternal_risk.data import LABELS

HIGH = LABELS.index("high")

SCORING = {
    "f1_macro": "f1_macro",
    "accuracy": "accuracy",
    # Missing a high-risk pregnancy is the costly mistake, so track it directly.
    "recall_high": make_scorer(recall_score, labels=[HIGH], average="macro"),
}


def repeated_splits(X, y, folds, repeats, seed, groups=None):
    """Materialised CV splits so every model is scored on identical folds."""
    splits = []
    for repeat in range(repeats):
        if groups is None:
            cv = StratifiedKFold(folds, shuffle=True, random_state=seed + repeat)
            splits.extend(cv.split(X, y))
        else:
            cv = StratifiedGroupKFold(folds, shuffle=True, random_state=seed + repeat)
            splits.extend(cv.split(X, y, groups))
    return splits


def score_model(model, X, y, splits):
    scores = cross_validate(model, X, y, cv=splits, scoring=SCORING)
    row = {}
    for metric in SCORING:
        values = scores[f"test_{metric}"]
        row[f"{metric}_mean"] = np.mean(values)
        row[f"{metric}_std"] = np.std(values)
    return row


def compare(models, X, y, splits):
    rows = [{"model": name, **score_model(model, X, y, splits)} for name, model in models.items()]
    table = pd.DataFrame(rows).sort_values("f1_macro_mean", ascending=False)
    return table.reset_index(drop=True)
