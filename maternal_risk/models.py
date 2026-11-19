import logging

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import make_pipeline as make_sampling_pipeline
except ImportError:
    SMOTE = None


def candidates(seed):
    """Models to compare, keyed by the name used in the results table."""
    models = {
        "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "logreg_balanced": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")
        ),
        "rf": _forest(seed),
        "rf_balanced": _forest(seed, class_weight="balanced"),
        "hist_gb": HistGradientBoostingClassifier(random_state=seed),
        "hist_gb_balanced": HistGradientBoostingClassifier(
            random_state=seed, class_weight="balanced"
        ),
    }

    if XGBClassifier is not None:
        models["xgb"] = _xgb(seed)
    else:
        log.warning("xgboost is not installed, skipping the XGBoost models")

    if SMOTE is not None:
        # imblearn's pipeline resamples during fit only, so validation folds
        # are never oversampled and no synthetic point leaks into scoring.
        models["rf_smote"] = make_sampling_pipeline(SMOTE(random_state=seed), _forest(seed))
        if XGBClassifier is not None:
            models["xgb_smote"] = make_sampling_pipeline(SMOTE(random_state=seed), _xgb(seed))
    else:
        log.warning("imbalanced-learn is not installed, skipping the SMOTE models")

    return models


def _forest(seed, **kwargs):
    return RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1, **kwargs)


def _xgb(seed):
    return XGBClassifier(
        n_estimators=300, learning_rate=0.1, max_depth=4, random_state=seed, n_jobs=-1
    )


# Small grids aimed at the one thing that matters with ~360 training rows:
# how far the trees are allowed to grow. Parameter names are prefixed at
# search time when the estimator sits inside a pipeline.
TUNING_GRIDS = {
    "rf": {"max_depth": [4, 6, 8, None], "min_samples_leaf": [1, 3, 5]},
    "hist_gb": {"max_depth": [2, 3, None], "learning_rate": [0.03, 0.1], "max_iter": [100, 300]},
    "xgb": {"max_depth": [2, 3, 4, 6], "learning_rate": [0.03, 0.1], "n_estimators": [100, 300]},
}


def tuning_grid(name, model):
    family = next((key for key in TUNING_GRIDS if name.startswith(key)), None)
    if family is None:
        return None
    grid = TUNING_GRIDS[family]
    if hasattr(model, "steps"):
        step = model.steps[-1][0]
        grid = {f"{step}__{param}": values for param, values in grid.items()}
    return grid
