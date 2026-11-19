import numpy as np
import pandas as pd
import pytest

from maternal_risk import data, evaluation, models
from maternal_risk.predict import predict


@pytest.fixture(scope="module")
def train_frame():
    df = data.clean(data.load_raw(data.load_config()["data_path"]))
    return data.train_test_split_grouped(df, test_fraction=0.2, seed=0)[0]


def test_cv_folds_never_share_a_feature_vector(train_frame):
    X, y = train_frame[data.FEATURES], train_frame[data.TARGET]
    groups = data.feature_groups(train_frame).to_numpy()
    splits = evaluation.repeated_splits(X, y, folds=5, repeats=2, seed=0, groups=groups)
    assert len(splits) == 10
    for fit_idx, val_idx in splits:
        assert not set(groups[fit_idx]) & set(groups[val_idx])


@pytest.mark.parametrize("name", ["logreg_balanced", "rf", "hist_gb"])
def test_models_fit_and_predict_valid_classes(train_frame, name):
    X, y = train_frame[data.FEATURES], train_frame[data.TARGET]
    model = models.candidates(seed=0)[name].fit(X, y)
    scored = predict(model, X.head(20))
    assert set(scored["predicted_risk"]) <= set(data.LABELS)
    assert np.allclose(scored[["p_low", "p_mid", "p_high"]].sum(axis=1), 1, atol=0.01)


def test_tuning_grid_is_prefixed_for_pipelines():
    assert models.tuning_grid("logreg", None) is None
    assert "max_depth" in models.tuning_grid("rf_balanced", models.candidates(0)["rf_balanced"])
    if "rf_smote" in models.candidates(0):
        grid = models.tuning_grid("rf_smote", models.candidates(0)["rf_smote"])
        assert all(key.startswith("randomforestclassifier__") for key in grid)


def test_predict_rejects_incomplete_input():
    with pytest.raises(ValueError, match="missing columns"):
        predict(None, pd.DataFrame({"Age": [30]}))
