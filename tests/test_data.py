import pandas as pd
import pytest

from maternal_risk import data


def make_raw(rows):
    return pd.DataFrame(rows, columns=data.FEATURES + [data.TARGET])


def test_clean_drops_duplicates_and_bad_heart_rate():
    raw = make_raw([
        [25, 120, 80, 7.0, 98.0, 70, "low risk"],
        [25, 120, 80, 7.0, 98.0, 70, "low risk"],
        [16, 120, 75, 7.9, 98.0, 7, "low risk"],
        [40, 140, 90, 15.0, 98.0, 80, "high risk"],
    ])
    cleaned = data.clean(raw)
    assert len(cleaned) == 2
    assert cleaned[data.TARGET].tolist() == [data.LABELS.index("low"), data.LABELS.index("high")]


def test_clean_drops_every_copy_of_a_contradictory_reading():
    raw = make_raw([
        [25, 120, 80, 7.0, 98.0, 70, "low risk"],
        [25, 120, 80, 7.0, 98.0, 70, "mid risk"],
        [30, 110, 70, 6.5, 98.0, 76, "low risk"],
    ])
    assert len(data.clean(raw)) == 1
    assert len(data.clean(raw, drop_conflicts=False)) == 3


def test_clean_does_not_modify_input():
    raw = make_raw([[25, 120, 80, 7.0, 98.0, 70, "mid risk"]])
    data.clean(raw)
    assert raw[data.TARGET].tolist() == ["mid risk"]


def test_unknown_label_is_rejected():
    with pytest.raises(ValueError, match="unexpected risk levels"):
        data.encode_target(pd.Series(["low risk", "severe risk"]))


def test_same_features_share_a_group_even_with_different_labels():
    raw = make_raw([
        [25, 120, 80, 7.0, 98.0, 70, "low risk"],
        [25, 120, 80, 7.0, 98.0, 70, "mid risk"],
        [30, 110, 70, 6.5, 98.0, 76, "low risk"],
    ])
    groups = data.feature_groups(data.clean(raw, drop_conflicts=False))
    assert groups[0] == groups[1] != groups[2]


def test_split_keeps_feature_vectors_on_one_side():
    df = data.clean(data.load_raw(data.load_config()["data_path"]))
    train, test = data.train_test_split_grouped(df, test_fraction=0.2, seed=0)
    assert len(train) + len(test) == len(df)
    overlap = train[data.FEATURES].merge(test[data.FEATURES].drop_duplicates(), on=data.FEATURES)
    assert overlap.empty
    assert set(test[data.TARGET]) == {0, 1, 2}
