"""How much does each cleaning step change the measured score?

Scores the same random forest three ways: on the file as published with plain
stratified folds, after removing duplicates, and after also removing readings
that appear with contradictory labels (the data the project trains on).
"""

import logging

import pandas as pd

from maternal_risk import data, evaluation, models

log = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = data.load_config()
    folds, repeats, seed = cfg["cv_folds"], cfg["cv_repeats"], cfg["seed"]
    forest = models.candidates(seed)["rf"]

    raw = data.load_raw(cfg["data_path"])
    X_raw, y_raw = raw[data.FEATURES], data.encode_target(raw[data.TARGET])
    naive = evaluation.score_model(
        forest, X_raw, y_raw, evaluation.repeated_splits(X_raw, y_raw, folds, repeats, seed)
    )

    def grouped_score(df):
        X, y = df[data.FEATURES], df[data.TARGET]
        splits = evaluation.repeated_splits(
            X, y, folds, repeats, seed, groups=data.feature_groups(df)
        )
        return evaluation.score_model(forest, X, y, splits)

    deduplicated = data.clean(raw, drop_conflicts=False)
    consistent = data.clean(raw)

    table = pd.DataFrame(
        [
            {"protocol": "raw file, random folds", "rows": len(raw), **naive},
            {"protocol": "deduplicated, grouped folds", "rows": len(deduplicated),
             **grouped_score(deduplicated)},
            {"protocol": "deduplicated, conflicts removed", "rows": len(consistent),
             **grouped_score(consistent)},
        ]
    )
    out = data.resolve(cfg["reports_dir"]) / "leakage_check.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False, float_format="%.4f")
    log.info("\n%s", table.to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
