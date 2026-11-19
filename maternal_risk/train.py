"""Compare models with grouped cross-validation, tune the winner, test it once."""

import json
import logging

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score
from sklearn.model_selection import GridSearchCV

from maternal_risk import data, evaluation, models

log = logging.getLogger(__name__)


def tune(name, model, X, y, splits):
    grid = models.tuning_grid(name, model)
    if grid is None:
        return model.fit(X, y), {}, None
    search = GridSearchCV(model, grid, scoring="f1_macro", cv=splits, n_jobs=-1)
    search.fit(X, y)
    return search.best_estimator_, search.best_params_, search.best_score_


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = data.load_config()
    reports = data.resolve(cfg["reports_dir"])
    (reports / "figures").mkdir(parents=True, exist_ok=True)

    raw = data.load_raw(cfg["data_path"])
    df = data.clean(raw)
    log.info("%d raw rows -> %d after cleaning", len(raw), len(df))

    train, test = data.train_test_split_grouped(df, cfg["test_fraction"], cfg["seed"])
    X_train, y_train = train[data.FEATURES], train[data.TARGET]
    X_test, y_test = test[data.FEATURES], test[data.TARGET]
    log.info("train %d rows, test %d rows", len(train), len(test))

    splits = evaluation.repeated_splits(
        X_train, y_train, cfg["cv_folds"], cfg["cv_repeats"], cfg["seed"],
        groups=data.feature_groups(train),
    )
    candidates = models.candidates(cfg["seed"])
    table = evaluation.compare(candidates, X_train, y_train, splits)
    table.to_csv(reports / "cv_results.csv", index=False, float_format="%.4f")
    log.info("\n%s", table.to_string(index=False, float_format="%.3f"))

    best_name = table.loc[0, "model"]
    best_model, best_params, tuned_cv_f1 = tune(
        best_name, candidates[best_name], X_train, y_train, splits
    )
    log.info("selected %s, tuned params: %s", best_name, best_params or "none")

    predictions = best_model.predict(X_test)
    report = classification_report(
        y_test, predictions, target_names=data.LABELS, output_dict=True, zero_division=0
    )
    metrics = {
        "model": best_name,
        "tuned_params": best_params,
        "cv_f1_macro_untuned": float(table.loc[0, "f1_macro_mean"]),
        "cv_f1_macro_tuned": tuned_cv_f1,
        "test_f1_macro": f1_score(y_test, predictions, average="macro"),
        "test_accuracy": report["accuracy"],
        "test_per_class": {label: report[label] for label in data.LABELS},
        "n_train": len(train),
        "n_test": len(test),
    }
    with open(reports / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("test macro F1 %.3f, accuracy %.3f", metrics["test_f1_macro"], metrics["test_accuracy"])

    ConfusionMatrixDisplay.from_predictions(
        y_test, predictions, display_labels=data.LABELS, cmap="Blues", colorbar=False
    )
    plt.title(f"{best_name} on the held-out test set")
    plt.tight_layout()
    plt.savefig(reports / "figures" / "confusion_matrix.png", dpi=150)
    plt.close()

    model_path = data.resolve(cfg["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, model_path)
    log.info("saved model to %s", model_path)


if __name__ == "__main__":
    main()
