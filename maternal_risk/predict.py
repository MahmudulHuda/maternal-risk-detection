"""Score a CSV of patient readings with the saved model.

    python -m maternal_risk.predict readings.csv -o scored.csv
"""

import argparse
import sys

import joblib
import pandas as pd

from maternal_risk import data


def predict(model, readings):
    missing = set(data.FEATURES) - set(readings.columns)
    if missing:
        raise ValueError(f"input is missing columns: {sorted(missing)}")
    X = readings[data.FEATURES]
    if X.isna().any().any():
        raise ValueError("input contains missing values; the model was trained on complete rows")

    out = readings.copy()
    out["predicted_risk"] = [data.LABELS[i] for i in model.predict(X)]
    probabilities = model.predict_proba(X)
    for i, label in enumerate(data.LABELS):
        out[f"p_{label}"] = probabilities[:, i].round(3)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input", help="CSV with the six feature columns")
    parser.add_argument("-o", "--output", help="where to write predictions (default: stdout)")
    args = parser.parse_args()

    model_path = data.resolve(data.load_config()["model_path"])
    if not model_path.exists():
        sys.exit(f"No model at {model_path}. Run `python -m maternal_risk.train` first.")

    scored = predict(joblib.load(model_path), pd.read_csv(args.input))
    scored.to_csv(args.output or sys.stdout, index=False)


if __name__ == "__main__":
    main()
