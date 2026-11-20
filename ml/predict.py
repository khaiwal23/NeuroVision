"""
Predict from a CSV containing the same extracted feature columns as the
research dataset. This is NOT compatible with the current webcam summary CSVs.
Usage:
    python ml/predict.py path/to/research_features.csv
"""
import sys
from pathlib import Path
import joblib
import pandas as pd

HERE = Path(__file__).resolve().parent
bundle = joblib.load(HERE / "model.joblib")
model = bundle["pipeline"]
columns = bundle["feature_columns"]

if len(sys.argv) < 2:
    raise SystemExit("Usage: python ml/predict.py <feature_rows.csv>")

rows = pd.read_csv(sys.argv[1])
missing = [c for c in columns if c not in rows.columns]
if missing:
    raise SystemExit(
        "Input does not contain the research dataset's required features. "
        "Missing columns: " + ", ".join(missing[:12])
    )

X = rows[columns].apply(pd.to_numeric, errors="coerce")
prob = model.predict_proba(X)[:, 1]
pred = model.predict(X)
for i, (label, score) in enumerate(zip(pred, prob), start=1):
    print(f"Row {i}: {'diagnosed_yes' if label == 1 else 'control_no'} "
          f"(experimental model score={score:.3f})")
print("Warning: research-data predictions only; not a medical diagnosis.")
