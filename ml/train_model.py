"""
Train/evaluate the experimental Parkinson's-vs-control model.
Usage from project root:
    python ml/train_model.py data/severity_dataset.csv
"""
import sys, json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report)

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "severity_dataset.csv"
OUT = Path(__file__).resolve().parent
df = pd.read_csv(CSV_PATH)

if "diagnosed" not in df.columns:
    raise ValueError("CSV must contain a 'diagnosed' column with yes/no labels.")
y = df["diagnosed"].astype(str).str.strip().str.lower().map({"no": 0, "yes": 1})
if y.isna().any():
    raise ValueError("The 'diagnosed' column must contain only yes/no values.")

excluded = {
    "Unnamed: 0", "diagnosed", "filename", "hand", "gender", "Rating",
    "Rating1", "Rating2", "Rating3", "Rating4", "Comment1", "Comment3",
    "Comment4", "CheckDifficult1", "CheckDifficult3", "CheckDifficult4"
}
X = df.drop(columns=[c for c in df.columns if c in excluded])
X = X.select_dtypes(include=[np.number])
X = X.loc[:, X.nunique(dropna=False) > 1]
feature_columns = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=400, class_weight="balanced",
        min_samples_leaf=2, random_state=42, n_jobs=-1
    ))
])
model.fit(X_train, y_train)
pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:, 1]
metrics = {
    "rows": int(len(df)),
    "features": len(feature_columns),
    "accuracy": float(accuracy_score(y_test, pred)),
    "precision_diagnosed": float(precision_score(y_test, pred, zero_division=0)),
    "recall_diagnosed": float(recall_score(y_test, pred, zero_division=0)),
    "f1_diagnosed": float(f1_score(y_test, pred, zero_division=0)),
    "roc_auc": float(roc_auc_score(y_test, prob)),
    "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    "note": "Exploratory record-level holdout only; not clinically validated."
}
# Refit final artifact on all records after reporting holdout metrics.
model.fit(X, y)
joblib.dump({"pipeline": model, "feature_columns": feature_columns}, OUT / "model.joblib")
(OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
(OUT / "feature_columns.json").write_text(json.dumps(feature_columns, indent=2), encoding="utf-8")
print(json.dumps(metrics, indent=2))
print("Saved model to", OUT / "model.joblib")
