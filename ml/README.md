# NeuroVision experimental ML module

## Contents
- `train_model.py`: trains/evaluates a Random Forest using the dataset's `diagnosed` yes/no label.
- `model.joblib`: model refitted on all 489 uploaded dataset rows.
- `metrics.json`: stratified 80/20 holdout evaluation from the fixed random seed.
- `predict.py`: predicts only from CSV rows containing the research dataset's matching extracted feature columns.
- `feature_columns.json`: required feature names.

## Setup
From the project root:
```bash
pip install pandas scikit-learn joblib
```

Place the downloaded research CSV at `data/severity_dataset.csv`, then run:
```bash
python ml/train_model.py data/severity_dataset.csv
```

For a compatible research feature CSV:
```bash
python ml/predict.py path/to/research_features.csv
```

## Critical compatibility limitation
The current NeuroVision webcam scripts save aggregate summaries:
- eye blink: duration, blink count, blink rate
- finger stability: duration, measurement count, average frame-to-frame pixel movement

The research dataset uses a different set of extracted finger-tapping features (wrist movement, periodicity, frequency, amplitude, speed, etc.). Therefore this model CANNOT be called on the current webcam CSVs. Do not display its output as a prediction from the webcam tests until the same feature extraction protocol is implemented and independently validated.

The holdout metrics are exploratory and record-level. The dataset may contain multiple recordings from the same participant; without participant-grouped splitting, the metrics can be optimistic. This is a research prototype, not a clinical diagnostic tool.
