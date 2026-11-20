
from flask import Flask, jsonify, request
from flask_cors import CORS
from pathlib import Path
import csv
import subprocess
import sys
import json
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)

# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"
ML_DIR = BASE_DIR / "ml"

MODEL_FILE = ML_DIR / "model.joblib"
METRICS_FILE = ML_DIR / "metrics.json"

TEST_SCRIPTS = {
    "eye-blink": BASE_DIR / "src" / "eye_blink.py",
    "finger-stability": BASE_DIR / "src" / "finger_stability_test.py"
}

active_tests = {}

# ============================================================
# 2. LOAD ML MODEL
# ============================================================

ml_model = None
feature_columns = []
ml_metrics = {}

try:
    model_bundle = joblib.load(MODEL_FILE)
    ml_model = model_bundle["pipeline"]
    feature_columns = model_bundle["feature_columns"]

    if METRICS_FILE.exists():
        with METRICS_FILE.open("r", encoding="utf-8") as file:
            ml_metrics = json.load(file)

    print("ML model loaded successfully.")

except Exception as error:
    print("ML model could not be loaded:", error)


# ============================================================
# 3. CSV READER
# ============================================================

def read_csv_file(file_path):

    if not file_path.exists():
        return {
            "available": False,
            "message": f"File not found: {file_path.name}",
            "count": 0,
            "records": []
        }

    try:
        with file_path.open(
            "r", newline="", encoding="utf-8-sig"
        ) as file:
            reader = csv.DictReader(file)
            records = list(reader)

        return {
            "available": True,
            "count": len(records),
            "records": records
        }

    except (OSError, csv.Error, UnicodeError) as error:
        return {
            "available": False,
            "message": str(error),
            "count": 0,
            "records": []
        }


# ============================================================
# 4. COMBINED REPORT
# ============================================================

def generate_combined_report():

    eye_data = read_csv_file(
        DATA_DIR / "eye_blink_results.csv"
    )

    finger_data = read_csv_file(
        DATA_DIR / "finger_stability_results.csv"
    )

    combined_records = []

    if eye_data["available"]:
        for record in eye_data["records"]:

            timestamp = record.get("Timestamp", "")

            combined_records.append({
                "Test": "Eye Blink",
                "Metric": "Blink Rate Per Minute",
                "Value": record.get("Blink_Rate_Per_Minute", ""),
                "Timestamp": timestamp
            })

            combined_records.append({
                "Test": "Eye Blink",
                "Metric": "Total Blinks",
                "Value": record.get("Total_Blinks", ""),
                "Timestamp": timestamp
            })

    if finger_data["available"]:
        for record in finger_data["records"]:

            timestamp = record.get("Timestamp", "")

            combined_records.append({
                "Test": "Finger Stability",
                "Metric": "Average Frame Movement Pixels",
                "Value": record.get(
                    "Average_Frame_Movement_Pixels", ""
                ),
                "Timestamp": timestamp
            })

            combined_records.append({
                "Test": "Finger Stability",
                "Metric": "Total Measurements",
                "Value": record.get("Total_Measurements", ""),
                "Timestamp": timestamp
            })

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    report_path = REPORTS_DIR / "combined_screening_report.csv"

    fieldnames = ["Test", "Metric", "Value", "Timestamp"]

    with report_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(combined_records)

    return {
        "available": True,
        "count": len(combined_records),
        "records": combined_records
    }


# ============================================================
# 5. HOME AND HEALTH APIs
# ============================================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "project": "NeuroVision - Parkinson's Research Prototype",
        "status": "running",
        "message": "Backend API is working",
        "ml_model_loaded": ml_model is not None,
        "disclaimer": "Research prototype only. Not a medical diagnosis."
    })


@app.route("/api/health", methods=["GET"])
def health():

    return jsonify({
        "status": "healthy",
        "backend": "Flask"
    })


# ============================================================
# 6. RESULTS APIs
# ============================================================

@app.route("/api/results", methods=["GET"])
def all_results():

    eye_data = read_csv_file(
        DATA_DIR / "eye_blink_results.csv"
    )

    finger_data = read_csv_file(
        DATA_DIR / "finger_stability_results.csv"
    )

    combined_data = generate_combined_report()

    return jsonify({
        "eye_blink": eye_data,
        "finger_stability": finger_data,
        "combined_report": combined_data
    })


@app.route("/api/results/eye-blink", methods=["GET"])
def eye_blink_results():

    return jsonify(
        read_csv_file(DATA_DIR / "eye_blink_results.csv")
    )


@app.route("/api/results/finger-stability", methods=["GET"])
def finger_stability_results():

    return jsonify(
        read_csv_file(DATA_DIR / "finger_stability_results.csv")
    )


@app.route("/api/results/combined-report", methods=["GET"])
def combined_report():

    return jsonify(generate_combined_report())


# ============================================================
# 7. ML STATUS API
# ============================================================

@app.route("/api/ml/status", methods=["GET"])
def ml_status():

    return jsonify({
        "model_loaded": ml_model is not None,
        "model_type": "Random Forest Classifier",
        "dataset": "Research finger-tapping dataset",
        "features_required": len(feature_columns),
        "metrics": ml_metrics,
        "webcam_compatible": False,
        "message": (
            "Model is trained on research finger-tapping features. "
            "Current webcam summary CSVs do not contain the same features."
        ),
        "disclaimer": (
            "Experimental research model. "
            "Not clinically validated and not a medical diagnosis."
        )
    })


# ============================================================
# 8. ML PREDICTION API
# ============================================================

@app.route("/api/ml/predict", methods=["POST"])
def ml_predict():

    if ml_model is None:
        return jsonify({
            "success": False,
            "message": "ML model is not loaded."
        }), 503

    data = request.get_json(silent=True)

    if not data or not isinstance(data.get("features"), dict):
        return jsonify({
            "success": False,
            "message": "Send JSON containing a features object."
        }), 400

    supplied_features = data["features"]

    missing = [
        column for column in feature_columns
        if column not in supplied_features
    ]

    if missing:
        return jsonify({
            "success": False,
            "message": "Required research dataset features are missing.",
            "missing_features": missing[:20],
            "missing_count": len(missing),
            "webcam_compatibility": False
        }), 400

    try:
        feature_row = pd.DataFrame(
            [[supplied_features[column] for column in feature_columns]],
            columns=feature_columns
        )

        feature_row = feature_row.apply(
            pd.to_numeric, errors="coerce"
        )

        if feature_row.isna().all(axis=1).iloc[0]:
            return jsonify({
                "success": False,
                "message": "No valid numeric feature values supplied."
            }), 400

        prediction = int(ml_model.predict(feature_row)[0])

        probabilities = ml_model.predict_proba(feature_row)[0]

        class_index = list(ml_model.classes_).index(1)
        diagnosed_score = float(probabilities[class_index])

        return jsonify({
            "success": True,
            "experimental_result": (
                "Research-model positive class"
                if prediction == 1
                else "Research-model control class"
            ),
            "predicted_label": prediction,
            "model_score": round(diagnosed_score, 4),
            "input_type": "Research dataset extracted features",
            "webcam_based_prediction": False,
            "disclaimer": (
                "Experimental dataset classification only. "
                "This result cannot diagnose or rule out Parkinson's disease."
            )
        })

    except Exception as error:

        return jsonify({
            "success": False,
            "message": f"Prediction failed: {str(error)}"
        }), 500


# ============================================================
# 9. START WEBCAM TESTS
# ============================================================

@app.route(
    "/api/tests/start/<test_name>",
    methods=["POST"]
)
def start_screening_test(test_name):

    if test_name not in TEST_SCRIPTS:
        return jsonify({
            "success": False,
            "message": "Unknown screening test."
        }), 404

    script_path = TEST_SCRIPTS[test_name]

    if not script_path.is_file():
        return jsonify({
            "success": False,
            "message": f"Test script not found: {script_path.name}"
        }), 404

    existing_process = active_tests.get(test_name)

    if existing_process and existing_process.poll() is None:
        return jsonify({
            "success": False,
            "message": "This test is already running."
        }), 409

    try:
        process = subprocess.Popen(
            [sys.executable, str(script_path)],
            cwd=str(BASE_DIR)
        )

        active_tests[test_name] = process

        return jsonify({
            "success": True,
            "message": f"{test_name} test started.",
            "pid": process.pid
        }), 200

    except Exception as error:

        return jsonify({
            "success": False,
            "message": str(error)
        }), 500


# ============================================================
# 10. TEST STATUS
# ============================================================

@app.route("/api/tests/status", methods=["GET"])
def screening_test_status():

    return jsonify({
        name: (
            "running"
            if process.poll() is None
            else "finished"
        )
        for name, process in active_tests.items()
    })


# ============================================================
# 11. RUN FLASK
# ============================================================

if __name__ == "__main__":

    print("=" * 55)
    print("NEUROVISION - PARKINSON'S RESEARCH PROTOTYPE")
    print("Flask Backend Starting...")
    print("ML Model:", "Loaded" if ml_model is not None else "Unavailable")
    print("=" * 55)

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
