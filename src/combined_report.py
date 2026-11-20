import csv
import os
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"

EYE_FILE = DATA_DIR / "eye_blink_results.csv"
FINGER_FILE = DATA_DIR / "finger_stability_results.csv"
REPORT_FILE = REPORTS_DIR / "combined_screening_report.csv"


def read_latest_result(file_path):
    if not file_path.exists():
        return None

    with open(file_path, "r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    return rows[-1] if rows else None


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    eye_result = read_latest_result(EYE_FILE)
    finger_result = read_latest_result(FINGER_FILE)

    print("\n========================================")
    print(" PARKINSON'S RESEARCH PROTOTYPE")
    print(" COMBINED TEST REPORT")
    print("========================================")
    print(f"Report generated: {datetime.now().isoformat(timespec='seconds')}")

    if eye_result:
        print("\nEYE BLINK TEST")
        print(f"Timestamp: {eye_result.get('Timestamp', 'N/A')}")
        print(f"Duration: {eye_result.get('Duration_Seconds', 'N/A')} seconds")
        print(f"Total blinks: {eye_result.get('Total_Blinks', 'N/A')}")
        print(f"Blink rate: {eye_result.get('Blink_Rate_Per_Minute', 'N/A')} per minute")
    else:
        print("\nEye blink results not found.")

    if finger_result:
        print("\nFINGER STABILITY TEST")
        print(f"Timestamp: {finger_result.get('Timestamp', 'N/A')}")
        print(f"Duration: {finger_result.get('Actual_Duration_Seconds', 'N/A')} seconds")
        print(f"Measurements: {finger_result.get('Total_Measurements', 'N/A')}")
        print(
            "Average frame movement: "
            f"{finger_result.get('Average_Frame_Movement_Pixels', 'N/A')} pixels"
        )
    else:
        print("\nFinger stability results not found.")

    print("\nNOTE: These are experimental measurements, not a Parkinson's diagnosis.")
    print("========================================")

    with open(REPORT_FILE, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Test", "Timestamp", "Duration_Seconds", "Metric", "Value"])

        if eye_result:
            writer.writerow([
                "Eye Blink",
                eye_result.get("Timestamp", ""),
                eye_result.get("Duration_Seconds", ""),
                "Total Blinks",
                eye_result.get("Total_Blinks", "")
            ])
            writer.writerow([
                "Eye Blink",
                eye_result.get("Timestamp", ""),
                eye_result.get("Duration_Seconds", ""),
                "Blink Rate Per Minute",
                eye_result.get("Blink_Rate_Per_Minute", "")
            ])

        if finger_result:
            writer.writerow([
                "Finger Stability",
                finger_result.get("Timestamp", ""),
                finger_result.get("Actual_Duration_Seconds", ""),
                "Total Measurements",
                finger_result.get("Total_Measurements", "")
            ])
            writer.writerow([
                "Finger Stability",
                finger_result.get("Timestamp", ""),
                finger_result.get("Actual_Duration_Seconds", ""),
                "Average Frame Movement Pixels",
                finger_result.get("Average_Frame_Movement_Pixels", "")
            ])

    print(f"\nCombined report saved to:\n{REPORT_FILE}")


if __name__ == "__main__":
    main()