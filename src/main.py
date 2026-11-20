import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent

SCRIPTS = {
    "1": ("Eye Blink Test", SRC_DIR / "eye_blink.py"),
    "2": ("Finger Stability Test", SRC_DIR / "finger_stability_test.py"),
    "3": ("Hand Tracking", SRC_DIR / "hand_tracking.py"),
    "4": ("Finger Tracking", SRC_DIR / "finger_tracking.py"),
    "5": ("Face Detection", SRC_DIR / "face_detection.py"),
    "6": ("Tremor Detection", SRC_DIR / "tremor_detection.py"),
}


def main():
    while True:
        print("\n" + "=" * 42)
        print(" PARKINSON'S SCREENING RESEARCH PROTOTYPE")
        print("=" * 42)
        print("1. Eye Blink Test")
        print("2. Finger Stability Test")
        print("3. Hand Tracking")
        print("4. Finger Tracking")
        print("5. Face Detection")
        print("6. Tremor Detection")
        print("0. Exit")
        print("=" * 42)

        choice = input("Select an option: ").strip()

        if choice == "0":
            print("Application closed.")
            break

        if choice not in SCRIPTS:
            print("Invalid option. Please try again.")
            continue

        name, script_path = SCRIPTS[choice]

        if not script_path.exists():
            print(f"File not found: {script_path}")
            continue

        print(f"\nStarting {name}...")
        print("Press ESC in the camera window to exit the test.")

        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(PROJECT_ROOT),
                check=False
            )

            if result.returncode != 0:
                print(f"{name} exited with an error.")

        except Exception as error:
            print(f"Could not start {name}: {error}")

        input("\nPress Enter to return to the main menu...")


if __name__ == "__main__":
    main()