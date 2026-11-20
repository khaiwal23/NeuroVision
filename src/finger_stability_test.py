import cv2
import mediapipe as mp
import os
import math
import time
import csv
from datetime import datetime


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "hand_landmarker.task"
)

RESULTS_DIR = os.path.join(PROJECT_ROOT, "data")
CSV_FILE = os.path.join(RESULTS_DIR, "finger_stability_results.csv")

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# 2. CHECK MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):
    print(f"Hand Landmarker model not found: {MODEL_PATH}")
    raise SystemExit(1)


# ============================================================
# 3. SAVE RESULTS
# ============================================================

def save_result(duration, measurements, average_movement):
    file_exists = os.path.exists(CSV_FILE) and os.path.getsize(CSV_FILE) > 0

    with open(CSV_FILE, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "Timestamp",
                "Actual_Duration_Seconds",
                "Total_Measurements",
                "Average_Frame_Movement_Pixels"
            ])

        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            round(duration, 2),
            measurements,
            round(average_movement, 4)
        ])

    print(f"Result saved to: {CSV_FILE}")


# ============================================================
# 4. MEDIAPIPE SETUP
# ============================================================

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)


# ============================================================
# 5. TEST SETTINGS
# ============================================================

TEST_DURATION = 30


# ============================================================
# 6. OPEN CAMERA
# ============================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera not found or unavailable.")
    raise SystemExit(1)


print("==========================================")
print("FINGER STABILITY TEST")
print("Duration: 30 seconds")
print("Keep your index finger in view.")
print("Press ESC to stop early.")
print("==========================================")


# ============================================================
# 7. VARIABLES
# ============================================================

previous_x = None
previous_y = None

movement_values = []

start_time = time.time()
end_reason = "completed"


# ============================================================
# 8. RUN TEST
# ============================================================

try:
    with HandLandmarker.create_from_options(options) as landmarker:

        while True:
            success, frame = camera.read()

            if not success:
                print("Frame not captured.")
                end_reason = "camera_error"
                break

            frame = cv2.flip(frame, 1)
            height, width, _ = frame.shape

            elapsed_time = time.time() - start_time
            remaining_time = max(0, TEST_DURATION - elapsed_time)

            if elapsed_time >= TEST_DURATION:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            result = landmarker.detect(mp_image)

            if result.hand_landmarks:
                hand_landmarks = result.hand_landmarks[0]
                index_tip = hand_landmarks[8]

                x = int(index_tip.x * width)
                y = int(index_tip.y * height)

                cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)

                if previous_x is not None and previous_y is not None:
                    movement = math.sqrt(
                        (x - previous_x) ** 2 +
                        (y - previous_y) ** 2
                    )

                    movement_values.append(movement)

                    average_movement = (
                        sum(movement_values) / len(movement_values)
                    )

                    cv2.putText(
                        frame,
                        f"Movement: {movement:.2f} pixels",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"Average: {average_movement:.2f} pixels",
                        (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 0),
                        2
                    )

                previous_x = x
                previous_y = y

            else:
                cv2.putText(
                    frame,
                    "Show your hand to the camera",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

                previous_x = None
                previous_y = None

            cv2.putText(
                frame,
                f"Time left: {remaining_time:.1f}s",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            cv2.imshow("Parkinson's Research Prototype - Finger Stability", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                end_reason = "stopped_early"
                print("Test stopped manually.")
                break

finally:
    camera.release()
    cv2.destroyAllWindows()


# ============================================================
# 9. FINAL RESULTS
# ============================================================

actual_duration = min(time.time() - start_time, TEST_DURATION)

print("\n==========================================")
print("FINGER STABILITY TEST RESULTS")
print("==========================================")
print(f"Actual duration: {actual_duration:.2f} seconds")
print(f"Test status: {end_reason}")
print(f"Total measurements: {len(movement_values)}")

if movement_values:
    final_average = sum(movement_values) / len(movement_values)

    print(f"Average frame movement: {final_average:.4f} pixels")

    save_result(
        actual_duration,
        len(movement_values),
        final_average
    )
else:
    print("No movement measurements collected; result not saved.")

print("==========================================")
print("Camera closed.")