import cv2
import mediapipe as mp
import numpy as np
import os
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
    "face_landmarker.task"
)

RESULTS_DIR = os.path.join(PROJECT_ROOT, "data")
CSV_FILE = os.path.join(RESULTS_DIR, "eye_blink_results.csv")

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# 2. CHECK MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):
    print(f"Face Landmarker model not found: {MODEL_PATH}")
    raise SystemExit(1)

print("Face Landmarker model found!")


# ============================================================
# 3. SAVE RESULTS TO CSV
# ============================================================

def save_blink_result(duration, total_blinks, blink_rate):
    file_exists = os.path.exists(CSV_FILE) and os.path.getsize(CSV_FILE) > 0

    with open(CSV_FILE, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "Timestamp",
                "Duration_Seconds",
                "Total_Blinks",
                "Blink_Rate_Per_Minute"
            ])

        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            round(duration, 2),
            total_blinks,
            round(blink_rate, 2)
        ])

    print(f"Result saved to: {CSV_FILE}")


# ============================================================
# 4. MEDIAPIPE SETUP
# ============================================================

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1
)


# ============================================================
# 5. DISTANCE AND EAR FUNCTIONS
# ============================================================

def distance(point1, point2):
    return np.sqrt(
        (point1[0] - point2[0]) ** 2 +
        (point1[1] - point2[1]) ** 2
    )


def calculate_ear(eye_points):
    vertical_1 = distance(eye_points[1], eye_points[5])
    vertical_2 = distance(eye_points[2], eye_points[4])
    horizontal = distance(eye_points[0], eye_points[3])

    if horizontal == 0:
        return 0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


# ============================================================
# 6. EYE LANDMARKS
# ============================================================

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


# ============================================================
# 7. TEST SETTINGS
# ============================================================

EAR_THRESHOLD = 0.20
TEST_DURATION = 30

blink_count = 0
eye_closed = False
test_started = False
test_complete = False
start_time = None
blink_rate = 0.0


# ============================================================
# 8. OPEN CAMERA
# ============================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera not found or unavailable.")
    raise SystemExit(1)

print("Camera connected successfully.")
print("Press SPACE to start the eye blink test.")
print("Press R to restart. Press ESC to exit.")


# ============================================================
# 9. RUN EYE BLINK TEST
# ============================================================

try:
    with FaceLandmarker.create_from_options(options) as landmarker:

        while True:
            success, frame = camera.read()

            if not success:
                print("Frame not captured.")
                break

            frame = cv2.flip(frame, 1)
            height, width, _ = frame.shape

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            result = landmarker.detect(mp_image)

            if result.face_landmarks:
                face_landmarks = result.face_landmarks[0]

                left_eye_points = []
                right_eye_points = []

                for index in LEFT_EYE:
                    landmark = face_landmarks[index]
                    left_eye_points.append((
                        int(landmark.x * width),
                        int(landmark.y * height)
                    ))

                for index in RIGHT_EYE:
                    landmark = face_landmarks[index]
                    right_eye_points.append((
                        int(landmark.x * width),
                        int(landmark.y * height)
                    ))

                left_ear = calculate_ear(left_eye_points)
                right_ear = calculate_ear(right_eye_points)
                average_ear = (left_ear + right_ear) / 2.0

                for point in left_eye_points + right_eye_points:
                    cv2.circle(frame, point, 3, (0, 255, 0), -1)

                if test_started:
                    elapsed_time = time.time() - start_time
                    remaining_time = max(0, TEST_DURATION - elapsed_time)

                    if average_ear < EAR_THRESHOLD:
                        if not eye_closed:
                            eye_closed = True
                    else:
                        if eye_closed:
                            blink_count += 1
                            eye_closed = False

                    cv2.putText(
                        frame, f"Time: {remaining_time:.1f} sec",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 255), 2
                    )

                    cv2.putText(
                        frame, f"Blinks: {blink_count}",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2
                    )

                    cv2.putText(
                        frame, f"EAR: {average_ear:.2f}",
                        (20, 120), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2
                    )

                    if elapsed_time >= TEST_DURATION:
                        test_started = False
                        test_complete = True

                        blink_rate = (blink_count / TEST_DURATION) * 60

                        save_blink_result(
                            TEST_DURATION,
                            blink_count,
                            blink_rate
                        )

                        print("\n========== EYE BLINK TEST COMPLETE ==========")
                        print(f"Duration: {TEST_DURATION} seconds")
                        print(f"Total blinks: {blink_count}")
                        print(f"Blink rate: {blink_rate:.2f} per minute")
                        print("==============================================")

                elif test_complete:
                    cv2.putText(
                        frame, "TEST COMPLETE",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 0), 2
                    )

                    cv2.putText(
                        frame, f"Total Blinks: {blink_count}",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2
                    )

                    cv2.putText(
                        frame, f"Blink Rate: {blink_rate:.2f}/min",
                        (20, 120), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2
                    )

                    cv2.putText(
                        frame, "Press R to Restart",
                        (20, 160), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2
                    )

                else:
                    cv2.putText(
                        frame, "Press SPACE to Start Test",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 255), 2
                    )

            else:
                cv2.putText(
                    frame, "Face not detected",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2
                )

            cv2.imshow("Parkinson's Research Prototype - Eye Blink Test", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == 32 and not test_started:
                blink_count = 0
                eye_closed = False
                test_started = True
                test_complete = False
                start_time = time.time()
                print("30-second eye blink test started.")

            elif key == ord("r"):
                blink_count = 0
                eye_closed = False
                test_started = True
                test_complete = False
                start_time = time.time()
                print("Eye blink test restarted.")

            elif key == 27:
                break

finally:
    camera.release()
    cv2.destroyAllWindows()
    print("Camera closed successfully.")