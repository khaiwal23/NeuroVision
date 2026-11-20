import cv2
import mediapipe as mp
import os
import math


# ============================================================
# MODEL PATH
# ============================================================

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "models",
    "hand_landmarker.task"
)


# ============================================================
# CHECK MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):
    print("Hand Landmarker model not found!")
    exit()


# ============================================================
# MEDIAPIPE SETUP
# ============================================================

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions


# ============================================================
# HAND LANDMARKER OPTIONS
# ============================================================

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)


# ============================================================
# OPEN CAMERA
# ============================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera Not Found")
    exit()

print("Camera Connected Successfully")
print("Finger Stability Test Started")
print("Keep your index finger steady")
print("Press ESC to exit")


# ============================================================
# VARIABLES
# ============================================================

previous_x = None
previous_y = None

movement_values = []


# ============================================================
# START HAND LANDMARKER
# ============================================================

with HandLandmarker.create_from_options(options) as landmarker:

    while True:

        success, frame = camera.read()

        if not success:
            print("Frame Not Captured")
            break


        # Mirror camera
        frame = cv2.flip(frame, 1)

        height, width, _ = frame.shape


        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        # Convert to MediaPipe Image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )


        # Detect hand
        result = landmarker.detect(mp_image)


        # ====================================================
        # HAND DETECTED
        # ====================================================

        if result.hand_landmarks:

            hand_landmarks = result.hand_landmarks[0]


            # Index Finger Tip = Landmark 8
            index_tip = hand_landmarks[8]


            # Convert normalized coordinates to pixels
            x = int(index_tip.x * width)
            y = int(index_tip.y * height)


            # Draw circle
            cv2.circle(
                frame,
                (x, y),
                15,
                (0, 255, 0),
                -1
            )


            # =================================================
            # MOVEMENT CALCULATION
            # =================================================

            if previous_x is not None:

                # Calculate distance between positions
                distance = math.sqrt(
                    (x - previous_x) ** 2 +
                    (y - previous_y) ** 2
                )


                # Save movement value
                movement_values.append(distance)


                # Calculate average movement
                average_movement = sum(
                    movement_values
                ) / len(movement_values)


                # Display current movement
                cv2.putText(
                    frame,
                    f"Current Movement: {distance:.2f}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )


                # Display average movement
                cv2.putText(
                    frame,
                    f"Average Movement: {average_movement:.2f}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )


            # Save current position
            previous_x = x
            previous_y = y


        else:

            cv2.putText(
                frame,
                "Show Your Index Finger",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )


            previous_x = None
            previous_y = None


        # ====================================================
        # SHOW CAMERA
        # ====================================================

        cv2.imshow(
            "Neurological Screening Tool - Finger Stability",
            frame
        )


        # ESC TO EXIT
        if cv2.waitKey(1) & 0xFF == 27:
            break


# ============================================================
# CLOSE CAMERA
# ============================================================

camera.release()
cv2.destroyAllWindows()


# ============================================================
# FINAL RESULT
# ============================================================

print("\n==============================")

print("FINGER STABILITY TEST COMPLETE")

print("==============================")


if len(movement_values) > 0:

    final_average = sum(
        movement_values
    ) / len(movement_values)

    print(
        f"Average Finger Movement: {final_average:.2f}"
    )

    print(
        f"Total Measurements: {len(movement_values)}"
    )

else:

    print("No movement data collected")


print("==============================")

print("Camera Closed Successfully")