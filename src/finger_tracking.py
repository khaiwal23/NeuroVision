import cv2
import mediapipe as mp
import os


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
    print(MODEL_PATH)
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
print("Index Finger Tracking Started")
print("Press ESC to exit")


# Previous finger position
previous_x = None
previous_y = None


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

            # First detected hand
            hand_landmarks = result.hand_landmarks[0]


            # Index Finger Tip = Landmark 8
            index_tip = hand_landmarks[8]


            # Convert normalized coordinates to pixels
            x = int(index_tip.x * width)
            y = int(index_tip.y * height)


            # Draw circle on index finger
            cv2.circle(
                frame,
                (x, y),
                15,
                (0, 255, 0),
                -1
            )


            # Display position
            cv2.putText(
                frame,
                f"X: {x}  Y: {y}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )


            # =================================================
            # MOVEMENT CALCULATION
            # =================================================

            if previous_x is not None:

                movement_x = abs(x - previous_x)
                movement_y = abs(y - previous_y)

                total_movement = movement_x + movement_y


                cv2.putText(
                    frame,
                    f"Movement: {total_movement}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )


            # Save current position
            previous_x = x
            previous_y = y


        # ====================================================
        # NO HAND DETECTED
        # ====================================================

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


            # Reset previous position
            previous_x = None
            previous_y = None


        # ====================================================
        # SHOW CAMERA
        # ====================================================

        cv2.imshow(
            "Neurological Screening Tool - Finger Tracking",
            frame
        )


        # ESC to exit
        if cv2.waitKey(1) & 0xFF == 27:
            break


# ============================================================
# CLOSE CAMERA
# ============================================================

camera.release()
cv2.destroyAllWindows()

print("Camera Closed Successfully")