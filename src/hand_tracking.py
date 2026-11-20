import cv2
import mediapipe as mp
import os


# ============================================================
# 1. MODEL PATH
# ============================================================

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "models",
    "hand_landmarker.task"
)


# ============================================================
# 2. CHECK MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):
    print("Hand Landmarker model not found!")
    print(MODEL_PATH)
    exit()

print("Hand Landmarker model found!")


# ============================================================
# 3. MEDIAPIPE SETUP
# ============================================================

BaseOptions = mp.tasks.BaseOptions

VisionRunningMode = mp.tasks.vision.RunningMode

HandLandmarker = mp.tasks.vision.HandLandmarker

HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions


# ============================================================
# 4. CONFIGURE HAND LANDMARKER
# ============================================================

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2
)


# ============================================================
# 5. OPEN CAMERA
# ============================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera Not Found")
    exit()

print("Camera Connected Successfully")
print("Hand Tracking Started")
print("Press ESC to exit")


# ============================================================
# 6. START HAND LANDMARKER
# ============================================================

with HandLandmarker.create_from_options(options) as landmarker:

    while True:

        success, frame = camera.read()

        if not success:
            print("Frame Not Captured")
            break


        # Mirror camera
        frame = cv2.flip(frame, 1)


        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        # Convert frame to MediaPipe Image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )


        # Detect hands
        result = landmarker.detect(mp_image)


        # ====================================================
        # IF HAND DETECTED
        # ====================================================

        if result.hand_landmarks:

            height, width, _ = frame.shape


            # Loop through all detected hands
            for hand_landmarks in result.hand_landmarks:


                # Loop through 21 landmarks
                for landmark in hand_landmarks:

                    x = int(landmark.x * width)
                    y = int(landmark.y * height)


                    # Draw landmark
                    cv2.circle(
                        frame,
                        (x, y),
                        5,
                        (0, 255, 0),
                        -1
                    )


        # ====================================================
        # DISPLAY TEXT
        # ====================================================

        cv2.putText(
            frame,
            "Hand Tracking Active",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


        # ====================================================
        # SHOW CAMERA
        # ====================================================

        cv2.imshow(
            "Neurological Screening Tool - Hand Tracking",
            frame
        )


        # ESC TO EXIT
        if cv2.waitKey(1) & 0xFF == 27:
            break


# ============================================================
# 7. CLOSE CAMERA
# ============================================================

camera.release()

cv2.destroyAllWindows()

print("Camera Closed Successfully")