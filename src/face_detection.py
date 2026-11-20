import cv2
import mediapipe as mp
import os

# Path to the face landmark model
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "models",
    "face_landmarker.task"
)

# Check if model exists
if not os.path.exists(MODEL_PATH):
    print("Face Landmarker model not found!")
    print(MODEL_PATH)
    exit()

print("Face Landmarker model found!")

# MediaPipe Tasks
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

# Configure Face Landmarker
options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1
)

# Open webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera Not Found")
    exit()

print("Camera Connected Successfully")
print("Press ESC to exit")

# Create Face Landmarker
with FaceLandmarker.create_from_options(options) as landmarker:

    while True:

        success, frame = camera.read()

        if not success:
            print("Frame Not Captured")
            break

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

        # Detect face landmarks
        result = landmarker.detect(mp_image)

        # Check if face detected
        if result.face_landmarks:

            for face_landmarks in result.face_landmarks:

                # Draw facial landmarks
                for landmark in face_landmarks:

                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])

                    cv2.circle(
                        frame,
                        (x, y),
                        1,
                        (0, 255, 0),
                        -1
                    )

        # Display camera
        cv2.imshow(
            "Neurological Screening Tool - Face Landmarks",
            frame
        )

        # ESC key
        if cv2.waitKey(1) == 27:
            break

# Release camera
camera.release()
cv2.destroyAllWindows()

print("Camera Closed Successfully")