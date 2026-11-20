import cv2

print("Opening Camera...")

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera Not Found")
    exit()

print("Camera Connected Successfully")

while True:

    success, frame = camera.read()

    if not success:
        print("Frame Not Captured")
        break

    cv2.imshow("Neurological Screening Tool v1.0", frame)

    key = cv2.waitKey(1)

    if key == 27:
        break

camera.release()

cv2.destroyAllWindows()

print("Camera Closed Successfully")