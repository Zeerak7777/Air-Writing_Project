"""
Quick standalone test: opens the camera, grabs a handful of frames,
and saves the last one as a JPEG -- completely separate from Flask,
so we can tell if the camera itself is the problem.
"""
import cv2

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print("Camera failed to open at all.")
else:
    print("Camera opened. Reading a few frames to let it warm up...")
    frame = None
    for i in range(10):
        success, frame = cap.read()
        print(f"Frame {i}: success={success}, "
              f"shape={frame.shape if success else None}, "
              f"mean_brightness={frame.mean() if success else None}")

    if frame is not None:
        cv2.imwrite("camera_test.jpg", frame)
        print("Saved camera_test.jpg -- open it in Finder and look at it.")

cap.release()
