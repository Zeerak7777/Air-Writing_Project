"""
app.py
------
Flask app: webcam -> MediaPipe hand tracking -> fingertip drawing ->
CNN letter recognition OR geometric shape recognition -> typed text.

Gestures:
  index finger only        -> draw (pen down)
  open palm (all 5 up)     -> pause / lift pen (letter NOT finished)
  index + middle only      -> toggle between LETTER mode and SHAPE mode
  fist                     -> finish: recognize using the active mode,
                               then clear the canvas
  pinky finger only        -> space
  thumb only                -> backspace

Run with:  python3 app.py     then open http://127.0.0.1:5000
"""

import cv2
import numpy as np
from flask import Flask, Response, render_template, jsonify

from hand_tracker import HandTracker
from recognizer import LetterRecognizer

app = Flask(__name__)

CAM_INDEX = 0
GESTURE_HOLD_FRAMES = 8
MIN_STROKE_POINTS = 5
CONFIDENCE_THRESHOLD = 0.4

tracker = HandTracker(max_hands=1)

try:
    recognizer = LetterRecognizer("model/emnist_cnn.h5")
    MODEL_LOADED = True
except Exception as e:
    print(f"[WARN] Could not load model ({e}). Run train_model.py first.")
    recognizer = None
    MODEL_LOADED = False

state = {
    "typed_text": "",
    "last_letter": "",
    "last_confidence": 0.0,
    "mode": "letter",
}

GESTURE_NONE = "none"
GESTURE_DRAW = "draw"
GESTURE_FIST = "fist"
GESTURE_PAUSE = "pause"
GESTURE_SPACE = "space"
GESTURE_THUMB = "thumb"
GESTURE_TOGGLE_MODE = "toggle_mode"


def classify_gesture(fingers):
    if fingers == [False, True, False, False, False]:
        return GESTURE_DRAW
    if fingers == [False, False, False, False, False]:
        return GESTURE_FIST
    if fingers == [True, True, True, True, True]:
        return GESTURE_PAUSE
    if fingers == [False, False, False, False, True]:
        return GESTURE_SPACE
    if fingers == [True, False, False, False, False]:
        return GESTURE_THUMB
    if fingers == [False, True, True, False, False]:
        return GESTURE_TOGGLE_MODE
    return GESTURE_NONE


def detect_shape(canvas, bbox):
    x_min, y_min, x_max, y_max = bbox
    pad = 20
    x_min = max(0, x_min - pad)
    y_min = max(0, y_min - pad)
    x_max = min(canvas.shape[1], x_max + pad)
    y_max = min(canvas.shape[0], y_max + pad)
    crop = canvas[y_min:y_max, x_min:x_max]
    if crop.size == 0:
        return None

    ys, xs = np.where(crop > 0)
    if len(xs) < 20:
        return None
    points = np.column_stack([xs, ys]).astype(np.int32)
    hull = cv2.convexHull(points)

    area = cv2.contourArea(hull)
    if area < 400:
        return None

    perimeter = cv2.arcLength(hull, True)
    if perimeter == 0:
        return None

    approx = cv2.approxPolyDP(hull, 0.02 * perimeter, True)
    vertices = len(approx)
    circularity = 4 * np.pi * area / (perimeter ** 2)

    if vertices == 3:
        return "triangle"
    if vertices == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspect = w / float(h) if h else 0
        return "square" if 0.85 <= aspect <= 1.15 else "rectangle"
    if vertices == 5:
        return "pentagon"
    if vertices == 6:
        return "hexagon"
    if circularity > 0.85:
        return "circle"
    return None


def generate_frames():
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_AVFOUNDATION)
    print(">>> Camera opened" if cap.isOpened() else "!!! CAMERA FAILED TO OPEN")

    canvas = None
    stroke_points = []
    prev_point = None
    gesture_counter = 0
    stable_gesture = GESTURE_NONE
    last_committed = GESTURE_NONE

    while True:
        success, frame = cap.read()
        if not success:
            print("!!! FAILED TO READ FRAME - stopping stream")
            break

        frame = cv2.flip(frame, 1)
        if canvas is None or canvas.shape[:2] != frame.shape[:2]:
            canvas = np.zeros(frame.shape[:2], dtype=np.uint8)

        results = tracker.process(frame)
        current_gesture = GESTURE_NONE
        fingertip = None

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            handedness = "Right"
            if results.multi_handedness:
                handedness = results.multi_handedness[0].classification[0].label
            tracker.draw_landmarks(frame, hand)
            px = tracker.get_landmark_pixel_coords(hand, frame.shape)
            current_gesture = classify_gesture(tracker.fingers_up(px, handedness))
            fingertip = px[8]

        if current_gesture == stable_gesture:
            gesture_counter += 1
        else:
            stable_gesture = current_gesture
            gesture_counter = 1
        is_stable = gesture_counter >= GESTURE_HOLD_FRAMES

        if stable_gesture == GESTURE_DRAW and fingertip is not None:
            last_committed = GESTURE_NONE
            if prev_point is not None:
                cv2.line(canvas, prev_point, fingertip, 255,
                         thickness=5, lineType=cv2.LINE_AA)
                stroke_points.append(fingertip)
            prev_point = fingertip
        else:
            prev_point = None

            if is_stable and stable_gesture != last_committed:
                if stable_gesture == GESTURE_FIST:
                    _commit(canvas, stroke_points)
                    canvas[:] = 0
                    stroke_points.clear()
                    last_committed = GESTURE_FIST

                elif stable_gesture == GESTURE_SPACE:
                    state["typed_text"] += " "
                    canvas[:] = 0
                    stroke_points.clear()
                    last_committed = GESTURE_SPACE

                elif stable_gesture == GESTURE_THUMB:
                    state["typed_text"] = state["typed_text"][:-1]
                    last_committed = GESTURE_THUMB

                elif stable_gesture == GESTURE_TOGGLE_MODE:
                    state["mode"] = "shape" if state["mode"] == "letter" else "letter"
                    canvas[:] = 0
                    stroke_points.clear()
                    last_committed = GESTURE_TOGGLE_MODE

        overlay = frame.copy()
        overlay[canvas > 0] = (0, 255, 0)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        cv2.putText(frame, f"Mode: {state['mode'].upper()}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        cv2.putText(frame, f"Gesture: {stable_gesture}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Last: {state['last_letter']} ({state['last_confidence']:.2f})",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
               + buffer.tobytes() + b"\r\n")

    cap.release()


def _commit(canvas, stroke_points):
    if len(stroke_points) < MIN_STROKE_POINTS:
        return
    ys, xs = np.where(canvas > 0)
    if len(xs) == 0:
        return
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

    if state["mode"] == "shape":
        shape = detect_shape(canvas, bbox)
        state["last_letter"] = shape or "?"
        state["last_confidence"] = 1.0 if shape else 0.0
        if shape:
            state["typed_text"] += shape + " "
        return

    if not MODEL_LOADED:
        return
    letter, confidence = recognizer.predict(canvas, bbox)
    state["last_letter"] = letter or "?"
    state["last_confidence"] = confidence
    if letter and confidence >= CONFIDENCE_THRESHOLD:
        state["typed_text"] += letter


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/get_text")
def get_text():
    return jsonify({
        "text": state["typed_text"],
        "last_letter": state["last_letter"],
        "last_confidence": state["last_confidence"],
        "mode": state["mode"],
    })


@app.route("/clear_text", methods=["POST"])
def clear_text():
    state["typed_text"] = ""
    return jsonify({"status": "cleared"})


if __name__ == "__main__":
    app.run(debug=True, threaded=True)
