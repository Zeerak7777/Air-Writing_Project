"""
hand_tracker.py
----------------
Wraps MediaPipe's Hands solution and adds helper logic to figure out
which fingers are "up" (extended) or "down" (folded), which we use
to decide the current gesture (draw / end-letter / space / backspace).

MediaPipe gives us 21 landmarks per hand, numbered like this:

        8   12  16  20
        |   |   |   |
        7   11  15  19
        |   |   |   |
    4   6   10  14  18
    |   |   |   |   |
    3   5---9---13--17
     \  |           /
      \ |          /
        0 (wrist)

Tip landmarks: thumb=4, index=8, middle=12, ring=16, pinky=20
"""

import cv2
import mediapipe as mp


class HandTracker:
    def __init__(self, max_hands=1, detection_conf=0.7, tracking_conf=0.7):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.tip_ids = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky

    def process(self, frame_bgr):
        """
        Runs hand detection on a BGR frame (as read by OpenCV).
        Returns the MediaPipe results object (may contain 0 or more hands).
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return self.hands.process(frame_rgb)

    def draw_landmarks(self, frame_bgr, hand_landmarks):
        """Draws the skeleton overlay on the frame (for visual feedback)."""
        self.mp_draw.draw_landmarks(
            frame_bgr, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
        )

    def get_landmark_pixel_coords(self, hand_landmarks, frame_shape):
        """
        Converts normalized (0-1) landmark coordinates into pixel
        coordinates for the given frame size.
        Returns a list of 21 (x, y) tuples.
        """
        h, w = frame_shape[:2]
        return [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]

    def fingers_up(self, landmarks_px, handedness_label="Right"):
        """
        Given pixel-coordinate landmarks, returns a list of 5 booleans:
        [thumb, index, middle, ring, pinky] -> True if extended ("up").

        Index/middle/ring/pinky: finger is "up" if the tip is above
        (smaller y) than the joint two positions below it (the PIP joint).

        Thumb: compares x instead of y, and the comparison direction
        flips depending on whether it's the left or right hand, since
        the thumb moves sideways rather than up/down.
        """
        fingers = []

        # Thumb (landmark 4) vs its IP joint (landmark 3)
        if handedness_label == "Right":
            fingers.append(landmarks_px[4][0] > landmarks_px[3][0])
        else:
            fingers.append(landmarks_px[4][0] < landmarks_px[3][0])

        # Other four fingers: tip vs pip joint (tip_id - 2)
        for tip_id in self.tip_ids[1:]:
            fingers.append(landmarks_px[tip_id][1] < landmarks_px[tip_id - 2][1])

        return fingers  # e.g. [False, True, False, False, False] = only index up
