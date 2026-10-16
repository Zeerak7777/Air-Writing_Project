"""
recognizer.py
-------------
Loads the trained EMNIST-letters CNN model and turns a raw drawing
canvas (a black image with a white stroke traced by the fingertip)
into a predicted letter.
"""

import numpy as np
import cv2
import string

LABELS = list(string.ascii_uppercase)


class LetterRecognizer:
    def __init__(self, model_path="model/emnist_cnn.h5"):
        import tensorflow as tf
        self.model = tf.keras.models.load_model(model_path)

    def preprocess(self, canvas, bbox, size=28, padding=20):
        x_min, y_min, x_max, y_max = bbox

        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(canvas.shape[1], x_max + padding)
        y_max = min(canvas.shape[0], y_max + padding)

        cropped = canvas[y_min:y_max, x_min:x_max]
        if cropped.size == 0:
            return None

        # Rotate FIRST, before square-padding -- rotating after padding
        # distorts proportions differently depending on the letter's
        # natural aspect ratio.
        # no orientation change

        h, w = cropped.shape
        side = max(h, w)
        square = np.zeros((side, side), dtype=np.uint8)
        y_off = (side - h) // 2
        x_off = (side - w) // 2
        square[y_off:y_off + h, x_off:x_off + w] = cropped

        resized = cv2.resize(square, (size, size), interpolation=cv2.INTER_AREA)

        normalized = resized.astype("float32") / 255.0
        return normalized.reshape(1, size, size, 1)

    def predict(self, canvas, bbox):
        model_input = self.preprocess(canvas, bbox)
        if model_input is None:
            return None, 0.0

        probs = self.model.predict(model_input, verbose=0)[0]
        idx = int(np.argmax(probs))
        confidence = float(probs[idx])
        return LABELS[idx], confidence
