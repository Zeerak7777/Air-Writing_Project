"""
One-time diagnostic: run app.py normally, draw ONE letter (pick something
asymmetric like F, P, or R -- avoid O, X, I which look the same rotated),
make a fist to commit it, then Ctrl+C the app and run this script.

It loads the last raw crop that was saved, tries all 8 possible
rotate/flip combinations, and prints what the model guesses for each --
so we can see exactly which transform gives the right answer.
"""
import cv2
import numpy as np
import tensorflow as tf
import glob
import os

model = tf.keras.models.load_model("model/emnist_cnn.h5")
LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

files = sorted(glob.glob("debug_letters/raw_*.png"), key=os.path.getmtime)
if not files:
    print("No raw debug image found. Make sure app.py has been patched")
    print("to save raw crops (see next message from Claude).")
    exit()

img = cv2.imread(files[-1], cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

transforms = {
    "identity":            lambda im: im,
    "rotate_90_cw":        lambda im: cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE),
    "rotate_90_ccw":       lambda im: cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE),
    "rotate_180":          lambda im: cv2.rotate(im, cv2.ROTATE_180),
    "flip_h":              lambda im: cv2.flip(im, 1),
    "flip_v":              lambda im: cv2.flip(im, 0),
    "transpose":           lambda im: cv2.transpose(im),
    "transpose_then_flip": lambda im: cv2.flip(cv2.transpose(im), 0),
}

print(f"Testing image: {files[-1]}\n")
for name, fn in transforms.items():
    transformed = fn(img)
    model_input = transformed.astype("float32") / 255.0
    model_input = model_input.reshape(1, 28, 28, 1)
    probs = model.predict(model_input, verbose=0)[0]
    idx = int(np.argmax(probs))
    conf = float(probs[idx])
    print(f"{name:22s} -> predicted: {LABELS[idx]}  (confidence: {conf:.2f})")
