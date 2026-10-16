# Air-Writing Typer

Write letters in the air with your fingertip, in front of your webcam, and
watch them get typed out — built with Flask, OpenCV, MediaPipe, and a small
CNN trained on the EMNIST handwritten-letters dataset.

## How it works

1. **Hand tracking (`hand_tracker.py`)** — MediaPipe's Hands model finds 21
   landmark points on your hand every frame, including the index fingertip
   (landmark #8). We compare each fingertip's position to its lower joint to
   figure out which fingers are extended ("up") vs folded ("down").

2. **Gesture state machine (`app.py`)** — the pattern of up/down fingers
   decides the mode:
   - ☝️ **Only index up** → "pen down": the fingertip's path is drawn onto
     an invisible canvas frame by frame.
   - ✊ **Fist** → "end of letter": the canvas stroke is cropped to its
     bounding box, resized to 28×28, and sent to the CNN for classification.
     The predicted letter is appended to the typed text, and the canvas
     clears for the next letter.
   - ✋ **Open palm** → insert a space.
   - 👍 **Thumb only** → backspace.

   Gestures are debounced (must hold for several consecutive frames) so a
   single flickery frame doesn't accidentally trigger an action.

3. **Letter recognition (`recognizer.py` + `train_model.py`)** — a small CNN
   (2 conv layers + dense layers) trained on EMNIST's 26-letter dataset.
   `train_model.py` downloads the dataset and trains/saves the model once;
   `recognizer.py` loads it and classifies each finished stroke.

4. **Flask (`app.py`)** — streams the annotated webcam feed to the browser
   as an MJPEG stream (`/video_feed`), and exposes the current typed text as
   JSON (`/get_text`) that the page polls twice a second.

## Setup

```bash
python -m venv venv
source venv/bin/activate      # venv\Scripts\activate on Windows
pip install -r requirements.txt

# Train the letter-recognition model (only needed once; downloads EMNIST
# automatically, takes a few minutes on CPU)
python train_model.py

# Run the app
python app.py
```

Then open **http://127.0.0.1:5000** in your browser. Allow camera access if
prompted by your OS.

## Tips for accuracy

- Draw letters fairly large and slowly, roughly how you'd trace them on a
  whiteboard — EMNIST's handwriting style is closer to natural cursive-ish
  print than tiny careful print.
- Good, even lighting helps MediaPipe track your hand reliably.
- If recognition seems consistently wrong, the most common culprit is
  camera mirroring/orientation — the code flips the frame horizontally
  (`cv2.flip`) to feel natural; if your letters come out mirrored, try
  removing that flip.
- `CONFIDENCE_THRESHOLD` in `app.py` controls how confident the model must
  be before committing a letter — lower it if it's saying "?" too often,
  raise it if it's guessing wrong too often.

## Extending it

- **Word-level smoothing**: feed the typed text through a spellchecker/
  autocomplete (e.g. `pyspellchecker`) to clean up misrecognitions.
- **Numbers/symbols**: EMNIST also has a `balanced` split with digits and
  punctuation if you want to expand beyond A-Z.
- **Multi-hand / two-handed shortcuts**: `max_hands` in `HandTracker` can be
  raised to 2, and you could dedicate the second hand to commands (space,
  backspace, enter) instead of overloading gestures on one hand.
- **Smoother strokes**: apply a small moving-average filter to the
  fingertip coordinates before drawing, to reduce jitter from hand tracking
  noise.

## Project structure

```
air_writing_project/
├── app.py              # Flask app + gesture state machine + video streaming
├── hand_tracker.py      # MediaPipe wrapper + finger-up detection
├── recognizer.py         # Loads CNN, preprocesses canvas, predicts letter
├── train_model.py        # One-time script to train the EMNIST CNN
├── requirements.txt
├── templates/
│   └── index.html
├── static/
│   └── style.css
└── model/                 # emnist_cnn.h5 gets saved here after training
```
