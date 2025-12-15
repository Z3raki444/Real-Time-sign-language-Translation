import cv2
import math
import time
import numpy as np
from collections import deque, Counter
from pathlib import Path
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

# -------------------- Config --------------------
CAM_INDEX = 0
MAX_HANDS = 1
OFFSET = 20
IMG_SIZE = 300                # canvas for classifier input
MIRROR = True                 # mirror webcam for natural preview
MODEL_PATH = "Model/keras_model.h5"
LABELS_PATH = "Model/labels.txt"
FALLBACK_LABELS = ["A", "B", "C"]
SMOOTH_WINDOW = 7             # number of recent predictions to smooth (mode)
CONF_THRESH = 0.65            # min probability to display class
BAR_W = 220                   # width of probability bar
# ------------------------------------------------

def load_labels(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            labels = [ln.strip() for ln in f if ln.strip()]
        if labels:
            return labels
    except Exception:
        pass
    return FALLBACK_LABELS

def center_pad_resize(img, target=IMG_SIZE):
    """Resize img to fit inside a square (target x target) with padding (white)."""
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        # Return a blank canvas to avoid downstream crashes
        return np.ones((target, target, 3), dtype=np.uint8) * 255

    aspect = h / w
    canvas = np.ones((target, target, 3), dtype=np.uint8) * 255

    if aspect > 1:  # tall
        k = target / h
        new_w = int(w * k)
        new_w = min(max(new_w, 1), target)
        resized = cv2.resize(img, (new_w, target))
        w_gap = (target - new_w) // 2
        canvas[:, w_gap:w_gap + new_w] = resized
    else:          # wide
        k = target / w
        new_h = int(h * k)
        new_h = min(max(new_h, 1), target)
        resized = cv2.resize(img, (target, new_h))
        h_gap = (target - new_h) // 2
        canvas[h_gap:h_gap + new_h, :] = resized

    return canvas

def safe_crop_with_offset(frame, x, y, w, h, offset=OFFSET):
    """Crop hand bbox with offset; pads with replicated border if outside frame."""
    H, W = frame.shape[:2]
    x1, y1 = x - offset, y - offset
    x2, y2 = x + w + offset, y + h + offset

    pad_left   = max(0, -x1)
    pad_top    = max(0, -y1)
    pad_right  = max(0, x2 - W)
    pad_bottom = max(0, y2 - H)

    x1c = max(0, x1)
    y1c = max(0, y1)
    x2c = min(W, x2)
    y2c = min(H, y2)

    crop = frame[y1c:y2c, x1c:x2c]
    if crop.size == 0:
        # return a tiny white canvas to avoid crashes; caller will resize anyway
        return np.ones((10, 10, 3), dtype=np.uint8) * 255

    if any([pad_left, pad_top, pad_right, pad_bottom]):
        crop = cv2.copyMakeBorder(
            crop, pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_REPLICATE
        )
    return crop

def draw_label_box(img, text, x, y, w, h, color=(255, 0, 255)):
    """Draw a filled label box above the hand rectangle."""
    pad_y = 50
    cv2.rectangle(img, (x, y - pad_y), (x + max(120, w // 3), y), color, cv2.FILLED)
    cv2.putText(img, text, (x + 8, y - 12), cv2.FONT_HERSHEY_COMPLEX, 0.9, (255, 255, 255), 2)

def draw_probs_panel(img, probs, labels, origin=(10, 120), bar_w=BAR_W, bar_h=24, gap=8):
    """Draw top-3 probabilities as horizontal bars."""
    if probs is None or len(probs) == 0:
        return
    # zip & sort by prob desc
    pairs = list(zip(labels, probs))
    pairs.sort(key=lambda p: p[1], reverse=True)
    top = pairs[:min(3, len(pairs))]

    x0, y0 = origin
    for i, (lbl, p) in enumerate(top):
        y = y0 + i * (bar_h + gap)
        cv2.putText(img, f"{lbl}", (x0, y + bar_h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        # bar bg
        cv2.rectangle(img, (x0 + 70, y), (x0 + 70 + bar_w, y + bar_h), (60, 60, 60), 2)
        # bar fill
        fill_w = int(bar_w * float(p))
        cv2.rectangle(img, (x0 + 70, y), (x0 + 70 + fill_w, y + bar_h), (0, 200, 255), cv2.FILLED)
        cv2.putText(img, f"{p*100:5.1f}%", (x0 + 80 + bar_w, y + bar_h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)

def main():
    # Load labels & model
    labels = load_labels(LABELS_PATH)
    classifier = Classifier(MODEL_PATH, LABELS_PATH)

    # Camera setup
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {CAM_INDEX}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    detector = HandDetector(maxHands=MAX_HANDS)

    # Smoothing
    pred_history = deque(maxlen=SMOOTH_WINDOW)

    prev_time = time.time()
    last_probs = None
    last_index = None

    while True:
        ok, img = cap.read()
        if not ok:
            print("⚠️ Frame grab failed; retrying...")
            continue

        if MIRROR:
            img = cv2.flip(img, 1)

        hands, img_draw = detector.findHands(img, draw=True)

        # FPS
        now = time.time()
        fps = 1.0 / (now - prev_time) if now > prev_time else 0.0
        prev_time = now

        cv2.putText(img_draw, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(img_draw, "Keys: [Q]/[ESC] Quit",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            # Robust crop & square canvas
            img_crop = safe_crop_with_offset(img, x, y, w, h, offset=OFFSET)
            img_white = center_pad_resize(img_crop, target=IMG_SIZE)

            # Get prediction
            probs, index = classifier.getPrediction(img_white, draw=False)  # probs is list of floats
            last_probs = probs
            last_index = index if 0 <= index < len(labels) else None

            # Smooth predicted index using mode over recent window
            pred_history.append(index)
            mode_index = Counter(pred_history).most_common(1)[0][0]

            # Choose display label & confidence
            if probs and 0 <= mode_index < len(probs):
                conf = float(probs[mode_index])
            else:
                conf = 0.0

            if last_index is not None and conf >= CONF_THRESH:
                display_text = f"{labels[mode_index]}  {conf*100:.1f}%"
            elif last_index is not None:
                display_text = f"{labels[mode_index]}  {conf*100:.1f}% (low)"
            else:
                display_text = "Detecting..."

            # Draw label & bbox
            draw_label_box(img_draw, display_text, x - OFFSET, y - OFFSET, w, h, color=(255, 0, 255))
            cv2.rectangle(img_draw, (x - OFFSET, y - OFFSET), (x + w + OFFSET, y + h + OFFSET), (255, 0, 255), 3)

            # Show helper windows
            cv2.imshow("ImageCrop", img_crop)
            cv2.imshow("ImageWhite", img_white)

        # Draw probabilities panel on the main frame (last seen)
        if last_probs is not None:
            draw_probs_panel(img_draw, last_probs, labels, origin=(10, 110), bar_w=BAR_W)

        cv2.imshow("Image", img_draw)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):  # q or ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
