import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import time
import numpy as np
from pathlib import Path
from cvzone.HandTrackingModule import HandDetector
from tensorflow.keras.models import load_model

# -------------------- Config --------------------
CAM_INDEX = 0
MAX_HANDS = 1
OFFSET = 20
PREVIEW_MIRROR = True
MODEL_INPUT = 224
THRESHOLD = 0.75  # you can raise later (0.75). Start lower to see output.
MODEL_PATH = "Model/keras_model.h5"
LABELS_PATH = "Model/labels.txt"
# ------------------------------------------------

def load_labels(path):
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f.readlines() if ln.strip()]

def center_pad_resize(img, target=MODEL_INPUT):
    h, w = img.shape[:2]
    canvas = np.ones((target, target, 3), dtype=np.uint8) * 255
    if h == 0 or w == 0:
        return canvas

    aspect = h / w
    if aspect > 1:  # tall
        k = target / h
        new_w = int(w * k)
        new_w = min(max(new_w, 1), target)
        resized = cv2.resize(img, (new_w, target))
        w_gap = (target - new_w) // 2
        canvas[:, w_gap:w_gap + new_w] = resized
    else:  # wide
        k = target / w
        new_h = int(h * k)
        new_h = min(max(new_h, 1), target)
        resized = cv2.resize(img, (target, new_h))
        h_gap = (target - new_h) // 2
        canvas[h_gap:h_gap + new_h, :] = resized

    return canvas

def safe_crop_with_offset(frame, x, y, w, h, offset=OFFSET):
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
        return np.ones((MODEL_INPUT, MODEL_INPUT, 3), dtype=np.uint8) * 255

    if any([pad_left, pad_top, pad_right, pad_bottom]):
        crop = cv2.copyMakeBorder(
            crop, pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_REPLICATE
        )
    return crop

def preprocess_for_model(img_bgr):
    sq = center_pad_resize(img_bgr, target=MODEL_INPUT)
    rgb = cv2.cvtColor(sq, cv2.COLOR_BGR2RGB)
    tensor = (rgb.astype(np.float32) / 127.5) - 1.0  # Teachable Machine
    return np.expand_dims(tensor, axis=0), sq

def draw_above_hand(img, bbox, letter, label, percent):
    """
    Draws:
      Line1: BIG letter
      Line2: Label: <label> | <percent>%
    above the hand bbox, clamped inside screen.
    """
    x, y, w, h = bbox
    H, W = img.shape[:2]

    font = cv2.FONT_HERSHEY_SIMPLEX
    pad = 8

    line1 = f"{letter}"
    line2 = f"Label: {label} | {percent}%"

    # Big letter size
    s1, t1 = 1.7, 3
    (tw1, th1), bl1 = cv2.getTextSize(line1, font, s1, t1)

    # Small details size
    s2, t2 = 0.75, 2
    (tw2, th2), bl2 = cv2.getTextSize(line2, font, s2, t2)

    box_w = max(tw1, tw2) + pad * 2
    box_h = (th1 + bl1) + (th2 + bl2) + pad * 3

    cx = x + w // 2
    box_x1 = int(cx - box_w // 2)
    box_y1 = int(y - box_h - 12)

    # Clamp inside screen
    box_x1 = max(0, min(box_x1, W - box_w - 1))
    if box_y1 < 0:
        box_y1 = 5  # if no space above, show at top

    box_x2 = box_x1 + box_w
    box_y2 = box_y1 + box_h

    # Draw background
    cv2.rectangle(img, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1)

    # Text positions
    tx1 = box_x1 + pad
    ty1 = box_y1 + pad + th1
    tx2 = box_x1 + pad
    ty2 = ty1 + bl1 + pad + th2

    cv2.putText(img, line1, (tx1, ty1), font, s1, (255, 255, 255), t1, cv2.LINE_AA)
    cv2.putText(img, line2, (tx2, ty2), font, s2, (200, 200, 200), t2, cv2.LINE_AA)

def main():
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")
    if not Path(LABELS_PATH).exists():
        raise FileNotFoundError(f"Missing labels file: {LABELS_PATH}")

    model = load_model(MODEL_PATH, compile=False)
    labels = load_labels(LABELS_PATH)

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {CAM_INDEX}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    detector = HandDetector(maxHands=MAX_HANDS)
    prev_time = time.time()

    last_letter = "?"
    last_percent = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        if PREVIEW_MIRROR:
            frame = cv2.flip(frame, 1)

        hands, img_draw = detector.findHands(frame, draw=True)

        # FPS
        now = time.time()
        fps = 1.0 / (now - prev_time) if now > prev_time else 0.0
        prev_time = now

        if hands:
            hand = hands[0]
            x, y, w, h = hand["bbox"]

            crop = safe_crop_with_offset(frame, x, y, w, h, offset=OFFSET)
            tensor, _ = preprocess_for_model(crop)

            preds = model.predict(tensor, verbose=0)[0]
            idx = int(np.argmax(preds))
            conf = float(preds[idx])
            percent = int(round(conf * 100))

            letter = labels[idx] if idx < len(labels) else "?"
            if conf < THRESHOLD:
                letter_to_show = "?"
                label_to_show = "Unknown"
            else:
                letter_to_show = letter
                label_to_show = letter

            last_letter = letter_to_show
            last_percent = percent

            draw_above_hand(img_draw, (x, y, w, h), last_letter, label_to_show, last_percent)

        # Top-left overlay
        cv2.putText(img_draw, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(img_draw, "Keys: [Q]/[ESC] Quit", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        cv2.imshow("Webcam", img_draw)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
