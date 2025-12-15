import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # quieter TF logs

import cv2
import math
import time
import numpy as np
from pathlib import Path
from cvzone.HandTrackingModule import HandDetector
from tensorflow.keras.models import load_model

# -------------------- Config --------------------
CAM_INDEX = 0
MAX_HANDS = 1
OFFSET = 20                 # extra pixels around bbox
PREVIEW_MIRROR = True       # mirror webcam preview
MODEL_INPUT = 224           # typical for Teachable Machine
THRESHOLD = 0.75            # show "Unknown" below this
SAVE_DIR = Path("Predictions")  # optional saves on 's' key

# Model folder paths
MODEL_PATH = "Model/keras_model.h5"
LABELS_PATH = "Model/labels.txt"
# ------------------------------------------------

def load_labels(path):
    with open(path, "r", encoding="utf-8") as f:
        labels = [ln.strip() for ln in f.readlines() if ln.strip()]
    return labels

def center_pad_resize(img, target=MODEL_INPUT):
    """Resize img to fit inside a square (target x target) with padding (white)."""
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.ones((target, target, 3), dtype=np.uint8) * 255

    aspect = h / w
    canvas = np.ones((target, target, 3), dtype=np.uint8) * 255

    if aspect > 1:  # Tall image
        k = target / h
        new_w = int(w * k)
        new_w = min(max(new_w, 1), target)
        resized = cv2.resize(img, (new_w, target))
        w_gap = (target - new_w) // 2
        canvas[:, w_gap:w_gap + new_w] = resized
    else:          # Wide image
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
        return np.ones((MODEL_INPUT, MODEL_INPUT, 3), dtype=np.uint8) * 255

    if any([pad_left, pad_top, pad_right, pad_bottom]):
        crop = cv2.copyMakeBorder(
            crop, pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_REPLICATE
        )
    return crop

def preprocess_for_model(img_bgr):
    """Convert BGR->RGB, resize to MODEL_INPUT, normalize to [-1,1] like Teachable Machine."""
    sq = center_pad_resize(img_bgr, target=MODEL_INPUT)
    rgb = cv2.cvtColor(sq, cv2.COLOR_BGR2RGB)
    tensor = (rgb.astype(np.float32) / 127.5) - 1.0  # [-1, 1]
    return np.expand_dims(tensor, axis=0), sq

def draw_label_above_hand(img, text, bbox, pad=6):
    """Draw a filled label centered above the hand bbox."""
    x, y, w, h = bbox
    # Compute text size
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.9
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)

    # Center the text horizontally above the bbox
    cx = x + w // 2
    tx = max(0, cx - tw // 2)
    ty = max(0, y - 10)  # start a little above the bbox

    # Box coordinates
    box_x1 = max(0, tx - pad)
    box_y1 = max(0, ty - th - pad)
    box_x2 = min(img.shape[1] - 1, tx + tw + pad)
    box_y2 = min(img.shape[0] - 1, ty + baseline + pad)

    # Draw filled rectangle (dark) and then text (white)
    cv2.rectangle(img, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1)
    cv2.putText(img, text, (tx, ty), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

def main():
    # Load model & labels
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")
    if not Path(LABELS_PATH).exists():
        raise FileNotFoundError(f"Missing labels file: {LABELS_PATH}")

    model = load_model(MODEL_PATH, compile=False)
    labels = load_labels(LABELS_PATH)

    # Prepare camera
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {CAM_INDEX}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    detector = HandDetector(maxHands=MAX_HANDS)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    prev_time = time.time()
    save_count = 0
    last_pred = ""
    last_conf = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("⚠️  Frame grab failed; retrying...")
            continue

        if PREVIEW_MIRROR:
            frame = cv2.flip(frame, 1)

        hands, img_draw = detector.findHands(frame, draw=True)

        # FPS
        now = time.time()
        fps = 1.0 / (now - prev_time) if now > prev_time else 0.0
        prev_time = now

        # Default model display
        display_sq = np.ones((MODEL_INPUT, MODEL_INPUT, 3), dtype=np.uint8) * 255

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            # Crop & preprocess
            crop = safe_crop_with_offset(frame, x, y, w, h, offset=OFFSET)
            tensor, display_sq = preprocess_for_model(crop)

            # Predict
            preds = model.predict(tensor, verbose=0)[0]  # (num_classes,)
            top_idx = int(np.argmax(preds))
            conf = float(preds[top_idx])
            label = labels[top_idx] if top_idx < len(labels) else f"Class {top_idx}"

            if conf >= THRESHOLD:
                last_pred, last_conf = label, conf
            else:
                last_pred, last_conf = "Unknown", conf

            # >>> Draw label above the hand (centered over bbox)
            draw_label_above_hand(img_draw, f"{last_pred} ({last_conf:.2f})", (x, y, w, h))

        # UI overlays (keep FPS and helper text)
        cv2.putText(img_draw, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(img_draw, "Keys: [S] Save crop  [Q]/[ESC] Quit", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        # Windows
        cv2.imshow("Webcam", img_draw)
        cv2.imshow("Model Input (224x224)", display_sq)

        # Keys
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            ts = time.strftime("%Y%m%d_%H%M%S")
            millis = int((time.time() % 1) * 1000)
            safe_label = last_pred.replace(" ", "_")
            out_name = SAVE_DIR / f"{safe_label}_{ts}_{millis}_{int(last_conf*100)}.jpg"
            cv2.imwrite(str(out_name), display_sq)
            save_count += 1
            print(f"✔ Saved #{save_count}: {out_name}")

        if key in (ord('q'), 27):  # q or ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
