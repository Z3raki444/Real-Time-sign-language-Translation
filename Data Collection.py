import cv2
import math
import time
import numpy as np
from pathlib import Path
from cvzone.HandTrackingModule import HandDetector

# -------------------- Config --------------------
CAM_INDEX = 0
MAX_HANDS = 1
OFFSET = 20
IMG_SIZE = 300
FOLDER = Path("Data/A")
MIRROR = False  # Mirror for natural webcam preview
# ------------------------------------------------

def center_pad_resize(img, target=IMG_SIZE):
    """Resize img to fit inside a square (target x target) with padding."""
    h, w = img.shape[:2]
    aspect = h / w
    canvas = np.ones((target, target, 3), dtype=np.uint8) * 255

    if aspect > 1:  # Tall image
        k = target / h
        new_w = int(w * k)
        new_w = min(new_w, target)
        resized = cv2.resize(img, (new_w, target))
        w_gap = (target - new_w) // 2
        canvas[:, w_gap:w_gap + new_w] = resized

    else:  # Wide image
        k = target / w
        new_h = int(h * k)
        new_h = min(new_h, target)
        resized = cv2.resize(img, (target, new_h))
        h_gap = (target - new_h) // 2
        canvas[h_gap:h_gap + new_h, :] = resized

    return canvas

def safe_crop_with_offset(frame, x, y, w, h, offset=OFFSET):
    """Crop hand bbox with offset; pads with border if it spills outside."""
    H, W = frame.shape[:2]
    x1, y1 = x - offset, y - offset
    x2, y2 = x + w + offset, y + h + offset

    # Padding needed if crop is outside image
    pad_left   = max(0, -x1)
    pad_top    = max(0, -y1)
    pad_right  = max(0, x2 - W)
    pad_bottom = max(0, y2 - H)

    # Clip for actual cropping
    x1c = max(0, x1)
    y1c = max(0, y1)
    x2c = min(W, x2)
    y2c = min(H, y2)

    crop = frame[y1c:y2c, x1c:x2c]

    if any([pad_left, pad_top, pad_right, pad_bottom]):
        crop = cv2.copyMakeBorder(
            crop, pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_REPLICATE
        )

    return crop

def main():
    FOLDER.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {CAM_INDEX}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    detector = HandDetector(maxHands=MAX_HANDS)
    save_count = 0
    prev_time = time.time()

    while True:
        ok, img = cap.read()
        if not ok:
            print("⚠️ Frame grab failed; retrying...")
            continue

        if MIRROR:
            img = cv2.flip(img, 1)

        hands, img_draw = detector.findHands(img, draw=True)

        # Calculate FPS
        now = time.time()
        fps = 1 / (now - prev_time) if now != prev_time else 0
        prev_time = now

        # UI Overlays
        cv2.putText(img_draw, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(img_draw, f"Saved: {save_count}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
        cv2.putText(img_draw, "Keys: [S] Save  [Q]/[ESC] Quit", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        img_white = None

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            img_crop = safe_crop_with_offset(img, x, y, w, h, offset=OFFSET)
            img_white = center_pad_resize(img_crop, target=IMG_SIZE)

            cv2.imshow("ImageCrop", img_crop)
            cv2.imshow("ImageWhite", img_white)

        cv2.imshow("Image", img_draw)

        key = cv2.waitKey(1) & 0xFF

        # Save
        if key == ord('s'):
            if img_white is None:
                print("⚠️ No hand detected — nothing to save.")
            else:
                ts = time.strftime("%Y%m%d_%H%M%S")
                millis = int((time.time() % 1) * 1000)
                filename = FOLDER / f"Image_{ts}_{millis}.jpg"
                cv2.imwrite(str(filename), img_white)
                save_count += 1
                print(f"✔ Saved #{save_count}: {filename}")

        # Quit
        if key in (ord('q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
