import os
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

# ---------- Config ----------
OFFSET = 20           # Extra space around the hand crop
IMG_SIZE = 300        # Size of the output square image
FOLDER = "Data/D"     # Folder to save images
DETECT_CONF = 0.7     # Detection confidence for hand detector
MAX_HANDS = 1         # Max hands to track
# ---------------------------

os.makedirs(FOLDER, exist_ok=True)

# Open webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # CAP_DSHOW is often more stable on Windows
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

# Warm-up a few frames so the first frames aren't None
for _ in range(10):
    cap.read()

# Hand detector (from cvzone -> mediapipe)
detector = HandDetector(maxHands=MAX_HANDS, detectionCon=DETECT_CONF)

counter = 0

while True:
    success, img = cap.read()
    if not success or img is None:
        # If the camera hiccups, just skip this iteration
        continue

    # Ensure contiguous memory (mediapipe can be picky on some Windows builds)
    img = np.ascontiguousarray(img)

    # Detect hands (guarded)
    try:
        hands, img_draw = detector.findHands(img, draw=True)
    except Exception as e:
        # If mediapipe throws for a bad frame, skip safely
        print("Detector warning:", e)
        continue

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']  # (x, y, width, height)

        # White background image (300x300)
        imgWhite = np.ones((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8) * 255

        # ---- SAFE CROPPING ----
        H, W = img.shape[:2]
        y1 = max(0, y - OFFSET)
        y2 = min(H, y + h + OFFSET)
        x1 = max(0, x - OFFSET)
        x2 = min(W, x + w + OFFSET)
        if y2 <= y1 or x2 <= x1:
            # Invalid bbox (sometimes happens when the hand is at the edge)
            cv2.imshow("Image", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        imgCrop = img[y1:y2, x1:x2]
        if imgCrop.size == 0:
            cv2.imshow("Image", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        # -----------------------

        # Use the CROPPED size to compute ratio (robust if bbox is clipped)
        ch, cw = imgCrop.shape[:2]
        aspectRatio = ch / cw

        if aspectRatio > 1:
            # taller than wide -> fit height
            k = IMG_SIZE / ch
            wCal = math.ceil(k * cw)
            imgResize = cv2.resize(imgCrop, (wCal, IMG_SIZE))
            wGap = (IMG_SIZE - wCal) // 2
            imgWhite[:, wGap:wGap + wCal] = imgResize
        else:
            # wider than tall -> fit width
            k = IMG_SIZE / cw
            hCal = math.ceil(k * ch)
            imgResize = cv2.resize(imgCrop, (IMG_SIZE, hCal))
            hGap = (IMG_SIZE - hCal) // 2
            imgWhite[hGap:hGap + hCal, :] = imgResize

        # Show windows
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)
        cv2.imshow("Image", img_draw)
    else:
        # No hands detected; still show live feed
        cv2.imshow("Image", img)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        # Save a timestamped image
        filename = os.path.join(FOLDER, f'Image_{int(time.time()*1000)}.jpg')
        cv2.imwrite(filename, imgWhite)
        counter += 1
        print("Saved:", counter, "->", filename)

    if key == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
