import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

# Open webcam
cap = cv2.VideoCapture(0)

# Hand detector (from cvzone)
detector = HandDetector(maxHands=1)

offset = 20          # Extra space around the hand crop
imgSize = 300        # Size of the output square image
folder = "Data/C"    # Folder to save images
counter = 0          # Saved image counter

while True:
    success, img = cap.read()              # Read frame
    hands, img = detector.findHands(img)   # Detect hands

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']          # Bounding box (x, y, width, height)

        # White background image (300x300)
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # ---- SAFE CROPPING TO AVOID ERRORS ----
        y1 = max(0, y - offset)
        y2 = min(img.shape[0], y + h + offset)
        x1 = max(0, x - offset)
        x2 = min(img.shape[1], x + w + offset)

        imgCrop = img[y1:y2, x1:x2]

        # Skip frame if crop is empty
        if imgCrop.size == 0:
            continue
        # ----------------------------------------

        aspectRatio = h / w

        # If height > width (vertical hand), match height first
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize

        # If width >= height (horizontal hand), match width first
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        # Show cropped hand + final white image
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    # Show original webcam image
    cv2.imshow("Image", img)

    key = cv2.waitKey(1)

    # Press 's' to save an image
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print("Saved:", counter)

    # Press 'q' to quit
    if key == ord("q"):
        break

# Close webcam and windows
cap.release()
cv2.destroyAllWindows()
