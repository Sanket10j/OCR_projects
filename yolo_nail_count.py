import cv2
import numpy as np
from ultralytics import YOLO


VIDEO_PATH = "D:/Downloads/video-part-1-code-test.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Error: Cannot open video file.")
    exit()

fgbg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=64, detectShadows=True)

falling_count = 0
tracked_objects = {}
next_id = 0

# Optical flow parameters
prev_gray = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (800, int(frame.shape[0] * 800 / frame.shape[1])))

    # Motion mask
    fgmask = fgbg.apply(frame)
    # Remove shadows (if detectShadows=True)
    fgmask[fgmask == 127] = 0

    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, np.ones((4, 4), np.uint8))
    fgmask = cv2.dilate(fgmask, None, iterations=2)

    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    current_objects = {}

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 200 or area > 1800:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h if h != 0 else 0
        if not (0.1 < aspect_ratio < 0.4):
            continue

        roi = frame[y:y+h, x:x+w]
        if roi.size == 0:
            continue

        # LAB color filtering to remove brown
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)

        # Brightness filter
        mean_L = np.mean(L)
        # A channel: 128 = neutral, >128 = reddish/brown, <128 = greenish/blue
        mean_A = np.mean(A)

        if mean_L < 160:  # too dark
            continue
        if mean_A > 128:  # brown/rust
            continue

        # Track
        cx, cy = x + w//2, y + h//2
        matched_id = None
        for obj_id, (px, py) in tracked_objects.items():
            if abs(cx - px) < 20 and abs(cy - py) < 35:
                matched_id = obj_id
                break

        if matched_id is None:
            matched_id = next_id
            next_id += 1
            falling_count += 1

        current_objects[matched_id] = (cx, cy)

        # Draw
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"Nail {matched_id}", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    tracked_objects = current_objects

    # Show count
    cv2.putText(frame, f"Falling Nails (Good): {falling_count}",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Improved Nail Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

print(f"Total proper (non-rusted) falling nails: {falling_count}")

