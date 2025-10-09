import cv2
import pytesseract
import numpy as np
from collections import defaultdict

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Path to your image
path = r"C:\Users\BAPS\Desktop\OCR\IMG-20250422-WA0021.jpg"

def ocr_text(image, box):

    x, y, w, h = box
    roi = image[y:y+h, x:x+w]

    # Preprocessing to improve yellow text accuracy
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # Improve contrast
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Sharpening
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    sharp = cv2.filter2D(gray, -1, kernel)

    # Binary Threshold
    _, thresh = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Try with psm 6 (block of text) or 7 (single line)
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(thresh, config=custom_config)
    return text.strip()


def extract_boxes_by_color(image, lower, upper, label="color"):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 100 and h > 20:
            boxes.append((x, y, w, h))
    return sorted(boxes, key=lambda b: b[1])

# Load image
# path = "IMG-20250422-WA0019.jpg"
image = cv2.imread(path)
if image is None:
    raise FileNotFoundError(f"Couldn't load image at: {path}")

# Define color bounds
yellow_lower = np.array([20, 100, 100])
yellow_upper = np.array([40, 255, 255])
black_lower = np.array([0, 0, 0])
black_upper = np.array([180, 255, 50])

yellow_boxes = extract_boxes_by_color(image, yellow_lower, yellow_upper, "yellow")
black_boxes = extract_boxes_by_color(image, black_lower, black_upper, "black")

output = defaultdict(list)

# Pair and OCR
for ybox in yellow_boxes:
    y_text = ocr_text(image, ybox)
    y_x, y_y, y_w, y_h = ybox

    # Match to closest black box BELOW
    best_match = None
    min_dist = float('inf')
    for bbox in black_boxes:
        b_x, b_y, b_w, b_h = bbox
        if b_y > y_y:
            dist = b_y - y_y
            if dist < min_dist:
                min_dist = dist
                best_match = bbox

    if best_match:
        b_text = ocr_text(image, best_match)
        key = y_text.replace("\n", " ").strip()
        val = b_text.replace("\n", " ").strip()
        if key and val:
            output[key].append(val)

# Result
print("Info:")
print(dict(output))