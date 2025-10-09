import cv2
import pytesseract
import numpy as np
from collections import defaultdict

# Set the Tesseract executable path (adjust if needed)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def ocr_text(image, box):
    x, y, w, h = box
    # print(x, y, w, h)
    roi = image[y:y+h, x:x+w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    custom_config = r'--oem 3 --psm 6'
    return pytesseract.image_to_string(thresh, config=custom_config).strip()

# ------------------------
# Box Detection Function
# ------------------------
def get_colored_boxes(image, lower, upper, min_area=500):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Number of contours detected: {len(contours)}")

    # Create a copy of the original image to draw bounding boxes
    box_visualization = image.copy()

    boxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            box = cv2.boundingRect(cnt)
            boxes.append(box)
            # Draw the bounding box on the image
            x, y, w, h = box
            cv2.rectangle(box_visualization, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green box

    # Display the image with bounding boxes
    cv2.imshow("Bounding Boxes", box_visualization)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return sorted(boxes, key=lambda b: b[1])  # Sort by y

# Load your image (change path accordingly)
image_path = r"C:\Users\BAPS\Desktop\OCR\IMG-20250422-WA0022.jpg"
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Cannot load image: {image_path}")

# HSV color range for t-shirt type labels (e.g., yellow)
type_lower = np.array([20, 100, 100])
type_upper = np.array([40, 255, 255])

# HSV color range for t-shirt IDs (e.g., black text boxes)
id_lower = np.array([0, 0, 0])
id_upper = np.array([180, 255, 50])

# Detect colored boxes
type_boxes = get_colored_boxes(image, type_lower, type_upper)
id_boxes = get_colored_boxes(image, id_lower, id_upper)

# OCR for each box
type_texts = [ocr_text(image, box) for box in type_boxes]
id_texts = [ocr_text(image, box) for box in id_boxes]


results = defaultdict(list)
for idx_id, id_box in enumerate(id_boxes):
    _, id_y, _, _ = id_box
    candidate_text = None
    max_y = -1
    for idx_type, type_box in enumerate(type_boxes):
        _, type_y, _, _ = type_box
        if type_y < id_y and type_y > max_y:
            max_y = type_y
            candidate_text = type_texts[idx_type]
    if candidate_text and id_texts[idx_id]:
        key = candidate_text.replace("\n", " ").strip()
        value = id_texts[idx_id].replace("\n", " ").strip()
        if key and value:
            results[key].append(value)


print("\n✅ Extracted T-shirt Data:")
for k, v in results.items():
    print(f"{k}: {v}")
