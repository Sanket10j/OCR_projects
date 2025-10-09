# # import pytesseract

# # # # # # to call application through library
# # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# # # # # from PIL import Image

# # # # # img = Image.open(r'C:\Users\BAPS\Desktop\OCR\IMG-20250422-WA0019.jpg')
# # # # # text = pytesseract.image_to_string(img)
# # # # # print(text)


# # import cv2
# # import pytesseract
# # from collections import defaultdict
# # import re

# # # # # Paths to the uploaded images
# # # # image_paths = [
# # # #     r"C:\Users\BAPS\Desktop\OCR\IMG-20250422-WA0019.jpg",
# # # #     r"C:\Users\BAPS\Desktop\OCR\IMG-20250422-WA0020.jpg",
# # # #     r"C:\Users\BAPS\Desktop\OCR\IMG-20250422-WA0021.jpg"
# # # # ]

# # # # # Function to extract text using pytesseract
# # # # def extract_text_from_image(image_path):
# # # #     image = cv2.imread(image_path)
# # # #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# # # #     text = pytesseract.image_to_string(gray)
# # # #     return text

# # # # # Dictionary to store results
# # # # # result = defaultdict(list)

# # # # # Regex patterns
# # # # # type_pattern = re.compile(r'^[A-Z\s\d]+$', re.MULTILINE)
# # # # # id_pattern = re.compile(r'DNO\s*\d+', re.IGNORECASE)

# # # # ls = []

# # # # # Process each image
# # # # for path in image_paths:
# # # #     raw_text = extract_text_from_image(path)
# # # #     ls.append(raw_text)

# # # # print(ls)



    
# # # # #     # Split into lines and filter
# # # # #     lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
    
# # # # #     current_type = None
# # # # #     for line in lines:
# # # # #         # Check if it's a T-shirt type
# # # # #         if type_pattern.match(line) and "DNO" not in line.upper():
# # # # #             current_type = line.strip()
# # # # #         elif current_type and id_pattern.search(line):
# # # # #             ids = id_pattern.findall(line)
# # # # #             result[current_type].extend(ids)

# # # # # Convert defaultdict to regular dict
# # # # # result_dict = dict(result)
# # # # # print(result_dict)


# # # import cv2
# # # import pytesseract
# # # import re
# # # from collections import defaultdict

# # # pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# # # # Set Tesseract path if on Windows
# # # def preprocess_image(image_path):
# # #     img = cv2.imread(image_path)
# # #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # #     gray = cv2.resize(gray, None, fx=2, fy=2)  # scale up
# # #     gray = cv2.GaussianBlur(gray, (5, 5), 0)  # reduce noise
# # #     _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# # #     return thresh

# # # # # Load and preprocess the image
# # # # def preprocess_image(image_path):
# # # #     img = cv2.imread(image_path)
# # # #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # # #     gray = cv2.resize(gray, None, fx=2, fy=2)  # scale up
# # # #     _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
# # # #     return thresh

# # # # Extract specific text from OCR results
# # # def extract_tshirt_data(text):
# # #     result = defaultdict(list)
# # #     lines = text.split('\n')
# # #     current_type = None

# # #     for line in lines:
# # #         line = line.strip()
# # #         if not line:
# # #             continue

# # #         # If line contains T-shirt type (e.g., DOUBLE LAFFER)
# # #         if "DOUBLE LAFFER" in line.upper():
# # #             current_type = line.strip()
# # #             result[current_type] = []
        
# # #         # If line contains DNO ID
# # #         elif current_type and re.search(r'DNO\s*\d+', line.upper()):
# # #             matches = re.findall(r'DNO\s*\d+', line.upper())
# # #             result[current_type].extend(matches)

# # #     return dict(result)

# # # # Main
# # # image_path = r"C:\Users\BAPS\Desktop\OCR\IMG-20250422-WA0019.jpg"  # change this to your path
# # # processed_img = preprocess_image(image_path)
# # # text = pytesseract.image_to_string(processed_img)
# # # tshirt_data = extract_tshirt_data(text)

# # # print(tshirt_data)

# # def preprocess_image(image_path):
# #     img = cv2.imread(image_path)
# #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# #     gray = cv2.resize(gray, None, fx=2, fy=2)  # scale up
# #     gray = cv2.GaussianBlur(gray, (5, 5), 0)  # reduce noise
# #     _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# #     return thresh

# # def extract_tshirt_data(text):
# #     result = defaultdict(list)
# #     lines = [line.strip().upper() for line in text.split('\n') if line.strip()]
# #     current_type = None

# #     for line in lines:
# #         print(f"Processing line: {line}")  # Debugging
# #         if "DOUBLE LAFFER" in line:
# #             print("Matched T-shirt type:", line)  # Debugging
# #             current_type = line.strip()
# #             result[current_type] = []
# #         elif current_type and re.search(r'DNO\s*\d+', line):
# #             print("Matched DNO ID:", line)  # Debugging
# #             matches = re.findall(r'DNO\s*\d+', line)
# #             result[current_type].extend(matches)

# #     return dict(result)

# # # Main
# # image_paths = [
# #     r"C:\Users\BAPS\Desktop\OCR\IMG-20250422-WA0019.jpg",
# #     r"C:\Users\BAPS\Desktop\OCR\IMG-20250422-WA0020.jpg",
# #     r"C:\Users\BAPS\Desktop\OCR\IMG-20250422-WA0021.jpg"
# # ]

# # for image_path in image_paths:
# #     processed_img = preprocess_image(image_path)
# #     text = pytesseract.image_to_string(processed_img)
# #     print("OCR Output:")
# #     print(text)  # Debugging
# #     tshirt_data = extract_tshirt_data(text)
# #     print(f"Data for {image_path}: {tshirt_data}")

# import cv2
# import pytesseract
# import numpy as np
# from collections import defaultdict
# import re

# # Set up Tesseract path (if needed on Windows)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# # Load the image
# image_path = r"C:\Users\BAPS\Desktop\OCR\IMG-20250422-WA0019.jpg"
# image = cv2.imread(image_path)
# hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# # Define HSV color ranges
# lower_yellow = np.array([20, 100, 100])
# upper_yellow = np.array([40, 255, 255])
# lower_black = np.array([0, 0, 0])
# upper_black = np.array([180, 255, 50])

# # Create masks
# mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
# mask_black = cv2.inRange(hsv, lower_black, upper_black)

# # Find contours
# contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# contours_black, _ = cv2.findContours(mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # Filter and sort boxes
# def get_boxes(contours):
#     return sorted(
#         [cv2.boundingRect(c) for c in contours if cv2.boundingRect(c)[2] > 50 and cv2.boundingRect(c)[3] > 20],
#         key=lambda b: (b[1], b[0])
#     )

# yellow_boxes = get_boxes(contours_yellow)
# black_boxes = get_boxes(contours_black)

# # OCR helper
# def extract_text(box):
#     x, y, w, h = box
#     roi = image[y:y+h, x:x+w]
#     gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#     return pytesseract.image_to_string(gray, config="--psm 6").strip()

# # Match yellow to black
# result = defaultdict(list)
# for ybox in yellow_boxes:
#     type_text = extract_text(ybox)
#     type_clean = re.sub(r"[^A-Z0-9 ]", "", type_text.upper())
#     if not type_clean:
#         continue

#     y_x, y_y, y_w, y_h = ybox
#     y_center_x = y_x + y_w // 2

#     for bbox in black_boxes:
#         b_x, b_y, b_w, b_h = bbox
#         b_center_x = b_x + b_w // 2

#         if b_y > y_y + y_h and abs(b_center_x - y_center_x) < 100:
#             id_text = extract_text(bbox)
#             id_clean = re.findall(r"DNO\s*\d+", id_text.upper())
#             if id_clean:
#                 result[type_clean].extend(id_clean)

# # Remove duplicates
# final_result = {k: list(set(v)) for k, v in result.items()}
# print(final_result)

import cv2
import pytesseract
import numpy as np

# Set path to tesseract executable if needed
# Example for Windows:
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load image
image_path = r"C:\Users\BAPS\Desktop\OCR\IMG-20250422-WA0021.jpg"
image = cv2.imread(image_path)

if image is None:
    raise Exception("Image not loaded. Check the path.")

# Resize for consistent processing
image = cv2.resize(image, (800, int(image.shape[0] * 800 / image.shape[1])))

# Convert image to HSV for color detection
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define yellow color range and create mask
yellow_lower = np.array([20, 100, 100])
yellow_upper = np.array([35, 255, 255])
yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

# Define black color range and create mask
black_lower = np.array([0, 0, 0])
black_upper = np.array([180, 255, 50])
black_mask = cv2.inRange(hsv, black_lower, black_upper)

# Function to extract text from a mask
def extract_text(mask, label):
    results = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 100 and h > 30:  # Filter small noise
            roi = image[y:y+h, x:x+w]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(gray, config="--psm 6").strip()
            if text:
                results.append((x, y, text))  # include position for sorting
    # Sort by y and x position
    results.sort(key=lambda tup: (tup[1], tup[0]))
    return [t[2] for t in results]

# Extract yellow (T-shirt types) and black (T-shirt IDs) texts
yellow_texts = extract_text(yellow_mask, "yellow")
black_texts = extract_text(black_mask, "black")

# Match closest black text below each yellow one
result_dict = {}
for i, y_text in enumerate(yellow_texts):
    y_x = i  # assume order matters, or further refine with spatial position
    black1 = black_texts[i*2] if i*2 < len(black_texts) else None
    black2 = black_texts[i*2+1] if i*2+1 < len(black_texts) else None
    result_dict[y_text] = list(filter(None, [black1, black2]))

# Print final dictionary
print("Extracted T-shirt Info:")
print(result_dict)
