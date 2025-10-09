import cv2
import numpy as np
import easyocr

# 1. Preprocess image
def preprocess(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    th = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    return img, th

# 2. Find possible captcha boxes
def find_captcha_boxes(binary_img, min_area=500, max_area=5000):
    cnts, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        aspect = w / float(h)
        if min_area < area < max_area and 2 < aspect < 8:
            boxes.append((x, y, w, h))
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    return boxes

# 3. OCR on the cropped captcha box
reader = easyocr.Reader(['en'])  # load once

def ocr_captcha(img, box):
    x, y, w, h = box
    crop = img[y:y+h, x:x+w]
    crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    result = reader.readtext(crop, detail=0)  # detail=0 means only text output
    return result, crop

# 4. Solve the captcha
def solve_captcha(img_path):
    img, th = preprocess(img_path)
    boxes = find_captcha_boxes(th)
    for box in boxes:
        text, crop = ocr_captcha(img, box)
        if text:
            candidate = text[0].replace(" ", "")  # remove spaces
            if 4 <= len(candidate) <= 6:  # assuming captcha is 4-6 chars
                return candidate, crop
    return None, None

# 5. Main execution
if __name__ == "__main__":
    img_path = r"D:\Desktop\test\a4.jpg"  # CHANGE to your image file
    captcha_text, captcha_img = solve_captcha(img_path)

    if captcha_text:
        print("Detected CAPTCHA:", captcha_text)
        cv2.imshow("Detected CAPTCHA Region", captcha_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Failed to detect a suitable CAPTCHA.")
