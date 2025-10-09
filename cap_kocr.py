# import re
# import cv2
# import keras_ocr

# # Initialize keras-ocr pipeline
# pipeline = keras_ocr.pipeline.Pipeline()

# # Preprocessing: denoise, threshold, remove stray lines
# def preprocess(image):
#     # Convert to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     # Adaptive threshold to binarize
#     thresh = cv2.adaptiveThreshold(
#         gray,
#         255,
#         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#         cv2.THRESH_BINARY_INV,
#         blockSize=15,
#         C=10
#     )
#     # Morphological open to remove thin lines
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#     opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
#     # Convert back to BGR for OCR compatibility
#     processed = cv2.cvtColor(opened, cv2.COLOR_GRAY2BGR)
#     return processed

# # Extract and solve captcha
# # Supports alphanumeric captchas and simple arithmetic ( +, -, *, / )
# def extract_captcha_text(image_path):
#     # Load image
#     image = cv2.imread(image_path)
#     if image is None:
#         raise FileNotFoundError(f"Image not found: {image_path}")

#     # Preprocess to remove noise/lines
#     processed = preprocess(image)

#     # Recognize text boxes (expects RGB)
#     rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
#     prediction_groups = pipeline.recognize([rgb])
#     if not prediction_groups or not prediction_groups[0]:
#         return ''

#     # Flatten predictions
#     texts = [text for text, box in prediction_groups[0]]
#     captcha = texts[0].replace(' ', '')

#     # Solve arithmetic if pattern matches
#     match = re.match(r"^(\d+)([\+\-\*/xX])(\d+)$", captcha)
#     if match:
#         a, op, b = match.groups()
#         a, b = int(a), int(b)
#         if op in ['x', 'X', '*']:
#             result = a * b
#         elif op == '+':
#             result = a + b
#         elif op == '-':
#             result = a - b
#         elif op == '/':
#             result = a / b
#         return str(int(result))

#     # Otherwise return raw text
#     return captcha

import re
import keras_ocr
from keras_ocr.tools import read

# Compile patterns
CAPTCHA_PATTERN = re.compile(r'^[A-Za-z0-9]{6}$')
MATH_PATTERN = re.compile(r'^\s*(\d+)\s*([\+\-\*xX÷\/])\s*(\d+)\s*$')

# Initialize Keras-OCR pipeline once
pipeline = keras_ocr.pipeline.Pipeline()


def extract_captcha_text(image_path: str) -> str:
    # Load image via Keras-OCR helper
    image = read(image_path)

    # Run OCR
    prediction_groups = pipeline.recognize([image])
    if not prediction_groups or not prediction_groups[0]:
        return ''

    # Extract all recognized strings
    candidates = [text.strip() for text, _ in prediction_groups[0] if text.strip()]

    # 1) Look for a math captcha match among candidates
    for s in candidates:
        if MATH_PATTERN.match(s):
            return _solve_math(s)

    # 2) Look for 6-char alphanumeric among candidates
    for s in candidates:
        cleaned = re.sub(r'[^A-Za-z0-9]', '', s)
        if CAPTCHA_PATTERN.match(cleaned):
            return cleaned

    # 3) As fallback, join all and re-test
    joined = ''.join(candidates)
    if MATH_PATTERN.match(joined):
        return _solve_math(joined)
    cleaned = re.sub(r'[^A-Za-z0-9]', '', joined)
    if CAPTCHA_PATTERN.match(cleaned):
        return cleaned

    # 4) Otherwise return the top candidate raw
    return candidates[0]


def _solve_math(expr: str) -> str:
    # Normalize division symbol
    expr = expr.replace('÷', '/')
    # Extract parts
    m = MATH_PATTERN.match(expr)
    if not m:
        return expr
    a, op, b = m.groups()
    a, b = int(a), int(b)
    # Normalize operator
    if op in ('x', 'X', '*'):
        result = a * b
    elif op == '+':
        result = a + b
    elif op == '-':
        result = a - b
    elif op == '/':
        # integer division if divisible, else float
        result = a / b
    else:
        return expr
    # Return integer if whole
    return str(int(result)) if result == int(result) else str(result)


if __name__ == '__main__':
    # Example usage
    files = [ r"D:\Desktop\test\a3.jpg"]
    for f in files:
        text = extract_captcha_text(f)
        print(f,"=>", text)
