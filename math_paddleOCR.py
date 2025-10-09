# Depencies:
# pip install paddlepaddle paddleocr
# pip install opencv-python numpy


import re
import cv2
import numpy as np
from paddleocr import PaddleOCR


# Initialize PaddleOCR
ocr = PaddleOCR(
    use_angle_cls=True,
    lang='en',
    det_db_score_mode="fast",
    det_db_thresh=0.2,
    det_db_box_thresh=0.2,
    det_db_unclip_ratio=1.7
)


# def deskew_image(image):
#     """
#     Deskew image to correct for rotated digits (e.g., 1 being slanted).
#     """
#     coords = np.column_stack(np.where(image > 0))
#     angle = cv2.minAreaRect(coords)[-1]
#     if angle < -45:
#         angle = -(90 + angle)
#     else:
#         angle = -angle
#     (h, w) = image.shape[:2]
#     center = (w // 2, h // 2)
#     M = cv2.getRotationMatrix2D(center, angle, 1.0)
#     rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
#     return rotated


def parse_math_expression(text):
    """
    Parse and validate math expressions like '3+2' or '7-4'.
    If the operator is not detected, assume '-' as fallback.
    """
    cleaned = text.replace(' ', '')
    match = re.match(r'^(\d{1,2})([+\-])(\d{1,2})$', cleaned)
    if match:
        left, op, right = match.groups()
    else:
        # Try to infer the operator if it's missing (e.g. '72' instead of '7-2')
        match = re.match(r'^(\d)(\d)$', cleaned)
        if match:
            left, right = match.groups()
            op = '-'  # Default fallback operator
        else:
            raise ValueError(f"Could not parse expression: {cleaned}")
    return f"{left}{op}{right}"


def extract_math_captcha(image_path: str, solve: bool = True) -> str:
    """
    Extract and optionally solve a math captcha from an image.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize for better OCR performance
    scale_percent = 300
    width = int(gray.shape[1] * scale_percent / 100)
    height = int(gray.shape[0] * scale_percent / 100)
    gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_LINEAR)

    # Preprocessing
    blur = cv2.GaussianBlur(gray, (7, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # cv2.imshow("Bilateral Filter", thresh)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # Deskew
    # thresh = deskew_image(thresh)

    # Invert and morph for clarity
    processed = cv2.bitwise_not(thresh)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel, iterations=1)

    # cv2.imshow("Bilateral Filter", thresh)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # OCR
    results = ocr.ocr(processed, cls=True)
    if not results or not results[0]:
        raise ValueError("No text detected by OCR.")

    # Sort OCR results left to right
    sorted_results = sorted(results[0], key=lambda r: r[0][0][0])
    recognized_segments = [text for (box, (text, conf)) in sorted_results]

    # print("DEBUG - OCR segments recognized:", recognized_segments)

    full_expression_raw = ''.join(recognized_segments)

    try:
        full_expression = parse_math_expression(full_expression_raw)
        print("DEBUG - Parsed math expression:", full_expression)
        if solve:
            result = eval(full_expression)
            print("DEBUG - Math result:", result)
            return str(result)
        else:
            return full_expression
    except Exception as e:
        raise ValueError("Extracted text is not a valid math expression.") from e


if __name__ == '__main__':
    test_image = r"image path"  # Update with your test image
    result = extract_math_captcha(test_image, solve=True)
    print(f"Final result: {result}")
