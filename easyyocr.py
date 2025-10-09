import os
import re
import cv2
import easyocr

# Initialize EasyOCR with English language.
reader = easyocr.Reader(['en'])

# Patterns for a 6-character captcha and a simple math captcha.
CAPTCHA_PATTERN = re.compile(r'^[A-Za-z0-9]{6}$')
MATH_PATTERN = re.compile(r'^\s*(\d+)\s*([\+\-\*xX÷\/])\s*(\d+)\s*$')

def _eval_simple_math(expr: str) -> int:
    m = MATH_PATTERN.match(expr)
    if not m:
        raise ValueError(f"Not a simple math captcha: {expr!r}")
    a, op, b = m.groups()
    a, b = int(a), int(b)
    # Normalize operator symbols: change 'x' to '+', '_' to '-' as per your original logic.
    op = op.lower().replace('x', '+').replace('_', '-')
    if op == '+':
        return a + b
    elif op == '-':
        return a - b
    else:
        raise ValueError(f"Unknown operator {op!r}")

def extract_captcha_text(image_path: str) -> str:
    """
    Extracts captcha text from an image.
    Preprocesses the image, runs OCR with EasyOCR,
    cleans the OCR output, and then returns either the 6-character captcha text
    or the evaluated result of a simple arithmetic expression.
    """
    # Run OCR on the image using EasyOCR; results are returned as a list of (bbox, text, confidence).
    results = reader.readtext(image_path)
    if not results or len(results) == 0:
        raise ValueError("OCR did not detect any text.")

    # Normalize OCR results into a list of (text, confidence).
    detected = []
    for bbox, text, conf in results:
        clean_text = text.strip()
        if clean_text:
            detected.append((clean_text, conf))
    
    if not detected:
        raise ValueError("No text detected in OCR results.")
    
    # Sort candidates by OCR confidence (highest first).
    detected.sort(key=lambda x: x[1], reverse=True)
    
    # 1. Look for an exact 6-character alphanumeric match.
    for txt, conf in detected:
        if CAPTCHA_PATTERN.fullmatch(txt):
            return txt
    
    # 2. Look for a valid math captcha and evaluate it.
    for txt, conf in detected:
        if MATH_PATTERN.fullmatch(txt):
            try:
                return str(_eval_simple_math(txt))
            except Exception:
                continue
    
    # 3. As a fallback, remove spaces and re-check.
    for txt, conf in detected:
        candidate = txt.replace(" ", "")
        if CAPTCHA_PATTERN.fullmatch(candidate):
            return candidate
        if MATH_PATTERN.fullmatch(candidate):
            try:
                return str(_eval_simple_math(candidate))
            except Exception:
                continue
    
    # 4. If no candidate qualifies, return the highest confidence text as fallback.
    return detected[0][0]

if __name__ == '__main__':
    # Example usage
    files = [ r"D:\Desktop\test\a2.jpg"]
    for f in files:
        text = extract_captcha_text(f)
        print(f,"=>", text)