# Depencies:
# pip install paddlepaddle paddleocr
# pip install opencv-python numpy



import re
import cv2
import numpy as np
from paddleocr import PaddleOCR

# 1) Define the 6-char alphanumeric pattern:
CAPTCHA_PATTERN = re.compile(r'^[a-z0-9]{6}$')

# 2) Initialize PaddleOCR
ocr = PaddleOCR(
    use_angle_cls=True,
    lang='en',
    det_db_score_mode="fast",
    det_db_thresh=0.2,
    det_db_box_thresh=0.2,
    det_db_unclip_ratio=2.0
)

def extract_captcha_text(image_path: str) -> str:
    """
    Reads an image of a 6-character alphanumeric captcha,
    removes noise lines, then uses PaddleOCR to extract
    exactly 6 alphanumeric chars. 
    """
    # -------------------------------------------------------------------------
    # STEP A: Load the image and convert to grayscale
    # -------------------------------------------------------------------------
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # -------------------------------------------------------------------------
    # STEP B: Blur and binarize (invert so text becomes white, background black)
    # -------------------------------------------------------------------------
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bw = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
     # 3. Remove long horizontal & vertical artifacts
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 7))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 35))
    h_removed = cv2.morphologyEx(bw, cv2.MORPH_OPEN, h_kernel)
    v_removed = cv2.morphologyEx(bw, cv2.MORPH_OPEN, v_kernel)
    no_lines = cv2.subtract(bw, cv2.bitwise_or(h_removed, v_removed))

    
    # -------------------------------------------------------------------------
    # STEP C: Morphological cleaning - remove small lines/holes
    #   - Increase kernel size or change morph operations if needed
    # -------------------------------------------------------------------------
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    cleaned = cv2.morphologyEx(no_lines, cv2.MORPH_OPEN, kernel, iterations=1)

    # -------------------------------------------------------------------------
    # STEP D: Invert back so text is dark on light background (standard for OCR)
    # -------------------------------------------------------------------------
    processed = cv2.bitwise_not(cleaned)

    # -------------------------------------------------------------------------
    # STEP E: OCR with PaddleOCR
    # -------------------------------------------------------------------------
    results = ocr.ocr(processed, cls=True)
    if not results or not results[0]:
        raise ValueError("No text detected by OCR.")

    # -------------------------------------------------------------------------
    # STEP F: Collect OCR segments (this is crucial for debugging)
    # -------------------------------------------------------------------------
    recognized_segments = []
    for box, (txt, conf) in results[0]:
        # Strip out leading/trailing whitespace from each chunk:
        recognized_segments.append(txt.strip())

    print("DEBUG - OCR segments recognized:", recognized_segments)

    # -------------------------------------------------------------------------
    # STEP G: 1st Attempt - Simply join all segments and see if we have 6 chars
    # -------------------------------------------------------------------------
    joined_all = "".join(recognized_segments)
    print(f"DEBUG - Joined segments: '{joined_all}'")

    if CAPTCHA_PATTERN.fullmatch(joined_all):
        return joined_all

    # -------------------------------------------------------------------------
    # STEP H: 2nd Attempt - Remove everything not alphanumeric, then check
    # -------------------------------------------------------------------------
    candidate_clean = re.sub(r'[^a-z0-9]', '', joined_all)
    print(f"DEBUG - Alphanumeric-only joined: '{candidate_clean}'")

    if CAPTCHA_PATTERN.fullmatch(candidate_clean):
        return candidate_clean

    # -------------------------------------------------------------------------
    # STEP I: 3rd Attempt - Maybe each segment individually is correct.
    #   Sometimes PaddleOCR returns a single segment for the entire captcha,
    #   or sometimes 6 segments for each character, etc.
    # -------------------------------------------------------------------------
    for seg in recognized_segments:
        seg_clean = re.sub(r'[^A-Za-z0-9]', '', seg)
        if CAPTCHA_PATTERN.fullmatch(seg_clean):
    #        print(f"DEBUG - Matched within single segment: '{seg_clean}'")
            return seg_clean

    # -------------------------------------------------------------------------
    # STEP J: If all else fails, raise an error
    # -------------------------------------------------------------------------
    raise ValueError("Failed to extract 6-char code cleanly")


if __name__ == '__main__':
    test_image = r"image path"  # Or your actual file
    text = extract_captcha_text(test_image)
    print(f"Final recognized CAPTCHA text = '{text}'")