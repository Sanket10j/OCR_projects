# import os
# import re
# import cv2
# from paddleocr import PaddleOCR

# # Initialize PaddleOCR with angle classification enabled
# ocr = PaddleOCR(
#     use_angle_cls=True,
#     lang='en',
#     det_db_score_mode="fast",
#     det_db_thresh=0.2,        # lower the detection confidence cutoff
#     det_db_box_thresh=0.2,
#     det_db_unclip_ratio=2.0     # lower the box‐accept threshold
# )



# # Patterns for a 6-character captcha and a simple math captcha
# CAPTCHA_PATTERN = re.compile(r'^[A-Za-z0-9]{6}$')
# MATH_PATTERN = re.compile(r'^\s*(\d+)\s*([\+\-\*xX÷\/])\s*(\d+)\s*$')

# def _eval_simple_math(expr: str) -> int:

#     m = MATH_PATTERN.match(expr)
#     if not m:
#         raise ValueError(f"Not a simple math captcha: {expr!r}")
#     a, op, b = m.groups()
#     a, b = int(a), int(b)
#     # Normalize operator symbols
#     op = op.lower().replace('x', '+').replace('_', '-').replace('.', '-')
#     if op == '+':
#         return a + b
#     elif op == '-':
#         return a - b
#     else:
#         raise ValueError(f"Unknown operator {op!r}")


# def preprocess_image(image_path: str):
#     """
#     Preprocess to maximize OCR on simple math:
#     1) gray + blur
#     2) Otsu threshold
#     3) morphological closing
#     4) invert to black-on-white
#     """
#     img = cv2.imread(image_path)
#     if img is None:
#         raise FileNotFoundError(f"Image not found: {image_path}")
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     gray = cv2.GaussianBlur(gray, (5,5), 0)

#     # Otsu threshold
#     _, thresh = cv2.threshold(gray, 0, 255,
#                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    

#     cv2.imshow("Bilateral Filter", thresh)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


#     # Close small gaps (kernel size can be tuned)
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

#     closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
#     cv2.imshow("Bilateral Filter", closed)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#     #Invert back so text is dark on light
#     processed = cv2.bitwise_not(closed)

#     cv2.imshow("Bilateral Filter", processed)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


#     return processed



# def extract_captcha_text(image_path: str) -> str:
#     """
#     Extracts the captcha text from the image.
#     It preprocesses the image, runs OCR, cleans the OCR output,
#     and then returns either the 6-character captcha text or the result of a
#     simple arithmetic expression.
#     """
#     processed_img = preprocess_image(image_path)
    
#     # Run OCR on the preprocessed image
#     results = ocr.ocr(processed_img , cls=True)
#     if not results or len(results) == 0:
#         raise ValueError("OCR did not detect any text.")
#     ocr_lines = results[0]
    
#     # Normalize OCR results into a list of (text, confidence)
#     detected = []
#     for box, (text, conf) in ocr_lines:
#         # Clean extra punctuation and whitespace
#         clean_text = text.strip()
#         if clean_text:
#             detected.append((clean_text, conf))
    
#     if not detected:
#         raise ValueError("No text detected in OCR results.")
    
#     # Sort candidates by OCR confidence (highest first)
#     # detected.sort(key=lambda x: x[1], reverse=True)
    
#     # 1. Look for an exact 6-character alphanumeric match.
#     for txt, conf in detected:
#         if CAPTCHA_PATTERN.fullmatch(txt):
#             return txt
    
#     # 2. Look for a valid math captcha and evaluate it.
#     for txt, conf in detected:
#         if MATH_PATTERN.fullmatch(txt):
#             try:
#                 return str(_eval_simple_math(txt))
#             except Exception as e:
#                 # Continue to next candidate if evaluation fails
#                 continue
    
#     # 3. As a fallback, try post-processing: remove spaces and re-check.
#     for txt, conf in detected:
#         candidate = txt.replace(" ", "")
#         if CAPTCHA_PATTERN.fullmatch(candidate):
#             return candidate
#         if MATH_PATTERN.fullmatch(candidate):
#             try:
#                 return str(_eval_simple_math(candidate))
#             except Exception:
#                 continue
    
#     # 4. If no candidate qualifies, return the highest confidence text as fallback.
#     return detected[0][0]


# if __name__ == '__main__':
#     # Example usage
#     files = [ r"D:\Desktop\test\a2.jpg"]
#     for f in files:
#         text = extract_captcha_text(f)
#         print(f,"=>", text)



# # if __name__ == "__main__":
# #     folder_path = r"D:\Desktop\test"
# #     for file_name in os.listdir(folder_path):
# #         img_path = os.path.join(folder_path, file_name)
# #         try:
# #             result = extract_captcha_text(img_path)
# #             print(f"Detected CAPTCHA: {result}")
# #         except Exception as e:
# #             print(f"Error processing image {img_path}: {e}")






# Dependencies (install via pip):
#   pip install opencv-python paddleocr numpy

import re
# import cv2
# import numpy as np
# from paddleocr import PaddleOCR

# # Initialize PaddleOCR
# ocr = PaddleOCR(
#     use_angle_cls=True,
#     lang='en',
#     det_db_score_mode="fast",
#     det_db_thresh=0.2,
#     det_db_box_thresh=0.2,
#     det_db_unclip_ratio=2.0
# )

# # Only accept exactly 6 alphanumeric chars
# CAPTCHA_PATTERN = re.compile(r'^[A-Za-z0-9]{6}$')

# def preprocess_image(image_path: str) -> np.ndarray:
#     """
#     1) Grayscale + blur
#     2) Otsu threshold to isolate text & noise
#     3) Morphological opening to remove thin lines
#     4) Invert so text is dark on light
#     """
#     img = cv2.imread(image_path)
#     if img is None:
#         raise FileNotFoundError(f"Image not found: {image_path}")
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5,5), 0)

#     # Threshold to get binary image (text + noise = white)
#     _, bw = cv2.threshold(
#         blurred, 0, 255,
#         cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
#     )

#     # Use a thin rectangular kernel to remove wiggly line artifacts
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
#     clean = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=2)

#     # Invert back: text black on white
#     return cv2.bitwise_not(clean)


# def extract_captcha_text(image_path: str) -> str:
#     """
#     Returns the 6-character captcha text from the image.
#     Raises ValueError if no valid 6-char string is found.
#     """
#     img = preprocess_image(image_path)

#     # PaddleOCR returns a list of lists: [[ (box, (text,conf)), ... ], ...]
#     results = ocr.ocr(img, cls=True)
#     if not results or not results[0]:
#         raise ValueError("No text detected")

#     # Flatten and strip
#     candidates = []
#     for box, (text, conf) in results[0]:
#         txt = text.strip()
#         if txt:
#             candidates.append((txt, conf))

#     if not candidates:
#         raise ValueError("Empty OCR output")

#     # 1) Look for exact 6-char match
#     for txt, _ in sorted(candidates, key=lambda x: x[1], reverse=True):
#         if CAPTCHA_PATTERN.fullmatch(txt):
#             return txt

#     # 2) Fallback: strip spaces and try again
#     for txt, _ in sorted(candidates, key=lambda x: x[1], reverse=True):
#         merged = txt.replace(" ", "")
#         if CAPTCHA_PATTERN.fullmatch(merged):
#             return merged

#     # 3) As a last resort, return the very highest-confidence string
#     return candidates[0][0]


# if __name__ == "__main__":
#     import glob

#     # Example usage: process all .jpg/.png in a folder
#     for img_path in glob.glob("captchas/*.[jp][pn]g"):
#         try:
#             code = extract_captcha_text(img_path)
#             print(f"{img_path} → {code}")
#         except Exception as e:
#             print(f"{img_path} → ERROR: {e}")


###################################

# import re
# import cv2
# import numpy as np
# from paddleocr import PaddleOCR

# # Initialize once at module scope
# ocr = PaddleOCR(
#     use_angle_cls=True,
#     lang='en',
#     det_db_score_mode="fast",
#     det_db_thresh=0.2,
#     det_db_box_thresh=0.2,
#     det_db_unclip_ratio=2.0
# )

# CAPTCHA_PATTERN = re.compile(r'^[A-Za-z0-9]{6}$')


# # def extract_captcha_text(image_path: str) -> str:
# #     """
# #     Reads an image, removes random lines, runs OCR, and returns
# #     the 6-character alphanumeric captcha string.
# #     """
# #     # 1. Load & grayscale
# #     img = cv2.imread(image_path)
# #     if img is None:
# #         raise FileNotFoundError(f"Cannot read {image_path}")
# #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# #     # 2. Blur + binarize (invert so text+noise==white)
# #     blur = cv2.GaussianBlur(gray, (5,5), 0)
    

# #     _, bw = cv2.threshold(
# #         blur, 0, 255,
# #         cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
# #     )

# #     # Use a thin rectangular kernel to remove wiggly line artifacts
# #     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
# #     cleaned = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=2)



# #     # 5. Invert back for standard OCR input (text dark on light)
# #     processed = cv2.bitwise_not(cleaned)

# #     # 6. PaddleOCR
# #     results = ocr.ocr(processed, cls=True)
# #     if not results or not results[0]:
# #         raise ValueError("No text detected")

# #     # 7. Collect candidates and pick the first perfect match
# #     for box, (txt, conf) in results[0]:
# #         txt = txt.strip()
# #         if CAPTCHA_PATTERN.fullmatch(txt):
# #             return txt

# #     # 8. Fallback: strip spaces/punctuation and re-test
# #     for box, (txt, conf) in results[0]:
# #         candidate = re.sub(r'[^A-Za-z0-9]', '', txt)
# #         if CAPTCHA_PATTERN.fullmatch(candidate):
# #             return candidate

# #     # 9. If still no perfect match, raise
# #     raise ValueError("Failed to extract 6-char code cleanly")

# def extract_captcha_text(image_path: str) -> str:
#     """
#     Reads an image, removes random lines, runs OCR, and returns
#     the 6-character alphanumeric captcha string.
#     """
#     # 1. Load & grayscale
#     img = cv2.imread(image_path)
#     if img is None:
#         raise FileNotFoundError(f"Cannot read {image_path}")
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # 2. Blur + binarize (invert so text+noise==white)
#     blur = cv2.GaussianBlur(gray, (5, 5), 0)
#     _, bw = cv2.threshold(
#         blur, 0, 255,
#         cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
#     )

#     # 3. Remove wiggly line artifacts with morphological opening
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#     cleaned = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=2)

#     # 4. Invert back for standard OCR input (text dark on light)
#     processed = cv2.bitwise_not(cleaned)

#     # 5. Run PaddleOCR
#     results = ocr.ocr(processed, cls=True)
#     if not results or not results[0]:
#         raise ValueError("No text detected")

#     # 6. Concatenate all detected text segments
#     candidate = "".join(txt.strip() for box, (txt, conf) in results[0])
#     if CAPTCHA_PATTERN.fullmatch(candidate):
#         return candidate

#     # 7. Fallback: Remove non-alphanumeric characters and re-test
#     candidate = re.sub(r'[^A-Za-z0-9]', '', candidate)
#     if CAPTCHA_PATTERN.fullmatch(candidate):
#         return candidate

#     # 8. If still no perfect match, raise an error
#     raise ValueError("Failed to extract 6-char code cleanly")


# if __name__ == '__main__':
#     # Example usage
#     files = [ r"D:\Desktop\test\a1.jpg"]
#     for f in files:
#         text = extract_captcha_text(f)
#         print(f,"=>", text)

###################################



# import re
# import cv2
# import numpy as np
# from paddleocr import PaddleOCR

# # 1) Define the 6-char alphanumeric pattern:
# ALPHANUMERIC_6_PATTERN = re.compile(r'^[a-z0-9]{6}$')
# MATH_EXPR_PATTERN = re.compile(r'^\s*\d\s*[\+\-]\s*\d\s*$')

# # 2) Initialize PaddleOCR
# ocr = PaddleOCR(
#     use_angle_cls=True,
#     lang='en',
#     det_db_score_mode="fast",
#     det_db_thresh=0.2,
#     det_db_box_thresh=0.2,
#     det_db_unclip_ratio=2.0
# )

# def extract_captcha_text(image_path: str) -> str:
#     """
#     Reads an image of a 6-character alphanumeric captcha,
#     removes noise lines, then uses PaddleOCR to extract
#     exactly 6 alphanumeric chars. 
#     """
#     # -------------------------------------------------------------------------
#     # STEP A: Load the image and convert to grayscale
#     # -------------------------------------------------------------------------
#     img = cv2.imread(image_path)
#     if img is None:
#         raise FileNotFoundError(f"Cannot read {image_path}")
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # -------------------------------------------------------------------------
#     # STEP B: Blur and binarize (invert so text becomes white, background black)
#     # -------------------------------------------------------------------------
#     blur = cv2.GaussianBlur(gray, (5, 5), 0)
#     _, bw = cv2.threshold(
#         blur, 0, 255,
#         cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
#     )
#      # 3. Remove long horizontal & vertical artifacts
#     h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 7))
#     v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 35))
#     h_removed = cv2.morphologyEx(bw, cv2.MORPH_OPEN, h_kernel)
#     v_removed = cv2.morphologyEx(bw, cv2.MORPH_OPEN, v_kernel)
#     no_lines = cv2.subtract(bw, cv2.bitwise_or(h_removed, v_removed))

#     cv2.imshow("Bilateral Filter", no_lines)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     # -------------------------------------------------------------------------
#     # STEP C: Morphological cleaning - remove small lines/holes
#     #   - Increase kernel size or change morph operations if needed
#     # -------------------------------------------------------------------------
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
#     cleaned = cv2.morphologyEx(no_lines, cv2.MORPH_OPEN, kernel, iterations=1)

#     cv2.imshow("Bilateral Filter", cleaned)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#     # -------------------------------------------------------------------------
#     # STEP D: Invert back so text is dark on light background (standard for OCR)
#     # -------------------------------------------------------------------------
#     processed = cv2.bitwise_not(cleaned)

    

#     # -------------------------------------------------------------------------
#     # STEP E: OCR with PaddleOCR
#     # -------------------------------------------------------------------------
#     results = ocr.ocr(processed, cls=True)
#     if not results or not results[0]:
#         raise ValueError("No text detected by OCR.")

#     # -------------------------------------------------------------------------
#     # STEP F: Collect OCR segments (this is crucial for debugging)
#     # -------------------------------------------------------------------------
#     recognized_segments = []
#     for box, (txt, conf) in results[0]:
#         # Strip out leading/trailing whitespace from each chunk:
#         recognized_segments.append(txt.strip())

#     print("DEBUG - OCR segments recognized:", recognized_segments)

#     # -------------------------------------------------------------------------
#     # STEP G: Join all recognized segments
#     # -------------------------------------------------------------------------
#     joined_all = "".join(recognized_segments)
#     print(f"DEBUG - Joined segments: '{joined_all}'")

#     # Check for 6-char alphanumeric
#     alnum_candidate = re.sub(r'[^a-z0-9]', '', joined_all.lower())
#     if ALPHANUMERIC_6_PATTERN.fullmatch(alnum_candidate):
#         return alnum_candidate

#     # Check for math expression like '3+4', ' 7 - 2 '
#     math_candidate = re.sub(r'[^0-9+\- ]', '', joined_all)
#     if MATH_EXPR_PATTERN.fullmatch(math_candidate):
#         return re.sub(r'\s+', '', math_candidate)  # Normalize and return e.g. '3-4'

#     # -------------------------------------------------------------------------
#     # STEP H: Check each individual segment
#     # -------------------------------------------------------------------------
#     for seg in recognized_segments:
#         seg_clean = seg.strip().lower()
#         seg_alnum = re.sub(r'[^a-z0-9]', '', seg_clean)
#         if ALPHANUMERIC_6_PATTERN.fullmatch(seg_alnum):
#             return seg_alnum

#         seg_math = re.sub(r'[^0-9+\- ]', '', seg_clean)
#         if MATH_EXPR_PATTERN.fullmatch(seg_math):
#             return re.sub(r'\s+', '', seg_math)

#     # -------------------------------------------------------------------------
#     # STEP I: If all fails
#     # -------------------------------------------------------------------------
#     raise ValueError("Failed to extract a valid 6-char code or math expression.")



# if __name__ == '__main__':
#     test_image = r"D:\Desktop\test\a18.png"  # Or your actual file
#     text = extract_captcha_text(test_image)
#     print(f"Final recognized CAPTCHA text = '{text}'")

#######################################################################################


# import re
# import cv2
# import numpy as np
# from paddleocr import PaddleOCR

# ALPHANUMERIC_6_PATTERN = re.compile(r'^[a-z0-9]{6}$')
# MATH_EXPR_PATTERN = re.compile(r'^\s*\d\s*[\+\-]\s*\d\s*$')

# ocr = PaddleOCR(use_angle_cls=True, lang='en')

# def preprocess_for_alphanumeric(img):
#     """Preprocessing for 6-char alphanumeric captchas."""
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray, (5, 5), 0)
#     _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

#     h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 7))
#     v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 35))
#     h_removed = cv2.morphologyEx(bw, cv2.MORPH_OPEN, h_kernel)
#     v_removed = cv2.morphologyEx(bw, cv2.MORPH_OPEN, v_kernel)
#     no_lines = cv2.subtract(bw, cv2.bitwise_or(h_removed, v_removed))

#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
#     cleaned = cv2.morphologyEx(no_lines, cv2.MORPH_OPEN, kernel)
#     return cv2.bitwise_not(cleaned)

# def preprocess_for_math(img):
#     """Simpler preprocessing for math expressions like '3+2' or '9 - 1'."""
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
#     cv2.imshow("Bilateral Filter", thresh)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     return cv2.bitwise_not(thresh)
    


# def run_ocr(image):
#     """Apply PaddleOCR and return joined text segments."""
#     results = ocr.ocr(image, cls=True)
#     if not results or not results[0]:
#         raise ValueError("No text detected by OCR.")
#     return "".join(txt.strip() for box, (txt, conf) in results[0])

# def extract_captcha_text(image_path: str) -> str:
#     img = cv2.imread(image_path)
#     if img is None:
#         raise FileNotFoundError(f"Cannot read {image_path}")

#     # First: try math preprocessing
#     processed_math = preprocess_for_math(img)
#     text_math = run_ocr(processed_math, label="math")
    
#     clean_math = re.sub(r'[^0-9+\- ]', '', text_math)
#     if MATH_EXPR_PATTERN.fullmatch(clean_math):
#         expr = re.sub(r'\s+', '', clean_math)
#         print(f"Recognized math expression: '{expr}'")
#         try:
#             return str(eval(expr))  # 3+2 => 5
#         except Exception as e:
#             print(f"Math eval failed: {e}")

#     # Second: try alphanumeric preprocessing
#     processed_alnum = preprocess_for_alphanumeric(img)
#     text_alnum = run_ocr(processed_alnum)
#     clean_alnum = re.sub(r'[^a-z0-9]', '', text_alnum.lower())
#     if ALPHANUMERIC_6_PATTERN.fullmatch(clean_alnum):
#         return clean_alnum

#     raise ValueError("Failed to extract a valid 6-char code or math expression.")

# if __name__ == '__main__':
#     test_image = r"D:\Desktop\test\a18.png"
#     text = extract_captcha_text(test_image)
#     print(f"Final CAPTCHA result = '{text}'")

##########################################################################################

# import re
# import cv2
# import numpy as np
# from paddleocr import PaddleOCR

# # Pattern for math expression like "3+2", " 7 - 4"
# MATH_EXPR_PATTERN = re.compile(r'^\s*\d\s*[\+\-]\s*\d\s*$')

# # Initialize OCR
# ocr = PaddleOCR(
#     use_angle_cls=True,
#     lang='en',
#     det_db_score_mode="fast",
#     det_db_thresh=0.2,
#     det_db_box_thresh=0.2,
#     det_db_unclip_ratio=2.0
# )

# def extract_math_captcha(image_path: str, solve: bool = True) -> str:
#     """
#     Extract and optionally solve a math captcha (e.g., '3 + 2' -> '5').
#     """
#     # Load image
#     img = cv2.imread(image_path)
#     if img is None:
#         raise FileNotFoundError(f"Cannot read image: {image_path}")
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # Resize image for better OCR performance
#     scale_percent = 300  # enlarge by 300%
#     width = int(gray.shape[1] * scale_percent / 100)
#     height = int(gray.shape[0] * scale_percent / 100)
#     gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_LINEAR)


#     # Preprocessing for math captchas
#     blur = cv2.GaussianBlur(gray, (7, 5), 0)
#     _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

#     # Adaptive thresholding
#     # thresh = cv2.adaptiveThreshold(
#     #     gray, 255,
#     #     cv2.ADAPTIVE_THRESH_MEAN_C,
#     #     cv2.THRESH_BINARY_INV,
#     #     blockSize=35,
#     #     C=20
#     # )
#     cv2.imshow("Bilateral Filter", thresh)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


#     # Invert to make text dark on white
#     processed = cv2.bitwise_not(thresh)
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
#     processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)


#     cv2.imshow("Bilateral Filter", processed)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

    
#     # OCR
#     results = ocr.ocr(processed, cls=True)
#     if not results or not results[0]:
#         raise ValueError("No text detected by OCR.")

#     # Sort OCR boxes left-to-right using x-coordinate
#     sorted_results = sorted(results[0], key=lambda r: r[0][0][0])  # r[0][0][0] is x of top-left point

#     # Extract sorted boxes and texts
#     boxes = [box for (box, (text, conf)) in sorted_results]
#     recognized_segments = [text for (box, (text, conf)) in sorted_results]

#     print("DEBUG - OCR segments recognized:", recognized_segments)

#     # Heuristic: insert missing minus sign if only two segments are detected
#     # Heuristic: if only two segments, assume missing operator and insert '-'
#     if len(recognized_segments) == 2:
#         recognized_segments.insert(1, '-')
#         print("DEBUG - Heuristically inserted minus sign.")

#     full_expression = ''.join(recognized_segments).replace(' ', '')
#     print("DEBUG - Cleaned math expression:", repr(full_expression))


#     if solve:
#         try:
#             result = eval(full_expression)
#             print("DEBUG - Math result:", result)
#             return str(result)
#         except Exception as e:
#             raise ValueError("Extracted text is not a valid math expression.") from e

#     return full_expression


# if __name__ == '__main__':
#     test_image = r"D:\Desktop\test\a20.png"  # Your math captcha image path
#     result = extract_math_captcha(test_image, solve=True)
#     print(f"Final result: {result}")


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


    cv2.imshow("Bilateral Filter", thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Deskew
    # thresh = deskew_image(thresh)

    # Invert and morph for clarity
    processed = cv2.bitwise_not(thresh)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel, iterations=1)

    cv2.imshow("Bilateral Filter", thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    # OCR
    results = ocr.ocr(processed, cls=True)
    if not results or not results[0]:
        raise ValueError("No text detected by OCR.")

    # Sort OCR results left to right
    sorted_results = sorted(results[0], key=lambda r: r[0][0][0])
    recognized_segments = [text for (box, (text, conf)) in sorted_results]

    print("DEBUG - OCR segments recognized:", recognized_segments)

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
    test_image = r"D:\Desktop\test\a11.png"  # Update with your test image
    result = extract_math_captcha(test_image, solve=True)
    print(f"Final result: {result}")
