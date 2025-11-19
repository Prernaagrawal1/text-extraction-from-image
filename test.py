# test.py  — robust preprocessing + multi-config OCR
import cv2, os
import pytesseract
from pathlib import Path
import numpy as np

# If tesseract is NOT on PATH, uncomment and set the path:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

IN_IMG = "image/sample.png"               # your original
PRE_IMG = "image/sample_preprocessed.png" # saved preprocessed image
OUT_DIR = Path("image/ocr_debug")
OUT_DIR.mkdir(parents=True, exist_ok=True)

if not os.path.exists(IN_IMG):
    raise FileNotFoundError(f"Place your image at {IN_IMG}")

# 1. Read as color then convert to gray
orig = cv2.imread(IN_IMG, cv2.IMREAD_COLOR)
if orig is None:
    raise FileNotFoundError("Failed to open " + IN_IMG)
gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)

# 2. Upscale if small (helps tiny text)
h, w = gray.shape
scale = 1.5 if max(h, w) < 1000 else 1.0
if scale != 1.0:
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

# 3. Auto-detect white text on black background and invert if needed
# If mean intensity is low (dark overall) and text appears lighter, invert.
mean = np.mean(gray)
# compute a simple foreground darkness metric using Otsu threshold
_, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# otsu will give white foreground; check which is foreground by counting white pixels
white_frac = np.count_nonzero(otsu == 255) / (otsu.size)
# If image mostly white (white_frac > 0.5) then text likely black on white (do nothing).
# If image mostly black (white_frac < 0.5) then text likely white on black -> invert.
if white_frac < 0.5:
    gray = cv2.bitwise_not(gray)
    inverted = True
else:
    inverted = False

# 4. Increase contrast using CLAHE
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
gray = clahe.apply(gray)

# 5. Slight blur to remove tiny noise, then Otsu threshold for crisp binary
gray = cv2.GaussianBlur(gray, (3,3), 0)
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 6. Morphological closing to fill gaps inside letters (useful for bold fonts)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

# 7. Save preprocessing outputs for inspection
cv2.imwrite(PRE_IMG, clean)
cv2.imwrite(str(OUT_DIR/"step_gray.png"), gray)
cv2.imwrite(str(OUT_DIR/"step_thresh.png"), thresh)
cv2.imwrite(str(OUT_DIR/"step_clean.png"), clean)

print(f"[info] Preprocessing done. inverted_taken = {inverted}")
print(f"[info] Preprocessed image saved to: {PRE_IMG}")

# 8. Prepare OCR configs to try
psm_list = [3, 6, 7]   # good defaults
oem = 3
results = []

for psm in psm_list:
    cfg = f"--oem {oem} --psm {psm}"
    txt = pytesseract.image_to_string(clean, config=cfg).strip()
    results.append((psm, cfg, txt))
# also try a trimmed config forcing common letters (if needed)
cfg_whitelist = "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz --oem 3 --psm 7"
txt_whitelist = pytesseract.image_to_string(clean, config=cfg_whitelist).strip()
results.append(("whitelist_alpha_psm7", cfg_whitelist, txt_whitelist))

# 9. Print and save results
for i, (psm, cfg, txt) in enumerate(results, 1):
    header = f"RESULT {i}  (psm={psm})"
    print("\n" + "="*len(header))
    print(header)
    print("="*len(header))
    if txt:
        print(txt)
    else:
        print("[no text extracted]")
    # save
    out_file = OUT_DIR / f"ocr_result_psm_{psm}.txt"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(f"config: {cfg}\n\n")
        f.write(txt)
    print(f"[saved] {out_file}")

print("\n[done]")
