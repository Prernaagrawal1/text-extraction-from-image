import cv2, os
import pytesseract
from pathlib import Path
import csv

IN_DIR = Path("image")
OUT_CSV = "image/ocr_results.csv"
paths = sorted([p for p in IN_DIR.iterdir() if p.suffix.lower() in (".png",".jpg",".jpeg")])

with open(OUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["filename","psm","text"])

    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # auto-invert if needed (same logic as before)
        _, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        white_frac = (otsu==255).sum() / otsu.size
        if white_frac < 0.5:
            img = cv2.bitwise_not(img)

        # optional upscale + CLAHE
        h,w = img.shape
        if max(h,w) < 1000:
            img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        img = clahe.apply(img)
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # choose psm that worked for you, or try list
        psm = 6
        cfg = f"--oem 3 --psm {psm}"
        text = pytesseract.image_to_string(img, config=cfg).strip()
        writer.writerow([p.name, psm, text])

print("Saved:", OUT_CSV)
