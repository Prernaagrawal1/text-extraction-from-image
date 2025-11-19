import os
import csv
import pytesseract
from pytesseract import Output
from PIL import Image

IMAGE_DIR = "image"
OUT_CSV = "ocr_confidence_results.csv"

def get_conf(path):
    data = pytesseract.image_to_data(Image.open(path), output_type=Output.DICT)
    confs = [int(c) for c in data['conf'] if c != '-1']
    return sum(confs)/len(confs) if confs else 0

with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["filename", "avg_confidence"])
    for fname in os.listdir(IMAGE_DIR):
        if fname.lower().endswith((".png",".jpg",".jpeg")):
            path = os.path.join(IMAGE_DIR, fname)
            conf = get_conf(path)
            w.writerow([fname, f"{conf:.2f}"])

print("Saved:", OUT_CSV)
