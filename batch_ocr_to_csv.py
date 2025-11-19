# batch_ocr_to_csv.py
import cv2, os, csv
import pytesseract
from pathlib import Path
import numpy as np
from pytesseract import Output

IMG_DIR = Path("image")
OUT_DIR = IMG_DIR / "ocr_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = OUT_DIR / "ocr_results.csv"

EXTS = {".png", ".jpg", ".jpeg"}
PSM_TRY = [6, 7, 3]   # order to try; choose the first non-empty result

def preprocess(img):
    # img: BGR color image -> returns binary cleaned image
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    # upscale if small
    h, w = gray.shape
    if max(h,w) < 1000:
        gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
    # auto invert if needed
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    white_frac = (otsu == 255).sum() / otsu.size
    if white_frac < 0.5:
        gray = cv2.bitwise_not(gray)
    # contrast and denoise
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    # binarize + close
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    return clean

def annotate_and_save_boxes(orig_color, out_path, cfg="--oem 3 --psm 6"):
    data = pytesseract.image_to_data(orig_color, output_type=Output.DICT, config=cfg)
    img = orig_color.copy()
    for i, txt in enumerate(data['text']):
        t = txt.strip()
        if not t:
            continue
        x,y,w,h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
    cv2.imwrite(str(out_path), img)

def process_file(path: Path):
    orig = cv2.imread(str(path))
    if orig is None:
        return (path.name, None, "")
    pre = preprocess(orig)
    pre_path = OUT_DIR / f"{path.stem}_pre.png"
    cv2.imwrite(str(pre_path), pre)
    chosen_psm = None
    chosen_text = ""
    for psm in PSM_TRY:
        cfg = f"--oem 3 --psm {psm}"
        text = pytesseract.image_to_string(pre, config=cfg).strip()
        if text:
            chosen_psm = psm
            chosen_text = text
            break
    # If still empty, save last attempt psm for traceability
    if chosen_psm is None:
        chosen_psm = PSM_TRY[0]
    # save annotated boxes on original (for context)
    boxes_out = OUT_DIR / f"{path.stem}_boxes.png"
    annotate_and_save_boxes(orig, boxes_out, cfg=f"--oem 3 --psm {chosen_psm}")
    return (path.name, chosen_psm, chosen_text)

def main():
    files = [p for p in sorted(IMG_DIR.iterdir()) if p.suffix.lower() in EXTS and p.parent != OUT_DIR]
    if not files:
        print("No input images found in 'image/'")
        return
    rows = []
    for p in files:
        print("Processing:", p.name)
        fn, psm, txt = process_file(p)
        rows.append((fn, psm, txt.replace("\n", " ").strip()))
        print(f" -> {fn}  psm={psm}  len(text)={len(txt)}")
    # write CSV
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename","psm","text"])
        for r in rows:
            writer.writerow(r)
    print("Saved CSV:", CSV_PATH)
    print("All outputs in:", OUT_DIR)

if __name__ == "__main__":
    main()
