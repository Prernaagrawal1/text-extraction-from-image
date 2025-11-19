# noisy_search.py
# Tries multiple preprocessing parameter combinations on one noisy image,
# runs OCR (uppercase whitelist) and ranks results by simple avg confidence.
import cv2, os, itertools, numpy as np
import pytesseract
from pytesseract import Output
from pathlib import Path

INPUT = "image/text_006_noisy.png"   # change if needed
OUT_DIR = Path("image/noisy_search_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

if not os.path.exists(INPUT):
    raise FileNotFoundError("Put your noisy image at: " + INPUT)

orig = cv2.imread(INPUT)
if orig is None:
    raise FileNotFoundError("Failed to open " + INPUT)

# parameter ranges to try
median_sizes = [3, 5, 7]           # median blur kernel (odd)
nlm_h_vals = [15, 30, 45]          # fastNlMeansDenoising h
adaptive_params = [(31,8), (41,6), (31,12)]  # (blockSize, C)
min_areas = [80, 150, 300]
psm_list = [7, 6]                  # try single-line and single-block
whitelist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

results = []

def preprocess_with_params(img, med_k, nlm_h, adapt_block, adapt_C, min_area):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if med_k > 1:
        gray = cv2.medianBlur(gray, med_k)
    gray = cv2.fastNlMeansDenoising(gray, h=nlm_h, templateWindowSize=7, searchWindowSize=21)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=1)
    th = cv2.adaptiveThreshold(opened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, adapt_block, adapt_C)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    clean = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel2, iterations=1)
    # connected components filter
    nb, labels, stats, _ = cv2.connectedComponentsWithStats(clean, connectivity=8)
    mask = np.zeros(clean.shape, dtype="uint8")
    for i in range(1, nb):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            mask[labels == i] = 255
    return mask

def ocr_and_confidence(img, psm):
    cfg = f"--oem 3 --psm {psm} -c tessedit_char_whitelist={whitelist}"
    # use image_to_data to get confidences
    data = pytesseract.image_to_data(img, config=cfg, output_type=Output.DICT)
    texts = []
    confs = []
    n = len(data['text'])
    for i in range(n):
        t = data['text'][i].strip()
        try:
            conf = float(data['conf'][i])
        except:
            conf = -1.0
        if t:
            texts.append(t)
            confs.append(conf if conf >= 0 else 0.0)
    full_text = " ".join(texts).strip()
    avg_conf = (sum(confs)/len(confs)) if len(confs)>0 else -1.0
    return full_text, avg_conf, data

# iterate combinations
comb_index = 0
for med_k, nlm_h, (bsize, C), min_area, psm in itertools.product(
        median_sizes, nlm_h_vals, adaptive_params, min_areas, psm_list):
    comb_index += 1
    clean = preprocess_with_params(orig, med_k, nlm_h, bsize, C, min_area)
    text, conf, data = ocr_and_confidence(clean, psm)
    # save debug image for top candidates later, but small preview for each
    preview_name = OUT_DIR / f"preview_{comb_index}_m{med_k}_h{nlm_h}_b{bsize}_C{C}_a{min_area}_psm{psm}.png"
    cv2.imwrite(str(preview_name), clean)
    results.append({
        "index": comb_index,
        "med": med_k,
        "nlm_h": nlm_h,
        "block": bsize,
        "C": C,
        "min_area": min_area,
        "psm": psm,
        "text": text,
        "conf": conf,
        "preview": str(preview_name)
    })
    print(f"[tried #{comb_index}] med={med_k} nlm_h={nlm_h} block={bsize} C={C} min_area={min_area} psm={psm} -> text='{text}' conf={conf:.2f}")

# sort by confidence (desc) then by length of text (desc)
results_sorted = sorted(results, key=lambda r: (r['conf'], len(r['text'])), reverse=True)

# save top 6 results and print
print("\nTOP RESULTS:")
for i, r in enumerate(results_sorted[:6], 1):
    print(f"{i}. idx={r['index']} conf={r['conf']:.2f} psm={r['psm']} med={r['med']} nlm_h={r['nlm_h']} block={r['block']} C={r['C']} min_area={r['min_area']} text='{r['text']}' preview={r['preview']}")
    # also save an annotated version showing boxes on original
    cfg = f"--oem 3 --psm {r['psm']} -c tessedit_char_whitelist={whitelist}"
    data = pytesseract.image_to_data(cv2.imread(r['preview']), config=cfg, output_type=Output.DICT)
    # make a color version to draw boxes
    box_img = cv2.cvtColor(cv2.imread(r['preview']), cv2.COLOR_GRAY2BGR)
    for j,t in enumerate(data['text']):
        if t.strip():
            x,y,w,h = data['left'][j], data['top'][j], data['width'][j], data['height'][j]
            cv2.rectangle(box_img, (x,y), (x+w, y+h), (0,255,0), 1)
    outbox = OUT_DIR / f"top_{i}_idx{r['index']}.png"
    cv2.imwrite(str(outbox), box_img)
    print("    saved preview box:", outbox)

# write a summary CSV
import csv
csv_path = OUT_DIR / "search_summary.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["index","med","nlm_h","block","C","min_area","psm","conf","text","preview"])
    for r in results_sorted:
        w.writerow([r['index'], r['med'], r['nlm_h'], r['block'], r['C'], r['min_area'], r['psm'], r['conf'], r['text'], r['preview']])
print("\nSummary written to:", csv_path)
print("Top preview images + box images saved to:", OUT_DIR)
