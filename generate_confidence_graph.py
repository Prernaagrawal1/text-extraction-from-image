import csv
import matplotlib.pyplot as plt

files = []
conf = []

with open("ocr_confidence_results.csv") as f:
    r = csv.DictReader(f)
    for row in r:
        files.append(row["filename"])
        conf.append(float(row["avg_confidence"]))

plt.figure(figsize=(12,6))
plt.plot(files, conf, marker='o')
plt.xticks(rotation=90)
plt.ylabel("Tesseract Confidence (%)")
plt.title("OCR Confidence Across Images")
plt.tight_layout()
plt.savefig("ocr_confidence_graph.png")
print("Saved: ocr_confidence_graph.png")
