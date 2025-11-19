# Text Extraction from Images using OCR & Otsu Thresholding

This project performs **text extraction (OCR)** from a batch of images using **image preprocessing**, **Tesseract OCR**, and **confidence score analysis**.  
It also generates confidence CSV files and a confidence graph to analyse OCR accuracy.

This project was developed for text extraction testing, accuracy checking, and preprocessing evaluation.

---

## 📁 Project Structure

ocr_otsu/
├── image/ # Input test images (10 images)
├── batch_ocr.py # Run OCR on a single image
├── batch_ocr_to_csv.py # Run OCR on all images & save to CSV
├── create_ocr_confidence_csv.py # Generate CSV containing confidence values
├── generate_confidence_graph.py # Plot OCR confidence graph
├── noisy_search.py # Experiment script for noisy image OCR
├── ocr_confidence_graph.png # Output graph (OCR confidence)
├── ocr_confidence_results.csv # Output CSV (text + confidence)
└── test.py # Preprocessing + sample OCR test


---

## 🔧 Requirements

Install packages:

Install **Tesseract OCR** on Windows:  
Download from: https://github.com/tesseract-ocr/tesseract

If needed, set the path inside your scripts:

```python
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

▶️ Running the Project
1. Run OCR on all images and save results to CSV
python batch_ocr_to_csv.py


Output CSV:

ocr_confidence_results.csv

2. Generate OCR confidence CSV
python create_ocr_confidence_csv.py

3. Generate the confidence graph
python generate_confidence_graph.py


Output graph:

ocr_confidence_graph.png

4. Test preprocessing + OCR
python test.py

5. Run OCR on noisy images
python noisy_search.py

📊 Results
✔ OCR Output CSV

File: ocr_confidence_results.csv
Contains:

Extracted text

Character confidences

Word-level OCR scores

✔ Confidence Graph

File: ocr_confidence_graph.png
Shows confidence distribution across all images.

🧠 Concepts Used

Image preprocessing

Grayscale conversion

Gaussian blur

Otsu's thresholding

OCR using Tesseract

Confidence score analysis

Text extraction into CSV

Visualization with Matplotlib

🚀 Future Scope

Improve preprocessing using morphological operations

Add spelling correction on OCR text

Add GUI for uploading images

Add character-level accuracy graph

Deploy as a web app
