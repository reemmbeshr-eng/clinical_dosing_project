# 💊 Pediatric Clinical Dosing Assistant

A clinical decision support system for pediatric drug dosing,
integrating rule-based logic, renal-adjusted dosing, and AI-assisted
drug recognition.

## 🚀 Features
- Pediatric dose calculation (mg/kg, mg/m²)
- Renal-adjusted dosing based on GFR / CrCl
- AI-assisted drug identification from images
- Dose preparation and vial reconstitution guidance
- Safety alerts for high-risk doses

## 🧠 Clinical Logic
- Standard dosing vs renal-adjusted dosing selected before calculation
- Renal ranges matched dynamically from structured database rows
- No unsafe assumptions when renal data is missing

## 🛠️ Tools & Technologies

### 🔹 Programming & Frameworks
- **Python 3** — core application logic
- **Streamlit** — interactive clinical UI

---

### 🔹 Clinical Logic
- Custom rule-based dosing engine
- Clear separation between:
  - Standard dosing
  - Renal-adjusted dosing
- Explainable, traceable calculations

---

### 🔹 Database & Data Layer
- **PostgreSQL** — structured clinical reference storage
- **psycopg2** — database connectivity
- Renal dosing stored as independent rows for scalability

---

### 🔹 Machine Learning
- **PyTorch**
- **ResNet-18** for image-based drug classification
- **Torchvision** for image preprocessing

---

### 🔹 Image Processing & OCR
- **OpenCV** — image preprocessing
- **Tesseract OCR** — text extraction
- **Pillow (PIL)** — image handling

---

### 🔹 LLM Integration
- **Ollama (Local LLM Runtime)**
  - Drug & indication extraction from free text
  - Natural-language explanation of dose calculations
  - Local execution for data privacy


## 📂 Project Structure
See folder structure above.

## ⚠️ Disclaimer
This tool is for educational and research purposes only.
It does not replace clinical judgment.

## 📌 Future Work
- Auto-calculation of CrCl
- Contraindication alerts
- Expanded drug database
