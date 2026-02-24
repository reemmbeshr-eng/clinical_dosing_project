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

## 🛠️ Tech Stack
- Python
- Streamlit (UI)
- PostgreSQL
- PyTorch (ResNet18)
- OCR (Tesseract)
- Ollama (LLM assistance)

## 📂 Project Structure
See folder structure above.

## ⚠️ Disclaimer
This tool is for educational and research purposes only.
It does not replace clinical judgment.

## 📌 Future Work
- Auto-calculation of CrCl
- Contraindication alerts
- Expanded drug database
