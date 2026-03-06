An AI-powered application that combines Computer Vision, NLP, and clinical logic to assist in medical decision workflows.

Built as an interactive Streamlit application.

🚀 System Components
🩻 Chest X-ray Pneumonia Detection

Detects pneumonia from chest X-ray images.

Technologies used

PyTorch → building the CNN model

Computer Vision → medical image classification

Image preprocessing → resizing and tensor transformation

🧾 Clinical Text Processing

Extracts drug name and indication from clinical text queries.

Technologies used

NLP parsing → extract medical entities

LLM (Ollama) → assist entity extraction from text

💊 Drug Image Recognition

Detects drug information from vial images.

Technologies used

CNN image classification → drug name detection

OCR → extract text from drug labels

Image processing → detect vial strength

👶 Pediatric Dose Calculation

Calculates drug dosing based on patient parameters.

Technologies used

Rule-based clinical logic → dose calculation

Python algorithms → mg/kg and mg/m² calculations

🧠 Renal Dose Adjustment

Adjusts dosing according to kidney function.

Technologies used

Rule-based decision logic → GFR-based dose selection

💉 Drug Preparation & Dilution

Determines preparation instructions for drug administration.

Technologies used

Text parsing → extract preparation instructions

Mathematical calculations → concentration and dilution

⚠️ Safety Evaluation

Performs dose safety validation.

Technologies used

Clinical safety rules → detect unsafe dosing

🤖 AI Clinical Explanation

Explains the calculated dose and decision logic.

Technologies used

LLM (Ollama) → generate clinical explanation

🧠 Tech Stack

## Programming

- Python

## Framework

- Streamlit

## Machine Learning

- PyTorch
- CNN models

## Computer Vision

- Image classification
- OCR extraction

## NLP

- Deterministic NLP
- LLM-assisted extraction

## AI Explanation

- Ollama LLM