#C:\Program Files\Tesseract-OCR
# ai_vision_service.py
import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


import cv2


def extract_text_from_image(image_path: str) -> str:
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]

    text = pytesseract.image_to_string(gray, config="--psm 6")
    return text.lower()


import re

def extract_vial_strength_mg(ocr_text: str):
    """
    Extract vial strength in mg from noisy OCR text.
    """

    text = ocr_text.lower()

    numbers = re.findall(r"\d+", text)

    if not numbers:
        return None

    common_strengths = {50, 100, 250, 500, 750, 1000, 2000}

    for n in numbers:
        val = int(n)

        if val in common_strengths:
            return val

        if val in {1, 2}:
            return val * 1000

    return None


