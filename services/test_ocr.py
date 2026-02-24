from .ai_vision_service import extract_text_from_image, extract_vial_strength_mg

text = extract_text_from_image("ampicillin.jpg")
print("OCR text:", text)

strength = extract_vial_strength_mg(text)
print("Detected vial strength (mg):", strength)
