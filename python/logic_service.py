import re
import math

# -------------------------
# Dose type detection
# -------------------------

def detect_dose_type(dosage_text):
    text = dosage_text.lower()

    if "mg/kg/day" in text:
        return "MG_KG_DAY"
    elif "mg/kg/dose" in text:
        return "MG_KG_DOSE"
    elif "mg/m2/day" in text or "mg/m²/day" in text:
        return "MG_M2_DAY"
    else:
        return "UNKNOWN"


# -------------------------
# Extract dose range (mg)
# -------------------------

def extract_dose_range(dosage_text):
    numbers = re.findall(r"\d+\.?\d*", dosage_text)
    numbers = [float(n) for n in numbers]

    if len(numbers) >= 2:
        return numbers[0], numbers[1]
    elif len(numbers) == 1:
        return numbers[0], numbers[0]
    else:
        return None


# -------------------------
# Extract max daily dose (mg)
# -------------------------

def extract_max_daily_dose(dosage_text):
    match = re.search(r"max(imum)? daily dose[:\s]*([\d\.]+)\s*(mg|g)", dosage_text.lower())
    if not match:
        return None

    value = float(match.group(2))
    unit = match.group(3)

    if unit == "g":
        value *= 1000

    return value


# -------------------------
# Extract dosing interval
# -------------------------

def extract_doses_per_day(dosage_text):
    text = dosage_text.lower()

    if "q6h" in text or "every 6 hours" in text:
        return 4
    if "q8h" in text or "every 8 hours" in text:
        return 3
    if "q12h" in text or "every 12 hours" in text:
        return 2

    # default once daily
    return 1


# -------------------------
# BSA (Mosteller)
# -------------------------

def calculate_bsa(weight, height):
    return math.sqrt((weight * height) / 3600)


# -------------------------
# Core dose calculation
# -------------------------

def calculate_pediatric_dose(
    dosage_text,
    weight,
    height=None
):
    dose_type = detect_dose_type(dosage_text)
    dose_range = extract_dose_range(dosage_text)

    if not dose_range:
        return None

    low, high = dose_range

    # Step 1: total daily dose
    if dose_type == "MG_KG_DAY":
        daily_low = low * weight
        daily_high = high * weight

    elif dose_type == "MG_KG_DOSE":
        daily_low = low * weight
        daily_high = high * weight

    elif dose_type == "MG_M2_DAY":
        bsa = calculate_bsa(weight, height)
        daily_low = low * bsa
        daily_high = high * bsa

    else:
        return None

    # Step 2: apply max daily dose
    max_daily = extract_max_daily_dose(dosage_text)
    if max_daily:
        daily_low = min(daily_low, max_daily)
        daily_high = min(daily_high, max_daily)

    # Step 3: divide doses
    doses_per_day = extract_doses_per_day(dosage_text)

    per_dose_low = daily_low / doses_per_day
    per_dose_high = daily_high / doses_per_day

    return {
        "daily_dose_mg": (daily_low, daily_high),
        "dose_per_administration_mg": (per_dose_low, per_dose_high),
        "doses_per_day": doses_per_day
    }
