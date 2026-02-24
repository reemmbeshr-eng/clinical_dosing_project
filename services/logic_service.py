import re
import math

# Detect dose type
def detect_dose_type(dosage_text):
    text = dosage_text.lower()
    if "mg/kg/day" in text:
        return "MG_KG_DAY"
    if "mg/kg/dose" in text:
        return "MG_KG_DOSE"
    if "mg/m2/day" in text or "mg/m²/day" in text:
        return "MG_M2_DAY"
    return "UNKNOWN"


# Extract numeric dose range
def extract_dose_range(dosage_text):
    numbers = re.findall(r"\d+\.?\d*", dosage_text)
    nums = [float(n) for n in numbers]
    if not nums:
        return None
    if len(nums) == 1:
        return nums[0], nums[0]
    return nums[0], nums[1]


# Extract max daily dose (mg)
def extract_max_daily_dose(dosage_text):
    match = re.search(r"max(imum)? daily dose[:\s]*([\d\.]+)\s*(mg|g)", dosage_text.lower())
    if not match:
        return None
    value = float(match.group(2))
    unit = match.group(3)
    if unit == "g":
        value *= 1000
    return value


# BSA (Mosteller)
def calculate_bsa(weight, height):
    return math.sqrt((weight * height) / 3600)


# Core pediatric dose calculation
def calculate_pediatric_dose_base(dosage_text, weight, height=None):
    dose_type = detect_dose_type(dosage_text)
    dose_range = extract_dose_range(dosage_text)
    if not dose_range:
        return None

    low, high = dose_range

    if dose_type == "MG_KG_DAY":
        daily_low = low * weight
        daily_high = high * weight

    elif dose_type == "MG_KG_DOSE":
        per_dose_low = low * weight
        per_dose_high = high * weight
        return {
            "dose_type": dose_type,
            "per_dose_mg": (per_dose_low, per_dose_high)
        }

    elif dose_type == "MG_M2_DAY":
        if height is None:
            return None
        bsa = calculate_bsa(weight, height)
        daily_low = low * bsa
        daily_high = high * bsa

    else:
        return None

    # apply max daily dose if present
    max_daily = extract_max_daily_dose(dosage_text)
    if max_daily:
        daily_low = min(daily_low, max_daily)
        daily_high = min(daily_high, max_daily)

    return {
        "dose_type": dose_type,
        "daily_dose_mg": (daily_low, daily_high)
    }


# Interval-based division (user-defined)
def divide_daily_dose(daily_low, daily_high, interval_hours):
    doses_per_day = 24 / interval_hours
    per_dose_low = daily_low / doses_per_day
    per_dose_high = daily_high / doses_per_day
    return doses_per_day, per_dose_low, per_dose_high