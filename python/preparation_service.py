import re

def extract_concentration(preparation_text):
    """
    Extract reconstitution concentration (mg/mL) from preparation text
    """
    match = re.search(r"(\d+\.?\d*)\s*mg\s*/\s*mL", preparation_text.lower())
    if match:
        return float(match.group(1))
    return None


def calculate_withdrawal_volume(required_dose_mg, concentration_mg_per_ml):
    """
    Calculate volume to withdraw (mL)
    """
    return required_dose_mg / concentration_mg_per_ml

def extract_max_concentration(preparation_text):
    """
    Extract maximum allowed final concentration (mg/mL) if present
    """
    match = re.search(r"not exceed\s*(\d+\.?\d*)\s*mg/mL", preparation_text.lower())
    if match:
        return float(match.group(1))
    return None


def extract_reconstitution_table(preparation_text):
    """
    Extract vial strength and reconstitution volume mapping from text.
    Returns dict like {500: 5, 1000: 7.4}
    """
    table = {}

    # 25 / 250 / 500 mg vials: 5 mL
    multi_match = re.search(
        r"(\d+)\s*mg,\s*(\d+)\s*mg,\s*or\s*(\d+)\s*mg vials:\s*(\d+\.?\d*)\s*mL",
        preparation_text.lower()
    )
    if multi_match:
        vols = float(multi_match.group(4))
        for i in range(1, 4):
            table[int(multi_match.group(i))] = vols

    # single vial lines like "1,000 mg vial: 7.4 mL"
    singles = re.findall(
        r"([\d,]+)\s*mg vial:\s*(\d+\.?\d*)\s*mL",
        preparation_text.lower()
    )

    for strength, vol in singles:
        table[int(strength.replace(",", ""))] = float(vol)

    return table