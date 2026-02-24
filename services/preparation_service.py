import re

def fallback_reconstitution(preparation_text):
    """
    Rule-based fallback extraction.
    Returns partial data if found.
    """

    result = {
        "reconstitution": [],
        "max_concentration_mg_ml": None
    }

    # max concentration
    max_match = re.search(
        r"(not exceed|max).*?(\d+\.?\d*)\s*mg\s*/\s*mL",
        preparation_text.lower()
    )
    if max_match:
        result["max_concentration_mg_ml"] = float(max_match.group(2))

    # vial + volume (simple patterns)
    pairs = re.findall(
        r"([\d,]+)\s*mg.*?(\d+\.?\d*)\s*mL",
        preparation_text.lower()
    )

    for vial, vol in pairs:
        result["reconstitution"].append({
            "vial_mg": int(vial.replace(",", "")),
            "volume_ml": float(vol)
        })

    return result