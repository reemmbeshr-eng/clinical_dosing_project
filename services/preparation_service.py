import re


def rule_based_reconstitution(preparation_text):
    """
    Deterministic rule-based extraction.
    Handles:
    - Reconstitution (vial mg + mL)
    - Dilution-only instructions
    - Max concentration limits
    """

    result = {
        "reconstitution": [],
        "max_concentration_mg_ml": None,
        "dilution_only": False
    }

    text = preparation_text.lower()

    # ---------------------------------
    # Extract max concentration limit
    # ---------------------------------
    max_match = re.search(
        r"(not exceed|≤|<=|maximum|max).*?(\d+\.?\d*)\s*mg\s*/\s*m[l]",
        text
    )

    if max_match:
        result["max_concentration_mg_ml"] = float(max_match.group(2))

    # ---------------------------------
    # Extract reconstitution patterns
    # ---------------------------------

    pattern1 = re.findall(
        r"(\d[\d,]*)\s*mg\s*vial\s*[:\-]\s*(\d+\.?\d*)\s*m[l]",
        text
    )

    pattern2 = re.findall(
        r"reconstitute\s*(\d[\d,]*)\s*mg\s*vial.*?(\d+\.?\d*)\s*m[l]",
        text
    )

    pattern3 = re.findall(
        r"(\d[\d,]*)\s*mg\s*vial.*?(\d+\.?\d*)\s*m[l]",
        text
    )

    all_pairs = pattern1 + pattern2 + pattern3

    seen = set()

    for vial, vol in all_pairs:
        vial_clean = int(vial.replace(",", ""))
        vol_clean = float(vol)

        if (vial_clean, vol_clean) not in seen:
            seen.add((vial_clean, vol_clean))
            result["reconstitution"].append({
                "vial_mg": vial_clean,
                "volume_ml": vol_clean
            })

    # ---------------------------------
    # Detect dilution-only case
    # ---------------------------------
    if not result["reconstitution"]:
        dilution_match = re.search(
            r"(dilute|final concentration).*?(\d+\.?\d*)\s*mg\s*/\s*m[l]",
            text
        )
        if dilution_match:
            result["dilution_only"] = True

    return result