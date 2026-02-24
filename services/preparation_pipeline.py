from .ai_preparation_service import extract_reconstitution_ai
from .preparation_service import fallback_reconstitution

def extract_reconstitution(preparation_text):
    """
    Final unified pipeline:
    AI first → fallback rules fill missing data
    """

    ai_data = extract_reconstitution_ai(preparation_text)
    rule_data = fallback_reconstitution(preparation_text)

    final = {
        "reconstitution": ai_data["reconstitution"],
        "max_concentration_mg_ml": ai_data["max_concentration_mg_ml"]
    }

    # if AI failed to extract reconstitution list
    if not final["reconstitution"]:
        final["reconstitution"] = rule_data["reconstitution"]

    # if AI failed to extract max concentration
    if final["max_concentration_mg_ml"] is None:
        final["max_concentration_mg_ml"] = rule_data["max_concentration_mg_ml"]

    return final