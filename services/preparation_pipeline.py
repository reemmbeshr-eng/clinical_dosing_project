from .ai_preparation_service import extract_reconstitution_ai
from .preparation_service import rule_based_reconstitution


def extract_reconstitution(preparation_text):
    """
    Deterministic-first preparation pipeline.
    """

    rule_data = rule_based_reconstitution(preparation_text)
    ai_data = extract_reconstitution_ai(preparation_text)

    discrepancy = False

    if ai_data and ai_data.get("reconstitution"):
        if ai_data["reconstitution"] != rule_data["reconstitution"]:
            discrepancy = True

    return {
        "reconstitution": rule_data["reconstitution"],
        "max_concentration_mg_ml": rule_data["max_concentration_mg_ml"],
        "dilution_only": rule_data["dilution_only"],
        "ai_discrepancy": discrepancy
    }