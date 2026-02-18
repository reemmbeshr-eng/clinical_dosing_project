import subprocess
import json

def extract_drug_and_indication_from_text(query: str):
    """
    Uses Ollama to extract drug name and indication from a free-text clinical query.
    Returns (drug, indication)
    """

    prompt = f"""
You are a medical NLP assistant.

From the following clinical sentence, extract:
1) Drug name
2) Indication (disease/condition)

Return ONLY valid JSON in this format:
{{ "drug": "...", "indication": "..." }}

Sentence:
{query}
"""

    result = subprocess.run(
        ["ollama", "run", "mistral"],
        input=prompt.encode("utf-8"),
        capture_output=True
    )

    output = result.stdout.decode("utf-8", errors="ignore").strip()

    try:
        data = json.loads(output)
        return data.get("drug"), data.get("indication")
    except Exception:
        return None, None
