import subprocess
import json

def extract_reconstitution_ai(preparation_text: str):
    """
    AI-first extraction.
    Always returns a fixed structure.
    """

    prompt = f"""
You are a clinical pharmacy assistant.

Extract reconstitution data from the text below.

Return ONLY valid JSON:
{{
  "reconstitution": [
    {{"vial_mg": 500, "volume_ml": 5}}
  ],
  "max_concentration_mg_ml": 30
}}

If something is not mentioned, use null or empty list.

Text:
{preparation_text}
"""

    result = subprocess.run(
        ["ollama", "run", "mistral"],
        input=prompt.encode("utf-8"),
        capture_output=True
    )

    output = result.stdout.decode("utf-8", errors="ignore").strip()

    try:
        data = json.loads(output)
    except Exception:
        data = {}

    return {
        "reconstitution": data.get("reconstitution") or [],
        "max_concentration_mg_ml": data.get("max_concentration_mg_ml")
    }