import subprocess
import json


def extract_reconstitution_ai(preparation_text: str):
    """
    AI extraction layer (secondary validation).
    Returns structured data safely.
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

    try:
        result = subprocess.run(
            ["ollama", "run", "mistral"],
            input=prompt.encode("utf-8"),
            capture_output=True,
            timeout=20
        )

        output = result.stdout.decode("utf-8", errors="ignore").strip()

        json_start = output.find("{")
        json_end = output.rfind("}") + 1
        clean_json = output[json_start:json_end]

        data = json.loads(clean_json)

    except Exception:
        data = {}

    return {
        "reconstitution": data.get("reconstitution") or [],
        "max_concentration_mg_ml": data.get("max_concentration_mg_ml")
    }