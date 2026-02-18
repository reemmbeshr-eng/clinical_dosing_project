import subprocess
import json

def extract_reconstitution_from_text(preparation_text: str):
    """
    Uses Ollama to extract reconstitution data from unstructured preparation text.

    Returns:
    {
        "reconstitution": [
            {"vial_mg": int, "volume_ml": float},
            ...
        ],
        "max_concentration_mg_ml": float or None
    }
    """

    prompt = f"""
You are a clinical pharmacy assistant.

From the following drug preparation text, extract:

1) Reconstitution information as a list of:
   - vial strength in mg
   - volume to add in mL

2) Maximum allowed final concentration in mg/mL (if mentioned).

Return ONLY valid JSON in this exact format:
{{
  "reconstitution": [
    {{"vial_mg": 500, "volume_ml": 5}},
    {{"vial_mg": 1000, "volume_ml": 7.4}}
  ],
  "max_concentration_mg_ml": 30
}}

If any value is not mentioned, use null.

Preparation text:
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
        return data
    except Exception:
        return None
