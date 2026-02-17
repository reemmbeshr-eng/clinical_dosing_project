import subprocess

"""
AI service module using a local LLM (Ollama)
to provide explainable pediatric dose calculations.
"""


def explain_dose_with_ollama(
    drug,
    indication,
    dose_type,
    calculated_low,
    calculated_high,
    weight=None,
    height=None
):
    """
    Uses a local language model to explain
    pediatric dose calculations in clinical language.
    """

    prompt = f"""
You are a clinical assistant explaining pediatric drug dosing.

Drug: {drug}
Indication: {indication}
Dose type: {dose_type}
Calculated dose range: {calculated_low:.2f} to {calculated_high:.2f} mg
"""

    if weight:
        prompt += f"Patient weight: {weight} kg\n"
    if height:
        prompt += f"Patient height: {height} cm\n"

    prompt += """
Explain clearly how this dose was calculated.
Do not recommend changes.
Do not prescribe.
Only explain the calculation in simple clinical language.
"""

    result = subprocess.run(
            ["ollama", "run", "mistral"],
            input=prompt,
            text=True,
            capture_output=True,
            check=False
        )


    return result.stdout.strip()
