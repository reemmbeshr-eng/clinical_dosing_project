from reference_service import get_drug_reference
from logic_service import calculate_pediatric_dose
from ai_service import explain_dose_with_ollama

print("\n--- Pediatric Drug Dosing System ---\n")

# -------------------------
# User input
# -------------------------
drug = input("Enter drug name: ")
indication = input("Enter indication: ")

# -------------------------
# Get reference from DB
# -------------------------
ref = get_drug_reference(drug, indication)

if not ref:
    print("No reference found.")
    exit()

print("\n--- REFERENCE ---")
print("Dosage:", ref["dosage"])
print("Administration:", ref["administration"])
print("Preparation:", ref["preparation"])

# -------------------------
# Required patient parameters
# -------------------------
weight = float(input("\nEnter patient weight (kg): "))

height = None
if "mg/m2" in ref["dosage"].lower() or "mg/m²" in ref["dosage"].lower():
    height = float(input("Enter patient height (cm): "))

# -------------------------
# Dose calculation
# -------------------------
dose_result = calculate_pediatric_dose(
    dosage_text=ref["dosage"],
    weight=weight,
    height=height
)

if not dose_result:
    print("Dose calculation failed.")
    exit()

dl, dh = dose_result["daily_dose_mg"]
pl, ph = dose_result["dose_per_administration_mg"]

print("\n--- CALCULATED DOSE ---")
print(f"Total daily dose: {dl:.2f} – {dh:.2f} mg/day")
print(f"Doses per day: {dose_result['doses_per_day']}")
print(f"Dose per administration: {pl:.2f} – {ph:.2f} mg")

# -------------------------
# AI explanation (Ollama)
# -------------------------
print("\n--- AI EXPLANATION ---")

explanation = explain_dose_with_ollama(
    drug=drug,
    indication=indication,
    dose_type="Pediatric dosing with divisions and maximum limit",
    calculated_low=pl,
    calculated_high=ph,
    weight=weight,
    height=height
)

print(explanation)

# -------------------------
# Renal adjustment note
# -------------------------
if ref["renal_adjustment"] and ref["renal_adjustment"].lower() != "no":
    print("\n⚠ Renal adjustment note:")
    print(ref["renal_adjustment_dose"])