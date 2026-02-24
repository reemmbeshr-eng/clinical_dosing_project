from .reference_service import get_drug_reference
from .logic_service import (
    calculate_pediatric_dose_base,
    divide_daily_dose
)
from .ai_input_service import extract_drug_and_indication_from_text
from .preparation_pipeline import extract_reconstitution
from .safety_service import (
    generate_safety_flags,
    format_safety_comment
)


# Shared variables

pl = ph = None          # dose per administration (mg)
dl = dh = None          # daily dose (mg/day)
interval_hours = None
renal_active = False


def section(title):
    print("\n" + "=" * 50)
    print(title.upper())
    print("=" * 50)



# App start

print("\nPEDIATRIC DRUG DOSING ASSISTANT")
print("-" * 50)


# Clinical query

print("\nEnter clinical query:")
query = input("> ")

drug, indication = extract_drug_and_indication_from_text(query)

if not drug or not indication:
    print("❌ Could not understand the clinical query.")
    exit()

print(f"\nDetected drug: {drug}")
print(f"Detected indication: {indication}")


# Get reference

ref = get_drug_reference(drug, indication)

if not ref:
    print("❌ No reference found.")
    exit()

section("Dosage Reference")
print(ref["dosage"])


# Patient parameters

weight = float(input("\nEnter patient weight (kg): "))

height = None
if "mg/m2" in ref["dosage"].lower() or "mg/m²" in ref["dosage"].lower():
    height = float(input("Enter patient height (cm): "))


# Renal decision

renal_active = False
gfr_value = None

renal_choice = input(
    "\nIs renal impairment present? (y/n): "
).strip().lower()

if renal_choice == "y":
    renal_active = True
    gfr_value = float(
        input("If yes, enter patient GFR (mL/min/1.73 m²): ")
    )




# Dose calculation

section("Dose Calculation")

# ---------- RENAL DOSING PATH ----------
if renal_active:
    print("\nReference dosing used (renal):")
    print(ref["renal_adjustment_dose"])

    import re

    text = ref["renal_adjustment_dose"].lower()

    # استخراج GFR range
    gfr_match = re.search(r"gfr\s*(\d+)\s*to\s*(\d+)", text)
    if not gfr_match:
        print("❌ Could not extract GFR range from renal reference.")
        exit()

    gfr_low = float(gfr_match.group(1))
    gfr_high = float(gfr_match.group(2))

    # check GFR applicability
    if not (gfr_low <= gfr_value <= gfr_high):
        print(f"❌ Renal reference applies to GFR {gfr_low}–{gfr_high}.")
        print(f"Patient GFR = {gfr_value}. No matching renal dosing available.")
        exit()

    dose_match = re.search(
    r"(\d+)\s*to\s*(\d+)\s*mg/kg/dose",
    text
)

    if not dose_match:
        print("❌ Could not extract renal dose (mg/kg/dose).")
        exit()

    dose_low = float(dose_match.group(1))
    dose_high = float(dose_match.group(2))


    

    interval_match = re.search(r"every\s+(\d+)", text)
    interval_hours = int(interval_match.group(1)) if interval_match else None

    pl = weight * dose_low
    ph = weight * dose_high

    print(f"Dose per administration: {pl:.2f} – {ph:.2f} mg")
    if interval_hours:
        print(f"Dosing interval: every {interval_hours} hours")


# ---------- STANDARD PATH ----------
else:
    print("\nReference dosing used:")
    print(ref["dosage"])

    base_result = calculate_pediatric_dose_base(
        dosage_text=ref["dosage"],
        weight=weight,
        height=height
    )

    if not base_result:
        print("❌ Dose calculation failed.")
        exit()

    if base_result["dose_type"] == "MG_KG_DOSE":
        pl, ph = base_result["per_dose_mg"]
        print(f"Dose per administration: {pl:.2f} – {ph:.2f} mg")

    else:
        dl, dh = base_result["daily_dose_mg"]
        print(f"Total daily dose: {dl:.2f} – {dh:.2f} mg/day")

        interval_hours = float(input("Enter dosing interval (hours): "))
        doses_per_day, pl, ph = divide_daily_dose(dl, dh, interval_hours)

        print(f"Doses per day: {int(doses_per_day)}")
        print(f"Dose per administration: {pl:.2f} – {ph:.2f} mg")


# Administration

section("Administration")
print(ref["administration"])


# Reconstitution / Preparation

section("Reconstitution")

prep_data = extract_reconstitution(ref["preparation"])

if prep_data and prep_data.get("reconstitution"):
    recon_list = prep_data["reconstitution"]
    max_conc = prep_data.get("max_concentration_mg_ml")

    print("Available reconstitution options from reference:")
    for item in recon_list:
        print(f"- {item['vial_mg']} mg vial → add {item['volume_ml']} mL")

    try:
        vial_strength = int(input("Select vial strength (mg): "))
    except ValueError:
        vial_strength = None

    selected = next(
        (x for x in recon_list if x["vial_mg"] == vial_strength),
        None
    )

    if selected:
        concentration = selected["vial_mg"] / selected["volume_ml"]
        print(f"\nResulting concentration: {concentration:.2f} mg/mL")

        volume_low = pl / concentration
        volume_high = ph / concentration

        print(f"Volume to withdraw per dose: {volume_low:.2f} – {volume_high:.2f} mL")

        if max_conc and concentration > max_conc:
            print("\n⚠ WARNING:")
            print(f"Concentration exceeds recommended maximum ({max_conc} mg/mL).")
            print("Further dilution is required before administration.")
    else:
        print("⚠ Selected vial strength not found in reference data.")

else:
    print("⚠ AI could not extract reconstitution data from reference.")
    print("Please refer to preparation instructions manually.")


# Safety check

print("\n--- SAFETY COMMENT ---")

flags = generate_safety_flags(
    daily_dose_mg=dh if not renal_active else None,
    max_daily_dose_mg=None,
    dose_per_administration_mg=ph,
    withdrawal_volume_ml=None
)



comments = format_safety_comment(flags)
for c in comments:
    print(c)
