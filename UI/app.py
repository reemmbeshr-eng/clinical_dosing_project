import sys
import os
import streamlit as st
from PIL import Image
import re
# Fix ROOT PATH
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Imports
from services.ai_input_service import extract_drug_and_indication_from_text
from services.reference_service import get_drug_reference
from services.logic_service import calculate_pediatric_dose_base, divide_daily_dose
from services.preparation_pipeline import extract_reconstitution
from services.safety_service import generate_safety_flags, format_safety_comment
from ML.inference import predict_drug_from_image


def select_renal_dose(renal_text: str, gfr: float):
    if not renal_text:
        return None

    lines = renal_text.split("\n")

    for line in lines:
        numbers = [float(n) for n in re.findall(r"\d+\.?\d*", line)]

        if len(numbers) >= 2:
            low, high = numbers[0], numbers[1]
            if low <= gfr <= high:
                return line

        elif len(numbers) == 1:
            if "less" in line.lower() or "<" in line:
                if gfr < numbers[0]:
                    return line
            if "greater" in line.lower() or ">" in line:
                if gfr > numbers[0]:
                    return line

    return None
# Page config
st.set_page_config(
    page_title="Pediatric Dosing Assistant",
    page_icon="💊",
    layout="wide"
)

st.title("💊 Pediatric Dosing Assistant")

# Session State
if "drug" not in st.session_state:
    st.session_state.drug = None
if "indication" not in st.session_state:
    st.session_state.indication = None
if "dose_result" not in st.session_state:
    st.session_state.dose_result = None

# Tabs
tab1, tab2, tab3 = st.tabs([
    "🧾 Clinical Input",
    "🧮 Dose & Preparation",
    "⚠️ Safety"
])

# TAB 1 — Clinical Input
with tab1:
    st.subheader("Clinical Input")

    mode = st.radio("Input type", ["Text", "Image"])

    if mode == "Text":
        query = st.text_area("Enter clinical query")
        if st.button("Analyze text"):
            drug, indication = extract_drug_and_indication_from_text(query)
            st.session_state.drug = drug
            st.session_state.indication = indication

    if mode == "Image":
        uploaded = st.file_uploader("Upload drug image", type=["jpg", "png", "jpeg"])
        if uploaded:
            image = Image.open(uploaded)
            st.image(image, width=250)

            drug_pred, conf = predict_drug_from_image(image)
            st.info(f"AI detected: {drug_pred} ({conf:.0%})")

            st.session_state.drug = st.text_input(
                "Confirm or edit drug name",
                value=drug_pred
            )

    if st.session_state.drug:
        st.text_input(
            "Selected drug",
            value=st.session_state.drug,
            disabled=True
        )
        st.session_state.indication = st.text_input(
            "Indication",
            value=st.session_state.indication or ""
        )

# TAB 2 — Dose & Preparation
with tab2:
    if not st.session_state.drug:
        st.info("Please provide clinical input first")
        st.stop()

    ref = get_drug_reference(
        st.session_state.drug,
        st.session_state.indication
    )

    if not ref:
        st.error("No reference found")
        st.stop()

    st.subheader("📘 Drug Reference")

    # RENAL DECISION (BEFORE DOSE)
    renal_status = st.radio(
        "Renal status",
        ["Normal renal function", "Renal impairment"]
    )

    # Decide dosage reference
    if renal_status == "Normal renal function":
        dosage_text = ref["dosage"]
        st.success("Using standard dosing")

    else:
        gfr = st.number_input(
            "Enter GFR (mL/min/1.73 m²)",
            min_value=0.0,
            value=30.0
        )

        renal_text = ref.get("renal_adjustment_dose")

        st.subheader("DEBUG renal text from DB")
        st.code(renal_text)

        if not renal_text:
            st.error("No renal dosing data available")
            st.stop()

        selected_dose = select_renal_dose(renal_text, gfr)

        if not selected_dose:
            st.error("No renal dosing matches this GFR value")
            st.stop()

        dosage_text = selected_dose
        st.warning(f"Using renal-adjusted dosing for GFR = {gfr}")

    # Patient Data
    st.subheader("👶 Patient Data")

    weight = st.number_input("Weight (kg)", min_value=0.1, value=10.0)

    height = None
    if "mg/m2" in dosage_text.lower():
        height = st.number_input("Height (cm)", min_value=30.0, value=80.0)

    interval = None
    if "day" in dosage_text.lower():
        interval = st.number_input("Interval (hours)", min_value=1.0, value=8.0)

    # STEP 1 — Calculate Dose
    if st.button("🧮 Calculate Dose"):
        base = calculate_pediatric_dose_base(
            dosage_text,
            weight,
            height
        )

        if base["dose_type"] == "MG_KG_DOSE":
            low, high = base["per_dose_mg"]
        else:
            dl, dh = base["daily_dose_mg"]
            _, low, high = divide_daily_dose(dl, dh, interval)

        st.success(
            f"💊 Dose per administration: {low:.1f} – {high:.1f} mg"
        )

        st.session_state.dose_result = {
            "low": low,
            "high": high
        }

    # STEP 2 — Preparation
    if st.session_state.dose_result:
        prep = extract_reconstitution(ref["preparation"])

        st.subheader("💉 Vial Information")

        vial_mg = st.number_input(
            "Vial strength you have (mg)",
            min_value=1.0,
            step=50.0
        )

        if st.button("🧪 Calculate Preparation"):
            max_conc = prep.get("max_concentration_mg_ml", 30)
            required_dose = st.session_state.dose_result["high"]

            reconstitution_volume = vial_mg / max_conc
            withdraw_volume = required_dose / max_conc
            vials_needed = required_dose / vial_mg

            st.success("✅ Preparation Instructions")

            st.markdown(
                f"""
                • Reconstitute **{vial_mg:.0f} mg vial** with  
                  **{reconstitution_volume:.1f} mL**

                • Withdraw **{withdraw_volume:.1f} mL**  
                  to give **{required_dose:.0f} mg**
                """
            )

            if vials_needed > 1:
                st.info(f"≈ {vials_needed:.1f} vials required")

# TAB 3 — Safety
with tab3:
    if not st.session_state.dose_result:
        st.info("Please calculate dose first")
        st.stop()

    flags = generate_safety_flags(
        dose_per_administration_mg=st.session_state.dose_result["high"]
    )

    for f in format_safety_comment(flags):
        st.warning(f)