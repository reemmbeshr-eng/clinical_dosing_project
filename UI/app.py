import sys
import os
import streamlit as st
from PIL import Image
import re
import tempfile

# ======================================================
# Fix ROOT PATH
# ======================================================
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# ======================================================
# Imports
# ======================================================
from services.ai_input_service import extract_drug_and_indication_from_text
from services.reference_service import get_drug_reference
from services.logic_service import calculate_pediatric_dose_base, divide_daily_dose
from services.preparation_pipeline import extract_reconstitution
from services.safety_service import generate_safety_flags, format_safety_comment
from services.ai_service import explain_dose_with_ollama
from ML.inference import predict_drug_from_image
from services.ai_vision_service import extract_text_from_image, extract_vial_strength_mg

# ======================================================
# Renal Selector
# ======================================================
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


# ======================================================
# Page config
# ======================================================
st.set_page_config(
    page_title="Pediatric Dosing Assistant",
    page_icon="💊",
    layout="wide"
)

st.title("💊 Pediatric Dosing Assistant")

# ======================================================
# Session State
# ======================================================
if "drug" not in st.session_state:
    st.session_state.drug = None
if "indication" not in st.session_state:
    st.session_state.indication = None
if "dose_result" not in st.session_state:
    st.session_state.dose_result = None
if "dose_type" not in st.session_state:
    st.session_state.dose_type = None
if "explanation" not in st.session_state:
    st.session_state.explanation = None

# ======================================================
# Tabs
# ======================================================
tab1, tab2, tab3 = st.tabs([
    "🧾 Clinical Input",
    "🧮 Dose & Preparation",
    "⚠️ Safety & AI Explanation"
])

# ======================================================
# TAB 1 — Clinical Input
# ======================================================
with tab1:
    st.subheader("Clinical Input")

    mode = st.radio("Input type", ["Text", "Image"])

    # -------------------------
    # TEXT MODE
    # -------------------------
    if mode == "Text":
        query = st.text_area("Enter clinical query")

        if st.button("Analyze text"):
            drug, indication = extract_drug_and_indication_from_text(query)
            
            st.session_state.drug = drug
            st.session_state.indication = indication

            if drug and indication:
                st.success(f"Detected drug: {drug}")
                st.info(f"Indication: {indication}")
            else:
                st.error("Could not extract clinical entities.")

    # -------------------------
    # IMAGE MODE
    # -------------------------
    if mode == "Image":
        uploaded = st.file_uploader("Upload drug image", type=["jpg", "png", "jpeg"])

        if uploaded:
            image = Image.open(uploaded)
            st.image(image, width=250)

            # =====================================
            # 1️⃣ CNN Drug Classification
            # =====================================
            drug_pred, conf = predict_drug_from_image(image)

            st.subheader("🧠 CNN Drug Detection")
            st.success(f"Detected Drug: {drug_pred}")
            st.caption(f"Confidence: {conf:.0%}")

            # نخزن اسم الدواء
            st.session_state.drug = drug_pred

            # =====================================
            # 2️⃣ OCR Strength Detection
            # =====================================
            import tempfile

            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                image.save(tmp_file.name)
                temp_path = tmp_file.name

            ocr_text = extract_text_from_image(temp_path)
            vial_strength = extract_vial_strength_mg(ocr_text)

            st.subheader("🔎 OCR Strength Detection")
            st.text_area("Raw OCR Text", ocr_text, height=100)

            if vial_strength:
                st.success(f"Detected Vial Strength: {vial_strength} mg")
                st.session_state.vial_strength_detected = vial_strength
            else:
                st.warning("Could not detect vial strength from image")
        # -------------------------
        # Selected Drug Display
        # -------------------------
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
# ======================================================
# TAB 2 — Dose & Preparation
# ======================================================
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

    renal_status = st.radio(
        "Renal status",
        ["Normal renal function", "Renal impairment"]
    )

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

        if not renal_text:
            st.error("No renal dosing data available")
            st.stop()

        selected_dose = select_renal_dose(renal_text, gfr)

        if not selected_dose:
            st.error("No renal dosing matches this GFR value")
            st.stop()

        dosage_text = selected_dose
        st.warning(f"Using renal-adjusted dosing for GFR = {gfr}")

    # ==============================
    # Patient Data
    # ==============================
    st.subheader("👶 Patient Data")

    weight = st.number_input("Weight (kg)", min_value=0.1, value=10.0)

    height = None
    if "mg/m2" in dosage_text.lower():
        height = st.number_input("Height (cm)", min_value=30.0, value=80.0)

    interval = None
    if "day" in dosage_text.lower():
        interval = st.number_input("Interval (hours)", min_value=1.0, value=8.0)

    # ==============================
    # STEP 1 — Calculate Dose
    # ==============================
    if st.button("🧮 Calculate Dose"):

        base = calculate_pediatric_dose_base(
            dosage_text,
            weight,
            height
        )

        st.session_state.dose_type = base["dose_type"]

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

        st.session_state.explanation = None  # reset old explanation

    
    # ==============================
    # STEP 2 — Preparation
    # ==============================
    if st.session_state.dose_result:

        st.subheader("💉 Preparation")

        prep = extract_reconstitution(ref["preparation"])

        

        # -----------------------------
        # Case 1: Reconstitution available
        # -----------------------------
        if prep.get("reconstitution"):

            recon_list = prep["reconstitution"]
            max_conc = prep.get("max_concentration_mg_ml")

            st.markdown("### Available reconstitution options:")

            for item in recon_list:
                st.write(f"- {item['vial_mg']} mg vial → add {item['volume_ml']} mL")

            vial_strength = st.selectbox(
                "Select vial strength (mg)",
                [item["vial_mg"] for item in recon_list]
            )

            selected = next(
                (x for x in recon_list if x["vial_mg"] == vial_strength),
                None
            )

            if selected:

                concentration = selected["vial_mg"] / selected["volume_ml"]

                st.success(f"Resulting concentration: {concentration:.2f} mg/mL")

                low_dose = st.session_state.dose_result["low"]
                high_dose = st.session_state.dose_result["high"]

                volume_low = low_dose / concentration
                volume_high = high_dose / concentration

                st.markdown("### 💊 Volume to withdraw per dose:")
                st.write(f"{volume_low:.2f} – {volume_high:.2f} mL")

                if max_conc and concentration > max_conc:
                    st.error(
                        f"⚠ Concentration exceeds recommended maximum ({max_conc} mg/mL)."
                    )

                    # نحسب dilution المطلوب
                    required_total_volume_low = low_dose / max_conc
                    required_total_volume_high = high_dose / max_conc

                    extra_diluent_low = required_total_volume_low - volume_low
                    extra_diluent_high = required_total_volume_high - volume_high

                    st.markdown("### 💧 Additional Dilution Required:")

                    st.write(
                        f"To reach ≤ {max_conc} mg/mL:"
                    )

                    st.write(
                        f"• Add additional diluent: "
                        f"{extra_diluent_low:.2f} – {extra_diluent_high:.2f} mL"
                    )

                    st.write(
                        f"• Final total volume per dose: "
                        f"{required_total_volume_low:.2f} – {required_total_volume_high:.2f} mL"
                    )

        # -----------------------------
        # Case 2: Dilution only
        # -----------------------------
        elif prep.get("dilution_only"):
            st.info("No reconstitution required. Dilution instructions only.")
            if prep.get("max_concentration_mg_ml"):
                st.write(
                    f"Maximum final concentration: "
                    f"{prep['max_concentration_mg_ml']} mg/mL"
                )

        # -----------------------------
        # Case 3: Nothing structured found
        # -----------------------------
        else:
            st.warning("No structured preparation data detected.")

# ======================================================
# TAB 3 — Safety & AI Explanation
# ======================================================
with tab3:

    if not st.session_state.dose_result:
        st.info("Please calculate dose first")
        st.stop()

    st.subheader("⚠️ Safety Evaluation")

    high = st.session_state.dose_result["high"]

    flags = generate_safety_flags(
        dose_per_administration_mg=high
    )

    formatted_flags = format_safety_comment(flags)

    if formatted_flags:
        for f in formatted_flags:
            st.warning(f)
    else:
        st.success("No safety flags detected")

    # ---------------------------------------------
    # AI Dose Explanation
    # ---------------------------------------------
    st.divider()
    st.subheader("🧠 AI Dose Explanation")

    if st.button("Explain Dose Calculation with AI"):

        with st.spinner("Generating clinical explanation..."):

            st.session_state.explanation = explain_dose_with_ollama(
                drug=st.session_state.drug,
                indication=st.session_state.indication,
                dose_type=st.session_state.dose_type,
                calculated_low=st.session_state.dose_result["low"],
                calculated_high=st.session_state.dose_result["high"]
            )

    if st.session_state.explanation:
        st.info(st.session_state.explanation)