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
from ML.drug_classification.inference import predict_drug_from_image
from services.ai_vision_service import extract_text_from_image, extract_vial_strength_mg
from services.logic_service import select_renal_dose

from ML.CXR.model import PneumoniaCNN
import torch
from torchvision import transforms


# ======================================================
# Load Pneumonia Model
# ======================================================

pneumonia_model = PneumoniaCNN()

pneumonia_model.load_state_dict(
    torch.load("pneumonia_cnn.pth", map_location="cpu")
)

pneumonia_model.eval()

cxr_transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])


# ======================================================
# Page config
# ======================================================
st.set_page_config(
    page_title="AI Clinical Decision Support",
    page_icon="🩺",
    layout="wide"
)

# ======================================================
# Custom UI Styling
# ======================================================

st.markdown("""
<style>

.main-title {
    font-size:38px;
    font-weight:700;
    color:#0f6ab4;
}

.subtitle {
    font-size:18px;
    color:#555;
}

.card {
    background-color:#f8fbff;
    padding:20px;
    border-radius:12px;
    border:1px solid #e6eef8;
    margin-bottom:20px;
}

.metric-card {
    background-color:#f0f7ff;
    padding:20px;
    border-radius:10px;
    border-left:6px solid #0f6ab4;
}

.section-title {
    font-size:22px;
    font-weight:600;
    margin-bottom:10px;
}

</style>
""", unsafe_allow_html=True)


# ======================================================
# Header
# ======================================================

st.markdown('<div class="main-title">🩺 AI Clinical Decision Support System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Pneumonia detection from Chest X-ray with pediatric dosing assistant</div>', unsafe_allow_html=True)

st.divider()

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

tab0, tab1, tab2, tab3 = st.tabs([
    "🩻 Chest X-ray Screening",
    "🧾 Clinical Input",
    "🧮 Dose & Preparation",
    "⚠️ Safety & AI Explanation"
])


# ======================================================
# TAB 0 — Chest X-ray Screening
# ======================================================

with tab0:

    st.markdown('<div class="section-title">🩻 Chest X-ray Pneumonia Screening</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)

    cxr_file = st.file_uploader(
        "Upload Chest X-ray image",
        type=["jpg","png","jpeg"]
    )

    if cxr_file:

        image = Image.open(cxr_file).convert("RGB")

        st.image(image, width=300)

        img = cxr_transform(image)
        img = img.unsqueeze(0)

        with torch.no_grad():

            output = pneumonia_model(img)

            probs = torch.softmax(output, dim=1)

            pred = torch.argmax(output,1).item()

            confidence = probs[0][pred].item()

        classes = ["Normal","Pneumonia"]

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Prediction", classes[pred])

        with col2:
            st.metric("Confidence", f"{confidence:.0%}")

        if pred == 0:

            st.success(
                "Normal chest X-ray. No signs of pneumonia.\n\n"
                "Antibiotics are not recommended."
            )

            st.stop()

        else:

            st.warning(
                "Possible pneumonia detected.\n\n"
                "Proceed to clinical dosing assistant."
            )

    st.markdown('</div>', unsafe_allow_html=True)


# ======================================================
# TAB 1 — Clinical Input
# ======================================================

with tab1:

    st.markdown('<div class="section-title">🧾 Clinical Input</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)

    mode = st.radio("Input type", ["Text", "Image"])

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

    if mode == "Image":

        uploaded = st.file_uploader("Upload drug image", type=["jpg", "png", "jpeg"])

        if uploaded:

            image = Image.open(uploaded)

            st.image(image, width=250)

            drug_pred, conf = predict_drug_from_image(image)

            st.subheader("🧠 CNN Drug Detection")

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Detected Drug", drug_pred)

            with col2:
                st.metric("Model Confidence", f"{conf:.0%}")

            st.session_state.drug = drug_pred

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

    st.markdown('</div>', unsafe_allow_html=True)


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

    st.markdown('<div class="section-title">📘 Drug Reference</div>', unsafe_allow_html=True)

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

    st.markdown('<div class="section-title">👶 Patient Information</div>', unsafe_allow_html=True)

    weight = st.number_input("Weight (kg)", min_value=0.1, value=10.0)

    height = None

    if "mg/m2" in dosage_text.lower():
        height = st.number_input("Height (cm)", min_value=30.0, value=80.0)

    interval = None

    if "day" in dosage_text.lower():
        interval = st.number_input("Interval (hours)", min_value=1.0, value=8.0)


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

        st.markdown('<div class="metric-card">', unsafe_allow_html=True)

        st.metric(
            label="Dose per Administration",
            value=f"{low:.1f} – {high:.1f} mg"
        )

        st.markdown('</div>', unsafe_allow_html=True)

        st.session_state.dose_result = {
            "low": low,
            "high": high
        }

        st.session_state.explanation = None


    if st.session_state.dose_result:

        st.markdown('<div class="section-title">💉 Drug Preparation</div>', unsafe_allow_html=True)

        prep = extract_reconstitution(ref["preparation"])

        if prep.get("reconstitution"):

            recon_list = prep["reconstitution"]

            max_conc = prep.get("max_concentration_mg_ml")

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

                st.write(f"Volume to withdraw: {volume_low:.2f} – {volume_high:.2f} mL")
                if max_conc and concentration > max_conc:

                    st.error(
                        f"⚠ Concentration exceeds recommended maximum ({max_conc} mg/mL)."
                    )

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

        elif prep.get("dilution_only"):

            st.info("No reconstitution required. Dilution instructions only.")

        else:

            st.warning("No structured preparation data detected.")


# ======================================================
# TAB 3 — Safety & AI Explanation
# ======================================================

with tab3:

    if not st.session_state.dose_result:
        st.info("Please calculate dose first")
        st.stop()

    st.markdown('<div class="section-title">⚠️ Safety Evaluation</div>', unsafe_allow_html=True)

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

    st.divider()

    st.markdown('<div class="section-title">🧠 AI Clinical Explanation</div>', unsafe_allow_html=True)

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