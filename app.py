import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the Model and Dictionaries
try:
    model = joblib.load('Antibiotic_Resistance_Model.pkl')
    organism_map = joblib.load('organism_map.pkl')
    antibiotic_map = joblib.load('antibiotic_map.pkl')
except FileNotFoundError:
    st.error("⚠️ Error: .pkl files not found. Make sure they are in the same folder!")
    st.stop()

# App Title & Design
st.set_page_config(page_title="Resistance Predictor", page_icon="🏥")
st.title("🏥 Antibiotic Resistance AI")
st.caption("Developed by Sujal Das | Supervisor: Ms. Shital Hajare")

st.markdown("---")

# Input Form
with st.form("prediction_form"):
    st.header("Patient Diagnosis")
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Patient Age", min_value=0, max_value=120, value=45)
        gender = st.selectbox("Gender", ["Male", "Female"])
    
    with col2:
        # Using the dictionaries to populate dropdowns
        organism_name = st.selectbox("Detected Organism", sorted(organism_map.keys()))
        antibiotic_name = st.selectbox("Antibiotic", sorted(antibiotic_map.keys()))
    
    submit = st.form_submit_button("Predict Resistance 🔍")

# Logic
# Logic when button is clicked
if submit:
    # Encoding Inputs
    try:
        gender_code = 1 if gender == "Female" else 0
        org_code = organism_map[organism_name]
        abx_code = antibiotic_map[antibiotic_name]
        
        # Prediction
        input_data = [[age, gender_code, org_code, abx_code]]
        prediction = model.predict(input_data)[0]
        
        # Get Probability (Confidence Score)
        probs = model.predict_proba(input_data)[0]
        confidence = np.max(probs) * 100  # Take the highest probability

        # --- IMPROVED DISPLAY SECTION ---
        st.divider()
        
        # Create two columns for a dashboard layout
        r_col1, r_col2 = st.columns([3, 1])

        with r_col1:
            if prediction == 1:
                st.error(f"⚠️ **RESISTANT DETECTED**")
                st.markdown(f"The AI predicts that **{antibiotic_name}** will **FAIL** against **{organism_name}**.")
            else:
                st.success(f"✅ **SUSCEPTIBLE (EFFECTIVE)**")
                st.markdown(f"The AI predicts that **{antibiotic_name}** will successfully treat **{organism_name}**.")

        with r_col2:
            # Show a big bold metric number
            st.metric(label="AI Confidence", value=f"{confidence:.1f}%")
            # Visual progress bar
            if confidence > 80:
                st.progress(confidence / 100, text="High Certainty")
            else:
                st.progress(confidence / 100, text="Moderate Certainty")

    except Exception as e:
        st.error(f"Error processing data: {e}")

st.markdown("---")
st.caption("Developed by Sujal Das")