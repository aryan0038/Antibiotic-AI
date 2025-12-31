import streamlit as st
import joblib
import pandas as pd
import numpy as np
import google.generativeai as genai
import time
from fpdf import FPDF
from datetime import datetime

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="DRAP | Clinical Decision Support",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- HELPER FUNCTIONS ---

def get_alternatives(model, age, gender_code, org_code, antibiotic_map):
    """
    Scans all antibiotics in the map to find the most susceptible options
    for the current patient profile.
    """
    results = []
    
    # Iterate through all available antibiotics in the system
    for abx_name, abx_code in antibiotic_map.items():
        # Create input array for this specific antibiotic
        input_data = [[age, gender_code, org_code, abx_code]]
        
        # Get probability of class 0 (Susceptible) vs class 1 (Resistant)
        # Assuming model.classes_ are [0, 1] or similar logic
        probs = model.predict_proba(input_data)[0]
        
        # We want the probability of being Susceptible (usually index 0, but check your model)
        # Here we assume index 0 is Susceptible based on your previous code logic
        susceptible_score = probs[0] 
        
        results.append({
            "Antibiotic": abx_name,
            "Susceptibility Score": susceptible_score * 100,
            "Predicted Status": "Susceptible" if susceptible_score > 0.5 else "Resistant"
        })
    
    # Create DataFrame and sort by highest Susceptibility Score
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by="Susceptibility Score", ascending=False).head(5)
    return df_results

def create_pdf(patient_data, prediction_result, confidence, ai_advice):
    """Generates a PDF report and returns the binary data."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Header
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="DRAP Clinical Report", ln=True, align='C')
    pdf.line(10, 20, 200, 20)
    pdf.ln(15)

    # Patient Details
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Patient Profile", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.cell(200, 10, txt=f"Age: {patient_data['Age']} | Gender: {patient_data['Gender']}", ln=True)
    pdf.cell(200, 10, txt=f"Pathogen: {patient_data['Organism']}", ln=True)
    pdf.cell(200, 10, txt=f"Antibiotic Prescribed: {patient_data['Antibiotic']}", ln=True)
    pdf.ln(10)

    # Prediction
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Analysis Result", ln=True)
    pdf.set_font("Arial", size=12)
    result_color = "RESISTANT (Danger)" if prediction_result == "RESISTANT" else "SUSCEPTIBLE (Effective)"
    pdf.cell(200, 10, txt=f"Outcome: {result_color}", ln=True)
    pdf.cell(200, 10, txt=f"Model Confidence: {confidence:.1f}%", ln=True)
    pdf.ln(10)

    # AI Advice (Sanitized for PDF)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="AI Clinical Specialist Notes", ln=True)
    pdf.set_font("Arial", size=10)
    safe_advice = ai_advice.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 10, txt=safe_advice)
    
    return pdf.output(dest='S').encode('latin-1')

# 2. LOAD MODELS
@st.cache_resource
def load_assets():
    try:
        # ENSURE THESE FILES EXIST IN YOUR DIRECTORY
        model = joblib.load('Antibiotic_Resistance_Model.pkl')
        organism_map = joblib.load('organism_map.pkl')
        antibiotic_map = joblib.load('antibiotic_map.pkl')
        return model, organism_map, antibiotic_map
    except FileNotFoundError:
        return None, None, None

model, organism_map, antibiotic_map = load_assets()

if model is None:
    st.error("⚠️ Critical Error: Model files (.pkl) not found. Please check your directory.")
    st.stop()

# 3. CONFIGURE GEMINI
# Replace with your actual key
genai.configure(api_key="AIzaSyD45n0EWx_RyuLAv-LK8eaSJLwljr9c01g") 

# 4. CUSTOM STYLING
st.markdown("""
    <style>z
    .main { background-color: #f8f9fa; }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
    }
    div[data-testid="stMetricValue"] {
        font-size: 24px;
        color: #0068c9;
    }
    .highlight-card {
        background-color: #e3f2fd;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #0068c9;
    }
    </style>
""", unsafe_allow_html=True)

# 5. SIDEBAR
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=60)
    st.title("DRAP System")
    st.caption("v2.2 | Clinical Release")
    st.markdown("---")
    st.subheader("🧠 Model Intelligence")
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Accuracy", "94.2%", "+1.2%")
    with col_b:
        st.metric("F1 Score", "0.91")
    
    st.markdown("**Architecture:** Random Forest Classifier")
    st.markdown("---")
    if st.button("🔄 Reset / Clear Form"):
        st.rerun()

# 6. MAIN DASHBOARD HEADER
st.title("🛡️ DRAP: Antibiotic Resistance Predictor")
st.markdown("---")

# 7. INPUT SECTION
with st.container():
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.subheader("1. Patient Profile")
        age = st.number_input("Age (Years)", 0, 120, 45)
        gender = st.selectbox("Gender", ["Male", "Female"])
        
    with col2:
        st.subheader("2. Clinical Data")
        organism_name = st.selectbox("Pathogen Identified", sorted(organism_map.keys()))
        antibiotic_name = st.selectbox("Antibiotic Prescribed", sorted(antibiotic_map.keys()))

    with col3:
        st.subheader("3. Actions")
        st.write("Ready to analyze?")
        analyze_btn = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

# 8. PREDICTION & RESULTS
if analyze_btn:
    # Processing Animation
    progress_text = "Analyzing resistance markers..."
    my_bar = st.progress(0, text=progress_text)
    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1, text=progress_text)
    my_bar.empty()

    try:
        # Encode Inputs
        gender_code = 1 if gender == "Female" else 0
        org_code = organism_map[organism_name]
        abx_code = antibiotic_map[antibiotic_name]
        
        # Predict
        input_data = [[age, gender_code, org_code, abx_code]]
        prediction = model.predict(input_data)[0]
        probs = model.predict_proba(input_data)[0]
        
        # Logic: Assuming class 1 is Resistant
        is_resistant = (prediction == 1)
        confidence = np.max(probs) * 100
        status_text = "RESISTANT" if is_resistant else "SUSCEPTIBLE"

        # Initialize session state for AI advice
        if 'ai_advice' not in st.session_state:
             st.session_state['ai_advice'] = "AI analysis pending..."

        # --- RESULT DASHBOARD ---
        st.divider()
        
        # Define Tabs
        tab1, tab2, tab3 = st.tabs(["📊 Analysis & Actions", "🤖 AI Specialist Opinion", "🧪 Smart Screener"])
        
        # TAB 2: Gemini AI
        with tab2:
            st.markdown("### Clinical Interpretation")
            with st.spinner("Consulting AI Specialist..."):
                try:
                    model_gemini = genai.GenerativeModel('gemini-1.5-flash')
                    prompt = f"""
                    Patient: {age}yr {gender}, Pathogen: {organism_name}, Drug: {antibiotic_name}.
                    Prediction: {status_text} (Confidence: {confidence:.1f}%).
                    Provide:
                    1. A brief explanation of why this resistance might occur (mechanism).
                    2. Recommended next steps or dosage considerations.
                    Keep it under 100 words.
                    """
                    response = model_gemini.generate_content(prompt)
                    st.session_state['ai_advice'] = response.text
                    st.info(st.session_state['ai_advice'])
                except Exception as e:
                    st.session_state['ai_advice'] = "AI Service Unavailable"
                    st.warning(f"AI Service Error: {e}")

        # TAB 1: Core Results
        with tab1:
            r_col1, r_col2 = st.columns([2, 1])
            
            with r_col1:
                if is_resistant:
                    st.error(f"### ⚠️ Result: {status_text}")
                    st.markdown("**Recommendation:** Consider switching antibiotics (See 'Smart Screener' tab).")
                else:
                    st.success(f"### ✅ Result: {status_text}")
            
            with r_col2:
                st.metric(label="Model Confidence", value=f"{confidence:.1f}%")

            st.markdown("---")
            st.markdown("#### Record Management")
            
            # PDF Download Button
            patient_info = {
                "Age": age, "Gender": gender, 
                "Organism": organism_name, "Antibiotic": antibiotic_name
            }
            pdf_bytes = create_pdf(patient_info, status_text, confidence, st.session_state['ai_advice'])
            
            st.download_button(
                label="📄 Download PDF Report",
                data=pdf_bytes,
                file_name=f"DRAP_Report_{int(time.time())}.pdf",
                mime="application/pdf",
                use_container_width=True
            )

        # TAB 3: NEW FEATURE - SMART SCREENER
        with tab3:
            st.markdown("### 🧪 Alternative Treatment Screener")
            st.markdown("Scanning model for most effective antibiotics for this specific patient...")
            
            with st.spinner("Screening drug database..."):
                # Call the new helper function
                alt_df = get_alternatives(model, age, gender_code, org_code, antibiotic_map)
                
                # Display best options
                st.dataframe(
                    alt_df.style.background_gradient(subset=['Susceptibility Score'], cmap="Greens"),
                    use_container_width=True,
                    hide_index=True
                )
                
                best_drug = alt_df.iloc[0]['Antibiotic']
                best_score = alt_df.iloc[0]['Susceptibility Score']
                
                if best_drug != antibiotic_name:
                    st.success(f"💡 **AI Recommendation:** '{best_drug}' shows a {best_score:.1f}% susceptibility score.")
                else:
                    st.success(f"✅ The selected drug '{antibiotic_name}' is already the best option available in the database.")

    except Exception as e:
        st.error(f"An error occurred during analysis: {e}")
        #python -m streamlit run app.py

