import streamlit as st
import requests
import json
import pandas as pd

# --- Configuration ---
# Your FastAPI server should be running locally at this address
FASTAPI_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="Heart Disease Risk Predictor (MLOps Project)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Utility Function to Post Data to FastAPI ---
def predict_risk(data):
    """Sends input data to the FastAPI prediction endpoint."""
    try:
        response = requests.post(FASTAPI_URL, json=data)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("Connection Error: Could not connect to the FastAPI server. Please ensure the server is running at http://127.0.0.1:8000.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"API Error: Server returned {response.status_code}. Details: {response.json().get('detail', 'No details available.')}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

# --- UI Layout and Input Fields ---

st.title("🫀 Heart Disease Risk Assessment Tool")
st.markdown("Enter the patient's clinical parameters below to get a risk prediction from the deployed Random Forest model.")

with st.form("risk_assessment_form"):
    
    # --- Row 1: Age, Sex, Chest Pain ---
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Numeric Input: Age
        age = st.number_input("Age (Years)", min_value=25, max_value=90, value=50, step=1, key='age')
        
        # Categorical Dropdown: Sex (Maps to sex_label)
        sex_label = st.selectbox(
            "Gender",
            options=["Male", "Female"],
            index=0,
            key='sex_label'
        )

    with col2:
        # Numeric Input: Trestbps (Resting BP)
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80.0, max_value=250.0, value=120.0, step=5.0, key='trestbps')

        # Categorical Dropdown: CP (Maps to cp_label)
        cp_label = st.selectbox(
            "Chest Pain Type",
            options=["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"],
            index=3, # Asymptomatic is often the most critical
            key='cp_label'
        )
        
    with col3:
        # Numeric Input: Cholesterol
        chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100.0, max_value=600.0, value=200.0, step=10.0, key='chol')

        # Raw FBS Input (Maps to fbs_raw, used by computed field)
        fbs_raw = st.number_input("Fasting Blood Sugar (mg/dl)", min_value=60, max_value=150, value=90, step=1, key='fbs_raw')


    st.markdown("---")
    st.subheader("ECG, Exercise, and Thalassemia")

    # --- Row 2: ECG, Max HR, Exang ---
    col4, col5, col6 = st.columns(3)
    
    with col4:
        # Categorical Dropdown: RestECG
        restecg_label = st.selectbox(
            "Resting ECG Results",
            options=["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"],
            index=0,
            key='restecg_label'
        )
        
        # Categorical Dropdown: Exang
        exang_label = st.selectbox(
            "Exercise Induced Angina",
            options=["No", "Yes"],
            index=0,
            key='exang_label'
        )

    with col5:
        # Numeric Input: Thalachh
        thalachh = st.number_input("Max Heart Rate Achieved", min_value=60.0, max_value=220.0, value=150.0, step=5.0, key='thalachh')

        # Numeric Input: Oldpeak
        oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=6.5, value=1.0, step=0.1, key='oldpeak')

    with col6:
        # Categorical Dropdown: Thal
        thal_label = st.selectbox(
            "Thalassemia Type",
            options=["Normal", "Fixed defect", "Reversible defect"],
            index=0,
            key='thal_label'
        )
        
        # Numerical Dropdown: CA (Number of Major Vessels)
        ca = st.selectbox(
            "Number of Major Vessels (0-3)",
            options=[0, 1, 2, 3],
            index=0,
            key='ca'
        )

        # Numerical Dropdown: Slope
        slope = st.selectbox(
            "Peak Exercise ST Segment Slope",
            options=[0, 1, 2],
            index=1,
            key='slope'
        )


    # --- Submit Button ---
    st.markdown("---")
    submitted = st.form_submit_button("Predict Heart Risk", type="primary")

# --- Results Display ---

if submitted:
    # 1. Collect data from all input fields in the format required by the Pydantic model
    input_data = {
        "age": age,
        "sex_label": sex_label,
        "cp_label": cp_label,
        "trestbps": trestbps,
        "chol": chol,
        "fbs_raw": fbs_raw,
        "restecg_label": restecg_label,
        "thalachh": thalachh,
        "exang_label": exang_label,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal_label": thal_label
    }
    
    # 2. Call the FastAPI endpoint
    with st.spinner('Analyzing patient data...'):
        result = predict_risk(input_data)

    # 3. Display Results
    if result:
        st.subheader("Prediction Results")
        
        risk_level = result['risk_level'] # This will be "High Risk" or "Low Risk"
        risk_prob = result['risk_score_probability']
        
        # Color coding the output based on risk level string (FIXED LOGIC)
        if "High Risk" in risk_level:
            st.error(f"🔴 WARNING: {risk_level} DETECTED")
        else:
            st.success(f"🟢 {risk_level} DETECTED")

        colA, colB = st.columns(2)
        
        with colA:
            st.metric("Risk Assessment", risk_level)
            
        with colB:
            st.metric("Probability of High Risk", f"{risk_prob * 100:.2f}%", help="This is the model's calculated probability of heart disease (Class 1).")
        
        st.info("Remember: This prediction is generated by an ML model and is NOT a substitute for professional medical advice.")