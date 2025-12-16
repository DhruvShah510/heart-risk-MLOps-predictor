from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, computed_field, model_validator
from typing import Literal, Annotated, Any, Dict
import joblib
import pandas as pd

# --- MAPPING DICTIONARIES (Optional but helpful for clarity) ---
# We don't strictly need the dictionary for the computed fields below, 
# but they are good for robust validation (which we now include).

# --- 1. Load the Model Pipeline ---
try:
    # Adjust path to point to the root directory
    MODEL_PATH = '../rf_pipeline.joblib' 
    model_pipeline = joblib.load(MODEL_PATH)
    print(f"INFO: Model loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    # If loading inside the container, the path is simply 'rf_pipeline.joblib'
    MODEL_PATH = 'rf_pipeline.joblib'
    try:
        model_pipeline = joblib.load(MODEL_PATH)
        print(f"INFO: Model loaded successfully (Container Path: {MODEL_PATH})")
    except FileNotFoundError:
        print(f"FATAL ERROR: Model file not found.")
        model_pipeline = None
        
# --- 2 defining class for the fastapi oblject
class HeartFeatures(BaseModel):
    """
    Defines the expected input features for the heart disease prediction.
    Computed fields transform human-readable strings and raw values into 
    the numerical codes required by the Random Forest model.
    """
    # ----------------------------------------------------
    # INPUT FIELDS (User-facing names/types)
    # ----------------------------------------------------
    age: Annotated[int, Field(ge=25, le=90, description="Age of the patient (years).")]
    # Note: Using sex_label for the string input, and 'sex' will be the computed numerical field.
    sex_label: Annotated[str, Field(description="Gender (Male or Female).")]
    cp_label: Annotated[str, Field(description="Chest pain type (Typical angina, Atypical angina, Non-anginal pain, Asymptomatic).")]
    trestbps: Annotated[float, Field(ge=80, le=250, description="Resting Blood Pressure (mm Hg).")]
    chol: Annotated[float, Field(ge=100, le=600, description="Serum Cholesterol (mg/dl).")]
    # Fbs is the raw measurement value
    fbs_raw: Annotated[int, Field(ge=60,le=150,description="Fasting blood sugar level (mg/dl).")]
    restecg_label: Annotated[str, Field(description="Resting ECG results (Normal, ST-T wave abnormality, Left ventricular hypertrophy).")]
    thalachh: Annotated[float, Field(ge=60, le=220, description="Maximum heart rate achieved.")]
    exang_label: Annotated[str, Field(description="Exercise induced angina (Yes or No).")]
    oldpeak: Annotated[float, Field(ge=0.0, le=6.5, description="ST depression induced by exercise relative to rest.")]
    slope: Annotated[Literal[0, 1, 2], Field(description="The slope of the peak exercise ST segment (0=Upsloping, 1=Flat, 2=Downsloping).")]
    ca: Annotated[Literal[0, 1, 2, 3], Field(description="Number of major vessels (0-3).")] 
    thal_label: Annotated[str, Field(description="Thalassemia type (Normal, Fixed defect, Reversible defect).")]

    # ----------------------------------------------------
    # COMPUTED FIELDS (Must match the 13 feature names of the RF model)
    # ----------------------------------------------------
        
    @computed_field
    @property
    def sex(self) -> Literal[0, 1]:
        """Maps 'Male' or 'Female' to 1 or 0."""
        label = self.sex_label.lower()
        
        if label == "male":
            return 1
        elif label == "female":
            return 0
        else:
            # Raises a validation error if the string is neither 'male' nor 'female'
            raise ValueError(f"Invalid gender label: Must be 'Male' or 'Female'.")
        
    @computed_field
    @property
    def cp(self) -> Literal[0, 1, 2, 3]:
        """Maps Chest Pain Type strings to numerical codes (0, 1, 2, 3)."""
        cp_map = {
            "Typical angina": 0,
            "Atypical angina": 1,
            "Non-anginal pain": 2,
            "Asymptomatic": 3
        }
        if self.cp_label in cp_map:
            return cp_map[self.cp_label]
        else:
            raise ValueError(f"Invalid CP label: '{self.cp_label}'. Check documentation for valid options.")

    @computed_field
    @property
    def fbs(self) -> Literal[0, 1]:
        """Computes the binary Fasting Blood Sugar feature (1 if > 120, 0 otherwise)."""
        # Note: Using fbs_raw from input field
        return 1 if self.fbs_raw > 120 else 0
        
    @computed_field
    @property
    def restecg(self) -> Literal[0, 1, 2]:
        """Maps Resting ECG strings to numerical codes (0, 1, 2)."""
        ecg_map = {
            "Normal": 0,
            "ST-T wave abnormality": 1,
            "Left ventricular hypertrophy": 2
        }
        if self.restecg_label in ecg_map:
            return ecg_map[self.restecg_label]
        else:
            raise ValueError(f"Invalid RestECG label: '{self.restecg_label}'.")
        
    @computed_field
    @property
    def exang(self) -> Literal[0, 1]:
        """Maps Exercise Angina strings to 1 (Yes) or 0 (No)."""
        if self.exang_label.lower() == "yes":
            return 1
        elif self.exang_label.lower() == "no":
            return 0
        else:
            raise ValueError(f"Invalid Exang label: Must be 'Yes' or 'No'.")

    @computed_field
    @property
    def thal(self)-> Literal[1, 2, 3]:
        """Maps Thalassemia strings to numerical codes (1, 2, 3)."""
        thal_map = {
            "Normal": 1,
            "Fixed defect": 2,
            "Reversible defect": 3
        }
        if self.thal_label in thal_map:
            return thal_map[self.thal_label]
        else:
            raise ValueError(f"Invalid Thalassemia label: '{self.thal_label}'.")


# Note: In your FastAPI endpoint, you will extract all fields *except* the input_labels (e.g., sex_label, fbs_raw)
# and use the computed fields (sex, cp, fbs, etc.) to build the Pandas DataFrame for prediction.
    
# --- 3. Initialize FastAPI App ---

app = FastAPI(
    title="Heart Disease Risk Predictor API",
    description="Random Forest model deployed with FastAPI for risk assessment.",
    version="1.0.0"
)

# methods (get, post)
@app.get('/')
def home():
    return {'message':'this is the home page'}


@app.post('/predict')
def predict_heart(data: HeartFeatures):

    if model_pipeline is None:
        return JSONResponse(status_code=500, content={'message': "Model not loaded."})

    # Define the exact order of the 13 features expected by the pipeline (Mandatory)
    FEATURE_ORDER = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalachh', 
        'exang', 'oldpeak', 'slope', 'ca', 'thal'
    ]

    input_df = pd.DataFrame([{
        'age': data.age,
        'sex' : data.sex,
        'cp' : data.cp,
        'trestbps' : data.trestbps,
        'chol' : data.chol,
        'fbs' : data.fbs,
        'restecg' : data.restecg,
        'thalachh' : data.thalachh,
        'exang' : data.exang,
        'oldpeak' : data.oldpeak,
        'slope' : data.slope,
        'ca' : data.ca,
        'thal' : data.thal
    }], columns = FEATURE_ORDER)

    # 3. Predict and return
    prediction = model_pipeline.predict(input_df)[0]
    risk_score = model_pipeline.predict_proba(input_df)[0][1]

    #--- THE CRITICAL FIX ---
    # Cast NumPy int64 (prediction) and float64 (risk_score) to native Python types
    return JSONResponse(
        status_code=200, 
        content={
            'predicted_category': int(prediction),
            'risk_score_probability': round(float(risk_score), 4), # Safely cast float64
            'risk_level': "High Risk" if prediction == 1 else "Low Risk"
        }
    )

    
