from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle

# Load Isolation Forest model
with open("1isolation_forest_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load the fitted RobustScaler
with open("scaler.pkl", "rb") as s:
    scaler = pickle.load(s)

# Define continuous variables that were scaled during training
continuous_vars = ['age', 'cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'glucose']

# Define input schema
class PatientData(BaseModel):
    male: int
    age: float
    education: float
    cigsPerDay: float
    BPMeds: float
    prevalentStroke: int
    prevalentHyp: int
    diabetes: int
    totChol: float
    sysBP: float
    diaBP: float
    BMI: float
    glucose: float

# Create FastAPI app
app = FastAPI()

@app.get("/")
def root():
    return {
        "message": "🔍 Welcome to the Framingham Anomaly Detection API using Isolation Forest. Use /detect to POST patient data."
    }

@app.post("/detect")
def detect_anomaly(data: PatientData):
    df = pd.DataFrame([data.dict()])

    # Scale only continuous variables
    df[continuous_vars] = scaler.transform(df[continuous_vars])

    # Predict anomaly (1 = normal, -1 = anomaly)
    prediction = model.predict(df)[0]

    # Anomaly score (lower score = more anomalous)
    score = model.decision_function(df)[0]

    result = "Normal" if prediction == 1 else "Anomaly"

    return {
        "Prediction_Label": result,
        "Prediction_Code": int(prediction),
        "Anomaly_Score": round(float(score), 4)
    }
