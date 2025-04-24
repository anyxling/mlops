# lab8app/main.py
from fastapi import FastAPI
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
import joblib

app = FastAPI()

# Load model from MLflow
model = joblib.load("../model.joblib")

# Endpoint to trigger predictions on a fixed CSV
@app.get("/predict")
def predict():
    df = pd.read_csv("x_test_scaled.csv")
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    pred = model.predict(df.sample(1, random_state=42)).tolist()
    return {"predictions": pred}