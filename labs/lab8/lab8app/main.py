# lab8app/main.py
from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib

app = FastAPI()

# Load model once at startup
model = joblib.load("reddit_model_pipeline.joblib")

class RedditComment(BaseModel):
    text: str

@app.post("/predict")
def predict(comment: RedditComment):
    # Predict probability of removal (1 = removed, 0 = not removed)
    proba = model.predict_proba([comment.text])[0]
    pred = int(proba[1] > 0.5)
    return {"prediction": pred, "probability_removed": proba[1]}
