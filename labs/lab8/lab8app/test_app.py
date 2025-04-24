# lab8app/test_app.py
import requests

response = requests.get("http://127.0.0.1:8000/predict")
print("Predictions:", response.json())