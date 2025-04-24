# lab8app/test_app.py
import requests

data = {"text": "This comment might get removed!"}
response = requests.post("http://127.0.0.1:8000/predict", json=data)

print("Response from API:")
print(response.json())
