import requests
import os

API_URL = os.getenv("INFERENCE_API_URL", "http://localhost:5001").rstrip("/")

def get_predictions(batch_number: int):
    url = f"{API_URL}/run-inference"
    payload = {"batch_number": batch_number}
    response = requests.post(url, json=payload, timeout=30)

    if response.status_code != 200:
        raise ValueError(f"API Error: {response.text}")

    data = response.json()

    if "predictions_array" not in data:
        raise ValueError("API Error: missing predictions")

    return data


