import requests
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger

project_root = Path(__file__).resolve().parent.parent.parent.parent


logger.add(f"{project_root}/logs/test_inference_api.log",
         rotation="00:00",        # New file daily
         retention="1 week",      # Keep 1 week of logs
         format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")


test_batches = np.arange(30,35,1)

for b in test_batches:
    r = requests.post("https://inference-api-242173489543.europe-west3.run.app/run-inference", json={"batch_number": int(b)})
    r_metrics = requests.get("https://inference-api-242173489543.europe-west3.run.app/latest_monitoring_metrics")
    logger.info(f"getting json data for batch number: {b}")
    data = r.json()

    logger.info("test api status:", r.status_code)

    try:
        data = r.json()
        data_metrics = r_metrics.json()
    except Exception:
        print("Non-JSON response:\n", r.text)
        continue

    logger.info("response keys from API:", list(data.keys()))

    if r.status_code != 200 or data.get("status") != "success":
        logger.error("API ERROR:", data)
        continue

    # safe access
    logger.info("RMSE from api testing:", data.get("root mean squared error score"))
    logger.info("num_predictions from api testing:", data.get("num_predictions"))

    assert r.status_code == 200
    assert data["num_predictions"] > 800
    assert data["root mean squared error score"] > 0
    assert data["root mean squared error score"] < 0.29 #set to max allowed rmse score
    assert data["status"] == "success"
    assert type(data["aeration_drift_detected"]) == bool
    assert type(data["substrate_flow_rate_drift_detected"]) == bool

