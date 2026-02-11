import sys
import os
import time
import random
from flask import Flask, jsonify,request,Response
from pathlib import Path
import pandas as pd
import numpy as np
from threading import Lock

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID


from loguru import logger
from dotenv import load_dotenv
from functools import lru_cache

from datetime import datetime, timedelta, timezone
from shapash.explainer.smart_explainer import SmartExplainer


from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST

logger.info("creating counter and gauges to monitor batch rmse scores and drift metrics")
REQUESTS = Counter("inference_requests_total", "Total inference requests", ["endpoint", "status"])
INFERENCE_LATENCY = Histogram("inference_latency_buckets", "Inference latency", buckets=(0.1,0.25,0.5,1,2,5,10,30))
INFERENCE_LATENCY_DURATION = Gauge("inference_latency_seconds", "Inference latency")

RMSE_LATEST = Gauge("model_rmse_latest", "Latest RMSE score observed", ["model_name"])
FEATURE_DRIFT_SCORE = Gauge("latest_feature_drift_score", "Latest feature drift", ["feature_name"])  # optional
SHARE_OF_DRIFTED_FEATURES = Gauge("share_of_drifted_features", "Latest share of drifted features")  # optional
PREDICTIONS_UPLOADED = Gauge("predictions_uploaded","predictions_uploaded", "Number of prediction files uploaded to GCS")

logger.info("initializing object to store latest inference output for grafana")
_LATEST_LOCK = Lock()
_LATEST = {
    "updated_at": None,
    "batch_number": None,
    "rmse": None,
    "share_of_drifted_columns": None,
    "aeration_drift_score": None,
    "substrate_flow_rate_drift_score": None,
    "global_drift_status": None,
    "drift_report_uri": None,
    "predictions_uri": None,
}

from google.cloud import storage

project_root = Path("/app")
logger.add(f"{project_root}/inference_runner.log",
         rotation="00:00",        # New file daily
         retention="1 week",      # Keep 1 week of logs
         format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")

logger.info("changing directory to inference api folder")
os.chdir(project_root/"entrypoint")
#os.chdir(project_root)

load_dotenv()

logger.info("setting storage client for GCP access")
storage_client = storage.Client()

logger.info("appending relevant folders to sys path")
sys.path.append(str(project_root))
sys.path.append(os.path.join(project_root, 'src'))
sys.path.append(os.path.join(project_root, 'common'))
sys.path.append(str(project_root / 'app-ml' /'src'))


from common.utils import *
from common.gcp_functions import *
from common.data_classes import *
from common.data_manager import DataManager


from pipelines.pipeline_runner import PipelineRunner



app = Flask(__name__)

config_path = project_root / 'config' / 'config.yaml'
config = read_config(config_path)
data_manager = DataManager(config)

pipeline_runner = PipelineRunner(config,data_manager)


import os
import logging
from threading import Lock
from google.cloud import storage
from catboost import CatBoostRegressor

logger = logging.getLogger(__name__)



_DF_CACHE = {"df": None, "loaded_at": 0.0}
_DF_LOCK = Lock()

model_path = project_root / 'models' /  'prod' /'trained_catboost_model.cbm'
logger.info(f"inference model loaded from {model_path}")

def _update_latest(**kwargs):
    with _LATEST_LOCK:
        _LATEST.update(kwargs)
        _LATEST["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")


def load_model_once():
    """
    Loads catboost model once on startup for inference

    Model is saved ephemerally

    Returns:

    Trained catboost model

    """

    #inference_model = CatBoostRegressor()
    #inference_model.load_model(str(model_path))
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "databricks")
    registry_uri = os.getenv("MLFLOW_REGISTRY_URI", "databricks")

    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "/Shared/penicillin time series model")

    logger.info("setting up mlflow tracking server")
    mlflow_client = configure_mlflow(tracking_uri_key=tracking_uri, registry_uri_key=registry_uri,
                                     experiment_name=experiment_name)
    logger.info("loading model from mlflow")

    mlflow_model_path = os.environ["MODEL_MLFLOW_URI"]
    inference_model = mlflow.sklearn.load_model(mlflow_model_path)

    return inference_model

def safe_float(x):
    """
    Prevents None objects being converted to float

    Args:
        x:

    Returns:
        either float(x) or None
    """
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None

def get_production_df(ttl_seconds: int = 15 * 60):
    """
    Lazy-load production df on first request, refresh every ttl_seconds.
    Prevents loading big DF at import time (reduces baseline Cloud Run memory).

    maintains dictionary of dataframe and load time. If dataframe has not been loaded yet, load dataframe, else retrive
    already saved dataframe. Avoids causing OOM errors in the container at runtime
    """
    now = time.time()
    with _DF_LOCK:
        #if _DF_CACHE has not df value yet, or more time has passed between now and when dataframe was last loaded,
        # then only run the data transformation function to create feature df.
        if _DF_CACHE["df"] is None or (now - _DF_CACHE["loaded_at"]) > ttl_seconds:
            logger.info("Loading/refreshing production df from GCS (TTL=%ss)", ttl_seconds)
            df = pipeline_runner.run_data_transformation()
            _DF_CACHE["df"] = df
            _DF_CACHE["loaded_at"] = now
    return _DF_CACHE["df"]


logger.info("initializing inference model")
model = load_model_once()

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

@app.get("/latest_monitoring_metrics")
def status():
    with _LATEST_LOCK:
        payload = dict(_LATEST)  # shallow copy

    # If nothing has run yet:
    if payload["updated_at"] is None:
        return jsonify({
            "status": "empty",
            "message": "No inference has been run yet. Call POST /run-inference first."
        }), 200

    return jsonify({"status": "ok", **payload}), 200


@app.route('/run-inference', methods=['POST'])
def run_inference():
    start = time.time()
    try:
        data = request.get_json(silent=True)
        if data is None or "batch_number" not in data:
            REQUESTS.labels(endpoint="/run-inference", status="400").inc()
            return jsonify({"status": "error", "message": "batch_number missing"}), 400

        logger.info("reading in batch number")
        batch_number = int(data["batch_number"])
     
        #Load and transform data
        logger.info("transforming data")
        df_inference = get_production_df(ttl_seconds=15 * 60)

        # Retrieve relevant dataset with input batch number
        logger.info("retrieving batch data")
        df_inference_batch = df_inference[df_inference['batch_number'] == batch_number]

        if df_inference_batch.empty:
            return jsonify({
                "status": "error",
                "message": f"No rows found for batch_number={batch_number}. "
                           f"Available batches example: {sorted(df_inference['batch_number'].dropna().unique())[:20]}"
            }), 404

        # Run inference and obtain rmse score and predictions array
        logger.info(f"running inference on batch number  {batch_number}")
        rmse_score, predictions,report,report_uri = pipeline_runner.run_inference(df_inference_batch, batch_number,model)
        logger.info("run inference successful")
        logger.info(f"rmse score is {rmse_score} for batch number {batch_number}")

        bucket = os.environ["REPORT_BUCKET"]

        logger.info(f"uploading predictions to GCP bucket")
        predictions_uri = None
        try:
            predictions_uri = upload_predictions_parquet(client=storage_client,bucket_name=bucket
                                    ,prefix="batch_predictions",batch_number=batch_number,
                                   predictions=predictions,
                                   rmse=rmse_score)
        except Exception as e:
            logger.info("predictions could not be uploaded")

        #PREDICTIONS_UPLOADED.labels(bucket="pennicilin_batch_yield").inc()


        #setting rmse score of batch prediction for prometheus
        RMSE_LATEST.labels(model_name="my_model").set(float(rmse_score))
        REQUESTS.labels(endpoint="/run-inference", status="200").inc()
        logger.info(f"RMSE metric set to {rmse_score}")

        logger.info(f"prediction took {time.time()-start} seconds")
        INFERENCE_LATENCY_DURATION.set(time.time() - start)

        #extracting drift information
        metrics = report.as_dict()["metrics"]

        # find the DataDriftTable metric
        drift_table = next(
            m for m in metrics if m["metric"] == "DataDriftTable"
        )

        drift_status = next(
            m for m in metrics if m["metric"] == "DatasetDriftMetric"
        )

        aeration_drift_score = drift_table["result"]["drift_by_columns"]["aeration_rate"]["drift_score"]
        logger.info(f"setting feature drift score for aeration rate to {aeration_drift_score}")
        aeration_drift_detected = drift_table["result"]["drift_by_columns"]["aeration_rate"]["drift_detected"]

        substrate_flow_rate_drift_score = drift_table["result"]["drift_by_columns"]["substrate_flow_rate"]["drift_score"]
        logger.info(f"setting feature drift score for subtrate flow rate to {substrate_flow_rate_drift_score}")
        substrate_flow_rate_drift_detected = drift_table["result"]["drift_by_columns"]["substrate_flow_rate"][
            "drift_detected"]

        FEATURE_DRIFT_SCORE.labels(feature_name="aeration_rate").set(float(aeration_drift_score))
        FEATURE_DRIFT_SCORE.labels(feature_name="substrate_flow_rate").set(float(substrate_flow_rate_drift_score))
        logger.info("feature drift scores for aeration and substrate flow rates set")

        share_of_drifted_columns = drift_table["result"].get("share_of_drifted_columns")
        share_of_drifted_columns_safe_float = safe_float(share_of_drifted_columns)

        if share_of_drifted_columns_safe_float is not None:
            SHARE_OF_DRIFTED_FEATURES.set(share_of_drifted_columns_safe_float)
            logger.info("share of drifted features metric set")


        global_drift_status = drift_status["result"]["dataset_drift"]
        logger.info(f"global drift status is {global_drift_status}")

        #logger.info(f"setting predictions uploaded metric")
        #PREDICTIONS_UPLOADED.set(int(len(predictions)))

        logger.info("updating latest metrics for grafana")
        _update_latest(
            batch_number=batch_number,
            rmse=safe_float(rmse_score),
            share_of_drifted_columns=share_of_drifted_columns_safe_float,
            aeration_drift_score=safe_float(aeration_drift_score),
            substrate_flow_rate_drift_score=safe_float(substrate_flow_rate_drift_score),
            global_drift_status=bool(global_drift_status),
            drift_report_uri=report_uri,
            predictions_uri=predictions_uri,
        )

        return jsonify({
            "status": "success",
            "batch_number": batch_number,
            "predictions_uri": predictions_uri, #save in gcp bucket
            "drift_report_uri": report_uri,
            "num_predictions":len(predictions),
            "predictions_array":predictions,
            "aeration_drift_detected": aeration_drift_detected,
            "substrate_flow_rate_drift_detected":substrate_flow_rate_drift_detected,
            "global_drift_status": global_drift_status,
            "root mean squared error score": float(rmse_score),
            "share_of_drifted_columns": share_of_drifted_columns_safe_float

        })
    


    except Exception as e:
        REQUESTS.labels(endpoint="/run-inference", status="500").inc()
        return jsonify({"status": "error", "message": str(e)}), 500




@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})



if __name__ == "__main__":
    # Load configuration using utils function
   

    # Initialize the modules
    # Prepare production database every time the script starts
    #data_manager.initialize_prod_database(project_root=project_root)


    # Start the app
    logger.info("starting flask app on port 8080")
    app.run(host="0.0.0.0", port=8080)