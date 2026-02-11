import os,sys
from statistics import correlation

from catboost import CatBoostRegressor
from dotenv import load_dotenv
from loguru import logger
from datetime import datetime
import json
from google.cloud import storage


import pandas as pd
from pathlib import Path
import numpy as np
from typing import Dict, Any


from common.utils import load_model
from common.data_manager import DataManager


from functools import lru_cache
import mlflow.pyfunc
from sklearn.metrics import root_mean_squared_error,make_scorer

from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric,DatasetCorrelationsMetric,ColumnCorrelationsMetric,ColumnDistributionMetric
from evidently.metrics import ColumnDriftMetric
from evidently.metrics import ColumnSummaryMetric
from evidently.metrics import DatasetDriftMetric
from evidently.metrics import DatasetMissingValuesMetric
from evidently.report import Report
from evidently.metric_preset import ClassificationPreset, DataDriftPreset, DataQualityPreset


from google.cloud import storage


#project_root = Path(__file__).resolve().parent.parent.parent.parent
project_root = Path("/app")


logger.add(f"{project_root}/inference_runner.log",
         rotation="00:00",        # New file daily
         retention="1 week",      # Keep 1 week of logs
         format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")




"""
Initializing mlflow tracking server to load and manage models

Uri is being loaded from .env file in the pipelines folder

"""

load_dotenv()
logger.info("set mlflow tracking URI")
mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))


logger.info("initialize gcp storage client")
client = storage.Client()


logger.add(f"{project_root}/logs/inference_pipeline.log",
         rotation="00:00",        
         format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
         level="INFO")

logger.add(f"{project_root}/logs/errors.log",
         level="ERROR")






class InferencePipeline:
    """
    A pipeline for making predictions using a trained model.

    This class handles:
    - Loading a trained model
    - Preparing input data for inference
    - Making predictions
    - Post-processing predictions

    Args:
        config (Dict[str, Any]): Configuration dictionary containing inference parameters
    """
    def __init__(self,config: Dict[str, Any]) -> None:
        """
        Initializes the InferencePipeline with configuration data.

        Args:
            config (Dict[str, Any]): Configuration dictionary containing inference parameters
        """
        self.config = config
        self.reference_data_path = project_root / 'data' / 'monitoring' / 'monitoring_data.parquet.gzip'

        self.data_manager = DataManager(config=config)

    @lru_cache(maxsize=1)
    def load_model(self):
        print("Loading model from MLflow...")
        model = load_model(base_path=self.config['pipeline_runner']['model_path'])
        return model
    
    def evaluate_batch_run(self,predictions:np.array,true_values:np.array,batch_number:int) -> float:

        """
        Evaluates recently finished batch run for regression metrics
        
        """

        model = self.load_model()


        rmse_score = root_mean_squared_error(true_values,predictions)

        logger.info(f"rmse score of batch number {batch_number} is {rmse_score}")

        return rmse_score
    
    def monitor_data_drift(self,current_batch_data:pd.DataFrame,current_batch_number:int):

        """
        Method which checks the current data drift and produces an evidently report

        Is executed after each batch inference run. Reference data is sampled from a list of batches not used in training (contained
        in the feature engineering pipeline)

        Relies on an Evidently report to monitor drift in the incoming, raw, unprocessed features.
        
        """
        import random

        reference_data =  self.data_manager.load_data(self.reference_data_path)

        reference_batch_number = random.choice(self.config['feature_engineering']['batch_numbers_drift_monitoring'])


        reference_data_batch = reference_data[reference_data['batch_number'] == reference_batch_number]



        print("producing drift report")
        logger.info(f"producing drift report for batch number {reference_batch_number}")
 
        
        report = Report([
        DataDriftPreset(drift_share=0.5,num_stattest='wasserstein',num_stattest_threshold=0.25),
        DatasetCorrelationsMetric(),
        DatasetMissingValuesMetric(),
            ])


        # we are examining drift in the raw original features
        drift_columns = self.config['preprocessing']['input_features']
        report.run(reference_data=reference_data[drift_columns],
                          current_data=current_batch_data[drift_columns])


    
        today = datetime.now()
        year = today.year
        month = today.month
        day = today.day
        hour = today.hour
        minute = today.minute

        timestamp = f"{year}_{month}_{day}_{hour}_{minute}"
        report_path = f'{project_root}/reports/drift_report_batchnum({current_batch_number})_{timestamp}.html'

        report.save_html(str(report_path))

        metadata = {
        "current_batch": int(current_batch_number),
        "reference_batch": int(reference_batch_number),
        "report_file": report_path,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        metadata_path = project_root / "reports" / "latest_drift_report.json"

        with open(metadata_path, "w") as f:
            json.dump(metadata, f)


        bucket_name = os.getenv("REPORT_BUCKET","pennicilin_batch_yield")  # set in Cloud Run env vars
        logger.info(f"bucket for uploading drift reports {bucket_name} accessed from env")
        gcs_html_uri = None
        gcs_latest_json_uri = None

        if bucket_name:
            logger.info("initiating storage bucket client to upload drift reports")

            bucket = client.bucket(bucket_name)

            gcs_prefix = f"drift_reports/batch={current_batch_number}"
            gcs_html_path = f"{gcs_prefix}/latest.html"
            gcs_json_path = f"{gcs_prefix}/latest.json"

            bucket.blob(gcs_html_path).upload_from_filename(str(report_path))
            bucket.blob(gcs_json_path).upload_from_string(
                json.dumps(metadata),
                content_type="application/json",
            )


            gcs_html_uri = f"gs://{bucket_name}/{gcs_html_path}"
            gcs_json_uri = f"gs://{bucket_name}/{gcs_json_path}"
            metadata["gcs_latest_html"] = gcs_html_uri
            metadata["gcs_latest_json"] = gcs_json_uri


        #return reference_data_batch, report.get_html(),report,gcs_html_uri
        return report,gcs_html_uri

    def monitor_data_drift_gcs(self,current_batch_data: pd.DataFrame, current_batch_number: int):


        import random

        # 1) Pick reference batch + build reference/current frames
        reference_data = self.data_manager.load_data(self.reference_data_path)
        reference_batch_number = random.choice(self.config["feature_engineering"]["batch_numbers_drift_monitoring"])
        reference_data_batch = reference_data[reference_data["batch_number"] == reference_batch_number]

        drift_columns = self.config["preprocessing"]["input_features_drift_report"]

        report = Report([
                DataDriftPreset(drift_share=0.5, num_stattest="wasserstein", num_stattest_threshold=0.25),
                DatasetCorrelationsMetric(),
                DatasetMissingValuesMetric(),
        ])

        report.run(
                reference_data=reference_data[drift_columns],
                current_data=current_batch_data[drift_columns],
            )

        # 3) Save HTML locally (container filesystem)
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        reports_dir = Path(project_root) / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        local_html_path = reports_dir / f"drift_report_batch={current_batch_number}_{ts}.html"
        report.save_html(str(local_html_path))

        metadata = {
                "current_batch": int(current_batch_number),
                "reference_batch": int(reference_batch_number),
                "created_at": ts,
                "local_report_file": str(local_html_path),
            }
        local_metadata_path = reports_dir / "latest_drift_report.json"
        local_metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

        # 5) Upload to GCS if REPORT_BUCKET is set
        bucket_name = os.getenv("REPORT_BUCKET")  # set in Cloud Run env vars
        gcs_html_uri = None
        gcs_latest_json_uri = None

        if bucket_name:
            client = storage.Client()
            bucket = client.bucket(bucket_name)

            gcs_prefix = f"drift_reports/batch={current_batch_number}"
            gcs_html_path = f"{gcs_prefix}/latest.html"
            gcs_json_path = f"{gcs_prefix}/latest.json"

            bucket.blob(gcs_html_path).upload_from_filename(str(local_html_path))
            bucket.blob(gcs_json_path).upload_from_string(
                    json.dumps(metadata),
                    content_type="application/json",
                )

            gcs_html_uri = f"gs://{bucket_name}/{gcs_html_path}"
            gcs_latest_json_uri = f"gs://{bucket_name}/{gcs_json_path}"

            metadata["gcs_latest_html"] = gcs_html_uri
            metadata["gcs_latest_json"] = gcs_latest_json_uri

            return reference_data_batch, report.get_html(), report, {
                "local_html": str(local_html_path),
                "gcs_latest_html": gcs_html_uri,
                "gcs_latest_json": gcs_latest_json_uri,
                "metadata": metadata,
            }

    def run(self, x: pd.DataFrame, model:CatBoostRegressor) -> pd.DataFrame:
        """
        Execute the complete inference pipeline.

        This method:
        1. Makes predictions using the loaded model
        2. Gets current timestamp
        3. Passes predictions and timestamp to postprocessing pipeline

        Args:
            x (pd.DataFrame): Input DataFrame containing features for prediction
            model (CatBoostRegressor): Trained model of latest version.

        Returns:
            y_pred: Predictions array of chosen batch number
        """
        # Load the model
        #model = load_model(base_path=model_path)
      

        # Make prediction
        y_pred = (model.predict(x))

        #x['prediction'] = y_pred
        # Take the last point prediction only
        #y_pred = y_pred[-1]
        return y_pred
    

