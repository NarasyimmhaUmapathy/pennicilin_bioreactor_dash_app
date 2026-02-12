import sys
import os
from pathlib import Path
import numpy as np
from flask.cli import load_dotenv
from loguru import logger

from google.oauth2 import service_account
from google.cloud import storage

import mlflow
from threading import Lock


from pipelines.pipeline_runner import PipelineRunner
from common.data_manager import DataManager
from common.utils import read_config, configure_mlflow


#project_root = Path("/app")
project_root = Path(__file__).resolve().parents[2]
#os.chdir(project_root/"entrypoint")
os.chdir(project_root)

sys.path.append(str(project_root))
sys.path.append(os.path.join(project_root, 'common'))
sys.path.append(str(project_root / 'app-ml' /'src'))




logger.add(f"{project_root}/logs/test_api.log",
         rotation="00:00",        # New file daily
         retention="1 week",      # Keep 1 week of logs
         format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")




config_path = project_root / 'config' / 'config.yaml'
model_path = project_root / 'models' /  'prod' /'trained_catboost_model.cbm'
config = read_config(config_path)
data_manager = DataManager(config)

data_path = os.path.join(
             config['data_manager']['prod_data_folder'],
            config['data_manager']['prod_database_name']
        )

data_path_test = os.path.join(config['data_manager']['prod_data_folder'],
            'docker_test_data.parquet'
        )

batch_number = 5

load_dotenv()
logger.info("creating credentials for GCP")
cred_path = project_root / 'app-ml' / 'entrypoint' / 'sa-key.json'

creds = service_account.Credentials.from_service_account_file(
    cred_path
)
client = storage.Client(credentials=creds, project=creds.project_id)

MODEL = None
_MODEL_LOCK = Lock()


_DF_CACHE = {"df": None, "loaded_at": 0.0}
_DF_LOCK = Lock()

# --- Configure MLflow (Databricks)  ---
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "databricks")
registry_uri = os.getenv("MLFLOW_REGISTRY_URI", "databricks")

experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "/Shared/penicillin time series model")

logger.info("setting up mlflow tracking server")
mlflow_client = configure_mlflow(tracking_uri_key=tracking_uri, registry_uri_key=registry_uri,
                                 experiment_name=experiment_name)
logger.info("loading model from mlflow")

model = mlflow.sklearn.load_model("models:/workspace.default.catboost_model/11")

pipeline_runner = PipelineRunner(config=config, data_manager=data_manager)
df_inference = pipeline_runner.run_data_transformation()

df_inference_batch = df_inference[df_inference['batch_number'] == batch_number]

rmse_score, predictions, report, report_uri = pipeline_runner.run_inference(df_inference, batch_number,model=model)
print(rmse_score)
print(min(predictions),max(predictions),np.mean(predictions))
print(f" {len(predictions)} as number of predictions")
