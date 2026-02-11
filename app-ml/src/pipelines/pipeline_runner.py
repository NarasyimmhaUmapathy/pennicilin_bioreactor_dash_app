import os
import sys
from pathlib import Path

from catboost import CatBoostRegressor
from loguru import logger

project_root = Path("/app")
#project_root = Path(__file__).resolve().parent.parent.parent.parent

#project_root = Path(__file__).resolve().parent.parent.parent.parent # Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'app-ml' /'src'))
sys.path.append(str(project_root / 'models'))


import pandas as pd
import numpy as np
from typing import Dict, Any
from common.data_manager import DataManager
from common.gcp_functions import load_csv_from_gcs
from common.data_classes import  InferenceValidator
from pipelines.preprocessing import PreprocessingPipeline
from pipelines.feature_engineering import FeatureEngineeringPipeline
from pipelines.training import TrainingPipeline
from pipelines.inference import InferencePipeline
from pipelines.postprocessing import PostprocessingPipeline

from sklearn.metrics import root_mean_squared_error

# logger 


logger.add(f"{project_root}/logs/pipeline_runner.log",
         rotation="00:00",        # New file daily
         retention="1 week",      # Keep 1 week of logs
         format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")



class PipelineRunner:
    """
    A class that orchestrates the execution of all stages in the ML pipeline.

    This includes:
    - Preprocessing
    - Feature engineering
    - Training
    - Inference
    - Postprocessing

    Attributes:
        config (Dict[str, Any]): Configuration dictionary.
        data_manager (DataManager): Manages loading/saving and transformation of data.
        real_time_data (pd.DataFrame): Cached real-time production data for inference.
        current_database_data (pd.DataFrame): Cached production database data for inference.
        prod_data_path (str): Path to the production database file.
        preprocessing_pipeline (PreprocessingPipeline): Handles data preprocessing steps.
        feature_eng_pipeline (FeatureEngineeringPipeline): Handles feature engineering steps.
        training_pipeline (TrainingPipeline): Handles model training steps.
        inference_pipeline (InferencePipeline): Handles inference steps.
        postprocessing_pipeline (PostprocessingPipeline): Handles postprocessing steps.
    """

    def __init__(self, config: Dict[str, Any], data_manager: DataManager):
        """
        Initialize the pipeline runner and its pipeline components.

        Args:
            config (Dict[str, Any]): Dictionary containing all pipeline configurations.
            data_manager (DataManager): Instance for managing I/O operations on data.
        """
        self.config = config
        self.data_manager = data_manager



        # Initialize individual pipeline components
        self.preprocessing_pipeline = PreprocessingPipeline(config=config)
        self.feature_eng_pipeline = FeatureEngineeringPipeline(config=config)
        self.training_pipeline = TrainingPipeline(config=config)
        self.inference_pipeline = InferencePipeline(config=config)
        self.postprocessing_pipeline = PostprocessingPipeline(config=config)

        # Load real-time data
        #self.real_time_data = self.data_manager.load_data(
         #   os.path.join(
          #      config['data_manager']['prod_data_folder'],
           #     config['data_manager']['real_time_data_prod_name']
           # )
       # )

        # Path to current production database
        self.prod_data_path = os.path.join(project_root,
                config['data_manager']['prod_data_folder'],
                config['data_manager']['prod_database_name']
            )

        self.reference_data_path = os.path.join(
             self.config['data_manager']['monitoring_data_folder'],
            self.config['data_manager']['monitoring_database_name']
        )

        self.model_path = project_root / 'models' / 'trained_catboost_model'


        

        # Load existing production database
        #self.current_database_data = self.data_manager.load_data(self.prod_data_path)




    def run_data_transformation(self) -> pd.DataFrame:
        '''

        Loadas production data from folder in gcp storage bucket

        Performs transformations and feature engineering on incoming batch data, and computes target column
        for inference

        Returns dataframe with necessary features engineered for inference, and the target column
        '''
        df = load_csv_from_gcs("pennicilin_batch_yield","production_data","production_data.csv")
        df = self.preprocessing_pipeline.run(df)
        df = self.feature_eng_pipeline.run(df=df)
        df_with_target  = self.training_pipeline.make_target(df, target_params=self.config['training']['target_params'])
        
        return df_with_target


    def run_training(self) -> pd.DataFrame:
        """
        Run the full training pipeline:
        1. Loads and preprocesses data
        2. Performs feature engineering, and partitions specified batch number for monitoring purposes
        3. Trains the model and logs it and other metrics in mlflow if threshold evaluation metric is passed
        4. Saves the trained model locally for inference

        Returns:
            dataframe with engineered features and target column
        """
        df = load_csv_from_gcs("pennicilin_batch_yield","production_data","production_data.csv")
        df = self.preprocessing_pipeline.run(df=df)
        df = self.feature_eng_pipeline.run(df=df)
        df_with_target  = self.training_pipeline.make_target(df, target_params=self.config['training']['target_params'])
        
        trained_model = self.training_pipeline.train_log_model(df)
        #self.postprocessing_pipeline.run_train(model=trained_model)
        return df_with_target

    def run_inference(self,df:pd.DataFrame,batch_number:int,model:CatBoostRegressor) -> tuple:
        """
        Run the full inference pipeline:
        1. Load real-time data for the current timestamp
        2. Append to the production database
        3. Prepare the latest batch
        4. Preprocess, transform, and predict
        5. Postprocess and store the prediction
        6. Update the production database

        Args:
            df: Production data for inference
            batch_number: Batch number of data to run inference on
            model: inference model

        Returns:
            object of class InferenceValidator, containing rmse score, predictions array and drift report
        """
        import random,math
      
        
        df_batch = df[df['batch_number'] == batch_number]
        
        # Step 5: Run inference
        y_pred = self.inference_pipeline.run(df_batch,model)

        #convert predictions array to list for api compatibility
        logger.info("converting predictions to a list of arrays of type float32")
        y_pred_rounded = np.round(np.asarray(y_pred, dtype=np.float32),3).tolist()

        logger.info("computing rmse score")
        rmse_score = root_mean_squared_error(df_batch['target_pennicilin'],y_pred)


        """
        Load reference data, filter data to the current batch dataset,
        and compute share of drifted features and drift scores
        """
        drift_columns = self.config['preprocessing']['input_features']


        logger.info("computing drift report")
        report,report_uri= self.inference_pipeline.monitor_data_drift(current_batch_data=df_batch,current_batch_number=batch_number)

        logger.info("drift report computed")

        from datetime import date

        today = date.today()

        today = today.strftime("%d/%m/%Y")

     

        logger.info("returning rmse score, predictions and drift report as inference validator object values")
        inference_validator = InferenceValidator(rmse_score=rmse_score,
                                                 y_pred=y_pred_rounded,
                                                 report=report,
                                                 report_uri=report_uri)

        return (inference_validator.rmse_score,
                inference_validator.y_pred,
                inference_validator.report,
                inference_validator.report_uri)

    
        

