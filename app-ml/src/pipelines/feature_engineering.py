import pandas as pd
from pathlib import Path
from typing import Dict, Any, List

import sys,os
from loguru import logger


project_root = Path("/app")
#project_root = Path(__file__).resolve().parent.parent.parent.parent


logger.add(f"{project_root}/logs/feature_engineering.log",
         rotation="00:00",        # New file daily
         retention="1 week",      # Keep 1 week of logs
         format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")




class FeatureEngineeringPipeline:
    """
    A pipeline for creating and engineering features from preprocessed data.

    This class handles feature engineering steps including:
    - Creating lag features for time series data

    Args:
        config (Dict[str, Any]): Configuration dictionary containing feature engineering parameters
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config['feature_engineering']
        self.config_training = config['training']
        self.config_global = config

    @staticmethod
    def add_lag_feats(df: pd.DataFrame, params: Dict[str, List[int]]) -> pd.DataFrame:
        """
        Add lag features to the DataFrame based on specified parameters.

        Args:
            df (pd.DataFrame): Input DataFrame
            params (Dict[str, List[int]]): Dictionary containing feature names and their corresponding lag periods.
                Example: {
                    'col1': [1, 2, 3],
                    'col2': [1, 5, 10],
                    ...
                }

        Returns:
            pd.DataFrame: DataFrame with added lag features
        """
        for feat, lags in params.items():
            for lag in lags:
                df[f'{feat}_lag_{lag}'] = df[feat].shift(lag).bfill()
        return df 
    
    @staticmethod
    def add_delta_feats(df: pd.DataFrame, params: Dict[str, List[int]]) -> pd.DataFrame:
        """
        Add differenced features to the DataFrame based on specified parameters.

        Args:
            df (pd.DataFrame): Input DataFrame
            params (Dict[str, List[int]]): Dictionary containing feature names and their corresponding delta periods.
                Example: {
                    'col1': [1, 2, 3],
                    'col2': [1, 5, 10],
                    ...
                }

        Returns:
            pd.DataFrame: DataFrame with added delta features
        """
        for target in params:
                df[f'{target}_delta_{target}'] = df[target].shift(-2).ffill()
        return df 
    

    @staticmethod
    def interpolate_target(df: pd.DataFrame, params: Dict[str, List[int]],interpolation_method:str) -> pd.DataFrame:
        """
        Impute missing values in target or targets, based on spline interpolation.

        Args:
            df (pd.DataFrame): Input DataFrame
            params (Dict[str, List[int]]): Dictionary containing target names.
                Example: {
                    'col1': [1, 2, 3],
                    'col2': [1, 5, 10],
                    ...
                }

        Returns:
            pd.DataFrame: DataFrame with interpolated target features
        """
        for target in params:
                df[f'{target}_interpolated'] = df[target].interpolate(option=f'{interpolation_method}').fillna(method='bfill')
                logger.info(f"interpolated {target} column according to {interpolation_method} method ")
        return df
    

    def split_data_production_and_monitoring_reference(self,df:pd.DataFrame,monitoring_batch_numbers:list[int]
                                                       ,monitoring_reference_data_path:str):
        
        """
        Args:
            df: input dataframe,
            monitoring_batch_numbers: batch numbers to be set aside for drift monitoring purposes
            monitoring_reference_data_path: path in local or cloud file system to save dataset containing monitoring batches in

        Separates batch runs according to monitoring batch numbers parameter for
        drift monitoring during production inference.

        Returns.
            training dataframe with monitoring batch numbers removed
        
        """
        
         
        df_monitoring_reference = df[df['batch_number'].isin([86,87,88,89])]

        df_training = df[~df.isin({'batch_number':self.config['batch_numbers_drift_monitoring']})]

       # assert df_training['batch_number'].nunique() > len(monitoring_batch_numbers )


        df_monitoring_reference.to_parquet(monitoring_reference_data_path, index=False)

        return df_training
    
         


    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute the complete feature engineering pipeline on the input DataFrame.

        Adds lagged features, delta features, and interpolates target based on chose interpolation method.

        Reserves a portion of batch numbers for drift monitoring purposes later

        Args:
            df (pd.DataFrame): Input DataFrame to be processed
            inference: If function is to be used as part of real time inference.
+
        Returns:
            pd.DataFrame: DataFrame with engineered features including lag features
        """
        df = self.add_lag_feats(df, self.config['lag_params'])
        df = self.add_delta_feats(df,self.config['delta_params'])
        df = self.interpolate_target(df,self.config['targets_interpolate'],interpolation_method='spline')

        monitoring_path  = os.path.join(project_root,
            self.config_global['data_manager']['monitoring_data_folder'],
            self.config_global['data_manager']['monitoring_database_name']
        )

        #if not inference:
        if os.path.exists(monitoring_path):
            logger.info("no monitoring reference dataframe exists yet,creating one")
            df = self.split_data_production_and_monitoring_reference(df=df,
                                                            monitoring_batch_numbers=self.config['batch_numbers_drift_monitoring']
                                                            ,monitoring_reference_data_path=monitoring_path)

        return df
    
