import pandas as pd

import pandera.pandas as pa

from typing import Dict, List
from pathlib import Path
import sys
import os
import yaml
from loguru import logger


#project_root = Path("/app")
project_root = Path(__file__).resolve().parent.parent.parent.parent


logger.add(f"{project_root}/logs/preprocessing.log",
         rotation="00:00",        # New file daily
         retention="1 week",      # Keep 1 week of logs
         format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")


class Schema(pa.DataFrameModel):
     '''
     Data validation class for incoming raw data columns with pandera.

     Checks attributes such as values having to be greater than zero
     
     '''
     co2_percent: float = pa.Field(ge=0)
     oxygen_percent: float = pa.Field(ge=0)
     dissolved_oxygen_concentration: float = pa.Field(ge=1)
     vessel_weight: float = pa.Field(ge=55000)
     pH : float = pa.Field(ge=2)
     temperature: float = pa.Field(ge=200)
     raman_spectra: float = pa.Field(ge=0)
     aeration_rate: int = pa.Field(ge=20)
     paa_flow : float = pa.Field(ge=0)
     substrate_flow_rate : int = pa.Field(ge=0)
   
class PreprocessingPipeline:
    """
    A pipeline for preprocessing the raw data.

    This class handles the preprocessing steps including:
    - Column renaming
    - Column dropping

    Args:
        config (Dict[str, str]): Configuration dictionary containing preprocessing parameters
    """
    def __init__(self, config: Dict[str, str]):
        self.config = config['preprocessing']

    @staticmethod
    def rename_columns(df, column_mapping):
        """
        Rename columns in the dataset using a mapping dictionary.

        Args:
            df (pd.DataFrame): Input DataFrame
            column_mapping (dict): Dictionary mapping old column names to new column names.
                                 Example: {'old_name': 'new_name'}

        Returns:
            pd.DataFrame: DataFrame with renamed columns
        """
        return df.rename(columns=column_mapping)
    



    def run(self, df: pd.DataFrame):
        """
        Execute the complete preprocessing pipeline on the input DataFrame.

        Additionally validates dataframe columns against expected schema, and asserts that the correct columns
        are passed on to further pipeline steps

        Args:
            df (pd.DataFrame): Input DataFrame to be preprocessed

        Returns:
            pd.DataFrame: Preprocessed DataFrame with renamed columns, dropped columns, and target variable
        """
        df.reset_index(drop=True, inplace=True)
        df = self.rename_columns(df, self.config['column_mapping'])
        df = df[self.config['input_features'] + self.config['targets_renamed']  ]

        df_nonfaultybatches = df[~df.isin({'batch_number':self.config['faulty_batch_numbers']})]

        assert df_nonfaultybatches.columns.tolist() == self.config['input_features'] + self.config['targets_renamed']

        try:
            logger.info("validating preprocessed dataframe against pandera Schema")
            Schema.validate(df_nonfaultybatches)
        except pa.errors.SchemaError as exc:
            logger.error(exc)
            print(exc)
        logger.info("preprocessed dataframe schema validated against pandera Schema")
        return df_nonfaultybatches
    

# class done with extended validation function for input schema and logical field values


