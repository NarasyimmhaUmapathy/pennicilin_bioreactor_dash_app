from dataclasses import dataclass
from evidently.report import Report
import numpy as np
import pandas as pd
from typing import List


@dataclass()
class CrossValidationData:
    """
    Class to validate result of cross validation for parameter tuning
    """
    X_train: pd.DataFrame
    y_train: pd.DataFrame
    X_val: pd.DataFrame
    y_val: pd.DataFrame


@dataclass()
class InferenceValidator:
    """
    Class to validate run inference function of pipeline
    """

    y_pred : List[np.array]
    report : Report
    rmse_score : float
    report_uri: str



@dataclass()
class TrainResult:
    rmse_cv: float
    training_data_uri: str
    model_name: str
    run_id: str | None = None
    run_name: str | None = None
    num_features_training: int | None = None
    target_variable: str | None = None

