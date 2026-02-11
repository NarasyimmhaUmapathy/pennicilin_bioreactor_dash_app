import pandas as pd
import numpy as np
from datetime import date
from typing import Dict,Any,List
from pathlib import Path
import logging


from .utils import read_config

from evidently.metric_preset import ClassificationPreset, DataDriftPreset, DataQualityPreset
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric,DatasetCorrelationsMetric,ColumnCorrelationsMetric,ColumnDistributionMetric
from evidently.metrics import ColumnDriftMetric
from evidently.metrics import ColumnSummaryMetric
from evidently.metrics import DatasetDriftMetric
from evidently.metrics import DatasetMissingValuesMetric
from evidently.report import Report
from evidently import ColumnMapping

from shapash.explainer.smart_explainer import SmartExplainer



import matplotlib.pyplot as plt


def monitoring_report(drift_share:float,
                      statistical_method_num_features:str,
                      drift_score_threshold_num_features:float,
                      additional_reports:List[Report]) -> List[Report]:
    """

    Args:
    Params for DataDriftPreset:
        drift_share: share of drifted features from all input features for global drift to be true
        statistical_method_num_features: stat method to measure drift for numerical features
        drift_score_threshold_num_features: threshold of drift score, upon reached, drift is set to be true
    Additional reports:
        Additional reports of evidently format for monitoring (max 2)

    Returns:
        report: Evidently drift report or reports

    """
    data_drift_report = Report([
        DataDriftPreset(drift_share=drift_share, num_stattest=statistical_method_num_features,
                        num_stattest_threshold=drift_score_threshold_num_features),
        additional_reports[0],
        additional_reports[1],
        ])



    return data_drift_report,custom_report_1,custom_report_2

def log_feature_importance_catboost(model, X, artifact_dir:str,project_root:Path,top_n=30):
    """

    Args:
        model:  trained model
        X: validation dataset features
        artifact_dir: directory to store model feature importances
        project_root: root directory to store model feature importances
        top_n: top_n for feature importances

    Returns:

        fi: Feature Importance dataframe

    """
    importances = model.get_feature_importance(type="PredictionValuesChange")
    fi = pd.DataFrame({
        "feature": X.columns,
        "importance": importances
    }).sort_values("importance", ascending=False)


    #  Log a quick bar plot (best for eyeballing)
    top = fi.head(top_n).iloc[::-1]  # reverse for horizontal bar chart
    plt.figure()
    plt.barh(top["feature"], top["importance"])
    plt.xlabel("Importance (PredictionValuesChange)")
    plt.title(f"Top {top_n} Feature Importances")
    logging.info("saving feature importances plot locally")
    png_path = project_root/artifact_dir/f"feature_importance_top{top_n}.png"
    plt.savefig(png_path, dpi=150)
    plt.close()

    return fi




def return_mapping(target:str,prediction:str):

    column_mapping = ColumnMapping()

    column_mapping.target = target
    column_mapping.prediction = prediction
  #  column_mapping.prediction = ''
   # column_mapping.numerical_features = numerical_features


    return column_mapping




def make_shapash_report_html(model, X_focus, y_focus, *, title="Batch explanation"):
    xpl = SmartExplainer()

    # For regression, provide model and data; Shapash will compute contributions.
    xpl.compile(
        x=X_focus,
        model=model,
        y_pred=y_focus,      # optional, but speeds & ensures consistent with your pipeline output
    )

    # Export as HTML report
    # (Depending on Shapash version: to_html / export / save_html — keep whichever your install supports.)
    # Common pattern:
    html = xpl.to_html(title=title)
    return html


def interesting_region_indices(y, slope_eps=1e-4, min_len=120):
    """
    Identify rising -> peak -> falling region.
    Returns (i_start, i_peak, i_end)
    """
    y = np.asarray(y, dtype=float)
    n = len(y)

    if n < 3:
        i_peak = int(np.nanargmax(y))
        return 0, i_peak, n - 1

    i_peak = int(np.nanargmax(y))
    dy = np.diff(y)

    i_start = i_peak
    while i_start > 1 and dy[i_start - 1] > slope_eps:
        i_start -= 1

    i_end = i_peak
    while i_end < n - 2 and dy[i_end] < -slope_eps:
        i_end += 1

    # Enforce minimum window size
    if (i_end - i_start + 1) < min_len:
        half = min_len // 2
        i_start = max(0, i_peak - half)
        i_end = min(n - 1, i_peak + half)

    return i_start, i_peak, i_end


