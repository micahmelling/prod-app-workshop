import os
from copy import deepcopy

import pandas as pd
from auto_shap.auto_shap import produce_shap_values_and_summary_plots
from sklearn.pipeline import Pipeline

from helpers.utils import make_directories_if_not_exists
from helpers.clean import transform_data_with_pipeline


def produce_shap_values_and_plots(pipeline: Pipeline, x_df: pd.DataFrame, model_uid: str) -> None:
    """
    Produces SHAP values and plots using the auto-shap wrapper library.
    """
    save_path = os.path.join(
        "modeling", "model_results", model_uid, "diagnostics", "shap"
    )
    pipeline_ = deepcopy(pipeline)
    model = pipeline_.named_steps["model"]
    x_df = transform_data_with_pipeline(pipeline_, x_df)
    make_directories_if_not_exists([save_path])
    produce_shap_values_and_summary_plots(model, x_df, save_path)
