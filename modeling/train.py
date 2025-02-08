# To make the demo easier, we will try to automatically solve any path issues
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.resolve().parent))

from typing import Tuple, List

import pandas as pd
from sklearn.model_selection import train_test_split

from modeling.config import (model_named_tuple, evaluation_named_tuple, MODEL_TRAINING_LIST, MODEL_EVALUATION_LIST,
                             CV_FOLDS, CV_SCORER, STATIC_PARAM_GRID, TARGET, TEST_SET_PERCENTAGE, DATA_PATH)
from modeling.model import train_model
from modeling.pipeline import get_pipeline
from modeling.evaluate import run_omnibus_model_evaluation
from modeling.explain import produce_shap_values_and_plots
from helpers.utils import create_uid


def read_raw_data(path):
    return pd.read_csv(path)


def create_x_y_split(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    y = df[target]
    x = df.drop(target, axis=1)
    return x, y


def create_train_test_split(x: pd.DataFrame, y: pd.DataFrame, test_size: float = 0.25) -> (
        Tuple)[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
    return x_train, x_test, y_train, y_test


def create_training_and_testing_data(df: pd.DataFrame, target: str, test_set_percentage: float = 0.25) -> (
        Tuple)[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x, y = create_x_y_split(df, target)
    x_train, x_test, y_train, y_test = create_train_test_split(x, y, test_set_percentage)
    return x_train, x_test, y_train, y_test


def train_evaluate_explain_model(x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.DataFrame,
                                 y_test: pd.DataFrame, model_training_list: List[model_named_tuple],
                                 cv_strategy: int, cv_scoring: str, static_param_space: dict,
                                 evaluation_list: List[evaluation_named_tuple]) -> None:
    for model in model_training_list:
        model_uid = create_uid(base_string=model.model_name)
        best_pipeline = train_model(x_train=x_train, y_train=y_train, get_pipeline_function=get_pipeline,
                                    model_uid=model_uid, model=model.model, param_space=model.param_space,
                                    iterations=model.iterations, cv_strategy=cv_strategy, cv_scoring=cv_scoring,
                                    static_param_space=static_param_space)
        run_omnibus_model_evaluation(estimator=best_pipeline, x_df=x_test, target=y_test, model_uid=model_uid,
                                     evaluation_list=evaluation_list)
        produce_shap_values_and_plots(pipeline=best_pipeline, x_df=x_test, model_uid=model_uid)


def main(data_path: str, target: str, test_set_percentage: float, model_training_list: List[model_named_tuple],
         cv_strategy: int, cv_scoring: str, static_param_space: dict,
         evaluation_list: List[evaluation_named_tuple]) -> None:
    student_df = read_raw_data(data_path)
    x_train, x_test, y_train, y_test = create_training_and_testing_data(
        student_df,
        target=target,
        test_set_percentage=test_set_percentage
    )
    train_evaluate_explain_model(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        model_training_list=model_training_list,
        cv_strategy=cv_strategy,
        cv_scoring=cv_scoring,
        static_param_space=static_param_space,
        evaluation_list=evaluation_list
    )


if __name__ == "__main__":
    main(
        data_path=DATA_PATH,
        target=TARGET,
        test_set_percentage=TEST_SET_PERCENTAGE,
        model_training_list=MODEL_TRAINING_LIST,
        cv_strategy=CV_FOLDS,
        cv_scoring=CV_SCORER,
        static_param_space=STATIC_PARAM_GRID,
        evaluation_list=MODEL_EVALUATION_LIST
    )
