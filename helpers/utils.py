import os
import datetime

import pytz
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline


def make_directories_if_not_exists(directories_list: list) -> None:
    """
    Makes directories in the current working directory if they do not exist.
    """
    for directory in directories_list:
        if not os.path.exists(directory):
            os.makedirs(directory)


def save_pipeline(pipeline: Pipeline, model_uid: str, subdirectory: str) -> None:
    """
    Saves a modeling pipeline locally as a pkl file into the model_uid's directory.
    """
    save_directory = os.path.join('modeling', 'model_results', model_uid, subdirectory)
    make_directories_if_not_exists([save_directory])
    joblib.dump(pipeline, os.path.join(save_directory, 'model.pkl'), compress=9)


def save_cv_scores(df: pd.DataFrame, model_uid: str, subdirectory: str) -> None:
    """
    Saves cross validation scores locally as a csv file into the model_uid's directory.
    """
    save_directory = os.path.join('modeling', 'model_results', model_uid, subdirectory)
    make_directories_if_not_exists([save_directory])
    df.to_csv(os.path.join(save_directory, 'cv_scores.csv'), index=False)


def create_uid(base_string):
    """
    Creates a UID by concatenating the current timestamp to base_string.

    :param base_string: the base string
    :returns: unique string
    """
    tz = pytz.timezone('US/Central')
    now = str(datetime.datetime.now(tz))
    now = now.replace(' ', '').replace(':', '').replace('.', '').replace('-', '')
    uid = base_string + '_' + now
    return uid

