from copy import deepcopy
from typing import Union

import pandas as pd
from sklearn.pipeline import Pipeline


def fill_missing_values_static(df: pd.DataFrame, fill_value: str) -> pd.DataFrame:
    """
    Fills all missing values in a dataframe with fill_value.
    """
    return df.fillna(value=fill_value)


def clip_numeric_feature(df: pd.DataFrame, col: str, clip_lower: Union[int, float],
                         clip_upper: Union[int, float]) -> pd.DataFrame:
    """Clips the upper and lower bounds of a numeric feature."""
    df[col] = df[col].clip(lower=-clip_lower, upper=clip_upper)
    return df


def drop_features(df: pd.DataFrame, features_to_drop: list) -> pd.DataFrame:
    """Drops a list of features."""
    return df.drop(features_to_drop, axis=1, errors='ignore')


def transform_data_with_pipeline(pipeline: Pipeline, x_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms x_df with the pre-processing steps defined in the pipeline.
    """
    pipeline_ = deepcopy(pipeline)
    pipeline_.steps.pop(len(pipeline_) - 1)
    x_df = pipeline_.transform(x_df)

    num_features = pipeline_.named_steps['preprocessor'].named_transformers_.get('numeric_transformer').named_steps[
        'dict_vectorizer'].feature_names_
    cat_features = pipeline_.named_steps['preprocessor'].named_transformers_.get('categorical_transformer').named_steps[
        'dict_vectorizer'].feature_names_

    x_df = pd.DataFrame(x_df, columns=num_features + cat_features)

    return x_df
