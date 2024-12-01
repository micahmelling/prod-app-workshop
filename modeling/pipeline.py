from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.feature_extraction import DictVectorizer

from helpers.clean import fill_missing_values_static, clip_numeric_feature, drop_features
from helpers.engineer import FeaturesToDict, TakeLog
from modeling.config import FEATURES_TO_DROP


def get_pipeline(model):
    """
    Generates a scikit-learn modeling pipeline with model as the final step.
    """
    numeric_transformer = Pipeline(steps=[
        ('age_clipper', FunctionTransformer(clip_numeric_feature, validate=False,
                                            kw_args={'col': 'Age', 'clip_lower': 15, 'clip_upper': 18})),
        ('log_creator', TakeLog(columns=['StudyTimeWeekly'])),
        ('dict_creator', FeaturesToDict()),
        ('dict_vectorizer', DictVectorizer(sparse=False)),
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ('missing_filler', FunctionTransformer(fill_missing_values_static, validate=False,
                                               kw_args={'fill_value': 'unknown'})),
        ('dict_creator', FeaturesToDict()),
        ('dict_vectorizer', DictVectorizer(sparse=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric_transformer', numeric_transformer, selector(dtype_include='number')),
            ('categorical_transformer', categorical_transformer, selector(dtype_exclude='number'))
        ],
        remainder='passthrough',
    )

    pipeline = Pipeline(steps=[
        ('feature_dropper', FunctionTransformer(drop_features, validate=False,
                                                kw_args={'features_to_drop': FEATURES_TO_DROP})),
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    return pipeline
