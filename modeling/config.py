from collections import namedtuple

from hyperopt import hp
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (log_loss, balanced_accuracy_score, roc_auc_score)


FEATURES_TO_DROP = ['StudentID', 'GPA']
TARGET = 'gpa_b_and_up'
DATA_PATH = 'data/raw/student_performance.csv'
CV_SCORER = 'neg_mean_squared_error'
CV_FOLDS = 5
TEST_SET_PERCENTAGE = 0.25
STATIC_PARAM_GRID = None


FOREST_PARAM_GRID = {
    'model__estimator__max_depth': hp.uniformint('model__max_depth', 3, 16),
    'model__estimator__min_samples_leaf': hp.uniform('model__min_samples_leaf', 0.001, 0.01),
    'model__estimator__max_features': hp.choice('model__max_features', ['log2', 'sqrt']),
}

GBOOST_PARAM_GRID = {
    'model__estimator__learning_rate': hp.uniform('model__learning_ratee', 0.01, 0.5),
    'model__estimator__max_iter': hp.randint('model__n_estimators', 75, 150),
    'model__estimator__max_depth': hp.randint('model__max_depth', 3, 16),
    'model__estimator__min_samples_leaf': hp.choice('model__min_samples_leaf', [10, 20]),
}


model_named_tuple = namedtuple('model_config', {'model_name', 'model', 'param_space', 'iterations'})
MODEL_TRAINING_LIST = [
    model_named_tuple(model_name='random_forest', model=CalibratedClassifierCV(RandomForestClassifier(n_estimators=500)),
                      param_space=FOREST_PARAM_GRID, iterations=25),
    model_named_tuple(model_name='hist_gb', model=CalibratedClassifierCV(HistGradientBoostingClassifier()),
                      param_space=GBOOST_PARAM_GRID, iterations=25),
]


evaluation_named_tuple = namedtuple('model_evaluation', {'scorer_callable', 'metric_name',
                                                         'evaluation_column'})
MODEL_EVALUATION_LIST = [
    evaluation_named_tuple(scorer_callable=balanced_accuracy_score, metric_name='balanced_accuracy',
                           evaluation_column='predicted_class'),
    evaluation_named_tuple(scorer_callable=roc_auc_score, metric_name='mae', evaluation_column='1_prob'),
    evaluation_named_tuple(scorer_callable=log_loss, metric_name='log_loss', evaluation_column='1_prob'),
]
