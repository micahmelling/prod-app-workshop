import pandas as pd

from hyperopt import fmin, tpe, Trials, space_eval
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from helpers.utils import save_pipeline, save_cv_scores


def train_model(x_train: pd.DataFrame, y_train: pd.DataFrame, get_pipeline_function: callable, model_uid: str,
                model: BaseEstimator, param_space: dict, iterations: int, cv_strategy: int,
                cv_scoring: str, static_param_space: dict or None = None) -> Pipeline:
    """
    Trains a machine learning model, optimizes the hyperparameters, and saves the serialized model.
    """
    print(f'training {model_uid}...')

    pipeline = get_pipeline_function(model)

    if static_param_space:
        param_space.update(static_param_space)

    cv_scores_df = pd.DataFrame()

    def _model_objective(params):
        pipeline.set_params(**params)
        score = cross_val_score(pipeline, x_train, y_train, cv=cv_strategy, scoring=cv_scoring, n_jobs=-1)

        temp_cv_scores_df = pd.DataFrame(score)
        temp_cv_scores_df = temp_cv_scores_df.reset_index()
        temp_cv_scores_df['index'] = 'fold_' + temp_cv_scores_df['index'].astype(str)
        temp_cv_scores_df = temp_cv_scores_df.T
        temp_cv_scores_df = temp_cv_scores_df.add_prefix('fold_')
        temp_cv_scores_df = temp_cv_scores_df.iloc[1:]
        temp_cv_scores_df['mean'] = temp_cv_scores_df.mean(axis=1)
        temp_cv_scores_df['std'] = temp_cv_scores_df.std(axis=1)
        temp_params_df = pd.DataFrame(params, index=list(range(0, len(params) + 1)))
        temp_cv_scores_df = pd.concat([temp_params_df, temp_cv_scores_df], axis=1)
        temp_cv_scores_df = temp_cv_scores_df.dropna()
        nonlocal cv_scores_df
        cv_scores_df = pd.concat([cv_scores_df, temp_cv_scores_df], axis=0)

        return 1 - score.mean()

    trials = Trials()
    best = fmin(_model_objective, param_space, algo=tpe.suggest, max_evals=iterations, trials=trials)
    best_params = space_eval(param_space, best)

    cv_scores_df = cv_scores_df.sort_values(by=['mean'], ascending=False)
    cv_scores_df = cv_scores_df.reset_index(drop=True)
    cv_scores_df = cv_scores_df.reset_index()
    cv_scores_df = cv_scores_df.rename(columns={'index': 'ranking'})
    save_cv_scores(cv_scores_df, model_uid, 'cv_scores')

    pipeline.set_params(**best_params)
    pipeline.fit(x_train, y_train)
    save_pipeline(pipeline, model_uid, 'model')
    return pipeline
