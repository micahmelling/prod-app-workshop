import os
from typing import List, Union

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.base import RegressorMixin
from sklearn.metrics import roc_curve, auc
from sklearn.pipeline import Pipeline

from helpers.utils import make_directories_if_not_exists
from modeling.config import evaluation_named_tuple


def make_predict_vs_actual_dataframe(estimator: Union[Pipeline, RegressorMixin], x_df: pd.DataFrame,
                                     target: Union[pd.Series, pd.DataFrame], class_cutoff: float = 0.50) -> pd.DataFrame:
    """
    Creates a dataframe of predictions vs. actuals.
    """
    target.name = 'target'
    df = pd.concat(
        [
            pd.DataFrame(estimator.predict_proba(x_df), columns=['0_prob', '1_prob']),
            target.reset_index(drop=True)
        ],
        axis=1)
    df['predicted_class'] = np.where(df['1_prob'] >= class_cutoff, 1, 0)
    df = df[['predicted_class'] + [col for col in df.columns if col != 'predicted_class']]
    return df


def make_full_predictions_dataframe(estimator: Union[Pipeline, RegressorMixin], model_uid: str, x_df: pd.DataFrame,
                                    target: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
    """
    Produces a dataframe consisting of a point estimate, a lower bound, an upper bound, and the actual value.
    """
    df = make_predict_vs_actual_dataframe(estimator, x_df, target)
    df = df[[target.name, 'predicted_class', '0_prob', '1_prob']]
    x_df = x_df.reset_index(drop=True)
    df = pd.concat([df, x_df], axis=1)
    save_path = os.path.join('modeling', 'model_results', model_uid, 'diagnostics', 'predictions')
    make_directories_if_not_exists([save_path])
    df.to_csv(os.path.join(save_path, 'predictions_vs_actuals.csv'), index=False)
    return df


def _evaluate_model(target: Union[pd.Series, pd.DataFrame], prediction_series: pd.Series, scorer: callable,
                    metric_name: str) -> pd.DataFrame:
    """
    Applies a scorer function to evaluate predictions against the ground-truth labels.
    """
    score = scorer(target, prediction_series)
    df = pd.DataFrame({metric_name: [score]})
    return df


def run_and_save_evaluation_metrics(predictions_df: pd.DataFrame, model_uid: str,
                                    evaluation_list: List[evaluation_named_tuple]) -> None:
    """
    Runs a series of evaluations metrics on a model's predictions and writes the results locally.
    """
    main_df = pd.DataFrame()
    for evaluation_config in evaluation_list:
        temp_df = _evaluate_model(predictions_df['target'], predictions_df[evaluation_config.evaluation_column],
                                  evaluation_config.scorer_callable, evaluation_config.metric_name)
        main_df = pd.concat([main_df, temp_df], axis=1)
    main_df = main_df.T
    main_df.reset_index(inplace=True)
    main_df.columns = ['scoring_metric', 'holdout_score']
    save_path = os.path.join('modeling', 'model_results', model_uid, 'diagnostics', 'evaluation_files')
    make_directories_if_not_exists([save_path])
    main_df.to_csv(os.path.join(save_path, 'evaluation_scores.csv'), index=False)


def plot_calibration_curve(predictions_df: pd.DataFrame, n_bins: int,
                           bin_strategy: str, model_uid: str) -> None:
    """
    Produces a calibration plot and saves it locally. The raw data behind the plot is also written locally.
    """
    try:
        prob_true, prob_pred = calibration_curve(predictions_df['target'], predictions_df['1_prob'], n_bins=n_bins,
                                                 strategy=bin_strategy)
        fig, ax = plt.subplots()
        plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label='model')
        line = mlines.Line2D([0, 1], [0, 1], color='black')
        transform = ax.transAxes
        line.set_transform(transform)
        ax.add_line(line)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xticks(np.arange(0, 1.1, 0.1))
        plt.yticks(np.arange(0, 1.1, 0.1))
        fig.suptitle(f' {bin_strategy.title()} Calibration Plot {n_bins} Requested Bins')
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('True Probability in Each Bin')
        plt.legend()
        plt.savefig(os.path.join('modeling', 'model_results', model_uid, 'diagnostics', 'evaluation_plots',
                                 f'{bin_strategy}_{n_bins}_calibration_plot.png'))
        plt.clf()
        calibration_df = pd.DataFrame({'prob_true': prob_true, 'prob_pred': prob_pred})
        calibration_df.to_csv(os.path.join('modeling', 'model_results', model_uid, 'diagnostics', 'evaluation_files',
                                           f'{bin_strategy}_{n_bins}_calibration_summary.csv'), index=False)
    except Exception as e:
        print(e)


def calculate_probability_lift(predictions_df: pd.DataFrame, model_uid: str, n_bins: int = 10) -> None:
    """
    Calculates the lift provided by the probability estimates. Lift is determined by how much improvement is experienced
    by using the predicted probabilities over assuming that each observation has the same probability of being in the
    positive class (i.e. applying the overall rate of occurrence of the positive class to all observations).

    This process takes the following steps:
    - find the overall rate of occurrence of the positive class
    - cut the probability estimates into n_bins
    - for each bin, calculate:
       - the average predicted probability
       - the actual probability
    -  for each bin, calculate
       - the difference between the average predicted probability and the true probability
       - the difference between the overall rate of occurrence and the true probability
    - take the sum of the absolute value for each the differences calculated in the previous step
    - take the ratio of the two sums, with the base rate sum as the numerator

    Values above 1 indicate the predicted probabilities have lift over simply assuming each observation has the same
    probability.
    """
    target = predictions_df['target'].reset_index(drop=True)
    prediction_series = predictions_df['1_prob'].reset_index(drop=True)
    df = pd.concat([target, prediction_series], axis=1)
    columns = list(df)
    class_col = columns[0]
    proba_col = columns[1]
    base_rate = df[class_col].mean()

    df['1_prob_bin'] = pd.qcut(df[proba_col], q=n_bins, labels=list(range(1, n_bins + 1)))
    grouped_df = df.groupby('1_prob_bin', observed=False).agg({proba_col: 'mean', class_col: 'mean'})
    grouped_df.reset_index(inplace=True)
    grouped_df['1_prob_diff'] = grouped_df[proba_col] - grouped_df[class_col]
    grouped_df['base_rate_diff'] = base_rate - grouped_df[class_col]

    prob_diff = grouped_df['1_prob_diff'].abs().sum()
    base_rate_diff = grouped_df['base_rate_diff'].abs().sum()
    lift = base_rate_diff / prob_diff
    pd.DataFrame({'lift': [lift]}).to_csv(os.path.join('modeling', 'model_results', model_uid, 'diagnostics',
                                                       'evaluation_files', 'proba_lift.csv'), index=False)
    return lift


def plot_roc_auc(predictions_df: pd.DataFrame, model_uid: str, pos_label: int = 1) -> None:
    """
    Plots ROC curve.
    """
    fpr, tpr, _ = roc_curve(predictions_df['target'], predictions_df['1_prob'], pos_label=pos_label)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig(os.path.join('modeling', 'model_results', model_uid, 'diagnostics', 'evaluation_plots', 'roc_curve.png'))
    plt.clf()


def run_omnibus_model_evaluation(estimator: Union[Pipeline, RegressorMixin],
                                 x_df: pd.DataFrame, target: Union[pd.Series, pd.DataFrame], model_uid: str,
                                 evaluation_list: List[evaluation_named_tuple]) -> None:
    """
    Runs a series of model evaluation techniques. Namely, providing scores of various metrics on the entire dataset
    and on segments of the dataset.
    """
    make_directories_if_not_exists([
        os.path.join('modeling', 'model_results', model_uid, 'diagnostics', 'predictions'),
        os.path.join('modeling', 'model_results', model_uid, 'diagnostics', 'evaluation_files'),
        os.path.join('modeling', 'model_results', model_uid, 'diagnostics', 'evaluation_plots'),
    ])
    predictions_df = make_full_predictions_dataframe(estimator, model_uid, x_df, target)
    run_and_save_evaluation_metrics(predictions_df, model_uid, evaluation_list)
    plot_calibration_curve(predictions_df, 10, 'uniform', model_uid)
    plot_calibration_curve(predictions_df, 10, 'quantile', model_uid)
    calculate_probability_lift(predictions_df, model_uid)
    plot_roc_auc(predictions_df, model_uid)
