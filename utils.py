"""utils.py
Shared utility functions for regression evaluation.

@author Gina Sprint
@date 12/15/23
"""
import numpy as np
import scipy.stats as stats
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

def compute_reg_metrics(actual_scores, predicted_scores, label):
    result_dict = {}
    result_dict[f"{label}_missing_rate%"] = predicted_scores.isnull().sum() / len(predicted_scores) * 100.0
    non_nan_indexes = predicted_scores[~predicted_scores.isnull()].index
    predicted_scores = predicted_scores.loc[non_nan_indexes]
    actual_scores = actual_scores.loc[non_nan_indexes]
    if len(actual_scores) >= 2:
        if label in ["", "overall"]:
            r, r_pval = stats.pearsonr(actual_scores, predicted_scores)
            result_dict[f"{label}_r"] = r
            result_dict[f"{label}_r_pval"] = r_pval
        mae = mean_absolute_error(actual_scores, predicted_scores)
        result_dict[f"{label}_MAE"] = mae
        rmse = np.sqrt(mean_squared_error(actual_scores, predicted_scores))
        result_dict[f"{label}_RMSE"] = rmse

    result_dict[f"{label}_#"] = len(actual_scores)
    return result_dict

def compute_metrics_per_score_label(df):
    result_dict = compute_reg_metrics(df["actual_score"], df["predicted_score"], "")
    all_result_dict = {"overall": result_dict}

    for score_label, score_df in df.groupby("score_label"):
        result_dict = compute_reg_metrics(score_df["actual_score"], score_df["predicted_score"], "")
        all_result_dict[score_label] = result_dict

    result_df = pd.DataFrame(all_result_dict).T
    return result_df