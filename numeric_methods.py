"""numeric_methods.py
Runs the following forecasting methods over numeric data
- Naive baselines
- Sci-kit Learn regressors
- Neural Forecast neural-network based forecasters

@author Gina Sprint
@date 12/15/23
"""
import os
import argparse
import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import TimesNet, Autoformer, Informer, FEDformer
from neuralforecast.losses.numpy import mae, mse
from neuralforecast.losses.pytorch import MAE

from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

RANDOM_SEED = 1
pd.options.display.float_format = '{:.3f}'.format

# naive baselines
class CopyYesterday():
    def fit(self, X_train, y_train):
        pass

    def predict(self, X_test):
        yesterdays = [X_test.iloc[i, -1] for i in range(len(X_test))]
        return yesterdays

class CopyLastWeek():
    def fit(self, X_train, y_train):
        pass

    def predict(self, X_test):
        last_weeks = []
        for i in range(len(X_test)):
            if len(X_test.iloc[i]) >= 7:
                last_weeks.append(X_test.iloc[i, -7])
            else:
                last_weeks.append(np.nan)
        return last_weeks

class HistoricalAverage():
    def fit(self, X_train, y_train):
        pass
    
    def predict(self, X_test):
        avgs = [X_test.iloc[i].mean() for i in range(len(X_test))]
        return avgs

# neural network-based forecasters
def prepare_for_neuralforecast(fname, score_label):
    df = pd.read_csv(fname, index_col=0)
    if score_label != "all":
        df = df.loc[df.index.str.contains(score_label)]
    df.index.name = "unique_id"
    df = df.reset_index()
    dates = df.columns[df.columns.str.contains("date")]
    cols = ["unique_id"] + list(dates)
    dates_df = df[cols]
    dates_df.set_index("unique_id", inplace=True)
    df = df.drop(dates, axis=1)
    df = df.melt(id_vars=["unique_id"], var_name="ds", value_name="y")
    df = df.sort_values(["unique_id", "ds"]) # sorting by ds here can be an issue because T1, T10, T11 is in sorted order but not the order we want
    for ind in df.index:
        ser = df.loc[ind]
        uid = ser["unique_id"]
        timepoint = ser["ds"]
        timepoint_date = dates_df.loc[uid, f"{timepoint}date"]
        df.loc[ind] = ser.replace(timepoint, timepoint_date)
    df = df.dropna() # because some series are shorter than others and had NaNs for those date labels
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values(["unique_id", "ds"])

    return df

def fit_predict_nf(models, train_df, test_df, y_test):
    nf = NeuralForecast(models=models, freq='M')
    val_size = 2 # this is the horizon of the validation set which is created by windowing
    nf.fit(df=train_df, val_size=val_size)
    pred_df = nf.predict(test_df)
    pred_df = pred_df.merge(y_test, on="unique_id")
    return pred_df

def load_data_for_nf(dataset_dirname, input_size, horizon, score_label):
    train_df = prepare_for_neuralforecast(os.path.join(dataset_dirname, "train_nf.csv"), score_label)
    test_df = prepare_for_neuralforecast(os.path.join(dataset_dirname, "test.csv"), score_label)
    y_test_inds = []
    for i, ind in enumerate(test_df.index):
        if (i + 1) % (input_size + horizon) == 0:
            y_test_inds.append(ind)
    y_test = test_df.loc[y_test_inds]
    test_df = test_df.drop(y_test_inds) # remove y_col
    return train_df, test_df, y_test

def run_nf_exp(dataset_dirname, models, input_size, horizon, score_label):
    train_df, test_df, y_test = load_data_for_nf(dataset_dirname, input_size, horizon, score_label)
    pred_df = fit_predict_nf(models, train_df, test_df, y_test)
    return pred_df

# sci-kit learn based regressors
def load_data_for_sklearn(dataset_dirname, score_label, temp_df):
    train_df = pd.read_csv(os.path.join(dataset_dirname, "train.csv"), index_col=0)
    val_df = pd.read_csv(os.path.join(dataset_dirname, "val.csv"), index_col=0)
    test_df = pd.read_csv(os.path.join(dataset_dirname, "test.csv"), index_col=0).sort_index()

    if score_label != "all":
        train_df = train_df.loc[train_df.index.str.contains(score_label)]
        val_df = val_df.loc[val_df.index.str.contains(score_label)]
        test_df = test_df.loc[test_df.index.str.contains(score_label)]

    timepoint_nums = sorted([int(col[1:]) for col in temp_df.columns if "date" not in col])
    y_col_name = f"T{timepoint_nums[-1]}"
    date_col_names = [f"T{num}date" for num in timepoint_nums]
    y_val = val_df[y_col_name]
    col_names_to_drop = [y_col_name] + date_col_names
    val_df = val_df.drop(col_names_to_drop, axis=1) # remove y_col and dates
    # train_df = pd.concat([train_df, val_df], ignore_index=True).sort_index()
    y_train = train_df[y_col_name]
    train_df = train_df.drop(col_names_to_drop, axis=1) # remove y_col
    y_test = test_df[y_col_name]
    test_df = test_df.drop(col_names_to_drop, axis=1) # remove y_col
    return train_df, y_train, val_df, y_val, test_df, y_test

def fit_predict_sklearn(models, X_train, y_train, X_val, y_val, X_test, y_test):
    preds_dict = {}
    for reg in models:
        if "XGB" in reg.__class__.__name__:
            reg.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        else:
            reg.fit(X_train, y_train)
        preds = reg.predict(X_test)
        model_label = reg.__class__.__name__
        preds_dict[model_label] = preds
    pred_df = pd.DataFrame(preds_dict)
    pred_df["sk_y"] = y_test.values
    return pred_df

def run_sklearn_exp(dataset_dirname, models, score_label, temp_df):
    X_train, y_train, X_val, y_val, X_test, y_test = load_data_for_sklearn(dataset_dirname, score_label, temp_df)
    pred_df = fit_predict_sklearn(models, X_train, y_train, X_val, y_val, X_test, y_test)
    return pred_df

def compute_results(model_list, pred_df):
    res_dicts = {}
    for model in model_list:
        model_label = model.__class__.__name__
        pred_ser = pred_df[model_label]
        pred_ser = pred_ser.apply(lambda x: round(x)) # round to nearest integer
        res_dict = {"MAE": mae(pred_df["y"], pred_ser),
                    "RMSE": np.sqrt(mse(pred_df["y"], pred_ser))}
        for score_label, score_df in pred_df.groupby("score_label"):
            score_pred_ser = score_df[model_label]
            score_pred_ser = score_pred_ser.apply(lambda x: round(x)) # round to nearest integer
            res_dict.update({f"{score_label}_MAE": mae(score_df["y"], score_pred_ser),
                    f"{score_label}_RMSE": np.sqrt(mse(score_df["y"], score_pred_ser))})
        res_dicts[model_label] = res_dict
    res_df = pd.DataFrame(res_dicts)
    return res_df

def setup_and_run(dirname, output_dirname):
    horizon = 1
    input_size = 5
    max_steps = 500
    val_check_steps = 10
    early_stop_patience_steps = 10

    temp_df = pd.read_csv(os.path.join(dirname, "train.csv"), index_col=0)
    timepoint_nums = [int(col[1:]) for col in temp_df.columns if "date" not in col]
    input_size = max(timepoint_nums) - horizon
    nf_models = [
            Autoformer(random_seed=RANDOM_SEED,
                    h=horizon,
                    input_size=input_size,
                    hidden_size = 16,
                    conv_hidden_size = 32,
                    n_head=2,
                    loss=MAE(),
                    scaler_type='robust',
                    learning_rate=1e-3,
                    max_steps=max_steps,
                    val_check_steps=val_check_steps,
                    early_stop_patience_steps=early_stop_patience_steps),
            Informer(random_seed=RANDOM_SEED,
                    h=horizon,
                    input_size=input_size,
                    hidden_size = 16,
                    conv_hidden_size = 32,
                    n_head = 2,
                    loss=MAE(),
                    scaler_type='robust',
                    learning_rate=1e-3,
                    max_steps=max_steps,
                    val_check_steps=val_check_steps,
                    early_stop_patience_steps=early_stop_patience_steps),
            FEDformer(random_seed=RANDOM_SEED,
                    h=horizon,
                    input_size=input_size,
                    hidden_size = 64,
                    conv_hidden_size = 128,
                    n_head = 8,
                    loss=MAE(),
                    scaler_type='robust',
                    learning_rate=1e-3,
                    max_steps=max_steps,
                    batch_size=2,
                    windows_batch_size=32,
                    val_check_steps=val_check_steps,
                    early_stop_patience_steps=early_stop_patience_steps),
            TimesNet(random_seed=RANDOM_SEED,
                    h=horizon,
                    input_size=input_size,
                    max_steps=max_steps,
                    val_check_steps=val_check_steps,
                    early_stop_patience_steps=early_stop_patience_steps,
                    top_k=3)]
    sk_models = [LinearRegression(),
                SVR(),
                XGBRegressor(objective='reg:squarederror', eval_metric=mean_squared_error, early_stopping_rounds=10),
                GradientBoostingRegressor(),
                RandomForestRegressor(),
                CopyYesterday(),
                CopyLastWeek(),
                HistoricalAverage()]

    score_label = "all" # "stress"
    nf_pred_df = run_nf_exp(dirname, nf_models, input_size, horizon, score_label=score_label)
    sk_pred_df = run_sklearn_exp(dirname, sk_models, score_label, temp_df)
    pred_df = pd.concat([nf_pred_df, sk_pred_df], axis=1)
    pred_df["score_label"] = pred_df["unique_id"].apply(lambda x: x.split("_")[-1])
    assert np.allclose(pred_df["y"].values, pred_df["sk_y"].values) # make sure they are in the same order
    pred_df = pred_df.drop("sk_y", axis=1)

    res_df = compute_results(nf_models + sk_models, pred_df)
    if not os.path.exists(output_dirname):
        os.mkdir(output_dirname)
    res_df.to_csv(os.path.join(output_dirname, f"numeric_results.csv"))
    print(res_df)

if __name__ == "__main__":
    # example run
    # python numeric_methods.py daily_ema_numeric daily_ema_numeric_results
    parser = argparse.ArgumentParser(description="Run numeric forecasting methods")
    parser.add_argument("-i",
                        type=str,
                        dest="numeric_directory_name",
                        default="daily_ema_numeric",
                        help="The directory to read the time series from")
    parser.add_argument("-r",
                        type=str,
                        dest="results_directory_name",
                        default="daily_ema_numeric_results",
                        help="The directory to write the results to")
    args = parser.parse_args()

    setup_and_run(args.numeric_directory_name, args.results_directory_name)