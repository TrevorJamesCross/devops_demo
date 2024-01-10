"""
DevOps Demo: Train Model
Author: Trevor Cross
Last Updated: 01/05/24

Train & evaluate linear regression model on preprocessed (aggregated w/ lag) data.
"""

# ----------------------
# ---Import Libraries---
# ----------------------

# import standard libraries
import pandas as pd
import numpy as np

# imprt sklearn libraries
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression

# import model saving libraries
from mlem.api import save

# import support libraries
import sys
from os.path import expanduser

# import toolbox
sys.path.append(f"{expanduser('~')}/projects/devops_demo/src")
from toolbox import *

# define random state
rs = 81

# -------------------------------
# ---Pull Data & Define Splits---
# -------------------------------

# define data path
data_path = f"{expanduser('~')}/projects/devops_demo/data/preprocessed/ts_data_monthly.csv"

# pull data from path
df = pd.read_csv(data_path, parse_dates=[0], index_col=0)

# remove rows with nan
df.dropna(inplace=True)

# create cols for month & year
df['month'] = df.index.month
df['year'] = df.index.year

# pop target values
target_values = pd.DataFrame(df.pop('values'))

# define timeseries splits
forecast_size = 1
tscv = TimeSeriesSplit(n_splits=20, test_size=forecast_size)

# ----------------------------
# ---Train & Evaluate Model---
# ----------------------------

# define list to hold true & predicted values
true_vals = list()
pred_vals = list()

# define list of dicts to hold model evaluation metrics
metrics_list = list()

# iterate splits
for trn_ind, tst_ind in tscv.split(df):

    # get features
    trn_features = df.iloc[trn_ind]
    tst_features = df.iloc[tst_ind]

    # get target values
    trn_targets = target_values.iloc[trn_ind]
    tst_targets = target_values.iloc[tst_ind]

    # define & fit model
    model = LinearRegression()
    model.fit(trn_features, trn_targets)

    # predict test values
    preds = pd.DataFrame(model.predict(tst_features), index=tst_targets.index, columns=['values'])

    # append true and predicted values to lists
    true_vals.append(tst_targets)
    pred_vals.append(preds)

    # evaluate model predictions
    metrics_list.append(eval_metrics(tst_targets.values, preds.values))

# aggregate metrics
avg_metrics = dict()
key_metrics = metrics_list[0].keys()
for key in key_metrics:
    vals = [dict_[key] for dict_ in metrics_list]
    avg_val = sum(vals)/len(vals)
    avg_metrics[key] = avg_val

# -----------------------------------
# ---Plot & Save Model Predictions---
# -----------------------------------

# define save path
save_path = f"{expanduser('~')}/projects/devops_demo/reports/figures/validation_forecasts.png"

# plot forecasts
plot_forecast_cv(target_values, true_vals, pred_vals, save_path=save_path)

# --------------------------
# ---Save Model & Metrics---
# --------------------------

# refit model on all data
model = LinearRegression()
model.fit(df, target_values)

# define save path
base_path = f"{expanduser('~')}/projects/devops_demo/models"

# save model using MLEM
model_save_path = base_path+'/lr_model'
save(model, model_save_path, sample_data=df)
print(f"\nSaved model {model} to path {model_save_path}")

# save model metrics
metrics_save_path = base_path+'/metrics.json'
dict_to_json(avg_metrics, metrics_save_path)
print(f"\nSaved model metrics to path {metrics_save_path}")
