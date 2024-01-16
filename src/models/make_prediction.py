"""
DevOps Demo: Make Prediction
Author: Trevor Cross
Last Updated: 01/11/24

Pull model, make next step prediction, and append prediction to local CSV.
"""

# ----------------------
# ---Import Libraries---
# ----------------------

# import standard libraries
import pandas as pd
import numpy as np

# import iterative.ai libraries
from mlem.api import save, load

# import support libraries
import sys
import os
import yaml

# import toolbox
sys.path.append("src")
from toolbox import *

# --------------------------
# ---Read Parameters File---
# --------------------------

# read parameters file to get number of lagged values
params = yaml.safe_load(open("params.yaml"))["preprocess"]

# obtain parameters
num_lagged_feats = params["num_lagged_features"]

# --------------------------------------
# ---Pull Model & Obtain Model Inputs---
# --------------------------------------

# pull model
model = load('models/lr_model')

# pull last row of processed data
last_rows  = pd.read_csv('data/preprocessed/ts_data_monthly.csv', parse_dates=[0], index_col=0).tail(num_lagged_feats)

# obtain model inputs
lagged_values = last_rows['values'].tolist()
lagged_values.reverse()

prev_date = last_rows.iloc[-1].name
month = int(prev_date.month) + 1
year = int(prev_date.year)
if month == 13:
    month = 1
    year += 1

# create input data
data = lagged_values + [month, year]
col_names = last_rows.columns[last_rows.columns.str.startswith('lagged_values_')].tolist() + ['month', 'year']
model_input = pd.DataFrame(data=[data], columns=col_names)

# ----------------------------------
# ---Make Prediction & Append CSV---
# ----------------------------------

# predict using pulled model & convert prediction to string
pred = model.predict(model_input)[0][0]

# convert date to string
date = str(year) + '-' + str(month)

# read live predictions CSV, or create one if not exists
if os.path.exists('models/live_preds.csv'):
    live_df = pd.read_csv('models/live_preds.csv', parse_dates=[0], index_col=0)
else:
    print(f"\nCreating new file path models/live_preds.csv")
    with open('models/live_preds.csv', 'w') as file:
        file.write(",predictions")
    live_df = pd.read_csv('models/live_preds.csv', parse_dates=[0], index_col=0)

# replace current prediction, or append if no matching index
try:
    live_df.loc[date, 'predictions'] = pred
    print(f"\nReplaced index {date} with new prediction within CSV path models/live_preds.csv")
except KeyError:
    new_row = pd.DataFrame(data={'predictions': pred}, index=[date])
    live_df = pd.concat([live_df, new_row], axis=0)
    print(f"\nAppended index {date} to CSV path models/live_preds.csv")

# export live_df
save(live_df, 'models/live_preds.csv')
