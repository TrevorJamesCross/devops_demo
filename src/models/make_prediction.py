"""
DevOps Demo: Make Prediction
Author: Trevor Cross
Last Updated: 01/08/24

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
from os.path import expanduser

# import toolbox
sys.path.append(f"{expanduser('~')}/projects/devops_demo/src")
from toolbox import *

# --------------------------------------
# ---Pull Model & Obtain Model Inputs---
# --------------------------------------

# define model path
model_path = f"{expanduser('~')}/projects/devops_demo/artifacts/lr_model"

# pull model
model = load(model_path)

# define base data path
base_path = f"{expanduser('~')}/projects/devops_demo/data"

# pull last row of processed data
last_row  = pd.read_csv(base_path+'/preprocessed/ts_data_monthly.csv', parse_dates=[0], index_col=0).iloc[-1]

# obtain model inputs
prev_date = last_row.name

month = int(prev_date.month) + 1
year = int(prev_date.year)
if month == 13:
    month = 1
    year += 1

lagged_value = last_row['values']

# create input data
model_input = pd.DataFrame(data=[[lagged_value, month, year]], columns=['lagged_values', 'month', 'year'])

# ----------------------------------
# ---Make Prediction & Append CSV---
# ----------------------------------

# predict using pulled model & convert prediction to string
pred = model.predict(model_input)[0][0]

# convert date to string
date = str(year) + '-' + str(month)

# read live predictions CSV
live_df = pd.read_csv(base_path+'/predictions/live_preds.csv', parse_dates=[0], index_col=0)

# replace current prediction, or append if no matching index
try:
    live_df.loc[date, 'prediction'] = pred
    print(f"\nReplaced index {date} with new prediction within CSV path {base_path+'/predictions/live_preds.csv'}")
except KeyError:
    new_row = pd.DataFrame(data={'prediction': pred}, index=[date])
    live_df = pd.concat([live_df, new_row], axis=0)
    print(f"\nAppended index {date} to CSV path {base_path+'/predictions/live_preds.csv'}")

# export live_df
save(live_df, base_path+'/predictions/live_preds.csv')
