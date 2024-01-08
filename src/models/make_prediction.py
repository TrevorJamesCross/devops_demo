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
from mlem.api import load

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

month = prev_date.month
year = prev_date.year

lagged_value = last_row['values']

# create input data
model_input = pd.DataFrame(data=[[lagged_value, month, year]], columns=['lagged_values', 'month', 'year'], index=[prev_date])

# ----------------------------------
# ---Make Prediction & Append CSV---
# ----------------------------------

# predict using pulled model & convert prediction to string
pred = str(model.predict(model_input)[0][0])

# convert date to string
prev_date = str(prev_date.year) + '-' + str(prev_date.month)

# append to CSV
with open(base_path+'/predictions/live_preds.csv', 'a') as preds_file:
    preds_file.write(f"{prev_date},{pred}")
print(f"\nAppended data '{prev_date},{pred}' to CSV path {base_path+'/predictions/live_preds.csv'}")
