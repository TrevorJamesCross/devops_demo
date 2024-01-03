"""
DevOps Demo: Train Models
Author: Trevor Cross
Last Updated: 01/02/24

Train multiple ARIMA models using statsmodels libraries.
"""

# ----------------------
# ---Import Libraries---
# ----------------------

# import standard libraries
import pandas as pd
import numpy as np

# import statsmodels libraries
from statsmodels.tsa.arima.model import ARIMA

# import support libraries
import sys
from os.path import expanduser

# import toolbox
sys.path.append(f"{expanduser('~')}/projects/devops_demo/src")
from toolbox import *

# define random state
rs = 81

# -----------------------
# ---Pull & Split Data---
# -----------------------

# define data path
data_path = f"{expanduser('~')}/projects/devops_demo/data/raw/product_sales_data.csv"

# pull data from path
df = pd.read_csv(data_path)
print(df)
raise
# generate train/test splits by product id
splits_gen = split_by_cat_generator(df, 'prod_id', test_size=0.2, shuffle=False, rs=rs)

# ---------------------------
# ---Define & Train Models---
# ---------------------------

# iterate examples
for exam_num in range(3):

    # get example timeseries split & plotting info
    trn_df, tst_df = next(splits_gen)
    prod_id = trn_df['prod_id'].iloc[0]

    # transform dfs
    trn_df.index = trn_df['sale_date']
    trn_df = trn_df['units_sold']

    tst_df.index = tst_df['sale_date']
    tst_df = tst_df['units_sold']

    # define & fit arima model
    model = ARIMA(trn_df, order=(2, 1, 2))
    results = model.fit()

    # get model forecast on test data
    forecast_steps = len(tst_df)
    forecast = results.get_forecast(steps=forecast_steps)

    # plot timeseries & forecast
    plot_forecast(trn_df, tst_df, forecast)
    raise
# ---------------------------------------
# ---Evaluate & Plot Model Predictions---
# ---------------------------------------

# -------------------------
# ---Save Models & Plots---
# -------------------------
