"""
DevOps Demo: Train Models
Author: Trevor Cross
Last Updated: 12/24/23

Train multiple ARIMA models on each product id using statsmodels libraries.
"""

# ----------------------
# ---Import Libraries---
# ----------------------

# import standard libraries
import pandas as pd
import numpy as np

# import statsmodels libraries
from statsmodels.tsa.statespace.sarimax import SARIMAX

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

# generate train/test splits by product id
splits_gen = split_by_cat_generator(df, 'prod_id', test_size=0.2, shuffle=False, rs=rs)

# ---------------------------
# ---Define & Train Models---
# ---------------------------

# define ARIMA parameters as dict
arima_params = {}

# define models
splits = next(splits_gen)
trn_split_df = splits[0][['sale_date', 'units_sold']]
trn_split_df.index = trn_split_df['sale_date']
del trn_split_df['sale_date']

arima_model = SARIMAX(trn_split_df, trend='c', order=(1,1,1))
arima_model.fit(disp=False)

# ---------------------------------------
# ---Evaluate & Plot Model Predictions---
# ---------------------------------------

# -------------------------
# ---Save Models & Plots---
# -------------------------
