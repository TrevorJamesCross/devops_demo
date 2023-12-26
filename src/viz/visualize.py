"""
DevOps Demo: Visualize Data
Author: Trevor Cross
Last Updated: 12/24/23

Visualize and explore raw data to aid in decision making.
"""

# ----------------------
# ---Import Libraries---
# ----------------------

# import standard libraries
import pandas as pd
import numpy as np

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

# ------------------------------
# ---Plot Example Time Series---
# ------------------------------

# get example time series split & plotting info
trn_df, tst_df = next(splits_gen)
prod_id = trn_df['prod_id'].loc[0]

# transform dfs
trn_df.index = trn_df['sale_date']
trn_df = trn_df['units_sold']

tst_df.index = tst_df['sale_date']
tst_df = tst_df['units_sold']

# plot timeseries
save_path = f"{expanduser('~')}/projects/devops_demo/reports/figures/example_ts.png"
plot_differenced_ts(trn_df, save_path=save_path, differencing_param=2, title=prod_id)
