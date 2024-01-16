"""
DevOps Demo: Visualize Data
Author: Trevor Cross
Last Updated: 01/11/24

Visualize and explore raw data to aid in decision making.
"""

# ----------------------
# ---Import Libraries---
# ----------------------

# import standard libraries
import pandas as pd

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

# read parameters file
params = yaml.safe_load(open("params.yaml"))["visualize"]

# obtain parameters
diff_param = params["differencing_param"]
num_lags = params["num_lags_to_plot"]

# -----------------------
# ---Pull & Split Data---
# -----------------------

# pull data from path
df_daily = pd.read_csv('data/raw/ts_data.csv', parse_dates=[0], index_col=0)
df_monthly = pd.read_csv('data/preprocessed/ts_data_monthly.csv', parse_dates=[0], index_col=0)

# drop lagged values from df_monthly
df_monthly = df_monthly.loc[:, ~df_monthly.columns.str.startswith('lagged_values_')]

# ----------------------------------
# ---Plot Differenced Time Series---
# ----------------------------------

# define base save path for figures
base_path = os.path.join('reports', 'figures')

# plot timeseries
plot_differenced_ts(df_daily,
                    save_path=base_path+'/daily_timeseries.png',
                    differencing_param=diff_param,
                    )

plot_differenced_ts(df_monthly,
                    save_path=base_path+'/monthly_timeseries.png',
                    differencing_param=diff_param,
                    )

# ---------------------
# ---Plot ACF & PACF---
# ---------------------

# plot ACF & PACF
plot_acf(df_daily,
         save_path=base_path+'/daily_acf.png',
         differencing_param=diff_param,
         num_xticks=num_lags
         )

plot_acf(df_monthly,
         save_path=base_path+'/monthly_acf.png',
         differencing_param=diff_param,
         num_xticks=num_lags
         )
