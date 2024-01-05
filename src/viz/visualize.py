"""
DevOps Demo: Visualize Data
Author: Trevor Cross
Last Updated: 01/05/24

Visualize and explore raw data to aid in decision making.
"""

# ----------------------
# ---Import Libraries---
# ----------------------

# import standard libraries
import pandas as pd

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
base_path = f"{expanduser('~')}/projects/devops_demo/data"

# pull data from path
df_daily = pd.read_csv(base_path+'/raw/ts_data.csv', parse_dates=[0], index_col=0)
df_monthly = pd.read_csv(base_path+'/preprocessed/ts_data_monthly.csv', parse_dates=[0], index_col=0)

# drop lagged values from df_monthly
df_monthly.drop(columns=['lagged_values'], inplace=True)

# ----------------------------------
# ---Plot Differenced Time Series---
# ----------------------------------

# define base save path for figures
base_path = f"{expanduser('~')}/projects/devops_demo/reports/figures"

# plot timeseries
plot_differenced_ts(df_daily,
                    save_path=base_path+'/daily_timeseries.png',
                    differencing_param=1,
                    )

plot_differenced_ts(df_monthly,
                    save_path=base_path+'/monthly_timeseries.png',
                    differencing_param=1,
                    )

# ---------------------
# ---Plot ACF & PACF---
# ---------------------

# plot ACF & PACF
plot_acf(df_daily,
         save_path=base_path+'/daily_acf.png',
         differencing_param=1,
         num_xticks=25
         )

plot_acf(df_monthly,
         save_path=base_path+'/monthly_acf.png',
         differencing_param=1,
         num_xticks=25
         )
