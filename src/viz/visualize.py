"""
DevOps Demo: Visualize Data
Author: Trevor Cross
Last Updated: 01/02/24

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
data_path = f"{expanduser('~')}/projects/devops_demo/data/raw/ts_data.csv"

# pull data from path
df = pd.read_csv(data_path, index_col=0)

# ------------------------------
# ---Plot Example Time Series---
# ------------------------------

# plot timeseries
base_path = f"{expanduser('~')}/projects/devops_demo/reports/figures/"
save_path = base_path + f"timeseries.png"

plot_differenced_ts(df,
                    save_path=save_path,
                    differencing_param=1,
                    )

# ---------------------
# ---Plot ACF & PACF---
# ---------------------

# plot ACF & PACF
save_path = base_path + f"autocorrelation.png"
plot_acf(df,
         save_path=save_path,
         differencing_param=1,
         num_xticks=25
         )
