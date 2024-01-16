"""
DevOps Demo: Preprocess Data
Author: Trevor Cross
Last Updated: 01/11/24

Preprocess raw (synthesized) data by aggregating it to the monthly level & appending
lagged values.
"""

# ----------------------
# ---Import Libraries---
# ----------------------

# import standard libraries
import pandas as pd

# import iterative.ai libraries
from mlem.api import save

# import support libraries
import os
import yaml

# --------------------------
# ---Read Parameters File---
# --------------------------

# read parameters file
params = yaml.safe_load(open("params.yaml"))["preprocess"]

# obtain parameters
num_lagged_feats = params["num_lagged_features"]

# --------------------
# ---Pull Raw Data---
# -------------------

# define data path
data_path = os.path.join('data', 'raw', 'ts_data.csv')

# pull data from path
df = pd.read_csv(data_path, parse_dates=[0], index_col=0)

# ---------------------
# ---Preprocess Data---
# ---------------------

# aggregate data to monthly level
df['month'] = df.index.month
df['year'] = df.index.year
df_monthly_agg = df.groupby(['year', 'month']).mean()
df_monthly_agg.index = df_monthly_agg.index.get_level_values('year').astype(str) + '-' + df_monthly_agg.index.get_level_values('month').astype(str)

# append lagged values columns
for lag_num in range(num_lagged_feats):
    df_monthly_agg[f"lagged_values_{lag_num+1}"] = df_monthly_agg['values'].shift(lag_num+1)

# ------------------------------
# ---Export Preprocessed Data---
# ------------------------------

# save df_monthly_agg using mlem
output_path = os.path.join('data', 'preprocessed', 'ts_data_monthly.csv')
save(df_monthly_agg, output_path)
print(f"\nSaved aggregated time series data to path {output_path}")
