"""
DevOps Demo: Synthesize Data
Author: Trevor Cross
Last Updated: 01/01/24

Create time series data using a gaussian process, & adding white noise & autoregressive
effects.
"""

#%% ----------------------
#-- ---Import Libraries---
#-- ----------------------

# import standard libraries
import numpy as np
import pandas as pd

# import datetime libraries
import datetime

# import model libraries
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from statsmodels.tsa.ar_model import AutoReg

# import plotting libraries
import matplotlib.pyplot as plt

# import iterative.ai libraries
from mlem.api import save

# import support libraries
import random as rand
from os.path import expanduser

# set random seed
rs = 81
rand.seed(rs)

#-- -----------------------------
#-- ---Create Gaussian Process---
#-- -----------------------------

# define function to represent target values for Gaussian Process
def target_func(X):
    func = np.cos(X) + 2*np.log(X + 1)
    return func.flatten()

# define data & obtain function values
num_days = 180
rng = np.random.default_rng()
gp_data = np.sort(rng.choice(num_days, size=num_days//3, replace=False))
gp_target = target_func(gp_data)

# define the gaussian process kernel
gp_kernel = Matern()

# define gaussian process regression object & fit
gpr = GaussianProcessRegressor(kernel=gp_kernel)
gpr.fit(gp_data.reshape(-1, 1), gp_target.reshape(-1, 1))

#-- ---------------------------------------
#-- ---Define Dates & Obtain Predictions---
#-- ---------------------------------------

# define range of dates
date_delta = datetime.timedelta(days=num_days)
date_end = datetime.date.today()
date_start = date_end - date_delta

step_len = datetime.timedelta(days=1)
date_range = [date_start + step_num*step_len for step_num in range(date_delta.days)]

# obtain predictions from gaussian process model
mean_targets = gpr.predict(np.arange(0, num_days, step=1).reshape(-1, 1), return_std=False)

#-- -----------------
#-- ---Apply Noise---
#-- -----------------

# define white noise
white_noise = 0.5*np.random.normal(size=num_days)

# define red (autoregressive) noise
ar_model = AutoReg(white_noise, lags=3)
ar_result = ar_model.fit()
red_noise = ar_result.predict()

# prepend zeros to red_noise
np.put(red_noise, range(3), [0]*3)

# apply noise to mean targets
time_series = mean_targets + white_noise + red_noise

#-- ------------------------
#-- ---Plot & Export Data---
#-- ------------------------

# plot data
plt.figure()
plt.style.use('bmh')
plt.plot([date_range[ind] for ind in gp_data], gp_target, label="Original")
plt.plot(date_range, mean_targets, label="GP Predicted")
plt.plot(date_range, time_series, label="Fully Synthesized")
plt.title("Synthesized Time Series")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()

# save plot
save_path = f"{expanduser('~')}/projects/devops_demo/reports/figures/synthesized_data.png"
plt.savefig(save_path)

# define df with synthesized data
df = pd.DataFrame(index=date_range, data=time_series, columns=['values'])

# save df using mlem
output_path = f"{expanduser('~')}/projects/devops_demo/data/raw/product_sales_data.csv"
save(df, output_path)
print(f"\nSaved time series data to path {output_path}")
