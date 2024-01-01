"""
DevOps Demo: Toolbox
Author: Trevor Cross
Last Updated: 12/28/23

Series of functions used to assist in project development
"""

# ----------------------
# ---Import Libraries---
# ----------------------

# import standard libraries
import pandas as pd
import numpy as np

# import plotting libraries
import matplotlib.pyplot as plt

# import statistics libraries
import statsmodels.api as sm

# import scikit-learn functions
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             mean_absolute_percentage_error)

# -------------------------------------
# ---Define Data Wrangling Functions---
# -------------------------------------

# define function to generate splits by category column
def split_by_cat_generator(df, cat_col, test_size=0.20, shuffle=False, rs=81):
    for cat_name in df[cat_col].unique():
        cat_df = df[df[cat_col] == cat_name]
        yield train_test_split(cat_df, test_size=test_size, shuffle=shuffle, random_state=rs)

# -------------------------------
# ---Define Plotting Functions---
# -------------------------------

# define function to plot (differenced) timeseries
def plot_differenced_ts(series, save_path=None, differencing_param=1, title="Time Series", num_xticks=10):

    # set style params
    plt.style.use('bmh')

    # define figure & axes
    fig, axes = plt.subplots(2, 1, figsize=(20, 14))

    # plot timeseries
    axes[0].plot(series.index, series.values)
    axes[0].set(title=title,
                ylabel="Units Sold",
                xticks = series.index[range(0, len(series), len(series)//num_xticks)]
                )

    # plot differenced timeseries
    diff_series = series
    for order in range(differencing_param):
        diff_series = diff_series.diff()
        axes[1].plot(diff_series.index, diff_series.values, label=f"Order {order + 1}")

    axes[1].set(title=title + " (Differenced)",
                ylabel=None,
                xticks = diff_series.index[range(0, len(series), len(series)//num_xticks)]
                )
    axes[1].legend()
    axes[1].plot(diff_series.index, [0]*len(series), 'r')

    # save plot
    if save_path != None:
        plt.savefig(save_path)
        print(f"\nSaved plot {title} to path {save_path}")
    else:
        plt.show()


# define function to plot ACF & PACF
def plot_acf(series, save_path=None, differencing_param=1, title=None, num_xticks=25):

    # set style params
    plt.style.use('bmh')

    # define figure & axes
    fig, axes = plt.subplots(2, 1, figsize=(20, 14))
    if title != None:
        fig.suptitle(title, fontsize=20)

    # difference timeseries
    diff_series = series
    for order in range(differencing_param):
        diff_series = diff_series.diff()

    # plot ACF
    fig = sm.graphics.tsa.plot_acf(diff_series.values[1:], lags=num_xticks, ax=axes[0])

    # plot PACF
    fig = sm.graphics.tsa.plot_pacf(diff_series.values[1:], lags=num_xticks, ax=axes[1])

    # save plot
    if save_path != None:
        plt.savefig(save_path)
        print(f"\nSaved plot {title} to path {save_path}")
    else:
        plt.show()

# define function to plot timeseries forecast
def plot_forecast(orig_ts, test_ts, forecast_obj, title="Forecast", save_path=None, num_xticks=10):

    # set style params
    plt.style.use('bmh')

    # define figure
    plt.figure()
    plt.title(title)

    # plot original timeseries
    plt.plot(orig_ts.index, orig_ts.values, label="Original Time Series")

    # plot test timeseries
    plt.plot(test_ts.index, test_ts.values, label="Test Time Series")

    # plot forecasts
    plt.plot(test_ts.index, forecast_obj.predicted_mean.values, label="Forecast")

    # plot confidence intervals
    conf_ints = forecast_obj.conf_int(alpha=0.05)
    plt.fill_between(test_ts.index, conf_ints.iloc[:, 0], conf_ints.iloc[:, 1], alpha=0.2, label='Confidence Intervals')

    # prettify
    plt.title(title)

    ts = pd.concat([orig_ts, test_ts], ignore_index=False)
    plt.xticks(ts.index[range(0, len(ts), len(ts)//num_xticks)])

    plt.legend()

    # save plot
    if save_path != None:
        plt.savefig(save_path)
        print(f"\nSaved plot {title} to path {save_path}")
    else:
        plt.show()

# ---------------------------------------
#---Define Model Evaluation Functions---
#---------------------------------------

# define function to evaluate model across different metrics
def eval_metrics(true_values, predicted_values):
    return {'MSE': mean_squared_error(true_values, predicted_values, squared=True),
            'RMSE': mean_squared_error(true_values, predicted_values, squared=False),
            'MAE':mean_absolute_error(true_values, predicted_values),
            'MAPE': mean_absolute_percentage_error(true_values, predicted_values)
            }
