"""
DevOps Demo: Toolbox
Author: Trevor Cross
Last Updated: 01/11/23

Series of functions used to assist in project development
"""

# ----------------------
# ---Import Libraries---
# ----------------------

# import standard libraries
import pandas as pd

# import plotting libraries
import matplotlib.pyplot as plt

# import statistics libraries
import statsmodels.api as sm

# import scikit-learn libraries
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             mean_absolute_percentage_error)

# import support libraries
import json

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
                ylabel="Value",
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
def plot_forecast_cv(ts, true_ts_list, predicted_ts_list, title="CV Forecasts", save_path=None, num_xticks=10):

    # set style params
    plt.style.use('bmh')

    # define figure
    plt.figure(figsize=(20, 14))

    # plot original timeseries
    ts.index = pd.to_datetime(ts.index)
    plt.plot(ts.index, ts.values, label="Original Time Series", color='k')

    # iterate lists
    for ts_num, packed_ts in enumerate(zip(true_ts_list, predicted_ts_list)):

        # unpack time series
        true_ts = packed_ts[0]
        predicted_ts = packed_ts[1]

        # convert dates
        true_ts.index = pd.to_datetime(true_ts.index)
        predicted_ts.index = pd.to_datetime(predicted_ts.index)

        # plot true timeseries
        plt.plot(true_ts.index,
                 true_ts.values,
                 label="True Time Series" if ts_num == 0 else "_nolegend_",
                 marker='o' if len(true_ts) == 1 else None,
                 color='b'
                 )

        # plot predicted timeseries
        plt.plot(predicted_ts.index,
                 predicted_ts.values,
                 label="Forecast" if ts_num == 0 else "_nolegend_",
                 marker='o' if len(predicted_ts) == 1 else None,
                 color='r')

    # specify xticks
    xticks = list(ts.index[range(0, len(ts), len(ts)//num_xticks)])
    xticks.append(ts.index[-1])

    # prettify
    plt.title(title)
    plt.xticks(xticks, rotation=20)
    plt.legend()

    # save plot
    if save_path != None:
        plt.savefig(save_path)
        print(f"\nSaved plot {title} to path {save_path}")
    else:
        plt.show()

# ---------------------------------------
# ---Define Model Evaluation Functions---
# ---------------------------------------

# define function to evaluate model across different metrics
def eval_metrics(true_values, predicted_values):
    return {'MSE': mean_squared_error(true_values, predicted_values, squared=True),
            'RMSE': mean_squared_error(true_values, predicted_values, squared=False),
            'MAE':mean_absolute_error(true_values, predicted_values),
            'MAPE': mean_absolute_percentage_error(true_values, predicted_values)
            }

# ------------------------------
# ---Define Support Functions---
# ------------------------------

# define function to save dictionary as JSON file
def dict_to_json(my_dict, file_path):
    with open(file_path, "w+") as file:
        json.dump(my_dict, file, indent=4)
