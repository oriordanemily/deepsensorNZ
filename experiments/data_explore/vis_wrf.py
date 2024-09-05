# Plot ERA5

#%%

import os

import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cf
import seaborn as sns
import scipy.stats as stats
from scipy.stats import gamma, bernoulli
from scipy.optimize import minimize

from nzdownscale.dataprocess.wrf import ProcessWRF
from nzdownscale.dataprocess.utils import PlotData

#%% Load ERA5 data

wrf = ProcessWRF()
plotnz = PlotData()

# %%
def neg_log_likelihood(params, non_zero_data):
    alpha, beta = params[0], params[1]
    return -np.sum(gamma.logpdf(non_zero_data, alpha, scale=1/beta))

#%% Load specific year only

for var in [
        'temperature',
        'precipitation',
        '10m_u_component_of_wind',
        '10m_v_component_of_wind',
        'surface_pressure',
        'surface_solar_radiation_downwards',
        ]:

    ds = wrf.load_ds([2024], [1], [var])
    da = wrf.ds_to_da(ds, var)

    # Plot timeslice
    # ax = plotnz.nz_map_with_coastlines()
    # da_to_plot = da.isel(time=0)
    # da_to_plot.plot()
    # plt.title(f'ERA5-land: {var}\n{da_to_plot["time"].values}')
    # # plt.savefig('./tmp/fig.png')
    # plt.show()

    # Plot histogram of all values
    fig, ax = plt.subplots()
    da.plot.hist(ax=ax, bins=100, density=True, label=f'WRF 2024 Jan {var} histogram')
    print('Plotted histogram')

    # # plot gaussian
    mean = da.mean().values
    std = da.std().values
    x = np.linspace(mean - 3*std, mean + 3*std, 100)

    ax.plot(x, stats.norm.pdf(x, mean, std), c='r', 
            label = f'N({mean:.2f}, {std:.2f})')
    print('Plotted gaussian')

    plt.title(f'WRF: {var} histogram')
    # # plot bernoulli-gamma
   
    # Aggregate data across latitudes and longitudes
    data = da.mean(dim=['latitude', 'longitude'])

    # Calculate Bernoulli parameter (probability of success)
    p = data.mean().values

    # Calculate Gamma parameters using method of moments
    def gamma_params(data):
        mean = np.mean(data)
        var = np.var(data)
        shape = mean**2 / var
        scale = var / mean
        return shape, scale

    # Extract data values for calculation
    data_values = data.values.flatten()
    shape, scale = gamma_params(data_values)

    # Plot histogram of the data points
    # fig, ax = plt.subplots()
    # ax.hist(data_values, bins=100, density=True, alpha=0.5, label='Data Histogram')

    # Plot Bernoulli distribution
    # bernoulli_dist = bernoulli(p)
    # x_bernoulli = np.array([0, 1])
    # y_bernoulli = bernoulli_dist.pmf(x_bernoulli)
    # ax.stem(x_bernoulli, y_bernoulli, 'r', markerfmt='ro', label=f'Bernoulli PMF p={p:.2f}', basefmt=" ")

    # Plot Gamma distribution
    x_gamma = np.linspace(0, data_values.max(), 100)
    y_gamma = gamma.pdf(x_gamma, a=shape, scale=scale)
    ax.plot(x_gamma, y_gamma, 'g-', label=f'Gamma PDF shape={shape:.2f}, scale={scale:.2f}')

    # Add titles and labels
    # ax.set_title(f'Fitted Bernoulli-Gamma Distributions for {var}')
    ax.set_xlabel('Data Values')
    ax.set_ylabel('Density')
    ax.legend()


    plt.show()


# %%