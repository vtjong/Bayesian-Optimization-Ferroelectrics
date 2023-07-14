# Prep training data
from copyreg import pickle
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import sys
import gzip
import numpy as np
import pandas as pd
import pickle as pk
import torch
import plotly.express as px
sys.path.append('..')
sys.path.insert(0, '../src')

def read_dat(dir="/Users/valenetjong/Bayesian-Optimization-Ferroelectrics/data/",
            src_file = "Bolometer_readings_PulseForge.xlsx", sheet= "Combined"):
    file = dir + src_file
    fe_data = pd.read_excel(file, sheet_name=sheet, usecols=['Energy density new cone (J/cm^2)',
                            'Time (ms)','2 Qsw/(U+|D|) 1e6cycles'])
    fe_data.dropna(subset=['2 Qsw/(U+|D|) 1e6cycles'], inplace=True)
    return fe_data

def display_data(fe_data):
    """
    [display_data(fe_data)] creates a cross-section scatter plot of all combinations
    of the four input parameters and single output parameter.
    """
    # Plot each cross-section
    fig = px.scatter_matrix(fe_data, dimensions=['Energy density new cone (J/cm^2)', 
    "Time (ms)", "2 Qsw/(U+|D|) 1e6cycles"])
    fig.update_layout(margin=dict(r=20, l=10, b=10, t=10))
    fig.update_layout(height=1000)
    fig.show()
    
def grid_helper(grid_size, num_params, grid_bounds):
    """
    [grid_helper(grid_size, num_params, grid_bounds)] returns a grid of dimensions
    [grid_size] by [num_params], filled in with data from array [grid_bounds].
    """
    grid = torch.zeros(grid_size, num_params)
    f_grid_diff = lambda i, x, y : float((x[i][1] - x[i][0]) / (y-2))
    for i in range(num_params):
        grid_diff = f_grid_diff(i, grid_bounds, grid_size)
        grid[:, i] = torch.linspace(grid_bounds[i][0] - grid_diff, 
                                    grid_bounds[i][1] + grid_diff, grid_size)
    return grid

def grid_maker(train_x):
    """
    [grid_maker(train_x)] creates grids to be used for gaussian process predictions.
    It outputs the dimension of the grid [num_params] and two grid utility 
    tensors [test_grid] and [test_x].
    """
    # Define grid between bounds of RTA time, RTA temp
    num_params = train_x.size(dim=1)
    grid_bounds = [(train_x[:,i].min(), train_x[:,i].max()) for i in range(num_params)]
    grid = grid_helper(20, num_params, grid_bounds)

    # Set up test_grid for predictions
    n = 30
    test_grid = grid_helper(n, num_params, grid_bounds)

    # Create 4D grid
    args = (test_grid[:, i] for i in range(num_params))
    test_x = torch.cartesian_prod(*args)
    test_x.shape
    return num_params, test_grid, test_x

def datasetmaker(fe_data):
    """
    [datasetmaker(fe_data)] filters and transforms the data in pandas df [fe_data] 
    into two tensors, [train_x] for input and [train_y] for output tensors. 
    """
    T_scaler = StandardScaler()
    # Filter training data 
    mask = ~np.isnan(fe_data['2 Qsw/(U+|D|) 1e6cycles'])
    energy_den = fe_data['Energy density new cone (J/cm^2)'][mask].values
    time = fe_data['Time (ms)'][mask].values
    # print(energy_den)
    # print(time)
    train_x = np.array([fe_data['Energy density new cone (J/cm^2)'][mask].values, 
                        fe_data['Time (ms)'][mask].values]).transpose()
    column_mean = np.mean(train_x, axis=0)
    column_sd = np.std(train_x, axis=0) 
    train_x-= column_mean
    train_x/= column_sd
    train_x = torch.Tensor(train_x)
    train_y = torch.Tensor(fe_data['2 Qsw/(U+|D|) 1e6cycles'][mask].values)
    return column_mean, column_sd, train_x, train_y

def save_dataset():
    """
    [save_dataset()] writes transformed data and parameters to csv.
    """
    dir = "quickload/"
    fe_data = read_dat()
    # display_data(fe_data)
    column_mean, column_sd, train_x, train_y = datasetmaker(fe_data)
    num_params, test_grid, test_x = grid_maker(train_x)
    params = [column_mean, column_sd, num_params]

    with gzip.open(dir + "train_x", "wb") as f: pk.dump([train_x], f)
    with gzip.open(dir + "train_y", "wb") as f: pk.dump([train_y], f)
    with gzip.open(dir + "test_grid", "wb") as f: pk.dump([test_grid], f)
    with gzip.open(dir + "test_x", "wb") as f: pk.dump([test_x], f)
    with gzip.open(dir + "params", "wb") as f: pk.dump(params, f)

    # return column_mean, column_sd, train_x, train_y, num_params, test_grid, test_x

save_dataset()