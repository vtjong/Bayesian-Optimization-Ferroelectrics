# Prep training data
from sklearn.preprocessing import StandardScaler
import sys, os
import numpy as np
import pandas as pd
import torch
import plotly.express as px
sys.path.append('..')
sys.path.insert(0, '../src')

def make_dummy_data(): 
    # Load data
    fe_data = pd.read_csv("/Users/valenetjong/Bayesian-Optimization-" + 
    "Ferroelectrics/data/KHM005_KHM006_quartz_HZO_samples.csv", index_col=0)
    fe_data_len = len(fe_data['Thickness (nm)'])

    # Add duty cycle data
    duty_cycle_list = np.array([0.45, 0.55, 0.65])
    duty_cycles = np.random.choice(duty_cycle_list, size=fe_data_len)
    fe_data['Duty Cycle'] = duty_cycles

    # Add num pulses data
    num_pulses_list = np.array([15, 25])
    num_pulses = np.random.choice(num_pulses_list, size=fe_data_len)
    fe_data['Num Pulses'] = num_pulses

    # Rearrange columns
    cols = list(fe_data.columns.values) 
    idx = cols.index("Flash time (msec)")
    cols = cols[:idx+1] + cols[-2:] + cols[idx+1:-2]
    fe_data = fe_data[cols]

    # Write data back to csv
    os.makedirs('../data', exist_ok=True) 
    fe_data.to_csv('../data/KHM005_KHM006_quartz_HZO_samples2.csv') 

    # Load manipulated data
    fe_data = pd.read_csv('../data/KHM005_KHM006_quartz_HZO_samples2.csv', 
    index_col=0)
    return fe_data

def display_data(fe_data):
    # Plot each cross-section
    fig = px.scatter_matrix(fe_data, dimensions=["Flash voltage (kV)", 
    "Flash time (msec)", "Duty Cycle", "Num Pulses", "Pr (uC/cm2), Pristine state"])
    # fig.update_layout(margin=dict(r=20, l=10, b=10, t=10))
    fig.update_layout(height=1000)
    fig.show()
    
def grid_helper(grid_size, num_params, grid_bounds):
    grid = torch.zeros(grid_size, num_params)
    f_grid_diff = lambda i, x, y : float((x[i][1] - x[i][0]) / (y-2))
    for i in range(num_params):
        grid_diff = f_grid_diff(i, grid_bounds, grid_size)
        grid[:, i] = torch.linspace(grid_bounds[i][0] - grid_diff, 
                                    grid_bounds[i][1] + grid_diff, grid_size)
    return grid

def filter(fe_data):
    T_scaler = StandardScaler()
    # Filter training data 
    mask = ~np.isnan(fe_data['Pr (uC/cm2), Pristine state'])
    train_x = torch.Tensor(np.array([fe_data['Flash voltage (kV)'][mask].values, 
                        fe_data['Flash time (msec)'][mask].values, 
                        fe_data['Duty Cycle'][mask].values,
                        fe_data['Num Pulses'][mask].values])).T
    train_y = torch.Tensor(fe_data['Pr (uC/cm2), Pristine state'][mask].values)
    return train_x, train_y

def grid_maker(train_x):
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
    return num_params, test_x

def dataset():
    fe_data = make_dummy_data()
    display_data(fe_data)
    train_x, train_y = filter(fe_data)
    num_params, test_x = grid_maker(train_x)
    return train_x, train_y, num_params, test_x

