import os, glob
import sys
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
from scipy.stats import spearmanr
from scipy.ndimage import gaussian_filter1d
from adjustText import adjust_text
import plotly.graph_objects as go

def prettyplot():
    """
    prettyplot()——some aesthetically pleasing settings. 
    """
    plt.style.use('bmh')
    mpl.rcParams['lines.linewidth'] = 1.5
    plt.rcParams['figure.dpi'] = 200
    plt.rcParams['font.size'] = 14

def vis_pred(train_x, train_y, test_grid, pred_labels):
    fig = go.Figure(data=[go.Surface(z=pred_labels.numpy(), 
                                    x=test_grid[:,0],
                                    y=test_grid[:,1],
                                    opacity = 0.8,
                                    colorscale = "Burg",
                                    colorbar=dict(thickness=15, len=0.5),
                                    name='GP regression')])
    
    fig.add_trace(go.Scatter3d(x=train_x[:,0],
                              y=train_x[:,1],
                            z=train_y.numpy(), mode='markers', 
                            marker={'color':'#72356c'}, name='training data'))
    fig.update_layout(width=1000, height=800,
                    # legend=dict(orientation="h", yanchor="top", y=1.02, 
                    # xanchor="left",x=1), margin=dict(r=0, l=0, b=0, t=0), 
                        scene=dict(
                        xaxis_title="Pulse Width (msec)",
                        yaxis_title="Energy density new cone (J/cm^2)",
                        zaxis_title='2 Qsw/(U+|D|) 1e6')
                    )
    fig.update_layout(template="ggplot2")
    camera = dict(up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0),
                    eye=dict(x=2.75, y=1.75, z=1))
    fig.update_layout(scene_camera=camera)
    fig.show()

def vis_acq(train_x, train_y, test_grid, pred_labels, upper_surf, lower_surf, acqs):
    fig = go.Figure(data=[go.Surface(z=pred_labels.numpy(), 
                                    x=test_grid[:,0],
                                    y=test_grid[:,1],
                                    opacity = 0.8,
                                    colorscale = "Burg",
                                    colorbar=dict(thickness=15, len=0.5),
                                    name='GP regression')])

    fig.add_trace(go.Surface(z=upper_surf, x=test_grid[:,0],
                    y=test_grid[:,1], opacity=0.2, colorscale = "Burg", showscale=False))

    fig.add_trace(go.Surface(z=lower_surf, x=test_grid[:,0],
                            y=test_grid[:,1], 
                            colorscale = "Burg",
                            opacity=0.2, showscale=False))
    fig.add_trace(go.Scatter3d(x=train_x[:,0], 
                                y=train_x[:,1], 
                                z=train_y.numpy(), 
                                mode='markers', 
                                name='training data', 
                                marker={'color':'#72356c'}))
    
    for acq_name, acq_val in acqs.items():
        fig.add_trace(go.Scatter3d(x=[test_grid[acq_val[1], 0]], 
                                y=[test_grid[acq_val[0], 1]],
                                z=[pred_labels[acq_val[0], acq_val[1]]], mode='markers', 
                                name='max(' + acq_name + ')')) 

    fig.update_layout(width=1200, height=750,
                    margin=dict(r=20, l=10, b=15, t=10),
                    legend=dict(orientation="h", yanchor="bottom", 
                                y=0.9, xanchor="right",x=0.85),
                    scene=dict(
                        xaxis_title="Pulse Width (msec)",
                        yaxis_title="Energy density new cone (J/cm^2)",
                        zaxis_title='2 Qsw/(U+|D|) 1e6')
                    )

    fig.update_layout(template="ggplot2")
    camera = dict(
    up=dict(x=0, y=0, z=1),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=2, y=0.3, z=0.75))

    fig.update_layout(scene_camera=camera)

    fig.show()

# get error metrics
def get_err(train_y, y_preds_mean):
    round_three = lambda val: np.round(val, 3) 
    rmse = np.sqrt(mean_squared_error(train_y, y_preds_mean))
    mae = mean_absolute_error(train_y,  y_preds_mean)
    spearman = spearmanr(train_y, y_preds_mean)[0]
    r2 = r2_score(train_y, y_preds_mean)
    return [round_three(i) for i in [rmse, mae, spearman, r2]]

def plot_err(axes, train_y, y_preds_mean, y_uncer, err_vals, train_or_test, fs):
    # training versus actuals plot: make plot, set title and axes values
    ax = axes[0]
    ax.scatter(train_y, y_preds_mean, color = "#72356c")
    low_lim, upp_lim = int(min(train_y).item()), int(np.ceil(max(train_y).item()))
    ax.plot(np.linspace(low_lim, upp_lim), np.linspace(low_lim, upp_lim), 'k--')
    ax.set_xlim(low_lim, upp_lim)
    ax.set_ylim(low_lim, upp_lim)
    
    # sort train_y to enable fill_between to exhibit correct functionality
    # plot 95% confidence interval
    sorted_train_y, sorted_indices_y = np.sort(train_y), np.argsort(train_y)
    low_bound = (y_preds_mean-y_uncer)[sorted_indices_y]
    upp_bound = (y_preds_mean+y_uncer)[sorted_indices_y]
    ax.fill_between(sorted_train_y, low_bound, upp_bound, color = "grey", alpha = 0.3, label="95% confidence interval")    
    ax.set_xlabel('Ground Truth 2 Qsw/(U+|D|) 1e6', fontsize = fs)
    ax.set_ylabel('Prediction 2 Qsw/(U+|D|) 1e6', fontsize = fs)
    title_type = "Training" if train_or_test == "train" else "Test"
    ax.set_title('GP ' + title_type + ' Results' + " (MAE=%.2f" % err_vals[1]+' [%])', fontsize = fs)
    ax.legend()

def plot_training_loss(axes, loss_lst, fs):
    ax = axes[1]
    ax.plot(np.arange(len(loss_lst))*500,loss_lst, 'o-', color = "#72356c")
    ax.set_xlabel('Epoch', fontsize = fs)
    ax.set_ylabel('Marginal Log Likelihood Loss', fontsize = fs)
    ax.set_title('Training Loss' + " (Loss=%.2f" % loss_lst[-1] + ')', fontsize = fs)

def plot_gp_res(train_y, y_preds_mean, loss_lst, y_uncer, train_or_test="train"):
    prettyplot()
    fig, axes = plt.subplots(1, 3, figsize=(5.5*3, 4.5))
    fs = 14

    # print error metrics
    err_vals = get_err(train_y, y_preds_mean)
    data = {'Metric': ['RMSE', 'MAE', 'Spearman', 'R² score'], 'Value': err_vals}
    df = pd.DataFrame(data)
    print(df)

    # plot error, training/testing loss
    plot_err(axes, train_y, y_preds_mean, y_uncer, err_vals, train_or_test, fs)
    if loss_lst: plot_training_loss(axes, loss_lst, fs)
    else: axes[1].axis("off")
    axes[2].axis("off")

    for i in range(len(axes)):
        axes[i].tick_params(direction='in', length=5, width=1, labelsize = fs*.8)
    plt.subplots_adjust(wspace = 0.4)
    
    plt.show()