import os, glob
import sys
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
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

def vis_pund(pund_data, iv_data, device, sheet_name):
    plt.clf()
    plt.plot(pund_data[:,0], pund_data[:,1], c = '#580F41', label = "PUND")
    plt.plot(iv_data[:,0], iv_data[:,1], c = '#A9561E', label = "Current")
    plt.xlabel('time (s)')
    plt.ylabel('V (V)')
    dir = "/Users/valenetjong/Bayesian-Optimization-Ferroelectrics/plots"
    if sheet_name == 'Data':
        iter = '3 cycle'
    elif sheet_name == 'Append1':
        iter = '10^6 cycle'
    elif sheet_name == 'Append2':
        iter = '10^7 cycle'
    plt.title(device + " PUND "+ iter)
    plt.legend(loc="upper right")
    plt.savefig(dir + "/PUND/" + device + "_" + iter +"_PUND-plot")   

def vis_pv(pv_data, pv_pos_tup, pv_neg_tup, device):
    """
    vis_pv(pv_data, pv_pos_tup, pv_neg_tup, device) saves a plot of 
    the shifted PV curve, as well as labelled max and min points.
    """
    plt.clf()
    plt.plot(pv_data[:,0], pv_data[:,1], c = '#580F41', label = "raw")
    plt.plot(pv_pos_tup[0], pv_pos_tup[1], 'C3o')
    plt.plot(pv_neg_tup[0], pv_neg_tup[1], 'C3o')
    plt.xlabel('Vforce (V)')
    plt.ylabel('Charge Density (mC/cm^2)')
    together = [pv_pos_tup, pv_neg_tup]
    texts=[]
    for i_x, i_y in together:
        texts.append(plt.text(i_x, i_y, r'({0:.3f}, {1:.3E})'.format(i_x, i_y), 
        c="black", fontsize='medium', 
        bbox=dict(facecolor='white', alpha=0.5)))
    dir = "/Users/valenetjong/Bayesian-Optimization-Ferroelectrics/plots"
    plt.title(device + " PV")
    plt.legend(loc="upper right")
    adjust_text(texts, only_move={'points':'y', 'texts':'y'})
    plt.savefig(dir + "/PV/" + device + "_" + "PV-plot")   
    # plt.show() 

def vis_iv(iv_data, iv_filt, pos_tup, neg_tup, device):
    """
    vis_iv(iv_data, iv_filt, pos_tup, neg_tup, device) saves a plot of 
    the IV curve, with raw and filtered data, as well as labelled max and min 
    points.
    """
    plt.clf()
    plt.plot(iv_data[:,0], iv_data[:,1], c = '#580F41', label = "raw")
    plt.plot(iv_data[:,0], iv_filt, c = '#A9561E', label = "filtered")
    plt.plot(pos_tup[0], pos_tup[1], 'C3o')
    plt.plot(neg_tup[0], neg_tup[1], 'C3o')
    plt.xlabel('Vforce (V)')
    plt.ylabel('Imeas (mC)')
    col = [pos_tup, neg_tup]
    for i_x, i_y in col:
        plt.text(i_x, i_y, r'({0:.3f}, {1:.3E})'.format(i_x, i_y), 
        c="black", fontsize='medium', 
        bbox=dict(facecolor='white', alpha=0.5))
    dir = "/Users/valenetjong/Bayesian-Optimization-Ferroelectrics/plots"
    plt.title(device + " IV")
    plt.legend(loc="upper right")
    plt.savefig(dir + "/IV/" + device + "_" + "IV-plot")    
    # plt.show()
    
def vis_pred(train_x, train_y, test_grid, pred_labels):
    fig = go.Figure(data=[go.Surface(z=pred_labels.numpy().T, 
                                    x=test_grid[:,0],
                                    y=test_grid[:,1],
                                    opacity = 0.8,
                                    name='GP regression')])
    fig.add_trace(go.Scatter3d(x=train_x[:,0],
                              y=train_x[:,1],
                            z=train_y.numpy(), mode='markers', 
                            marker={'color':'darkgreen'}, name='training data'))
    fig.update_layout( width=1000, height=800,
                    # legend=dict(orientation="h", yanchor="top", y=1.02, 
                    # xanchor="left",x=1), margin=dict(r=20, l=10, b=10, t=10), 
                        scene=dict(
                        xaxis_title="Pulse Width (msec)",
                        yaxis_title="Energy density new cone (J/cm^2)",
                        zaxis_title='2 Qsw/(U+|D|) 1e6')
                    )
    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=-2, y=-2.5, z=1.75)
    )

    fig.update_layout(scene_camera=camera)
    fig.show()

def vis_acq(train_x, train_y, test_grid, pred_labels, upper_surf, lower_surf, acqs):
    # Get back real values from standardized version

    pi, ei, ca, th, ucb = acqs["PI"], acqs["EI"], acqs["CA"], acqs["TH"], acqs["UCB"]

    fig = go.Figure(data=[go.Surface(z = upper_surf, x=test_grid[:,0],
                    y=test_grid[:,1], opacity=0.5, showscale=False)])
    fig.add_trace(go.Surface(z=lower_surf, x=test_grid[:,0],
                            y=test_grid[:,1], 
                            opacity=0.2, showscale=False))
    fig.add_trace(go.Scatter3d(x=train_x[:,0], 
                                y=train_x[:,1], 
                                z=train_y.numpy(), 
                                mode='markers', 
                                name='training data', 
                                marker={'color':'darkgreen'}))

    for acq_name, acq_val in acqs.items():
        fig.add_trace(go.Scatter3d(x=[test_grid[acq_val[1], 0]], 
                                y=[test_grid[acq_val[0], 1]],
                                z=[pred_labels[acq_val[0], acq_val[1]]], mode='markers', 
                                name='max(' + acq_name + ')')) 

    fig.update_layout( width=1000, height=600,
                    margin=dict(r=20, l=10, b=15, t=10),
                    legend=dict(orientation="h", yanchor="bottom", 
                                y=1.02, xanchor="right",x=1),
                    scene=dict(
                        xaxis_title="Pulse Width (msec)",
                        yaxis_title="Energy density new cone (J/cm^2)",
                        zaxis_title='2 Qsw/(U+|D|) 1e6')
                    )
    dir = "/Users/valenetjong/Bayesian-Optimization-Ferroelectrics/plots"
    fig.show()