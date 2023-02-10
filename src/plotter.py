import os, glob
import sys
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from adjustText import adjust_text

def prettyplot():
    """
    prettyplot()——some aesthetically pleasing settings. 
    """
    plt.style.use('seaborn')
    mpl.rcParams['lines.linewidth'] = 1.5
    plt.rcParams['figure.dpi'] = 200

def vis_pv(pv_data, pv_pos_tup, pv_neg_tup, device):
    """
    vis_pv(pv_data, pv_pos_tup, pv_neg_tup, device) saves a plot of 
    the shifted PV curve, as well as labelled max and min points.
    """
    plt.clf()
    plt.plot(pv_data[:,0], pv_data[:,1], c = '#580F41', label = "raw")
    plt.plot(pv_pos_tup[0], pv_pos_tup[1], 'C3o')
    plt.plot(pv_neg_tup[0], pv_neg_tup[1], 'C3o')
    plt.xlabel('Charge Density')
    plt.ylabel('Vforce')
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
    plt.xlabel('Imeas')
    plt.ylabel('Vforce')
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
