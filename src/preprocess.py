import os, glob
import sys
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from plotter import prettyplot, vis_iv, vis_pv

def get_devicelen(device): return int(device[:device.find('um')])

def scale_pr(device, pr): return pr/(get_devicelen(device)**2*10**(-14))

def get_dev(type, file):
    """
    get_dev(type, file) finds the device names given [file] and type of 
    file [type].
    """
    file = file.split(type,1)[1]
    file = file[:file.rfind('.')]
    if file.rfind('_after') != -1:
        file = file[:file.rfind('_after')]
    return file

def process_PV(df, file, print_flag_IV=False, print_flag_PV=False):
    """
    process_PV(df, file, print_flag_IV, print_flag_PV) calculates the 
    Pr, Vc, and Imprint values for the P-V and I-V loops for a given [file] 
    and returns an updated dataframe; if print_flag = True, 
    runs plot production code.
    """
    device = get_dev("PV_", file)  
    dev_row = df[df["device"] == device].index.to_numpy()[0]
    try: 
        PV_df = pd.read_excel(file, sheet_name='Append2', usecols=['Vforce','Charge'])
        IV_df = pd.read_excel(file, sheet_name='Append2', usecols=['Vforce','Imeas'])
    except:
        print("No Append2")
        return df
    pv_data = np.array(PV_df)
    iv_data = np.array(IV_df)

    # Unit conversion
    pv_data[:,1] = scale_pr(device, pv_data[:,1])
    
    # Shift data
    max, min = np.amax(pv_data,axis=0)[1],np.amin(pv_data,axis=0)[1]
    pv_data[:,1] -= np.mean((max, min))

    # Find Pr, Vc and imprint for P-V
    dir_Δ = 300 # 300 marks data point of direction change
    Pr = pv_data[np.argwhere(pv_data[dir_Δ:,0] < 0)[0]+dir_Δ,1][0]
    Vc_neg = pv_data[np.argwhere(pv_data[dir_Δ:,1] < 0)[0]+dir_Δ,0][0]
    Vc_pos = pv_data[np.argwhere(pv_data[50:,1] > 0)[0]+50,0][0]
    Vc = np.mean((Vc_pos, np.abs(Vc_neg)))
    Imprint = np.mean((Vc_pos,Vc_neg))

    df.at[dev_row,"Pr (mC/cm^2)"] = Pr
    df.at[dev_row,"Vc (P-V)"] = Vc
    df.at[dev_row,"Imprint (P-V)"] = Imprint

    # Produce I-V plots
    if print_flag_IV:
        pv_neg = pv_data[np.argwhere(pv_data[:,0]==Vc_neg)[0][0]][1]
        pv_neg_tup = (Vc_neg, pv_neg)
        pv_pos = pv_data[np.argwhere(pv_data[:,0]==Vc_pos)[0][0]][1]
        pv_pos_tup = (Vc_pos, pv_pos)
        vis_pv(pv_data, pv_neg_tup, pv_pos_tup, device)

    # Find Vc and imprint for I-V
    iv_filt = gaussian_filter1d(iv_data[:,1],2)
    arg_vc_pos, arg_vc_neg = np.argmax(iv_filt), np.argmin(iv_filt)
    Vc_pos_iv, Imeas_Vc_pos = iv_data[:,0][arg_vc_pos], iv_filt[arg_vc_pos]
    Vc_neg_iv, Imeas_Vc_neg = iv_data[:,0][arg_vc_neg], iv_filt[arg_vc_neg]
    df.at[dev_row, "Vc (I-V)"] = np.mean((Vc_pos_iv, np.abs(Vc_neg_iv)))
    df.at[dev_row, "Imprint (I-V)"] = np.mean((Vc_pos_iv, Vc_neg_iv))

    if print_flag_PV: 
        pos_tup, neg_tup = (Vc_pos_iv, Imeas_Vc_pos), (Vc_neg_iv, Imeas_Vc_neg)
        vis_iv(iv_data, iv_filt, pos_tup, neg_tup, device)  
    return df

def process_PUND(df, file):  
    device = get_dev("PUND_", file)  
    return df

def process_endurance(df, file):
    """
    process_endurance(df, file) finds the endurance of a given [file] and returns
    an updated dataframe. 
    """
    device = get_dev("endurance_", file)
    dev_row = df[df["device"] == device].index.to_numpy()[0]
    PV_df = pd.read_excel(file, usecols=['iteration','P','Qsw'])
    data = np.array(PV_df)
    # print(data)

    # Find iteration before occurrence of breakdown
    breakdown = np.argmax(data[:,1]>=1e-9)
    df.at[dev_row,"endurance"] = data[breakdown -1][0] if breakdown!=0 else 0
    
    # Find max Pr (before break)
    dat_bef_brk = data[:breakdown] if breakdown!=0 else data
    Q_sw = np.amax(dat_bef_brk, axis=0)[2]
    Pr_max = 0.5 * Q_sw
    df.at[dev_row,"max Pr (mC/cm^2)"] = scale_pr(device, Pr_max)

    # Find Pr at 10^6 by applying polyfit & eval at 1e6
    p = np.polyfit(dat_bef_brk[:,0], 0.5*dat_bef_brk[:,2], deg=2)
    Pr_1e6 = 0.5 * np.polyval(p, 1e6)
    df.at[dev_row,"10^6 Pr (mC/cm^2)"] = scale_pr(device, Pr_1e6)
    return df

def read_file(dir, idx):
    """
    read_file(dir, idx) reads all files in subdirectory specified by [dir]
    and [idx] and returns relevant information, written to a pandas df.
    """
    subdir = dir + str(idx)
    files = subdir + '/*.xls'
    files_exp = glob.glob(files, recursive = True)
    df = init_df(files_exp)

    for filename in files_exp:
        if 'PUND' in filename:
            df = process_PUND(df, filename)
        elif 'PV' in filename:
            df = process_PV(df, filename)
        elif 'endurance' in filename:
            df = process_endurance(df, filename)
    return df

def init_df(files_exp):
    """
    init_df(files_exp) initializes a dataframe with all device names in current
    subdirectory whose filenames are in [files_exp].  
    """
    end_files = [file for file in files_exp if "endurance" in file]
    PV_files = [file for file in files_exp if "PV" in file]
    # print("num end", len(end_files))
    # print("num pv", len(PV_files))

    # If missing endurance files, use PV files to extract device names
    if len(end_files) < len(PV_files):
        row_names = [file.split("PV_",1)[1] for file in PV_files]
        row_names = [file[:file.rfind('.')] for file in row_names]
        row_names = [file[:file.rfind('_after')] if "after" in file else 
        file for file in row_names]
        n_devices = len(PV_files)
    else:
        n_devices = len(end_files)
        row_names = [file.split("endurance_",1)[1] for file in end_files]
        row_names = [file[:file.rfind('.')] for file in row_names]
    n_rows = 14
    dat = np.zeros((n_devices, n_rows))
    col_names = ["device", "Pr (mC/cm^2)", "Vc (P-V)", "Imprint (P-V)", "Vc (I-V)", 
    "Imprint (I-V)", "endurance", "10^6 Pr (mC/cm^2)", "10^6 Vc", "10^6 Imprint", 
    "10^7 Pr (mC/cm^2)", "10^7 Vc", "10^7 Imprint", "max Pr (mC/cm^2)"] 
    df = pd.DataFrame(dat, columns=col_names)
    df['device'] = row_names
    return df

def main(dir, num_subdirs=23):
    """
    main(dir, num_subdirs=23) operates as the main caller function to read in 
    all raw data in various subdirectories and write out processed df data.
    """
    for idx in range(1, num_subdirs+1):
        df = read_file(dir, idx)
        df.to_csv(dir[:dir.rfind("/")] + "/processed/"+ str(idx)+ ".csv") 

# Update with file path on your local device 
prettyplot()
dir = '/Users/valenetjong/Bayesian-Optimization-Ferroelectrics/data/KHM010_'
main(dir)