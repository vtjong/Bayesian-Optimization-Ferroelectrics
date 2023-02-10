import sys
import pandas as pd
import numpy as np
import xlwings as xw

def read_joulemeter(in_file="data/MKS_Ophir_joulemeter_readings.xlsx", 
                    out_file="data/Bolometer_readings_PulseForge.xlsx"):
    """
    [read_joulemeter(in_file, out_file)] reads in_file and out_file and writes
    energy density values from in_file to respective column in out_file. 
    """
    df = pd.read_excel(in_file, sheet_name="Sheet2")
    df2 = pd.read_excel(out_file, sheet_name="Combined")
    cond1 = lambda i,j: sheet.range((i,3)).value == df.iloc[j-2, 0] # voltage
    cond2 = lambda i,j: sheet.range((i,4)).value == df.iloc[j-2, 1] # time
    book = xw.Book(out_file) 
    sheet = book.sheets[1]               
    for i in range(2, len(df2)): 
        for j in range(2, len(df)):     
            if cond1 and cond2: sheet.range((i,7)).value = df.iloc[j-2, 2]  