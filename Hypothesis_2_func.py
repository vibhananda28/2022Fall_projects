# Importing Libraries
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import datetime
from IPython.display import display, HTML

def clean_data1(df1):
    alist = df1["Country"].unique()
    for i1 in alist:
        df2 = df1.loc[df1["Country"]==i1]
        len_recovery = len(df2["Recovery%"].unique())
        len_municipal = len(df2["Municipal_waste_generated_percapita"].unique())
        if(len_recovery<3 or len_municipal<3):
            del_row = df1.loc[df1["Country"]==i1].index
            df1 = df1.drop(del_row)
    
    return df1

def time_series(ts1, variable):
    
    fig, ax = plt.subplots(figsize=(10,5))
    for key1 in ts1["Country"].unique():
        ts2 = ts1[ts1["Country"]==key1]
        x1 = ts2["Year"]
        y1 = ts2[variable]
        plt.plot(x1, y1, label = key1)    


    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, shadow=True, ncol=5)
    plt.ylabel(variable)
    
    plt.title(str(variable + " as a function of time"))
    return fig, ax

def time_series_subplots(ts1, variable):
    tot_num = len(ts1["Country"].unique())
    num_col = 4 # number of figures to display in one row
    #Calculating number of rows
    num_row = tot_num
    if(tot_num%num_col==0):
        num_rows = tot_num//num_col
    else:
        num_rows = (tot_num//num_col) + 1
    
    fig, axs = plt.subplots(figsize=(20,30), nrows = num_rows, ncols = num_col)
    cnt = 0
    fig.tight_layout(pad=4.0)
    for key1 in ts1["Country"].unique():
        ts2 = ts1[ts1["Country"]==key1]
        x1 = ts2["Year"]
        y1 = ts2[variable]

        i1 = cnt//4
        j1 = cnt%4

        axs[i1,j1].set_title(key1)
        axs[i1,j1].set_ylabel(variable)
        axs[i1,j1].set_xlabel("Time")
        axs[i1,j1].plot(x1, y1)
        cnt+=1
    
    
    # Deleting the empty figures
    j1 = -1
    for i1 in range((num_col*num_rows) - tot_num):
        fig.delaxes(axs[-1][-j1])
        j1-=1
    
    return fig, axs