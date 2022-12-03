# Importing Libraries
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

import datetime
from IPython.display import display, HTML

def clean_data_hyp3(df1):
    alist = df1["Country"].unique()
    for i1 in alist:
        df2 = df1.loc[df1["Country"]==i1]
        len_population = len(df2["Population_total"].unique())
        len_municipal = len(df2["Municipal_waste_generated_percapita"].unique())
        len_forest = len(df2["Forest_area"].unique())
        
        if(len_population<3 or len_municipal<3 or len_forest<3):
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

        i1 = cnt//num_col
        j1 = cnt%num_col

        axs[i1,j1].set_title(key1)
        axs[i1,j1].set_ylabel(variable)
        axs[i1,j1].set_xlabel("Time")
        axs[i1,j1].plot(x1, y1)
        cnt+=1
    
    
    # Deleting the empty figures
    j1 = -1
    for i1 in range((num_col*num_rows) - tot_num):
        fig.delaxes(axs[-1][j1])
        j1-=1
    
    return fig, axs

def calculate_correlation_hyp3(df1):
    alist = []
    for i1 in df1['Country'].unique():
        df2 = df1.loc[df1['Country']==i1][['Country', "Forest_area", "Municipal_waste_generated_percapita"]]
        df2 = df2.dropna()
        corr, _ = pearsonr(df2["Forest_area"], df2["Municipal_waste_generated_percapita"])
        alist.append([i1, corr])
    
    return alist

def country_groups(alist):
    alist1 = [] # for countries with high negative correlation
    alist2 = [] # for countries with high positive correlation
    alist3 = [] # for countries with low correlation
    
    for i1 in range(len(alist)):
        if(alist[i1][1]<-0.25):
            alist1.append(alist[i1])
        elif(alist[i1][1]>0.25):
            alist2.append(alist[i1])
        else:
            alist3.append(alist[i1])

    return alist1, alist2, alist3

def group_time_series_hyp3(ts1, corr_list, figure_size):
    tot_num = len(corr_list)*2
    num_col = 4 # number of figures to display in one row
    #Calculating number of rows
    num_row = tot_num
    if(tot_num%num_col==0):
        num_rows = tot_num//num_col
    else:
        num_rows = (tot_num//num_col) + 1
    
    fig, axs = plt.subplots(figsize=figure_size, nrows = num_rows, ncols = num_col)
    cnt = 0
    fig.tight_layout(pad=4.0)
    
    for key1 in corr_list:
        ts2 = ts1[ts1["Country"]==key1[0]]
        x1 = ts2["Year"]
        y1 = ts2["Forest_area"]
        y2 = ts2["Municipal_waste_generated_percapita"]

        i1 = cnt//num_col
        j1 = cnt%num_col

        axs[i1,j1].set_title(key1[0])
        axs[i1,j1].set_ylabel("Forest_area")
        axs[i1,j1].set_xlabel("Time")
        axs[i1,j1].scatter(x1, y1)
        
        axs[i1,j1+1].set_title(key1[0])
        axs[i1,j1+1].set_ylabel("Municipal_waste_generated_percapita")
        axs[i1,j1+1].set_xlabel("Time")
        axs[i1,j1+1].scatter(x1, y2, color = "orange")
        cnt+=2
    
    
    # Deleting the empty figures
    j1 = -1
    for i1 in range((num_col*num_rows) - tot_num):
        fig.delaxes(axs[-1][j1])
        j1-=1
    
    return fig, axs