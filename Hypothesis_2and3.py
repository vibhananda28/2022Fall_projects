# Importing Libraries
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

import datetime
from IPython.display import display, HTML

# # Uncomment these lines if pycharm shows an error Like "AttributeError: module 'backend_interagg' has no attribute 'FigureCanvas'"
# # Link - https://stackoverflow.com/questions/73745245/error-using-matplotlib-in-pycharm-has-no-attribute-figurecanvas
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
# # Uncomment these lines if pycharm shows an error Like "AttributeError: module 'backend_interagg' has no attribute 'FigureCanvas'"

# Hypothesis 2 and 3 common function
def clean_data1(df1, vlist):
    """
    Function to clean dataframe and returned the cleaned dataframe. Function removes rows with very less data
    or all nan values present in the columns used for analysis.
    :param df1: Input original data frame
    :param vlist: List of variables to consider while cleaning dataframe
    :return: Cleaned dataframe

    >>> clean_df = pd.read_csv("clean_df.csv")
    >>> clean_data1(clean_df, ["Recovery%", "Municipal_waste_generated_percapita"]) # doctest: +ELLIPSIS
          Unnamed: 0  Year  ... Population_urban% Population_urban_growth%
    0              0  1990  ...            62.960                 0.332494
    ...
    >>> clean_df = pd.read_csv("clean_df.csv")
    >>> clean_data1(clean_df, ["Forest_area", "Municipal_waste_generated_percapita"]) # doctest: +ELLIPSIS
          Unnamed: 0  Year  ... Population_urban% Population_urban_growth%
    0              0  1990  ...            62.960                 0.332494
    ...
    """

    alist = df1["Country"].unique()
    for i1 in alist:
        df2 = df1.loc[df1["Country"] == i1]
        for j1 in vlist:
            len_var = len(df2[j1].unique())
            if (len_var<3):
                del_row = df1.loc[df1["Country"] == i1].index
                df1 = df1.drop(del_row)
                break

    return df1

# Common function for hypothesis 2 and 3
def time_series(ts1, variable):
    """
    The function generates the plots of time series for a given variable.
    :param ts1: The input data frame having the variable data and Years data
    :param variable: The variable for which plot needs to be generated over time
    :return: figure and axis of the plot

    >>> clean_df = pd.read_csv("clean_df.csv") # doctest: +ELLIPSIS
    >>> clean_df = clean_data1(clean_df, ["Recovery%", "Municipal_waste_generated_percapita"]) # doctest: +ELLIPSIS
    >>> time_series(clean_df, "Recovery%")
    (<Figure size 1000x500 with 1 Axes>, <AxesSubplot: title={'center': 'Recovery% as a function of time'}, ylabel='Recovery%'>)
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    for key1 in ts1["Country"].unique():
        ts2 = ts1[ts1["Country"] == key1]
        x1 = ts2["Year"]
        y1 = ts2[variable]
        plt.plot(x1, y1, label=key1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, shadow=True, ncol=5)
    plt.ylabel(variable)

    plt.title(str(variable + " as a function of time"))
    return fig, ax

# Common function for hypothesis 2 and 3
def time_series_subplots(ts1, variable):
    """
    This function is used to create various subplots of the time series for a given variable over a period
    of time in years.
    :param ts1: The input dataframe
    :param variable: The variable for which time plots are plotted
    :return: figure and axis of the plot

    >>> clean_df = pd.read_csv("clean_df.csv")
    >>> clean_df = clean_data1(clean_df, ["Recovery%", "Municipal_waste_generated_percapita"])
    >>> time_series_subplots(clean_df, "Recovery%") # doctest: +ELLIPSIS
    (<Figure size 2000x3000 with 37 Axes>, array([[<AxesSubplot: title={'center': 'Austria'}, xlabel='Time', ylabel='Recovery%'>,
    ...
    """
    tot_num = len(ts1["Country"].unique())
    num_col = 4  # number of figures to display in one row
    # Calculating number of rows
    num_row = tot_num
    if (tot_num % num_col == 0):
        num_rows = tot_num // num_col
    else:
        num_rows = (tot_num // num_col) + 1

    fig, axs = plt.subplots(figsize=(20, 30), nrows=num_rows, ncols=num_col)
    cnt = 0
    fig.tight_layout(pad=4.0)
    for key1 in ts1["Country"].unique():
        ts2 = ts1[ts1["Country"] == key1]
        x1 = ts2["Year"]
        y1 = ts2[variable]

        i1 = cnt // num_col
        j1 = cnt % num_col

        axs[i1, j1].set_title(key1)
        axs[i1, j1].set_ylabel(variable)
        axs[i1, j1].set_xlabel("Time")
        axs[i1, j1].plot(x1, y1)
        cnt += 1

    # Deleting the empty figures
    j1 = -1
    for i1 in range((num_col * num_rows) - tot_num):
        fig.delaxes(axs[-1][j1])
        j1 -= 1

    return fig, axs

# Hypothesis 2 and 3 common function
def calculate_correlation(df1, vlist):
    """
    Calculates the correlation between the variables given in vlist
    :param df1: Input dataframe
    :param vlist: List of variables to consider for correlation
    :return: The correlation values beween variables in vlist for different countries

    >>> clean_df = pd.read_csv("clean_df.csv")
    >>> clean_df = clean_data1(clean_df, ["Recovery%", "Municipal_waste_generated_percapita"])
    >>> calculate_correlation(clean_df, ["Recovery%", "Municipal_waste_generated_percapita"]) # doctest: +ELLIPSIS
    [['Austria', 0.8118070209887488],...

    >>> clean_df = pd.read_csv("clean_df.csv")
    >>> clean_df = clean_data1(clean_df, ["Forest_area", "Municipal_waste_generated_percapita"])
    >>> calculate_correlation(clean_df, ["Forest_area", "Municipal_waste_generated_percapita"]) # doctest: +ELLIPSIS
    [['Austria', 0.8267992985009449], ['Belgium', -0.32137698966537276],...
    """

    alist = []
    for i1 in df1['Country'].unique():
        df2 = df1.loc[df1['Country'] == i1][['Country', vlist[0], vlist[1]]]
        df2 = df2.dropna()
        corr, _ = pearsonr(df2[vlist[0]], df2[vlist[1]])
        alist.append([i1, corr])

    return alist

# Common function for hypothesis 2 and 3
def country_groups(alist):
    """
    Calculates the different groups of countries based on the correlation values for them.
    :param alist: List of countries with the correlation values
    :return: 3 groups of countries based on the correlation values

    >>> clean_df = pd.read_csv("clean_df.csv")
    >>> clean_df = clean_data1(clean_df, ["Recovery%", "Municipal_waste_generated_percapita"])
    >>> alist = calculate_correlation(clean_df, ["Recovery%", "Municipal_waste_generated_percapita"])
    >>> country_groups(alist)# doctest: +ELLIPSIS
    ([['Germany', -0.3385490316420527], ['Spain', -0.33825072508548054],...
    """
    alist1 = []  # for countries with high negative correlation
    alist2 = []  # for countries with high positive correlation
    alist3 = []  # for countries with low correlation

    for i1 in range(len(alist)):
        if (alist[i1][1] < -0.25):
            alist1.append(alist[i1])
        elif (alist[i1][1] > 0.25):
            alist2.append(alist[i1])
        else:
            alist3.append(alist[i1])

    return alist1, alist2, alist3

# Hypothesis 2 and 3 common function
def group_time_series(ts1, corr_list, figure_size, vlist):
    """
    Creates a plot for a given group (corr_list) that compares the variables in vlist
    variables over time for different countries
    :param ts1: The input dataframe having data
    :param corr_list: The input group of countries that were group based on their correlation values
    :param figure_size: The figure size of the whole plot
    :param vlist: List of variables to consider while generating group time series plots
    :return: Figure and axis of the plot

    >>> clean_df = pd.read_csv("clean_df.csv")
    >>> clean_df = clean_data1(clean_df, ["Recovery%", "Municipal_waste_generated_percapita"])
    >>> alist = calculate_correlation(clean_df, ["Recovery%", "Municipal_waste_generated_percapita"])
    >>> corr_list1, corr_list2, corr_list3 = country_groups(alist)
    >>> group_time_series(clean_df, corr_list1, (20,20), ["Recovery%", "Municipal_waste_generated_percapita"])# doctest: +ELLIPSIS
    (<Figure size 2000x2000 with 20 Axes>, array([[<AxesSubplot: title={'center': 'Germany'}, xlabel='Time', ylabel='Recovery%'>,...

    >>> clean_df = pd.read_csv("clean_df.csv")
    >>> clean_df = clean_data1(clean_df, ["Forest_area", "Municipal_waste_generated_percapita"])
    >>> alist = calculate_correlation(clean_df, ["Forest_area", "Municipal_waste_generated_percapita"])
    >>> corr_list1, corr_list2, corr_list3 = country_groups(alist)
    >>> group_time_series(clean_df, corr_list1, (20,20), ["Forest_area", "Municipal_waste_generated_percapita"])# doctest: +ELLIPSIS
    (<Figure size 2000x2000 with 34 Axes>, array([[<AxesSubplot: title={'center': 'Belgium'}, xlabel='Time', ylabel='Forest_area'>,...
    """
    tot_num = len(corr_list) * 2
    num_col = 4  # number of figures to display in one row
    # Calculating number of rows
    num_row = tot_num
    if (tot_num % num_col == 0):
        num_rows = tot_num // num_col
    else:
        num_rows = (tot_num // num_col) + 1

    fig, axs = plt.subplots(figsize=figure_size, nrows=num_rows, ncols=num_col)
    cnt = 0
    fig.tight_layout(pad=4.0)

    for key1 in corr_list:
        ts2 = ts1[ts1["Country"] == key1[0]]
        x1 = ts2["Year"]
        y1 = ts2[vlist[0]]
        y2 = ts2[vlist[1]]

        i1 = cnt // num_col
        j1 = cnt % num_col

        axs[i1, j1].set_title(key1[0])
        axs[i1, j1].set_ylabel(vlist[0])
        axs[i1, j1].set_xlabel("Time")
        axs[i1, j1].scatter(x1, y1)

        axs[i1, j1 + 1].set_title(key1[0])
        axs[i1, j1 + 1].set_ylabel(vlist[1])
        axs[i1, j1 + 1].set_xlabel("Time")
        axs[i1, j1 + 1].scatter(x1, y2, color="orange")
        cnt += 2

    # Deleting the empty figures
    j1 = -1
    for i1 in range((num_col * num_rows) - tot_num):
        fig.delaxes(axs[-1][j1])
        j1 -= 1

    return fig, axs


# if __name__ == "__main__":
#
#     # clean_df = pd.read_csv("clean_df.csv")
#     # clean_df = clean_data1(clean_df)
#     # fig, ax = time_series(clean_df, "Recovery%")
#     # ax.plot()
#     # plt.show()