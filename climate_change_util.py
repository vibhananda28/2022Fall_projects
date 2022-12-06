
#Import packages
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import datetime
from IPython.display import display, HTML


def clean_data(dataframe,column_list):
    """
    Function to drop rows with missing countries or variable names. Also, replaces ".." values with nans.
    :param dataframe: dataframe/file object name
    :param column_list: column with missing values
    :return: returns cleaned dataframe
    """
    for column in column_list:
        dataframe = dataframe[dataframe[column].notna()]
    dataframe = dataframe.replace("..", np.nan)
    return dataframe


def long_to_wide(df,pivot_index_list,pivot_column_var,pivot_value_var,sort_by_list):
    """
    Function to reshape a dataframe from long to wide shape.
    :param df: dataframe/file object name
    :param pivot_index_list: variable list for pivoting
    :param pivot_column_var: column name for pivoting
    :param pivot_value_var: variable name with values for pivoting to wide shape
    :param sort_by_list: variable list to sort the dataframe by (not implemented yet)
    :return: returns wide shaped dataframe
    """
    df_final = pd.pivot(df,
                        index=pivot_index_list, #['Year', 'COU', 'Country']
                        columns=pivot_column_var, #'Variable'
                        values=pivot_value_var) #'Value'
    #df_final.sort_values(by=sort_by_list, #['Country', 'Year']
                         #ascending=[True, True])
    return df_final


def wide_to_long(df,stubname_var,i_list,j_var,reindex_list):
    """
    Function to reshape a dataframe from wide to long shape.
    :param df: dataframe/file object name
    :param stubname_var: variable stubname to use for reshaping
    :param i_list: variable list to keep after reshaping
    :param j_var: variable name for reshaping
    :param reindex_list: variable list for resetting index after reshaping
    :return: returns long shaped dataframe
    """
    df_clean = df.rename(columns=lambda x: re.sub(r'^\d+ \[(YR\d+)\]$', r'\1', x))
    df_reshaped = pd.wide_to_long(df_clean.reset_index(), stubnames=stubname_var,#["YR"]
                                  i=i_list, #["index", "Country Name", "Country Code"]
                                  j=j_var, #"Year"
                                  suffix='\d+')
    df_reshaped = df_reshaped.reset_index().reindex(reindex_list, #["Country Code", "Country Name", "Year", "Series Name", "YR"]
                                                     axis=1)
    return df_reshaped


def convert_dtype(df, float_list, int_list, str_list):
    """
    Function to convert list of variables that are by default of object type, to relevant type.
    :param df: dataframe/file object name
    :param float_list: variable list to be converted to float type
    :param int_list: variable list to be converted to integer type
    :param str_list: variable list to be converted to string type
    :return:
    """
    for var in float_list:
        df[var] = df[var].astype('float64', errors='raise')
    for var in int_list:
        df[var] = df[var].astype('int64', errors='raise')
    for var in str_list:
        df[var] = df[var].astype('string')


def density_plots(df,var):
    """
    Function to plot histograms with density plots for each column from a variable list.
    :param df: dataframe/file object name
    :param var: variable name for plotting
    :return: returns a histogram plot for the variable
    """
    sns.displot(data=df[var], kde=True)
    plt.title(var)
    plt.show()


def median_trend_plots(df,year_var,plot_var):
    """
    Function to plot median trend line plots for each column from a variable list.
    :param df: dataframe/file object name
    :param year_var: time variable
    :param plot_var: variable name for plotting
    :return: returns a median trend line plot for the variable
    """
    df.groupby(year_var)[plot_var].median().plot()
    plt.title(plot_var)
    plt.show()
