
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
    >>> clean_df = pd.read_csv("clean_df.csv")
    >>> clean_data(clean_df,['Composting%', 'Disposal%', 'Landfill%'])
          Unnamed: 0  Year  ... Population_urban% Population_urban_growth%
    0              0  1990  ...            62.960                 0.332494
    1              1  1990  ...            96.377                 0.386473
    2              2  1990  ...            73.926                 0.945759
    5              5  1990  ...            75.351                 0.400879
    8              8  1990  ...            65.838                -0.990579
    ...          ...   ...  ...               ...                      ...
    1071        1071  2020  ...            60.043                -0.165066
    1072        1072  2020  ...            66.310                 0.931922
    1073        1073  2020  ...            53.760                 0.143424
    1074        1074  2020  ...            55.118                 1.208242
    1075        1075  2020  ...            87.977                 1.028940
    <BLANKLINE>
    [811 rows x 45 columns]
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
    >>> df = pd.read_csv('./Data/MUNW_27112022020812290.csv', usecols=[0,1,3,5,7,9,11,12,14])
    >>> long_to_wide(df,['Year','COU','Country'],'Variable','Value',['Country', 'Year'])
    Variable                               % Composting  ...  Waste from households
    Year COU Country                                     ...                                          
    1990 AUT Austria                             23.171  ...               2504.000
         BEL Belgium                              5.983  ...               2884.000
         CHE Switzerland                          6.341  ...               2733.700
         CHN China (People's Republic of)           NaN  ...                    NaN
         DEU Germany                                NaN  ...                    NaN
    ...                                             ...  ...                    ...
    2020 POL Poland                              12.030  ...              11288.283
         PRT Portugal                            14.289  ...                  0.000
         SVK Slovak Republic                     13.720  ...               1336.614
         SVN Slovenia                            18.142  ...                628.439
         SWE Sweden                              18.262  ...               3775.526
    <BLANKLINE>
    [1200 rows x 31 columns]
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
    >>> df = pd.read_csv('./Data/Data.csv')
    >>> wide_to_long(df,"YR",["index","Country Name","Country Code"],"Year",["Country Code", "Country Name", "Year", "Series Name", "YR"])
           Country Code  ...   YR
    0               ARG  ...   ..
    1               ARG  ...   ..
    2               ARG  ...   ..
    3               ARG  ...   ..
    4               ARG  ...   ..
    ...             ...  ...  ...
    234817          NaN  ...  NaN
    234818          NaN  ...  NaN
    234819          NaN  ...  NaN
    234820          NaN  ...  NaN
    234821          NaN  ...  NaN
    <BLANKLINE>
    [234822 rows x 5 columns]
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
    >>> clean_df = pd.read_csv("clean_df.csv")
    >>> convert_dtype(clean_df,['Female_education_attainment_bach'],['Population_female'],['COU'])
    Unnamed: 0                               int64
    Year                                     int64
    COU                                     string
    Country                                 object
    Composting%                            float64
    Disposal%                              float64
    Landfill%                              float64
    Material_recovery%                     float64
    Recovery%                              float64
    Recycling%                             float64
    Amt_for_recovery_operations            float64
    Composting                             float64
    Electronic_waste                       float64
    Household_waste                        float64
    Landfill                               float64
    Municipal_waste_generated              float64
    Municipal_waste_generated_percapita    float64
    Municipal_waste_generated_1990         float64
    Municipal_waste_generated_2000         float64
    Municipal_waste_treated                float64
    Recycling                              float64
    Total_Incineration                     float64
    Waste_from_households                  float64
    Country_y                               object
    CO2_emissions_percapita                float64
    Female_education_attainment_bach       float64
    Male_education_attainment_bach         float64
    Total_education_attainment_bach        float64
    Energy_use_percapita                   float64
    Forest_area                            float64
    GDP_2015_USD                           float64
    GDP_growth%                            float64
    GDP_percapita_2015_USD                 float64
    GDP_percapita_growth%                  float64
    Education_expense_%_of_gdp             float64
    Education_expense_%_of_total_exp       float64
    Population_density                     float64
    Population_growth%                     float64
    Population_female                        int64
    Population_female_%                    float64
    Population_total                         int64
    Poverty_headcount%_ppp                 float64
    Poverty_headcount%_national_line       float64
    Population_urban%                      float64
    Population_urban_growth%               float64
    dtype: object
    """
    for var in float_list:
        df[var] = df[var].astype('float64', errors='raise')
    for var in int_list:
        df[var] = df[var].astype('int64', errors='raise')
    for var in str_list:
        df[var] = df[var].astype('string')
    return df.dtypes

def density_plots(df,var):
    """
    Function to plot histograms with density plots for each column from a variable list.
    :param df: dataframe/file object name
    :param var: variable name for plotting
    :return: returns a histogram plot for the variable

    >>> clean_df = pd.read_csv("clean_df.csv")
    >>> clean_df = clean_data(clean_df,['Material_recovery%', 'Electronic_waste'])
    >>> density_plots(clean_df, 'Material_recovery%') # doctest: +ELLIPSIS

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
