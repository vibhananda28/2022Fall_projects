
#Import packages
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from scipy.stats import spearmanr
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


#Hypothesis 2 and 3 functions

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
    (<Figure size 1000x500 with 1 Axes>, <AxesSubplot:title={'center':'Recovery% as a function of time'}, ylabel='Recovery%'>)
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
    (<Figure size 2000x3000 with 37 Axes>, array([[<AxesSubplot:title={'center':'Austria'}, xlabel='Time', ylabel='Recovery%'>,
            <AxesSubplot:title={'center':'Belgium'}, xlabel='Time', ylabel='Recovery%'>,
            <AxesSubplot:title={'center':'Switzerland'}, xlabel='Time', ylabel='Recovery%'>,
            <AxesSubplot:title={'center':'Germany'}, xlabel='Time', ylabel='Recovery%'>],
           [<AxesSubplot:title={'center':'Spain'}, xlabel='Time', ylabel='Recovery%'>,
            <AxesSubplot:title={'center':'United Kingdom'}, xlabel='Time', ylabel='Recovery%'>,
            <AxesSubplot:title={'center':'Greece'}, xlabel='Time', ylabel='Recovery%'>,
            <AxesSubplot:title={'center':'Hungary'}, xlabel='Time', ylabel='Recovery%'>],
           [<AxesSubplot:title={'center':'Italy'}, xlabel='Time', ylabel='Recovery%'>,
            <AxesSubplot:title={'center':'Japan'}, xlabel='Time', ylabel='Recovery%'>,
            <AxesSubplot:title={'center':'Korea'}, xlabel='Time', ylabel='Recovery%'>,
            <AxesSubplot:title={'center':'Luxembourg'}, xlabel='Time', ylabel='Recovery%'>],
           [<AxesSubplot:title={'center':'Netherlands'}, xlabel='Time', ylabel='Recovery%'>,
            <AxesSubplot:title={'center':'Norway'}, xlabel='Time', ylabel='Recovery%'>,
            <AxesSubplot:title={'center':'Poland'}, xlabel='Time', ylabel='Recovery%'>,
            <AxesSubplot:title={'center':'Portugal'}, xlabel='Time', ylabel='Recovery%'>],
           [<AxesSubplot:title={'center':'Sweden'}, xlabel='Time', ylabel='Recovery%'>,
            <AxesSubplot:title={'center':'United States'}, xlabel='Time', ylabel='Recovery%'>,
            <AxesSubplot:title={'center':'Mexico'}, xlabel='Time', ylabel='Recovery%'>,
            <AxesSubplot:title={'center':'Türkiye'}, xlabel='Time', ylabel='Recovery%'>],
           [<AxesSubplot:title={'center':'Australia'}, xlabel='Time', ylabel='Recovery%'>,
            <AxesSubplot:title={'center':'Finland'}, xlabel='Time', ylabel='Recovery%'>,
            <AxesSubplot:title={'center':'France'}, xlabel='Time', ylabel='Recovery%'>,
            <AxesSubplot:title={'center':'Slovak Republic'}, xlabel='Time', ylabel='Recovery%'>],
           [<AxesSubplot:title={'center':'Denmark'}, xlabel='Time', ylabel='Recovery%'>,
            <AxesSubplot:title={'center':'Czech Republic'}, xlabel='Time', ylabel='Recovery%'>,
            <AxesSubplot:title={'center':'Estonia'}, xlabel='Time', ylabel='Recovery%'>,
            <AxesSubplot:title={'center':'Ireland'}, xlabel='Time', ylabel='Recovery%'>],
           [<AxesSubplot:title={'center':'Iceland'}, xlabel='Time', ylabel='Recovery%'>,
            <AxesSubplot:title={'center':'Lithuania'}, xlabel='Time', ylabel='Recovery%'>,
            <AxesSubplot:title={'center':'Latvia'}, xlabel='Time', ylabel='Recovery%'>,
            <AxesSubplot:title={'center':'Slovenia'}, xlabel='Time', ylabel='Recovery%'>],
           [<AxesSubplot:title={'center':'Canada'}, xlabel='Time', ylabel='Recovery%'>,
            <AxesSubplot:title={'center':'Chile'}, xlabel='Time', ylabel='Recovery%'>,
            <AxesSubplot:title={'center':'Israel'}, xlabel='Time', ylabel='Recovery%'>,
            <AxesSubplot:title={'center':'Colombia'}, xlabel='Time', ylabel='Recovery%'>],
           [<AxesSubplot:title={'center':'Costa Rica'}, xlabel='Time', ylabel='Recovery%'>,
            <AxesSubplot:>, <AxesSubplot:>, <AxesSubplot:>]], dtype=object))
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
    [['Austria', 0.5572858731924359],...

    >>> clean_df = pd.read_csv("clean_df.csv")
    >>> clean_df = clean_data1(clean_df, ["Forest_area", "Municipal_waste_generated_percapita"])
    >>> calculate_correlation(clean_df, ["Forest_area", "Municipal_waste_generated_percapita"]) # doctest: +ELLIPSIS
    [['Austria', 0.5701890989988877], ['Belgium', -0.5668334009820518],...
    """

    alist = []
    for i1 in df1['Country'].unique():
        df2 = df1.loc[df1['Country'] == i1][['Country', vlist[0], vlist[1]]]
        df2 = df2.dropna()
        corr, _ = spearmanr(df2[vlist[0]], df2[vlist[1]])
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
    ([['Belgium', -0.41086691086691085], ['Germany', -0.35014532910447393],...
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
    (<Figure size 2000x2000 with 22 Axes>, array([[<AxesSubplot:title={'center':'Belgium'}, xlabel='Time', ylabel='Recovery%'>,
            <AxesSubplot:title={'center':'Belgium'}, xlabel='Time', ylabel='Municipal_waste_generated_percapita'>,
            <AxesSubplot:title={'center':'Germany'}, xlabel='Time', ylabel='Recovery%'>,
            <AxesSubplot:title={'center':'Germany'}, xlabel='Time', ylabel='Municipal_waste_generated_percapita'>],
           [<AxesSubplot:title={'center':'Spain'}, xlabel='Time', ylabel='Recovery%'>,
            <AxesSubplot:title={'center':'Spain'}, xlabel='Time', ylabel='Municipal_waste_generated_percapita'>,
            <AxesSubplot:title={'center':'United Kingdom'}, xlabel='Time', ylabel='Recovery%'>,
            <AxesSubplot:title={'center':'United Kingdom'}, xlabel='Time', ylabel='Municipal_waste_generated_percapita'>],
           [<AxesSubplot:title={'center':'Hungary'}, xlabel='Time', ylabel='Recovery%'>,
            <AxesSubplot:title={'center':'Hungary'}, xlabel='Time', ylabel='Municipal_waste_generated_percapita'>,
            <AxesSubplot:title={'center':'Japan'}, xlabel='Time', ylabel='Recovery%'>,
            <AxesSubplot:title={'center':'Japan'}, xlabel='Time', ylabel='Municipal_waste_generated_percapita'>],
           [<AxesSubplot:title={'center':'Korea'}, xlabel='Time', ylabel='Recovery%'>,
            <AxesSubplot:title={'center':'Korea'}, xlabel='Time', ylabel='Municipal_waste_generated_percapita'>,
            <AxesSubplot:title={'center':'Netherlands'}, xlabel='Time', ylabel='Recovery%'>,
            <AxesSubplot:title={'center':'Netherlands'}, xlabel='Time', ylabel='Municipal_waste_generated_percapita'>],
           [<AxesSubplot:title={'center':'Australia'}, xlabel='Time', ylabel='Recovery%'>,
            <AxesSubplot:title={'center':'Australia'}, xlabel='Time', ylabel='Municipal_waste_generated_percapita'>,
            <AxesSubplot:title={'center':'Estonia'}, xlabel='Time', ylabel='Recovery%'>,
            <AxesSubplot:title={'center':'Estonia'}, xlabel='Time', ylabel='Municipal_waste_generated_percapita'>],
           [<AxesSubplot:title={'center':'Costa Rica'}, xlabel='Time', ylabel='Recovery%'>,
            <AxesSubplot:title={'center':'Costa Rica'}, xlabel='Time', ylabel='Municipal_waste_generated_percapita'>,
            <AxesSubplot:>, <AxesSubplot:>]], dtype=object))

    >>> clean_df = pd.read_csv("clean_df.csv")
    >>> clean_df = clean_data1(clean_df, ["Forest_area", "Municipal_waste_generated_percapita"])
    >>> alist = calculate_correlation(clean_df, ["Forest_area", "Municipal_waste_generated_percapita"])
    >>> corr_list1, corr_list2, corr_list3 = country_groups(alist)
    >>> group_time_series(clean_df, corr_list1, (20,20), ["Forest_area", "Municipal_waste_generated_percapita"])# doctest: +ELLIPSIS
    (<Figure size 2000x2000 with 32 Axes>, array([[<AxesSubplot:title={'center':'Belgium'}, xlabel='Time', ylabel='Forest_area'>,
            <AxesSubplot:title={'center':'Belgium'}, xlabel='Time', ylabel='Municipal_waste_generated_percapita'>,
            <AxesSubplot:title={'center':'Germany'}, xlabel='Time', ylabel='Forest_area'>,
            <AxesSubplot:title={'center':'Germany'}, xlabel='Time', ylabel='Municipal_waste_generated_percapita'>],
           [<AxesSubplot:title={'center':'Spain'}, xlabel='Time', ylabel='Forest_area'>,
            <AxesSubplot:title={'center':'Spain'}, xlabel='Time', ylabel='Municipal_waste_generated_percapita'>,
            <AxesSubplot:title={'center':'United Kingdom'}, xlabel='Time', ylabel='Forest_area'>,
            <AxesSubplot:title={'center':'United Kingdom'}, xlabel='Time', ylabel='Municipal_waste_generated_percapita'>],
           [<AxesSubplot:title={'center':'Hungary'}, xlabel='Time', ylabel='Forest_area'>,
            <AxesSubplot:title={'center':'Hungary'}, xlabel='Time', ylabel='Municipal_waste_generated_percapita'>,
            <AxesSubplot:title={'center':'Japan'}, xlabel='Time', ylabel='Forest_area'>,
            <AxesSubplot:title={'center':'Japan'}, xlabel='Time', ylabel='Municipal_waste_generated_percapita'>],
           [<AxesSubplot:title={'center':'Portugal'}, xlabel='Time', ylabel='Forest_area'>,
            <AxesSubplot:title={'center':'Portugal'}, xlabel='Time', ylabel='Municipal_waste_generated_percapita'>,
            <AxesSubplot:title={'center':'Mexico'}, xlabel='Time', ylabel='Forest_area'>,
            <AxesSubplot:title={'center':'Mexico'}, xlabel='Time', ylabel='Municipal_waste_generated_percapita'>],
           [<AxesSubplot:title={'center':'Türkiye'}, xlabel='Time', ylabel='Forest_area'>,
            <AxesSubplot:title={'center':'Türkiye'}, xlabel='Time', ylabel='Municipal_waste_generated_percapita'>,
            <AxesSubplot:title={'center':'Australia'}, xlabel='Time', ylabel='Forest_area'>,
            <AxesSubplot:title={'center':'Australia'}, xlabel='Time', ylabel='Municipal_waste_generated_percapita'>],
           [<AxesSubplot:title={'center':'Estonia'}, xlabel='Time', ylabel='Forest_area'>,
            <AxesSubplot:title={'center':'Estonia'}, xlabel='Time', ylabel='Municipal_waste_generated_percapita'>,
            <AxesSubplot:title={'center':'Slovenia'}, xlabel='Time', ylabel='Forest_area'>,
            <AxesSubplot:title={'center':'Slovenia'}, xlabel='Time', ylabel='Municipal_waste_generated_percapita'>],
           [<AxesSubplot:title={'center':'Canada'}, xlabel='Time', ylabel='Forest_area'>,
            <AxesSubplot:title={'center':'Canada'}, xlabel='Time', ylabel='Municipal_waste_generated_percapita'>,
            <AxesSubplot:title={'center':'Israel'}, xlabel='Time', ylabel='Forest_area'>,
            <AxesSubplot:title={'center':'Israel'}, xlabel='Time', ylabel='Municipal_waste_generated_percapita'>],
           [<AxesSubplot:title={'center':'Colombia'}, xlabel='Time', ylabel='Forest_area'>,
            <AxesSubplot:title={'center':'Colombia'}, xlabel='Time', ylabel='Municipal_waste_generated_percapita'>,
            <AxesSubplot:title={'center':'Costa Rica'}, xlabel='Time', ylabel='Forest_area'>,
            <AxesSubplot:title={'center':'Costa Rica'}, xlabel='Time', ylabel='Municipal_waste_generated_percapita'>]],
          dtype=object))
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