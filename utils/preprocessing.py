'''
Preprocessing functions for web scraping, cleanup, and variable preprocessing 
'''

import requests
import pandas as pd

def create_df_from_website(url):
    ''' 
    This function scrapes data from the website and converts it to a usable pandas framework.
    '''
    # read in json filetype
    r = requests.get(url)
    rounds = r.json()['rounds']
    # remove commas from integer strings
    for r in rounds:
        for i in r:
            r[i] = r[i].replace(",", "")
    # create pandas df from dictionary
    df = pd.DataFrame.from_dict(rounds,dtype='string')
    # specify column names as needed
    columns={"drawNumber": "id", 
             "drawDate": "date", 
             "drawName": "round type", 
             "drawSize":"invitations issued", 
             "drawCRS": "CRS cutoff",
             "drawText2": "type issued",
             "drawCutOff": "tie break rule",
             "dd18": "total applications",
             "dd1":  "crs_range_601_1200",
             "dd2":  "crs_range_501_600",
             "dd3":  "crs_range_451_500",
             "dd4":  "crs_range_491_500",
             "dd5":  "crs_range_481_490",
             "dd6":  "crs_range_471_480",
             "dd7":  "crs_range_461_470",
             "dd8":  "crs_range_451_460",
             "dd9":  "crs_range_401_450",
             "dd10": "crs_range_441_450",
             "dd11": "crs_range_431_440",
             "dd12": "crs_range_421_430",
             "dd13": "crs_range_411_420",
             "dd14": "crs_range_401_410",
             "dd15": "crs_range_351_400",
             "dd16": "crs_range_301_350",
             "dd17": "crs_range_000_300",
             }
    # specify dtypes as needed
    dtypes={#"drawNumber": "id", 
            "drawDate": "datetime64[ns]", 
            #"drawName": "round type", 
            "drawSize":"int64", 
            "drawCRS": "int64",
            #"drawText2": "type issued",
            #"drawCutOff": "tie break rule",
            "dd18": "int64",
            "dd1":  "int64",
            "dd2":  "int64",
            "dd3":  "int64",
            "dd4":  "int64",
            "dd5":  "int64",
            "dd6":  "int64",
            "dd7":  "int64",
            "dd8":  "int64",
            "dd9":  "int64",
            "dd10": "int64",
            "dd11": "int64",
            "dd12": "int64",
            "dd13": "int64",
            "dd14": "int64",
            "dd15": "int64",
            "dd16": "int64",
            "dd17": "int64",
            }
    #set names and dtypes
    df = df.astype(dtype=dtypes)
    df = df.rename(columns=columns)
    # extract only columns of interest
    vars_to_keep=['date','round type','invitations issued','CRS cutoff']
    df= df[vars_to_keep]
    #setup date itself as the index
    df = df.set_index('date')

    return df

def cleanup_df_general_rounds(df):
    ''' 
    this function extracts only the general rounds and cleans up data slightly
    '''
    # combine general rounds into one type
    df["round type"] = df["round type"].replace({'No Program Specified': 'General'})
    # extract gral rounds only and remove outliers
    df = df[(df["round type"] == "General") & ((df["CRS cutoff"] < 700) & (df["CRS cutoff"] > 100))]

    return df

def calculate_date_vars(df):
    ''' 
    this function calculates extra variables based on date
    '''
    # get date column
    dates = df.index.to_series()
    # get vars
    df['month'] = dates.dt.month
    df['year'] = dates.dt.year
    df['dayofweek'] = dates.dt.dayofweek
    df['quarter'] = dates.dt.quarter

    return df

def calculate_rolling_averages(df,roll_times=['30D','60D','90D','180D']):
    ''' 
    This function calculates the rolling averages point wise for the uneven datetimes
    '''
    # flip data from oldest to newest
    df = df.iloc[::-1].copy()
    # get mean CRS in the past N months prior to this value
    for r in roll_times:
        df['roll_'+r] = df['CRS cutoff'].rolling(r, min_periods=1,closed='left').mean()
    # flip from new to old
    df = df.iloc[::-1]

    return df

def calculate_offset_windows(df,offset_value=[-1,-2,-3]):
    # remove nans so that you dont use their datetimes
    df = df.dropna().copy()
    # get date column
    dates = df.index.to_series()
    # get window per each offset values
    for offset in offset_value:
        #add previous CRS
        df['CRS'+str(offset)] = df['CRS cutoff'].shift(offset)
        #calculate quickly dt
        df['dt'+str(offset)] = dates.diff(periods=offset).abs().dt.days#.shift(offset)
    
    return df.dropna()

def calculate_independent_vars(df):
    ''' 
    this function calculates all independent variables in one go.
    '''
    # calculate rolling averages
    df = calculate_rolling_averages(df)
    # calculate offset windows
    df = calculate_offset_windows(df)
    # calculate data variables
    df = calculate_date_vars(df)

    return df
