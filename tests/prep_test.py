import os
import sys
import pandas as pd

# import utils
here = os.path.dirname(__file__)
sys.path.append(os.path.join(here, '..'))
from utils import preprocessing as prep
from utils import splitting as splt

url = 'https://www.canada.ca/content/dam/ircc/documents/json/ee_rounds_123_en.json'
df = prep.create_df_from_website(url)

def test_create_df_from_website():

    expected_dict = {
                    'round type': 
                    {pd.Timestamp('2015-03-27 00:00:00'): 'No Program Specified',
                    pd.Timestamp('2015-03-20 00:00:00'): 'No Program Specified',
                    pd.Timestamp('2015-02-27 00:00:00'): 'No Program Specified',
                    pd.Timestamp('2015-02-20 00:00:00'): 'Canadian Experience Class'},
                    'invitations issued': 
                    {pd.Timestamp('2015-03-27 00:00:00'): 1637,
                    pd.Timestamp('2015-03-20 00:00:00'): 1620,
                    pd.Timestamp('2015-02-27 00:00:00'): 1187,
                    pd.Timestamp('2015-02-20 00:00:00'): 849},
                    'CRS cutoff': 
                    {pd.Timestamp('2015-03-27 00:00:00'): 453,
                    pd.Timestamp('2015-03-20 00:00:00'): 481,
                    pd.Timestamp('2015-02-27 00:00:00'): 735,
                    pd.Timestamp('2015-02-20 00:00:00'): 808}
                    }
    
    actual_dict = df[-6:-2].copy().to_dict()

    assert expected_dict == actual_dict

def test_cleanup_df_general_rounds():
    expected_dict = {
        'round type': 
        {pd.Timestamp('2015-03-27 00:00:00'): 'General',
        pd.Timestamp('2015-03-20 00:00:00'): 'General'},
        'invitations issued': 
        {pd.Timestamp('2015-03-27 00:00:00'): 1637,
        pd.Timestamp('2015-03-20 00:00:00'): 1620},
        'CRS cutoff': 
        {pd.Timestamp('2015-03-27 00:00:00'): 453,
        pd.Timestamp('2015-03-20 00:00:00'): 481}
        }
    
    actual_dict = prep.cleanup_df_general_rounds(df[-6:-2].copy()).to_dict()

    assert expected_dict == actual_dict

def test_calculate_date_vars():
    expected_dict = {
                    'round type': 'General',
                    'invitations issued': 1980,
                    'CRS cutoff': 524,
                    'month': 3,
                    'year': 2024,
                    'dayofweek': 0,
                    'quarter': 1
                    }
    
    actual_dict = prep.calculate_date_vars(df.copy()).loc['2024-03-25'].to_dict()

    assert expected_dict == actual_dict

def test_calculate_rolling_averages():
    expected_dict = {
                    'round type': 'General',
                    'invitations issued': 1980,
                    'CRS cutoff': 524,
                    'roll_30D': 529.5,
                    'roll_60D': 533.75,
                    'roll_90D': 537.3333333333334,
                    'roll_180D': 540.875
                    }
    
    actual_dict = prep.calculate_rolling_averages(df[df['round type']=='General'].copy()).loc['2024-03-25'].to_dict()

    assert expected_dict == actual_dict

def test_calculate_offset_windows():
    expected_dict = {
                    'round type': 'General',
                    'invitations issued': 1980,
                    'CRS cutoff': 524,
                    'CRS-1': 525.0,
                    'dt-1': 13.0,
                    'CRS-2': 534.0,
                    'dt-2': 26.0,
                    'CRS-3': 535.0,
                    'dt-3': 41.0
                    }
    
    actual_dict = prep.calculate_offset_windows(df[df['round type']=='General'].copy()).loc['2024-03-25'].to_dict()

    assert expected_dict == actual_dict

def test_calculate_independent_vars():
    expected_dict = {
        'round type': 'General',
        'invitations issued': 1470,
        'CRS cutoff': 534,
        'roll_30D': 538.0,
        'roll_60D': 541.25,
        'roll_90D': 544.6666666666666,
        'roll_180D': 544.6666666666666,
        'CRS-1': 535.0,
        'dt-1': 15.0,
        'CRS-2': 541.0,
        'dt-2': 28.0,
        'CRS-3': 543.0,
        'dt-3': 36.0,
        'month': 2,
        'year': 2024,
        'dayofweek': 2,
        'quarter': 1
        }
    
    actual_dict = prep.calculate_independent_vars(df[df['round type']=='General'].copy()).loc['2024-02-28'].to_dict()

    assert expected_dict == actual_dict

def test_preprocess_query_date():
    expected_dict = {
                    'round type': 'General',
                    'invitations issued': 0.0,
                    'CRS cutoff': 0.0,
                    'roll_30D': 529.0,
                    'roll_60D': 529.0,
                    'roll_90D': 529.0,
                    'roll_180D': 535.0,
                    'CRS-1': 529.0,
                    'dt-1': 83.0,
                    'CRS-2': 549.0,
                    'dt-2': 96.0,
                    'CRS-3': 524.0,
                    'dt-3': 112.0,
                    'month': 7,
                    'year': 2024,
                    'dayofweek': 0,
                    'quarter': 3}
    
    actual_dict = prep.preprocess_query_date('2024-07-15').loc['2024-07-15'].to_dict()

    assert expected_dict == actual_dict

def test_split_test_train():
    expected_dtrain = {'round type': 
                       {pd.Timestamp('2015-02-27 00:00:00'): 'No Program Specified',
                        pd.Timestamp('2015-02-20 00:00:00'): 'Canadian Experience Class'},
                        'invitations issued': 
                        {pd.Timestamp('2015-02-27 00:00:00'): 1187,
                        pd.Timestamp('2015-02-20 00:00:00'): 849},
                        'CRS cutoff': 
                        {pd.Timestamp('2015-02-27 00:00:00'): 735,
                        pd.Timestamp('2015-02-20 00:00:00'): 808}}
    
    expected_dtest = {'round type': 
                    {pd.Timestamp('2015-03-27 00:00:00'): 'No Program Specified',
                    pd.Timestamp('2015-03-20 00:00:00'): 'No Program Specified'},
                    'invitations issued': 
                    {pd.Timestamp('2015-03-27 00:00:00'): 1637,
                    pd.Timestamp('2015-03-20 00:00:00'): 1620},
                    'CRS cutoff': 
                    {pd.Timestamp('2015-03-27 00:00:00'): 453,
                    pd.Timestamp('2015-03-20 00:00:00'): 481}}
    
    actual_dtrain, actual_dtest = splt.split_test_train(
        df[-6:-2].copy(),
        split_date='01-Mar-2015'
        )
    
    actual_dtrain = actual_dtrain.to_dict()
    actual_dtest = actual_dtest.to_dict()
    
    assert expected_dtrain == actual_dtrain
    assert expected_dtest == actual_dtest

def test_create_features():
    expected_vals = [779, 886]
    
    X, Y = splt.create_features(df[-1:],x_labels='invitations issued',y_labels='CRS cutoff')

    actual_vals = [
        X.loc['2015-01-31'],
        Y.loc['2015-01-31']
    ]

    assert expected_vals == actual_vals
