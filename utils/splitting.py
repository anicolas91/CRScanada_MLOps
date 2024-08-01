'''
Functions needed for splitting time series data into test and training
'''

def split_test_train(df,split_date='01-Jan-2023'):
    ''' 
    this function splits the time series into test and train data based on a cutoff date
    '''
    # get date column
    dates = df.index.to_series()
    # split it
    data_train = df.loc[dates <= split_date].copy()
    data_test = df.loc[dates > split_date].copy()
    return data_train, data_test

def create_features(data,x_labels=None,y_labels=None):
    ''' 
    This function creates X and Y features
    '''
    # set up fresh lists
    if x_labels is None:
        x_labels = ['dt-1','CRS-1']
    if y_labels is None:
        y_labels = ['CRS cutoff']
    # split in to X and Y
    X = data[x_labels]
    Y = data[y_labels]
    return X, Y
