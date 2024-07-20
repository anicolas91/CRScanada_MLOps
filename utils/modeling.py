'''
Functions needed for initializing and optimizing modeling instances
'''

import os
import pickle
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

def get_best_xgboost_model(X_train, y_train, X_test, y_test,verbose=True):
    ''' 
    this function is an auxiliary as it gets the best parameters for xgboost
    '''
    #set up parameter range to search
    parameters = {
        'n_estimators': [100,250,500],
        'learning_rate': [0.01,0.05,0.1, 0.5, 0.9],
        'max_depth': [10,15,20],
        'random_state': [42]
    }
    # set up eval set
    eval_set=[(X_train, y_train), (X_test, y_test)],
    #search the grid
    model = xgb.XGBRegressor(eval_set=eval_set, objective='reg:squarederror', verbose=False)
    clf = GridSearchCV(model, parameters)
    clf.fit(X_train, y_train,verbose=False)
    if verbose:
        # establish the best parameters as the precursors for the model
        print(f'Best params: {clf.best_params_}')
        print(f'Best validation score = {clf.best_score_}')
    #output the model
    model = xgb.XGBRegressor(**clf.best_params_, objective='reg:squarederror')

    return model

def save_model_to_pickle(model,filename='../models/lin_reg.bin'):
    ''' 
    this aux function saves the fitted model to a bin pickle file
    '''
    # create any directories if does not exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    #save file
    with open(filename, 'wb') as f_out:
        pickle.dump(model, f_out)

    return None
