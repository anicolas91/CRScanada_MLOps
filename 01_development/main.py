"""
CRS Canada immigration metrics analysis.

We investigate the Comprehensive Ranking System (CRS)
metrics that Canada uses to grant permanent residence invitations.

Goal: To project predicted cutoff values for the incoming cohort.

"""

# general libraries
import os
import sys

import mlflow
import matplotlib
from prefect import flow, task

# ML models
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

# import utils
here = os.path.dirname(__file__)
sys.path.append(os.path.join(here, '..'))
from utils import modeling as modl
from utils import splitting as splt
from utils import registering as regs
from utils import preprocessing as prep

# MLflow
# 0. Activate the `CRSenv` conda environment & kill any hanging mlflow with `pkill -f gunicorn`
# 1. Need to run on terminal: `mlflow ui --backend-store-uri sqlite:///mlflow.db`
#  or `mlflow server --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000'
# 2. MLflow can be found on [http://127.0.0.1:5000/](http://127.0.0.1:5000/)
# 3. We set the tracking uri on the python script
# 4. We set the experiment name to save runs. It the exp doesn't exist mlflow will create one.

# Prefect
# 0. Activate the `CRSenv` conda environment
# 1. Need to run on terminal: `prefect server start`
# 2. Prefect can be found on [http://127.0.0.1:4200](http://127.0.0.1:4200)
# 3. run the python script `python main.py`
# 4. start a run from the CLI with: `prefect deployment run main/CRS-canada-score-train-deploy`

# or...

# 3. Create a work pool with: `prefect work-pool create zoompool -t process`
#    (may error if pool already exists)
# 4. run the python script `python main.py`
# 5. Start a worker that polls your work pool with: `prefect worker start --pool 'zoompool'`
# 6. start a run from the CLI with: `prefect deployment run main/CRS-canada-score-train-deploy`

# General Setup
xgb.set_config(verbosity=0) # set xgb verbosity to none
#mlflow setup
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000/")
#mlflow.set_tracking_uri("sqlite:///mlflow.db") # this overrides the cli artifacts
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("CRS-score-canada")

# Cleaned up functions
@task(retries=3, retry_delay_seconds=2)
def preprocess_from_url(url):
    ''' 
    This function scrubs the url and returns a cleaned up dataframe with all independent vars
    '''
    df = prep.create_df_from_website(url)
    df = prep.cleanup_df_general_rounds(df)
    df = prep.calculate_independent_vars(df)

    return df

@task()
def create_split_features(df=None,split_date='01-Jan-2023',x_labels=None,y_labels=None):
    ''' 
    This function performs the splitting into test,train, and independent, dependent vars
    '''
    # set up fresh lists
    if x_labels is None:
        x_labels = ['dt-1','CRS-1']
    if y_labels is None:
        y_labels = ['CRS cutoff']
    # split data according to date
    dtrain, dtest = splt.split_test_train(df,split_date)
    # extract the labels
    X_train, y_train = splt.create_features(dtrain,x_labels,y_labels)
    X_test, y_test = splt.create_features(dtest,x_labels,y_labels)

    return X_train, y_train, X_test, y_test

@task()
def fit_and_evaluate_model(X_train,y_train,X_test,y_test,model_type='linear'):
    ''' 
    this function fits either an XGboost or a linear model and returns model/rmse
    '''
    # select model type
    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'xgboost':
        model = modl.get_best_xgboost_model(X_train, y_train,verbose=False)
    #fit the model to the training data
    model.fit(X_train, y_train) # pylint: disable=possibly-used-before-assignment
    # calculate the RMSE
    y_pred = y_test.copy()
    y_pred['CRS cutoff'] = model.predict(X_test)
    rmse = root_mean_squared_error(y_true=y_test,y_pred=y_pred)
    # return fitted model and rmse
    return model, rmse, y_pred

@task()
def plot_and_save_results(y_train,y_test,y_pred,filename='./figures/demo-file.png'):
    ''' 
    this aux function creates a pandas df plot and saves it as a png
    '''
    # do not show gui
    matplotlib.use('agg')
    #plt.ioff()
    # calculates rmse
    rmse = root_mean_squared_error(y_true=y_test,y_pred=y_pred)
    # create any directories if does not exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    # plot the figure via pandas
    ax = y_train.rename(columns={'CRS cutoff': 'Train Data'}).plot(
        figsize=(15, 5),
        title = f'rmse: {rmse:.3f}',
        ylabel='CRS cutoff',
        style='.-')
    y_test.rename(columns={'CRS cutoff': 'Test Data'}).plot(ax=ax,style='.-')
    y_pred.rename(columns={'CRS cutoff': 'Prediction from model'}).plot(ax=ax,style='.-',grid=True)
    #save the figure
    ax.figure.savefig(filename)
    return ax

@flow(log_prints=True)
def main():
    ''' 
    Main function to run the entire model fitting routine.
    '''
    # preprocessing
    df = preprocess_from_url(
        'https://www.canada.ca/content/dam/ircc/documents/json/ee_rounds_123_en.json')

    # create features and such
    split_date = '01-Jan-2023'

    # establish the combination of independent vars we wish to study
    xlabels_combos = [
        ['roll_30D','roll_60D','roll_90D','dt-1','CRS-1'],
        ['roll_30D','roll_60D','roll_90D','roll_180D','dt-1','CRS-1'],
        ['roll_30D','roll_60D','roll_90D','dt-1','CRS-1','month','year'],
        ['roll_30D','roll_60D','roll_90D','dt-1','CRS-1','dt-2','CRS-2','dt-3','CRS-3'],
        ['roll_30D','roll_60D','roll_90D','dt-1','CRS-1','dt-2','CRS-2','dt-3','CRS-3','month','year'], # pylint: disable=C0301
        ['roll_30D','roll_60D','dt-1','CRS-1','month','year','quarter'],
        ['dt-1','CRS-1','dt-2','CRS-2','dt-3','CRS-3'],
        ['dt-1','CRS-1','dt-2','CRS-2','dt-3','CRS-3','month','year'],
        ['roll_30D','roll_60D','roll_90D','roll_180D','dt-1','CRS-1','dt-2','CRS-2','dt-3','CRS-3','month','year','quarter'], # pylint: disable=C0301
        ['month','year','quarter','dayofweek']
    ]

    #specify model types
    model_types = ['linear','xgboost']

    # Main function
    # start runs at each model type and xlabel combo
    for model_type in model_types:
        for x_labels in xlabels_combos:
            # start a print statement
            print(f'modeling {model_type} with x labels {','.join(x_labels)}...')
            # start a run
            with mlflow.start_run():
                # obtain features
                X_train, y_train, X_test, y_test = create_split_features(
                                                    df=df,
                                                    split_date=split_date,
                                                    x_labels=x_labels
                                                    )
                # model fitting
                model,rmse,y_pred = fit_and_evaluate_model(X_train,y_train,X_test,y_test,model_type)
                # save model as bin file
                model_filename = f'./models/{model_type}_model.bin'
                modl.save_model_to_pickle(model,filename=model_filename)
                # plot fig and save as png
                figure_filename = f'./figures/{model_type}_model.png'
                plot_and_save_results(y_train,y_test,y_pred,filename=figure_filename)
                #register everything to mlflow
                #NOTE: there are other ways to register, but for our needs we are keeping it simple
                mlflow.set_tag("model_type",model_type)
                mlflow.set_tag("developer","andrea")
                mlflow.log_param("x_labels",x_labels)
                mlflow.log_param("split_date",split_date)
                mlflow.log_params(model.get_params())
                mlflow.log_metric("rmse",rmse)
                mlflow.log_artifact(local_path=model_filename, artifact_path="models")
                mlflow.log_artifact(local_path=figure_filename, artifact_path="figures")
            # print end rmse
            print(f'rmse: {rmse:.3f}')
    print('finished fitting models.')
    # register the best model and move to production
    print('registering the model with lowest RMSE and moving it to production...')
    regs.register_best_model()
    regs.stage_model_production()
    print('done.')


# call main fcn if calling this script directly
if __name__ == "__main__":
    # main.from_source(
    #     source="https://github.com/anicolas91/CRScanada_MLOps.git",
    #     entrypoint="main.py:main",
    # ).deploy(

    main.serve(
        name="CRS-canada-score-train-deploy",
        #work_pool_name="zoompool",
        tags=["training", "linear","xbgoost","generalCRS","deploy"],
        description="trains a model to predict the CRS cutoff score for the general rounds",
        version=1,
        cron="0 10 1,15 * *" # run this at 10am of the 1st n 15th day of every month
    )
