
# before running make sure you start up ...
# the CRS conda environment via:
# ```bash
# conda activate CRSenv
# ```
# mlflow via:
# ```bash
# mlflow server --backend-store-uri sqlite:///mlflow.db
# ```
# prefect via:
# ```bash
# prefect server start
# ```
# 
# and docker via:
# ```bash
# docker-compose up db adminer grafana --build
# ```
#
# run the python script as:
# ```bash
# python monitoring.py
# ```
#
# Go log into PostgreSQL and grafana, you should see things getting created in a cool existing dashboard 
#
#
# For **PostgreSQL** go to http://localhost:8080/ and on the login fill the following:
# - system: postgresQL
# - server: db
# - username: postgres
# - password: example
# - database: test
# 
# For **graphana** go to http://localhost:3000 and on the login fill the following:
# - user: admin
# - password: admin
# 
# Graphana will ask you to change the password. Do that, or just ignore it, whichever you'd like. In our case we used `oneprettybird` with the fancy formatting we usually use.
# 
#
# For **mlflow**, go to http://127.0.0.1:5000/
#
# For **prefect**, go to http://127.0.0.1:4200/
#
# neither of them need any credentials
#
#

import os
import sys
import time
import mlflow
import logging
import psycopg # to access the database
import warnings
import datetime
import pandas as pd

from prefect import task, flow

from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import ColumnDriftMetric, RegressionQualityMetric

# for a python script this is : 
here = os.path.dirname(__file__)
sys.path.append(os.path.join(here, '..'))

from utils import registering as regs
from utils import preprocessing as prep

# specify which kind of warning/info outputs are ok
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
warnings.simplefilter(action='ignore', category=FutureWarning)

# set up mlflow
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000/")
mlflow.set_tracking_uri(tracking_uri)

## INPUTS ##########################################################################
url = 'https://www.canada.ca/content/dam/ircc/documents/json/ee_rounds_123_en.json'

# global variable
SEND_TIMEOUT = 1

# we set up the beginning time, in this case is jan 1st 2024
begin = datetime.datetime(2024, 1, 1, 0, 0)

# Extract the number of days from the beginning of jan to today
ndays = (datetime.datetime.today() - begin).days+1

# load the model
model, x_labels, _, _ = regs.get_prod_info_from_registry(reg_model_name= "CRS_Model")

#just general column mapping so that evidently knows what is what
column_mapping = ColumnMapping(
    target='CRS cutoff',
    prediction='CRS prediction', # name on valiadation data
    numerical_features=x_labels,
    categorical_features=None
)

# sets up the report itself
report = Report(
    metrics = [
        ColumnDriftMetric(column_name='CRS cutoff',stattest='ed'), # drift in the actual crs
        RegressionQualityMetric()
    ]
)

# sets up the sql table
create_table_statement = """
drop table if exists crs_metrics;
create table crs_metrics(
	timestamp timestamp,
    predicted_crs float,
    actual_crs float,
    drift_actual_crs float,
    rmse_crs_current float,
    error_std_current float
)
"""

## functions ######################################################################
@task
def prep_db():
    ''' 
    This function connects to postgresql and either creates the table or adds to it
    '''
    with psycopg.connect("host=localhost port=5432 user=postgres password=example", autocommit=True) as conn: # we connect to host
        res = conn.execute("SELECT 1 FROM pg_database WHERE datname='test'") # this is an sql query
        if len(res.fetchall()) == 0: #if there is no database...
            conn.execute("create database test;") # we create the test database
        with psycopg.connect("host=localhost port=5432 dbname=test user=postgres password=example") as conn: # we connect to the db 
            conn.execute(create_table_statement) # we create the table

@task
def ref_curr_data_from_web(url,date_cutoff_start='01-Jan-2015',date_cutoff_ref = '01-Jan-2023'):
    ''' 
    This function calculates CRS predictions for all existing web data, and separantes into reference and current
    '''
    df = prep.create_df_from_website(url)
    df = prep.cleanup_df_general_rounds(df)
    df = prep.calculate_independent_vars(df)

    # calculate predictions for the entire data yolo
    df['CRS prediction'] = model.predict(df[x_labels]).round()

    # set up the CRS cutoff as float
    df['CRS cutoff'] =  df['CRS cutoff'].astype(float)

    # separate into reference and current data
    reference_data   = df[(df.index >= date_cutoff_start) & (df.index < date_cutoff_ref)]
    current_data_all = df[df.index >= date_cutoff_ref]

    return reference_data, current_data_all

@task
def calculate_metrics_atdate(query_date):
    ''' 
    This function calculates CRS predictions and the goodness metrics of the current model in production.
    NOTE: make sure teh query_date is after jan 2023
    '''
    # get all data up to the query_date
    df_todate = current_data_all.loc[current_data_all.index <= query_date].copy()
    # predict CRS only if date not on website, otherwise actually calculate all metrics
    if query_date not in current_data_all.index:
        # print('no CRS cutoff value available on CRS website. Adding a blank row.')
        # create a new row just for prediction
        df_query = df_todate[:0].copy()
        df_query.loc[query_date,'round type'] = 'General'
        df_query.loc[query_date,'invitations issued'] = 0 #dummy data
        df_query.loc[query_date,'CRS cutoff'] = 0 #dummy data 
        # concatenate to older data
        df_todate = pd.concat([df_query, df_todate])
        # recalculate x_labels for queried date not on website
        df_query_new = prep.calculate_independent_vars(df_todate[['round type','invitations issued','CRS cutoff']])[:1]
        #predict the crs value only
        y_pred = model.predict(df_query_new[x_labels])[0][0].round(0)
        # set up metrics
        predicted_crs = y_pred
        actual_crs = float('nan')
        drift_actual_crs = float('nan')
        rmse_crs_current = float('nan')
        error_std_current = float('nan')

    else:
        # get current data of the queried date and anything 3 months prior
        three_months_prior = pd.to_datetime(query_date) - datetime.timedelta(days=90)
        current_data = df_todate[(df_todate.index >= three_months_prior)]
        # run the report and convert to dictionary
        report.run(reference_data=reference_data,current_data=current_data,column_mapping=column_mapping)
        # convert to dictionary
        result = report.as_dict()
        #extract the metrics as needed
        predicted_crs = current_data['CRS prediction'][query_date]
        actual_crs = current_data['CRS cutoff'][query_date]
        drift_actual_crs = result['metrics'][0]['result']['drift_score']
        rmse_crs_current = result['metrics'][1]['result']['current']['rmse']
        error_std_current = result['metrics'][1]['result']['current']['error_std']

    return predicted_crs,actual_crs,drift_actual_crs,rmse_crs_current,error_std_current

@task
def insert_metrics_postgresql(curr,begin_date,days_from_begin):
    ''' 
    This function walks through days and inserts the metrics into the sql table
    '''
    # get the query date
    query_date = pd.to_datetime(begin_date) + datetime.timedelta(days=days_from_begin)
    # calculate the metrics
    pred_crs,act_crs,drift,rmse,errstd = calculate_metrics_atdate(query_date)
    # insert data into table day by day
    curr.execute(
        "insert into crs_metrics(timestamp, predicted_crs, actual_crs, drift_actual_crs,rmse_crs_current,error_std_current) values (%s, %s, %s, %s, %s, %s)",
        (query_date, pred_crs, act_crs, drift, rmse, errstd)
    )

@flow
def monitoring_backfill():
    '''
    This function calculates and exports metrics to the sql db every second to pretend it runs daily.
    '''
    prep_db() # just preps db and table
    last_send = datetime.datetime.now() - datetime.timedelta(seconds=1) # time of last sent
    with psycopg.connect("host=localhost port=5432 dbname=test user=postgres password=example", autocommit=True) as conn: # connected to db w credentials
        for i in range(0, ndays): #iterated n days  
            with conn.cursor() as curr: # inserted on cursor ACTUAL metrics and timestamp
                insert_metrics_postgresql(curr,begin,i)

            new_send = datetime.datetime.now()
            seconds_elapsed = (new_send - last_send).total_seconds() # for visuals we have a time delay calc
            if seconds_elapsed < SEND_TIMEOUT: # if its less than the 10 secs we just wait
                time.sleep(SEND_TIMEOUT - seconds_elapsed)
            while last_send < new_send: # if more than 10 secs passed then sure send the data
                last_send = last_send + datetime.timedelta(seconds=1)
            logging.info("data sent")


## main run #########################################################
if __name__ == '__main__':
    # get initial reference and current data
    reference_data, current_data_all = ref_curr_data_from_web(url,date_cutoff_ref='01-Jan-2023')

    # run the batch filling
    monitoring_backfill()




