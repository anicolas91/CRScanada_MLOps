'''
These functions basically register the best model and moves it to production in MLflow
Note: these functions need to be called after setting mlflow tracking uri
'''

import mlflow
from mlflow.tracking import MlflowClient


def register_best_model(experiment_id="1",model_name = "CRS_Model"):
    ''' 
    this function gets the model with the lowest RMSE and registers it into Models
    '''
    # start up mlflow client and connect to db
    client = MlflowClient()

    #obtain run with lowest RMSE
    best_performace_run = client.search_runs(
        experiment_ids=experiment_id,
        filter_string="metrics.rmse < 15",
        order_by=["metrics.rmse ASC"],
    )[0]

    # get run id and set Model name
    run_id = best_performace_run.info.run_id
    model_uri = f"runs:/{run_id}/artifacts/model"

    # register the model
    mlflow.register_model(model_uri=model_uri, name=model_name)

def stage_model_production(model_name = "CRS_Model"):
    '''
    this function moves the model into production
    '''
    client = MlflowClient()
    model_stage = "Production"
    all_transitions = client.get_latest_versions(name=model_name)
    model_version = len(all_transitions)
    
    client.transition_model_version_stage(
        name=model_name,
        version=model_version,
        stage=model_stage,
        archive_existing_versions=True
)