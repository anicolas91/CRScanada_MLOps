'''
These functions basically register the best model and moves it to production in MLflow
Note: these functions need to be called after setting mlflow tracking uri
'''

import ast
import pickle
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
        filter_string="metrics.rmse < 20",
        order_by=["metrics.rmse ASC"],
    )[0]

    # get run id and set Model name
    run_id = best_performace_run.info.run_id
    #model_uri = f"runs:/{run_id}/artifacts/model"
    model_uri = f"runs:/{run_id}/models"

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

def load_model_from_mlflow(run_id,model_type):
    ''' 
    given a run id and the type of model, download and read the model
    '''
    client = MlflowClient()

    # download the model to local folder
    model_filename = f'./models/{model_type}_model.bin'
    client.download_artifacts(run_id, model_filename,'./')

    # Load the pickle model
    with open(model_filename, "rb") as f:
        model = pickle.load(f)

    return model


def get_prod_info_from_registry(reg_model_name= "CRS_Model"):
    ''' 
    This function returns the model and other vars that was registered into production
    '''

    # get registered model
    client = MlflowClient()
    mymodel = client.get_registered_model(name= reg_model_name)

    # get run Id of the production model in the register
    run_id = None
    for lv in mymodel.latest_versions:
        stage = lv.current_stage
        if stage == 'Production':
            run_id = lv.run_id

    # Get the details of the run
    run_info = client.get_run(run_id)

    # Retrieve the x_label from parameters
    params = run_info.data.params
    x_labels = ast.literal_eval(params['x_labels'])

    # retrieve the label from the tags
    tags = run_info.data.tags
    model_type = tags['model_type']

    model = load_model_from_mlflow(run_id=run_id,model_type=model_type)

    return model, x_labels, model_type, run_id
