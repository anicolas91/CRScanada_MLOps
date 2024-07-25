## Predicting CRS scores
# This is a pipeline that given a specific date, will forecast an estimated CRS cutoff value.
import mlflow

from utils import preprocessing as prep
from utils import registering as regs

# setup MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")


def main_run(query_date):
    ''' 
    this is the main function to run during a prediction
    '''
    # establish the query date (to be replaced by a json input later)
    #query_date = '31-Jul-2024'
    # calculate all the usual x_labels
    df_query = prep.preprocess_query_date(query_date)
    # get model and the labels used to calculate it
    model, x_labels, _, _ = regs.get_prod_info_from_registry(reg_model_name= "CRS_Model")
    # get only x labels at the queried time
    X_vals = df_query[x_labels]
    #predict the CRS cutoff
    y_pred = model.predict(X_vals)[0][0]

    return y_pred


# run the main script see what happens
query_date = '31-Jul-2024'
y_pred =main_run(query_date)

#print value
print(f'for date {query_date} the predicted crs score is {y_pred:.0f}')