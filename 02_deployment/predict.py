## Predicting CRS scores
# This is a pipeline that given a specific date, will forecast an estimated CRS cutoff value.
import os
import sys
import mlflow

here = os.path.dirname(__file__)
sys.path.append(os.path.join(here, '..'))
from utils import registering as regs
from utils import preprocessing as prep
from flask import Flask, request, jsonify

# setup MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000/")

# set up app
app = Flask('CRS-prediction')

@app.route('/predict', methods=['POST'])
def main_run():
    ''' 
    this is the main function to run during a prediction
    '''
    # establish the query date (to be replaced by a json input later)
    #query_date = '31-Jul-2024'
    json_input = request.get_json()
    query_date = json_input['query_date']
    # calculate all the usual x_labels
    df_query = prep.preprocess_query_date(query_date)
    # get model and the labels used to calculate it
    model, x_labels, _, _ = regs.get_prod_info_from_registry(reg_model_name= "CRS_Model")
    # get only x labels at the queried time
    X_vals = df_query[x_labels]
    #predict the CRS cutoff
    CRS_pred = model.predict(X_vals)[0][0].round(0)
    #print response
    # print(f'for date {query_date} the predicted crs score is {CRS_pred:.0f}')
    #save prediction as dict
    result = {
        'query_date' : query_date,
        'CRS_pred': CRS_pred
    }
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host = '0.0.0.0', port=9696)

''' 
to run this app simply do:

gunicorn --bind=0.0.0.0:9696 predict:app

and then you can query via:

python test.py

'''