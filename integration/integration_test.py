'''
integration test for testing end-to-end
send request to docker+ web service
get and check the response
'''

import requests

# set up query date
query = {
    "query_date": "01-Jan-2024"
    }

# get api url
url = 'http://localhost:9696/predict'

if __name__ == "__main__":
    # get predicted CRS
    response = requests.post(url,json=query,timeout=10).json()
    # compare expected vs actual
    CRS_expected  = 530.0
    CRS_actual = response['CRS_pred']
    # print values
    print(f'predicted CRS: {CRS_expected}, actual CRS: {CRS_actual}')
    
    assert abs(CRS_expected-CRS_actual) < 10
    print('all good.') #wont print if assertion error btw


