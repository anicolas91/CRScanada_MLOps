import requests

query = {
    "query_date": "31-Jul-2024"
    }


# send post requests
url = 'http://localhost:9696/predict'
response = requests.post(url,json=query)
print(response.json())