"""
Python file just used for quick app testing.
"""
import requests

query = {
    "query_date": "31-Jul-2024"
    }


# send post requests
url = 'http://localhost:9696/predict'
response = requests.post(url,json=query,timeout=10)
print(response.json())
