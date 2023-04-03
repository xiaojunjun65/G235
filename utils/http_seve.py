import requests
import json
def requests_load(idx):
    headers = {"Content-Type": "application/json;charset=utf8"}
    url = "http://127.0.0.1:8388/miniapi/event/getcamera"
    _data = {

        
        "cameraid": str(idx)
    

    }

    res = requests.post(url=url, headers=headers, json=_data).text
    return res

