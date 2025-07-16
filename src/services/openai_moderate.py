import json
import requests
from tqdm import tqdm
from src.config import settings
import time

api_key = settings.openai_api_key


def assess_moderation_openai_api(text):
    # Define the URL and headers
    url = "https://api.openai.com/v1/moderations"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    # Define the payload
    payload = {"model": "omni-moderation-latest", "input": text}

    # Convert the payload to JSON
    payload_json = json.dumps(payload)

    # Send the POST request
    response = requests.post(url, headers=headers, data=payload_json)

    # Check if the request was successful
    if response.status_code == 200:
        response_json = response.json()["results"][0]["category_scores"]
        outputs = {}
        for k, v in response_json.items():
            outputs[k.replace("-", " ").replace("/", " ")] = {"result": v}
        return outputs
    else:
        raise Exception(f"Error: {response.status_code} - {response.text}")