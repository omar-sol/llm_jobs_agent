import requests
import json

FASTAPI_SERVER_URL = "http://localhost:8000"  # Change this to your FastAPI server URL


# def get_answer_non_streaming(user_input):
#     """
#     Function to get a non-streaming response from the FastAPI server.
#     """
#     url = f"{FASTAPI_SERVER_URL}/get_answer/"
#     data = {"user_input": user_input, "stream": False, "return_json": True}
#     response = requests.post(url, json=data)
#     return response.json()


# # Example usage
# user_question = "What are the necessary skills for a junior data scientist?"
# non_streaming_response = get_answer_non_streaming(user_question)
# print("Non-streaming response:", non_streaming_response)


import httpx
import json


def get_answer_streaming(user_input, timeout=60):
    """
    Function to get a streaming response from the FastAPI server.
    Correctly handles Server-Sent Events format.
    """
    url = f"{FASTAPI_SERVER_URL}/get_answer/"
    data = {"user_input": user_input, "stream": True, "return_json": True}
    with httpx.stream("POST", url, json=data, timeout=timeout) as response:
        for line in response.iter_lines():
            if line.startswith("data:"):
                try:
                    # Extract JSON data from SSE format
                    json_data = line[len("data: ") :]  # Remove the 'data: ' prefix
                    parsed_line = json.loads(json_data)
                    print(parsed_line)
                except json.JSONDecodeError as e:
                    # Handle JSON decode error
                    print("JSONDecodeError:", e)
                    print("Received data:", line)


# Example usage
user_question = "What are the necessary skills for a junior data scientist?"
print("\nStreaming response:")
get_answer_streaming(user_question, timeout=60)
