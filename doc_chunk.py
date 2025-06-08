import requests
response = requests.post(
    "http://127.0.0.1:11434/api/embed",
    json={"model": "saish_15/tethysai_research", "prompt": "What can you tell me about LLms"}
)
print(response.json())
