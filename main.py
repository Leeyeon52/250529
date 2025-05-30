import requests

test_data = ['i am happy', 'i want to go', 'i wake too early so i feel grumpy', 'i feel alarmed']
payload = {"infer_texts": test_data}  # 여기서 payload 정의

response = requests.post("http://127.0.0.1:8080/predict", json=payload)
print(response.json())
