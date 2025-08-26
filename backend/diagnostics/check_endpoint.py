import requests
import json

# Test the endpoint directly
url = "http://localhost:8000/api/v1/chat/chat"

# Simple test without attachment
data = {
    "question": "What is BPI?",
    "session_id": "test-simple"
}

print("Testing without attachment...")
try:
    response = requests.post(url, json=data)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print(f"Response: {response.json()}")
    else:
        print(f"Error: {response.text}")
except Exception as e:
    print(f"Request failed: {e}")

# Test with attachment
print("\nTesting with attachment...")
data_with_attachment = {
    "question": "What is my balance?",
    "extracted_text": "Current Balance: PHP 25,000.00",
    "session_id": "test-attachment"
}

try:
    response = requests.post(url, json=data_with_attachment)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print(f"Response: {response.json()}")
    else:
        print(f"Error: {response.text}")
except Exception as e:
    print(f"Request failed: {e}")