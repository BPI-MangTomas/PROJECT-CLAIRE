import requests
import json

# Test health endpoint
try:
    response = requests.get("http://localhost:8000/api/v1/health")
    print("Health Check Response:")
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"Health check failed: {e}")

# Test chat endpoint
try:
    response = requests.post(
        "http://localhost:8000/api/v1/chat/chat",
        json={"question": "Hello, how can I open an account?"}
    )
    print("\nChat Response:")
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"Chat failed: {e}")