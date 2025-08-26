"""
Simple test script for the CLAIRE API
"""

import requests
import json
from datetime import datetime
import time

API_BASE = "http://localhost:8000"

def test_health():
    """Test the health endpoint"""
    print("\n" + "="*50)
    print("Testing Health Endpoint")
    print("="*50)
    
    try:
        response = requests.get(f"{API_BASE}/api/v1/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Status: {data['status']}")
            print(f"Models Loaded:")
            for model, loaded in data['models_loaded'].items():
                status = "✅" if loaded else "❌"
                print(f"  {status} {model}: {loaded}")
            print(f"Vector DB Ready: {'✅' if data['vector_db_ready'] else '❌'}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure the backend is running!")
        return False
    return True

def test_chat(question):
    """Test the chat endpoint"""
    print("\n" + "="*50)
    print(f"Question: {question}")
    print("="*50)
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{API_BASE}/api/v1/chat/chat",
            json={
                "question": question,
                "session_id": "test-session-123"
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"\n📝 Answer:")
            print(f"   {data['answer']}")
            
            print(f"\n🌐 Language: {data['language']['language']} ({data['language']['confidence']:.2%})")
            print(f"😊 Emotion: {data['emotion']['emotion']} ({data['emotion']['confidence']:.2%})")
            print(f"⏱️  Processing Time: {data['processing_time']:.2f}s")
            
            if data.get('contexts'):
                print(f"\n📚 Retrieved Contexts ({len(data['contexts'])} documents):")
                for i, ctx in enumerate(data['contexts'], 1):
                    print(f"   {i}. {ctx['title']} (Score: {ctx['score']:.2%})")
                    
        else:
            print(f"❌ Chat request failed: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure the backend is running!")

def main():
    print("\n" + "🤖 CLAIRE API Test Script 🤖".center(50))
    
    # Test health first
    if not test_health():
        print("\n⚠️  Backend is not healthy. Please check the server.")
        return
    
    # Test questions
    test_questions = [
        "May online banking ba kayo? Paano mag-register?"
    ]
    
    print("\n" + "Starting Chat Tests".center(50, "="))
    
    for question in test_questions:
        test_chat(question)
        time.sleep(1)  # Small delay between requests
    
    print("\n" + "="*50)
    print("✅ All tests completed!")
    print("="*50)

if __name__ == "__main__":
    main()