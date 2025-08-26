"""Test OCR functionality"""

import requests
import time
from pathlib import Path

API_BASE = "http://localhost:8000"

def test_ocr_upload(file_path):
    """Test OCR file upload"""
    print(f"\nTesting OCR with: {file_path}")
    print("="*50)
    
    with open(file_path, 'rb') as f:
        files = {'file': (Path(file_path).name, f)}
        
        start_time = time.time()
        response = requests.post(
            f"{API_BASE}/api/v1/upload/extract-text",
            files=files
        )
        
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Success!")
        print(f"📄 Filename: {data['filename']}")
        print(f"📝 Extracted {data['char_count']} characters")
        print(f"⏱️  Processing time: {data['processing_time']:.2f}s")
        print(f"\n📖 Preview of extracted text:")
        print("-"*40)
        print(data['extracted_text'][:500])
        print("-"*40)
        return data['extracted_text']
    else:
        print(f"❌ Failed: {response.status_code}")
        print(f"Error: {response.text}")
        return None

def test_chat_with_attachment(question, extracted_text):
    """Test chat with extracted text"""
    print(f"\n📨 Sending question with attachment")
    print("="*50)
    
    response = requests.post(
        f"{API_BASE}/api/v1/chat/chat",
        json={
            "question": question,
            "extracted_text": extracted_text,
            "session_id": "test-ocr"
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Answer: {data['answer']}")
        print(f"📎 Has attachment: {data['has_attachment']}")
    else:
        print(f"❌ Failed: {response.status_code}")

# Test with sample files
if __name__ == "__main__":
    # Create a test text file
    test_file = "test_document.txt"
    with open(test_file, 'w') as f:
        f.write("""
        BPI Credit Card Statement
        
        Account Number: ****-****-****-1234
        Statement Date: January 15, 2024
        
        Current Balance: PHP 25,000.00
        Minimum Payment Due: PHP 1,250.00
        Payment Due Date: February 5, 2024
        
        Please pay on time to avoid late charges.
        """)
    
    # Test OCR
    extracted = test_ocr_upload(test_file)
    
    if extracted:
        # Test chat with extracted text
        test_chat_with_attachment(
            "What is my current balance and when is it due?",
            extracted
        )
    
    # Clean up
    Path(test_file).unlink()