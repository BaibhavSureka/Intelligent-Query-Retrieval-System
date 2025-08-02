#!/usr/bin/env python3
"""
Test script for the LLM-Powered Query Retrieval System
"""

import requests
import json
import time
import sys

# Configuration
BASE_URL = "http://localhost:8000"
BEARER_TOKEN = "88def649851a3e0861a60905001a92f0a9cdd621ade7686aa6be07cc91f1ed9b"

# Test document URL (from HackRx specification)
TEST_DOCUMENT_URL = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"

# Test questions
TEST_QUESTIONS = [
    "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
    "What is the waiting period for pre-existing diseases (PED) to be covered?",
    "Does this policy cover maternity expenses, and what are the conditions?",
    "What is the waiting period for cataract surgery?",
    "Are the medical expenses for an organ donor covered under this policy?"
]

def make_request(method, endpoint, data=None):
    """Make an HTTP request to the API"""
    url = f"{BASE_URL}{endpoint}"
    headers = {
        "Authorization": f"Bearer {BEARER_TOKEN}",
        "Content-Type": "application/json"
    }
    
    try:
        if method.upper() == "GET":
            response = requests.get(url, headers=headers)
        elif method.upper() == "POST":
            response = requests.post(url, headers=headers, json=data)
        elif method.upper() == "DELETE":
            response = requests.delete(url, headers=headers)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        return response
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None

def test_health_check():
    """Test the health check endpoint"""
    print("🔍 Testing health check...")
    response = make_request("GET", "/health")
    
    if response and response.status_code == 200:
        print("✅ Health check passed")
        return True
    else:
        print(f"❌ Health check failed: {response.status_code if response else 'No response'}")
        return False

def test_hackrx_endpoint():
    """Test the main HackRx endpoint"""
    print("\n🔍 Testing HackRx endpoint...")
    
    request_data = {
        "documents": TEST_DOCUMENT_URL,
        "questions": TEST_QUESTIONS[:3]  # Test with first 3 questions
    }
    
    print(f"📄 Document: {TEST_DOCUMENT_URL[:50]}...")
    print(f"❓ Questions: {len(request_data['questions'])} questions")
    
    start_time = time.time()
    response = make_request("POST", "/hackrx/run", request_data)
    end_time = time.time()
    
    if response and response.status_code == 200:
        response_data = response.json()
        print(f"✅ HackRx endpoint test passed")
        print(f"⏱️  Response time: {end_time - start_time:.2f} seconds")
        print(f"📝 Answers received: {len(response_data.get('answers', []))}")
        
        # Print first answer as sample
        if response_data.get('answers'):
            print(f"📋 Sample answer: {response_data['answers'][0][:100]}...")
        
        return True
    else:
        print(f"❌ HackRx endpoint test failed")
        if response:
            print(f"   Status code: {response.status_code}")
            print(f"   Response: {response.text}")
        return False

def test_single_query():
    """Test the single query endpoint"""
    print("\n🔍 Testing single query endpoint...")
    
    query_data = {
        "document_url": TEST_DOCUMENT_URL,
        "question": "What is the grace period for premium payment?"
    }
    
    response = make_request("POST", "/api/v1/query", query_data)
    
    if response and response.status_code == 200:
        response_data = response.json()
        print("✅ Single query test passed")
        print(f"📋 Answer: {response_data.get('answer', 'No answer')[:100]}...")
        return True
    else:
        print(f"❌ Single query test failed")
        if response:
            print(f"   Status code: {response.status_code}")
            print(f"   Response: {response.text}")
        return False

def test_config_endpoint():
    """Test the configuration endpoint"""
    print("\n🔍 Testing config endpoint...")
    
    response = make_request("GET", "/api/v1/config")
    
    if response and response.status_code == 200:
        config = response.json()
        print("✅ Config endpoint test passed")
        print(f"📊 Embedding model: {config.get('embedding_model', 'Unknown')}")
        print(f"🔧 Chunk size: {config.get('chunk_size', 'Unknown')}")
        return True
    else:
        print(f"❌ Config endpoint test failed")
        if response:
            print(f"   Status code: {response.status_code}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting API tests...")
    print(f"🌐 Base URL: {BASE_URL}")
    
    tests = [
        ("Health Check", test_health_check),
        ("HackRx Endpoint", test_hackrx_endpoint),
        ("Single Query", test_single_query),
        ("Config Endpoint", test_config_endpoint),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        sys.exit(0)
    else:
        print("⚠️  Some tests failed. Check the logs above.")
        sys.exit(1)

if __name__ == "__main__":
    main()