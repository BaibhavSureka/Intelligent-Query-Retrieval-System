#!/usr/bin/env python3
"""
Test script to validate RAG system improvements
"""

import requests
import json
import time
from typing import List, Dict

# Test configuration
API_URL = "http://localhost:8000/api/v1/hackrx/run"
AUTH_TOKEN = "88def649851a3e0861a60905001a92f0a9cdd621ade7686aa6be07cc91f1ed9b"

# Test document
TEST_DOCUMENT = "https://hackrx.blob.core.windows.net/assets/hackrx_6/policies/HDFHLIP23024V072223.pdf?sv=2023-01-03&st=2025-07-30T06%3A46%3A49Z&se=2025-09-01T06%3A46%3A00Z&sr=c&sp=rl&sig=9szykRKdGYj0BVm1skP%2BX8N9%2FRENEn2k7MQPUp33jyQ%3D"

# Test questions with expected answers
TEST_QUESTIONS = [
    {
        "question": "What is the grace period for premium payment under this policy?",
        "expected_keywords": ["30 days", "15 days", "yearly", "monthly"],
        "expected_accuracy": "high"
    },
    {
        "question": "What is the waiting period for pre-existing diseases to be covered?",
        "expected_keywords": ["36 months", "36 month", "pre-existing"],
        "expected_accuracy": "high"
    },
    {
        "question": "Are maternity expenses covered, and what are the eligibility conditions?",
        "expected_keywords": ["maternity", "2 deliveries", "covered"],
        "expected_accuracy": "high"
    },
    {
        "question": "Does the policy cover medical expenses for an organ donor?",
        "expected_keywords": ["organ donor", "not covered", "pre and post"],
        "expected_accuracy": "critical"  # This was the problematic one
    },
    {
        "question": "What benefits are provided under the preventive health check-up clause?",
        "expected_keywords": ["1%", "5000", "claim-free"],
        "expected_accuracy": "high"
    }
]

def test_rag_system():
    """Test the RAG system with the provided questions"""
    
    print("🧪 Testing RAG System Improvements")
    print("=" * 50)
    
    # Prepare request
    request_data = {
        "documents": TEST_DOCUMENT,
        "questions": [q["question"] for q in TEST_QUESTIONS]
    }
    
    headers = {
        "Authorization": f"Bearer {AUTH_TOKEN}",
        "Content-Type": "application/json"
    }
    
    try:
        # Measure response time
        start_time = time.time()
        
        response = requests.post(API_URL, json=request_data, headers=headers)
        
        end_time = time.time()
        response_time = end_time - start_time
        
        if response.status_code == 200:
            result = response.json()
            answers = result.get("answers", [])
            
            print(f"✅ Request successful (Response time: {response_time:.2f}s)")
            print(f"📄 Document processed: {TEST_DOCUMENT}")
            print(f"❓ Questions processed: {len(TEST_QUESTIONS)}")
            print(f"💬 Answers generated: {len(answers)}")
            print()
            
            # Analyze each answer
            for i, (question_data, answer) in enumerate(zip(TEST_QUESTIONS, answers)):
                print(f"Question {i+1}: {question_data['question']}")
                print(f"Answer: {answer}")
                
                # Check for expected keywords
                missing_keywords = []
                for keyword in question_data["expected_keywords"]:
                    if keyword.lower() not in answer.lower():
                        missing_keywords.append(keyword)
                
                if missing_keywords:
                    print(f"⚠️  Missing expected keywords: {missing_keywords}")
                else:
                    print("✅ All expected keywords found")
                
                # Special check for the organ donor question
                if "organ donor" in question_data["question"].lower():
                    if "not covered" in answer.lower() and "pre" in answer.lower() and "post" in answer.lower():
                        print("✅ Correctly identified organ donor pre/post expenses as NOT covered")
                    else:
                        print("❌ Incorrectly stated organ donor pre/post expenses are covered")
                
                print("-" * 50)
                
            # Performance metrics
            print(f"📊 Performance Metrics:")
            print(f"   - Total response time: {response_time:.2f}s")
            print(f"   - Average time per question: {response_time/len(TEST_QUESTIONS):.2f}s")
            print(f"   - Questions processed: {len(TEST_QUESTIONS)}")
            
        else:
            print(f"❌ Request failed with status code: {response.status_code}")
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")

def test_caching():
    """Test the caching mechanism"""
    
    print("\n🔄 Testing Caching Mechanism")
    print("=" * 30)
    
    request_data = {
        "documents": TEST_DOCUMENT,
        "questions": [TEST_QUESTIONS[0]["question"]]  # Just one question for caching test
    }
    
    headers = {
        "Authorization": f"Bearer {AUTH_TOKEN}",
        "Content-Type": "application/json"
    }
    
    try:
        # First request (should be slower)
        start_time = time.time()
        response1 = requests.post(API_URL, json=request_data, headers=headers)
        first_time = time.time() - start_time
        
        # Second request (should be faster due to caching)
        start_time = time.time()
        response2 = requests.post(API_URL, json=request_data, headers=headers)
        second_time = time.time() - start_time
        
        if response1.status_code == 200 and response2.status_code == 200:
            print(f"✅ First request time: {first_time:.2f}s")
            print(f"✅ Second request time: {second_time:.2f}s")
            
            if second_time < first_time * 0.8:  # 20% improvement expected
                print("✅ Caching is working (second request was faster)")
            else:
                print("⚠️  Caching may not be working as expected")
        else:
            print("❌ Request failed during caching test")
            
    except Exception as e:
        print(f"❌ Error during caching test: {str(e)}")

if __name__ == "__main__":
    test_rag_system()
    test_caching() 