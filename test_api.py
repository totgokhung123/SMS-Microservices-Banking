"""
Test script để test HDBank API endpoints
"""

import asyncio
import httpx
import json
from datetime import datetime

async def test_api_endpoints():
    """Test các API endpoints"""
    base_url = "http://127.0.0.1:8000"
    
    async with httpx.AsyncClient() as client:
        
        print("🧪 Testing HDBank Chatbot API Endpoints")
        print("=" * 50)
        
        # Test 1: Health Check
        try:
            print("\n1. Testing Health Check...")
            response = await client.get(f"{base_url}/health")
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                print(f"   Response: {json.dumps(response.json(), indent=2)}")
            else:
                print(f"   Error: {response.text}")
        except Exception as e:
            print(f"   Error: {e}")
        
        # Test 2: Detailed Health Check  
        try:
            print("\n2. Testing Detailed Health Check...")
            response = await client.get(f"{base_url}/health/detailed")
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"   Status: {data.get('status')}")
                print(f"   System CPU: {data.get('system', {}).get('cpu_usage', 'N/A')}%")
                print(f"   System Memory: {data.get('system', {}).get('memory_usage', 'N/A')}%")
                print(f"   Chatbot Service: {data.get('chatbot_service', {}).get('initialized', 'N/A')}")
            else:
                print(f"   Error: {response.text}")
        except Exception as e:
            print(f"   Error: {e}")
        
        # Test 3: Chat Endpoint
        try:
            print("\n3. Testing Chat Endpoint...")
            chat_data = {
                "message": "Tôi muốn biết về thẻ tín dụng HDBank",
                "user_id": "test_user_123",
                "conversation_id": "test_conv_123"
            }
            response = await client.post(f"{base_url}/api/v1/chat", json=chat_data)
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"   Answer: {data.get('answer', 'N/A')}")
                print(f"   Confidence: {data.get('confidence', 'N/A')}")
                print(f"   Response Time: {data.get('response_time', 'N/A')}s")
            else:
                print(f"   Error: {response.text}")
        except Exception as e:
            print(f"   Error: {e}")
        
        # Test 4: Search Endpoint
        try:
            print("\n4. Testing Search Endpoint...")
            search_data = {
                "query": "vay vốn HDBank",
                "top_k": 5
            }
            response = await client.post(f"{base_url}/api/v1/search", json=search_data)
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"   Total Results: {data.get('total_results', 'N/A')}")
                print(f"   Results Count: {len(data.get('results', []))}")
            else:
                print(f"   Error: {response.text}")
        except Exception as e:
            print(f"   Error: {e}")
        
        # Test 5: OpenAPI Schema
        try:
            print("\n5. Testing OpenAPI Schema...")
            response = await client.get(f"{base_url}/openapi.json")
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"   API Title: {data.get('info', {}).get('title', 'N/A')}")
                print(f"   API Version: {data.get('info', {}).get('version', 'N/A')}")
                print(f"   Available Paths: {len(data.get('paths', {}))}")
            else:
                print(f"   Error: {response.text}")
        except Exception as e:
            print(f"   Error: {e}")
        
        print("\n" + "=" * 50)
        print("✅ API Testing Complete!")

if __name__ == "__main__":
    asyncio.run(test_api_endpoints())