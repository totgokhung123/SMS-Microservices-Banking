"""
Simple test cho HDBank API
"""

import time
import requests
import json

def test_health_endpoint():
    """Test health endpoint"""
    try:
        print("Testing health endpoint...")
        response = requests.get("http://127.0.0.1:8000/health", timeout=5)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Health endpoint working!")
            print(f"Response: {json.dumps(data, indent=2)}")
            return True
        else:
            print(f"❌ Health endpoint failed: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed - server might not be running")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_openapi_endpoint():
    """Test OpenAPI schema endpoint"""
    try:
        print("\nTesting OpenAPI endpoint...")
        response = requests.get("http://127.0.0.1:8000/openapi.json", timeout=5)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ OpenAPI endpoint working!")
            print(f"API Title: {data.get('info', {}).get('title', 'N/A')}")
            print(f"API Version: {data.get('info', {}).get('version', 'N/A')}")
            print(f"Available Paths: {len(data.get('paths', {}))}")
            return True
        else:
            print(f"❌ OpenAPI endpoint failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing HDBank Chatbot API")
    print("=" * 40)
    
    # Give server time to start
    print("Waiting for server to be ready...")
    time.sleep(2)
    
    success = True
    success &= test_health_endpoint()
    success &= test_openapi_endpoint()
    
    print("\n" + "=" * 40)
    if success:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")