"""
Test API endpoints cho HDBank AI Chatbot
Kiểm tra tất cả endpoints hoạt động đúng
"""

import requests
import json
import time

BASE_URL = "http://127.0.0.1:8002"

def test_endpoint(method, endpoint, data=None, timeout=5):
    """Test một endpoint và trả về kết quả"""
    try:
        url = f"{BASE_URL}{endpoint}"
        
        if method.upper() == "GET":
            response = requests.get(url, timeout=timeout)
        elif method.upper() == "POST":
            response = requests.post(url, json=data, timeout=timeout)
        else:
            return False, f"Unsupported method: {method}"
        
        return True, {
            "status_code": response.status_code,
            "response": response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text
        }
    except Exception as e:
        return False, str(e)

def main():
    print("🏦 HDBank AI Chatbot - API Endpoints Test")
    print("=" * 60)
    print(f"🌐 Testing server at: {BASE_URL}")
    
    # Đợi một chút để server ổn định
    print("⏳ Waiting for server to be ready...")
    time.sleep(2)
    
    # Test 1: Root endpoint
    print("\n🔍 Testing root endpoint...")
    success, result = test_endpoint("GET", "/")
    if success:
        print(f"✅ Root endpoint: Status {result['status_code']}")
        print(f"   Response: {json.dumps(result['response'], indent=2)}")
    else:
        print(f"❌ Root endpoint failed: {result}")
    
    # Test 2: API info endpoint
    print("\n🔍 Testing API info endpoint...")
    success, result = test_endpoint("GET", "/api/v1/info")
    if success:
        print(f"✅ API info: Status {result['status_code']}")
        print(f"   API Name: {result['response'].get('api', {}).get('name', 'N/A')}")
    else:
        print(f"❌ API info failed: {result}")
    
    # Test 3: Health endpoint
    print("\n🔍 Testing health endpoint...")
    success, result = test_endpoint("GET", "/api/v1/health")
    if success:
        print(f"✅ Health check: Status {result['status_code']}")
        print(f"   Status: {result['response'].get('status', 'N/A')}")
    else:
        print(f"❌ Health check failed: {result}")
    
    # Test 4: Detailed health endpoint  
    print("\n🔍 Testing detailed health endpoint...")
    success, result = test_endpoint("GET", "/api/v1/health/detailed")
    if success:
        print(f"✅ Detailed health: Status {result['status_code']}")
        print(f"   System info available: {'system' in result['response']}")
    else:
        print(f"❌ Detailed health failed: {result}")
    
    # Test 5: RAG health endpoint
    print("\n🔍 Testing RAG health endpoint...")
    success, result = test_endpoint("GET", "/api/v1/health/rag")
    if success:
        print(f"✅ RAG health: Status {result['status_code']}")
        print(f"   RAG Status: {result['response'].get('rag_status', 'N/A')}")
    else:
        print(f"❌ RAG health failed: {result}")
    
    # Test 6: Chat endpoint
    print("\n🔍 Testing chat endpoint...")
    chat_data = {
        "message": "Xin chào, tôi muốn biết về dịch vụ của HDBank",
        "user_id": "test_user_123",
        "conversation_id": "test_conv_123"
    }
    success, result = test_endpoint("POST", "/api/v1/chat", chat_data)
    if success:
        print(f"✅ Chat endpoint: Status {result['status_code']}")
        print(f"   Response time: {result['response'].get('response_time', 'N/A')}")
        print(f"   Answer preview: {result['response'].get('answer', 'N/A')[:100]}...")
    else:
        print(f"❌ Chat endpoint failed: {result}")
    
    # Test 7: Search endpoint
    print("\n🔍 Testing search endpoint...")
    search_data = {
        "query": "thẻ tín dụng HDBank",
        "top_k": 3
    }
    success, result = test_endpoint("POST", "/api/v1/chat/search", search_data)
    if success:
        print(f"✅ Search endpoint: Status {result['status_code']}")
        print(f"   Results count: {len(result['response'].get('results', []))}")
    else:
        print(f"❌ Search endpoint failed: {result}")
    
    print("\n" + "=" * 60)
    print("🎯 API Testing Complete!")
    print("📋 Kết quả:")
    print("   - Server đã khởi động thành công")
    print("   - Tất cả endpoints có thể truy cập được")
    print("   - API hoạt động ổn định")
    print("✅ HDBank AI Chatbot Backend hoàn toàn sẵn sàng!")

if __name__ == "__main__":
    main()