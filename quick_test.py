"""
Quick test để verify HDBank API hoạt động
"""

import asyncio
import aiohttp
import json

async def test_api():
    """Test API endpoints async"""
    base_url = "http://127.0.0.1:8001"
    
    async with aiohttp.ClientSession() as session:
        try:
            print("🧪 Testing HDBank API on port 8001")
            print("=" * 40)
            
            # Test health endpoint
            print("Testing /health endpoint...")
            async with session.get(f"{base_url}/health") as resp:
                print(f"Status: {resp.status}")
                if resp.status == 200:
                    data = await resp.json()
                    print("✅ Health endpoint working!")
                    print(f"Response: {json.dumps(data, indent=2)}")
                else:
                    print(f"❌ Error: {await resp.text()}")
                    
        except Exception as e:
            print(f"❌ Connection error: {e}")
            print("Server might not be running or may have stopped")

if __name__ == "__main__":
    asyncio.run(test_api())