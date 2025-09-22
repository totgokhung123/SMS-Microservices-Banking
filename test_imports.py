"""
Test script để kiểm tra tất cả imports trong codebase HDBank
"""

import sys
import os
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

def test_imports():
    """Test tất cả các imports quan trọng"""
    
    print("🔍 Testing imports for HDBank AI Chatbot...")
    print("=" * 60)
    
    # Test 1: Core FastAPI modules
    try:
        print("Testing FastAPI imports...")
        import fastapi
        from fastapi import FastAPI, HTTPException, Depends
        from fastapi.middleware.cors import CORSMiddleware
        print("✅ FastAPI imports successful")
    except ImportError as e:
        print(f"❌ FastAPI import error: {e}")
        return False
    
    # Test 2: Pydantic models
    try:
        print("Testing Pydantic imports...")
        from pydantic import BaseModel, Field, ValidationError
        print("✅ Pydantic imports successful")
    except ImportError as e:
        print(f"❌ Pydantic import error: {e}")
        return False
    
    # Test 3: AI/ML libraries
    try:
        print("Testing AI/ML library imports...")
        import torch
        import numpy as np
        from sentence_transformers import SentenceTransformer
        import faiss
        print("✅ AI/ML library imports successful")
    except ImportError as e:
        print(f"❌ AI/ML library import error: {e}")
        return False
    
    # Test 4: Local modules - Utils
    try:
        print("Testing local utils imports...")
        from utils.logger import setup_logger, get_logger
        from utils.exceptions import ChatbotBaseException, RAGException, ConfigurationException
        print("✅ Utils imports successful")
    except ImportError as e:
        print(f"❌ Utils import error: {e}")
        return False
    
    # Test 5: Local modules - API
    try:
        print("Testing API modules imports...")
        from api.models import ChatRequest, ChatResponse, HealthResponse
        from api.dependencies import ChatbotServiceDep, check_rate_limit
        print("✅ API modules imports successful")
    except ImportError as e:
        print(f"❌ API modules import error: {e}")
        return False
    
    # Test 6: Local modules - Core
    try:
        print("Testing core modules imports...")
        from core.chatbot_service import ChatbotService, get_chatbot_service
        print("✅ Core modules imports successful")
    except ImportError as e:
        print(f"❌ Core modules import error: {e}")
        return False
    
    # Test 7: Local modules - RAG
    try:
        print("Testing RAG modules imports...")
        from rag import RAGPipeline, get_rag_info
        from rag.data_loader import DocumentLoader
        from rag.embedder import TextEmbedder
        from rag.vector_store import VectorStore
        from rag.retriever import RAGRetriever
        print("✅ RAG modules imports successful")
    except ImportError as e:
        print(f"❌ RAG modules import error: {e}")
        return False
    
    # Test 8: Main application
    try:
        print("Testing main application import...")
        from main import app
        print("✅ Main application import successful")
    except ImportError as e:
        print(f"❌ Main application import error: {e}")
        return False
    
    # Test 9: Routers
    try:
        print("Testing router imports...")
        from api.routers import chat, health
        print("✅ Router imports successful")
    except ImportError as e:
        print(f"❌ Router import error: {e}")
        return False
    
    print("=" * 60)
    print("🎉 ALL IMPORTS SUCCESSFUL!")
    return True

def test_basic_functionality():
    """Test basic functionality của các components"""
    
    print("\n🔍 Testing basic functionality...")
    print("=" * 60)
    
    try:
        # Test logger
        from utils.logger import get_logger
        logger = get_logger("test")
        logger.info("Test log message")
        print("✅ Logger functionality working")
        
        # Test ChatbotService initialization
        from core.chatbot_service import ChatbotService
        service = ChatbotService()
        print("✅ ChatbotService can be instantiated")
        
        # Test API models
        from api.models import ChatRequest
        request = ChatRequest(
            message="Test message",
            user_id="test_user",
            conversation_id="test_conv"
        )
        print("✅ API models validation working")
        
        # Test FastAPI app
        from main import app
        print(f"✅ FastAPI app created: {app.title}")
        
        print("=" * 60)
        print("🎉 ALL FUNCTIONALITY TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Functionality test error: {e}")
        return False

if __name__ == "__main__":
    print("🏦 HDBank AI Chatbot - Import & Functionality Test")
    print(f"📁 Project root: {project_root}")
    print(f"📁 Python path: {sys.path[0]}")
    
    # Run tests
    imports_ok = test_imports()
    if imports_ok:
        functionality_ok = test_basic_functionality()
        
        if functionality_ok:
            print("\n🎯 OVERALL RESULT: ✅ ALL TESTS PASSED")
            print("🚀 Ready to start server!")
        else:
            print("\n🎯 OVERALL RESULT: ❌ FUNCTIONALITY TESTS FAILED")
    else:
        print("\n🎯 OVERALL RESULT: ❌ IMPORT TESTS FAILED")