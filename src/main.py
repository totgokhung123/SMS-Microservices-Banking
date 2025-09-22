"""
HDBank Banking Chatbot - Main FastAPI Application
Entry point cho backend API server với RAG integration
"""

import os
import sys
import logging
from contextlib import asynccontextmanager
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Import local modules
from api.routers import chat, health
from core.chatbot_service import ChatbotService, get_chatbot_service
from utils.logger import setup_logger

# Setup logging
logger = setup_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    
    # Startup
    logger.info("🚀 Starting HDBank Chatbot API...")
    
    try:
        # Initialize RAG service
        chatbot_service = await get_chatbot_service()
        logger.info("✅ Chatbot service initialized successfully")
        
        # Add to app state
        app.state.chatbot_service = chatbot_service
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize chatbot service: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down HDBank Chatbot API...")
    
    if hasattr(app.state, 'chatbot_service'):
        await app.state.chatbot_service.cleanup()
        logger.info("✅ Chatbot service cleaned up")

# Create FastAPI application
app = FastAPI(
    title="HDBank AI Chatbot API",
    description="API backend cho HDBank AI Chatbot với RAG system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
    contact={
        "name": "HDBank AI Team",
        "email": "ai@hdbank.com.vn"
    },
    license_info={
        "name": "HDBank Internal License",
        "url": "https://www.hdbank.com.vn"
    }
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React dev server
        "http://localhost:8080",  # Vue dev server  
        "http://localhost:5000",  # Flutter web
        "https://hdbank.com.vn",  # Production domain
        "https://api.hdbank.com.vn",  # API domain
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Add trusted host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=[
        "localhost",
        "127.0.0.1",
        "hdbank.com.vn",
        "*.hdbank.com.vn"
    ]
)

# Global exception handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Global HTTP exception handler"""
    logger.error(f"HTTP Exception: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "message": exc.detail,
            "status_code": exc.status_code
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Global exception handler for all other exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "message": "Internal server error",
            "status_code": 500
        }
    )

# Include routers
app.include_router(
    health.router,
    prefix="/api/v1/health",
    tags=["health"]
)

app.include_router(
    chat.router,
    prefix="/api/v1/chat",
    tags=["chat"]
)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "HDBank AI Chatbot API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/api/v1/health"
    }

# API Info endpoint
@app.get("/api/v1/info")
async def api_info():
    """Get API information and stats"""
    return {
        "api": {
            "name": "HDBank AI Chatbot API",
            "version": "1.0.0",
            "description": "Backend API cho HDBank AI Chatbot với RAG system"
        },
        "features": [
            "RAG-powered conversation",
            "Vietnamese language support", 
            "Banking domain expertise",
            "Real-time responses",
            "Context-aware answers"
        ],
        "endpoints": {
            "chat": "/api/v1/chat/",
            "health": "/api/v1/health/",
            "docs": "/docs"
        }
    }

# Development server
if __name__ == "__main__":
    # Development configuration
    log_level = os.getenv("LOG_LEVEL", "info")
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "true").lower() == "true"
    
    logger.info(f"Starting development server on {host}:{port}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level,
        access_log=True
    )
