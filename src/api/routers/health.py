"""
Health Check Router - HDBank Chatbot API
Health monitoring endpoints cho system status
"""

import psutil
import time
from datetime import datetime, timezone
from typing import Dict, Any

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from api.dependencies import ChatbotServiceDep
from api.models import HealthResponse, SystemStats

router = APIRouter()

@router.get("/", response_model=HealthResponse)
async def health_check():
    """
    Basic health check endpoint
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc),
        uptime=time.time()
    )

@router.get("/detailed")
async def detailed_health_check(chatbot_service: ChatbotServiceDep):
    """
    Detailed health check với system metrics
    """
    try:
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # RAG service stats
        rag_stats = {}
        if chatbot_service:
            try:
                rag_stats = chatbot_service.get_stats()
            except Exception as e:
                rag_stats = {"error": str(e)}
        
        system_stats = SystemStats(
            cpu_usage=cpu_percent,
            memory_usage=memory.percent,
            memory_available=memory.available,
            memory_total=memory.total,
            disk_usage=disk.percent,
            disk_free=disk.free,
            disk_total=disk.total
        )
        
        # Determine overall status
        status = "healthy"
        if cpu_percent > 90 or memory.percent > 90 or disk.percent > 90:
            status = "degraded"
        
        if not chatbot_service or not chatbot_service.initialized:
            status = "unhealthy"
        
        return JSONResponse({
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "uptime": time.time(),
            "system": system_stats.model_dump(),
            "chatbot_service": {
                "initialized": chatbot_service.initialized if chatbot_service else False,
                "stats": rag_stats
            }
        })
        
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

@router.get("/rag")
async def rag_health_check(chatbot_service: ChatbotServiceDep):
    """
    Specific RAG system health check
    """
    if not chatbot_service:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unavailable",
                "message": "RAG service not initialized"
            }
        )
    
    try:
        # Get RAG stats
        stats = chatbot_service.get_stats()
        
        # Test basic functionality
        test_query = "thẻ tín dụng"
        test_result = await chatbot_service.search_documents(test_query, top_k=1)
        
        return JSONResponse({
            "status": "healthy",
            "chatbot_service": {
                "initialized": chatbot_service.initialized,
                "stats": stats,
                "test_query": test_query,
                "test_results": len(test_result),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        })
        
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

@router.get("/ready")
async def readiness_check(chatbot_service: ChatbotServiceDep):
    """
    Kubernetes readiness probe endpoint
    """
    try:
        if not chatbot_service:
            raise Exception("RAG service not available")
        
        # Check if service is ready
        stats = chatbot_service.get_stats()
        
        if chatbot_service.initialized:
            return JSONResponse({
                "status": "ready",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
        else:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "not_ready",
                    "message": "RAG service not fully initialized",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
            
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

@router.get("/live")
async def liveness_check():
    """
    Kubernetes liveness probe endpoint
    """
    return JSONResponse({
        "status": "alive",
        "timestamp": datetime.now(timezone.utc).isoformat()
    })