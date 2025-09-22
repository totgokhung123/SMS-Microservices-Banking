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
    Returns system status and basic info
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc),
        message="HDBank Chatbot API is running"
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
        
        # RAG service status
        rag_status = "unknown"
        rag_stats = {}
        
        if rag_service:
            try:
                rag_stats = rag_service.get_stats()
                rag_status = "healthy" if rag_stats.get('initialized', False) else "initializing"
            except Exception as e:
                rag_status = f"error: {str(e)}"
        
        health_data = {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "api": {
                "name": "HDBank Chatbot API",
                "version": "1.0.0",
                "uptime_seconds": time.time() - psutil.boot_time()
            },
            "system": {
                "cpu_percent": cpu_percent,
                "memory": {
                    "total_gb": round(memory.total / 1024**3, 2),
                    "available_gb": round(memory.available / 1024**3, 2),
                    "used_percent": memory.percent
                },
                "disk": {
                    "total_gb": round(disk.total / 1024**3, 2),
                    "free_gb": round(disk.free / 1024**3, 2),
                    "used_percent": round((disk.used / disk.total) * 100, 2)
                }
            },
            "rag_service": {
                "status": rag_status,
                "stats": rag_stats
            }
        }
        
        # Determine overall status
        if cpu_percent > 90 or memory.percent > 90:
            health_data["status"] = "degraded"
        
        if rag_status.startswith("error"):
            health_data["status"] = "unhealthy"
        
        return JSONResponse(content=health_data)
        
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": str(e)
            }
        )

@router.get("/rag")
async def rag_health_check(rag_service = Depends(get_rag_service)):
    """
    Specific health check cho RAG system
    """
    if not rag_service:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unavailable",
                "message": "RAG service not initialized"
            }
        )
    
    try:
        stats = rag_service.get_stats()
        
        # Test basic functionality
        test_query = "Thẻ tín dụng HDBank"
        test_result = await rag_service.search_documents(test_query, top_k=1)
        
        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "rag_service": {
                "initialized": stats.get('initialized', False),
                "vector_store": stats.get('vector_store', {}),
                "embedder": stats.get('embedder', {}),
                "test_query": {
                    "query": test_query,
                    "results_count": len(test_result),
                    "response_time_ms": "< 1000"  # Placeholder
                }
            }
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "message": f"RAG service error: {str(e)}",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

@router.get("/readiness")
async def readiness_check(rag_service = Depends(get_rag_service)):
    """
    Kubernetes-style readiness probe
    Returns 200 nếu service ready to handle requests
    """
    try:
        if not rag_service:
            return JSONResponse(
                status_code=503,
                content={"ready": False, "message": "RAG service not available"}
            )
        
        stats = rag_service.get_stats()
        if not stats.get('initialized', False):
            return JSONResponse(
                status_code=503,
                content={"ready": False, "message": "RAG service not initialized"}
            )
        
        return {"ready": True, "timestamp": datetime.now(timezone.utc).isoformat()}
        
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"ready": False, "error": str(e)}
        )

@router.get("/liveness")
async def liveness_check():
    """
    Kubernetes-style liveness probe
    Returns 200 nếu application đang running
    """
    return {
        "alive": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pid": psutil.Process().pid
    }
