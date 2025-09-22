"""
FastAPI Dependencies cho HDBank AI Chatbot
Dependency injection cho RAG service, authentication, rate limiting, etc.
"""

import asyncio
import time
from typing import Dict, Any, Optional, Annotated
from datetime import datetime, timezone
import logging

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# Import core services
from core import get_chatbot_service, ChatbotService

logger = logging.getLogger(__name__)

# Security scheme
security = HTTPBearer(auto_error=False)

# Rate limiting storage (in-memory - trong production nên dùng Redis)
request_counts: Dict[str, Dict[str, Any]] = {}

class RateLimiter:
    """Simple rate limiter implementation"""
    
    def __init__(self, requests_per_minute: int = 60, requests_per_hour: int = 1000):
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
    
    def is_allowed(self, client_id: str) -> bool:
        """
        Check if request is allowed based on rate limits
        
        Args:
            client_id: Client identifier (IP, user ID, etc.)
            
        Returns:
            bool: True if request allowed, False otherwise
        """
        now = datetime.now(timezone.utc)
        current_minute = int(now.timestamp() // 60)
        current_hour = int(now.timestamp() // 3600)
        
        if client_id not in request_counts:
            request_counts[client_id] = {
                "minute_count": 0,
                "hour_count": 0,
                "current_minute": current_minute,
                "current_hour": current_hour
            }
        
        client_data = request_counts[client_id]
        
        # Reset minute counter nếu sang phút mới
        if client_data["current_minute"] != current_minute:
            client_data["minute_count"] = 0
            client_data["current_minute"] = current_minute
        
        # Reset hour counter nếu sang giờ mới
        if client_data["current_hour"] != current_hour:
            client_data["hour_count"] = 0
            client_data["current_hour"] = current_hour
        
        # Check limits
        if client_data["minute_count"] >= self.requests_per_minute:
            return False
        
        if client_data["hour_count"] >= self.requests_per_hour:
            return False
        
        # Increment counters
        client_data["minute_count"] += 1
        client_data["hour_count"] += 1
        
        return True

# Global rate limiter instance
rate_limiter = RateLimiter()

async def get_rate_limiter() -> RateLimiter:
    """Dependency để get rate limiter"""
    return rate_limiter

async def check_rate_limit(
    request: Request,
    limiter: RateLimiter = Depends(get_rate_limiter)
) -> None:
    """
    Check rate limit cho request
    
    Args:
        request: FastAPI request object
        limiter: Rate limiter instance
        
    Raises:
        HTTPException: Nếu rate limit exceeded
    """
    # Get client ID (IP address hoặc user ID nếu authenticated)
    client_id = request.client.host if request.client else "unknown"
    
    # Check user ID từ headers nếu có
    user_id = request.headers.get("X-User-ID")
    if user_id:
        client_id = f"user_{user_id}"
    
    if not limiter.is_allowed(client_id):
        logger.warning(f"Rate limit exceeded for client: {client_id}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again later.",
            headers={"Retry-After": "60"}
        )

async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[Dict[str, Any]]:
    """
    Get current user từ authentication token (optional)
    
    Args:
        request: FastAPI request object
        credentials: HTTP Bearer token credentials
        
    Returns:
        User info dict hoặc None nếu không authenticated
    """
    # Check for user ID trong headers (simple auth)
    user_id = request.headers.get("X-User-ID")
    session_id = request.headers.get("X-Session-ID")
    
    if user_id:
        return {
            "user_id": user_id,
            "session_id": session_id,
            "authenticated": True,
            "auth_method": "header"
        }
    
    # Check Bearer token nếu có
    if credentials:
        # Trong thực tế sẽ validate JWT token
        # Hiện tại chỉ return basic info
        return {
            "user_id": "token_user",
            "session_id": None,
            "authenticated": True,
            "auth_method": "bearer",
            "token": credentials.credentials
        }
    
    # Anonymous user
    return {
        "user_id": None,
        "session_id": None,
        "authenticated": False,
        "auth_method": None
    }

async def get_conversation_id(request: Request) -> str:
    """
    Get hoặc generate conversation ID
    
    Args:
        request: FastAPI request object
        
    Returns:
        Conversation ID string
    """
    # Try to get từ headers
    conversation_id = request.headers.get("X-Conversation-ID")
    
    if conversation_id:
        return conversation_id
    
    # Try to get từ session
    session_id = request.headers.get("X-Session-ID")
    if session_id:
        return f"session_{session_id}"
    
    # Generate based on user ID và timestamp
    user_id = request.headers.get("X-User-ID", "anonymous")
    timestamp = int(time.time())
    
    return f"{user_id}_{timestamp}"

async def get_request_metadata(request: Request) -> Dict[str, Any]:
    """
    Get request metadata cho logging và analytics
    
    Args:
        request: FastAPI request object
        
    Returns:
        Request metadata dict
    """
    return {
        "method": request.method,
        "url": str(request.url),
        "path": request.url.path,
        "query_params": dict(request.query_params),
        "headers": dict(request.headers),
        "client_host": request.client.host if request.client else None,
        "user_agent": request.headers.get("User-Agent"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": id(request)  # Simple request ID
    }

# Common dependencies type annotations
ChatbotServiceDep = Annotated[ChatbotService, Depends(get_chatbot_service)]
RateLimiterDep = Annotated[RateLimiter, Depends(get_rate_limiter)]
CurrentUserDep = Annotated[Optional[Dict[str, Any]], Depends(get_current_user)]
ConversationIdDep = Annotated[str, Depends(get_conversation_id)]
RequestMetadataDep = Annotated[Dict[str, Any], Depends(get_request_metadata)]

# Dependency factory functions
def require_authentication():
    """
    Dependency factory để require authentication
    
    Returns:
        Dependency function
    """
    async def _require_auth(
        user: CurrentUserDep
    ) -> Dict[str, Any]:
        if not user or not user.get("authenticated"):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Bearer"}
            )
        return user
    
    return _require_auth

def require_api_key(api_key: str):
    """
    Dependency factory để require specific API key
    
    Args:
        api_key: Required API key
        
    Returns:
        Dependency function
    """
    async def _require_api_key(request: Request) -> None:
        provided_key = request.headers.get("X-API-Key")
        
        if not provided_key or provided_key != api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key"
            )
    
    return _require_api_key

async def validate_request_size(request: Request) -> None:
    """
    Validate request content size
    
    Args:
        request: FastAPI request object
        
    Raises:
        HTTPException: Nếu request quá lớn
    """
    content_length = request.headers.get("Content-Length")
    
    if content_length:
        size = int(content_length)
        max_size = 10 * 1024 * 1024  # 10MB
        
        if size > max_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Request too large. Maximum size: {max_size} bytes"
            )

async def health_check_dependency() -> Dict[str, Any]:
    """
    Health check dependency để monitor service status
    
    Returns:
        Health status dict
    """
    try:
        # Check chatbot service
        chatbot_service = await get_chatbot_service()
        
        return {
            "status": "healthy",
            "chatbot_service": "initialized" if chatbot_service.initialized else "not_initialized",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

# Background task dependency
class BackgroundTaskManager:
    """Manager cho background tasks"""
    
    def __init__(self):
        self.tasks = {}
        self.task_counter = 0
    
    async def add_task(self, coro, task_name: Optional[str] = None) -> str:
        """
        Add background task
        
        Args:
            coro: Coroutine to run
            task_name: Optional task name
            
        Returns:
            Task ID
        """
        self.task_counter += 1
        task_id = f"task_{self.task_counter}"
        
        if task_name:
            task_id = f"{task_name}_{self.task_counter}"
        
        task = asyncio.create_task(coro)
        self.tasks[task_id] = {
            "task": task,
            "created_at": datetime.now(timezone.utc),
            "name": task_name or "background_task"
        }
        
        return task_id
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status của background task"""
        if task_id not in self.tasks:
            return None
        
        task_info = self.tasks[task_id]
        task = task_info["task"]
        
        return {
            "task_id": task_id,
            "name": task_info["name"],
            "done": task.done(),
            "cancelled": task.cancelled(),
            "created_at": task_info["created_at"].isoformat(),
            "exception": str(task.exception()) if task.done() and task.exception() else None
        }

# Global background task manager
background_task_manager = BackgroundTaskManager()

async def get_background_task_manager() -> BackgroundTaskManager:
    """Dependency để get background task manager"""
    return background_task_manager
