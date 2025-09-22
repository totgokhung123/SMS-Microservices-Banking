"""
API Models for HDBank Chatbot
Pydantic models cho request/response validation
"""

from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from enum import Enum

try:
    from pydantic import BaseModel, Field, validator
except ImportError:
    # Fallback nếu pydantic chưa được cài
    class BaseModel:
        pass
    
    def Field(*args, **kwargs):
        return None
    
    def validator(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# Enums
class MessageType(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class ChatStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    PROCESSING = "processing"

class StreamType(str, Enum):
    START = "start"
    TEXT = "text"
    CONTEXT = "context"
    END = "end"
    ERROR = "error"

# Request Models
class ChatRequest(BaseModel):
    """Request model cho chat endpoint"""
    message: str = Field(..., min_length=1, max_length=2000, description="User message")
    conversation_id: Optional[str] = Field(None, description="Conversation ID")
    user_id: Optional[str] = Field(None, description="User ID")
    context_limit: Optional[int] = Field(5, ge=1, le=10, description="Number of context documents")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=1.0, description="Response creativity")
    
    @validator('message')
    def validate_message(cls, v):
        if not v or not v.strip():
            raise ValueError('Message không được rỗng')
        return v.strip()

class SearchRequest(BaseModel):
    """Request model cho document search"""
    query: str = Field(..., min_length=1, max_length=500, description="Search query")
    top_k: Optional[int] = Field(10, ge=1, le=50, description="Number of results")
    category: Optional[str] = Field(None, description="Document category filter")
    similarity_threshold: Optional[float] = Field(0.7, ge=0.0, le=1.0, description="Minimum similarity")

# Response Models
class Source(BaseModel):
    """Model cho source document information"""
    file_name: str
    similarity: float = Field(..., ge=0.0, le=1.0)
    content_preview: Optional[str] = None

class ChatMetadata(BaseModel):
    """Metadata cho chat response"""
    response_time_ms: float
    context_used: int
    context_length: int
    model_confidence: float = Field(..., ge=0.0, le=1.0)
    sources: List[Source] = []

class ChatResponse(BaseModel):
    """Response model cho chat endpoint"""
    message: str = Field(..., description="Bot response message")
    conversation_id: str = Field(..., description="Conversation ID")
    timestamp: datetime = Field(..., description="Response timestamp")
    status: ChatStatus = Field(ChatStatus.SUCCESS, description="Response status")
    metadata: Optional[ChatMetadata] = Field(None, description="Response metadata")

class StreamChatResponse(BaseModel):
    """Response model cho streaming chat"""
    type: StreamType = Field(..., description="Stream chunk type")
    conversation_id: str = Field(..., description="Conversation ID")
    timestamp: datetime = Field(default_factory=datetime.now, description="Chunk timestamp")
    data: Optional[Dict[str, Any]] = Field(None, description="Chunk data")

class SearchResult(BaseModel):
    """Model cho search result item"""
    content: str = Field(..., description="Document content")
    file_name: str = Field(..., description="Source file name")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Similarity score")
    chunk_id: str = Field(..., description="Chunk ID")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class SearchResponse(BaseModel):
    """Response model cho search endpoint"""
    query: str = Field(..., description="Original search query")
    results: List[SearchResult] = Field(..., description="Search results")
    total_found: int = Field(..., ge=0, description="Total results found")
    timestamp: datetime = Field(..., description="Search timestamp")

# Conversation Models
class Message(BaseModel):
    """Model cho single message trong conversation"""
    id: str = Field(..., description="Message ID")
    type: MessageType = Field(..., description="Message type")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(..., description="Message timestamp")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Message metadata")

class ConversationHistory(BaseModel):
    """Model cho conversation history"""
    conversation_id: str = Field(..., description="Conversation ID")
    messages: List[Message] = Field(..., description="Conversation messages")
    created_at: datetime = Field(..., description="Conversation creation time")
    updated_at: datetime = Field(..., description="Last update time")
    user_id: Optional[str] = Field(None, description="User ID")

# Health Check Models
class HealthResponse(BaseModel):
    """Basic health check response"""
    status: str = Field(..., description="Health status")
    timestamp: datetime = Field(..., description="Check timestamp")
    message: str = Field(..., description="Status message")

class SystemStats(BaseModel):
    """System statistics model"""
    cpu_percent: float = Field(..., ge=0.0, le=100.0)
    memory_used_percent: float = Field(..., ge=0.0, le=100.0)
    disk_used_percent: float = Field(..., ge=0.0, le=100.0)
    uptime_seconds: float = Field(..., ge=0.0)

class RAGStats(BaseModel):
    """RAG system statistics"""
    initialized: bool = Field(..., description="RAG system initialized")
    total_documents: int = Field(..., ge=0, description="Total indexed documents")
    total_chunks: int = Field(..., ge=0, description="Total text chunks")
    embedding_dimension: int = Field(..., ge=0, description="Embedding dimension")
    last_updated: Optional[datetime] = Field(None, description="Last index update")

class DetailedHealthResponse(BaseModel):
    """Detailed health check response"""
    status: str = Field(..., description="Overall health status")
    timestamp: datetime = Field(..., description="Check timestamp")
    api_version: str = Field(..., description="API version")
    system: SystemStats = Field(..., description="System metrics")
    rag_service: RAGStats = Field(..., description="RAG service stats")

# Error Models
class ErrorResponse(BaseModel):
    """Standard error response"""
    error: bool = Field(True, description="Error flag")
    message: str = Field(..., description="Error message")
    status_code: int = Field(..., description="HTTP status code")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")

# Configuration Models
class APIConfig(BaseModel):
    """API configuration model"""
    host: str = Field("0.0.0.0", description="API host")
    port: int = Field(8000, ge=1, le=65535, description="API port")
    workers: int = Field(1, ge=1, description="Number of workers")
    reload: bool = Field(False, description="Auto-reload on changes")
    log_level: str = Field("info", description="Logging level")

class RAGConfig(BaseModel):
    """RAG configuration model"""
    embedding_model: str = Field(..., description="Embedding model name")
    chunk_size: int = Field(512, ge=100, le=2000, description="Text chunk size")
    chunk_overlap: int = Field(64, ge=0, le=500, description="Chunk overlap")
    top_k: int = Field(5, ge=1, le=20, description="Default top-k results")
    similarity_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Similarity threshold")

# Export all models
__all__ = [
    # Enums
    "MessageType",
    "ChatStatus", 
    "StreamType",
    
    # Request models
    "ChatRequest",
    "SearchRequest",
    
    # Response models
    "ChatResponse",
    "StreamChatResponse", 
    "SearchResponse",
    "SearchResult",
    "Source",
    "ChatMetadata",
    
    # Conversation models
    "Message",
    "ConversationHistory",
    
    # Health models
    "HealthResponse",
    "SystemStats",
    "RAGStats", 
    "DetailedHealthResponse",
    
    # Error models
    "ErrorResponse",
    
    # Config models
    "APIConfig",
    "RAGConfig"
]