"""
Exception handling utilities cho HDBank AI Chatbot
Custom exceptions và error handling mechanisms
"""

import logging
import traceback
from typing import Any, Dict, Optional, Union
from datetime import datetime, timezone

from .logger import get_logger, log_error

logger = get_logger(__name__)

class ChatbotBaseException(Exception):
    """Base exception cho tất cả chatbot-related errors"""
    
    def __init__(self, message: str, error_code: str = None, 
                 details: Dict[str, Any] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        self.timestamp = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dict format"""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat()
        }

class RAGException(ChatbotBaseException):
    """Exception cho RAG system errors"""
    pass

class ModelException(ChatbotBaseException):
    """Exception cho AI model errors"""
    pass

class ConfigurationException(ChatbotBaseException):
    """Exception cho configuration errors"""
    pass

class ValidationException(ChatbotBaseException):
    """Exception cho input validation errors"""
    pass

class RateLimitException(ChatbotBaseException):
    """Exception cho rate limit exceeded"""
    pass

class AuthenticationException(ChatbotBaseException):
    """Exception cho authentication errors"""
    pass

class ServiceUnavailableException(ChatbotBaseException):
    """Exception khi service temporarily unavailable"""
    pass

# Specific RAG exceptions
class DocumentNotFoundException(RAGException):
    """Khi không tìm thấy document"""
    pass

class EmbeddingException(RAGException):
    """Khi có lỗi trong quá trình embedding"""
    pass

class VectorStoreException(RAGException):
    """Khi có lỗi với vector store operations"""
    pass

class RetrievalException(RAGException):
    """Khi có lỗi trong document retrieval"""
    pass

# Model-specific exceptions
class ModelLoadException(ModelException):
    """Khi không thể load AI model"""
    pass

class GenerationException(ModelException):
    """Khi có lỗi trong text generation"""
    pass

class TokenizationException(ModelException):
    """Khi có lỗi trong tokenization"""
    pass

# Error handling utilities
class ErrorHandler:
    """Central error handler class"""
    
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or get_logger(__name__)
    
    def handle_exception(self, error: Exception, 
                        context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Handle exception và return standardized error response
        
        Args:
            error: Exception instance
            context: Additional context information
            
        Returns:
            Standardized error response dict
        """
        context = context or {}
        
        # Log the error
        log_error(self.logger, error, context)
        
        # Create error response based on exception type
        if isinstance(error, ChatbotBaseException):
            return self._handle_chatbot_exception(error, context)
        elif isinstance(error, ValueError):
            return self._handle_validation_error(error, context)
        elif isinstance(error, PermissionError):
            return self._handle_permission_error(error, context)
        elif isinstance(error, FileNotFoundError):
            return self._handle_file_not_found(error, context)
        elif isinstance(error, ConnectionError):
            return self._handle_connection_error(error, context)
        else:
            return self._handle_generic_error(error, context)
    
    def _handle_chatbot_exception(self, error: ChatbotBaseException,
                                 context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle custom chatbot exceptions"""
        return {
            "error": True,
            "error_type": error.__class__.__name__,
            "error_code": error.error_code,
            "message": error.message,
            "details": error.details,
            "timestamp": error.timestamp.isoformat(),
            "context": context
        }
    
    def _handle_validation_error(self, error: ValueError,
                               context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle validation errors"""
        return {
            "error": True,
            "error_type": "ValidationError",
            "error_code": "VALIDATION_FAILED",
            "message": "Input validation failed",
            "details": {"validation_error": str(error)},
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "context": context
        }
    
    def _handle_permission_error(self, error: PermissionError,
                               context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle permission errors"""
        return {
            "error": True,
            "error_type": "PermissionError",
            "error_code": "ACCESS_DENIED",
            "message": "Access denied",
            "details": {"permission_error": str(error)},
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "context": context
        }
    
    def _handle_file_not_found(self, error: FileNotFoundError,
                             context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle file not found errors"""
        return {
            "error": True,
            "error_type": "FileNotFoundError", 
            "error_code": "FILE_NOT_FOUND",
            "message": "Required file not found",
            "details": {"file_error": str(error)},
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "context": context
        }
    
    def _handle_connection_error(self, error: ConnectionError,
                               context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle connection errors"""
        return {
            "error": True,
            "error_type": "ConnectionError",
            "error_code": "CONNECTION_FAILED",
            "message": "Connection failed",
            "details": {"connection_error": str(error)},
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "context": context
        }
    
    def _handle_generic_error(self, error: Exception,
                            context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle generic exceptions"""
        return {
            "error": True,
            "error_type": type(error).__name__,
            "error_code": "INTERNAL_ERROR",
            "message": "An internal error occurred",
            "details": {
                "error_message": str(error),
                "traceback": traceback.format_exc()
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "context": context
        }

# Global error handler instance
error_handler = ErrorHandler()

def handle_exception(error: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Global exception handler function
    
    Args:
        error: Exception to handle
        context: Additional context
        
    Returns:
        Standardized error response
    """
    return error_handler.handle_exception(error, context)

# Decorator cho automatic error handling
def handle_errors(default_response: Any = None, 
                 log_errors: bool = True,
                 reraise: bool = False):
    """
    Decorator để automatically handle errors trong functions
    
    Args:
        default_response: Default response khi có error
        log_errors: Whether to log errors
        reraise: Whether to reraise the exception
        
    Returns:
        Decorated function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    context = {
                        "function": func.__name__,
                        "args_count": len(args),
                        "kwargs_keys": list(kwargs.keys())
                    }
                    log_error(logger, e, context)
                
                if reraise:
                    raise
                
                return default_response
        
        # Preserve function metadata
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        
        return wrapper
    
    return decorator

def safe_execute(func, *args, default=None, log_errors=True, **kwargs):
    """
    Safely execute function với error handling
    
    Args:
        func: Function to execute
        *args: Function arguments
        default: Default return value on error
        log_errors: Whether to log errors
        **kwargs: Function keyword arguments
        
    Returns:
        Function result hoặc default value
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if log_errors:
            context = {
                "function": func.__name__ if hasattr(func, '__name__') else str(func),
                "args_count": len(args),
                "kwargs_keys": list(kwargs.keys())
            }
            log_error(logger, e, context)
        
        return default

# Context manager cho error handling
class ErrorContext:
    """Context manager cho automatic error handling"""
    
    def __init__(self, operation: str, default_response: Any = None,
                 log_errors: bool = True, reraise: bool = True):
        self.operation = operation
        self.default_response = default_response
        self.log_errors = log_errors
        self.reraise = reraise
        self.error = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.error = exc_val
            
            if self.log_errors:
                context = {"operation": self.operation}
                log_error(logger, exc_val, context)
            
            if not self.reraise:
                return True  # Suppress exception
        
        return False  # Let exception propagate

# Validation utilities
def validate_input(value: Any, validator_func, error_message: str = None):
    """
    Validate input với custom validator function
    
    Args:
        value: Value to validate
        validator_func: Function to validate với (returns bool)
        error_message: Custom error message
        
    Raises:
        ValidationException: Nếu validation fails
    """
    try:
        if not validator_func(value):
            raise ValidationException(
                error_message or f"Validation failed for value: {value}",
                error_code="VALIDATION_FAILED",
                details={"value": str(value)}
            )
    except Exception as e:
        if isinstance(e, ValidationException):
            raise
        
        raise ValidationException(
            f"Validation error: {e}",
            error_code="VALIDATION_ERROR",
            details={"validation_exception": str(e)}
        )

def validate_required_fields(data: Dict[str, Any], required_fields: list):
    """
    Validate required fields trong dict
    
    Args:
        data: Data dict to validate
        required_fields: List of required field names
        
    Raises:
        ValidationException: Nếu required fields missing
    """
    missing_fields = []
    
    for field in required_fields:
        if field not in data or data[field] is None:
            missing_fields.append(field)
    
    if missing_fields:
        raise ValidationException(
            f"Missing required fields: {missing_fields}",
            error_code="MISSING_REQUIRED_FIELDS",
            details={"missing_fields": missing_fields}
        )