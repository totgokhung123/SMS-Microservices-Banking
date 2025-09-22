"""
Logger utility cho HDBank AI Chatbot
Centralized logging configuration và utilities
"""

import logging
import logging.handlers
import sys
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
import os

class StructuredFormatter(logging.Formatter):
    """
    Custom formatter cho structured logging
    Output JSON format cho easy parsing
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        
        # Base log data
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread": record.thread,
            "process": record.process
        }
        
        # Add exception info nếu có
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields từ record
        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in {
                'name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 'filename',
                'module', 'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                'thread', 'threadName', 'processName', 'process', 'exc_info', 'exc_text',
                'stack_info', 'getMessage'
            }:
                extra_fields[key] = value
        
        if extra_fields:
            log_data["extra"] = extra_fields
        
        return json.dumps(log_data, ensure_ascii=False, default=str)

class RequestContextFilter(logging.Filter):
    """
    Filter để add request context vào log records
    """
    
    def filter(self, record: logging.LogRecord) -> bool:
        # Add request context nếu có trong thread local storage
        # Hiện tại để trống, sẽ implement sau với contextvars
        return True

def setup_logger(
    name: str = "hdbank_chatbot",
    level: str = "INFO",
    log_file: Optional[str] = None,
    structured: bool = True,
    include_console: bool = True
) -> logging.Logger:
    """
    Setup logger với configuration
    
    Args:
        name: Logger name
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        structured: Use structured JSON logging
        include_console: Include console handler
        
    Returns:
        Configured logger instance
    """
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatters
    if structured:
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # Console handler
    if include_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.addFilter(RequestContextFilter())
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        # Create log directory nếu chưa có
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        file_handler.addFilter(RequestContextFilter())
        logger.addHandler(file_handler)
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """
    Get logger instance với default configuration
    
    Args:
        name: Logger name (thường là __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)

# Utility functions cho logging
def log_function_call(logger: logging.Logger, func_name: str, 
                     args: tuple = (), kwargs: dict = None, 
                     level: str = "DEBUG"):
    """
    Log function call với parameters
    
    Args:
        logger: Logger instance
        func_name: Function name
        args: Function args
        kwargs: Function kwargs
        level: Log level
    """
    kwargs = kwargs or {}
    
    log_data = {
        "event": "function_call",
        "function": func_name,
        "args_count": len(args),
        "kwargs_keys": list(kwargs.keys())
    }
    
    # Add args và kwargs nếu không sensitive
    safe_args = []
    for arg in args:
        if isinstance(arg, (str, int, float, bool, list, dict)):
            safe_args.append(arg)
        else:
            safe_args.append(f"<{type(arg).__name__}>")
    
    log_data["args"] = safe_args[:3]  # Limit to first 3 args
    
    safe_kwargs = {}
    for key, value in kwargs.items():
        if key.lower() not in {'password', 'token', 'secret', 'key'}:
            if isinstance(value, (str, int, float, bool)):
                safe_kwargs[key] = value
            else:
                safe_kwargs[key] = f"<{type(value).__name__}>"
    
    log_data["kwargs"] = safe_kwargs
    
    getattr(logger, level.lower())(
        f"Calling {func_name}",
        extra=log_data
    )

def log_performance(logger: logging.Logger, operation: str, 
                   duration: float, metadata: Dict[str, Any] = None):
    """
    Log performance metrics
    
    Args:
        logger: Logger instance
        operation: Operation name
        duration: Duration in seconds
        metadata: Additional metadata
    """
    metadata = metadata or {}
    
    log_data = {
        "event": "performance",
        "operation": operation,
        "duration_seconds": round(duration, 4),
        **metadata
    }
    
    # Determine log level based on duration
    if duration > 5.0:
        level = "WARNING"
    elif duration > 2.0:
        level = "INFO"
    else:
        level = "DEBUG"
    
    getattr(logger, level.lower())(
        f"Operation {operation} completed in {duration:.4f}s",
        extra=log_data
    )

def log_error(logger: logging.Logger, error: Exception, 
              context: Dict[str, Any] = None, level: str = "ERROR"):
    """
    Log error với context information
    
    Args:
        logger: Logger instance
        error: Exception instance
        context: Additional context
        level: Log level
    """
    context = context or {}
    
    log_data = {
        "event": "error",
        "error_type": type(error).__name__,
        "error_message": str(error),
        **context
    }
    
    getattr(logger, level.lower())(
        f"Error occurred: {type(error).__name__}: {error}",
        exc_info=True,
        extra=log_data
    )

def log_api_request(logger: logging.Logger, method: str, path: str,
                   status_code: int, duration: float,
                   user_id: Optional[str] = None,
                   request_id: Optional[str] = None):
    """
    Log API request với standardized format
    
    Args:
        logger: Logger instance
        method: HTTP method
        path: Request path
        status_code: Response status code
        duration: Request duration
        user_id: User ID (nếu có)
        request_id: Request ID (nếu có)
    """
    log_data = {
        "event": "api_request",
        "method": method,
        "path": path,
        "status_code": status_code,
        "duration_seconds": round(duration, 4),
        "user_id": user_id,
        "request_id": request_id
    }
    
    # Determine log level based on status code
    if status_code >= 500:
        level = "ERROR"
    elif status_code >= 400:
        level = "WARNING"
    else:
        level = "INFO"
    
    getattr(logger, level.lower())(
        f"{method} {path} - {status_code} - {duration:.4f}s",
        extra=log_data
    )

def log_rag_operation(logger: logging.Logger, operation: str,
                     query: str, results_count: int,
                     duration: float, confidence: Optional[float] = None):
    """
    Log RAG operations với specific format
    
    Args:
        logger: Logger instance
        operation: RAG operation (search, retrieve, etc.)
        query: Search query
        results_count: Number of results
        duration: Operation duration
        confidence: Confidence score (nếu có)
    """
    log_data = {
        "event": "rag_operation",
        "operation": operation,
        "query_length": len(query),
        "results_count": results_count,
        "duration_seconds": round(duration, 4),
        "confidence": confidence
    }
    
    logger.info(
        f"RAG {operation}: {results_count} results in {duration:.4f}s",
        extra=log_data
    )

# Context managers cho automatic logging
class LoggedOperation:
    """Context manager cho automatic performance logging"""
    
    def __init__(self, logger: logging.Logger, operation: str,
                 metadata: Dict[str, Any] = None):
        self.logger = logger
        self.operation = operation
        self.metadata = metadata or {}
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now(timezone.utc).timestamp()
        self.logger.debug(f"Starting {self.operation}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = datetime.now(timezone.utc).timestamp() - self.start_time
        
        if exc_type:
            log_error(self.logger, exc_val, {
                "operation": self.operation,
                "duration_seconds": duration,
                **self.metadata
            })
        else:
            log_performance(self.logger, self.operation, duration, self.metadata)

# Initialize default logger
default_logger = setup_logger(
    name="hdbank_chatbot",
    level=os.getenv("LOG_LEVEL", "INFO"),
    log_file=os.getenv("LOG_FILE", "logs/api.log"),
    structured=True,
    include_console=True
)
