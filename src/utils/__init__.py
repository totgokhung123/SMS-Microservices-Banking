"""
Utilities package cho HDBank AI Chatbot
"""

from .logger import (
    setup_logger,
    get_logger,
    log_function_call,
    log_performance,
    log_error,
    log_api_request,
    log_rag_operation,
    LoggedOperation,
    default_logger
)

from .exceptions import (
    ChatbotBaseException,
    RAGException,
    ModelException,
    ConfigurationException,
    ValidationException,
    RateLimitException,
    AuthenticationException,
    ServiceUnavailableException,
    DocumentNotFoundException,
    EmbeddingException,
    VectorStoreException,
    RetrievalException,
    ModelLoadException,
    GenerationException,
    TokenizationException,
    ErrorHandler,
    error_handler,
    handle_exception,
    handle_errors,
    safe_execute,
    ErrorContext,
    validate_input,
    validate_required_fields
)

__all__ = [
    # Logger utilities
    'setup_logger',
    'get_logger',
    'log_function_call',
    'log_performance',
    'log_error',
    'log_api_request',
    'log_rag_operation',
    'LoggedOperation',
    'default_logger',
    
    # Exception classes
    'ChatbotBaseException',
    'RAGException',
    'ModelException',
    'ConfigurationException',
    'ValidationException',
    'RateLimitException',
    'AuthenticationException',
    'ServiceUnavailableException',
    'DocumentNotFoundException',
    'EmbeddingException',
    'VectorStoreException',
    'RetrievalException',
    'ModelLoadException',
    'GenerationException',
    'TokenizationException',
    
    # Error handling utilities
    'ErrorHandler',
    'error_handler',
    'handle_exception',
    'handle_errors',
    'safe_execute',
    'ErrorContext',
    'validate_input',
    'validate_required_fields'
]
