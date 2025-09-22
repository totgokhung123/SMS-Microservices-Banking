"""
Chat Router - HDBank Chatbot API
Main chat endpoints cho RAG-powered conversations
"""

import time
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
import json

from api.dependencies import (
    ChatbotServiceDep, 
    CurrentUserDep, 
    ConversationIdDep, 
    RequestMetadataDep,
    check_rate_limit
)
from api.models import (
    ChatRequest, 
    ChatResponse, 
    StreamChatResponse,
    ConversationHistory,
    SearchRequest,
    SearchResponse
)
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    chatbot_service: ChatbotServiceDep,
    background_tasks: BackgroundTasks = None
):
    """
    Main chat endpoint với RAG integration
    Xử lý câu hỏi của user và trả về câu trả lời từ AI
    """
    start_time = time.time()
    
    try:
        if not chatbot_service:
            raise HTTPException(
                status_code=503,
                detail="RAG service không available"
            )
        
        # Validate input
        if not request.message or len(request.message.strip()) < 3:
            raise HTTPException(
                status_code=400,
                detail="Message quá ngắn. Vui lòng nhập ít nhất 3 ký tự."
            )
        
        # Log request
        logger.info(f"Chat request: {request.message[:100]}...")
        
        # Generate conversation ID nếu không có
        conversation_id = request.conversation_id or str(uuid.uuid4())
        
        # Get context từ RAG
        context_result = await chatbot_service.get_context(
            request.message,
            top_k=request.context_limit or 5
        )
        
        # Generate response
        response = await chatbot_service.generate_response(
            message=request.message,
            context=context_result['context'],
            conversation_id=conversation_id,
            user_id=request.user_id
        )
        
        # Calculate metrics
        response_time = time.time() - start_time
        
        # Prepare response
        chat_response = ChatResponse(
            message=response['answer'],
            conversation_id=conversation_id,
            timestamp=datetime.now(timezone.utc),
            metadata={
                "response_time_ms": round(response_time * 1000, 2),
                "context_used": len(context_result['results']),
                "context_length": context_result['context_length'],
                "model_confidence": response.get('confidence', 0.8),
                "sources": [
                    {
                        "file_name": result.get('file_name', 'Unknown'),
                        "similarity": result.get('similarity_score', 0.0)
                    }
                    for result in context_result['results'][:3]  # Top 3 sources
                ]
            }
        )
        
        # Background task để log conversation
        if background_tasks:
            background_tasks.add_task(
                log_conversation,
                conversation_id,
                request.message,
                response['answer'],
                response_time
            )
        
        logger.info(f"Chat response generated in {response_time:.2f}s")
        return chat_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi xử lý chat: {str(e)}"
        )

@router.post("/search", response_model=SearchResponse)
async def search_documents(
    request: SearchRequest,
    chatbot_service: ChatbotServiceDep
):
    """
    Search documents trong knowledge base
    Trả về relevant documents cho query
    """
    try:
        if not chatbot_service:
            raise HTTPException(
                status_code=503,
                detail="RAG service không available"
            )
        
        # Search documents
        results = await chatbot_service.search_documents(
            query=request.query,
            top_k=request.top_k or 10,
            category=request.category,
            similarity_threshold=request.similarity_threshold
        )
        
        return SearchResponse(
            query=request.query,
            results=results,
            total_found=len(results),
            timestamp=datetime.now(timezone.utc)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi tìm kiếm: {str(e)}"
        )

@router.post("/stream")
async def stream_chat(
    request: ChatRequest,
    chatbot_service: ChatbotServiceDep
):
    """
    Streaming chat response
    Trả về response theo chunks để có trải nghiệm real-time
    """
    try:
        if not chatbot_service:
            raise HTTPException(
                status_code=503,
                detail="RAG service không available"
            )
        
        async def generate_stream():
            conversation_id = request.conversation_id or str(uuid.uuid4())
            
            # Send initial chunk
            initial_chunk = StreamChatResponse(
                type="start",
                conversation_id=conversation_id,
                timestamp=datetime.now(timezone.utc)
            )
            yield f"data: {initial_chunk.model_dump_json()}\n\n"
            
            try:
                # Get context
                context_result = await chatbot_service.get_context(
                    request.message,
                    top_k=request.context_limit or 5
                )
                
                # Send context chunk
                context_chunk = StreamChatResponse(
                    type="context",
                    conversation_id=conversation_id,
                    data={
                        "sources_found": len(context_result['results']),
                        "context_length": context_result['context_length']
                    }
                )
                yield f"data: {context_chunk.model_dump_json()}\n\n"
                
                # Generate response (simulated streaming)
                response = await chatbot_service.generate_response(
                    message=request.message,
                    context=context_result['context'],
                    conversation_id=conversation_id,
                    user_id=request.user_id
                )
                
                # Send response in chunks
                answer = response['answer']
                chunk_size = 20  # words per chunk
                words = answer.split()
                
                for i in range(0, len(words), chunk_size):
                    chunk_words = words[i:i + chunk_size]
                    chunk_text = ' '.join(chunk_words)
                    
                    text_chunk = StreamChatResponse(
                        type="text",
                        conversation_id=conversation_id,
                        data={"text": chunk_text}
                    )
                    yield f"data: {text_chunk.model_dump_json()}\n\n"
                    
                    # Small delay để simulate typing
                    await asyncio.sleep(0.1)
                
                # Send final chunk
                final_chunk = StreamChatResponse(
                    type="end",
                    conversation_id=conversation_id,
                    data={
                        "complete": True,
                        "full_response": answer,
                        "metadata": {
                            "sources": len(context_result['results']),
                            "confidence": response.get('confidence', 0.8)
                        }
                    }
                )
                yield f"data: {final_chunk.model_dump_json()}\n\n"
                
            except Exception as e:
                error_chunk = StreamChatResponse(
                    type="error",
                    conversation_id=conversation_id,
                    data={"error": str(e)}
                )
                yield f"data: {error_chunk.model_dump_json()}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream"
            }
        )
        
    except Exception as e:
        logger.error(f"Stream chat error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi streaming chat: {str(e)}"
        )

@router.get("/conversations/{conversation_id}", response_model=ConversationHistory)
async def get_conversation(
    conversation_id: str,
    chatbot_service: ChatbotServiceDep
):
    """
    Get conversation history
    """
    try:
        # This would typically come from a database
        # For now, return a placeholder
        return ConversationHistory(
            conversation_id=conversation_id,
            messages=[],
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )
        
    except Exception as e:
        logger.error(f"Get conversation error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Lỗi lấy lịch sử chat"
        )

@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """
    Delete conversation history
    """
    try:
        # Implement conversation deletion logic
        return {"message": f"Conversation {conversation_id} deleted"}
        
    except Exception as e:
        logger.error(f"Delete conversation error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Lỗi xóa lịch sử chat"
        )

# Background task function
async def log_conversation(
    conversation_id: str,
    user_message: str,
    bot_response: str,
    response_time: float
):
    """
    Background task để log conversation
    """
    try:
        # Log to file hoặc database
        logger.info(
            f"Conversation logged: {conversation_id} | "
            f"Response time: {response_time:.2f}s | "
            f"User: {user_message[:50]}... | "
            f"Bot: {bot_response[:50]}..."
        )
    except Exception as e:
        logger.error(f"Failed to log conversation: {e}")

# Import asyncio for streaming
import asyncio
