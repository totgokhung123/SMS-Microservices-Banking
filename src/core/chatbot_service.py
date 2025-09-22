"""
Chatbot Service - Core business logic cho HDBank AI Chatbot
Integrate RAG system với conversation management
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
import uuid
import json

# Import RAG system
try:
    from ..rag import RAGPipeline, get_rag_info
except ImportError:
    # Fallback nếu RAG chưa available
    class RAGPipeline:
        def __init__(self, *args, **kwargs):
            pass
        
        def initialize(self, *args, **kwargs):
            return False
        
        def search(self, *args, **kwargs):
            return []
        
        def get_context(self, *args, **kwargs):
            return {"context": "", "results": []}

# Setup logging
logger = logging.getLogger(__name__)

class ChatbotService:
    """
    Main service class cho HDBank AI Chatbot
    Handles conversation logic và RAG integration
    """
    
    def __init__(self, config_path: str = "config/rag_config.yaml"):
        """
        Initialize chatbot service
        
        Args:
            config_path: Path to RAG configuration
        """
        self.config_path = config_path
        self.rag_pipeline = None
        self.initialized = False
        self.conversation_cache = {}  # In-memory conversation storage
        self.system_prompt = self._get_system_prompt()
        
        logger.info("ChatbotService initialized")
    
    async def initialize(self) -> bool:
        """
        Initialize RAG pipeline và các components
        
        Returns:
            bool: True nếu initialization thành công
        """
        try:
            logger.info("Initializing RAG pipeline...")
            
            # Initialize RAG system
            self.rag_pipeline = RAGPipeline(self.config_path)
            success = self.rag_pipeline.initialize(force_rebuild=False)
            
            if success:
                self.initialized = True
                logger.info("✅ ChatbotService initialized successfully")
                return True
            else:
                logger.error("❌ Failed to initialize RAG pipeline")
                return False
                
        except Exception as e:
            logger.error(f"❌ ChatbotService initialization error: {e}", exc_info=True)
            return False
    
    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up ChatbotService...")
        self.conversation_cache.clear()
        self.initialized = False
    
    async def get_context(self, message: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Get relevant context cho message từ RAG system
        
        Args:
            message: User message
            top_k: Number of relevant documents
            
        Returns:
            Dict containing context và metadata
        """
        if not self.initialized or not self.rag_pipeline:
            logger.warning("RAG pipeline not initialized")
            return {"context": "", "results": [], "total_results": 0}
        
        try:
            return self.rag_pipeline.get_context(message, top_k)
        except Exception as e:
            logger.error(f"Error getting context: {e}")
            return {"context": "", "results": [], "total_results": 0}
    
    async def search_documents(self, query: str, top_k: int = 10, 
                             category: Optional[str] = None,
                             similarity_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Search documents trong knowledge base
        
        Args:
            query: Search query
            top_k: Number of results
            category: Document category filter
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of search results
        """
        if not self.initialized or not self.rag_pipeline:
            logger.warning("RAG pipeline not initialized")
            return []
        
        try:
            return self.rag_pipeline.search(query, top_k, category)
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    async def generate_response(self, message: str, context: str,
                              conversation_id: str,
                              user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate response từ AI model với context
        
        Args:
            message: User message
            context: Relevant context từ RAG
            conversation_id: Conversation ID
            user_id: User ID
            
        Returns:
            Dict containing generated response và metadata
        """
        try:
            # Get conversation history
            conversation = self.get_conversation(conversation_id)
            
            # Build prompt với context và history
            prompt = self._build_prompt(message, context, conversation)
            
            # Generate response (simplified - trong thực tế sẽ call LLM)
            response = await self._call_llm(prompt, message, context)
            
            # Update conversation history
            self._update_conversation(conversation_id, message, response['answer'], user_id)
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "answer": "Xin lỗi, tôi gặp sự cố khi xử lý câu hỏi của bạn. Vui lòng thử lại.",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def get_conversation(self, conversation_id: str) -> List[Dict[str, Any]]:
        """
        Get conversation history
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            List of conversation messages
        """
        return self.conversation_cache.get(conversation_id, [])
    
    def _update_conversation(self, conversation_id: str, user_message: str, 
                           bot_response: str, user_id: Optional[str] = None):
        """Update conversation history"""
        if conversation_id not in self.conversation_cache:
            self.conversation_cache[conversation_id] = []
        
        # Add user message
        self.conversation_cache[conversation_id].append({
            "id": str(uuid.uuid4()),
            "type": "user",
            "content": user_message,
            "timestamp": datetime.now(timezone.utc),
            "user_id": user_id
        })
        
        # Add bot response
        self.conversation_cache[conversation_id].append({
            "id": str(uuid.uuid4()),
            "type": "assistant", 
            "content": bot_response,
            "timestamp": datetime.now(timezone.utc)
        })
        
        # Keep only last 20 messages để tránh memory overflow
        if len(self.conversation_cache[conversation_id]) > 20:
            self.conversation_cache[conversation_id] = self.conversation_cache[conversation_id][-20:]
    
    def _build_prompt(self, message: str, context: str, 
                     conversation: List[Dict[str, Any]]) -> str:
        """
        Build prompt cho LLM với context và conversation history
        
        Args:
            message: Current user message
            context: Relevant context từ RAG
            conversation: Conversation history
            
        Returns:
            Formatted prompt string
        """
        # System prompt
        prompt_parts = [self.system_prompt]
        
        # Add context nếu có
        if context and context.strip():
            prompt_parts.append(f"\n### Thông tin tham khảo:\n{context}")
        
        # Add conversation history (last 5 exchanges)
        if conversation:
            recent_conversation = conversation[-10:]  # Last 5 user-bot exchanges
            prompt_parts.append("\n### Lịch sử trò chuyện:")
            
            for msg in recent_conversation:
                role = "Khách hàng" if msg["type"] == "user" else "Tư vấn viên"
                prompt_parts.append(f"{role}: {msg['content']}")
        
        # Add current message
        prompt_parts.append(f"\n### Câu hỏi hiện tại:\nKhách hàng: {message}")
        prompt_parts.append("\nTư vấn viên:")
        
        return "\n".join(prompt_parts)
    
    async def _call_llm(self, prompt: str, user_message: str, context: str) -> Dict[str, Any]:
        """
        Call LLM để generate response
        Hiện tại dùng rule-based response, sau này sẽ integrate với Qwen model
        
        Args:
            prompt: Formatted prompt
            user_message: Original user message
            context: Context từ RAG
            
        Returns:
            Dict with generated response
        """
        # Simulate LLM call với small delay
        await asyncio.sleep(0.1)
        
        # Rule-based responses based on keywords
        message_lower = user_message.lower()
        
        # Banking keywords và responses
        responses = {
            "thẻ tín dụng": "HDBank cung cấp nhiều loại thẻ tín dụng với ưu đãi hấp dẫn. Bạn có thể mở thẻ tín dụng tại các chi nhánh hoặc qua ứng dụng HDBank Mobile.",
            
            "chuyển khoản": "HDBank hỗ trợ chuyển khoản 24/7 qua Internet Banking, Mobile Banking và tại ATM. Phí chuyển khoản liên ngân hàng từ 1,100 VNĐ.",
            
            "vay vốn": "HDBank có các gói vay ưu đãi với lãi suất từ 6.99%/năm. Bạn có thể vay thế chấp, vay tín chấp hoặc vay mua nhà.",
            
            "tiết kiệm": "Gửi tiết kiệm HDBank với lãi suất hấp dẫn lên đến 7.2%/năm. Có thể gửi online qua ứng dụng HDBank Mobile.",
            
            "atm": "HDBank có hơn 1,000 ATM trên toàn quốc. Miễn phí rút tiền tại ATM HDBank, phí 3,300 VNĐ/lần tại ATM liên ngân hàng.",
            
            "internet banking": "HDBank Internet Banking cho phép bạn thực hiện mọi giao dịch 24/7. Đăng ký miễn phí tại chi nhánh hoặc qua ứng dụng."
        }
        
        # Find matching response
        response_text = "Cảm ơn bạn đã quan tâm đến dịch vụ HDBank. "
        
        for keyword, response in responses.items():
            if keyword in message_lower:
                response_text = response
                break
        else:
            # Fallback response nếu không match keyword
            if context and context.strip():
                response_text += "Dựa trên thông tin tôi tìm được, tôi khuyên bạn nên liên hệ hotline 1900 6060 để được tư vấn chi tiết hơn."
            else:
                response_text += "Vui lòng liên hệ hotline 1900 6060 hoặc đến chi nhánh gần nhất để được hỗ trợ tốt nhất."
        
        # Add context-based information
        if context and len(context) > 100:
            response_text += "\n\nThông tin này được tham khảo từ tài liệu chính thức của HDBank."
        
        return {
            "answer": response_text,
            "confidence": 0.85,
            "model": "hdbank-rule-based-v1",
            "tokens_used": len(prompt.split()),
            "response_time": 0.1
        }
    
    def _get_system_prompt(self) -> str:
        """Get system prompt cho chatbot"""
        return """Bạn là trợ lý AI tư vấn tài chính chuyên nghiệp của HDBank. 

Nhiệm vụ của bạn:
- Tư vấn các sản phẩm và dịch vụ ngân hàng của HDBank
- Trả lời các câu hỏi về thẻ tín dụng, vay vốn, tiết kiệm, chuyển khoản
- Hướng dẫn sử dụng dịch vụ Internet Banking, Mobile Banking
- Cung cấp thông tin chính xác, hữu ích và thân thiện

Nguyên tắc:
- Luôn lịch sự và chuyên nghiệp
- Cung cấp thông tin chính xác từ tài liệu chính thức
- Hướng dẫn khách hàng đến đúng kênh hỗ trợ khi cần
- Không đưa ra lời khuyên tài chính cá nhân
- Luôn đề xuất liên hệ hotline 1900 6060 cho các vấn đề phức tạp"""
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get service statistics
        
        Returns:
            Dict containing service stats
        """
        stats = {
            "initialized": self.initialized,
            "total_conversations": len(self.conversation_cache),
            "service_uptime": time.time(),  # Simplified
        }
        
        # Add RAG stats nếu available
        if self.rag_pipeline:
            try:
                rag_stats = self.rag_pipeline.get_stats()
                stats.update(rag_stats)
            except Exception as e:
                logger.warning(f"Could not get RAG stats: {e}")
        
        return stats

# Singleton instance
_chatbot_service_instance = None

async def get_chatbot_service() -> ChatbotService:
    """
    Get singleton chatbot service instance
    """
    global _chatbot_service_instance
    
    if _chatbot_service_instance is None:
        _chatbot_service_instance = ChatbotService()
        await _chatbot_service_instance.initialize()
    
    return _chatbot_service_instance
