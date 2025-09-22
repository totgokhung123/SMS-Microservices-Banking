"""
RAG System for HDBank Banking Chatbot
Complete Retrieval-Augmented Generation system với document loading, 
embedding, vector storage và intelligent retrieval.

Hệ thống RAG hoàn chỉnh hỗ trợ:
- Load và xử lý documents từ múltiple formats
- Embedding texts với Vietnamese language support
- FAISS vector database cho fast similarity search
- Intelligent retrieval với reranking và filtering
- Easy-to-use pipeline cho chatbot integration

Author: HDBank AI Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "HDBank AI Team"

import logging
from typing import List, Dict, Any, Optional

# Import main classes
try:
    from .data_loader import DocumentLoader, Document, Chunk, load_hdbank_documents, load_and_chunk_hdbank_documents
    from .embedder import TextEmbedder, create_embedder, embed_texts
    from .vector_store import VectorStore, create_vector_store, load_or_create_vector_store
    from .retriever import RAGRetriever, create_retriever, quick_search
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    RAG_COMPONENTS_AVAILABLE = True
    
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"Một số dependencies chưa được cài đặt: {e}")
    logger.warning("Vui lòng cài đặt: pip install sentence-transformers faiss-cpu PyYAML")
    RAG_COMPONENTS_AVAILABLE = False

# Export main classes và functions
__all__ = [
    # Main classes
    'DocumentLoader',
    'TextEmbedder', 
    'VectorStore',
    'RAGRetriever',
    
    # Data classes
    'Document',
    'Chunk',
    
    # Convenience functions
    'create_embedder',
    'create_vector_store',
    'create_retriever',
    'load_or_create_vector_store',
    'load_hdbank_documents',
    'load_and_chunk_hdbank_documents',
    'embed_texts',
    'quick_search',
    
    # Pipeline functions
    'build_rag_system',
    'create_rag_pipeline',
    'search_documents',
    
    # Utils
    'get_rag_info',
    'check_dependencies'
]

class RAGPipeline:
    """
    Complete RAG Pipeline - main class để sử dụng toàn bộ hệ thống RAG
    """
    
    def __init__(self, config_path: str = "config/rag_config.yaml"):
        """
        Khởi tạo complete RAG pipeline
        
        Args:
            config_path: Đường dẫn đến file config
        """
        if not RAG_COMPONENTS_AVAILABLE:
            raise ImportError("RAG components không available. Vui lòng cài đặt dependencies.")
            
        self.config_path = config_path
        self.retriever = None
        self._initialized = False
        
        logger.info("RAGPipeline được khởi tạo")
    
    def initialize(self, force_rebuild: bool = False) -> bool:
        """
        Initialize RAG system
        
        Args:
            force_rebuild: Có rebuild index không
            
        Returns:
            bool: True nếu thành công
        """
        try:
            logger.info("Đang initialize RAG system...")
            
            # Tạo retriever
            self.retriever = RAGRetriever(self.config_path)
            
            # Build index
            success = self.retriever.build_index(force_rebuild)
            
            if success:
                self._initialized = True
                logger.info("RAG system đã được initialize thành công")
            else:
                logger.error("Không thể initialize RAG system")
            
            return success
            
        except Exception as e:
            logger.error(f"Lỗi khi initialize RAG system: {e}")
            return False
    
    def search(self, query: str, top_k: int = 5, 
              category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search documents
        
        Args:
            query: Query string
            top_k: Số lượng kết quả
            category: Category để filter (optional)
            
        Returns:
            List[Dict]: Kết quả search
        """
        if not self._initialized:
            logger.warning("RAG system chưa được initialize. Đang initialize...")
            if not self.initialize():
                return []
        
        try:
            if category:
                return self.retriever.search_by_category(query, category, top_k)
            else:
                return self.retriever.retrieve(query, top_k)
        except Exception as e:
            logger.error(f"Lỗi khi search: {e}")
            return []
    
    def get_context(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Get context string cho LLM
        
        Args:
            query: Query string
            top_k: Số lượng documents
            
        Returns:
            Dict chứa context và metadata
        """
        if not self._initialized:
            if not self.initialize():
                return {'context': '', 'results': [], 'total_results': 0}
        
        try:
            return self.retriever.retrieve_with_context(query, top_k)
        except Exception as e:
            logger.error(f"Lỗi khi get context: {e}")
            return {'context': '', 'results': [], 'total_results': 0}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get thống kê về RAG system"""
        if not self._initialized:
            return {'initialized': False}
        
        stats = self.retriever.get_retrieval_stats()
        stats['initialized'] = True
        return stats

def build_rag_system(config_path: str = "config/rag_config.yaml", 
                    force_rebuild: bool = False) -> RAGPipeline:
    """
    Build complete RAG system
    
    Args:
        config_path: Đường dẫn config
        force_rebuild: Có rebuild index không
        
    Returns:
        RAGPipeline: Initialized RAG system
    """
    pipeline = RAGPipeline(config_path)
    pipeline.initialize(force_rebuild)
    return pipeline

def create_rag_pipeline(config_path: str = "config/rag_config.yaml") -> RAGPipeline:
    """
    Create RAG pipeline (không initialize ngay)
    
    Args:
        config_path: Đường dẫn config
        
    Returns:
        RAGPipeline: RAG pipeline chưa initialize
    """
    return RAGPipeline(config_path)

def search_documents(query: str, top_k: int = 5, 
                    config_path: str = "config/rag_config.yaml",
                    category: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Convenience function để search documents
    
    Args:
        query: Query string
        top_k: Số lượng kết quả
        config_path: Đường dẫn config
        category: Category filter
        
    Returns:
        List[Dict]: Kết quả search
    """
    pipeline = RAGPipeline(config_path)
    pipeline.initialize()
    return pipeline.search(query, top_k, category)

def get_rag_info() -> Dict[str, Any]:
    """
    Get thông tin về RAG system
    
    Returns:
        Dict chứa system info
    """
    info = {
        'version': __version__,
        'author': __author__,
        'components_available': RAG_COMPONENTS_AVAILABLE,
        'required_dependencies': [
            'sentence-transformers',
            'faiss-cpu',
            'PyYAML',
            'numpy',
            'scikit-learn'
        ]
    }
    
    if RAG_COMPONENTS_AVAILABLE:
        info['status'] = 'Ready'
    else:
        info['status'] = 'Missing dependencies'
    
    return info

def check_dependencies() -> Dict[str, bool]:
    """
    Kiểm tra dependencies
    
    Returns:
        Dict với status của từng dependency
    """
    dependencies = {}
    
    # Check các thư viện chính
    try:
        import sentence_transformers
        dependencies['sentence_transformers'] = True
    except ImportError:
        dependencies['sentence_transformers'] = False
    
    try:
        import faiss
        dependencies['faiss'] = True
    except ImportError:
        dependencies['faiss'] = False
    
    try:
        import yaml
        dependencies['yaml'] = True
    except ImportError:
        dependencies['yaml'] = False
    
    try:
        import numpy
        dependencies['numpy'] = True
    except ImportError:
        dependencies['numpy'] = False
    
    try:
        import sklearn
        dependencies['sklearn'] = True
    except ImportError:
        dependencies['sklearn'] = False
    
    return dependencies

# Print welcome message when imported
if RAG_COMPONENTS_AVAILABLE:
    logger.info("HDBank RAG System loaded successfully!")
    logger.info("Sẵn sàng để xử lý documents và search queries.")
else:
    logger.warning("HDBank RAG System - Missing dependencies!")
    logger.warning("Vui lòng cài đặt: pip install sentence-transformers faiss-cpu PyYAML")

# Example usage trong comments
"""
Example Usage:

# 1. Quick search
from src.rag import search_documents
results = search_documents("tôi muốn mở thẻ tín dụng", top_k=5)

# 2. Use pipeline
from src.rag import build_rag_system
rag = build_rag_system()
results = rag.search("làm thế nào để chuyển khoản")
context = rag.get_context("chuyển khoản qua internet banking")

# 3. Category search
results = rag.search("hạn mức thẻ", category="CARD")

# 4. Get system info
from src.rag import get_rag_info, check_dependencies
print(get_rag_info())
print(check_dependencies())
"""
