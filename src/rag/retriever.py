"""
Retriever Module for RAG System - HDBank Banking Chatbot
Module chính để thực hiện semantic search và ranking context
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import yaml
import re
from pathlib import Path

# Import local modules
from .data_loader import DocumentLoader, Chunk, Document
from .embedder import TextEmbedder
from .vector_store import VectorStore

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGRetriever:
    """
    Lớp chính thực hiện retrieval cho RAG system
    Kết hợp data loading, embedding và vector search
    """
    
    def __init__(self, config_path: str = "config/rag_config.yaml"):
        """
        Khởi tạo RAGRetriever
        
        Args:
            config_path: Đường dẫn đến file config
        """
        self.config = self._load_config(config_path)
        
        # Retrieval settings
        self.top_k = self.config['retrieval']['top_k']
        self.similarity_threshold = self.config['retrieval']['similarity_threshold']
        self.max_context_length = self.config['retrieval']['max_context_length']
        self.rerank = self.config['retrieval']['rerank']
        
        # Query processing settings
        self.expand_query = self.config['query_processing']['expand_query']
        self.min_query_length = self.config['query_processing']['min_query_length']
        
        # Initialize components
        self.document_loader = DocumentLoader(config_path)
        self.embedder = TextEmbedder(config_path)
        self.vector_store = VectorStore(config_path)
        
        # Cache cho documents và chunks
        self._documents_cache = None
        self._chunks_cache = None
        self._is_indexed = False
        
        logger.info("RAGRetriever đã được khởi tạo")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration từ YAML file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Không thể load config từ {config_path}: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Trả về config mặc định"""
        return {
            'retrieval': {
                'top_k': 5,
                'similarity_threshold': 0.7,
                'max_context_length': 2048,
                'rerank': True
            },
            'query_processing': {
                'expand_query': True,
                'min_query_length': 10
            }
        }
    
    def build_index(self, force_rebuild: bool = False) -> bool:
        """
        Build vector index từ documents
        
        Args:
            force_rebuild: Có rebuild nếu index đã tồn tại không
            
        Returns:
            bool: True nếu build thành công
        """
        try:
            # Kiểm tra xem đã có index chưa
            if not force_rebuild and self.vector_store.load_index():
                logger.info("Đã load index có sẵn")
                self._is_indexed = True
                return True
            
            logger.info("Đang build vector index...")
            
            # Load và chunk documents
            chunks = self.document_loader.load_and_chunk()
            if not chunks:
                logger.error("Không có chunks để build index")
                return False
            
            self._chunks_cache = chunks
            logger.info(f"Đã load {len(chunks)} chunks")
            
            # Tạo embeddings
            texts = [chunk.content for chunk in chunks]
            logger.info("Đang tạo embeddings...")
            embeddings = self.embedder.encode_texts(texts)
            
            if embeddings.size == 0:
                logger.error("Không thể tạo embeddings")
                return False
            
            # Tạo metadata cho vector store
            metadata = []
            chunk_ids = []
            for chunk in chunks:
                meta = {
                    'chunk_id': chunk.chunk_id,
                    'doc_id': chunk.doc_id,
                    'content': chunk.content,
                    'start_pos': chunk.start_pos,
                    'end_pos': chunk.end_pos,
                    'file_name': chunk.metadata.get('file_name', ''),
                    'file_path': chunk.metadata.get('file_path', ''),
                    'chunk_index': chunk.metadata.get('chunk_index', 0)
                }
                metadata.append(meta)
                chunk_ids.append(chunk.chunk_id)
            
            # Thêm vào vector store
            self.vector_store.add_vectors(embeddings, metadata, chunk_ids)
            
            # Lưu index
            self.vector_store.save_index()
            
            self._is_indexed = True
            logger.info(f"Đã build index thành công với {len(chunks)} chunks")
            
            return True
            
        except Exception as e:
            logger.error(f"Lỗi khi build index: {e}")
            return False
    
    def retrieve(self, query: str, top_k: Optional[int] = None,
                similarity_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents cho query
        
        Args:
            query: Query string
            top_k: Số lượng kết quả (override config nếu có)
            similarity_threshold: Ngưỡng similarity (override config nếu có)
            
        Returns:
            List[Dict]: Danh sách retrieved documents
        """
        if not self._is_indexed:
            logger.warning("Index chưa được build. Đang build...")
            if not self.build_index():
                logger.error("Không thể build index")
                return []
        
        # Validate query
        if not self._validate_query(query):
            return []
        
        # Process query
        processed_query = self._process_query(query)
        
        # Set parameters
        k = top_k if top_k is not None else self.top_k
        threshold = similarity_threshold if similarity_threshold is not None else self.similarity_threshold
        
        try:
            # Tạo query embedding
            query_embedding = self.embedder.encode_single_text(processed_query)
            
            if query_embedding.size == 0:
                logger.error("Không thể tạo query embedding")
                return []
            
            # Search trong vector store
            results = self.vector_store.search(
                query_embedding, 
                k=k * 2,  # Lấy nhiều hơn để có thể rerank
                similarity_threshold=threshold
            )
            
            if not results:
                logger.info(f"Không tìm thấy kết quả cho query: {query}")
                return []
            
            # Rerank nếu được enable
            if self.rerank and len(results) > 1:
                results = self._rerank_results(query, results)
            
            # Limit kết quả
            results = results[:k]
            
            logger.info(f"Retrieved {len(results)} documents cho query: {query[:50]}...")
            
            return results
            
        except Exception as e:
            logger.error(f"Lỗi khi retrieve: {e}")
            return []
    
    def retrieve_with_context(self, query: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        Retrieve documents và tạo context string để sử dụng với LLM
        
        Args:
            query: Query string
            top_k: Số lượng documents
            
        Returns:
            Dict chứa retrieved results và context string
        """
        results = self.retrieve(query, top_k)
        
        if not results:
            return {
                'results': [],
                'context': '',
                'total_results': 0,
                'context_length': 0
            }
        
        # Tạo context string
        context_parts = []
        total_length = 0
        
        for i, result in enumerate(results):
            content = result.get('content', '')
            file_name = result.get('file_name', 'Unknown')
            similarity = result.get('similarity_score', 0.0)
            
            # Format context piece
            context_piece = f"[Document {i+1}] (Source: {file_name}, Similarity: {similarity:.3f})\n{content}\n"
            
            # Kiểm tra độ dài
            if total_length + len(context_piece) > self.max_context_length:
                break
            
            context_parts.append(context_piece)
            total_length += len(context_piece)
        
        context = "\n".join(context_parts)
        
        return {
            'results': results,
            'context': context,
            'total_results': len(results),
            'context_length': len(context),
            'used_results': len(context_parts)
        }
    
    def _validate_query(self, query: str) -> bool:
        """
        Validate query
        
        Args:
            query: Query string
            
        Returns:
            bool: True nếu query hợp lệ
        """
        if not query or not query.strip():
            logger.warning("Query rỗng")
            return False
        
        if len(query.strip()) < self.min_query_length:
            logger.warning(f"Query quá ngắn (< {self.min_query_length} ký tự)")
            return False
        
        return True
    
    def _process_query(self, query: str) -> str:
        """
        Process và normalize query
        
        Args:
            query: Raw query
            
        Returns:
            str: Processed query
        """
        # Basic cleaning
        processed = query.strip()
        
        # Normalize whitespace
        processed = re.sub(r'\s+', ' ', processed)
        
        # Expand query nếu được enable
        if self.expand_query:
            processed = self._expand_query(processed)
        
        return processed
    
    def _expand_query(self, query: str) -> str:
        """
        Expand query với banking domain terms
        
        Args:
            query: Original query
            
        Returns:
            str: Expanded query
        """
        # Banking domain expansions
        expansions = {
            'thẻ': 'thẻ tín dụng thẻ ghi nợ card',
            'vay': 'vay vốn khoản vay loan',
            'gửi tiền': 'gửi tiền tiết kiệm deposit',
            'chuyển khoản': 'chuyển khoản transfer',
            'atm': 'atm rút tiền máy rút tiền',
            'internet banking': 'internet banking online banking ebanking',
            'mobile banking': 'mobile banking ứng dụng ngân hàng'
        }
        
        query_lower = query.lower()
        expanded_terms = []
        
        for term, expansion in expansions.items():
            if term in query_lower:
                expanded_terms.append(expansion)
        
        if expanded_terms:
            return query + " " + " ".join(expanded_terms)
        
        return query
    
    def _rerank_results(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rerank kết quả dựa trên additional criteria
        
        Args:
            query: Original query
            results: Danh sách kết quả cần rerank
            
        Returns:
            List[Dict]: Kết quả đã được rerank
        """
        # Simple reranking based on:
        # 1. Similarity score (đã có)
        # 2. Content length (prefer reasonable length)
        # 3. Source file priority
        
        # File priority mapping
        file_priorities = {
            'hdbank_qa.txt': 1.0,
            'hdbank_qna_articles': 0.9,
            'terms': 0.8,
            'default': 0.7
        }
        
        for result in results:
            file_name = result.get('file_name', '').lower()
            content_length = len(result.get('content', ''))
            similarity = result.get('similarity_score', 0.0)
            
            # File priority score
            file_score = file_priorities.get('default', 0.7)
            for pattern, score in file_priorities.items():
                if pattern in file_name:
                    file_score = score
                    break
            
            # Content length score (prefer 100-1000 chars)
            if 100 <= content_length <= 1000:
                length_score = 1.0
            elif content_length < 100:
                length_score = 0.8
            else:
                length_score = 0.9
            
            # Combined score
            combined_score = similarity * 0.7 + file_score * 0.2 + length_score * 0.1
            result['combined_score'] = combined_score
        
        # Sort by combined score
        results.sort(key=lambda x: x.get('combined_score', 0), reverse=True)
        
        return results
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """
        Lấy thống kê về retrieval system
        
        Returns:
            Dict chứa thống kê
        """
        stats = {
            'is_indexed': self._is_indexed,
            'config': {
                'top_k': self.top_k,
                'similarity_threshold': self.similarity_threshold,
                'max_context_length': self.max_context_length,
                'rerank': self.rerank
            }
        }
        
        # Vector store stats
        if self.vector_store:
            stats['vector_store'] = self.vector_store.get_stats()
        
        # Embedder stats
        if self.embedder:
            stats['embedder'] = self.embedder.get_embedding_info()
        
        # Document stats
        if self._chunks_cache:
            stats['chunks_count'] = len(self._chunks_cache)
            stats['chunks_stats'] = self.document_loader.get_chunk_statistics(self._chunks_cache)
        
        return stats
    
    def search_by_category(self, query: str, category: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Tìm kiếm theo category cụ thể
        
        Args:
            query: Query string
            category: Category để filter (ví dụ: "CARD", "LOAN", "ACCOUNT")
            top_k: Số lượng kết quả
            
        Returns:
            List[Dict]: Kết quả filtered theo category
        """
        # Retrieve tất cả results
        all_results = self.retrieve(query, top_k * 3)  # Lấy nhiều hơn để filter
        
        # Filter theo category (dựa trên file name hoặc content)
        category_keywords = {
            'CARD': ['thẻ', 'card', 'debit', 'credit'],
            'LOAN': ['vay', 'loan', 'khoản vay'],
            'ACCOUNT': ['tài khoản', 'account', 'mở tài khoản'],
            'TRANSFER': ['chuyển khoản', 'transfer'],
            'ATM': ['atm', 'rút tiền'],
            'EBANKING': ['internet banking', 'online banking', 'ebanking']
        }
        
        keywords = category_keywords.get(category.upper(), [])
        if not keywords:
            return all_results[:top_k]
        
        # Filter results
        filtered_results = []
        for result in all_results:
            content = result.get('content', '').lower()
            file_name = result.get('file_name', '').lower()
            
            # Check if any keyword appears in content or filename
            if any(keyword in content or keyword in file_name for keyword in keywords):
                filtered_results.append(result)
        
        return filtered_results[:top_k]

# Convenience functions
def create_retriever(config_path: str = "config/rag_config.yaml") -> RAGRetriever:
    """Tạo retriever instance"""
    return RAGRetriever(config_path)

def quick_search(query: str, top_k: int = 5, 
                config_path: str = "config/rag_config.yaml") -> List[Dict[str, Any]]:
    """Convenience function để search nhanh"""
    retriever = RAGRetriever(config_path)
    retriever.build_index()  # Build nếu chưa có
    return retriever.retrieve(query, top_k)
