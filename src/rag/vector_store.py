"""
Vector Store Module for RAG System - HDBank Banking Chatbot
Module để quản lý FAISS vector database
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import yaml
from pathlib import Path
import pickle
from dataclasses import asdict

try:
    import faiss
except ImportError:
    faiss = None
    logging.warning("FAISS không được cài đặt. Vui lòng cài đặt: pip install faiss-cpu hoặc faiss-gpu")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStore:
    """
    Lớp chính để quản lý FAISS vector database
    Hỗ trợ lưu trữ, tìm kiếm và quản lý embeddings
    """
    
    def __init__(self, config_path: str = "config/rag_config.yaml"):
        """
        Khởi tạo VectorStore
        
        Args:
            config_path: Đường dẫn đến file config
        """
        self.config = self._load_config(config_path)
        self.index_type = self.config['vector_store']['index_type']
        self.save_path = self.config['vector_store']['save_path']
        self.metadata_path = self.config['vector_store']['metadata_path']
        self.dimension = self.config['embedding']['dimension']
        
        # Khởi tạo
        self.index = None
        self.metadata = []  # Lưu metadata cho mỗi vector
        self.id_to_index = {}  # Mapping từ ID đến index trong FAISS
        self.index_to_id = {}  # Mapping ngược lại
        self.next_id = 0
        
        # Tạo thư mục nếu chưa có
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)
        
        # Khởi tạo FAISS index
        self._initialize_index()
        
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
            'vector_store': {
                'index_type': 'IndexFlatIP',
                'save_path': 'data/vector_db/faiss_index.bin',
                'metadata_path': 'data/vector_db/metadata.json'
            },
            'embedding': {
                'dimension': 768
            }
        }
    
    def _initialize_index(self):
        """Khởi tạo FAISS index"""
        if not faiss:
            raise ImportError("FAISS chưa được cài đặt")
            
        try:
            # Tạo index dựa trên type
            if self.index_type == "IndexFlatIP":
                # Inner Product (cho cosine similarity với normalized vectors)
                self.index = faiss.IndexFlatIP(self.dimension)
            elif self.index_type == "IndexFlatL2":
                # L2 distance
                self.index = faiss.IndexFlatL2(self.dimension)
            elif self.index_type == "IndexIVFFlat":
                # IVF index cho large dataset
                quantizer = faiss.IndexFlatIP(self.dimension)
                self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
            else:
                # Default fallback
                logger.warning(f"Unknown index type {self.index_type}, using IndexFlatIP")
                self.index = faiss.IndexFlatIP(self.dimension)
            
            logger.info(f"Đã khởi tạo FAISS index: {self.index_type}, dimension: {self.dimension}")
            
        except Exception as e:
            logger.error(f"Lỗi khi khởi tạo FAISS index: {e}")
            raise
    
    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict[str, Any]], 
                   ids: Optional[List[str]] = None) -> List[int]:
        """
        Thêm vectors vào index
        
        Args:
            vectors: Numpy array chứa vectors
            metadata: Metadata tương ứng cho mỗi vector
            ids: IDs tùy chọn cho vectors
            
        Returns:
            List[int]: Danh sách internal indices
        """
        if vectors.shape[0] != len(metadata):
            raise ValueError("Số lượng vectors và metadata phải bằng nhau")
        
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Vector dimension ({vectors.shape[1]}) không khớp với index dimension ({self.dimension})")
        
        # Normalize vectors nếu sử dụng cosine similarity
        if self.index_type == "IndexFlatIP":
            vectors = self._normalize_vectors(vectors)
        
        # Train index nếu cần (cho IVF index)
        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            if vectors.shape[0] >= 100:  # Cần ít nhất 100 vectors để train
                logger.info("Training IVF index...")
                self.index.train(vectors)
            else:
                logger.warning("Không đủ vectors để train IVF index")
        
        # Thêm vectors vào index
        start_idx = self.index.ntotal
        self.index.add(vectors)
        
        # Cập nhật metadata và mappings
        internal_indices = []
        for i, meta in enumerate(metadata):
            internal_idx = start_idx + i
            internal_indices.append(internal_idx)
            
            # Tạo ID nếu không có
            if ids and i < len(ids):
                vector_id = ids[i]
            else:
                vector_id = f"vec_{self.next_id}"
                self.next_id += 1
            
            # Cập nhật mappings
            self.id_to_index[vector_id] = internal_idx
            self.index_to_id[internal_idx] = vector_id
            
            # Thêm metadata
            meta_with_id = meta.copy()
            meta_with_id['vector_id'] = vector_id
            meta_with_id['internal_index'] = internal_idx
            self.metadata.append(meta_with_id)
        
        logger.info(f"Đã thêm {len(vectors)} vectors vào index. Tổng: {self.index.ntotal}")
        return internal_indices
    
    def search(self, query_vector: np.ndarray, k: int = 5, 
              similarity_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Tìm kiếm vectors tương tự nhất
        
        Args:
            query_vector: Vector query
            k: Số lượng kết quả trả về
            similarity_threshold: Ngưỡng similarity tối thiểu
            
        Returns:
            List[Dict]: Danh sách kết quả tìm kiếm
        """
        if self.index.ntotal == 0:
            logger.warning("Index rỗng")
            return []
        
        # Đảm bảo query_vector có shape đúng
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        
        if query_vector.shape[1] != self.dimension:
            raise ValueError(f"Query vector dimension ({query_vector.shape[1]}) không khớp với index dimension ({self.dimension})")
        
        # Normalize query vector nếu cần
        if self.index_type == "IndexFlatIP":
            query_vector = self._normalize_vectors(query_vector)
        
        # Tìm kiếm
        k = min(k, self.index.ntotal)  # Không thể tìm nhiều hơn số vectors có sẵn
        scores, indices = self.index.search(query_vector, k)
        
        # Xử lý kết quả
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1:  # FAISS trả về -1 nếu không tìm thấy
                continue
                
            # Áp dụng similarity threshold
            if similarity_threshold is not None and score < similarity_threshold:
                continue
            
            # Lấy metadata
            if idx < len(self.metadata):
                metadata = self.metadata[idx].copy()
                metadata['similarity_score'] = float(score)
                metadata['rank'] = i
                results.append(metadata)
        
        return results
    
    def search_by_ids(self, query_vector: np.ndarray, 
                     filter_ids: List[str], k: int = 5) -> List[Dict[str, Any]]:
        """
        Tìm kiếm với filter theo IDs
        
        Args:
            query_vector: Vector query
            filter_ids: Danh sách IDs để filter
            k: Số lượng kết quả
            
        Returns:
            List[Dict]: Kết quả tìm kiếm
        """
        # Tìm kiếm tất cả
        all_results = self.search(query_vector, k=self.index.ntotal)
        
        # Filter theo IDs
        filtered_results = [
            result for result in all_results 
            if result.get('vector_id') in filter_ids
        ]
        
        # Trả về top k
        return filtered_results[:k]
    
    def get_vector_by_id(self, vector_id: str) -> Optional[np.ndarray]:
        """
        Lấy vector theo ID
        
        Args:
            vector_id: ID của vector
            
        Returns:
            numpy.ndarray hoặc None nếu không tìm thấy
        """
        if vector_id not in self.id_to_index:
            return None
        
        internal_idx = self.id_to_index[vector_id]
        
        # FAISS không có method để lấy vector theo index trực tiếp
        # Cần implement riêng hoặc lưu vectors riêng
        logger.warning("get_vector_by_id chưa được implement đầy đủ")
        return None
    
    def delete_vectors(self, vector_ids: List[str]):
        """
        Xóa vectors theo IDs
        Note: FAISS không hỗ trợ xóa trực tiếp, cần rebuild index
        
        Args:
            vector_ids: Danh sách IDs cần xóa
        """
        logger.warning("FAISS không hỗ trợ xóa vectors trực tiếp. Cần rebuild index để xóa.")
        # TODO: Implement rebuild index without deleted vectors
    
    def save_index(self, save_path: Optional[str] = None, 
                  metadata_path: Optional[str] = None):
        """
        Lưu index và metadata
        
        Args:
            save_path: Đường dẫn lưu index (optional)
            metadata_path: Đường dẫn lưu metadata (optional)
        """
        if save_path is None:
            save_path = self.save_path
        if metadata_path is None:
            metadata_path = self.metadata_path
        
        try:
            # Lưu FAISS index
            faiss.write_index(self.index, save_path)
            logger.info(f"Đã lưu FAISS index tại: {save_path}")
            
            # Lưu metadata
            metadata_to_save = {
                'metadata': self.metadata,
                'id_to_index': self.id_to_index,
                'index_to_id': {str(k): v for k, v in self.index_to_id.items()},  # Convert int keys to string for JSON
                'next_id': self.next_id,
                'dimension': self.dimension,
                'index_type': self.index_type
            }
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata_to_save, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Đã lưu metadata tại: {metadata_path}")
            
        except Exception as e:
            logger.error(f"Lỗi khi lưu index: {e}")
            raise
    
    def load_index(self, save_path: Optional[str] = None, 
                  metadata_path: Optional[str] = None) -> bool:
        """
        Load index và metadata từ file
        
        Args:
            save_path: Đường dẫn file index
            metadata_path: Đường dẫn file metadata
            
        Returns:
            bool: True nếu load thành công
        """
        if save_path is None:
            save_path = self.save_path
        if metadata_path is None:
            metadata_path = self.metadata_path
        
        try:
            # Load FAISS index
            if os.path.exists(save_path):
                self.index = faiss.read_index(save_path)
                logger.info(f"Đã load FAISS index từ: {save_path}")
            else:
                logger.warning(f"File index không tồn tại: {save_path}")
                return False
            
            # Load metadata
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata_data = json.load(f)
                
                self.metadata = metadata_data.get('metadata', [])
                self.id_to_index = metadata_data.get('id_to_index', {})
                # Convert string keys back to int
                self.index_to_id = {int(k): v for k, v in metadata_data.get('index_to_id', {}).items()}
                self.next_id = metadata_data.get('next_id', 0)
                
                logger.info(f"Đã load metadata từ: {metadata_path}")
                logger.info(f"Tổng số vectors: {len(self.metadata)}")
            else:
                logger.warning(f"File metadata không tồn tại: {metadata_path}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Lỗi khi load index: {e}")
            return False
    
    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """
        Normalize vectors để sử dụng với cosine similarity
        
        Args:
            vectors: Input vectors
            
        Returns:
            numpy.ndarray: Normalized vectors
        """
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Tránh chia cho 0
        return vectors / norms
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Lấy thống kê về vector store
        
        Returns:
            Dict chứa thống kê
        """
        stats = {
            'total_vectors': self.index.ntotal if self.index else 0,
            'dimension': self.dimension,
            'index_type': self.index_type,
            'metadata_count': len(self.metadata),
            'next_id': self.next_id
        }
        
        if self.index and hasattr(self.index, 'is_trained'):
            stats['is_trained'] = self.index.is_trained
        
        return stats
    
    def clear(self):
        """Xóa tất cả dữ liệu trong vector store"""
        self._initialize_index()
        self.metadata = []
        self.id_to_index = {}
        self.index_to_id = {}
        self.next_id = 0
        logger.info("Đã xóa tất cả dữ liệu trong vector store")

# Convenience functions
def create_vector_store(config_path: str = "config/rag_config.yaml") -> VectorStore:
    """Tạo vector store instance"""
    return VectorStore(config_path)

def load_or_create_vector_store(config_path: str = "config/rag_config.yaml") -> VectorStore:
    """Load vector store nếu có, tạo mới nếu không"""
    store = VectorStore(config_path)
    store.load_index()  # Sẽ tạo mới nếu không load được
    return store
