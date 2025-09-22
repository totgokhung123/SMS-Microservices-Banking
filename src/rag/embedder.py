"""
Embedder Module for RAG System - HDBank Banking Chatbot
Module để tạo embeddings từ text sử dụng sentence-transformers
"""

import logging
from typing import List, Dict, Any, Optional, Union
import numpy as np
import yaml
from pathlib import Path
import json
import pickle
import hashlib

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None
    logging.warning("sentence-transformers không được cài đặt. Vui lòng cài đặt: pip install sentence-transformers")

try:
    import torch
except ImportError:
    torch = None
    logging.warning("PyTorch không được cài đặt")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextEmbedder:
    """
    Lớp chính để tạo embeddings từ text
    Hỗ trợ caching và batch processing
    """
    
    def __init__(self, config_path: str = "config/rag_config.yaml"):
        """
        Khởi tạo TextEmbedder
        
        Args:
            config_path: Đường dẫn đến file config
        """
        self.config = self._load_config(config_path)
        self.model_name = self.config['embedding']['model_name']
        self.model_kwargs = self.config['embedding']['model_kwargs']
        self.encode_kwargs = self.config['embedding']['encode_kwargs']
        self.dimension = self.config['embedding']['dimension']
        
        # Khởi tạo model
        self.model = None
        self.device = self._get_device()
        
        # Cache settings
        self.cache_enabled = self.config['performance']['cache_embeddings']
        self.cache_dir = Path("data/cache/embeddings")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        self._load_model()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration từ YAML file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Không thể load config từ {config_path}: {e}")
            # Return default config nếu không load được
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Trả về config mặc định"""
        return {
            'embedding': {
                'model_name': 'keepitreal/vietnamese-sbert',
                'model_kwargs': {'device': 'cpu'},
                'encode_kwargs': {
                    'normalize_embeddings': True,
                    'batch_size': 32
                },
                'dimension': 768
            },
            'performance': {
                'cache_embeddings': True
            }
        }
    
    def _get_device(self) -> str:
        """Xác định device để sử dụng"""
        if torch and torch.cuda.is_available():
            device = "cuda"
            logger.info(f"Sử dụng GPU: {torch.cuda.get_device_name()}")
        else:
            device = "cpu"
            logger.info("Sử dụng CPU")
        
        return device
    
    def _load_model(self):
        """Load sentence transformer model"""
        if not SentenceTransformer:
            raise ImportError("sentence-transformers chưa được cài đặt")
            
        try:
            logger.info(f"Đang load model: {self.model_name}")
            
            # Update device trong model_kwargs
            model_kwargs = self.model_kwargs.copy()
            model_kwargs['device'] = self.device
            
            self.model = SentenceTransformer(
                self.model_name,
                **model_kwargs
            )
            
            logger.info(f"Đã load model thành công. Device: {self.device}")
            
            # Kiểm tra dimension
            test_embedding = self.model.encode("test", show_progress_bar=False)
            actual_dim = len(test_embedding)
            if actual_dim != self.dimension:
                logger.warning(f"Dimension thực tế ({actual_dim}) khác với config ({self.dimension})")
                self.dimension = actual_dim
                
        except Exception as e:
            logger.error(f"Lỗi khi load model {self.model_name}: {e}")
            raise
    
    def encode_texts(self, texts: Union[str, List[str]], 
                    show_progress: bool = True) -> np.ndarray:
        """
        Encode texts thành embeddings
        
        Args:
            texts: Text hoặc danh sách texts cần encode
            show_progress: Có hiển thị progress bar không
            
        Returns:
            numpy.ndarray: Embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            return np.array([])
        
        # Kiểm tra cache trước
        if self.cache_enabled:
            cached_embeddings = self._get_cached_embeddings(texts)
            if cached_embeddings is not None:
                logger.info(f"Sử dụng cached embeddings cho {len(texts)} texts")
                return cached_embeddings
        
        try:
            logger.info(f"Đang encode {len(texts)} texts...")
            
            # Encode với sentence transformer
            embeddings = self.model.encode(
                texts,
                show_progress_bar=show_progress,
                **self.encode_kwargs
            )
            
            # Chuyển thành numpy array nếu cần
            if not isinstance(embeddings, np.ndarray):
                embeddings = np.array(embeddings)
            
            logger.info(f"Đã encode thành công. Shape: {embeddings.shape}")
            
            # Cache kết quả
            if self.cache_enabled:
                self._cache_embeddings(texts, embeddings)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Lỗi khi encode texts: {e}")
            raise
    
    def encode_single_text(self, text: str) -> np.ndarray:
        """
        Encode một text duy nhất
        
        Args:
            text: Text cần encode
            
        Returns:
            numpy.ndarray: Embedding vector
        """
        embeddings = self.encode_texts([text], show_progress=False)
        return embeddings[0] if len(embeddings) > 0 else np.array([])
    
    def _get_cache_key(self, texts: List[str]) -> str:
        """
        Tạo cache key từ danh sách texts
        
        Args:
            texts: Danh sách texts
            
        Returns:
            str: Cache key
        """
        # Tạo hash từ content và model name
        content = "|".join(texts) + f"|{self.model_name}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_cached_embeddings(self, texts: List[str]) -> Optional[np.ndarray]:
        """
        Lấy embeddings từ cache nếu có
        
        Args:
            texts: Danh sách texts
            
        Returns:
            numpy.ndarray hoặc None nếu không có cache
        """
        cache_key = self._get_cache_key(texts)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                return cached_data['embeddings']
            except Exception as e:
                logger.warning(f"Không thể load cache {cache_file}: {e}")
                
        return None
    
    def _cache_embeddings(self, texts: List[str], embeddings: np.ndarray):
        """
        Cache embeddings
        
        Args:
            texts: Danh sách texts
            embeddings: Embeddings tương ứng
        """
        try:
            cache_key = self._get_cache_key(texts)
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            
            cache_data = {
                'texts': texts,
                'embeddings': embeddings,
                'model_name': self.model_name,
                'timestamp': import_time.time()
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
                
        except Exception as e:
            logger.warning(f"Không thể cache embeddings: {e}")
    
    def batch_encode_documents(self, documents: List[Dict[str, Any]], 
                              content_field: str = 'content',
                              batch_size: Optional[int] = None) -> List[np.ndarray]:
        """
        Encode một batch documents
        
        Args:
            documents: Danh sách documents (dicts với content field)
            content_field: Tên field chứa text content
            batch_size: Kích thước batch (lấy từ config nếu None)
            
        Returns:
            List[np.ndarray]: Danh sách embeddings
        """
        if batch_size is None:
            batch_size = self.encode_kwargs.get('batch_size', 32)
        
        texts = [doc[content_field] for doc in documents if content_field in doc]
        
        if not texts:
            logger.warning("Không tìm thấy texts để encode")
            return []
        
        embeddings = []
        
        # Process theo batch
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.encode_texts(batch_texts)
            embeddings.extend(batch_embeddings)
        
        return embeddings
    
    def get_embedding_info(self) -> Dict[str, Any]:
        """
        Lấy thông tin về embedding model
        
        Returns:
            Dict chứa thông tin model
        """
        info = {
            'model_name': self.model_name,
            'dimension': self.dimension,
            'device': self.device,
            'cache_enabled': self.cache_enabled
        }
        
        if self.model:
            info['model_loaded'] = True
            # Thêm thông tin về model nếu có
            if hasattr(self.model, 'get_sentence_embedding_dimension'):
                info['actual_dimension'] = self.model.get_sentence_embedding_dimension()
        else:
            info['model_loaded'] = False
        
        return info
    
    def clear_cache(self):
        """Xóa tất cả cache"""
        if self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("*.pkl"):
                try:
                    cache_file.unlink()
                except Exception as e:
                    logger.warning(f"Không thể xóa {cache_file}: {e}")
            logger.info("Đã xóa tất cả cache")

# Import time for caching
import time as import_time

# Convenience functions
def create_embedder(config_path: str = "config/rag_config.yaml") -> TextEmbedder:
    """Tạo embedder instance"""
    return TextEmbedder(config_path)

def embed_texts(texts: Union[str, List[str]], 
               config_path: str = "config/rag_config.yaml") -> np.ndarray:
    """Convenience function để embed texts"""
    embedder = TextEmbedder(config_path)
    return embedder.encode_texts(texts)
