"""
Data Loader for RAG System - HDBank Banking Chatbot
Module để load và xử lý documents từ thư mục docs/
"""

import os
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import yaml
from dataclasses import dataclass

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Document:
    """Lớp đại diện cho một document"""
    content: str
    metadata: Dict[str, Any]
    doc_id: str
    source: str

@dataclass  
class Chunk:
    """Lớp đại diện cho một chunk text"""
    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    doc_id: str
    start_pos: int
    end_pos: int

class DocumentLoader:
    """Lớp chính để load và xử lý documents"""
    
    def __init__(self, config_path: str = "config/rag_config.yaml"):
        """
        Khởi tạo DocumentLoader
        
        Args:
            config_path: Đường dẫn đến file config
        """
        self.config = self._load_config(config_path)
        self.docs_dir = self.config['paths']['docs_dir']
        self.chunk_size = self.config['text_processing']['chunk_size']
        self.chunk_overlap = self.config['text_processing']['chunk_overlap']
        self.separator = self.config['text_processing']['separator']
        self.supported_formats = self.config['document_loader']['supported_formats']
        self.encoding = self.config['document_loader']['encoding']
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration từ YAML file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Không thể load config từ {config_path}: {e}")
            raise
    
    def load_documents(self, docs_dir: Optional[str] = None) -> List[Document]:
        """
        Load tất cả documents từ thư mục
        
        Args:
            docs_dir: Thư mục chứa documents (optional)
            
        Returns:
            List[Document]: Danh sách documents đã load
        """
        if docs_dir is None:
            docs_dir = self.docs_dir
            
        documents = []
        docs_path = Path(docs_dir)
        
        if not docs_path.exists():
            logger.error(f"Thư mục {docs_dir} không tồn tại")
            return documents
            
        logger.info(f"Đang load documents từ {docs_dir}")
        
        for file_path in docs_path.iterdir():
            if file_path.is_file() and file_path.suffix in self.supported_formats:
                try:
                    document = self._load_single_document(file_path)
                    if document:
                        documents.append(document)
                        logger.info(f"Đã load: {file_path.name}")
                except Exception as e:
                    logger.error(f"Lỗi khi load {file_path}: {e}")
                    
        logger.info(f"Đã load tổng cộng {len(documents)} documents")
        return documents
    
    def _load_single_document(self, file_path: Path) -> Optional[Document]:
        """
        Load một document từ file
        
        Args:
            file_path: Đường dẫn đến file
            
        Returns:
            Document hoặc None nếu có lỗi
        """
        try:
            with open(file_path, 'r', encoding=self.encoding) as f:
                content = f.read()
            
            # Làm sạch text
            content = self._clean_text(content)
            
            # Tạo metadata
            metadata = {
                'file_name': file_path.name,
                'file_path': str(file_path),
                'file_size': file_path.stat().st_size,
                'file_type': file_path.suffix,
                'created_time': file_path.stat().st_ctime
            }
            
            # Tạo doc_id từ tên file
            doc_id = self._generate_doc_id(file_path.name)
            
            return Document(
                content=content,
                metadata=metadata,
                doc_id=doc_id,
                source=str(file_path)
            )
            
        except Exception as e:
            logger.error(f"Lỗi khi đọc file {file_path}: {e}")
            return None
    
    def _clean_text(self, text: str) -> str:
        """
        Làm sạch text
        
        Args:
            text: Text cần làm sạch
            
        Returns:
            Text đã được làm sạch
        """
        # Loại bỏ ký tự đặc biệt không cần thiết
        text = re.sub(r'\r\n', '\n', text)  # Chuẩn hóa line breaks
        text = re.sub(r'\n{3,}', '\n\n', text)  # Giới hạn line breaks liên tiếp
        text = re.sub(r'[ \t]+', ' ', text)  # Chuẩn hóa spaces
        text = text.strip()
        
        return text
    
    def _generate_doc_id(self, filename: str) -> str:
        """
        Tạo doc_id từ tên file
        
        Args:
            filename: Tên file
            
        Returns:
            doc_id duy nhất
        """
        # Loại bỏ extension và ký tự đặc biệt
        base_name = Path(filename).stem
        doc_id = re.sub(r'[^a-zA-Z0-9_]', '_', base_name).lower()
        return doc_id
    
    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        """
        Chia documents thành các chunks nhỏ hơn
        
        Args:
            documents: Danh sách documents
            
        Returns:
            List[Chunk]: Danh sách chunks
        """
        all_chunks = []
        
        logger.info(f"Đang chia {len(documents)} documents thành chunks...")
        
        for doc in documents:
            chunks = self._chunk_single_document(doc)
            all_chunks.extend(chunks)
            
        logger.info(f"Đã tạo {len(all_chunks)} chunks từ {len(documents)} documents")
        return all_chunks
    
    def _chunk_single_document(self, document: Document) -> List[Chunk]:
        """
        Chia một document thành chunks
        
        Args:
            document: Document cần chia
            
        Returns:
            List[Chunk]: Danh sách chunks của document
        """
        text = document.content
        chunks = []
        
        # Chia text thành các phần dựa trên separator
        sections = text.split(self.separator)
        
        current_chunk = ""
        current_pos = 0
        chunk_index = 0
        
        for section in sections:
            section = section.strip()
            if not section:
                continue
                
            # Kiểm tra xem thêm section này có vượt quá chunk_size không
            potential_chunk = current_chunk + self.separator + section if current_chunk else section
            
            if len(potential_chunk) <= self.chunk_size:
                current_chunk = potential_chunk
            else:
                # Lưu chunk hiện tại nếu có
                if current_chunk:
                    chunk = self._create_chunk(
                        current_chunk, document, chunk_index, current_pos
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    
                    # Xử lý overlap
                    if self.chunk_overlap > 0:
                        overlap_text = current_chunk[-self.chunk_overlap:]
                        current_chunk = overlap_text + self.separator + section
                        current_pos += len(current_chunk) - self.chunk_overlap
                    else:
                        current_chunk = section
                        current_pos += len(current_chunk)
                else:
                    current_chunk = section
        
        # Lưu chunk cuối cùng
        if current_chunk:
            chunk = self._create_chunk(
                current_chunk, document, chunk_index, current_pos
            )
            chunks.append(chunk)
        
        return chunks
    
    def _create_chunk(self, content: str, document: Document, 
                     chunk_index: int, start_pos: int) -> Chunk:
        """
        Tạo một chunk object
        
        Args:
            content: Nội dung chunk
            document: Document gốc
            chunk_index: Index của chunk
            start_pos: Vị trí bắt đầu trong document
            
        Returns:
            Chunk object
        """
        chunk_id = f"{document.doc_id}_chunk_{chunk_index}"
        
        # Copy metadata từ document và thêm thông tin chunk
        metadata = document.metadata.copy()
        metadata.update({
            'chunk_index': chunk_index,
            'chunk_length': len(content),
            'parent_doc_id': document.doc_id
        })
        
        return Chunk(
            content=content,
            metadata=metadata,
            chunk_id=chunk_id,
            doc_id=document.doc_id,
            start_pos=start_pos,
            end_pos=start_pos + len(content)
        )
    
    def load_and_chunk(self, docs_dir: Optional[str] = None) -> List[Chunk]:
        """
        Convenience method để load và chunk documents trong một bước
        
        Args:
            docs_dir: Thư mục chứa documents
            
        Returns:
            List[Chunk]: Danh sách chunks
        """
        documents = self.load_documents(docs_dir)
        chunks = self.chunk_documents(documents)
        return chunks
    
    def get_chunk_statistics(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """
        Tính toán thống kê về chunks
        
        Args:
            chunks: Danh sách chunks
            
        Returns:
            Dict chứa thống kê
        """
        if not chunks:
            return {}
            
        chunk_lengths = [len(chunk.content) for chunk in chunks]
        
        stats = {
            'total_chunks': len(chunks),
            'total_documents': len(set(chunk.doc_id for chunk in chunks)),
            'avg_chunk_length': sum(chunk_lengths) / len(chunk_lengths),
            'min_chunk_length': min(chunk_lengths),
            'max_chunk_length': max(chunk_lengths),
            'total_text_length': sum(chunk_lengths)
        }
        
        return stats

# Convenience functions
def load_hdbank_documents(config_path: str = "config/rag_config.yaml") -> List[Document]:
    """Load tất cả documents HDBank"""
    loader = DocumentLoader(config_path)
    return loader.load_documents()

def load_and_chunk_hdbank_documents(config_path: str = "config/rag_config.yaml") -> List[Chunk]:
    """Load và chunk tất cả documents HDBank"""
    loader = DocumentLoader(config_path)
    return loader.load_and_chunk()
