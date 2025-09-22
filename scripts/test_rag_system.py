"""
Test Script for HDBank RAG System
Script để test toàn bộ RAG pipeline với dữ liệu HDBank
"""

import sys
import os
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_dependencies():
    """Test xem các dependencies đã được cài đặt chưa"""
    logger.info("🔍 Kiểm tra dependencies...")
    
    try:
        from src.rag import check_dependencies, get_rag_info
        
        deps = check_dependencies()
        info = get_rag_info()
        
        logger.info("📊 Trạng thái dependencies:")
        for dep, status in deps.items():
            status_icon = "✅" if status else "❌"
            logger.info(f"  {status_icon} {dep}: {status}")
        
        logger.info(f"📋 System info: {info}")
        
        missing_deps = [dep for dep, status in deps.items() if not status]
        if missing_deps:
            logger.error(f"❌ Missing dependencies: {missing_deps}")
            logger.error("Vui lòng cài đặt: pip install sentence-transformers faiss-cpu PyYAML scikit-learn")
            return False
        
        logger.info("✅ Tất cả dependencies đã sẵn sàng!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Lỗi khi kiểm tra dependencies: {e}")
        return False

def test_document_loading():
    """Test document loading"""
    logger.info("📚 Testing document loading...")
    
    try:
        from src.rag import DocumentLoader
        
        loader = DocumentLoader()
        
        # Test load documents
        documents = loader.load_documents()
        logger.info(f"✅ Loaded {len(documents)} documents")
        
        if documents:
            # Show first document info
            doc = documents[0]
            logger.info(f"📄 Sample document: {doc.doc_id}")
            logger.info(f"   Source: {doc.source}")
            logger.info(f"   Content length: {len(doc.content)}")
            logger.info(f"   Content preview: {doc.content[:100]}...")
        
        # Test chunking
        chunks = loader.chunk_documents(documents)
        logger.info(f"🔗 Created {len(chunks)} chunks")
        
        if chunks:
            # Show statistics
            stats = loader.get_chunk_statistics(chunks)
            logger.info("📊 Chunk statistics:")
            for key, value in stats.items():
                logger.info(f"   {key}: {value}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Lỗi khi test document loading: {e}")
        return False

def test_embedding():
    """Test embedding system"""
    logger.info("🔢 Testing embedding system...")
    
    try:
        from src.rag import TextEmbedder
        
        embedder = TextEmbedder()
        
        # Test single text embedding
        test_text = "Tôi muốn mở thẻ tín dụng HDBank"
        embedding = embedder.encode_single_text(test_text)
        
        logger.info(f"✅ Single embedding shape: {embedding.shape}")
        logger.info(f"   Sample values: {embedding[:5]}")
        
        # Test batch embedding
        test_texts = [
            "Làm thế nào để chuyển khoản qua internet banking?",
            "Hạn mức rút tiền ATM của HDBank là bao nhiêu?",
            "Cách mở tài khoản tiết kiệm HDBank"
        ]
        
        embeddings = embedder.encode_texts(test_texts)
        logger.info(f"✅ Batch embeddings shape: {embeddings.shape}")
        
        # Test embedding info
        info = embedder.get_embedding_info()
        logger.info("📋 Embedding info:")
        for key, value in info.items():
            logger.info(f"   {key}: {value}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Lỗi khi test embedding: {e}")
        return False

def test_vector_store():
    """Test vector store"""
    logger.info("🗃️ Testing vector store...")
    
    try:
        from src.rag import VectorStore, TextEmbedder
        
        # Create test data
        embedder = TextEmbedder()
        texts = [
            "Thẻ tín dụng HDBank có nhiều ưu đãi",
            "Internet banking HDBank rất tiện lợi", 
            "Gửi tiết kiệm HDBank lãi suất cao"
        ]
        
        embeddings = embedder.encode_texts(texts)
        metadata = [
            {'content': texts[0], 'category': 'CARD'},
            {'content': texts[1], 'category': 'EBANKING'},
            {'content': texts[2], 'category': 'SAVINGS'}
        ]
        
        # Test vector store
        store = VectorStore()
        store.clear()  # Clear any existing data
        
        # Add vectors
        indices = store.add_vectors(embeddings, metadata)
        logger.info(f"✅ Added {len(indices)} vectors to store")
        
        # Test search
        query_text = "tôi muốn biết về thẻ tín dụng"
        query_embedding = embedder.encode_single_text(query_text)
        
        results = store.search(query_embedding, k=2)
        logger.info(f"🔍 Search results for '{query_text}':")
        for i, result in enumerate(results):
            logger.info(f"   {i+1}. Score: {result['similarity_score']:.3f}")
            logger.info(f"      Content: {result['content']}")
            logger.info(f"      Category: {result['category']}")
        
        # Test stats
        stats = store.get_stats()
        logger.info("📊 Vector store stats:")
        for key, value in stats.items():
            logger.info(f"   {key}: {value}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Lỗi khi test vector store: {e}")
        return False

def test_full_rag_pipeline():
    """Test complete RAG pipeline"""
    logger.info("🚀 Testing complete RAG pipeline...")
    
    try:
        from src.rag import RAGPipeline
        
        # Create and initialize pipeline
        rag = RAGPipeline()
        success = rag.initialize(force_rebuild=True)
        
        if not success:
            logger.error("❌ Không thể initialize RAG pipeline")
            return False
        
        logger.info("✅ RAG pipeline initialized successfully")
        
        # Test queries
        test_queries = [
            "Làm thế nào để mở thẻ tín dụng HDBank?",
            "Hạn mức chuyển khoản internet banking là bao nhiêu?",
            "Cách rút tiền ATM HDBank",
            "Điều kiện mở tài khoản tiết kiệm",
            "Phí dịch vụ thẻ ghi nợ HDBank"
        ]
        
        logger.info("🔍 Testing search queries:")
        for query in test_queries:
            logger.info(f"\n📝 Query: {query}")
            
            # Test basic search
            results = rag.search(query, top_k=3)
            logger.info(f"   Found {len(results)} results")
            
            for i, result in enumerate(results[:2]):  # Show top 2
                score = result.get('similarity_score', 0)
                content = result.get('content', '')[:100]
                file_name = result.get('file_name', 'Unknown')
                
                logger.info(f"   {i+1}. Score: {score:.3f} | Source: {file_name}")
                logger.info(f"      Content: {content}...")
            
            # Test context generation
            context_result = rag.get_context(query, top_k=2)
            context_length = context_result.get('context_length', 0)
            logger.info(f"   Context length: {context_length} chars")
        
        # Test category search
        logger.info("\n🏷️ Testing category search:")
        category_query = "hạn mức thẻ tín dụng"
        results = rag.search(category_query, top_k=3, category="CARD")
        logger.info(f"Category search results for '{category_query}': {len(results)}")
        
        # Get system stats
        stats = rag.get_stats()
        logger.info("\n📊 RAG system stats:")
        logger.info(f"   Initialized: {stats.get('initialized', False)}")
        if 'vector_store' in stats:
            vs_stats = stats['vector_store']
            logger.info(f"   Total vectors: {vs_stats.get('total_vectors', 0)}")
            logger.info(f"   Index type: {vs_stats.get('index_type', 'Unknown')}")
        
        logger.info("✅ RAG pipeline test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Lỗi khi test RAG pipeline: {e}")
        return False

def test_convenience_functions():
    """Test convenience functions"""
    logger.info("⚡ Testing convenience functions...")
    
    try:
        from src.rag import quick_search, search_documents
        
        # Test quick search
        query = "thẻ tín dụng HDBank"
        results = quick_search(query, top_k=2)
        logger.info(f"✅ Quick search found {len(results)} results for '{query}'")
        
        # Test search documents with category
        results = search_documents(query, top_k=2, category="CARD")
        logger.info(f"✅ Category search found {len(results)} results")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Lỗi khi test convenience functions: {e}")
        return False

def main():
    """Main test function"""
    logger.info("🧪 Bắt đầu test HDBank RAG System")
    logger.info("=" * 60)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Document Loading", test_document_loading),
        ("Embedding", test_embedding),
        ("Vector Store", test_vector_store),
        ("Full RAG Pipeline", test_full_rag_pipeline),
        ("Convenience Functions", test_convenience_functions)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*20} {test_name} {'='*20}")
        
        try:
            if test_func():
                logger.info(f"✅ {test_name} PASSED")
                passed += 1
            else:
                logger.error(f"❌ {test_name} FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name} ERROR: {e}")
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"🏁 Test Summary: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 Tất cả tests PASSED! RAG system hoạt động tốt.")
        return True
    else:
        logger.error(f"⚠️ {total - passed} tests FAILED. Vui lòng kiểm tra lại.")
        return False

if __name__ == "__main__":
    # Ensure we're in the right directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir.parent)  # Go to project root
    
    try:
        success = main()
        exit_code = 0 if success else 1
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("\n⚠️ Test bị interrupt bởi user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Lỗi không mong muốn: {e}")
        sys.exit(1)