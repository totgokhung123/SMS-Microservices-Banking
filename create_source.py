import os

def create_project_structure(root_dir="source"):
    """
    Tạo cấu trúc thư mục và file cho project backend chatbot theo mô tả.
    Chạy script này để tự động tạo toàn bộ structure.
    """
    # Tạo root directory nếu chưa tồn tại
    os.makedirs(root_dir, exist_ok=True)
    
    # Root files
    files = [
        "README.md",
        ".gitignore",
        "pyproject.toml",
        "requirements.txt",
        "Dockerfile",
        "docker-compose.yml",
        ".env.example"
    ]
    for file in files:
        open(os.path.join(root_dir, file), 'w').close()
    
    # .github/workflows/
    workflows_dir = os.path.join(root_dir, ".github", "workflows")
    os.makedirs(workflows_dir, exist_ok=True)
    open(os.path.join(workflows_dir, "ci.yml"), 'w').close()
    
    # config/
    config_dir = os.path.join(root_dir, "config")
    os.makedirs(config_dir, exist_ok=True)
    config_files = [
        "__init__.py",
        "settings.py",
        "model_config.yaml",
        "rag_config.yaml",
        "api_config.yaml"
    ]
    for file in config_files:
        open(os.path.join(config_dir, file), 'w').close()
    
    # data/
    data_dir = os.path.join(root_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # data/raw/
    raw_dir = os.path.join(data_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    
    # data/raw/csv/
    os.makedirs(os.path.join(raw_dir, "csv"), exist_ok=True)
    
    # data/raw/docs/
    docs_dir = os.path.join(raw_dir, "docs")
    os.makedirs(docs_dir, exist_ok=True)
    open(os.path.join(docs_dir, "banking_policies.pdf"), 'w').close()  # Empty placeholder
    open(os.path.join(docs_dir, "financial_terms.html"), 'w').close()
    
    # data/raw/logs/
    os.makedirs(os.path.join(raw_dir, "logs"), exist_ok=True)
    
    # data/processed/
    processed_dir = os.path.join(data_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)
    
    # data/processed/train_split/
    os.makedirs(os.path.join(processed_dir, "train_split"), exist_ok=True)
    
    # data/processed/chunks/
    os.makedirs(os.path.join(processed_dir, "chunks"), exist_ok=True)
    
    # data/vector_db/
    vector_db_dir = os.path.join(data_dir, "vector_db")
    os.makedirs(vector_db_dir, exist_ok=True)
    open(os.path.join(vector_db_dir, "faiss_index.bin"), 'wb').close()  # Empty binary placeholder
    
    # src/
    src_dir = os.path.join(root_dir, "src")
    os.makedirs(src_dir, exist_ok=True)
    
    # src/__init__.py và main.py
    open(os.path.join(src_dir, "__init__.py"), 'w').close()
    open(os.path.join(src_dir, "main.py"), 'w').close()
    
    # src/models/
    models_dir = os.path.join(src_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    open(os.path.join(models_dir, "__init__.py"), 'w').close()
    models_files = ["fine_tune.py", "merge.py", "inference.py"]
    for file in models_files:
        open(os.path.join(models_dir, file), 'w').close()
    
    # src/rag/
    rag_dir = os.path.join(src_dir, "rag")
    os.makedirs(rag_dir, exist_ok=True)
    open(os.path.join(rag_dir, "__init__.py"), 'w').close()
    rag_files = ["data_loader.py", "embedder.py", "vector_store.py", "retriever.py"]
    for file in rag_files:
        open(os.path.join(rag_dir, file), 'w').close()
    
    # src/api/
    api_dir = os.path.join(src_dir, "api")
    os.makedirs(api_dir, exist_ok=True)
    open(os.path.join(api_dir, "__init__.py"), 'w').close()
    open(os.path.join(api_dir, "app.py"), 'w').close()
    open(os.path.join(api_dir, "dependencies.py"), 'w').close()
    
    # src/api/routers/
    routers_dir = os.path.join(api_dir, "routers")
    os.makedirs(routers_dir, exist_ok=True)
    open(os.path.join(routers_dir, "__init__.py"), 'w').close()
    open(os.path.join(routers_dir, "chat.py"), 'w').close()
    open(os.path.join(routers_dir, "health.py"), 'w').close()
    
    # src/utils/
    utils_dir = os.path.join(src_dir, "utils")
    os.makedirs(utils_dir, exist_ok=True)
    open(os.path.join(utils_dir, "__init__.py"), 'w').close()
    utils_files = ["logger.py", "validators.py", "data_utils.py"]
    for file in utils_files:
        open(os.path.join(utils_dir, file), 'w').close()
    
    # src/core/
    core_dir = os.path.join(src_dir, "core")
    os.makedirs(core_dir, exist_ok=True)
    open(os.path.join(core_dir, "__init__.py"), 'w').close()
    open(os.path.join(core_dir, "chatbot_service.py"), 'w').close()
    
    # scripts/
    scripts_dir = os.path.join(root_dir, "scripts")
    os.makedirs(scripts_dir, exist_ok=True)
    scripts_files = ["setup_data.py", "train_model.py", "deploy_model.py"]
    for file in scripts_files:
        open(os.path.join(scripts_dir, file), 'w').close()
    
    # tests/
    tests_dir = os.path.join(root_dir, "tests")
    os.makedirs(tests_dir, exist_ok=True)
    open(os.path.join(tests_dir, "__init__.py"), 'w').close()
    open(os.path.join(tests_dir, "conftest.py"), 'w').close()
    
    # tests/test_models/
    os.makedirs(os.path.join(tests_dir, "test_models"), exist_ok=True)
    
    # tests/test_rag/
    os.makedirs(os.path.join(tests_dir, "test_rag"), exist_ok=True)
    
    # tests/test_api/
    os.makedirs(os.path.join(tests_dir, "test_api"), exist_ok=True)
    
    # docs/
    docs_dir = os.path.join(root_dir, "docs")
    os.makedirs(docs_dir, exist_ok=True)
    open(os.path.join(docs_dir, "architecture.md"), 'w').close()
    open(os.path.join(docs_dir, "api_spec.md"), 'w').close()
    
    # monitoring/
    monitoring_dir = os.path.join(root_dir, "monitoring")
    os.makedirs(monitoring_dir, exist_ok=True)
    open(os.path.join(monitoring_dir, "prometheus.yml"), 'w').close()
    
    # monitoring/grafana/
    os.makedirs(os.path.join(monitoring_dir, "grafana"), exist_ok=True)
    
    print(f"Cấu trúc project đã được tạo thành công tại thư mục '{root_dir}'!")

if __name__ == "__main__":
    create_project_structure()