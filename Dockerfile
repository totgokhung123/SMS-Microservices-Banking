# HDBank AI Chatbot Backend - Multi-stage Docker build

# Stage 1: Base Python image với system dependencies
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    gcc \
    g++ \
    git \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash hdbank
WORKDIR /app
RUN chown hdbank:hdbank /app

# Stage 2: Dependencies installation
FROM base as dependencies

# Copy requirements first for better caching
COPY requirements.txt requirements_finetune.txt ./

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Optional: Install fine-tuning dependencies (comment out nếu không cần)
# RUN pip install -r requirements_finetune.txt

# Stage 3: Application
FROM dependencies as application

# Switch to non-root user
USER hdbank

# Copy application source code
COPY --chown=hdbank:hdbank . .

# Create necessary directories
RUN mkdir -p logs data/vector_db models

# Set Python path
ENV PYTHONPATH="/app:/app/src"

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Development stage với additional tools
FROM application as development

USER root

# Install development dependencies
RUN pip install \
    pytest \
    pytest-asyncio \
    black \
    isort \
    flake8 \
    jupyter \
    ipykernel

# Install development tools
RUN apt-get update && apt-get install -y \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

USER hdbank

# Development command với reload
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Production stage với optimizations
FROM application as production

# Production optimizations
ENV PYTHONOPTIMIZE=1

# Use gunicorn for production
RUN pip install gunicorn

# Production command
CMD ["gunicorn", "src.main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--access-logfile", "-", "--error-logfile", "-"]