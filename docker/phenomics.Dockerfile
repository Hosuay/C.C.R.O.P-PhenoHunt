# Phenomics Pipeline Dockerfile
# PhytoOracle-inspired containerized environment for C.C.R.O.P-PhenoHunt
#
# Sacred Geometry: Base image uses Python 3.11 (3 + 1 + 1 = 5, pentagon)

FROM python:3.11-slim

# Metadata
LABEL maintainer="C.C.R.O.P-PhenoHunt Team"
LABEL version="1.0.0"
LABEL description="Containerized phenomics pipeline for cannabis breeding research"
LABEL sacred_geometry="369"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    SACRED_SEED=369 \
    DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Image processing libraries
    libopencv-dev \
    python3-opencv \
    # Additional tools
    git \
    wget \
    curl \
    # Cleanup
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt /workspace/requirements.txt

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    # Additional phenomics-specific dependencies
    pip install opencv-python-headless>=4.8.0 scikit-image>=0.21.0

# Copy source code
COPY src/ /workspace/src/
COPY scripts/ /workspace/scripts/
COPY pipelines/ /workspace/pipelines/
COPY configs/ /workspace/configs/

# Create directories for data
RUN mkdir -p /workspace/data_examples/images \
             /workspace/data/processed \
             /workspace/data/results \
             /workspace/logs

# Make scripts executable
RUN chmod +x /workspace/scripts/*.py

# Set Python path
ENV PYTHONPATH=/workspace:$PYTHONPATH

# Default command: run pipeline in dry-run mode
CMD ["python", "/workspace/scripts/run_pipeline.py", "--config", "/workspace/pipelines/phenomics_pipeline.yml", "--dry-run"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.path.insert(0, '/workspace'); from src.phenomics.feature_extraction import FeatureExtractor; print('OK')" || exit 1

# Expose port for future API integration
EXPOSE 8050

# Volume mounts for data persistence
VOLUME ["/workspace/data", "/workspace/data_examples", "/workspace/logs"]
