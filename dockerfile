# Multi-stage Dockerfile for building NVIDIA Apex with CUDA

# Stage 1: Build NVIDIA Apex with CUDA
FROM nvidia/cuda:12.1-devel-ubuntu22.04 AS apex-builder

# Install Python 3.13.2
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
    python3.13 \
    python3.13-dev \
    python3.13-venv \
    python3.13-distutils \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.13 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.13 1 \
    && update-alternatives --set python3 /usr/bin/python3.13

# Install pip
RUN apt-get update && apt-get install -y python3-pip && rm -rf /var/lib/apt/lists/*

# Install build dependencies and Python build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    && pip install --upgrade pip setuptools wheel \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA support (required for Apex)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Copy Apex source code
COPY apex/ /tmp/apex/

# Set working directory
WORKDIR /tmp/apex

# Build Apex wheel with CUDA extensions
RUN NVCC_APPEND_FLAGS="--threads 4" APEX_PARALLEL_BUILD=8 APEX_CPP_EXT=1 APEX_CUDA_EXT=1 pip wheel -v --no-build-isolation .

# Copy the built wheel to a known location
RUN cp dist/*.whl /tmp/apex-wheel.whl

# Stage 2: Final application image
FROM nvidia/cuda:12.1-runtime-ubuntu22.04

# Install Python 3.13.2
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
    python3.13 \
    python3.13-venv \
    python3.13-distutils \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.13 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.13 1 \
    && update-alternatives --set python3 /usr/bin/python3.13

# Install pip
RUN apt-get update && apt-get install -y python3-pip && rm -rf /var/lib/apt/lists/*

# Copy PyTorch and Apex wheel from builder stage
COPY --from=apex-builder /usr/local/lib/python3.13/site-packages/torch* /usr/local/lib/python3.13/site-packages/
COPY --from=apex-builder /usr/local/lib/python3.13/site-packages/torchvision* /usr/local/lib/python3.13/site-packages/
COPY --from=apex-builder /usr/local/lib/python3.13/site-packages/torchaudio* /usr/local/lib/python3.13/site-packages/
COPY --from=apex-builder /tmp/apex-wheel.whl /tmp/

# Install Apex from wheel
RUN pip install --no-cache-dir /tmp/apex-wheel.whl

# Set working directory
WORKDIR /app

# Copy application files
COPY . /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port (adjust as needed)
EXPOSE 8000

# Default command
CMD ["python3", "main.py"]
