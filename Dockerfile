# Use the NVIDIA CUDA base image
FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04

# Set environment variables to avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    python3 \
    python3-pip \
    python-is-python3 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --break-system-packages torch torchvision torchaudio

# Install new node
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs

# Make sure necessary libraries are present for fully functional python scripts
RUN apt-get install -y ffmpeg libsm6 libxext6 libmagic1

# Our required modules
COPY requirements.txt /
RUN pip install --break-system-packages -r /requirements.txt