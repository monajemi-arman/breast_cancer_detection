# Use the NVIDIA CUDA base image
FROM docker.io/nvidia/cuda:13.3.0-cudnn-devel-ubuntu24.04

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

# Make sure necessary libraries are present for fully functional python scripts
RUN apt update
RUN apt-get install -y software-properties-common
RUN add-apt-repository universe
RUN apt-get update
RUN apt-get install -y ffmpeg libsm6 libxext6 libmagic1

# Our required modules
COPY requirements.txt /
RUN pip install --break-system-packages --ignore-installed -r /requirements.txt
