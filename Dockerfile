# Use the NVIDIA CUDA base image
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Set environment variables to avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Temporarily disable the NVIDIA repository
RUN mv /etc/apt/sources.list.d/cuda-ubuntu2204-x86_64.list /etc/apt/sources.list.d/cuda-ubuntu2204-x86_64.list.disabled

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    python3 \
    python3-pip \
    python-is-python3 \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Re-enable the NVIDIA repository
RUN mv /etc/apt/sources.list.d/cuda-ubuntu2204-x86_64.list.disabled /etc/apt/sources.list.d/cuda-ubuntu2204-x86_64.list

# Create a non-root user
RUN useradd -m user
WORKDIR /home/user

# Clone the repository
RUN git clone https://github.com/monajemi-arman/breast_cancer_detection.git

# Debugging step: List files to ensure they exist
RUN ls -lah /home/user/breast_cancer_detection

# Copy required files
COPY ./last.ckpt /home/user/breast_cancer_detection/classification_output/last.ckpt
COPY ./config.json /home/user/breast_cancer_detection/llm/config.json
COPY ./detectron.cfg.pkl /home/user/breast_cancer_detection/webapp/detectron.cfg.pkl
COPY ./model.pth /home/user/breast_cancer_detection/webapp/model.pth

# Install Python dependencies
RUN pip install torch torchvision torchaudio
RUN pip install -r /home/user/breast_cancer_detection/requirements.txt

# Copy and set up the mammoGraphyLabeling project
COPY ./mammoGraphyLabeling /home/user/mammoGraphyLabeling
WORKDIR /home/user/mammoGraphyLabeling

# Ensure dependencies are installed
RUN npm install
RUN npm install -g vite # Ensures Vite is available globally

# Switch to the non-root user
USER user

# Set the working directory for the breast_cancer_detection project
WORKDIR /home/user/breast_cancer_detection

# Expose ports
EXPOSE 3000-3006 33510-33530

# Command to start both services
CMD bash -c "ls -lah /home/user/breast_cancer_detection && python start_api_services.py & cd /home/user/mammoGraphyLabeling && npm run dev"
