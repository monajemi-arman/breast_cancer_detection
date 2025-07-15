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
    && rm -rf /var/lib/apt/lists/*

# Re-enable the NVIDIA repository
RUN mv /etc/apt/sources.list.d/cuda-ubuntu2204-x86_64.list.disabled /etc/apt/sources.list.d/cuda-ubuntu2204-x86_64.list

# Create a non-root user and set up its home directory with proper permissions
RUN useradd -m user && \
    chown -R user:user /home/user && \
    chmod -R u+w /home/user

WORKDIR /home/user

# Clone the repository as the non-root user
RUN git clone https://github.com/monajemi-arman/breast_cancer_detection.git

# Install Python dependencies
RUN pip install torch torchvision torchaudio
RUN pip install -r /home/user/breast_cancer_detection/requirements.txt

RUN chown -R user:user /home/user/*

# Switch to the non-root user
USER user

# Set the working directory
WORKDIR /home/user

# Check if the repository was cloned successfully and is not empty
RUN if [ -z "$(ls -A /home/user/breast_cancer_detection)" ]; then \
        echo "Error: breast_cancer_detection repository is empty or failed to clone."; \
        exit 1; \
    fi

# Debugging step: List files to ensure they exist
RUN ls -lah /home/user/breast_cancer_detection

# Copy required files and ensure they are owned by the non-root user
COPY --chown=user:user ./last.ckpt /home/user/breast_cancer_detection/classification_output/last.ckpt
COPY --chown=user:user ./config.json /home/user/breast_cancer_detection/llm/config.json
COPY --chown=user:user ./detectron.cfg.pkl /home/user/breast_cancer_detection/webapp/detectron.cfg.pkl
COPY --chown=user:user ./model.pth /home/user/breast_cancer_detection/webapp/model.pth

# Copy and set up the mammoGraphyLabeling project
COPY --chown=user:user ./mammoGraphyLabeling /home/user/mammoGraphyLabeling

USER root
# Temporarily disable the NVIDIA repository
RUN mv /etc/apt/sources.list.d/cuda-ubuntu2204-x86_64.list /etc/apt/sources.list.d/cuda-ubuntu2204-x86_64.list.disabled

# Install new node
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs

# Re-enable the NVIDIA repository
RUN mv /etc/apt/sources.list.d/cuda-ubuntu2204-x86_64.list.disabled /etc/apt/sources.list.d/cuda-ubuntu2204-x86_64.list

# Make sure necessary libraries are present for fully functional python scripts
RUN apt-get install -y ffmpeg libsm6 libxext6 libmagic1

USER user
WORKDIR /home/user/mammoGraphyLabeling

# Ensure dependencies are installed
RUN npm install

# Set the working directory for the breast_cancer_detection project
WORKDIR /home/user/breast_cancer_detection

# Expose ports
EXPOSE 3000-3006 33510-33530

# Command to start both services and keep the container alive on error
CMD bash -c "ls -lah /home/user/breast_cancer_detection && python start_api_services.py & cd /home/user/mammoGraphyLabeling && npm run dev || sleep infinity"