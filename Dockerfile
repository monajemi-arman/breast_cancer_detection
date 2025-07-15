# Use the NVIDIA CUDA UBI8 base image
FROM nvidia/cuda:12.8.1-cudnn-devel-ubi8

# Set environment variables to avoid interactive prompts
ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies using dnf
RUN dnf clean all && \
    dnf -y install \
    git \
    curl \
    python3 \
    python3-pip \
    nodejs \
    ffmpeg \
    libSM \
    libXext \
    file \
    shadow-utils \
    && dnf clean all

# Create a non-root user and set up its home directory
RUN useradd -m user && \
    chown -R user:user /home/user && \
    chmod -R u+w /home/user

WORKDIR /home/user

# Clone the repository as root (then chown)
RUN git clone https://github.com/monajemi-arman/breast_cancer_detection.git && \
    chown -R user:user /home/user/breast_cancer_detection

# Install Python dependencies
RUN pip3 install --upgrade pip && \
    pip3 install torch torchvision torchaudio && \
    pip3 install -r /home/user/breast_cancer_detection/requirements.txt

# Copy required files and set permissions
COPY --chown=user:user ./last.ckpt /home/user/breast_cancer_detection/classification_output/last.ckpt
COPY --chown=user:user ./config.json /home/user/breast_cancer_detection/llm/config.json
COPY --chown=user:user ./detectron.cfg.pkl /home/user/breast_cancer_detection/webapp/detectron.cfg.pkl
COPY --chown=user:user ./model.pth /home/user/breast_cancer_detection/webapp/model.pth
COPY --chown=user:user ./mammoGraphyLabeling /home/user/mammoGraphyLabeling

# Switch to the non-root user
USER user

# Verify the repository is not empty
RUN if [ -z "$(ls -A /home/user/breast_cancer_detection)" ]; then \
        echo "Error: breast_cancer_detection repository is empty or failed to clone."; \
        exit 1; \
    fi

# Debug: list contents
RUN ls -lah /home/user/breast_cancer_detection

# Install npm dependencies
WORKDIR /home/user/mammoGraphyLabeling
RUN npm install

# Set the working directory for app startup
WORKDIR /home/user/breast_cancer_detection

# Expose ports
EXPOSE 3000-3006 33510-33530

# Start the services and prevent container exit on error
CMD bash -c "ls -lah /home/user/breast_cancer_detection && python3 start_api_services.py & cd /home/user/mammoGraphyLabeling && npm run dev || sleep infinity"
