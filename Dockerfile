# CUDA 12.1 + cuDNN 9 + PyTorch 2.4 (Python 3.10+)
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

ARG DEBIAN_FRONTEND=noninteractive

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential git curl ca-certificates \
      ffmpeg libsm6 libxext6 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Python & CUDA cache env
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=UTF-8 \
    PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_PROGRESS_BAR=off \
    CUDA_CACHE_PATH=/tmp/cuda-cache \
    CUDA_CACHE_MAXSIZE=2147483647 \
    CUDA_MODULE_LOADING=LAZY \
    # Optional: faster matmuls on Ampere
    TORCH_ALLOW_TF32_CUBLAS=1 TORCH_ALLOW_TF32_CUDNN=1

# Fast local CUDA cache
RUN mkdir -p /tmp/cuda-cache && chmod 777 /tmp/cuda-cache

WORKDIR /app

# Tools & PyTorch extras (match CUDA 12.1)
RUN python -m pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cu121 \
      torchvision torchaudio

# App deps
COPY requirements.txt /tmp/requirements.txt
# Make sure requirements.txt does NOT pin torch/torchvision/torchaudio again
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# App code
COPY . /app

# (Optional) non-root user to match host UID/GID
# ARG USERNAME=appuser UID=1000 GID=1000
# RUN groupadd -g $GID $USERNAME && useradd -m -u $UID -g $GID $USERNAME \
#  && chown -R $USERNAME:$USERNAME /app /tmp/cuda-cache
# USER $USERNAME

# Default cmd is set via docker-compose



################################################################################ OLD

#FROM anibali/pytorch:1.4.0-cuda10.1

# Install python dependencies
#COPY requirements.txt .
#RUN pip install --ignore-installed -r requirements.txt

#RUN apt-get update && \
#    apt-get install -y --no-install-recommends \
#        tmux \
#        nano \
#        && \
#    rm -rf /var/lib/apt/lists/*


#CMD ["/bin/bash"]
