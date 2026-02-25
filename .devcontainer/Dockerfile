# Python development image with PyTorch (CUDA) and scientific stack
# Fedora 41, Python 3.12, PyTorch with CUDA 12.4
FROM fedora:41

# Install Python 3.12 and system deps for numpy/scipy (openblas)
# Note: Fedora 41 has no python3.12-pip package; we use ensurepip below
RUN dnf install -y \
    python3.12 \
    git \
    openblas \
    openblas-devel \
    && dnf clean all

# Bootstrap pip for Python 3.12 and symlink/wrapper for devcontainer
RUN python3.12 -m ensurepip --upgrade && \
    ln -sf /usr/bin/python3.12 /usr/local/bin/python && \
    printf '%s\n' '#!/bin/sh' 'exec /usr/bin/python3.12 -m pip "$@"' > /usr/local/bin/pip && chmod +x /usr/local/bin/pip

WORKDIR /workspace

# Install PyTorch with CUDA 12.4 (from PyTorch index)
RUN python3.12 -m pip install --no-cache-dir --upgrade pip && \
    python3.12 -m pip install --no-cache-dir torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

# Install scientific stack from requirements.txt
COPY requirements.txt .
RUN python3.12 -m pip install --no-cache-dir -r requirements.txt

CMD ["bash"]
