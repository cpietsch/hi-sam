ARG BASE_IMAGE=pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies (gcc needed to compile Polygon3, pyclipper)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    gcc \
    g++ \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Clone Hi-SAM repository
RUN git clone https://github.com/ymy-k/Hi-SAM.git /opt/Hi-SAM

# Install Hi-SAM dependencies
WORKDIR /opt/Hi-SAM
RUN pip install --no-cache-dir -r requirements.txt

# Install FastAPI app dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy application code
COPY app/ /app/app/

# Create model directory
RUN mkdir -p /models

# Copy download script
COPY scripts/ /app/scripts/

ENV HISAM_REPO_PATH=/opt/Hi-SAM
ENV MODEL_DIR=/models

WORKDIR /app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
