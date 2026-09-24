FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        cmake \
        ninja-build \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Piper training (piper1-gpl). Its torch wheel includes CUDA; the GPU is used
# when the container is started with GPU access (see docker-compose.yml).
COPY script/install_piper /tmp/install_piper
RUN /tmp/install_piper python3 /opt/piper1-gpl && rm -rf /root/.cache

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY export_dataset/ ./export_dataset/
COPY prompts/ ./prompts/

ENV PYTHONUNBUFFERED=1 \
    DATA_DIR=/data
VOLUME ["/data"]
EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
