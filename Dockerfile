# Cross-Modal Satellite Image Retrieval -- API deployment image.
#
# Build:    docker build -t cross-modal-retrieval .
# Run:      docker run -p 8000:8000 cross-modal-retrieval
#           (mount /app/models, /app/data, /app/outputs as volumes for your
#            trained model + datasets; set RETRIEVAL_* env vars to relocate)

FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Core + web dependencies first (better layer caching).
COPY requirements.txt requirements-web.txt ./
RUN pip install -r requirements.txt -r requirements-web.txt

# Optional real-data deps (tifffile/h5py); keep the build working if they fail.
COPY requirements-real-data.txt ./
RUN pip install -r requirements-real-data.txt || true

COPY . .

# Runtime artifacts are mounted volumes, not baked into the image.
VOLUME ["/app/models", "/app/data", "/app/outputs", "/app/embeddings", "/app/faiss", "/app/database"]

EXPOSE 8000

# Override the config via RETRIEVAL_CONFIG / RETRIEVAL_* env vars at runtime.
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
