FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    CHROMA_PATH=/app/data/chroma \
    ANONYMIZED_TELEMETRY=False \
    PORT=7860

# Hugging Face Spaces runs containers as user 1000
RUN useradd -m -u 1000 user
WORKDIR /app

# CPU-only torch keeps the image ~2GB smaller than the default CUDA build
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY --chown=user requirements.txt .
RUN pip install -r requirements.txt

COPY --chown=user . .
RUN chown -R user /app
USER user

# Download the embedding model and build the vector index into the image,
# so the app starts fast and needs no write access at runtime.
RUN python ingest.py

EXPOSE 7860
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
