FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CHROMA_PATH=/app/data/chroma \
    ANONYMIZED_TELEMETRY=False \
    PORT=10000

RUN useradd -m -u 1000 user
WORKDIR /app

COPY --chown=user requirements.txt .
RUN pip install -r requirements.txt

COPY --chown=user . .
RUN chown -R user /app
USER user

# Download the ONNX MiniLM model and build the vector index into the image,
# so the container starts ready to answer.
RUN python ingest.py

EXPOSE 10000
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
