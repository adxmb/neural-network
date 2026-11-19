FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir --no-deps .

COPY artifacts/model.npz artifacts/model.npz

RUN useradd --create-home appuser
USER appuser

ENV MODEL_PATH=/app/artifacts/model.npz \
    NUMBA_CACHE_DIR=/tmp/numba
EXPOSE 8000
CMD ["sh", "-c", "uvicorn accent_recognition.app:app --host 0.0.0.0 --port ${PORT:-8000}"]