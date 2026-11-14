# ---------- stage 1: build the environment ----------
FROM python:3.12-slim AS builder
ENV PIP_NO_CACHE_DIR=1
WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

# ---------- stage 2: small runtime image ----------
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/app/.hf_cache \
    ANONYMIZED_TELEMETRY=False \
    PORT=8501
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv

# Bake the embedding model into the image
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY Correction ./Correction
COPY RAGBase ./RAGBase
COPY RagCore.py WebApp.py ./

EXPOSE 8501
CMD ["sh", "-c", "streamlit run WebApp.py --server.port=${PORT} --server.address=0.0.0.0 --browser.gatherUsageStats=false"]