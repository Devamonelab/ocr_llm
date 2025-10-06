# -------- Base --------
    FROM python:3.11-slim AS runtime

    # System deps
    RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl && \
        rm -rf /var/lib/apt/lists/*
    
    WORKDIR /app
    
    # Install Python deps
    COPY requirements.txt .
    RUN pip install --no-cache-dir -r requirements.txt
    
    # Copy app code
    COPY app.py ./app.py
    COPY pipeline.py ./pipeline.py
    
    # Create outputs directory
    RUN mkdir -p /app/outputs
    
    # Env (no secrets in image)
    ENV PYTHONUNBUFFERED=1 \
        UVICORN_WORKERS=2 \
        HOST=0.0.0.0 \
        PORT=8000 \
        LOG_LEVEL=INFO
    
    EXPOSE 8000
    
    # Start API
    CMD ["sh", "-c", "uvicorn app:app --host ${HOST} --port ${PORT} --workers ${UVICORN_WORKERS}"]
    