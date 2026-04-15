# ==============================================================================
# Dockerfile — Medical Chatbot
# ==============================================================================
# Builds a production-ready container for the Flask application.
# Uses Gunicorn as the WSGI server for reliable multi-worker serving.
# ==============================================================================

# --- Base image ---
FROM python:3.10-slim

# --- Metadata ---
LABEL maintainer="ayushsharma"
LABEL description="RAG-based Medical Chatbot"

# --- Set working directory ---
WORKDIR /app

# --- Install system dependencies ---
# (gcc is needed by some Python packages with C extensions)
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# --- Install Python dependencies ---
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- Copy application code ---
COPY . .

# --- Expose the Flask port ---
EXPOSE 5000

# --- Run with Gunicorn (production WSGI server) ---
# 4 worker processes, bind to all interfaces on port 5000,
# 120s timeout (embedding model download on first request can be slow).
CMD ["gunicorn", \
     "--bind", "0.0.0.0:5000", \
     "--workers", "4", \
     "--timeout", "120", \
     "app:app"]
