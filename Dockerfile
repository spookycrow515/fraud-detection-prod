# Use a slim Python runtime
FROM python:3.10-slim

# Install system dependencies needed for libraries like compilation or OpenMP if required
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install dependencies first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the core Python code layers and the cached model binaries
COPY api.py model_io.py ./
COPY fraud_model_*.pkl ./

# Expose the API server port
EXPOSE 8000

# Boot the web server bound to all interfaces
# Change "0.0.0.0" to "8000" in your port flag
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]