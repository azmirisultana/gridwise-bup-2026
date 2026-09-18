# Multi-stage or slim build for BUP CSE Fest 2026 GridWise Backend
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    HOST=0.0.0.0

# Install system dependencies (including coinor-cbc solver if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    coinor-cbc \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY main.py .
COPY BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json .
COPY test_solution.py .

# Expose port
EXPOSE 8000

# Run the FastAPI service binding to 0.0.0.0:8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
