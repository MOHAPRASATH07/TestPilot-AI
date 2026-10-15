# TestPilot AI Container Specification
FROM python:3.12-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=5000
ENV HOST=0.0.0.0

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source and artifacts
COPY . .

# Expose Flask (5000), FastAPI (8000), and Streamlit (8501)
EXPOSE 5000
EXPOSE 8000
EXPOSE 8501

# Healthcheck
HEALTHCHECK CMD curl --fail http://localhost:5000/prioritize || curl --fail http://localhost:8000/health || exit 1

# Default command launches Flask SaaS application
CMD ["python", "app_flask.py"]
