# Python slim base
FROM python:3.11-slim

# System libs your PDF/text code might need (adjust if not needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils curl git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first (better layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the whole app
COPY . .

# FastAPI/Uvicorn defaults
ENV PYTHONUNBUFFERED=1 \
    PORT=8000 \
    GIT_PYTHON_REFRESH=quiet

EXPOSE 8000
CMD ["uvicorn","main:app","--host","0.0.0.0","--port","8000"]
