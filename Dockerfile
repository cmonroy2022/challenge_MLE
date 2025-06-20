# syntax=docker/dockerfile:1.2
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY challenge/ ./challenge/
COPY data/ ./data/

EXPOSE 8080

# ← forma shell: la expansión ${PORT:-8080} funciona
CMD uvicorn challenge.api:app --host 0.0.0.0 --port ${PORT:-8080}