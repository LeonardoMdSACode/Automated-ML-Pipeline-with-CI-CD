FROM python:3.10-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt requirements-dev.txt ./
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir -r requirements-dev.txt

# Copy repo
COPY . .

# ---- RUN TRAINING PIPELINE ONCE (BUILD TIME) ----
RUN python scripts/bootstrap.py

# HF Spaces port
EXPOSE 7860

# Start API only (NO training here)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
