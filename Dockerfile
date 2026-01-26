# Use official Python slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt requirements-dev.txt ./
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements-dev.txt

# Copy the entire repo
COPY . .

# Expose the port Hugging Face Spaces expects
EXPOSE 7860

# Command to run FastAPI app
# Replace 'main:app' with your FastAPI app instance path if different
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
