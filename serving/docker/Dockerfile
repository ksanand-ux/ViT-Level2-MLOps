# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy model and code
COPY vit_model.onnx /app/
COPY serving/api /app/

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    fastapi uvicorn[standard] onnxruntime torch torchvision Pillow python-multipart

# Expose port
EXPOSE 8080

# Run FastAPI app
CMD ["uvicorn", "inference:app", "--host", "0.0.0.0", "--port", "8080"]
