# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies needed for OpenCV image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Ensure upload, cache, and asset directories exist
RUN mkdir -p static/uploads static/assets

# Expose default port (Hugging Face Spaces default: 7860; Render/Railway inject $PORT)
EXPOSE 7860

# Run uvicorn dynamically binding to $PORT with fallback to 7860
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860}"]
