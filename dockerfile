# Use a base image with Python
FROM python:3.9-slim

# Install the necessary dependencies including OpenGL and X11 libraries
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libx11-dev \
    libxext-dev \
    libxrender-dev \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libfontconfig1 \
    libice6 \
    cmake \
    build-essential \
    --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies listed in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY ./app /app

ENV PYTHONPATH=/app:/app/app

# Expose port 8000 for FastAPI
EXPOSE 8000

# Run the FastAPI application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

