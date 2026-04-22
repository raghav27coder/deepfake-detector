# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Install system dependencies needed for OpenCV and other ML packages
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces require running as a non-root user with UID 1000
RUN useradd -m -u 1000 user

# Switch to the non-root user
USER user

# Set environment variables for the user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory
WORKDIR $HOME/app

# Copy the application code and set ownership to the new user
COPY --chown=user . $HOME/app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Ensure directories exist with the right permissions
RUN mkdir -p uploads && chmod -R 777 uploads
RUN mkdir -p models

# Make port 7860 available (Hugging Face default)
EXPOSE 7860

# Define environment variables
ENV FLASK_APP=app.py
ENV PYTHONUNBUFFERED=1

# Run gunicorn when the container launches, binding to port 7860
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "120", "app:app"]
