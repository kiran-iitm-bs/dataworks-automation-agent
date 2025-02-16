# Use a base image with Python and Node.js
FROM node:20-bookworm-slim AS node-base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PATH="/venv/bin:$PATH"

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-venv \
    npm \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment
RUN python3 -m venv /venv

# Set working directory
WORKDIR /app

# Copy code and requirements
COPY . /app
COPY requirements.txt /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8000

# Command to run the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]