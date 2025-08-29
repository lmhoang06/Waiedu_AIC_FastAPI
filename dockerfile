# Use Python 3.13.3 as base image
FROM python:3.13.3-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Install remaining dependencies (except apex due to no cuda)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Return to project root
WORKDIR /app

# Expose the port
EXPOSE 8250

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8250/ || exit 1

# Command to run the FastAPI server
CMD ["python", "main.py"]