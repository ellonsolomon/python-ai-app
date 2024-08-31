# Stage 1: Build environment
FROM python:3.10.12-slim AS builder

# Set working directory for the application
WORKDIR /usr/src/app

# Install system dependencies (if any)
# RUN apt-get update && apt-get install -y build-essential ...

# Copy requirements file and install dependencies in a virtual environment
COPY requirements.txt .
RUN python -m venv /opt/venv \
    && . /opt/venv/bin/activate \
    && pip install --no-cache-dir --prefer-binary -r requirements.txt \
    && pip install --no-cache-dir --prefer-binary tensorflow==2.16.1 \
    && pip install --no-cache-dir --prefer-binary dash pandas plotly flask numpy scikit-learn

# Stage 2: Final image
FROM python:3.10.12-slim

# Set environment variables for Python
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

# Copy virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Set working directory for the application
WORKDIR /usr/src/app

# Copy application files
COPY . .

# Make the start_services.sh script executable
RUN chmod +x ./start_services.sh

# Expose the necessary ports
EXPOSE 8050

# Use a non-root user
RUN useradd -m appuser
USER appuser

# Define the entry point
ENTRYPOINT ["./start_services.sh"]
