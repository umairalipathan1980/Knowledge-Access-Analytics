# Use Red Hat Universal Base Image (UBI) with Python 3.11
FROM registry.redhat.io/ubi9/python-311:1-72

# Set environment variables for Streamlit and OpenShift
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8080 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true

# Switch to root to install system dependencies
USER root

# Install system dependencies required for the application
RUN dnf update -y && \
    dnf install -y gcc gcc-c++ && \
    dnf clean all

# Create application directory and set permissions for OpenShift
WORKDIR /opt/app-root/src

# Copy requirements first for better Docker layer caching
COPY requirements.txt ./

# Install Python dependencies as root
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download spaCy model (required by the application)
RUN python -m spacy download en_core_web_sm

# Copy application source code
COPY . ./

# Create data directory for knowledge bases with proper permissions
RUN mkdir -p /opt/app-root/src/data && \
    chown -R 1001:0 /opt/app-root/src && \
    chmod -R g+rwX /opt/app-root/src

# Switch back to non-root user (OpenShift requirement)
USER 1001

# Expose port 8080 (OpenShift default)
EXPOSE 8080

# Health check for container readiness
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/_stcore/health || exit 1

# Run the application
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0", "--server.headless=true", "--browser.gatherUsageStats=false"]