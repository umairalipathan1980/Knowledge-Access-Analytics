# OpenShift-friendly base (Python 3.12, non-root ready)
FROM registry.access.redhat.com/ubi9/python-312:latest
 
# ---- Metadata ----
LABEL name="streamlit-app" \
      vendor="Your Team" \
      version="1.0.0" \
      release="1" \
      summary="Streamlit application for Rahti / OpenShift" \
      description="OpenShift-friendly Streamlit service running without root" \
      io.k8s.display-name="Streamlit App" \
      io.openshift.expose-services="8080:http"
 
# ---- Environment (logging, HOME, caches to /tmp) ----
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/opt/app-root/src \
    STREAMLIT_CONFIG_DIR=/opt/app-root/src/.streamlit \
    XDG_CACHE_HOME=/tmp \
    # Disable GPU/CUDA dependencies
    CUDA_VISIBLE_DEVICES="" \
    FORCE_CPU=1 \
    TORCH_FORCE_CPU=1 \
    # Prevent torch installation
    PIP_FIND_LINKS="" \
    # Disable HuggingFace/Transformers caching
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    TOKENIZERS_PARALLELISM=false
 
# Working directory (OpenShift convention)
WORKDIR /opt/app-root/src
 
# ---- Dependencies first ----
COPY requirements.txt constraints.txt ./
# Uncomment if packages need build tools (gcc, headers):
# USER 0
# RUN microdnf install -y gcc gcc-c++ make python3-devel && microdnf clean all
# USER 1001
# Install with constraints to block heavy ML packages
RUN python -m pip install --upgrade pip \
&& pip install --no-cache-dir --constraint constraints.txt -r requirements.txt
 
# ---- Application code ----
# Copy with correct ownership (uid=1001, gid=0)
COPY --chown=1001:0 . .
 
# ---- Writable dirs & permissions ----
# Create runtime directories that don't exist in source code
# .streamlit/ is needed for Streamlit runtime files
# Add mkdir only for: empty dirs, runtime dirs, or dirs in .dockerignore
RUN mkdir -p /opt/app-root/src/.streamlit \
&& chmod -R g=u /opt/app-root
 
# ---- Port (documentation) ----
EXPOSE 8080
 
# ---- Entrypoint ----
USER 1001
 
CMD ["streamlit", "run", "app.py", \
     "--server.port=8080", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.fileWatcherType=none"]