# OpenShift / Rahti 2 friendly UBI9 Python base image
FROM registry.access.redhat.com/ubi9/python-312:latest
 
LABEL name="streamlit-app" vendor="Your Team" version="1.0.0" \
      summary="Streamlit app for Rahti / OpenShift" \
      io.k8s.display-name="Streamlit App" io.openshift.expose-services="8080:http"
 
# Env for OpenShift compatibility + no CUDA
ENV HOME=/opt/app-root/src \
    STREAMLIT_CONFIG_DIR=/opt/app-root/src/.streamlit \
    XDG_CACHE_HOME=/tmp \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_APP_FILE=app.py \
    TORCH_CUDA_ARCH_LIST="" FORCE_CUDA="0" USE_CUDA=0
 
# Create app dir
WORKDIR /opt/app-root/src
 
# --- Dependencies first (cached layer) ---
COPY requirements.txt .
RUN python -m pip install --upgrade pip \
 && pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

# Create .streamlit directory early
RUN mkdir -p "$STREAMLIT_CONFIG_DIR"
 
# --- App code with proper ownership for OpenShift ---
COPY --chown=1001:0 . .
 
# Only set permissions on specific directories that need it
RUN chmod -R g=u /opt/app-root/src/.streamlit \
 && chmod g=u /opt/app-root/src
 
EXPOSE 8080
 
# Run as non-root (OpenShift may still assign a random UID)
USER 1001
 
CMD ["streamlit", "run", "app.py", \
     "--server.port=8080", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.fileWatcherType=none"]



















# # OpenShift / Rahti 2 friendly UBI9 Python base image
# FROM registry.access.redhat.com/ubi9/python-312:latest
 
# LABEL name="streamlit-app" vendor="Your Team" version="1.0.0" \
#       summary="Streamlit app for Rahti / OpenShift" \
#       io.k8s.display-name="Streamlit App" io.openshift.expose-services="8080:http"
 
# # Env for OpenShift compatibility + no CUDA
# ENV HOME=/opt/app-root/src \
#     STREAMLIT_CONFIG_DIR=/opt/app-root/src/.streamlit \
#     XDG_CACHE_HOME=/tmp \
#     PYTHONUNBUFFERED=1 \
#     STREAMLIT_APP_FILE=app.py \
#     TORCH_CUDA_ARCH_LIST="" FORCE_CUDA="0" USE_CUDA=0
 
# # Create app dir and grant group write (arbitrary UID in GID 0)
# WORKDIR /opt/app-root/src
# RUN mkdir -p /opt/app-root/src && chmod -R g=u /opt/app-root
 
# # --- Dependencies first (as root, system-wide) ---
# COPY requirements.txt .
# RUN python -m pip install --upgrade pip \
#  && pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
#  && pip install --no-cache-dir -r requirements.txt
 
# # --- App code with proper ownership for OpenShift ---
# COPY --chown=1001:0 . .
 
# # Ensure .streamlit exists and is writable
# RUN mkdir -p "$STREAMLIT_CONFIG_DIR" && chmod -R g=u /opt/app-root
 
# EXPOSE 8080
 
# # Run as non-root (OpenShift may still assign a random UID)
# USER 1001
 
# CMD ["streamlit", "run", "app.py", \
#      "--server.port=8080", \
#      "--server.address=0.0.0.0", \
#      "--server.headless=true", \
#      "--server.fileWatcherType=none"]