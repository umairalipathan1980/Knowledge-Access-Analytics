# OpenShift Deployment Guide

This directory contains the necessary files to deploy the Knowledge Access Analytics application on OpenShift.

## Prerequisites

- OpenShift cluster access
- `oc` CLI tool installed and logged in
- Docker/Podman for building the image

## Deployment Steps

### 1. Build and Push the Docker Image

```bash
# Build the image using the Red Hat UBI base
docker build -t knowledge-access-analytics:latest .

# Tag for your registry (replace with your registry URL)
docker tag knowledge-access-analytics:latest your-registry/knowledge-access-analytics:latest

# Push to registry
docker push your-registry/knowledge-access-analytics:latest
```

### 2. Update Deployment Configuration

Edit `deployment.yaml` to update the image reference:
```yaml
image: your-registry/knowledge-access-analytics:latest
```

### 3. Create OpenShift Resources

```bash
# Navigate to the OpenShift directory
cd openshift

# Create a new project (optional)
oc new-project knowledge-analytics

# Apply all configurations
oc apply -f secret.yaml
oc apply -f pvc.yaml
oc apply -f deployment.yaml
oc apply -f service.yaml
oc apply -f route.yaml
```

### 4. Update API Keys

Update the secret with your actual API keys:

```bash
oc patch secret api-keys -p='{"stringData":{"azure-api-key":"your-actual-azure-key","langchain-api-key":"your-actual-langchain-key"}}'
```

### 5. Access the Application

Get the route URL:
```bash
oc get route knowledge-access-analytics-route
```

## Configuration Files

- `deployment.yaml` - Main application deployment
- `service.yaml` - ClusterIP service for internal access
- `route.yaml` - OpenShift route for external access
- `pvc.yaml` - Persistent volume claim for data storage
- `secret.yaml` - API keys (update with real values)

## Security Features

- Runs as non-root user (UID 1001)
- Security contexts applied
- No privileged escalation
- Secrets managed separately
- TLS termination at route level

## Monitoring

The deployment includes:
- Liveness probe on `/_stcore/health`
- Readiness probe for startup checks
- Resource limits and requests
- Health check endpoint

## Storage

- Persistent volume for knowledge base data
- 10Gi storage allocation (adjust as needed)
- Data persisted across pod restarts