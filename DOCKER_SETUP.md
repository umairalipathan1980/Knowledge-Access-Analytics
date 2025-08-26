# Docker Setup Guide for WSL2

Docker is not currently available in this WSL2 environment. Here's how to set it up:

## Option 1: Docker Desktop (Recommended)

1. **Install Docker Desktop for Windows**:
   - Download from: https://docs.docker.com/desktop/install/windows-install/
   - Install Docker Desktop

2. **Enable WSL2 Integration**:
   - Open Docker Desktop
   - Go to Settings → Resources → WSL Integration
   - Enable integration with your WSL2 distro
   - Click "Apply & Restart"

3. **Verify Installation**:
   ```bash
   docker --version
   docker run hello-world
   ```

## Option 2: Install Docker Engine in WSL2

```bash
# Update packages
sudo apt update

# Install prerequisites
sudo apt install -y apt-transport-https ca-certificates curl gnupg lsb-release

# Add Docker GPG key
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# Add Docker repository
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io

# Start Docker service
sudo service docker start

# Add user to docker group (logout/login required)
sudo usermod -aG docker $USER
```

## Building and Running the Application

Once Docker is set up:

### Method 1: Use the automated script
```bash
./run-docker.sh
```

### Method 2: Manual commands
```bash
# Build the image
docker build -t knowledge-access-analytics:latest .

# Run the container
docker run -d --name knowledge-access-analytics -p 8080:8080 --env-file .env knowledge-access-analytics:latest

# View logs
docker logs -f knowledge-access-analytics

# Access the application
# Open browser to: http://localhost:8080
```

## Important Notes

1. **Environment Variables**: Make sure you have a `.env` file with your API keys:
   ```bash
   cp .env.example .env
   # Edit .env with your actual API keys
   ```

2. **Port Access**: The application will be available at:
   - Local: http://localhost:8080
   - Network: http://0.0.0.0:8080

3. **Data Persistence**: Knowledge bases will be stored inside the container. For persistence across container restarts, consider mounting a volume:
   ```bash
   docker run -d --name knowledge-access-analytics -p 8080:8080 --env-file .env -v $(pwd)/data:/opt/app-root/src/data knowledge-access-analytics:latest
   ```

## Troubleshooting

- **Permission denied**: Make sure your user is in the docker group
- **Port 8080 in use**: Change the port mapping `-p 8081:8080`
- **Build errors**: Check if all files are present and .dockerignore is not excluding required files