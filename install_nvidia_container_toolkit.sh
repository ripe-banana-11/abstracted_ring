#!/bin/bash
# Script to install NVIDIA Container Toolkit for Docker GPU support
# Run with: sudo bash install_nvidia_container_toolkit.sh

set -e

echo "=========================================="
echo "Installing NVIDIA Container Toolkit"
echo "=========================================="

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "Please run as root (use sudo)"
    exit 1
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed. Please install Docker first."
    echo "Install Docker with: curl -fsSL https://get.docker.com -o get-docker.sh && sudo sh get-docker.sh"
    exit 1
fi

# Check if nvidia-smi works (driver must be installed)
if ! command -v nvidia-smi &> /dev/null; then
    echo "Warning: nvidia-smi not found. Make sure NVIDIA driver is installed first."
    echo "Run: sudo bash install_nvidia_driver.sh"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Detect distribution
if [ -f /etc/os-release ]; then
    . /etc/os-release
    DISTRIBUTION=$ID$VERSION_ID
    echo "Detected distribution: $DISTRIBUTION"
else
    echo "Error: Cannot detect distribution"
    exit 1
fi

# Install prerequisites
echo ""
echo "Step 1: Installing prerequisites..."
apt-get update
apt-get install -y curl gnupg

# Add NVIDIA Container Toolkit repository
echo ""
echo "Step 2: Adding NVIDIA Container Toolkit repository..."
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Update package list
echo ""
echo "Step 3: Updating package list..."
apt-get update

# Install NVIDIA Container Toolkit
echo ""
echo "Step 4: Installing NVIDIA Container Toolkit..."
apt-get install -y nvidia-container-toolkit

# Configure Docker to use NVIDIA runtime
echo ""
echo "Step 5: Configuring Docker..."
nvidia-ctk runtime configure --runtime=docker

# Restart Docker daemon
echo ""
echo "Step 6: Restarting Docker daemon..."
systemctl restart docker

echo ""
echo "=========================================="
echo "Installation complete!"
echo "=========================================="
echo ""
echo "NVIDIA Container Toolkit has been installed and configured."
echo ""
echo "Verify installation with:"
echo "  docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi"
echo ""
echo "If the command above shows GPU information, Docker GPU support is working!"
echo ""

