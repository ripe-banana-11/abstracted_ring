#!/bin/bash
# Script to install NVIDIA driver on Ubuntu
# Run with: sudo bash install_nvidia_driver.sh

set -e

echo "=========================================="
echo "Installing NVIDIA Driver"
echo "=========================================="

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "Please run as root (use sudo)"
    exit 1
fi

# Detect Ubuntu version
if [ -f /etc/os-release ]; then
    . /etc/os-release
    UBUNTU_VERSION=$VERSION_ID
    echo "Detected Ubuntu version: $UBUNTU_VERSION"
else
    echo "Error: Cannot detect Ubuntu version"
    exit 1
fi

# Method 1: Using ubuntu-drivers (Recommended - easiest method)
echo ""
echo "Method 1: Installing recommended NVIDIA driver using ubuntu-drivers..."
echo "This will automatically detect and install the best driver for your GPU."
echo ""

# Install ubuntu-drivers-common if not present
if ! command -v ubuntu-drivers &> /dev/null; then
    echo "Installing ubuntu-drivers-common..."
    apt-get update
    apt-get install -y ubuntu-drivers-common
fi

# Detect and install recommended driver
echo "Detecting recommended driver for your GPU..."
ubuntu-drivers devices

echo ""
echo "Installing recommended NVIDIA driver..."
ubuntu-drivers autoinstall

# Alternative Method 2: Install specific driver version
# Uncomment and modify if you need a specific version
# echo "Installing NVIDIA driver 535 (or specify your version)..."
# apt-get update
# apt-get install -y nvidia-driver-535

# Alternative Method 3: Using NVIDIA official repository
# Uncomment if you prefer using NVIDIA's official repository
# echo "Setting up NVIDIA official repository..."
# distribution=$(. /etc/os-release;echo $ID$VERSION_ID | sed -e 's/\.//g')
# wget https://developer.download.nvidia.com/compute/cuda/repos/$distribution/x86_64/cuda-keyring_1.1-1_all.deb
# dpkg -i cuda-keyring_1.1-1_all.deb
# apt-get update
# apt-get install -y nvidia-driver-535  # or your preferred version

echo ""
echo "=========================================="
echo "Driver installation complete!"
echo "=========================================="
echo ""
echo "IMPORTANT: You need to REBOOT your system for the driver to work."
echo ""
echo "After reboot, verify installation with:"
echo "  nvidia-smi"
echo ""
echo "If nvidia-smi shows your GPU information, the driver is working correctly."
echo ""

