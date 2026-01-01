#!/bin/bash
# Script to install CUDA 12.8 on Ubuntu 22.04
# Run with: sudo bash install_cuda_12.8.sh

set -e

echo "=========================================="
echo "Installing CUDA 12.8 for Ubuntu 22.04"
echo "=========================================="

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "Please run as root (use sudo)"
    exit 1
fi

# Remove old CUDA keyring if exists
rm -f /etc/apt/sources.list.d/cuda-*.list 2>/dev/null || true

# Download and install CUDA keyring
echo "Step 1: Installing CUDA repository keyring..."
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
rm cuda-keyring_1.1-1_all.deb

# Update package list
echo "Step 2: Updating package list..."
apt-get update

# Install CUDA 12.8
echo "Step 3: Installing CUDA 12.8 toolkit..."
apt-get install -y cuda-toolkit-12-8

# Set up environment variables
echo "Step 4: Setting up environment variables..."
cat >> /etc/profile.d/cuda.sh << 'EOF'
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/compat:$LD_LIBRARY_PATH
EOF

# Create symlink for compatibility
if [ ! -L /usr/local/cuda ]; then
    ln -sf /usr/local/cuda-12.8 /usr/local/cuda
fi

# Verify installation
echo ""
echo "=========================================="
echo "Installation complete!"
echo "=========================================="
echo ""
echo "CUDA 12.8 has been installed to: /usr/local/cuda-12.8"
echo ""
echo "To use CUDA 12.8 in current session, run:"
echo "  export CUDA_HOME=/usr/local/cuda-12.8"
echo "  export PATH=\$CUDA_HOME/bin:\$PATH"
echo "  export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$CUDA_HOME/compat:\$LD_LIBRARY_PATH"
echo ""
echo "Or restart your terminal/shell to load automatically."
echo ""
echo "Verify installation with:"
echo "  /usr/local/cuda-12.8/bin/nvcc --version"
echo ""

