#!/bin/bash
# Script to remove CUDA 12.4 and use only CUDA 12.8
# Run with: sudo bash remove_cuda_12.4.sh

set -e

echo "=========================================="
echo "Removing CUDA 12.4 and setting CUDA 12.8 as default"
echo "=========================================="

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "Please run as root (use sudo)"
    exit 1
fi

# Step 1: Remove all CUDA 12.4 packages
echo ""
echo "Step 1: Removing CUDA 12.4 packages..."
apt-get remove --purge -y \
    cuda-toolkit-12-4* \
    cuda-libraries-12-4* \
    cuda-command-line-tools-12-4* \
    cuda-compiler-12-4* \
    cuda-crt-12-4* \
    cuda-cudart-12-4* \
    cuda-cudart-dev-12-4* \
    cuda-cuobjdump-12-4* \
    cuda-cupti-12-4* \
    cuda-cupti-dev-12-4* \
    cuda-cuxxfilt-12-4* \
    cuda-documentation-12-4* \
    cuda-driver-dev-12-4* \
    cuda-gdb-12-4* \
    cuda-nsight-12-4* \
    cuda-nsight-compute-12-4* \
    cuda-nsight-systems-12-4* \
    cuda-nvcc-12-4* \
    cuda-nvdisasm-12-4* \
    cuda-nvml-dev-12-4* \
    cuda-nvprof-12-4* \
    cuda-nvprune-12-4* \
    cuda-nvrtc-12-4* \
    cuda-nvrtc-dev-12-4* \
    cuda-nvtx-12-4* \
    cuda-nvvm-12-4* \
    cuda-nvvp-12-4* \
    cuda-opencl-12-4* \
    cuda-opencl-dev-12-4* \
    cuda-profiler-api-12-4* \
    cuda-sanitizer-12-4* \
    cuda-tools-12-4* \
    cuda-visual-tools-12-4* \
    libcufile-12-4* \
    libcusolver-12-4* \
    libcusolver-dev-12-4* \
    2>/dev/null || true

# Remove NCCL packages tied to CUDA 12.4 (if they exist)
apt-get remove --purge -y libnccl2 libnccl-dev 2>/dev/null || true

# Clean up
apt-get autoremove -y
apt-get autoclean

# Step 2: Update alternatives to use only CUDA 12.8
echo ""
echo "Step 2: Updating CUDA alternatives to use 12.8..."
update-alternatives --remove-all cuda 2>/dev/null || true
update-alternatives --install /usr/local/cuda cuda /usr/local/cuda-12.8 100

update-alternatives --remove-all cuda-12 2>/dev/null || true
update-alternatives --install /usr/local/cuda-12 cuda-12 /usr/local/cuda-12.8 100

# Step 3: Remove CUDA 12.4 directory
echo ""
echo "Step 3: Removing /usr/local/cuda-12.4 directory..."
if [ -d "/usr/local/cuda-12.4" ]; then
    rm -rf /usr/local/cuda-12.4
    echo "   Removed /usr/local/cuda-12.4"
else
    echo "   Directory /usr/local/cuda-12.4 not found (already removed)"
fi

# Step 4: Update environment variables in /etc/profile.d/cuda.sh
echo ""
echo "Step 4: Updating environment variables..."
cat > /etc/profile.d/cuda.sh << 'EOF'
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/compat:$LD_LIBRARY_PATH
EOF

# Step 5: Update .bashrc if it has CUDA 12.4 references
if [ -f ~/.bashrc ]; then
    echo ""
    echo "Step 5: Updating ~/.bashrc..."
    # Remove old CUDA 12.4 entries
    sed -i '/CUDA_HOME.*12\.4/d' ~/.bashrc
    sed -i '/cuda-12\.4/d' ~/.bashrc
    
    # Add CUDA 12.8 if not present
    if ! grep -q "CUDA_HOME=/usr/local/cuda-12.8" ~/.bashrc; then
        echo "" >> ~/.bashrc
        echo "# CUDA 12.8 Environment Variables" >> ~/.bashrc
        echo "export CUDA_HOME=/usr/local/cuda-12.8" >> ~/.bashrc
        echo "export PATH=\$CUDA_HOME/bin:\$PATH" >> ~/.bashrc
        echo "export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$CUDA_HOME/compat:\$LD_LIBRARY_PATH" >> ~/.bashrc
    fi
fi

# Step 6: Verify installation
echo ""
echo "=========================================="
echo "Removal complete!"
echo "=========================================="
echo ""
echo "Verifying CUDA 12.8 installation:"
/usr/local/cuda-12.8/bin/nvcc --version
echo ""
echo "Current CUDA default:"
which nvcc
nvcc --version
echo ""
echo "To use CUDA 12.8 in current session, run:"
echo "  export CUDA_HOME=/usr/local/cuda-12.8"
echo "  export PATH=\$CUDA_HOME/bin:\$PATH"
echo "  export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$CUDA_HOME/compat:\$LD_LIBRARY_PATH"
echo ""
echo "Or restart your terminal/shell to load automatically."
echo ""

