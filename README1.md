# 3D Generation Pipeline

Automated pipeline for generating 3D models from 2D images.

## Requirements

- **Docker** and **Docker Compose**
- **NVIDIA GPU** with CUDA 12.x support
- At least **80GB VRAM** (61GB+ recommended)

## Installation

### NVIDIA Driver Installation

**Important:** Install the NVIDIA driver before installing CUDA.

#### Quick Installation (Recommended)

Use the provided script to automatically detect and install the recommended driver:

```bash
sudo bash install_nvidia_driver.sh
```

After installation, **reboot your system**:
```bash
sudo reboot
```

After reboot, verify the driver is working:
```bash
nvidia-smi
```

#### Manual Installation Methods

**Method 1: Using ubuntu-drivers (Easiest)**
```bash
sudo apt update
sudo apt install ubuntu-drivers-common
ubuntu-drivers devices  # Shows recommended driver
sudo ubuntu-drivers autoinstall  # Installs recommended driver
sudo reboot
```

**Method 2: Install specific driver version**
```bash
sudo apt update
sudo apt install nvidia-driver-535  # Replace 535 with your desired version
sudo reboot
```

**Method 3: Using NVIDIA official repository**
```bash
# Add NVIDIA repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID | sed -e 's/\.//g')
wget https://developer.download.nvidia.com/compute/cuda/repos/$distribution/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install nvidia-driver-535  # Replace with your desired version
sudo reboot
```

**Verify installation:**
```bash
nvidia-smi  # Should show GPU information
```

### CUDA Installation

After installing the NVIDIA driver, install CUDA 12.8:

```bash
sudo bash install_cuda_12.8.sh
```

### NVIDIA Container Toolkit (Required for Docker GPU Support)

**Important:** You must install NVIDIA Container Toolkit to use `--gpus all` with Docker.

#### Quick Installation

Use the provided script:

```bash
sudo bash install_nvidia_container_toolkit.sh
```

#### Manual Installation

```bash
# Add NVIDIA Container Toolkit repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker
sudo nvidia-ctk runtime configure --runtime=docker

# Restart Docker
sudo systemctl restart docker
```

**Verify Docker GPU support:**
```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

If this shows GPU information, Docker GPU support is working correctly.

### Docker (building)
```bash
docker build -f docker/Dockerfile -t forge3d-pipeline:latest .
```

## Run pipeline

Copy `.env.sample` to `.env` and configure if needed

- Start with docker-compose 

```bash
cd docker
docker-compose up -d --build
```

- Start with docker run
```bash
docker run --name 404 --gpus all \
  -v ./pipeline_service:/workspace/ \
  -p 8095:8095 \
  -p 10006:10006 \
  forge3d-pipeline:latest

docker start -a 404

docker exec 404 cat /var/log/vllm.log
```

- Start with docker run and env file
```bash
docker run --gpus all -p 10006:10006 --env-file .env forge3d-pipeline:latest
```

- Start with docker run and env file and bound directory (Useful for active development)
```bash
docker run --gpus all -v ./pipeline_service:/workspace/pipeline_service -p 10006:10006 --env-file .env forge3d-pipeline:latest
```

## API Usage

**Seed parameter:**
- `seed: 42` - Use specific seed for reproducible results
- `seed: -1` - Auto-generate random seed (default)

### Endpoint 1: File upload (returns binary PLY)

```bash
curl -X POST "http://localhost:10006/generate" \
  -F "prompt_image_file=@cr7.png" \
  -F "seed=42" \
  -o model.ply
```

### Endpoint 2: File upload (returns binary SPZ)

```bash
curl -X POST "http://localhost:10006/generate-spz" \
  -F "prompt_image_file=@image.png" \
  -F "seed=42" \
  -o model.spz
```

### Endpoint 3: Base64 (returns JSON)

```bash
curl -X POST "http://localhost:10006/generate_from_base64" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt_type": "image",
    "prompt_image": "<base64_encoded_image>",
    "seed": 42
  }'
```

### Endpoint 4: Health check (returns JSON)

```bash
curl -X GET "http://localhost:10006/health" \
  -H "Content-Type: application/json" 
```