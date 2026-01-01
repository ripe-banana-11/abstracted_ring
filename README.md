# 3D Generation Pipeline

Automated pipeline for generating 3D models from 2D images.

## Requirements

- **Docker** and **Docker Compose**
- **NVIDIA GPU** with CUDA 12.x support
- At least **80GB VRAM** (61GB+ recommended)

## Installation

### 1. Install GPU drivers, CUDA, and Docker
Run the setup script with root privileges:
```bash
sudo bash set_up_gpu.sh
```

### 2. Reboot the system
A reboot is required to load the new kernel and NVIDIA driver:
```bash
sudo reboot
```

###3. Verify GPU installation

After rebooting, verify that the NVIDIA driver is correctly installed:
```bash
nvidia-smi
```

## Docker (building)
```bash
sudo docker build -f docker/Dockerfile -t forge3d-pipeline:latest .
```

## Run pipeline

Copy `.env.sample` to `.env` and configure if needed

- Start with docker-compose 

```bash
cd docker
sudo docker-compose up -d --build
```

- Start with docker run
```bash
sudo docker run --name 404 --gpus all \
  -v ./pipeline_service:/workspace/ \
  -p 8095:8095 \
  -p 10006:10006 \
  forge3d-pipeline:latest

sudo docker start -a 404

sudo docker exec 404 cat /var/log/vllm.log
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