## build docker image:
sudo docker build -f docker/Dockerfile -t forge3d-pipeline:latest .
## run docker:
sudo docker run --name 404 --gpus all \
  -v ./pipeline_service:/workspace/ \
  -p 8095:8095 \
  -p 10006:10006 \
  forge3d-pipeline:latest

sudo docker start -a 404

sudo docker exec 404 cat /var/log/vllm.log

sudo docker run --name test5 --gpus all \
  -v $(pwd)/pipeline_service:/workspace \
  -v $(pwd)/logs:/var/log \
  -p 8095:8095 \
  -p 10006:10006 \
  forge3d-pipeline:latest