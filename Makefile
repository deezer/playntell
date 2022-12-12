DOCKER_TAG=playntell
DOCKER_NAME=playntell
DEVICE=0

DOCKER_PARAMS=  -it --rm --name=${DOCKER_NAME} 
DOCKER_PARAMS+= --gpus '"device=${DEVICE}"'

build:
	docker build -t ${DOCKER_TAG} .

run: stop build
	nvidia-docker run -d $(DOCKER_PARAMS) ${DOCKER_TAG}
	docker logs -f ${DOCKER_TAG}

run-bash:
# Mount current directory in the container
	nvidia-docker run $(DOCKER_PARAMS) -v ${PWD}:/workspace -w /workspace ${DOCKER_TAG} /bin/bash

exec:
	nvidia-docker exec -it ${DOCKER_TAG} /bin/bash

stop:
	docker stop ${DOCKER_TAG} || true && docker rm ${DOCKER_TAG} || true

logs:
	docker logs -f ${DOCKER_TAG}