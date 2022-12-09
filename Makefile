DOCKER_TAG=playntellpublic
DOCKER_NAME=playntellpublic
DEVICE=0

DOCKER_PARAMS=  -it --rm --name=${DOCKER_NAME} 
DOCKER_PARAMS+= --gpus '"device=${DEVICE}"'
# Mount current directory in the container
DOCKER_PARAMS+= -v ${PWD}:/workspace -w /workspace
# DOCKER_PARAMS+= -v /data/nfs/analysis/interns/ggabbolini/:/workspace/data/
# Mount /data directory (which contains NFS mounts on the Research VMs)
# DOCKER_PARAMS+= -v /data:/data 

build:
	docker build -t ${DOCKER_TAG} .

run: stop build
	nvidia-docker run -d $(DOCKER_PARAMS) ${DOCKER_TAG}
	# docker run -d $(DOCKER_PARAMS) ${DOCKER_TAG}
	docker logs -f ${DOCKER_TAG}

run-bash: build
	nvidia-docker run $(DOCKER_PARAMS) ${DOCKER_TAG} /bin/bash

exec:
	nvidia-docker exec -it ${DOCKER_TAG} /bin/bash

stop:
	docker stop ${DOCKER_TAG} || true && docker rm ${DOCKER_TAG} || true

logs:
	docker logs -f ${DOCKER_TAG}