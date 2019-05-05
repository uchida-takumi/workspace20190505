# THIS_DIR is the directory path of this file.
THIS_DIR=$(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
DOCKER_DIR='/docker_work'

# Define operation
docker-build:
	docker build -t hogefuga:latest .

docker-run-echo:
	docker run --rm -v $(THIS_DIR):$(DOCKER_DIR) \
		hogefuga:latest \
		echo "Docker run from a Makefile operation."

docker-run-bash:
	docker run -it --rm -v $(THIS_DIR):$(DOCKER_DIR) \
		hogefuga:latest \
		/bin/bash
