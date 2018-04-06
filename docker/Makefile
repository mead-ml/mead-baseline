help:
	@cat Makefile

MODEL_STORE?=/data/model-store
EMBEDDINGS_DIR?=/data/embeddings
CHECKPOINTS_DIR?=/data/model-checkpoints
TASKS_DIR?=/data/tasks
HOST_NETWORK?=default
GPU?=0
DOCKER=GPU=$(GPU) nvidia-docker

build:
	docker build --network $(HOST_NETWORK) -t baseline -f ./Dockerfile ../

bash: build
	${DOCKER} run --rm -it --network ${HOST_NETWORK} -v ${MODEL_STORE}:/data/model-store -v ${EMBEDDINGS}:/data/embeddings -v ${CHECKPOINTS_DIR}:/data/model-checkpoints -v ${TASK_DIR}:/data/tasks baseline
