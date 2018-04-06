help:
	@cat Makefile

MODEL_STORE?=/data/model-store
EMBEDDINGS_DIR?=/data/embeddings
CHECKPOINTS_DIR?=/data/model-checkpoints
DATASETS_DIR?=/data/datasets
HOST_NETWORK?=default
GPU?=0
DOCKER=NV_GPU=$(GPU) nvidia-docker

build:
	docker build --network $(HOST_NETWORK) -t baseline -f ./Dockerfile ../

bash: build
	$(DOCKER) run --rm -it --network ${HOST_NETWORK} -v ${MODEL_STORE}:/data/model-store -v ${EMBEDDINGS}:/data/embeddings -v ${CHECKPOINTS_DIR}:/data/model-checkpoints -v ${DATASETS_DIR}:/data/datasets baseline
