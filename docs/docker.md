To run the experiments inside a docker container, the following commands should be run in the [docker directory](../docker).

First, install [docker CE (engine)](https://docs.docker.com/install/linux/docker-ce/ubuntu/) and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker).  If you want to run without sudo, [add yourself to the docker group](`sudo usermod -a -G docker <user> && reboot`).  Otherwise run sudo before each command below:

```
docker build --network=host -t baseline -f Dockerfile ../
nvidia-docker run --net=host -v /data:/data:ro -it baseline bash
```


We assume all necessary files (datasests, embeddings etc.) are stored in `/data/` in your machine.

For convenience, we also provide a [build script](../docker/build_docker.sh), and a [run script](../docker/run_docker.sh) which can be run as `./run_docker.sh -g 0 -n <container_name>`, eg. `./run_docker.sh -g 0 -n test`. The script assumes that following directories exist in your machine:


- `/data/embeddings`: location for embedding files (W2v, Glove), see [embeddings.json](../python/mead/config/embeddings.json) in `mead`.

- `/data/datasets`: dataset locations (PTB, CoNLL), see [datasets.json](../python/mead/config/datasets.json) in `mead`.

- `/data/model-checkpoints`: directory to store _saved_ models (model graph + data)

- `/data/model-store`: directory to store _exported_ models.

