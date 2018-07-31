To run the experiments inside a docker container, the following commands should be run in the [docker directory](../docker).

First, install [docker CE (engine)](https://docs.docker.com/install/linux/docker-ce/ubuntu/) and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker).  If you want to run without sudo, [add yourself to the docker group](`sudo usermod -a -G docker <user> && reboot`).  Otherwise run sudo before each command below:

```
docker build --network=host -t baseline -f Dockerfile ../
nvidia-docker run --net=host -v /data:/data:ro -it baseline bash
```


We assume all necessary files (datasests, embeddings etc.) are stored in `/data/` in your machine.

For convenience, we also provide a [build script](../docker/build_docker.sh), and a [run script](../docker/run_docker.sh). The usual pipeline is this:

- build a docker image: `./build_docker.sh tf` or `./build_docker.sh pyt` 

- run the container: `./run_docker.sh -g <gpu_number> -n <container_name> -t <image_type>`, eg. `./run_docker.sh -g 0 -n test -t tf`.
