To run the experiments inside a docker container, use the following commands in this directory:

```
docker build --network=host -t baseline -f Dockerfile ../
nvidia-docker run --net=host -v /data:/data:ro -it baseline bash
```

We assume all necessary files (datasests, embeddings etc.) are stored in `/data/` in your machine.

For convenience, we also provide a [script](docker.sh), which can be run as `./docker.sh -g 0 -n <container_name>`, eg. `./docker.sh -g 0 -n test`. The script assumes that following directories exist in your machine:


- /data/embeddings: location for embedding files (W2v, Glove), see [embeddings.json](../python/mead/embeddings.json) in `mead`.  

- /data/datasets: dataset locations (PTB, CoNLL), see [datasets.json](../python/mead/datasets.json) in `mead`.

- /data/model-checkpoints: directory to store _saved_ models (model graph + data)   

- /data/model-store: directory to store _exported_ models.    

