ARG TF_VERSION=1.5.0

FROM tensorflow/tensorflow:${TF_VERSION}-gpu-py3

RUN apt-get update && \
    apt-get install -y g++ make git vim && \
    pip install visdom pymongo click && \
    pip install http://download.pytorch.org/whl/cu90/torch-0.3.1-cp35-cp35m-linux_x86_64.whl  && \
    pip install torchvision && \
    jupyter nbextension enable --py widgetsnbextension

VOLUME ["/baseline", "/data/embeddings", "/data/model-store", "/data/datasets", "/data/model-checkpoints"]

ENV PYTHONPATH='$PYTHONPATH:/baseline'

WORKDIR /baseline
CMD ["bash"]
