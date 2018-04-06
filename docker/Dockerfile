ARG TF_VERSION=1.5.0

FROM tensorflow/tensorflow:${TF_VERSION}-gpu-py3
LABEL authors="Dan Pressel, Sagnik Choudhury, Yanjie Zhao, Matt Barta"

RUN apt-get update && \
    apt-get install -y g++ make git vim && \
    pip install visdom pymongo click && \
    pip install http://download.pytorch.org/whl/cu90/torch-0.3.1-cp35-cp35m-linux_x86_64.whl  && \
    pip install torchvision && \
    jupyter nbextension enable --py widgetsnbextension

COPY ./python ./baseline

VOLUME ["/data/embeddings", "/data/model-store", "/data/tasks", "/data/model-checkpoints"]

WORKDIR ./baseline
CMD ["bash"]
