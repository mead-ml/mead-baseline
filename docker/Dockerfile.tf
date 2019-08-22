ARG TF_VERSION=1.12.0

FROM python:3.6
FROM tensorflow/tensorflow:${TF_VERSION}-gpu-py3

RUN apt-get update && \
    apt-get install -y g++ make git vim && \
    pip install --upgrade pip && \ 
    pip install visdom pymongo pyyaml && \
    jupyter nbextension enable --py widgetsnbextension

COPY python /baseline/python
COPY docs /baseline/docs

RUN  cd /baseline/python/ && bash ./install_dev.sh baseline no_test

ADD https://github.com/mead-ml/xpctl/archive/master.tar.gz ./xpctl-master.tar.gz
RUN tar xzf xpctl-master.tar.gz && \
    cd ./xpctl-master/ && \
    pip install -e .

ADD https://github.com/mead-ml/hpctl/archive/master.tar.gz ./hpctl-master.tar.gz
RUN tar xzf hpctl-master.tar.gz && \
    cd ./hpctl-master/ && \
    pip install -e .

VOLUME ["/data/embeddings", "/data/model-store", "/data/datasets", "/data/model-checkpoints"]

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

WORKDIR /baseline/python
CMD ["bash"]
