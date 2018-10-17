ARG CUDA_VERSION=9
ARG CUDNN_VERSION=7

FROM nvidia/cuda:${CUDA_VERSION}.0-cudnn${CUDNN_VERSION}-devel

RUN apt-get update
RUN apt-get install -y apt-utils g++ make git vim cython cmake mercurial python3-setuptools python3-dev
RUN easy_install3 pip
RUN alias python=python3
RUN pip install --upgrade pip
RUN pip install cython
RUN BACKEND=cuda pip install git+https://github.com/clab/dynet#egg=dynet
RUN pip install visdom pymongo pyyaml jupyter
RUN jupyter nbextension enable --py widgetsnbextension

COPY python /baseline/python
COPY docs /baseline/docs

RUN echo "alias python=python3" >> ~/.bashrc

RUN  cd /baseline/python/ && bash ./install_dev.sh baseline no_test && bash ./install_dev.sh xpctl no_test

RUN cd /baseline/python/hpctl && pip install -e .[docker]

VOLUME ["/data/embeddings", "/data/model-store", "/data/datasets", "/data/model-checkpoints"]

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

WORKDIR /baseline/python
CMD ["bash"]
