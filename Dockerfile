FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 AS base

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive

RUN sed -i 's|http://archive.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' /etc/apt/sources.list

RUN apt-get update && \
    apt-get install -y vim tmux zip unzip wget git cmake build-essential software-properties-common curl libibverbs-dev ca-certificates iproute2 ffmpeg libsm6 libxext6 && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.11 python3.11-venv python3.11-dev python3-pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

RUN python -m pip install --upgrade pip

RUN pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

RUN pip install packaging ninja opencv-python

# download torch and torchvision wheels into whls filefold
COPY whls/*.whl /app/

RUN pip install --no-cache-dir torch-*.whl torchvision*.whl

WORKDIR /workspace

# download flash-attention source code
COPY flash-attention /workspace/flash-attention

RUN cd flash-attention && pip install --no-cache-dir -v -e .

# download fast-hadamard-transform source code
COPY fast-hadamard-transform /workspace/fast-hadamard-transform

RUN cd fast-hadamard-transform && pip install --no-cache-dir -v -e .

COPY requirements/runtime.txt /app/

RUN pip install --no-cache-dir -r /app/runtime.txt

RUN rm -rf /app
