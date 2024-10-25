FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN sed -i 's|http://archive.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' /etc/apt/sources.list

RUN apt-get update && \
    apt-get install -y vim wget git cmake build-essential software-properties-common curl libibverbs-dev ca-certificates iproute2 && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.11 python3.11-venv python3.11-dev python3-pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

RUN python -m pip install --upgrade pip

RUN pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

COPY whls/*.whl /app/

RUN pip install --no-cache-dir torch-*.whl torchvision*.whl

COPY requirements/runtime.txt /app/

RUN pip install --no-cache-dir -r runtime.txt

WORKDIR /workspace

COPY fast-hadamard-transform /workspace/fast-hadamard-transform

RUN cd fast-hadamard-transform && pip install --no-cache-dir -v -e .

RUN rm -rf /app
