#!/bin/bash

current_directory=$(pwd)
llmc=$(echo "$current_directory" | sed 's/\/ci_check$//')
export PYTHONPATH=$llmc:$PYTHONPATH

config=${llmc}/ci_check/gptq_w_only.yml 

nnodes=1
nproc_per_node=1
MASTER_ADDR=127.0.0.1
MASTER_PORT=$((10000 + RANDOM % 20000))

RANDOM=$(python -c 'import uuid; print(uuid.uuid4())')
task_id=$RANDOM

cd ../scripts

torchrun \
    --nnodes $nnodes \
    --nproc_per_node $nproc_per_node \
    --rdzv_id $task_id \
    --rdzv_backend c10d \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    ${llmc}/llmc/__main__.py --config $config --task_id $task_id \
