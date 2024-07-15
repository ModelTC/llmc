#!/bin/bash

gpu_id=2
export CUDA_VISIBLE_DEVICES=$gpu_id

llmc=/mnt/nvme1/yongyang/projects/llm/emnlp/tmp/llmc
export PYTHONPATH=$llmc:$PYTHONPATH

task_name=awq_w3a16_v2_2

nohup \
python -m llmc --config ../configs/quantization/Awq/awq_w3a16_v2.yml \
> ${task_name}.log 2>&1 &

echo $! > ${task_name}.pid

