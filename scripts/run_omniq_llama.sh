#!/bin/bash

gpu_id=0
export CUDA_VISIBLE_DEVICES=$gpu_id

llmc=llmc_path
export PYTHONPATH=$llmc:$PYTHONPATH

task_name=llm_quant_exp

nohup \
python -m llmc --config ../configs/quantization/OmniQuant/omniq_llama_w8a8_fakequant_eval.yml \
> ${task_name}.log 2>&1 &

echo $! > ${task_name}.pid

