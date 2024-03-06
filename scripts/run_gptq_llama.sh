#!/bin/bash

gpu_id=0
export CUDA_VISIBLE_DEVICES=$gpu_id

llmc=llmc_path
export PYTHONPATH=$llmc:$PYTHONPATH

task_name=llm_quant_exp

nohup \
python -m llmc --config ../configs/quantization/GPTQ/gptq_w4a16_fakequant_eval_general.yml \
> ${task_name}.log 2>&1 &

echo $! > ${task_name}.pid