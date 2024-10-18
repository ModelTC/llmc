#!/bin/bash

gpu_id=0
export CUDA_VISIBLE_DEVICES=$gpu_id

llmc=llmc_path
export PYTHONPATH=$llmc:$PYTHONPATH


task_name=rtn_w8a8_fakequant_eval
echo "${task_name} running..."
python -m llmc --config ../configs/quantization/RTN/rtn_w8a8_fakequant_eval.yml \
> ${task_name}.log 2>&1 


task_name=smoothquant_llama_w8a8_fakequant_eval_general
echo "${task_name} running..."
python -m llmc --config ../configs/quantization/SmoothQuant/smoothquant_llama_w8a8_fakequant_eval_general.yml \
> ${task_name}.log 2>&1 


task_name=osplus_llama_w8a8_fakequant_eval_general
echo "${task_name} running..."
python -m llmc --config ../configs/quantization/OsPlus/osplus_llama_w8a8_fakequant_eval_general.yml \
> ${task_name}.log 2>&1 
