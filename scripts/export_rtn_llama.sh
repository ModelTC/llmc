#!/bin/bash

gpu_id=0
export CUDA_VISIBLE_DEVICES=$gpu_id

llmc=llmc_path
export PYTHONPATH=$llmc:$PYTHONPATH

python -m llmc --config ../configs/quantization/RTN/rtn_w4a16.yml 
