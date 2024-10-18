#!/bin/bash

current_directory=$(pwd)
llmc=$(echo "$current_directory" | sed 's/\/ci_check$//')
export PYTHONPATH=$llmc:$PYTHONPATH


cd ../scripts

python -m llmc --config ../ci_check/awq_w4a16_fakequant_eval.yml 
