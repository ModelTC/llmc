export CUDA_VISIBLE_DEVICES=4,5,6,7
llmc=./llmc
lm_eval=./llmc/lm-evaluation-harness
export PYTHONPATH=$llmc:$PYTHONPATH
export PYTHONPATH=$llmc:$lm_eval:$PYTHONPATH
# Replace the config file (i.e., RTN with algorithm-transformed model path or notate quant with original model path) 
# with the one you want to use. `--quarot` is depend on the transformation algorithm used before.
accelerate launch --multi_gpu --num_processes 4 llmc/tools/llm_eval.py \
    --config llmc/configs/quantization/RTN/rtn_quarot.yml \
    --model hf \
    --quarot \
    --tasks lambada_openai,arc_easy \
    --model_args parallelize=False,trust_remote_code=True \
    --batch_size 64 \
    --output_path ./save/lm_eval \
    --log_samples