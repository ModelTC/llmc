# Installation of llmc

```
git clone https://github.com/ModelTC/llmc.git
pip install -r requirements.txt
```

llmc does not need to be installed. To use llmc you only need to add this to the script.
```
PYTHONPATH=[llmc's save path]:$PYTHONPATH
```

# Prepare the model

Currently, llmc only supports models in the Hugging Face format. In the case of Qwen2-0.5B, the model can be found [here](https://huggingface.co/Qwen/Qwen2-0.5B). 

A simple download example can be used: 
```
pip install -U hf-transfer

HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download --resume-download Qwen/Qwen2-0.5B --local-dir Qwen2-0.5B
```

# Download the datasets

The datasets required by llmc can be divided into calibration datasets and eval datasets. The calibration dataset can be downloaded [here](https://github.com/ModelTC/llmc/blob/main/tools/download_calib_dataset.py), and the eval dataset can be downloaded [here](https://github.com/ModelTC/llmc/blob/main/tools/download_eval_dataset.py).

Of course, llmc also supports online download of datasets, as long as the download in the config is set to True.


# Set Configs

In the case of smoothquant, the config is [here](https://github.com/ModelTC/llmc/blob/main/configs/quantization/SmoothQuant/smoothquant_llama_w8a8_fakequant_eval.yml).

```
base:
    seed: &seed 42
model:
    type: Qwen2 # Set the model name, which can support Llama, Qwen2, Llava, Gemma2 and other models.
    path: # Set model weight path.
    torch_dtype: auto
calib:
    name: pileval
    download: False
    path: # Set calibration dataset path.
    n_samples: 512
    bs: 1
    seq_len: 512
    preproc: pileval_smooth
    seed: *seed
eval:
    eval_pos: [pretrain, transformed, fake_quant]
    name: wikitext2
    download: False
    path: # Set eval dataset path.
    bs: 1
    seq_len: 2048
quant:
    method: SmoothQuant
    weight:
        bit: 8
        symmetric: True
        granularity: per_channel
    act:
        bit: 8
        symmetric: True
        granularity: per_token
save:
    save_trans: True # Set to True to save the adjusted weights.
    save_path: ./save
```

# Start to run

Once you are prepared above, you can run the following commands
```
PYTHONPATH=[llmc's save path]:$PYTHONPATH \
python -m llmc \
--config configs/quantization/SmoothQuant/smoothquant_llama_w8a8_fakequant_eval.yml
```
Under scripts file folder, llmc also provides a lot of running [scripts](https://github.com/ModelTC/llmc/tree/main/scripts) for your reference

```
#!/bin/bash

gpu_id=0 # Set the GPU id used.
export CUDA_VISIBLE_DEVICES=$gpu_id

llmc= # Set the save path of llmc.
export PYTHONPATH=$llmc:$PYTHONPATH

task_name=smoothquant_llama_w8a8_fakequant_eval # Set task_name, the file name used to save the log.

# Select a config to run.
nohup \
python -m llmc \
--config ../configs/quantization/SmoothQuant/smoothquant_llama_w8a8_fakequant_eval.yml \
> ${task_name}.log 2>&1 &

echo $! > ${task_name}.pid
```

# FAQ

**<font color=red> Q1 </font>** 

ValueError: Tokenizer class xxx does not exist or is not currently imported.

**<font color=green> Solution </font>** 

pip install transformers --upgrade

**<font color=red> Q2 </font>** 

If you are running a large model and a single gpu card cannot store the entire model, then the gpu memory will be out during eval.

**<font color=green> Solution </font>** 

Use per block for inference, turn on inference_per_block, and increase bs appropriately to improve inference speed without exploding the gpu memory.
```
bs: 10
inference_per_block: True
```

**<font color=red> Q3 </font>** 

Exception: ./save/transformed_model existed before. Need check.

**<font color=green> Solution </font>** 

The saving path is an existing directory and needs to be changed to a non-existing saving directory.
