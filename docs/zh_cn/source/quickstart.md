# llmc的安装

```
git clone https://github.com/ModelTC/llmc.git
pip install -r requirements.txt
```

llmc无需安装，使用llmc只需在脚本中添加
```
PYTHONPATH=llmc的下载路径:$PYTHONPATH
```

# 准备模型

llmc目前仅支持hugging face格式的模型。以Qwen2-0.5B为例，可以在[这里](https://huggingface.co/Qwen/Qwen2-0.5B)找到模型。下载方式可以参考[这里](https://zhuanlan.zhihu.com/p/663712983)

大陆地区用户还可以使用[hugging face镜像](https://hf-mirror.com/)

一个简单的下载例子可以参考
```
pip install -U hf-transfer

HF_ENDPOINT=https://hf-mirror.com HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download --resume-download Qwen/Qwen2-0.5B --local-dir Qwen2-0.5B
```

# 下载数据集

llmc需要的数据集可以分为校准数据集和测试数据集。校准数据集可以在[这里](https://github.com/ModelTC/llmc/blob/main/tools/download_calib_dataset.py)下载，测试数据集可以在[这里](https://github.com/ModelTC/llmc/blob/main/tools/download_eval_dataset.py)下载

当然llmc也支持在线下载数据集，只需要在config中的download设置为True即可。

# 设置config

以smoothquant为例，config在[这里](https://github.com/ModelTC/llmc/blob/main/configs/quantization/SmoothQuant/smoothquant_llama_w8a8_fakequant_eval.yml)

```
base:
    seed: &seed 42
model:
    type: Qwen2 # 设置模型名,可支持Llama,Qwen2,Llava,Gemma2等模型
    path: # 设置模型权重路径
    torch_dtype: auto
calib:
    name: pileval
    download: False
    path: # 设置校准数据集路径
    n_samples: 512
    bs: 1
    seq_len: 512
    preproc: pileval_smooth
    seed: *seed
eval:
    eval_pos: [pretrain, transformed, fake_quant]
    name: wikitext2
    download: False
    path: # 设置测试数据集路径
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
    save_trans: True # 设置为True，可以保存下调整之后的权重
    save_path: ./save
```

# 开始运行

做好上面的准备之后，可以通过以下的命令运行
```
PYTHONPATH=llmc的下载路径:$PYTHONPATH \
python -m llmc \
--config configs/quantization/SmoothQuant/smoothquant_llama_w8a8_fakequant_eval.yml
```
llmc在scripts下，也提供了很多的运行[脚本](https://github.com/ModelTC/llmc/tree/main/scripts)供大家参考

```
#!/bin/bash

gpu_id=0 # 设置使用的GPU id
export CUDA_VISIBLE_DEVICES=$gpu_id

llmc= # 设置llmc的下载路径
export PYTHONPATH=$llmc:$PYTHONPATH

task_name=smoothquant_llama_w8a8_fakequant_eval # 设置task_name，用于保存log的文件名

# 选择某个config运行
nohup \
python -m llmc \
--config ../configs/quantization/SmoothQuant/smoothquant_llama_w8a8_fakequant_eval.yml \
> ${task_name}.log 2>&1 &

echo $! > ${task_name}.pid
```

# 常见问题

**<font color=red> 问题1 </font>** 

ValueError: Tokenizer class xxx does not exist or is not currently imported.

**<font color=green> 解决方法 </font>** 

pip install transformers --upgrade

**<font color=red> 问题2 </font>** 

下载数据集卡住，下载不下来

**<font color=green> 解决方法 </font>** 

大陆地区可能需要在vpn环境下才能正常访问hugging face的数据集

**<font color=red> 问题3 </font>** 

如果运行的是一个很大的模型，单卡显存放不下整个模型，那么eval的时候，会爆显存

**<font color=green> 解决方法 </font>** 

使用per block进行推理测试，打开inference_per_block，在不爆显存的前提下，适当提高bs以提高推理速度
```
bs: 10
inference_per_block: True
```

**<font color=red> 问题4 </font>** 

Exception: ./save/transformed_model existed before. Need check.

**<font color=green> 解决方法 </font>** 

保存的路径是一个已经存在的目录，需要换个不存在的保存目录

