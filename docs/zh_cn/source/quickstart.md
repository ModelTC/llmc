# LLMC的安装

```
git clone https://github.com/ModelTC/llmc.git
cd llmc/
pip install -r requirements.txt
```

# 准备模型

**LLMC**目前仅支持`hugging face`格式的模型。以`Qwen2-0.5B`为例，可以在[这里](https://huggingface.co/Qwen/Qwen2-0.5B)找到模型。下载方式可以参考[这里](https://zhuanlan.zhihu.com/p/663712983)

大陆地区用户还可以使用[hugging face镜像](https://hf-mirror.com/)

一个简单的下载例子可以参考
```
pip install -U hf-transfer

HF_ENDPOINT=https://hf-mirror.com HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download --resume-download Qwen/Qwen2-0.5B --local-dir Qwen2-0.5B
```

# 下载数据集

**LLMC**需要的数据集可以分为`校准数据集`和`测试数据集`。`校准数据集`可以在[这里](https://github.com/ModelTC/llmc/blob/main/tools/download_calib_dataset.py)下载，`测试数据`集可以在[这里](https://github.com/ModelTC/llmc/blob/main/tools/download_eval_dataset.py)下载

当然**LLMC**也支持在线下载数据集，只需要在`config`中的`download`设置为True即可。

```yaml
calib:
    name: pileval
    download: True
```

# 设置配置文件

所有的`配置文件`都在[这里](https://github.com/ModelTC/llmc/blob/main/configs/)可以找到，同时关于`配置文件`的说明请参考[此章节](https://llmc-zhcn.readthedocs.io/en/latest/configs.html)
以SmoothQuant为例，`config`在[这里](https://github.com/ModelTC/llmc/blob/main/configs/quantization/methods/SmoothQuant/smoothquant_w_a.yml)

```yaml
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
    save_vllm: True # 当设置为True时，可以保存真实量化的整型模型，并通过VLLM推理引擎进行推理
    save_trans: False # 当设置为True，可以保存下调整之后的浮点权重
    save_path: ./save
```
有关于`save`的更多选项和说明，请参照[此章节](https://llmc-zhcn.readthedocs.io/en/latest/configs.html)


**LLMC**在`configs/quantization/methods`路径下，提供了很多的[算法配置文件](https://github.com/ModelTC/llmc/tree/main/configs/quantization/methods)供大家参考。

# 开始运行

**LLMC**无需安装，只需在[运行脚本](https://github.com/ModelTC/llmc/blob/main/scripts/run_llmc.sh)中将`/path/to/llmc`修改为**LLMC**的`本地路径`即可。
```bash
llmc=/path/to/llmc
export PYTHONPATH=$llmc:$PYTHONPATH
```

根据你想运行的算法，需相应修改[运行脚本](https://github.com/ModelTC/llmc/blob/main/scripts/run_llmc.sh)中的配置路径。例如，`${llmc}/configs/quantization/methods/SmoothQuant/smoothquant_w_a.yml`对应的是 SmoothQuant 量化的配置文件。`task_name`用于指定**LLMC**运行时生成的`日志文件名称`。

```bash
task_name=smooth_w_a
config=${llmc}/configs/quantization/methods/SmoothQuant/smoothquant_w_a.yml
```

当在运行脚本中，修改完相应的LLMC路径和config路径后，运行即可：

```bash
bash run_llmc.sh
```

# 量化推理

假设你在配置文件中指定了保存`真实量化`模型的选项，例如 `save_vllm: True`，那么保存的`真实量化模型`即可直接用于对应的`推理后端`执行，具体可参照[文档](https://llmc-zhcn.readthedocs.io/en/latest)的`量化推理后端`章节。

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

