# 模型精度测试

## 精度测试流程

llmc支持基础的ppl(perplexity，困惑度)评测，但是更多的下游任务评测，llmc本身并不支持。

常见的做法使用评测工具直接对模型进行推理测试，目前已有的评测工具包括但不限于

1. [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)

2. [opencompass](https://github.com/open-compass/opencompass)

但是这种评测方法评测效率不高，我们推荐使用**推理引擎评测工具分离**的方式进行模型精度评测，模型由推理引擎进行推理，并以api的形式serving起来，评测工具对该api进行评测。这种方式有以下的好处：

1. 使用高效的推理引擎进行模型推理，可以加速整个评测进程

2. 将模型的推理和模型的评测分离开，各自负责份内专业的事，代码结构更清晰

3. 使用推理引擎推理模型，更符合实际部署的场景，和模型实际部署的精度更容易对齐

我们在此推荐并介绍使用以下的模型的压缩-部署-评测流程：**llmc压缩-lightllm推理-opencompass评测**

以下是相关工具的链接：

1. llmc，大模型压缩工具，[[github](https://github.com/ModelTC/llmc),[文档](https://llmc-zhcn.readthedocs.io/en/latest/)]

2. lightllm，大模型推理引擎，[[github](https://github.com/ModelTC/lightllm)]

3. opencompass，大模型评测工具，[[github](https://github.com/open-compass/opencompass),[文档](https://opencompass.readthedocs.io/zh-cn/latest/)]


## lightllm推理引擎的使用

[lightllm](https://github.com/ModelTC/llmc)官方仓库有着更详细的文档，这里仅给出一个简单快速入门的使用文档

<font color=792ee5> 起一个float模型的服务 </font>

**安装lightllm**

```
git clone https://github.com/ModelTC/lightllm.git
cd lightllm
pip install -v -e .
```

**起服务**

```
python -m lightllm.server.api_server --model_dir 模型路径            \
                                     --host 0.0.0.0                 \
                                     --port 1030                    \
                                     --nccl_port 2066               \
                                     --max_req_input_len 6144       \
                                     --max_req_total_len 8192       \
                                     --tp 2                         \
                                     --trust_remote_code            \
                                     --max_total_token_num 120000
```

上述命令将在本机的1030端口，起一个2卡的服务

上述命令可以通过tp的数量设置，在tp张卡上进行TensorParallel推理，适用于较大的模型的推理。

上述命令中的max_total_token_num，会影响测试过程中的吞吐性能，可以根据[lightllm文档](https://github.com/ModelTC/lightllm/blob/main/docs/ApiServerArgs.md)，进行设置。只要不爆显存，往往设置越大越好。

如果要在同一个机器上起多个lightllm服务，需要重新设定上面的port和nccl_port，不要有冲突即可。


<font color=792ee5> 对服务进行简单测试 </font>

执行下面的python脚本

```
import requests
import json

url = 'http://localhost:1030/generate'
headers = {'Content-Type': 'application/json'}
data = {
    'inputs': 'What is AI?',
    "parameters": {
        'do_sample': False,
        'ignore_eos': False,
        'max_new_tokens': 128,
    }
}
response = requests.post(url, headers=headers, data=json.dumps(data))
if response.status_code == 200:
    print(response.json())
else:
    print('Error:', response.status_code, response.text)
```

若上述脚本是有正常返回，说明服务正常

<font color=792ee5> 起一个量化模型的服务 </font>

```
python -m lightllm.server.api_server --model_dir 模型路径            \
                                     --host 0.0.0.0                 \
                                     --port 1030                    \
                                     --nccl_port 2066               \
                                     --max_req_input_len 6144       \
                                     --max_req_total_len 8192       \
                                     --tp 2                         \
                                     --trust_remote_code            \
                                     --max_total_token_num 120000   \
                                     --mode triton_w4a16
```

上述命令加了一个`--mode triton_w4a16`，表示使用了w4a16的naive量化

起完服务，同样需要验证一下服务是否正常

上述的命令使用的模型路径是原始预训练的模型，并没有经过llmc调整。可以按照llmc的文档，打开save_trans，保存一个调整之后的模型，然后再运行上述的naive量化服务命令

## opencompass评测工具的使用

[opencompass](https://github.com/open-compass/opencompass)官方仓库有着更详细的文档，这里仅给出一个简单快速入门的使用文档

**安装opencompass**

```
git clone https://github.com/open-compass/opencompass.git
cd opencompass
pip install -v -e .
```

**修改配置文件**

配置文件在[这里](https://github.com/open-compass/opencompass/blob/main/configs/eval_lightllm.py)，这个配置文件是用于opencompass来评测lightllm的api服务的精度，需要注意的是里面的`url`里面的port，要和上述的lightllm的服务port保持一致

评测的数据集选择，需要修改这部分代码

```
with read_base():
    from .summarizers.leaderboard import summarizer
    from .datasets.humaneval.deprecated_humaneval_gen_a82cae import humaneval_datasets
```

上述的代码片段，表示测试humaneval数据集，更多的数据集测试支持，可以查看[这里](https://github.com/open-compass/opencompass/tree/main/configs/datasets)

**数据集下载**

需要根据opencompass的[文档](https://opencompass.readthedocs.io/zh-cn/latest/get_started/installation.html#id2)，最好数据集的准备

**运行精度测试**

修改好上述的配置文件后，即可运行下面的命令
```
python run.py configs/eval_lightllm.py
```
当模型完成推理和指标计算后，我们便可获得模型的评测结果。其中会在当前目录下生成output文件夹，logs子文件夹记录着评测中的日志，最后生成summary子文件会记录所测数据集的精度

## 常见问题

**<font color=red> 问题1 </font>** 

opencompass中的数据集配置文件，同一个数据集有不同的后缀，表示的是什么意思

**<font color=green> 解决方法 </font>** 

不同后缀表示不同的prompt模板，详细的opencompass问题，可以查看opencompass文档

**<font color=red> 问题2 </font>** 

llama模型的humaneval的测试精度过低

**<font color=green> 解决方法 </font>** 

可能需要将opencompass提供的数据集中的humaneval的jsonl文件里面每一条末尾的\n给删除，再重新测试一下

**<font color=red> 问题3 </font>** 

测试速度还是不够快

**<font color=green> 解决方法 </font>** 

可以考虑lightllm起服务时的max_total_token_num参数设置是否合理，过小的设置，会导致测试并发偏低

