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

xxx

<font color=792ee5> 对服务进行简单测试 </font>

xxx

<font color=792ee5> 起一个量化模型的服务 </font>

xxx


## opencompass评测工具的使用

[opencompass](https://github.com/open-compass/opencompass)官方仓库有着更详细的文档，这里仅给出一个简单快速入门的使用文档


## 常见问题

xxx
