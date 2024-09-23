# MLC LLM量化推理

[MLC LLM](https://github.com/mlc-ai/mlc-llm) 是一个专为大语言模型设计的机器学习编译器和高性能部署引擎。其使命是让每个人都能够在自己的平台上本地开发、优化和部署 AI 模型。

**MLC LLM** 支持直接加载由 **AutoAWQ** 导出的真实量化模型。由于 **LLMC** 与 **AutoAWQ** 已无缝集成，**AutoAWQ** 作为 **LLMC** 与 **MLC LLM** 之间的桥梁，极大简化了量化模型的加载与部署流程。




## 1.1 环境准备

要使用 **MLC LLM** 进行量化推理，首先需要安装并配置 **MLC LLM** 环境，以CUDA 12.2为例：
```bash
python -m pip install --pre -U -f https://mlc.ai/wheels mlc-llm-nightly-cu122 mlc-ai-nightly-cu122
```

## 1.2 量化格式

与 [**AutoAWQ**](https://llmc-zhcn.readthedocs.io/en/latest/backend/autoawq.html) 相同。


## 1.3 使用LLMC量化模型


### 1.3.1 校准数据

在本章节中，我们使用**Pileval**和**Wikitext**两个学术数据集作为校准数据，有关于校准数据的下载和预处理请参考[章节](https://llmc-zhcn.readthedocs.io/en/latest/configs.html)。

在实际使用中，建议应使用真实部署场景的数据进行离线量化校准。


### 1.3.2 量化算法的选择


**W4A16**

在 W4A16 的量化设置下，我们建议使用 LLMC 中的 AWQ 算法。

具体实现可以参考 AWQ W4A16 的权重量化 [配置文件](https://github.com/ModelTC/llmc/tree/main/configs/quantization/backend/mlcllm/awq_w4a16.yml)

```yaml
# configs/quantization/backend/mlcllm/awq_w4a16.yml
quant:
    method: Awq
    weight:
        bit: 4
        symmetric: True
        granularity: per_group
        group_size: 128
        pack_version: gemm_pack
    special:
        trans: True
        trans_version: v2
        weight_clip: True
    quant_out: True  
```

请注意，此步骤中需要将 `pack_version` 参数设置为 `gemm_pack`，它表示将 int4 数据打包成 `torch.int32`。**MLC LLM** 支持加载由 **AutoAWQ** 的 `GEMM` 内核格式对应的整型权重。


此外，如果 AWQ 无法满足精度需求，还可以尝试其他算法，例如 [GPTQ](https://github.com/ModelTC/llmc/tree/main/configs/quantization/backend/mlcllm/gptq_w4a16.yml)。同时，我们建议使用[此章节](https://llmc-zhcn.readthedocs.io/en/latest/practice/awq_omni.html)中介绍的 **AWQ+OmniQuant 组合算法**，以进一步提升精度。我们也提供了相应的[配置文件](https://github.com/ModelTC/llmc/tree/main/configs/quantization/backend/mlcllm/w4a16_combin)供参考。



### 1.3.3 真实量化模型导出

```yaml
save:
    save_mlcllm: True
    save_path: /path/to/save_for_mlcllm_awq_w4/
```
请注意，务必将 `save_mlcllm` 设置为 `True`。对于 **W4A16** 的量化设置，LLMC 会将权重打包为 `torch.int32` 形式导出，便于 **MLC LLM** 直接加载，并且会同时导出量化参数。


### 1.3.4 运行LLMC

修改运行脚本中的配置文件路径并运行：

```bash
# scripts/run_llmc.sh
llmc=llmc_path
export PYTHONPATH=$llmc:$PYTHONPATH

task_name=awq_for_mlcllm
config=${llmc}/configs/quantization/backend/mlcllm/awq_w4a16.yml
```
等LLMC运行结束后，真实量化的模型就会存储在`save.save_path`路径

## 1.4 使用MLC LLM推理模型


### 1.4.1 生成 MLC 配置

第一步是生成 **MLC LLM** 的配置文件。

```bash
export LOCAL_MODEL_PATH=/path/to/llama2-7b-chat/   # 本地模型存放路径
export MLC_MODEL_PATH=./dist/llama2-7b-chat-MLC/  # 处理后模型的 MLC 存放路径
export QUANTIZATION=q4f16_autoawq            # 量化选项, LLMC目前只支持q4f16_autoawq格式的量化
export CONV_TEMPLATE=llama-2            # 对话模板选项


mlc_llm gen_config $LOCAL_MODEL_PATH \
    --quantization $QUANTIZATION \
    --conv-template $CONV_TEMPLATE \
    -o $MLC_MODEL_PATH
```
配置生成命令接收本地模型路径、**MLC LLM** 输出的目标路径、**MLC LLM** 中的对话模板名称以及量化格式。这里的量化选项 `q4f16_autoawq` 表示使用 **AutoAWQ** 中的 `w4a16` 量化格式，而对话模板 `llama-2` 是 **MLC LLM** 中 `Llama-2` 模型的模板。



### 1.4.2 编译模型库

以下是在 **MLC LLM** 中编译模型库的示例命令：

```bash
export MODEL_LIB=$MLC_MODEL_PATH/lib.so
mlc_llm compile $MLC_MODEL_PATH -o $MODEL_LIB
```

### 1.4.3 转换模型权重

在这一步，我们将模型权重转换为 **MLC LLM** 格式。

```bash
export LLMC_MODEL_PATH=/path/to/save_for_mlcllm_awq_w4/ #LLMC导出的真实量化模型
mlc_llm convert_weight $LOCAL_MODEL_PATH \
  --quantization $QUANTIZATION \
  -o $MLC_MODEL_PATH \
  --source-format awq \
  --source $LLMC_MODEL_PATH/mlcllm_quant_model/model.safetensors

```
在上述模型转换过程中，将 `$LLMC_MODEL_PATH` 替换为 `save.save_path`，`--source-format` 表示 **LLMC** 传递给 **MLC LLM** 的是 **AutoAWQ** 格式的权重，而 `--source` 指的是 **LLMC** 导出的真实量化张量，即存储在 `save.save_path` 的权重张量。转换完成后的结果将存放在 **MLC LLM** 使用 `-o` 指定的输出路径，之后即可直接用于 **MLC LLM** 的推理。


### 1.4.4 运行MLC LLM引擎

我们提供了一个运行 **MLC LLM** 引擎进行推理的[示例](https://github.com/ModelTC/llmc/blob/main/examples/backend/mlcllm/infer_with_mlcllm.py)。

将[示例](https://github.com/ModelTC/llmc/blob/main/examples/backend/mlcllm/infer_with_mlcllm.py)中的 `model_path` 替换为**MLC LLM** 的输出路径，然后运行以下命令即可完成推理：

```bash
cd examples/backend/mlcllm

python infer_with_mlcllm.py
```
