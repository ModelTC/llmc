# VLLM量化推理

[VLLM](https://github.com/vllm-project/vllm) 是一个专门为满足大规模语言模型推理需求设计的高效后端。它通过优化内存管理和计算效率，能够显著加速推理过程。

LLMC 支持导出 VLLM 所需的量化模型格式，并通过其强大的多算法支持（如 AWQ、GPTQ、QuaRot 等），能够在保证推理速度的同时保持较高的量化精度。通过 LLMC 和 VLLM 的结合，用户可以在不牺牲精度的情况下实现推理加速和内存优化，使其非常适合需要高效处理大规模语言模型的场景



## 1.1 环境准备

要使用 VLLM 进行量化推理，首先需要安装并配置 VLLM 环境：
```bash
pip install vllm
```

## 1.2 量化格式

在 **VLLM** 的定点整型量化中，支持以下几种常见格式：

- **W4A16**：权重为 int4，激活为 float16；
- **W8A16**：权重为 int8，激活为 float16；
- **W8A8**：权重为 int8，激活为 int8；
- **权重 per-channel/group 量化**：按通道或按组进行量化；
- **激活 per-token 动态量化**：针对每个 token 的动态量化方式，进一步提升量化精度和效率。

因此，在使用 **LLMC** 进行模型量化时，必须确保权重和激活的比特数设置为 VLLM 支持的格式。


## 1.3 使用LLMC量化模型


### 1.3.1 校准数据

在本章节中，我们使用**pieval**和**wikitext**两个学术数据集作为校准数据，有关于校准数据的下载和预处理请参考[章节](https://llmc-zhcn.readthedocs.io/en/latest/configs.md)。

在实际使用中，建议应使用真实部署场景的数据进行离线量化校准。


### 1.3.2 量化算法的选择

**W8A16**

在 W8A16 的量化设置下，大语言模型的精度通常不会出现明显问题。在这种情况下，我们建议使用最简单的 RTN（Round to Nearest）算法，该算法不需要额外的校准步骤，运行速度较快。

具体实现可以参考 RTN W8A16 的权重量化 [配置文件](https://github.com/ModelTC/llmc/tree/main/configs/quantization/backend/vllm/rtn_w8a16.yml)

```yaml
# configs/quantization/backend/vllm/rtn_w8a16.yml
quant:
    method: RTN
    weight:
        bit: 8
        symmetric: True
        granularity: per_group
        group_size: 128
        int_range: [-128, 127]
        pack_mode: vllm_pack
```
请注意，在此步骤中需要将 `pack_mode` 参数设置为 `vllm_pack`, 这会将8-bit的权重`打包`为`torch.int32`的格式供VLLM直接加载推理。

**W4A16**

在 W4A16 的量化设置下，RTN（Round to Nearest）不能保证精度无问题，因此需要使用一些高阶量化算法来维持模型的精度。在这种情况下，我们建议使用 LLMC 中的 AWQ 算法.


具体实现可以参考 AWQ W4A16 的权重量化 [配置文件](https://github.com/ModelTC/llmc/tree/main/configs/quantization/backend/vllm/awq_w4a16.yml)

```yaml
# configs/quantization/backend/vllm/awq_w4a16.yml
quant:
    method: Awq
    weight:
        bit: 4
        symmetric: True
        granularity: per_group
        group_size: 128
        int_range: [-8, 7]
        pack_mode: vllm_pack
    special:
        trans: True
        trans_version: v2
        weight_clip: True
    quant_out: True  
```
请注意，在此步骤中需要将 `pack_mode` 参数设置为 `vllm_pack`, 这会将4-bit的权重`打包`为`torch.int32`的格式存储，供VLLM直接加载推理。


此外，如果 AWQ 无法满足精度需求，我们建议使用 [章节](https://llmc-zhcn.readthedocs.io/en/practice/awq_omni.md)介绍的 **AWQ+OmniQuant 组合算法** 来进一步提升精度。在此也给出相应的[配置文件](https://github.com/ModelTC/llmc/tree/main/configs/quantization/backend/vllm/w4a16_combin)


**W8A8**

在 W8A8 的量化设置下，我们同样建议使用 AWQ 算法。AWQ 在大多数情况下的表现优于 SmoothQuant 和 OS+，能够提供更好的量化精度。

具体的实现可以参考 AWQ W8A8 的 [配置文件](https://github.com/ModelTC/llmc/tree/main/configs/quantization/backend/vllm/awq_w8a8.yml)。

```yaml
# configs/quantization/backend/vllm/awq_w8a8.yml
quant:
    method: Awq
    weight:
        bit: 8
        symmetric: True
        granularity: per_channel
        group_size: -1
        int_range: [-128, 127]
    act:
        bit: 8
        symmetric: True
        granularity: per_token
        int_range: [-128, 127]
    special:
        trans: True
        trans_version: v2
        weight_clip: True
    quant_out: True 
```

此外，如果 AWQ 无法满足精度需求，我们建议使用 [章节](https://llmc-zhcn.readthedocs.io/en/practice/quarot_gptq.md) 介绍的 **Quarot+GPTQ 组合算法** 来进一步提升精度。在此也给出相应的[配置文件](https://github.com/ModelTC/llmc/tree/main/configs/quantization/backend/vllm/w8a8_combin)


### 1.3.3 真实量化模型导出

```yaml
save:
    save_vllm: True
    save_path: /path/to/save_for_vllm_rtn_w8a16/
```
请注意，务必将 `save_vllm` 设置为 `True`。对于 **W4A16** 和 **W8A16** 的量化设置，LLMC 会将权重打包为 `torch.int32` 形式导出，便于 VLLM 直接加载，并且会同时导出量化参数。

对于 **W8A8** 的量化设置，LLMC 会将权重量化为 `torch.int8` 形式导出，便于 VLLM 直接加载，同时也会导出相关的量化参数。


### 1.3.4 运行LLMC

修改运行脚本中的配置文件路径并运行：

```bash
# scripts/run_llmc.sh
llmc=llmc_path
export PYTHONPATH=$llmc:$PYTHONPATH

task_name=rtn_for_vllm
config=${llmc}/configs/quantization/backend/vllm/rtn_w8a16.yml
```
等LLMC运行结束后，真实量化的模型就会存储在`save.save_path`路径

## 1.4 使用VLLM推理模型

### 1.4.1 离线推理

我们构建了一个使用 **vLLM** 对数据集进行离线批量推理的[示例](https://github.com/ModelTC/llmc/tree/main/llmc/examples/backend/infer_with_vllm.py)。只需要将 `save.save_path` 路径下保存的模型替换为 [示例](https://github.com/ModelTC/llmc/tree/main/examples/backend/infer_with_vllm.py) 中的 `model_path`，然后运行以下命令即可：

```bash
cd examples/backend

python infer_with_vllm.py
```



### 1.4.1 推理服务

vLLM 可以作为一个实现 OpenAI API 协议的服务器进行部署。这使得 vLLM 可以作为使用 OpenAI API 的应用程序的即插即用替代方案。默认情况下，它会在 http://localhost:8000 启动服务器。你可以通过 --host 和 --port 参数来指定地址。`model_path`替换成保存的`量化模型`即可。

启动服务：

```
vllm serve model_path 
```

调用服务：

```
curl http://localhost:8000/v1/completions \
-H "Content-Type: application/json" \
-d '{
    "model": "model_path",
    "prompt": "What is the AI?",
    "max_tokens": 128,
    "temperature": 0
}'
```