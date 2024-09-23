# AutoAWQ量化推理

[AutoAWQ](https://github.com/casper-hansen/AutoAWQ) 是一个易于使用的 4-bit 权重量化模型的包。与 FP16 相比，**AutoAWQ** 能将模型速度提高 3 倍，并将内存需求减少 3 倍。**AutoAWQ** 实现了激活感知权重量化 (AWQ) 算法，用于对大型语言模型进行量化。

**LLMC** 支持导出 **AutoAWQ** 所需的量化格式，并兼容多种算法，不仅限于 AWQ。相比之下，**AutoAWQ**  仅支持 AWQ 算法，而 **LLMC**  可以通过 GPTQ、AWQ 和 Quarot 等算法导出真实量化模型，供 **AutoAWQ**  直接加载，并使用 **AutoAWQ** 的 GEMM 和 GEMV 内核实现推理加速。


## 1.1 环境准备

要使用 **AutoAWQ** 进行量化推理，首先需要安装并配置 **AutoAWQ** 环境：
```bash
INSTALL_KERNELS=1 pip install git+https://github.com/casper-hansen/AutoAWQ.git
# NOTE: This installs https://github.com/casper-hansen/AutoAWQ_kernels
```

## 1.2 量化格式

在 **AutoAWQ** 的定点整型量化中，支持以下几种常见格式：

- **W4A16**：权重为 int4，激活为 float16；
- **权重 per-channel/group 量化**：按通道或按组进行量化；
- **权重非对称量化**：量化参数包括scale和zero point；

因此，在使用 **LLMC** 进行模型量化时，必须确保权重和激活的比特数设置为 **AutoAWQ**支持的格式。


## 1.3 使用LLMC量化模型


### 1.3.1 校准数据

在本章节中，我们使用**Pileval**和**Wikitext**两个学术数据集作为校准数据，有关于校准数据的下载和预处理请参考[章节](https://llmc-zhcn.readthedocs.io/en/latest/configs.html)。

在实际使用中，建议应使用真实部署场景的数据进行离线量化校准。


### 1.3.2 量化算法的选择


**W4A16**

在 W4A16 的量化设置下，我们建议使用 LLMC 中的 AWQ 算法。

具体实现可以参考 AWQ W4A16 的权重量化 [配置文件](https://github.com/ModelTC/llmc/tree/main/configs/quantization/backend/autoawq/awq_w4a16.yml)

```yaml
# configs/quantization/backend/autoawq/awq_w4a16.yml
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

请注意，此步骤中需要将 `pack_version` 参数设置为 `gemm_pack` 或 `gemv_pack`，它们分别对应将 int4 数据打包成 `torch.int32` 的两种方式，适用于 **AutoAWQ** 中加载 `GEMM` 和 `GEMV` 内核时的不同需求。关于 `GEMM` 和 `GEMV` 的区别，请参阅此[链接](https://github.com/casper-hansen/AutoAWQ/tree/main?tab=readme-ov-file#int4-gemm-vs-int4-gemv-vs-fp16)。


此外，如果 AWQ 无法满足精度需求，还可以尝试其他算法，例如 [GPTQ](https://github.com/ModelTC/llmc/tree/main/configs/quantization/backend/autoawq/gptq_w4a16.yml)。同时，我们建议使用[此章节](https://llmc-zhcn.readthedocs.io/en/latest/practice/awq_omni.html)中介绍的 **AWQ+OmniQuant 组合算法**，以进一步提升精度。我们也提供了相应的[配置文件](https://github.com/ModelTC/llmc/tree/main/configs/quantization/backend/autoawq/w4a16_combin)供参考。




### 1.3.3 真实量化模型导出

```yaml
save:
    save_autoawq: True
    save_path: /path/to/save_for_autoawq_awq_w4/

```
请注意，务必将 `save_autoawq` 设置为 `True`。对于 **W4A16** 的量化设置，LLMC 会将权重打包为 `torch.int32` 形式导出，便于 **AutoAWQ** 直接加载，并且会同时导出量化参数。


### 1.3.4 运行LLMC

修改运行脚本中的配置文件路径并运行：

```bash
# scripts/run_llmc.sh
llmc=llmc_path
export PYTHONPATH=$llmc:$PYTHONPATH

task_name=awq_for_autoawq
config=${llmc}/configs/quantization/backend/autoawq/awq_w4a16.yml
```
等LLMC运行结束后，真实量化的模型就会存储在`save.save_path`路径

## 1.4 使用AutoAWQ推理模型


### 1.4.1 离线推理

我们提供了一个使用 **AutoAWQ** 进行离线推理的[示例](https://github.com/ModelTC/llmc/blob/main/examples/backend/autoawq/infer_with_autoawq.py)。

首先，需要将 **AutoAWQ** 的仓库克隆到本地：

```bash
git clone https://github.com/casper-hansen/AutoAWQ.git
```

接着，将[示例](https://github.com/ModelTC/llmc/blob/main/examples/backend/autoawq/infer_with_autoawq.py)中的 `autoawq_path` 替换为你本地的 **AutoAWQ** 仓库路径，并将[示例](https://github.com/ModelTC/llmc/blob/main/examples/backend/autoawq/infer_with_autoawq.py)中的 `model_path`替换为`save.save_path` 中保存的模型路径。然后运行以下命令即可完成推理：

```bash
cd examples/backend/autoawq

CUDA_VISIBLE_DEVICES=0 python infer_with_autoawq.py
```
