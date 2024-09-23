
# AutoAWQ Quantized Inference

[AutoAWQ](https://github.com/casper-hansen/AutoAWQ) is an easy-to-use package for 4-bit weight quantization models. Compared to FP16, **AutoAWQ** can speed up models by 3 times and reduce memory requirements by 3 times. **AutoAWQ** implements the Activation-aware Weight Quantization (AWQ) algorithm for quantizing large language models.

**LLMC** supports exporting the quantization format required by **AutoAWQ** and is compatible with various algorithms, not limited to AWQ. In contrast, **AutoAWQ** only supports the AWQ algorithm, while **LLMC** can export real quantized models via algorithms like GPTQ, AWQ, and Quarot for **AutoAWQ** to load directly and use **AutoAWQ**'s GEMM and GEMV kernels to achieve inference acceleration.


## 1.1 Environment Setup

To perform quantized inference using **AutoAWQ**, first, you need to install and configure the **AutoAWQ** environment:
```bash
INSTALL_KERNELS=1 pip install git+https://github.com/casper-hansen/AutoAWQ.git
# NOTE: This installs https://github.com/casper-hansen/AutoAWQ_kernels
```

## 1.2 Quantization Formats

In **AutoAWQ**'s fixed-point integer quantization, the following common formats are supported:

- **W4A16**: weights are int4, activations are float16;
- **Weight per-channel/group quantization**: quantization is performed per-channel or per-group;
- **Weight asymmetric quantization**: quantization parameters include scale and zero point;

Therefore, when quantizing models using **LLMC**, make sure that the bit-width for weights and activations is set to a format supported by **AutoAWQ**.


## 1.3 Using LLMC for Model Quantization

### 1.3.1 Calibration Data

In this chapter, we use the **Pileval** and **Wikitext** academic datasets as calibration data. For downloading and preprocessing calibration data, refer to [this chapter](https://llmc-en.readthedocs.io/en/latest/configs.html).

In practical use, we recommend using real deployment data for offline quantization calibration.


### 1.3.2 Choosing a Quantization Algorithm


**W4A16**

Under the W4A16 quantization setting, we recommend using the AWQ algorithm in LLMC.

You can refer to the [AWQ W4A16 weight quantization configuration file](https://github.com/ModelTC/llmc/tree/main/configs/quantization/backend/autoawq/awq_w4a16.yml) for the specific implementation.

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

Please note that in this step, the `pack_version` parameter needs to be set to either `gemm_pack` or `gemv_pack`, which correspond to two ways of packing int4 data into `torch.int32`, suitable for loading `GEMM` and `GEMV` kernels in **AutoAWQ** for different needs. For the difference between `GEMM` and `GEMV`, please refer to this [link](https://github.com/casper-hansen/AutoAWQ/tree/main?tab=readme-ov-file#int4-gemm-vs-int4-gemv-vs-fp16).

Additionally, if AWQ does not meet the precision requirements, other algorithms such as [GPTQ](https://github.com/ModelTC/llmc/tree/main/configs/quantization/backend/autoawq/gptq_w4a16.yml) can also be tried. We also recommend the **AWQ+OmniQuant combination algorithm** introduced in this [section](https://llmc-en.readthedocs.io/en/latest/practice/awq_omni.html) to further improve accuracy. Corresponding [configuration files](https://github.com/ModelTC/llmc/tree/main/configs/quantization/backend/autoawq/w4a16_combin) are provided for reference.


### 1.3.3 Exporting Real Quantized Model

```yaml
save:
    save_autoawq: True
    save_path: /path/to/save_for_autoawq_awq_w4/
```
Make sure to set `save_autoawq` to `True`. For the W4A16 quantization setting, LLMC will export the weights packed into `torch.int32` format for **AutoAWQ** to load directly, along with exporting the quantization parameters.


### 1.3.4 Running LLMC

Modify the configuration file path in the run script and execute:

```bash
# scripts/run_llmc.sh
llmc=llmc_path
export PYTHONPATH=$llmc:$PYTHONPATH

task_name=awq_for_autoawq
config=${llmc}/configs/quantization/backend/autoawq/awq_w4a16.yml
```
Once LLMC finishes running, the real quantized model will be stored in the `save.save_path`.


## 1.4 Using AutoAWQ for Inference


### 1.4.1 Offline Inference

We provide an [example](https://github.com/ModelTC/llmc/blob/main/examples/backend/autoawq/infer_with_autoawq.py) of using **AutoAWQ** for offline inference.

First, clone the **AutoAWQ** repository locally:

```bash
git clone https://github.com/casper-hansen/AutoAWQ.git
```

Next, replace `autoawq_path` in the [example](https://github.com/ModelTC/llmc/blob/main/examples/backend/autoawq/infer_with_autoawq.py) with the local path to your **AutoAWQ** repository, and replace `model_path` in the [example](https://github.com/ModelTC/llmc/blob/main/examples/backend/autoawq/infer_with_autoawq.py) with the path where the model is saved in `save.save_path`. Then run the following command to complete inference:

```bash
cd examples/backend/autoawq

CUDA_VISIBLE_DEVICES=0 python infer_with_autoawq.py
```