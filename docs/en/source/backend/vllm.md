
# VLLM Quantized Inference

[VLLM](https://github.com/vllm-project/vllm) is an efficient backend specifically designed to meet the inference needs of large language models. By optimizing memory management and computational efficiency, it significantly speeds up the inference process.

LLMC supports exporting quantized model formats required by VLLM and, through its strong multi-algorithm support (such as AWQ, GPTQ, QuaRot, etc.), can maintain high quantization accuracy while ensuring inference speed. The combination of LLMC and VLLM enables users to achieve inference acceleration and memory optimization without sacrificing accuracy, making it ideal for scenarios requiring efficient handling of large-scale language models.

## 1.1 Environment Setup

To use VLLM for quantized inference, first, install and configure the VLLM environment:

```bash
pip install vllm
```

## 1.2 Quantization Formats

In **VLLM**'s fixed-point integer quantization, the following common formats are supported:

- **W4A16**: Weights are int4, activations are float16;
- **W8A16**: Weights are int8, activations are float16;
- **W8A8**: Weights are int8, activations are int8;
- **Per-channel/group quantization**: Quantization is applied per channel or per group;
- **Per-token dynamic quantization**: Dynamic quantization per token, which further improves quantization accuracy and efficiency.

Therefore, when quantizing models with **LLMC**, make sure that the bit settings for weights and activations are in formats supported by VLLM.

## 1.3 Using LLMC for Model Quantization

### 1.3.1 Calibration Data

In this chapter, we use the **Pieval** and **Wikitext** academic datasets as calibration data. For downloading and preprocessing calibration data, refer to [this chapter](https://llmc-zhcn.readthedocs.io/en/latest/configs.md).

In practical use, we recommend using real deployment data for offline quantization calibration.

### 1.3.2 Quantization Algorithm Selection

**W8A16**

In the W8A16 quantization setting, large language models typically do not experience significant accuracy degradation. In this case, we recommend using the simple RTN (Round to Nearest) algorithm, which does not require additional calibration steps and runs quickly.

You can refer to the RTN W8A16 weight quantization [configuration file](https://github.com/ModelTC/llmc/tree/main/configs/quantization/backend/vllm/rtn_w8a16.yml).

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

Make sure to set the `pack_mode` parameter to `vllm_pack`, which packs 8-bit weights into `torch.int32` format for direct VLLM loading and inference.

**W4A16**

In the W4A16 quantization setting, RTN (Round to Nearest) cannot ensure accuracy, so higher-order quantization algorithms are needed to maintain model accuracy. In this case, we recommend using the AWQ algorithm from LLMC.

You can refer to the AWQ W4A16 weight quantization [configuration file](https://github.com/ModelTC/llmc/tree/main/configs/quantization/backend/vllm/awq_w4a16.yml).

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

Make sure to set the `pack_mode` parameter to `vllm_pack`, which packs 4-bit weights into `torch.int32` format for direct VLLM loading and inference.

If AWQ cannot meet accuracy requirements, we recommend using the **AWQ + OmniQuant combination algorithm** described in [this chapter](https://llmc-en.readthedocs.io/en/latest/practice/awq_omni.html) to further improve accuracy. The corresponding [configuration file](https://github.com/ModelTC/llmc/tree/main/configs/quantization/backend/vllm/w4a16_combin) is also provided.

**W8A8**

In the W8A8 quantization setting, we also recommend using the AWQ algorithm. AWQ generally outperforms SmoothQuant and OS+ in most cases, providing better quantization accuracy.

You can refer to the AWQ W8A8 quantization [configuration file](https://github.com/ModelTC/llmc/tree/main/configs/quantization/backend/vllm/awq_w8a8.yml).

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

If AWQ cannot meet accuracy requirements, we recommend using the **Quarot + GPTQ combination algorithm** described in [this chapter](https://llmc-en.readthedocs.io/en/latest/practice/quarot_gptq.html) to further improve accuracy. The corresponding [configuration file](https://github.com/ModelTC/llmc/tree/main/configs/quantization/backend/vllm/w8a8_combin) is also provided.

### 1.3.3 Exporting Quantized Models

```yaml
save:
    save_vllm: True
    save_path: /path/to/save_for_vllm_rtn_w8a16/
```

Make sure to set `save_vllm` to `True`. For **W4A16** and **W8A16** quantization settings, LLMC will export the weights in `torch.int32` format for direct VLLM loading, and it will also export the quantization parameters.

For **W8A8** quantization settings, LLMC will export the weights in `torch.int8` format for direct VLLM loading, along with the relevant quantization parameters.

### 1.3.4 Running LLMC

Modify the configuration file path in the run script and execute:

```bash
# scripts/run_llmc.sh
llmc=llmc_path
export PYTHONPATH=$llmc:$PYTHONPATH

task_name=rtn_for_vllm
config=${llmc}/configs/quantization/backend/vllm/rtn_w8a16.yml
```

After LLMC finishes running, the real quantized model will be stored at the `save.save_path`.

## 1.4 Using VLLM for Inference

### 1.4.1 Offline Inference

We provide an [example](https://github.com/ModelTC/llmc/tree/main/llmc/examples/backend/infer_with_vllm.py) for performing offline batch inference on datasets using **vLLM**. Simply replace the model saved at `save.save_path` with the `model_path` in the [example](https://github.com/ModelTC/llmc/tree/main/examples/backend/infer_with_vllm.py) and run the following command:

```bash
cd examples/backend

python infer_with_vllm.py
```

### 1.4.1 Inference Service

vLLM can be deployed as a server that implements the OpenAI API protocol. This allows vLLM to be used as a drop-in replacement for applications using the OpenAI API. By default, it starts the server at http://localhost:8000. You can specify the address with `--host` and `--port` arguments. Replace `model_path` with the saved `quantized model`.

Start the server:

```
vllm serve model_path 
```

Query the server:

```
curl http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{
    "model": "model_path",
    "prompt": "What is the AI?",
    "max_tokens": 128,
    "temperature": 0
}'
```