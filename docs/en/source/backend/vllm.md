
# VLLM Quantized Inference

[VLLM](https://github.com/vllm-project/vllm) is an efficient backend specifically designed to meet the inference needs of large language models. By optimizing memory management and computational efficiency, it significantly speeds up the inference process.

**LLMC** supports exporting quantized model formats required by **VLLM** and, through its strong multi-algorithm support (such as AWQ, GPTQ, QuaRot, etc.), can maintain high quantization accuracy while ensuring inference speed. The combination of **LLMC** and **VLLM** enables users to achieve inference acceleration and memory optimization without sacrificing accuracy, making it ideal for scenarios requiring efficient handling of large-scale language models.

## 1.1 Environment Setup

To use **VLLM** for quantized inference, first, install and configure the **VLLM** environment:

```bash
pip install vllm
```

## 1.2 Quantization Formats

In **VLLM**'s fixed-point integer quantization, the following common formats are supported:

- **W4A16**: Weights are int4, activations are float16.
- **W8A16**: Weights are int8, activations are float16.
- **W8A8**: Weights are int8, activations are int8.
- **FP8 (E4M3, E5M2)**: Weights are float8, activations are float8.
- **Per-channel/group weight quantization**: Quantization applied per channel or group.
- **Per-tensor weight quantization**: Quantization applied per tensor.
- **Per-token dynamic activation quantization**: Dynamic quantization for each token to further improve precision.
- **Per-tensor static activation quantization**: Static quantization for each tensor to enhance efficiency.
- **Symmetric weight/activation quantization**: Quantization parameters include scale.

Therefore, when quantizing models with **LLMC**, make sure that the bit settings for weights and activations are in formats supported by **VLLM**.

## 1.3 Using LLMC for Model Quantization

### 1.3.1 Calibration Data

In this chapter, we use the **Pileval** and **Wikitext** academic datasets as calibration data. For downloading and preprocessing calibration data, refer to [this chapter](https://llmc-en.readthedocs.io/en/latest/configs.html).

In practical use, we recommend using real deployment data for offline quantization calibration.

### 1.3.2 Choosing a Quantization Algorithm

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
        need_pack: True
```

Make sure to set the `need_pack` parameter to `True`, which packs 8-bit weights into `torch.int32` format for direct **VLLM** loading and inference.

**W4A16**

In the W4A16 quantization setting, RTN (Round to Nearest) cannot ensure accuracy, so higher-order quantization algorithms are needed to maintain model accuracy. In this case, we recommend using the AWQ algorithm from **LLMC**.

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
        need_pack: True
    special:
        trans: True
        trans_version: v2
        weight_clip: True
    quant_out: True  
```

Make sure to set the `need_pack` parameter to `True`, which packs 4-bit weights into `torch.int32` format for direct **VLLM** loading and inference.

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
    act:
        bit: 8
        symmetric: True
        granularity: per_token
    special:
        trans: True
        trans_version: v2
        weight_clip: True
    quant_out: True 
```

If AWQ cannot meet accuracy requirements, we recommend using the **Quarot + GPTQ combination algorithm** described in [this chapter](https://llmc-en.readthedocs.io/en/latest/practice/quarot_gptq.html) to further improve accuracy. The corresponding [configuration file](https://github.com/ModelTC/llmc/tree/main/configs/quantization/backend/vllm/w8a8_combin) is also provided.

**FP8-Dynamic**

In FP8 quantization, **LLMC** supports weight quantization per-channel and activation quantization dynamically per-token. In this case, the RTN (Round to Nearest) algorithm is sufficient. However, we recommend using the AWQ algorithm for better quantization accuracy. For implementation details, refer to the AWQ FP8 [configuration file](https://github.com/ModelTC/llmc/tree/main/configs/quantization/backend/vllm/fp8/awq_fp8.yml).

```yaml
# configs/quantization/backend/vllm/fp8/awq_fp8.yml
quant:
    method: Awq
    quant_type: float_quant
    weight:
        # Support ["e4m3", "e5m2"]
        bit: e4m3
        symmetric: True
        granularity: per_channel
        use_qtorch: True
    act:
        # Support ["e4m3", "e5m2"]
        bit: e4m3
        symmetric: True
        granularity: per_token
        use_qtorch: True
    special:
        trans: True
        trans_version: v2
        weight_clip: True
    quant_out: True
```

Ensure that `quant_type` is set to `float_quant` to indicate floating-point quantization. Additionally, set `use_qtorch` to `True`, as **LLMC**'s FP8 implementation depends on certain functionalities from the [QPyTorch](https://github.com/Tiiiger/QPyTorch) library.

Install [QPyTorch](https://github.com/Tiiiger/QPyTorch) with the following command:

```bash
pip install qtorch
```

**FP8-Static**

In FP8 quantization, **LLMC** also supports weight quantization per-tensor and activation quantization statically per-tensor. In this case, we recommend using the AWQ algorithm while adjusting the activation ranges. Refer to the AWQ FP8 static quantization [configuration file](https://github.com/ModelTC/llmc/tree/main/configs/quantization/backend/vllm/fp8/awq_fp8_static.yml).

```yaml
# configs/quantization/backend/vllm/fp8/awq_fp8_static.yml
quant:
    method: Awq
    quant_type: float-quant
    weight:
        # Support ["e4m3", "e5m2"]
        bit: e4m3
        symmetric: True
        granularity: per_tensor
        use_qtorch: True
    act:
        # Support ["e4m3", "e5m2"]
        bit: e4m3
        symmetric: True
        granularity: per_tensor
        use_qtorch: True
        static: True
```

### 1.3.3 Exporting Real Quantized Model

```yaml
save:
    save_vllm: True
    save_path: /path/to/save_for_vllm_rtn_w8a16/
```

Make sure to set `save_vllm` to `True`. For **W4A16** and **W8A16** quantization settings, **LLMC** will export the weights in `torch.int32` format for direct **VLLM** loading, and it will also export the quantization parameters.

For **W8A8** quantization settings, **LLMC** will export the weights in `torch.int8` format for direct **VLLM** loading, along with the relevant quantization parameters.

### 1.3.4 Running LLMC

Modify the configuration file path in the run script and execute:

```bash
# scripts/run_llmc.sh
llmc=llmc_path
export PYTHONPATH=$llmc:$PYTHONPATH

task_name=rtn_for_vllm
config=${llmc}/configs/quantization/backend/vllm/rtn_w8a16.yml
```

After **LLMC** finishes running, the real quantized model will be stored at the `save.save_path`.

## 1.4 Using VLLM for Inference

### 1.4.1 Offline Inference

We have provided an [example](https://github.com/ModelTC/llmc/blob/main/examples/backend/vllm/infer_with_vllm.py) for performing offline batch inference on a dataset using **VLLM**. You only need to replace the `model_path` in the [example](https://github.com/ModelTC/llmc/blob/main/examples/backend/vllm/infer_with_vllm.py) with the `save.save_path` path, and then run the following command:

```bash
cd examples/backend/vllm

python infer_with_vllm.py
```

### 1.4.2 Inference Service

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
