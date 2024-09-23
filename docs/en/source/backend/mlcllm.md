
# MLC LLM Quantized Inference

[MLC LLM](https://github.com/mlc-ai/mlc-llm) is a machine learning compiler and high-performance deployment engine specifically designed for large language models. Its mission is to enable everyone to develop, optimize, and deploy AI models natively on their platforms.

**MLC LLM** supports directly loading real quantized models exported by **AutoAWQ**. Since **LLMC** is seamlessly integrated with **AutoAWQ**, **AutoAWQ** acts as a bridge between **LLMC** and **MLC LLM**, greatly simplifying the loading and deployment process of quantized models.

## 1.1 Environment Setup

To perform quantized inference using **MLC LLM**, you first need to install and configure the **MLC LLM** environment. For example, with CUDA 12.2:

```bash
python -m pip install --pre -U -f https://mlc.ai/wheels mlc-llm-nightly-cu122 mlc-ai-nightly-cu122
```

## 1.2 Quantization Format

The quantization format is the same as in [**AutoAWQ**](https://llmc-en.readthedocs.io/en/latest/backend/autoawq.html).

## 1.3 Using LLMC for Model Quantization

### 1.3.1 Calibration Data

In this section, we use **Pileval** and **Wikitext** as calibration datasets. For details on downloading and preprocessing calibration data, please refer to [this section](https://llmc-en.readthedocs.io/en/latest/configs.html).

For actual use, it is recommended to use data from real deployment scenarios for offline quantization calibration.

### 1.3.2 Choosing a Quantization Algorithm

**W4A16**

For W4A16 quantization settings, we recommend using the AWQ algorithm from LLMC.

You can refer to the AWQ W4A16 weight quantization [configuration file](https://github.com/ModelTC/llmc/tree/main/configs/quantization/backend/mlcllm/awq_w4a16.yml):

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

Please note that the `pack_version` parameter needs to be set to `gemm_pack`, which means int4 data is packed into `torch.int32`. **MLC LLM** supports loading integer weights corresponding to **AutoAWQ**'s `GEMM` kernel format.

Additionally, if AWQ does not meet the accuracy requirements, other algorithms can be explored, such as [GPTQ](https://github.com/ModelTC/llmc/tree/main/configs/quantization/backend/mlcllm/gptq_w4a16.yml). We also recommend the **AWQ+OmniQuant combined algorithm** introduced in [this section](https://llmc-en.readthedocs.io/en/latest/practice/awq_omni.html) to further improve accuracy. The corresponding [configuration files](https://github.com/ModelTC/llmc/tree/main/configs/quantization/backend/mlcllm/w4a16_combin) are available for reference.

### 1.3.3 Exporting Real Quantized Models

```yaml
save:
    save_mlcllm: True
    save_path: /path/to/save_for_mlcllm_awq_w4/
```

Make sure to set `save_mlcllm` to `True`. For **W4A16** quantization settings, LLMC will export the weights in `torch.int32` format, making it easy for **MLC LLM** to load, and will also export the quantization parameters.

### 1.3.4 Running LLMC

Modify the configuration file path in the script and run:

```bash
# scripts/run_llmc.sh
llmc=llmc_path
export PYTHONPATH=$llmc:$PYTHONPATH

task_name=awq_for_mlcllm
config=${llmc}/configs/quantization/backend/mlcllm/awq_w4a16.yml
```

After LLMC finishes running, the real quantized model will be stored in the `save.save_path` directory.

## 1.4 Using MLC LLM for Inference

### 1.4.1 Generate MLC Configuration

The first step is to generate the **MLC LLM** configuration file.

```bash
export LOCAL_MODEL_PATH=/path/to/llama2-7b-chat/   # Local model storage path
export MLC_MODEL_PATH=./dist/llama2-7b-chat-MLC/  # Path for storing the processed MLC model
export QUANTIZATION=q4f16_autoawq            # Quantization option, LLMC currently only supports the q4f16_autoawq format
export CONV_TEMPLATE=llama-2            # Conversation template option

mlc_llm gen_config $LOCAL_MODEL_PATH     --quantization $QUANTIZATION     --conv-template $CONV_TEMPLATE     -o $MLC_MODEL_PATH
```

The configuration generation command takes in the local model path, the target path for **MLC LLM** output, the conversation template name in **MLC LLM**, and the quantization format. Here, the quantization option `q4f16_autoawq` represents using **AutoAWQ**'s `w4a16` quantization format, and the conversation template `llama-2` is the template for the **Llama-2** model in **MLC LLM**.

### 1.4.2 Compile Model Library

Here is an example command to compile the model library in **MLC LLM**:

```bash
export MODEL_LIB=$MLC_MODEL_PATH/lib.so
mlc_llm compile $MLC_MODEL_PATH -o $MODEL_LIB
```

### 1.4.3 Convert Model Weights

In this step, we convert the model weights to **MLC LLM** format.

```bash
export LLMC_MODEL_PATH=/path/to/save_for_mlcllm_awq_w4/ # LLMC-exported real quantized model
mlc_llm convert_weight $LOCAL_MODEL_PATH   --quantization $QUANTIZATION   -o $MLC_MODEL_PATH   --source-format awq   --source $LLMC_MODEL_PATH/mlcllm_quant_model/model.safetensors
```

In the above model conversion process, replace `$LLMC_MODEL_PATH` with `save.save_path`. The `--source-format` parameter indicates that **LLMC** is passing **AutoAWQ** format weights to **MLC LLM**, and `--source` points to the real quantized tensor exported by **LLMC**, which is stored in `save.save_path`. The converted result will be stored in the output path specified by **MLC LLM** using the `-o` option, and can be used for **MLC LLM** inference.

### 1.4.4 Running the MLC LLM Engine

We provide an example of running the **MLC LLM** engine for inference [here](https://github.com/ModelTC/llmc/blob/main/examples/backend/mlcllm/infer_with_mlcllm.py).

Replace the `model_path` in the [example](https://github.com/ModelTC/llmc/blob/main/examples/backend/mlcllm/infer_with_mlcllm.py) with the output path of **MLC LLM**, then run the following command to complete the inference:

```bash
cd examples/backend/mlcllm

python infer_with_mlcllm.py
```
