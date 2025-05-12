
# lightx2v 量化推理

[lightx2v](https://github.com/ModelTC/lightx2v) 是一个专为满足视频生成模型推理需求设计的高效后端。它通过优化内存管理和计算效率，能够显著加速推理过程。

**LLMC** 支持导出 lightx2v 所需的量化模型格式，并通过其对多种量化算法的强大支持（如 AWQ、GPTQ、SmoothQuant 等），能够在保证推理速度的同时保持较高的量化精度。将 **LLMC** 与 **lightx2v** 结合使用，可以在不牺牲精度的前提下实现推理加速和内存优化，非常适合需要高效处理视频生成模型的应用场景。

---

## 1.1 环境准备

要使用 **lightx2v** 进行量化推理，首先需要安装并配置相关环境：

```bash
# 克隆仓库及其子模块
git clone https://github.com/ModelTC/lightx2v.git lightx2v && cd lightx2v
git submodule update --init --recursive

# 创建并激活 conda 环境
conda create -n lightx2v python=3.11 && conda activate lightx2v
pip install -r requirements.txt

# 为避免版本冲突，单独安装 transformers
pip install transformers==4.45.2

# 安装 flash-attention 2
cd lightx2v/3rd/flash-attention && pip install --no-cache-dir -v -e .

# 安装 flash-attention 3（仅在 Hopper 架构下）
cd lightx2v/3rd/flash-attention/hopper && pip install --no-cache-dir -v -e .
```

---

## 1.2 量化格式

**lightx2v** 支持以下几种常见的定点量化格式：

- **W8A8**：权重和激活均为 int8；
- **FP8 (E4M3)**：权重和激活均为 float8；
- **权重 per-channel 量化**；
- **激活 per-token 动态量化**，进一步提升精度；
- **对称量化**（仅使用 scale 参数）。

使用 **LLMC** 进行模型量化时，必须确保权重和激活的比特数符合 **lightx2v** 所支持的格式。

---

## 1.3 使用 LLMC 进行模型量化

### 1.3.1 校准数据

以 Wan2.1 模型在 I2V 任务为例，校准数据示例可在[此目录](https://github.com/ModelTC/llmc/tree/main/assets/wan_i2v/calib)中找到，用户可根据需求添加更多数据。

### 1.3.2 量化算法选择

#### **W8A8**

推荐使用 **SmoothQuant** 算法，配置参考如下 [配置文件](https://github.com/ModelTC/llmc/tree/main/configs/quantization/video_gen/wan_i2v/smoothquant_w_a.yaml)：

```yaml
quant:
  video_gen:
    method: SmoothQuant
    weight:
      bit: 8
      symmetric: True
      granularity: per_channel
    act:
      bit: 8
      symmetric: True
      granularity: per_token
    special:
      alpha: 0.75
```

如果 SmoothQuant 无法满足精度需求，可以尝试使用 **AWQ**，相关配置请参考 [AWQ 配置文件](https://github.com/ModelTC/llmc/tree/main/configs/quantization/video_gen/wan_i2v/awq_w_a.yaml)。

#### **FP8 动态量化**

对于 FP8 格式，LLMC 支持权重 per-channel、激活 per-token 动态量化。推荐仍使用 **SmoothQuant**，参考配置如下：

```yaml
quant:
  video_gen:
    method: SmoothQuant
    weight:
      quant_type: float-quant
      bit: e4m3
      symmetric: True
      granularity: per_channel
      use_qtorch: True
    act:
      quant_type: float-quant
      bit: e4m3
      symmetric: True
      granularity: per_token
      use_qtorch: True
    special:
      alpha: 0.75
```

请确保将 `quant_type` 设置为 `float-quant`，并将 `use_qtorch` 设置为 `True`，因为 LLMC 的浮点量化依赖于 [QPyTorch](https://github.com/Tiiiger/QPyTorch)。

安装 QPyTorch：

```bash
pip install qtorch
```

### 1.3.3 导出真实量化模型

```yaml
save:
  save_lightx2v: True
  save_path: /path/to/save_for_lightx2v/
```

务必将 `save_lightx2v` 设置为 `True`。LLMC 会将权重以 `torch.int8` 或 `torch.float8_e4m3fn` 形式导出，供 lightx2v 直接使用，并附带相应的量化参数。

### 1.3.4 运行 LLMC

编辑运行脚本中的配置路径：

```bash
# scripts/run_llmc.sh
llmc=llmc_path
export PYTHONPATH=$llmc:$PYTHONPATH

task_name=sq_for_lightx2v
config=${llmc}/configs/quantization/video_gen/wan_i2v/smoothquant_w_a.yaml
```

运行完成后，真实量化模型会保存在 `save.save_path` 中。

### 1.3.5 模型评估

以 Wan2.1 在 I2V 任务为例，测试数据在[此目录](https://github.com/ModelTC/llmc/tree/main/assets/wan_i2v/eval)，配置参考如下：

```yaml
eval:
  eval_pos: [fake_quant]
  type: video_gen
  name: i2v
  download: False
  path: ../assets/wan_i2v/eval/
  bs: 1
  target_height: 480
  target_width: 832
  num_frames: 81
  guidance_scale: 5.0
  output_video_path: ./output_videos_sq/
```

LLMC 会生成使用伪量化模型生成的视频结果。

---

## 1.4 使用 lightx2v 进行模型推理

### 1.4.1 权重结构转换

LLMC 导出后，需将模型结构转换为 lightx2v 支持的格式，可使用 [转换脚本](https://github.com/ModelTC/lightx2v/blob/main/examples/diffusers/converter.py)：

```bash
python converter.py -s /path/to/save_for_lightx2v/ -o /path/to/output/ -d backward
```

转换后的模型将保存在 `/path/to/output/`。

### 1.4.2 离线推理

编辑 [推理脚本](https://github.com/ModelTC/lightx2v/blob/main/scripts/run_wan_i2v_advanced_ptq.sh)，设置 `model_path` 为 `/path/to/output/`，`lightx2v_path` 为本地路径，然后运行：

```bash
bash run_wan_i2v_advanced_ptq.sh
```
