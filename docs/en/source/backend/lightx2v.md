# lightx2v Quantized Inference

[lightx2v](https://github.com/ModelTC/lightx2v) is an efficient backend designed specifically to meet the inference demands of video generation models. By optimizing memory management and computational efficiency, it significantly accelerates the inference process.

**LLMC** supports exporting quantized model formats required by **lightx2v** and offers strong support for multiple quantization algorithms (such as AWQ, GPTQ, SmoothQuant, etc.), maintaining high quantization accuracy while improving inference speed. Combining **LLMC** with **lightx2v** enables accelerated inference and memory optimization without compromising accuracy, making it ideal for scenarios that require efficient video model processing.

---

## 1.1 Environment Setup

To use **lightx2v** for quantized inference, first install and configure the environment:

```bash
# Clone the repository and its submodules
git clone https://github.com/ModelTC/lightx2v.git lightx2v && cd lightx2v
git submodule update --init --recursive

# Create and activate the conda environment
conda create -n lightx2v python=3.11 && conda activate lightx2v
pip install -r requirements.txt

# Reinstall transformers separately to bypass version conflicts
pip install transformers==4.45.2

# Install flash-attention 2
cd lightx2v/3rd/flash-attention && pip install --no-cache-dir -v -e .

# Install flash-attention 3 (only if using Hopper architecture)
cd lightx2v/3rd/flash-attention/hopper && pip install --no-cache-dir -v -e .
```

---

## 1.2 Quantization Formats

**lightx2v** supports several fixed-point quantization formats:

- **W8A8**: int8 for weights and activations.
- **FP8 (E4M3)**: float8 for weights and activations.
- **Weight per-channel quantization**.
- **Activation per-token dynamic quantization** for improved precision.
- **Symmetric quantization** for both weights and activations (uses only scale).

When using **LLMC** to quantize models, ensure the bit-width of weights and activations matches supported **lightx2v** formats.

---

## 1.3 Quantizing Models with LLMC

### 1.3.1 Calibration Data

For example, for the Wan2.1 model on the I2V task, a calibration dataset is provided in the [directory](https://github.com/ModelTC/llmc/tree/main/assets/wan_i2v/calib). Users can add more samples as needed.

### 1.3.2 Choosing Quantization Algorithm

#### **W8A8**

We recommend using **SmoothQuant** for W8A8 settings.  
Refer to the SmoothQuant W8A8 [configuration file](https://github.com/ModelTC/llmc/tree/main/configs/quantization/video_gen/wan_i2v/smoothquant_w_a.yaml):

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

If SmoothQuant does not meet the precision requirement, use **AWQ** for better accuracy. See the corresponding [configuration](https://github.com/ModelTC/llmc/tree/main/configs/quantization/video_gen/wan_i2v/awq_w_a.yaml).

#### **FP8-Dynamic**

LLMC supports FP8 quantization with per-channel weights and per-token dynamic activations. SmoothQuant is again recommended. See the SmoothQuant FP8 [configuration](https://github.com/ModelTC/llmc/tree/main/configs/quantization/backend/lightx2v/fp8/awq_fp8.yml):

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

Ensure `quant_type` is set to `float-quant` and `use_qtorch` to `True`, as **LLMC** uses [QPyTorch](https://github.com/Tiiiger/QPyTorch) for float quantization.

Install QPyTorch with:

```bash
pip install qtorch
```

### 1.3.3 Exporting the Quantized Model

```yaml
save:
  save_lightx2v: True
  save_path: /path/to/save_for_lightx2v/
```

Set `save_lightx2v` to `True`. LLMC will export weights as `torch.int8` or `torch.float8_e4m3fn` for direct loading in **lightx2v**, along with quantization parameters.

### 1.3.4 Running LLMC

Edit the config path in the run script and execute:

```bash
# scripts/run_llmc.sh
llmc=llmc_path
export PYTHONPATH=$llmc:$PYTHONPATH

task_name=sq_for_lightx2v
config=${llmc}/configs/quantization/video_gen/wan_i2v/smoothquant_w_a.yaml
```

After LLMC completes, the quantized model is saved to `save.save_path`.

### 1.3.5 Evaluation

For the I2V task with the Wan2.1 model, an evaluation dataset is provided [here](https://github.com/ModelTC/llmc/tree/main/assets/wan_i2v/eval). Set the following in the config file:

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

LLMC will generate evaluation videos using the pseudo-quantized model.

---

## 1.4 Inference with lightx2v

### 1.4.1 Weight Structure Conversion

After LLMC exports the model, convert its structure to match **lightx2v** requirements using the [conversion script](https://github.com/ModelTC/lightx2v/blob/main/examples/diffusers/converter.py):

```bash
python converter.py -s /path/to/save_for_lightx2v/ -o /path/to/output/ -d backward
```

The converted model will be saved under `/path/to/output/`.

### 1.4.2 Offline Inference

Edit the [inference script](https://github.com/ModelTC/lightx2v/blob/main/scripts/run_wan_i2v_advanced_ptq.sh), set `model_path` to `/path/to/output/` and `lightx2v_path` to your local lightx2v path, then run:

```bash
bash run_wan_i2v_advanced_ptq.sh
```
