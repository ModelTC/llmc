
# AWQ

## 1.1 Weight-only Quantization

AWQ performs well in most weight-only quantization scenarios, but it performs poorly in `low-bit` quantization (especially `2-bit`). This is because AWQ uses a `symmetric strategy` for weight clipping regardless of whether symmetric or asymmetric quantization is used.

In LLMC, we have improved the AWQ method by aligning its `weight clipping` strategy with the `quantization strategy`, such that `asymmetric quantization` uses `asymmetric clipping` and `symmetric quantization` uses `symmetric clipping`, resulting in better performance, especially in low-bit quantization.

### 1.1.1 Algorithm Configuration

The specific implementation can be found in the AWQ weight-only quantization [configuration file](https://github.com/ModelTC/llmc/tree/main/configs/quantization/methods/Awq/awq_w_only.yml).

```yaml
# configs/quantization/methods/Awq/awq_w_only.yml
quant:
    method: Awq
    weight:
        bit: 4
        symmetric: False
        granularity: per_group
        group_size: 128
    special:
        trans: True
        # The options for "trans_version" include "v1" and "v2". 
        # But their results don't differ significantly.
        trans_version: v2
        weight_clip: True 
```

### 1.1.2 Running the Algorithm

Simply modify the configuration file path in the [run script](https://github.com/ModelTC/llmc/tree/main/scripts/run_llmc.sh) and execute:

```bash
# scripts/run_llmc.sh
llmc=llmc_path
export PYTHONPATH=$llmc:$PYTHONPATH

task_name=awq_w4a16
config=${llmc}/configs/quantization/methods/Awq/awq_w_only.yml
```

With this improvement, AWQ-LLMC can achieve better accuracy compared to the [original method](https://github.com/mit-han-lab/llm-awq), especially showing significant improvement in 2-bit quantization.

If `clip_sym` is not specified in `config.quant.special`, its value will default to the same as `config.quant.weight.symmetric`. If you want to reproduce academic-level accuracy, you can add `clip_sym` to the config and set it to `True`:

```yaml
quant:
   special:
        clip_sym: True
```

## 1.2 Weight-Activation Quantization

In addition, unlike the original method, AWQ in LLMC supports weight-activation quantization. Compared to [OS+](https://arxiv.org/abs/2304.09145) and [SmoothQuant](https://arxiv.org/abs/2211.10438), which only support scaling transformations for `ln` and `fc` layers, AWQ provides more options for equivalent transformation locations.

AWQ also uses grid search to find the optimal scaling factor for weight transformations, thus often achieving better results in weight-activation quantization.

### 1.2.1 Algorithm Configuration

The specific implementation can be found in the AWQ weight-activation quantization [configuration file](https://github.com/ModelTC/llmc/tree/main/configs/quantization/methods/Awq/awq_w_a.yml).

```yaml
# configs/quantization/methods/Awq/awq_w_a.yml
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
        # The options for "trans_version" include "v1" and "v2".
        trans_version: v2
        weight_clip: True 
```

Simply modify the configuration file path in the [run script](https://github.com/ModelTC/llmc/tree/main/scripts/run_llmc.sh) and execute:

### 1.2.2 Running the Algorithm

```bash
# scripts/run_llmc.sh
llmc=llmc_path
export PYTHONPATH=$llmc:$PYTHONPATH

task_name=awq_w8a8
config=${llmc}/configs/quantization/methods/Awq/awq_w_a.yml
```

In weight-activation quantization, AWQ-LLMC can achieve better results than algorithms like SmoothQuant.
