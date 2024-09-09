# AWQ + OmniQuant

OmniQuant 使用 可学习的权重截断（Learnable Weight Clipping，`LWC`）和 可学习的等效变换（Learnable Equivalent Transformation，`LET`）来优化量化模型，与非基于学习的算法相比，往往能够获得更好的性能。然而，由于训练过程中的不稳定性以及对超参数的敏感性，OmniQuant 需要大量时间来精细调整超参数，这不仅增加了训练成本，还容易导致训练效果不佳。

为了解决这些问题，在 LLMC 中，我们对 OmniQuant 进行了改进。我们使用 AWQ 生成`截断参数`和`变换参数`，并将其分别作为 OmniQuant 中 `LWC` 和 `LET` 的初始化。事实证明，这种优质的初始化能够大幅缩短 OmniQuant 的训练时间，同时提升其精度表现。


## 1.1 仅权重量化

以 `w4a16g128` 设置为例，我们提供了[AWQ 和 OmniQuant 的组合配置文件](https://github.com/ModelTC/llmc/tree/main/configs/quantization/combination/awq_comb_omni/w4a16g128)。


### 1.1.1 运行AWQ

**第一步**，运行与 AWQ 相关的[配置文件](https://github.com/ModelTC/llmc/tree/main/configs/quantization/combination/awq_comb_omni/w4a16g128/step_1_awq.yml)。请注意，在此步骤中需要将 `save_trans` 参数设置为 `True` 以保存经过变换的模型。

```yaml
# configs/quantization/combination/awq_comb_omni/w4a16g128/step_1_awq.yml

save:
    # Save the AWQ-transformed model for omniquant.
    save_trans: True
    save_fake: False
    save_path: /path/to/save_awq_trans/
```
运行脚本：
```bash
# scripts/run_llmc.sh
llmc=llmc_path
export PYTHONPATH=$llmc:$PYTHONPATH

task_name=step_1_awq
config=${llmc}/configs/quantization/combination/awq_comb_omni/w4a16g128/step_1_awq.yml
```
### 1.1.2 运行OmniQuant

**第二步**，加载AWQ变换过的模型并运行与 OmniQuant 相关的[配置文件](https://github.com/ModelTC/llmc/tree/main/configs/quantization/combination/awq_comb_omni/w4a16g128/step_2_omniq.yml)。
请注意，在此步骤中需要将 `search_clip_init` 参数设置为 `True` 以使用AWQ网格搜索得到`截断参数`初始化`LWC`。

```yaml
# configs/quantization/combination/awq_comb_omni/w4a16g128/step_2_omniq.yml
model:
    type: model_type
    # Load AWQ-transformed model
    path: /path/to/save_awq_trans/transformed_model
    torch_dtype: auto
```
```yaml
quant:
    special:
            search_clip_init: True
```

运行脚本：
```bash
# scripts/run_llmc.sh

llmc=llmc_path
export PYTHONPATH=$llmc:$PYTHONPATH

task_name=step_2_omni
config=${llmc}/configs/quantization/combination/awq_comb_omni/w4a16g128/step_2_omniq.yml
```

通过上述两个步骤的运行，LLMC 在 仅权重量化 的场景下，可以取得比 OmniQuant [原论文](https://arxiv.org/abs/2308.13137)更好的结果，更重要的是，LLMC 仅需 5 个 epoch 就能达到这一效果，远少于[原论文](https://arxiv.org/abs/2308.13137)中所需的 20或者40 个 epoch，大大减少了训练时间。

请注意, 在 仅权重量化 中，AWQ 的`截断参数`和`变换参数`不需要存储以供 OmniQuant 使用，只需保存一个经过变换的模型即可。这是因为 Learnable Equivalent Transformation (`LET`) 主要针对激活量化中的`异常值`（Outlier）现象，因此在仅权重量化中，OmniQuant 无需使用 `LET`。与此同时，使用 AWQ 的`截断参数`来初始化 Learnable Weight Clipping (`LWC`) 会在 LLMC 中的 OmniQuant 代码中自动完成。

## 1.2 权重-激活量化

以 `w8a8` 设置为例，我们提供了[AWQ 和 OmniQuant 的组合配置文件](https://github.com/ModelTC/llmc/tree/main/configs/quantization/combination/awq_comb_omni/w8a8)。


### 1.2.1 运行AWQ

**第一步**，运行与 AWQ 相关的[配置文件](https://github.com/ModelTC/llmc/tree/main/configs/quantization/combination/awq_comb_omni/w8a8/step_1_awq.yml)。请注意，在此步骤中需要将 `save_clip` 和`save_scale`参数设置为 `True` 以保存`截断参数`和`变换参数`。 注意，权重的校准方式要选择 learnable，因为只有 learnable方式得到的`截断参数`支持保存和加载。

```yaml
# configs/quantization/combination/awq_comb_omni/w8a8/step_1_awq.yml
quant:
    weight:
        bit: 8
        symmetric: False
        granularity: per_channel
        group_size: -1
        calib_algo: learnable
    act:
        bit: 8
        symmetric: False
        granularity: per_token
        calib_algo: minmax
```

```yaml
save:
    save_scale: True
    scale_path: /path/to/scale/awq_w8a8.pth
    save_clip: True
    clip_path: /path/to/clip/awq_w8a8.pth
```

运行脚本：
```bash
# scripts/run_llmc.sh
llmc=llmc_path
export PYTHONPATH=$llmc:$PYTHONPATH

task_name=step_1_awq
config=${llmc}/configs/quantization/combination/awq_comb_omni/w8a8/step_1_awq.yml
```

### 1.2.2 运行OmniQuant

**第二步**：加载 AWQ 生成的`截断参数`和`变换参数`, 在此步骤中，加载 AWQ 产出的`截断参数`和`变换参数`，以供 OmniQuant 中的 `LWC` 和 `LET` 进行初始化训练。运行与 OmniQuant 相关的[配置文件](https://github.com/ModelTC/llmc/tree/main/configs/quantization/combination/awq_comb_omni/w8a8/step_2_omniq.yml)。

```yaml
# configs/quantization/combination/awq_comb_omni/w8a8/step_2_omniq.yml
quant:
    special:
       # Use AWQ's search clip factors to initialize OmniQuant's clip factors, 
        # Then refine them through learning (LWC). 
        search_clip_init: True
        load_clip: True
        clip_path: /path/to/scale/awq_w8a8.pth
        # Use AWQ's search scale factors to initialize OmniQuant's scale factors, 
        # Then refine them through learning (LET).
        search_scale_init: True
        scale_path: /path/to/clip/awq_w8a8.pth
```

请注意，在此步骤中需要将 `search_scale_init` 和 `search_clip_init` 参数设置为 `True`，以使用 AWQ 网格搜索得到的 `截断参数` 和 `变换参数` 初始化 `LWC` 和 `LET`。

运行脚本：
```bash
# scripts/run_llmc.sh
llmc=llmc_path
export PYTHONPATH=$llmc:$PYTHONPATH

task_name=step_2_omniq
config=${llmc}/configs/quantization/combination/awq_comb_omni/w8a8/step_2_omniq.yml
```
通过上述两个步骤的运行，LLMC 在 权重-激活量化 设置下可以取得比 [原论文](https://arxiv.org/abs/2308.13137)中更好的结果，且仅仅需要5个epoch。
