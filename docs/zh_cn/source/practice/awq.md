# AWQ

## 1.1 仅权重量化


AWQ 在仅权重量化（weight-only quantization）的大多数情况下表现出色，但在`低比特`（尤其是 `2-bit`）量化时效果较差。这是因为 AWQ 无论是对称量化还是非对称量化，都采用了 `对称策略` 来截断权重。

在 LLMC 中，我们对 AWQ 方法进行了改进，将其`权重截断`的策略修改为了和`量化策略`保持一致，例如`非对称量化`使用`非对称截断`，`对称量化`使用`对称截断`， 获得了更优的结果，尤其是在低比特量化。

### 1.1.1 算法配置

具体实现可以参考 AWQ 的权重量化 [配置文件](https://github.com/ModelTC/llmc/tree/main/configs/quantization/methods/Awq/awq_w_only.yml)

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

### 1.1.2 算法运行

只需修改 [运行脚本](https://github.com/ModelTC/llmc/tree/main/scripts/run_llmc.sh) 中的配置文件路径，然后执行即可：

运行脚本：
```bash
# scripts/run_llmc.sh
llmc=llmc_path
export PYTHONPATH=$llmc:$PYTHONPATH

task_name=awq_w4a16
config=${llmc}/configs/quantization/methods/Awq/awq_w_only.yml
```

通过这一改进，AWQ-LLMC 可以取得比于 [原始方法](https://github.com/mit-han-lab/llm-awq) 更好的精度表现，尤其在2-bit量化，表现出显著的改善。


如果在 `config.quant.special` 中未指定 `clip_sym`，那么它的取值将与 `config.quant.weight.symmetric` 保持一致。如果想复现学术精度，可以将 `clip_sym` 写到config里并设置为 `True`：

```yaml
quant:
   special:
        clip_sym: True
```


## 1.2 权重-激活量化

此外，与原始方法不同，LLMC 中的 AWQ 还支持权重-激活量化。相比于 OS+ 和 SmoothQuant 仅支持对 `ln` 和 `fc` 层进行缩放变换，AWQ 提供了更多等价变换的位置选择。

同时，AWQ 通过网格搜索（Grid Search）寻找权重变换的最优缩放因子，因此在权重-激活量化方面通常能够取得更优异的效果。

### 1.2.1 算法配置

具体可以参考 AWQ 的权重-激活量化 [配置文件](https://github.com/ModelTC/llmc/tree/main/configs/quantization/methods/Awq/awq_w_a.yml)

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

只需修改 [运行脚本](https://github.com/ModelTC/llmc/tree/main/scripts/run_llmc.sh) 中的配置文件路径，然后执行即可：

### 1.2.2 算法运行

运行脚本：
```bash
# scripts/run_llmc.sh
llmc=llmc_path
export PYTHONPATH=$llmc:$PYTHONPATH

task_name=awq_w8a8
config=${llmc}/configs/quantization/methods/Awq/awq_w_a.yml
```

在权重-激活量化中，AWQ-LLMC 可以取得比SmoothQuant等算法更好的结果
