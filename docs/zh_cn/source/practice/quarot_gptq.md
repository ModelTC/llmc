# Quarot + GPTQ


## 1.1 权重-激活量化

QuaRot 旨在通过引入 `旋转矩阵`（如 `Hadamard 变换`）来优化大型语言模型的量化性能，使得模型的所有部分（包括权重、激活值）能够实现高效的 权重-激活量化。这种技术通过平滑激活值的分布，消除其中的`异常值`（Outliers），从而简化量化过程。

然而，由于 QuaRot 所使用的旋转矩阵具有随机性，其结果往往波动较大。为了解决这一问题，在 LLMC 中，我们可以采用 `QuaRot + GPTQ` 的组合策略。在施加 QuaRot 旋转后的权重上使用 GPTQ 重建量化输出，通过微调权重使得量化结果更加稳定和优异。(详细的分析见我们的[论文](https://arxiv.org/abs/2405.06001v2))

请注意，运行 QuaRot 需要 **Hadamard 变换 kernel** 的支持。此 kernel 的安装可以参考该 [仓库](https://github.com/spcl/QuaRot)。

### 1.1.1 运行Quarot

**第一步**，运行与 Quarot 相关的[配置文件](https://github.com/ModelTC/llmc/tree/main/configs/quantization/combination/quarot_comb_gptq/w8a8/step_1_quarot.yml)。请注意，在此步骤中需要将 `save_trans` 参数设置为 `True` 以保存经过变换的模型。

```yaml
# configs/quantization/combination/quarot_comb_gptq/w8a8/step_1_quarot.yml

save:
    # Save the Quarot-transformed model.
    save_trans: True
    save_fake: False
    save_path: /path/to/save_quarot_trans_for_gptq/
```
运行脚本：
```bash
# scripts/run_llmc.sh
llmc=llmc_path
export PYTHONPATH=$llmc:$PYTHONPATH

task_name=step_1_quarot
config=${llmc}/configs/quantization/combination/quarot_comb_gptq/w8a8/step_1_quarot.yml
```
### 1.1.2 运行GPTQ

**第二步**，加载 Quarot 变换过的模型并运行与 GPTQ 相关的[配置文件](https://github.com/ModelTC/llmc/tree/main/configs/quantization/combination/quarot_comb_gptq/w8a8/step_2_gptq.yml)。

```yaml
# configs/quantization/combination/quarot_comb_gptq/w8a8/step_2_gptq.yml
model:
    type: Llama
    # Load Quarot-transformed model
    path: /path/to/save_quarot_trans_for_gptq/transformed_model
    torch_dtype: auto
```

运行脚本：
```bash
# scripts/run_llmc.sh

llmc=llmc_path
export PYTHONPATH=$llmc:$PYTHONPATH

task_name=step_2_gptq
config=${llmc}/configs/quantization/combination/quarot_comb_gptq/w8a8/step_2_gptq.yml

```
请注意，在 QuaRot 和 GPTQ 中都有 `online_rotate` 选项，务必确保两个配置文件中的该选项保持一致。该选项表示是否对激活进行在线旋转处理，这对提升精度有很大帮助，但不利于实际部署。有关在线旋转的详细说明，请参考[原 QuaRot 论文](https://arxiv.org/abs/2404.00456)。
```yaml
quant:
    special:
        online_rotate: True
```

通过上述两个步骤的运行，LLMC 在 权重-激活量化 设置下可以取得比单独使用 Quarot 算法更好的结果
