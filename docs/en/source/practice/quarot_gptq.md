
# Quarot + GPTQ

## 1.1 Weight-Activation Quantization

QuaRot aims to optimize the quantization performance of large language models by introducing a `rotation matrix` (such as the `Hadamard transform`), enabling efficient weight-activation quantization across all parts of the model (including weights and activations). This technique smooths the distribution of activation values, eliminating `outliers` and simplifying the quantization process.

However, due to the randomness of the rotation matrix used by QuaRot, the results tend to fluctuate. To address this issue, in LLMC, we can adopt the `QuaRot + GPTQ` combination strategy. By applying GPTQ to reconstruct quantized outputs on the weights transformed by QuaRot, we can fine-tune the weights to stabilize and improve the quantization results. (For detailed analysis, see our [paper](https://arxiv.org/abs/2405.06001v2)).

Please note that running QuaRot requires support for the **Hadamard transform kernel**. The installation of this kernel can be referenced in this [repository](https://github.com/spcl/QuaRot).

### 1.1.1 Running Quarot

**Step One**, run the QuaRot-related [configuration file](https://github.com/ModelTC/llmc/tree/main/configs/quantization/combination/quarot_comb_gptq/w8a8/step_1_quarot.yml). Note that in this step, the `save_trans` parameter must be set to `True` to save the transformed model.

```yaml
# configs/quantization/combination/quarot_comb_gptq/w8a8/step_1_quarot.yml

save:
    # Save the QuaRot-transformed model.
    save_trans: True
    save_fake: False
    save_path: /path/to/save_quarot_trans_for_gptq/
```

Run the script:
```bash
# scripts/run_llmc.sh
llmc=llmc_path
export PYTHONPATH=$llmc:$PYTHONPATH

task_name=step_1_quarot
config=${llmc}/configs/quantization/combination/quarot_comb_gptq/w8a8/step_1_quarot.yml
```

### 1.1.2 Running GPTQ

**Step Two**, load the QuaRot-transformed model and run the GPTQ-related [configuration file](https://github.com/ModelTC/llmc/tree/main/configs/quantization/combination/quarot_comb_gptq/w8a8/step_2_gptq.yml).

```yaml
# configs/quantization/combination/quarot_comb_gptq/w8a8/step_2_gptq.yml
model:
    type: Llama
    # Load QuaRot-transformed model
    path: /path/to/save_quarot_trans_for_gptq/transformed_model
    torch_dtype: auto
```

Run the script:
```bash
# scripts/run_llmc.sh

llmc=llmc_path
export PYTHONPATH=$llmc:$PYTHONPATH

task_name=step_2_gptq
config=${llmc}/configs/quantization/combination/quarot_comb_gptq/w8a8/step_2_gptq.yml
```

Please note that both QuaRot and GPTQ have an `online_rotate` option. Be sure to keep this option consistent across both configuration files. This option indicates whether to apply online rotation to activations, which can greatly improve accuracy but may hinder practical deployment. For more details on online rotation, please refer to the [original QuaRot paper](https://arxiv.org/abs/2404.00456).

```yaml
quant:
    special:
        online_rotate: True
```

By following these two steps, LLMC can achieve better results in weight-activation quantization compared to using the QuaRot algorithm alone.
