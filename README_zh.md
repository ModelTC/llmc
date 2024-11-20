# LLMC: 准确高效的LLM压缩工具

<img src="./imgs/llmc.png" alt="llmc" style="zoom:35%;" />

<div align="center">

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![arXiv](https://img.shields.io/badge/LLMC-2405.06001-b31b1b)](https://arxiv.org/abs/2405.06001)
[![GitHub Stars](https://img.shields.io/github/stars/ModelTC/llmc.svg?style=social&label=Star&maxAge=60)](https://github.com/ModelTC/llmc)
![visitors](https://komarev.com/ghpvc/?username=llmc&label=visitors)
[![Discord Banner](https://img.shields.io/discord/1139835312592392214?logo=discord&logoColor=white)](https://discord.com/invite/NfJzbkK3jY)
[![QQ](https://img.shields.io/badge/QQ-EB1923?logo=tencent-qq&logoColor=white)](http://qm.qq.com/cgi-bin/qm/qr?_wv=1027&k=I9IGPWWj8uuRXWH3_ELWjouf6gkIMgUl&authKey=GA3WbFAsm90ePJf%2FCbc7ZyXXq4ShQktlBaLxgqS5yuSPAsr3%2BDKMRdosUiLYoilO&noverify=0&group_code=526192592)
[![Doc](https://img.shields.io/badge/docs-English-99cc2)](https://llmc-en.readthedocs.io/en/latest/)
[![Doc](https://img.shields.io/badge/文档-中文-99cc2)](https://llmc-zhcn.readthedocs.io/en/latest/)

</div>

**\[ English | [中文](README_zh.md) | [日本語](README_ja.md) \]**

**LLMC** 是一个开箱即用的工具，专为压缩LLM设计，利用最先进的压缩算法提高效率并减少模型体积，同时不影响预测精度。

**英文文档**在[此处](https://llmc-en.readthedocs.io/en/latest/)。

**中文文档**在[此处](https://llmc-zhcn.readthedocs.io/en/latest/)。

**Docker hub**在[此处](https://hub.docker.com/r/llmcompression/llmc)。

**阿里云docker**: `registry.cn-hangzhou.aliyuncs.com/yongyang/llmcompression:[tag]`

你可以通过以下命令下载可以运行llmc的docker镜像，中国大陆用户推荐使用阿里云docker。

docker hub

```
docker pull llmcompression/llmc:pure-latest
```

阿里云docker

```
docker pull registry.cn-hangzhou.aliyuncs.com/yongyang/llmcompression:pure-latest
```

**社区**:

- [Discord 服务器](https://discord.com/invite/NfJzbkK3jY)
- [腾讯QQ群](http://qm.qq.com/cgi-bin/qm/qr?_wv=1027&k=I9IGPWWj8uuRXWH3_ELWjouf6gkIMgUl&authKey=GA3WbFAsm90ePJf%2FCbc7ZyXXq4ShQktlBaLxgqS5yuSPAsr3%2BDKMRdosUiLYoilO&noverify=0&group_code=526192592)

## 最新消息

- **2024年11月20日:** 🔥 我们现已全面支持✨`DeepSeekv2(2.5)`等`MOE`模型以及✨`Qwen2VL`、`Llama3.2`等`VLM`模型的量化。支持的量化方案包括✅整型量化、✅浮点量化，以及✅AWQ、✅GPTQ、✅SmoothQuant 和 ✅Quarot 等先进算法。

- **2024年11月12日:** 🔥 我们新增对各种模型和算法的💥`激活静态 per-tensor量化`支持，涵盖✅整型量化和✅浮点量化，进一步优化性能和效率。同时支持导出`✨真实量化模型`，并使用 [VLLM](https://github.com/vllm-project/vllm)和[SGLang](https://github.com/sgl-project/sglang)后端进行推理加速，具体请参阅[VLLM文档](https://llmc-zhcn.readthedocs.io/en/latest/backend/vllm.html)和[SGLang文档](https://llmc-zhcn.readthedocs.io/en/latest/backend/sglang.html)。

- **2024年9月26日:** 🔥 我们现在支持从🚀 `LLMC`导出💥 `FP8 量化（E4M3，E5M2）`模型到一些先进的推理后端，例如[VLLM](https://github.com/vllm-project/vllm)和[SGLang](https://github.com/sgl-project/sglang)。关于详细使用方法，请参阅[VLLM文档](https://llmc-zhcn.readthedocs.io/en/latest/backend/vllm.html)和[SGLang文档](https://llmc-zhcn.readthedocs.io/en/latest/backend/sglang.html)。

- **2024年9月24日:** 🔥 我们正式发布了 ✨`Llama-3.1-405B` 的 ✅INT4 和 ✅INT8 模型，这些模型通过 🚀`LLMC` 使用 `save_lightllm` 模式进行量化。你可以在[此处](https://huggingface.co/Dongz/llama31-405b-quant)下载模型参数。

- **2024年9月23日:** 🔥 我们现在支持从 🚀`LLMC` 导出 ✨`真正量化的(INT4, INT8)` 模型到先进推理后端，例如 [VLLM](https://github.com/vllm-project/vllm), [SGLang](https://github.com/sgl-project/sglang), [AutoAWQ](https://github.com/casper-hansen/AutoAWQ), 和 [MLC-LLM](https://github.com/mlc-ai/mlc-llm) 用于量化推理部署，从而实现 ✨`减少内存使用` 和 ✨`加快推理速度`。
  详细使用方法，请参考 [VLLM 文档](https://llmc-zhcn.readthedocs.io/en/latest/backend/vllm.html)、[SGLang 文档](https://llmc-zhcn.readthedocs.io/en/latest/backend/sglang.html)、[AutoAWQ 文档](https://llmc-zhcn.readthedocs.io/en/latest/backend/autoawq.html) 和 [MLC-LLM 文档](https://llmc-zhcn.readthedocs.io/en/latest/backend/mlcllm.html)。

- **2024年9月9日:** 🔥 我们提供了一些最佳实践配置，帮助提升性能（参见最佳实践[此处](https://llmc-zhcn.readthedocs.io/en/latest/)）。

- **2024年9月3日:** 🔥 我们支持通过[opencompass](https://github.com/open-compass/opencompass) 评估 🚀`LLMC` 模型。请参考此[文档](https://llmc-zhcn.readthedocs.io/en/latest/advanced/model_test_v2.html)试用！

- **2024年8月22日:** 🔥我们支持许多小型语言模型，包括当前SOTA的 [SmolLM](https://huggingface.co/collections/HuggingFaceTB/smollm-6695016cad7167254ce15966)(参见[支持的模型列表](#supported-model-list))。

- **2024年8月22日:** 🔥此外，我们还支持通过我们修改的 [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) 进行下游任务评估 🤗。具体操作，用户可以先采用 `save_trans` 模式（参见 [配置](https://llmc-zhcn.readthedocs.io/en/latest/configs.html) 中的 `save` 部分）保存权重修改后的模型。在获得转换模型后，可以直接参考 [run_lm_eval.sh](scripts/run_lm_eval.sh) 对量化模型进行评估。更多细节请见[此处](https://llmc-zhcn.readthedocs.io/en/latest/advanced/model_test_v1.html)。

- **2024年7月23日:** 🍺🍺🍺 我们发布了全新的基准论文：

  [**LLMC: Benchmarking Large Language Model Quantization with a Versatile Compression Toolkit**](https://arxiv.org/abs/2405.06001v2)。

  [Ruihao Gong\*](https://xhplus.github.io/), [Yang Yong\*](https://github.com/helloyongyang), [Shiqiao Gu\*](https://github.com/gushiqiao), [Yushi Huang\*](https://github.com/Harahan), [Chengtao Lv](https://scholar.google.com/citations?user=r8vseSUAAAAJ&hl=en), [Yunchen Zhang](https://scholar.google.com/citations?user=glkWFyUAAAAJ&hl=en), [Xianglong Liu📧](https://xlliu-beihang.github.io/), [Dacheng Tao](https://scholar.google.com/citations?user=RwlJNLcAAAAJ&hl=en)

  (\* 表示同等贡献，📧 表示通讯作者。)

<details close>
<summary>历史消息</summary>

- **2024年7月16日:** 🔥我们现在支持 Wanda/Naive（幅度）进行 LLM 稀疏化和逐层混合比特量化！

- **2024年7月14日:** 🔥我们现在支持基于旋转的量化 QuaRot！

- **2024年5月17日:** 🚀 我们现在支持一些先进的大型模型，例如 LLaVA、Mixtral、LLaMA V3 和 Qwen V2。快来试试吧！

- **2024年5月13日:** 🍺🍺🍺 我们发布了量化基准论文：

  [**LLM-QBench: A Benchmark Towards the Best Practice for Post-training Quantization of Large Language Models**](https://arxiv.org/abs/2405.06001)。

  [Ruihao Gong\*](https://xhplus.github.io/), [Yang Yong\*](https://github.com/helloyongyang), [Shiqiao Gu\*](https://github.com/gushiqiao), [Yushi Huang\*](https://github.com/Harahan), [Yunchen Zhang](https://scholar.google.com/citations?user=glkWFyUAAAAJ&hl=en), [Xianglong Liu📧](https://xlliu-beihang.github.io/), [Dacheng Tao](https://scholar.google.com/citations?user=RwlJNLcAAAAJ&hl=en)

  (\* 表示同等贡献，📧 表示通讯作者。)

  <div align=center>
   <img src="./imgs/best_practice.png" alt="comp" width="800" />
  </div>

  我们模块化且公平地基准测试了量化技术，考虑了校准成本、推理效率和量化准确性。在多种模型和数据集上进行了近600次实验，得出了三个关于校准数据、算法管道和量化配置选择的有见地的结论。基于这些结论，设计了一种LLM后训练量化管道的最佳实践，以在各种场景下实现最佳的准确性和效率平衡。

- **2024年3月7日:** 🚀 我们发布了一个功能强大且高效的LLM压缩工具的量化部分。值得注意的是，我们的基准论文即将发布😊。

</details>

## 亮点功能

- 💥**综合算法支持**: 提供广泛的 ✨`SOTA压缩算法` 支持，包括 ✅量化、✅混合精度量化 和 ✅稀疏化，同时保持与原始仓库一致的精度。我们还提供 ✨`量化最佳实践`（参见✨`最佳实践` 章节[此处](https://llmc-zhcn.readthedocs.io/en/latest/)），确保最佳性能和效率。

- 💥**支持的格式**: 支持 ✨`量化`（整型和浮点）和 ✨`稀疏化`，具体包括 ✅权重激活量化、✅权重量化、✅混合精度量化，以及 ✅结构化 和 ✅非结构化稀疏化。

- 💥**广泛模型支持**: 支持多种 ✨`LLM模型`，包括 ✅LLama、✅Mistral、✅InternLM2、✅Qwen2 等，以及 ✅MOE(DeepSeekv2, Deepseekv2.5) 和 ✅VLM(Llama3.2-vision, Qwen2-vl) 模型（参见[支持的模型列表](#supported-model-list)）。

- 💥**多后端兼容性**: 无缝集成多个后端，增强部署灵活性。多种量化设置和模型格式兼容广泛的后端和硬件平台，例如 ✅VLLM、✅Sglang、✅LightLLM、✅MLC-LLM 和 ✅AutoAWQ，使其高度灵活（参见✨`推理后端` 章节 [此处](https://llmc-zhcn.readthedocs.io/en/latest/)）。

- 💥**性能效率**: 支持大规模LLM的量化，例如 ✨`Llama3.1-405B` 和 ✨`DeepSeekV2-236B`，并可在 `单个 A100/H100/H800 GPU` 上评估 PPL。

## 使用指南

请参阅 🚀`快速入门`章节[此处](https://llmc-zhcn.readthedocs.io/en/latest/)。

## 支持的模型列表

✅ [BLOOM](https://huggingface.co/bigscience/bloom)

✅ [LLaMA](https://github.com/facebookresearch/llama)

✅ [LLaMA V2](https://huggingface.co/meta-llama)

✅ [StarCoder](https://github.com/bigcode-project/starcoder)

✅ [OPT](https://huggingface.co/docs/transformers/model_doc/opt)

✅ [Falcon](https://huggingface.co/docs/transformers/model_doc/falcon)

✅ [InternLM2](https://huggingface.co/internlm)

✅ [Mistral](https://huggingface.co/docs/transformers/model_doc/mistral)

✅ [LLaMA V3](https://huggingface.co/meta-llama)

✅ [Mixtral](https://huggingface.co/docs/transformers/model_doc/mixtral)

✅ [Qwen V2](https://github.com/QwenLM/Qwen2)

✅ [LLaVA](https://github.com/haotian-liu/LLaVA)

✅ [InternLM2.5](https://huggingface.co/internlm)

✅ [StableLM](https://github.com/Stability-AI/StableLM)

✅ [Gemma2](https://huggingface.co/docs/transformers/main/en/model_doc/gemma2)

✅ [Phi2](https://huggingface.co/microsoft/phi-2)

✅ [Phi 1.5](https://huggingface.co/microsoft/phi-1_5)

✅ [MiniCPM](https://github.com/OpenBMB/MiniCPM)

✅ [SmolLM](https://huggingface.co/collections/HuggingFaceTB/smollm-6695016cad7167254ce15966)

✅ [DeepSeekv2.5](https://huggingface.co/deepseek-ai/DeepSeek-V2.5)

✅ [LLaMA V3.2 Vision](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision)

✅ [Qwen MOE](https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B)

✅ [Qwen2-VL](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)

✅ [InternVL2](https://huggingface.co/OpenGVLab/InternVL2-2B)

你可以参考 `llmc/models/*.py` 文件添加自己的模型类型。

## 支持的后端列表

✅ [VLLM](https://github.com/vllm-project/vllm)

✅ [LightLLM](https://github.com/ModelTC/lightllm)

✅ [Sglang](https://github.com/sgl-project/sglang)

✅ [MLC-LLM](https://github.com/mlc-ai/mlc-llm)

✅ [AutoAWQ](https://github.com/casper-hansen/AutoAWQ)

## 支持的算法列表

### 量化

✅ Naive

✅ [AWQ](https://arxiv.org/abs/2306.00978)

✅ [GPTQ](https://arxiv.org/abs/2210.17323)

✅ [SmoothQuant](https://arxiv.org/abs/2211.10438)

✅ [OS+](https://arxiv.org/abs/2304.09145)

✅ [OmniQuant](https://arxiv.org/abs/2308.13137)

✅ [NormTweaking](https://arxiv.org/abs/2309.02784)

✅ [AdaDim](https://arxiv.org/pdf/2309.15531.pdf)

✅ [QUIK](https://arxiv.org/abs/2310.09259)

✅ [SpQR](https://arxiv.org/abs/2306.03078)

✅ [DGQ](https://arxiv.org/abs/2310.04836)

✅ [OWQ](https://arxiv.org/abs/2306.02272)

✅ [LLM.int8()](https://arxiv.org/abs/2208.07339)

✅ [HQQ](https://mobiusml.github.io/hqq_blog/)

✅ [QuaRot](https://arxiv.org/abs/2404.00456)

✅ [SpinQuant](https://arxiv.org/abs/2405.16406) **([见此分支](https://github.com/ModelTC/llmc/tree/dev_spinquant))**

✅ [TesseraQ](https://arxiv.org/abs/2410.19103)

### 剪枝

✅ Naive（Magnitude）

✅ [Wanda](https://arxiv.org/abs/2306.11695)

✅ [ShortGPT](https://arxiv.org/abs/2403.03853)

## 鸣谢

我们的代码参考了以下仓库：

- https://github.com/mit-han-lab/llm-awq
- https://github.com/mit-han-lab/smoothquant
- https://github.com/OpenGVLab/OmniQuant
- https://github.com/IST-DASLab/gptq
- https://github.com/ModelTC/Outlier_Suppression_Plus
- https://github.com/IST-DASLab/QUIK
- https://github.com/Vahe1994/SpQR
- https://github.com/ilur98/DGQ
- https://github.com/xvyaward/owq
- https://github.com/TimDettmers/bitsandbytes
- https://github.com/mobiusml/hqq
- [https://github.com/spcl/QuaRot](https://github.com/spcl/QuaRot)
- [https://github.com/locuslab/wanda](https://github.com/locuslab/wanda)
- [https://github.com/EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [https://github.com/facebookresearch/SpinQuant](https://github.com/facebookresearch/SpinQuant)
- [https://github.com/Intelligent-Computing-Lab-Yale/TesseraQ](https://github.com/Intelligent-Computing-Lab-Yale/TesseraQ)

## Star 历史

[![Star History Chart](https://api.star-history.com/svg?repos=ModelTC/llmc&type=Timeline)](https://star-history.com/#ModelTC/llmc&Timeline)

## 引用

## 引用

如果您认为我们的 LLM-QBench 论文/llmc 工具对您的研究有用或相关，请务必引用我们的论文：

```
@misc{llmc,
   author = {llmc contributors},
   title = {llmc: Towards Accurate and Efficient LLM Compression},
   year = {2024},
   publisher = {GitHub},
   journal = {GitHub repository},
   howpublished = {\url{https://github.com/ModelTC/llmc}},
}

@misc{gong2024llmqbench,
      title={LLM-QBench: A Benchmark Towards the Best Practice for Post-training Quantization of Large Language Models},
      author={Ruihao Gong and Yang Yong and Shiqiao Gu and Yushi Huang and Yunchen Zhang and Xianglong Liu and Dacheng Tao},
      year={2024},
      eprint={2405.06001},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}

@misc{gong2024llmcbenchmarkinglargelanguage,
      title={LLMC: Benchmarking Large Language Model Quantization with a Versatile Compression Toolkit},
      author={Ruihao Gong and Yang Yong and Shiqiao Gu and Yushi Huang and Chentao Lv and Yunchen Zhang and Xianglong Liu and Dacheng Tao},
      year={2024},
      eprint={2405.06001},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2405.06001},
}
```
