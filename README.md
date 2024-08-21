# llmc: Towards Accurate and Efficient LLM Compression

<img src="./imgs/llmc.png" alt="llmc" style="zoom:35%;" />

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![arXiv](https://img.shields.io/badge/LLMC-2405.06001-b31b1b)](https://arxiv.org/abs/2405.06001)
[![GitHub Stars](https://img.shields.io/github/stars/ModelTC/llmc.svg?style=social&label=Star&maxAge=60)](https://github.com/ModelTC/llmc)
![visitors](https://komarev.com/ghpvc/?username=llmc&label=visitors)
[![Discord Banner](https://img.shields.io/discord/1139835312592392214?logo=discord&logoColor=white)](https://discord.gg/qZKUDfhm)
[![QQ](https://img.shields.io/badge/QQ-EB1923?logo=tencent-qq&logoColor=white)](http://qm.qq.com/cgi-bin/qm/qr?_wv=1027&k=I9IGPWWj8uuRXWH3_ELWjouf6gkIMgUl&authKey=GA3WbFAsm90ePJf%2FCbc7ZyXXq4ShQktlBaLxgqS5yuSPAsr3%2BDKMRdosUiLYoilO&noverify=0&group_code=526192592)
[![Doc](https://img.shields.io/badge/docs-English-99cc2)](https://llmc-en.readthedocs.io/en/latest/)
[![Doc](https://img.shields.io/badge/文档-中文-99cc2)](https://llmc-zhcn.readthedocs.io/en/latest/)

**\[ English | [中文](README_zh.md) | [日本語](README_ja.md) \]**

**llmc** is an off-the-shell tool designed for compressing LLM, leveraging state-of-the-art compression algorithms to enhance efficiency and reduce model size without compromising performance.

**English doc** is [here](https://llmc-en.readthedocs.io/en/latest/).

**Chinese doc** is [here](https://llmc-zhcn.readthedocs.io/en/latest/).

**Community**:

- [Discord Server](https://discord.gg/qZKUDfhm)
- [Tencent QQ Group](http://qm.qq.com/cgi-bin/qm/qr?_wv=1027&k=I9IGPWWj8uuRXWH3_ELWjouf6gkIMgUl&authKey=GA3WbFAsm90ePJf%2FCbc7ZyXXq4ShQktlBaLxgqS5yuSPAsr3%2BDKMRdosUiLYoilO&noverify=0&group_code=526192592)

## News

- **Aug 22, 2024:** 🔥We support lots of small language models, including current SOTA [SmolLM](https://huggingface.co/collections/HuggingFaceTB/smollm-6695016cad7167254ce15966)(see [Supported Model List](#supported-model-list)). Additionally, we also support down stream task evaluation through our modified [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) 🤗. Specifically, people can first employ `save_trans` mode(see `save` part in [Configuration](#configuration)) to save a weight modified model. After obtaining the transformed model, they can directly evaluate the quantized model referring to [run_lm_eval.sh](scripts/run_lm_eval.sh). More details can be found in [here](https://llmc-en.readthedocs.io/en/latest/advanced/model_test.html).

- **Jul 23, 2024:** 🍺🍺🍺 We release a brand new version benchmark paper:

  [**LLMC: Benchmarking Large Language Model Quantization with a Versatile Compression Toolkit**](https://arxiv.org/abs/2405.06001v2).

  [Ruihao Gong\*](https://xhplus.github.io/), [Yang Yong\*](https://github.com/helloyongyang), [Shiqiao Gu\*](https://github.com/gushiqiao), [Yushi Huang\*](https://github.com/Harahan), [Chengtao Lv](https://scholar.google.com/citations?user=r8vseSUAAAAJ&hl=en), [Yunchen Zhang](https://scholar.google.com/citations?user=glkWFyUAAAAJ&hl=en), [Xianglong Liu📧](https://xlliu-beihang.github.io/), [Dacheng Tao](https://scholar.google.com/citations?user=RwlJNLcAAAAJ&hl=en)

  (\* denotes equal contribution, 📧 denotes corresponding author.)

  <div align=center>
   <img src="./imgs/K.png" alt="comp" width="800" />
  </div>

  Instead of focusing on the best practice, We modularly and fairly benchmark LLM quantization considering calibration data, algorithms, and data formats. With detailed observation and analysis, we provide various types of novel points for performance and method improvements under different configurations. With the powerful toolkit LLMC and comprehensive insights, future LLM researchers can efficiently integrate suitable algorithms and low-bit formats for their applications, thereby democratizing the compression of large language models.

- **Jul 16, 2024:** 🔥We support Wanda/Naive(Magnitude) for llm sparsification and layer-wise mix bits quantization now!

- **Jul 14, 2024:** 🔥We support rotation based quantization QuaRot now!

- **Jul 4, 2024:** 📱 We open our discussion channel. If you have any questions, please join our community:

  - [Discord Server](https://discord.gg/qZKUDfhm)
  - [Tencent QQ Group](http://qm.qq.com/cgi-bin/qm/qr?_wv=1027&k=I9IGPWWj8uuRXWH3_ELWjouf6gkIMgUl&authKey=GA3WbFAsm90ePJf%2FCbc7ZyXXq4ShQktlBaLxgqS5yuSPAsr3%2BDKMRdosUiLYoilO&noverify=0&group_code=526192592)

- **May 17, 2024:** 🚀 We support some advanced large models, e.g., LLaVA, Mixtral, LLaMA V3 and Qwen V2 now. Have a try!

- **May 13, 2024:** 🍺🍺🍺 We release our quantization benchmark paper:

  [**LLM-QBench: A Benchmark Towards the Best Practice for Post-training Quantization of Large Language Models**](https://arxiv.org/abs/2405.06001).

  [Ruihao Gong\*](https://xhplus.github.io/), [Yang Yong\*](https://github.com/helloyongyang), [Shiqiao Gu\*](https://github.com/gushiqiao), [Yushi Huang\*](https://github.com/Harahan), [Yunchen Zhang](https://scholar.google.com/citations?user=glkWFyUAAAAJ&hl=en), [Xianglong Liu📧](https://xlliu-beihang.github.io/), [Dacheng Tao](https://scholar.google.com/citations?user=RwlJNLcAAAAJ&hl=en)

  (\* denotes equal contribution, 📧 denotes corresponding author.)

  <div align=center>
   <img src="./imgs/best_practice.png" alt="comp" width="800" />
  </div>

  We modularly and fairly benchmark the quantization techniques considering calibration cost, inference efficiency, and quantized accuracy. Near 600 experiments on diverse models and datasets provide three insightful takeaways
  on the calibration data, algorithm pipeline, and quantization configuration selection. Based on the takeaways, a best practice for the LLM PTQ pipeline is designed, to achieve the best accuracy and efficiency performance balance
  under various scenarios.

- **Mar 7, 2024:** 🚀 We release the quantization part of a powerful and efficient LLM compression tool. Notably, our benchmark paper is coming soon😊.

## Highlight Feature

- Quantize LLMs, e.g., Llama2-70B, OPT-175B,  and evaluate their PPL on only one A100/H100/H800 GPU💥.
- SOTA compression algorithms [align with the origin repos](benchmark/align.md), for users to choose from, and users can sequentially employ multiple algorithms on one LLM💥.
- Transformed model (`save_trans`  mode in `quant` part in [Configuration](#configuration)) exported by our tool with a specifical compression algorithm can go through naive quantization by multiple backends, e.g., [Lightllm](https://github.com/ModelTC/lightllm), [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) to get a specifical-compression-algorithm-optimized model, which the corresponding backend can infer 💥.
- Our compressed model (`save_lightllm`  mode in `quant` part in [Configuration](#configuration)) with a shallow memory footprint can be directly inferred by [Lightllm](https://github.com/ModelTC/lightllm)💥.

## Usage

1. Clone this repository and install packages:

   ```shell
   # install packages
   cd llmc
   pip install -r requirements.txt
   ```

2. Prepare models and data.

   ```shell
   # After downloading LLMs from huggingface, prepare calibration and evaluation data as follows:
   cd tools
   python download_calib_dataset.py --save_path [calib data path]
   python download_eval_dataset.py --save_path [eval data path]
   ```

3. Choose an algorithm to quantize your model:

   ```shell
   # Here's an example about Awq:
   cd scripts
   # Modify the path of llmc, ``llmc_path``, in the bash file. You can also choose one config
   # placed in ``llmc/configs/quantization/Awq/`` to quantize your model, or your own
   # config referring to those we provide by changing the ``--config`` argument in run_awq_llama.sh.
   bash run_awq_llama.sh
   ```

## Configuration

To help users design their configs, we now explain some universal configurations in all configs we provide under `llmc/configs/`:

- `model`:

  ```yaml
  model:
      # Replace by the name of the class in ``llmc/models/*.py``.
      type: Llama
      # Replace by the path of your model.
      path: model path
      torch_dtype: auto
  ```

- `calib`:

  ```yaml
  # Note: some algorithms do not need ``calib``, like naive... So, you can remove this part.
  calib:
      # Replace by the calibration data name, e.g., pileval, c4, wikitext2, or ptb, downloaded before.
      name: pileval
      download: False
      # Replace by the path of one of the calibration data, e.g., pileval, c4, wikitext2, or ptb,
      # downloaded before.
      path: calib data path
      n_samples: 128
      bs: -1
      seq_len: 512
      # Replace by the function name in ``llmc/data/dataset/specified_preproc.py``.
      preproc: general
      seed: *seed
  ```

- `eval`:

  ```yaml
  # If you want to evaluate PPL of your pretrained/transformed/fake_quant model.
  eval:
      # You can evaluate the pretrain, transformed, fake_quant model, and set the position
      # you want to evaluate.
      eval_pos: [pretrain, transformed, fake_quant]
      # Replace by the name of the eval data, e.g., c4, wikitext2, ptb or [c4, wikitext2],
      # downloaded before.
      name: wikitext2
      download: False
      path: eval data path
      # For 70B model eval, bs can be set to 20, and inference_per_block can be set to True.
      # For 7B / 13B model eval, bs can be set to 1, and inference_per_block can be set to False.
      bs: 1
      inference_per_block: False
      seq_len: 2048
  ```

- `save`:

  ```yaml
  save:
      # ``save_trans`` is True, which means you want to export the transformed model, e.g., parameter-modified
      # model, whose performance and structure are the same as the original model, and users can
      # utilize naive quantization to the transformed model to obtain the same performance as
      # the specifical-algorithm-quantized model.
      save_trans: False
      # ``save_lightllm`` or ``save_trtllm`` is True, which means you want to export a real quant model, e.g.,
      # low-bit weights with weight and activation quantization parameters.
      save_lightllm: False
      # ``save_fake`` is True means you want to export fake_quant model, e.g.,
      # dequantized weight with activation quantization parameters.
      save_fake: False
      save_path: ./save
  ```

- `quant`:

  ```yaml
  quant:
      # Replace by the class name in ``llmc/compression/quantization/*.py``
      method: OmniQuant
      # weight-only quantization does not have ``act`` part.
      weight:
          bit: 8
          symmetric: True
          # Quantization granularity: per_channel, per_tensor, per_head (not recommended).
          granularity: per_channel
          group_size: -1
          # Calibration algorithms: learnble, mse, and minmax (default).
          calib_algo: learnable
          # Utilize Stright-Through Estimation, which is necessary for learnable
          # calibration algorithms.
          ste: True
      act:
          bit: 8
          symmetric: True
          # Quantization granularity: per_token, per_tensor
          granularity: per_token
          ste: True
          # Static quantization (quantization during calibration)or dynamic
          # quantization (quantization during inference).
          static: True
      # This part is designed for specific algorithms, users can refer to
      # those we provide to design their own.
      special:
          let: True
          lwc_lr: 0.01
          let_lr: 0.005
          use_shift: False
          alpha: 0.5
          deactive_amp: True
          epochs: 20
          wd: 0
      # If quant_out is True, employ the outputs of the former quantized block as the
      # calibration data of the proceeding block.
      quant_out: True
  ```

## Supported Model List

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

You can add your own model type referring to files under `llmc/models/*.py`.

## Supported Algorithm List

### Quantization

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

### Pruning

✅ Naive(Magnitude)

✅ [Wanda](https://arxiv.org/abs/2306.11695)

✅ [ShortGPT](https://arxiv.org/abs/2403.03853)

## Acknowledgments

We develop our code referring to the following repos:

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

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=ModelTC/llmc&type=Timeline)](https://star-history.com/#ModelTC/llmc&Timeline)

## Citation

If you find our LLM-QBench paper/llmc toolkit useful or relevant to your research, please kindly cite our paper:

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
