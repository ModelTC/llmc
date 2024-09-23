# llmc: 正確で効率的なLLM圧縮に向けて

<img src="./imgs/llmc.png" alt="llmc" style="zoom:35%;" />

[![ライセンス](https://img.shields.io/badge/ライセンス-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![arXiv](https://img.shields.io/badge/LLMC-2405.06001-b31b1b)](https://arxiv.org/abs/2405.06001)
[![GitHub スター](https://img.shields.io/github/stars/ModelTC/llmc.svg?style=social&label=Star&maxAge=60)](https://github.com/ModelTC/llmc)
![訪問者](https://komarev.com/ghpvc/?username=llmc&label=visitors)
[![Discord バナー](https://img.shields.io/discord/1139835312592392214?logo=discord&logoColor=white)](https://discord.gg/qZKUDfhm)
[![QQ](https://img.shields.io/badge/QQ-EB1923?logo=tencent-qq&logoColor=white)](http://qm.qq.com/cgi-bin/qm/qr?_wv=1027&k=I9IGPWWj8uuRXWH3_ELWjouf6gkIMgUl&authKey=GA3WbFAsm90ePJf%2FCbc7ZyXXq4ShQktlBaLxgqS5yuSPAsr3%2BDKMRdosUiLYoilO&noverify=0&group_code=526192592)
[![Doc](https://img.shields.io/badge/docs-English-99cc2)](https://llmc-en.readthedocs.io/en/latest/)
[![Doc](https://img.shields.io/badge/文档-中文-99cc2)](https://llmc-zhcn.readthedocs.io/en/latest/)

**\[ [English](README.md) | [中文](README_zh.md) | 日本語 \]**

**llmc** は、最先端の圧縮アルゴリズムを活用して、パフォーマンスを損なうことなく効率を向上させ、モデルサイズを削減することを目的とした、オフ・ザ・シェルフのツールです。

**英語のドキュメント**は[こちら](https://llmc-en.readthedocs.io/en/latest/)です。

**docker hub**は[こちら](https://hub.docker.com/r/llmcompression/llmc)です。

**コミュニティ**:

- [Discord サーバー](https://discord.gg/qZKUDfhm)
- [Tencent QQ グループ](http://qm.qq.com/cgi-bin/qm/qr?_wv=1027&k=I9IGPWWj8uuRXWH3_ELWjouf6gkIMgUl&authKey=GA3WbFAsm90ePJf%2FCbc7ZyXXq4ShQktlBaLxgqS5yuSPAsr3%2BDKMRdosUiLYoilO&noverify=0&group_code=526192592)

## ニュース

- **2024年9月9日:** 🔥 量子化された LLM の [vLLM](https://github.com/vllm-project/vllm) へのエクスポートを修正しました ([こちら](https://llmc-en.readthedocs.io/en/latest/backend/vllm.html) を参照)。さらに、優れたパフォーマンスに向けた ベスト プラクティスの構成 もいくつか提供しています (ベスト プラクティスは [こちら](https://llmc-en.readthedocs.io/en/latest/) を参照)。

* **2024年9月3日:** 🚀 私たちはOpenCompassの精度評価をサポートしました。ドキュメントは[こちら](https://llmc-en.readthedocs.io/en/latest/advanced/model_test_v2.html)を参照してください。ぜひご利用ください！

* **2024年8月22日：** 🔥私たちは、現在の最先端技術である[SmolLM](https://huggingface.co/collections/HuggingFaceTB/smollm-6695016cad7167254ce15966)（[サポートされているモデルリスト](#supported-model-list)を参照）を含む多くの小型言語モデルをサポートしています。さらに、改良された[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) 🤗 を通じてダウンストリームタスクの評価もサポートしています。具体的には、まず `save_trans` モード（[設定](#設定)の `save` 部分を参照）を使用して、変更されたモデルの重みを保存します。変換後のモデルを取得した後、[run_lm_eval.sh](scripts/run_lm_eval.sh)を参照して量子化モデルを直接評価することができます。詳細は[こちら](https://llmc-en.readthedocs.io/en/latest/advanced/model_test.html)で確認できます。

* **2024年7月23日:** 🍺🍺🍺 新しいバージョンのベンチマーク ペーパーをリリースします:

  [**LLMC: 多用途の圧縮ツールキットを使用した大規模言語モデル量子化のベンチマーク**](https://arxiv.org/abs/2405.06001v2)。

  [Ruihao Gong\*](https://xhplus.github.io/)、[Yang Yong\*](https://github.com/helloyongyang)、[Shiqiao Gu\*](https://github.com/gushiqiao)、[Yushi Huang\*](https://github.com/Harahan)、[Chengtao Lv](https://scholar.google.com/citations?user=r8vseSUAAAAJ&hl=en)、[Yunchen Zhang](https://scholar.google.com/citations?user=glkWFyUAAAAJ&hl=en)、[Xianglong Liu📧](https://xlliu-beihang.github.io/)、[Dacheng Tao](https://scholar.google.com/citations?user=RwlJNLcAAAAJ&hl=en)

  (\* は同等の貢献、📧 は対応する貢献を表します著者。)

  <div align=center>
  <img src="./imgs/K.png" alt="comp" width="800" />
  </div>

  ベストプラクティスに焦点を当てるのではなく、キャリブレーションデータ、アルゴリズム、データ形式を考慮して、LLM量子化をモジュール式かつ公平にベンチマークします。詳細な観察と分析により、さまざまな構成でパフォーマンスと方法を改善するためのさまざまなタイプの新しいポイントを提供します。強力なツールキットLLMCと包括的な洞察により、将来のLLM研究者は、アプリケーションに適したアルゴリズムと低ビット形式を効率的に統合し、大規模な言語モデルの圧縮を民主化できます。

* **2024年7月16日:** 🔥現在、llmのスパース化と層間混合ビット量子化のためのWanda/Naive(Magnitude)をサポートしています！

* **2024年7月14日:** 🔥現在、回転ベースの量子化QuaRotをサポートしています！

* **2024年7月4日:** 📱 ディスカッションチャンネルを開設しました。質問がある場合は、コミュニティに参加してください:

  - [Discord サーバー](https://discord.gg/qZKUDfhm)
  - [Tencent QQ グループ](http://qm.qq.com/cgi-bin/qm/qr?_wv=1027&k=I9IGPWWj8uuRXWH3_ELWjouf6gkIMgkUl&authKey=GA3WbFAsm90ePJf%2FCbc7ZyXXq4ShQktlBaLxgqS5yuSPAsr3%2BDKMRdosUiLYoilO&noverify=0&group_code=526192592)

* **2024年5月17日:** 🚀 現在、LLaVA、Mixtral、LLaMA V3、Qwen V2などの高度な大規模モデルをサポートしています。試してみてください！

* **2024年5月13日:** 🍺🍺🍺 量子化ベンチマーク論文を発表しました:

  [**LLM-QBench: 大規模言語モデルのポストトレーニング量子化のベストプラクティスに向けたベンチマーク**](https://arxiv.org/abs/2405.06001).

  [Ruihao Gong\*](https://xhplus.github.io/), [Yang Yong\*](https://github.com/helloyongyang), [Shiqiao Gu\*](https://github.com/gushiqiao), [Yushi Huang\*](https://github.com/Harahan), [Yunchen Zhang](https://scholar.google.com/citations?user=glkWFyUAAAAJ&hl=en), [Xianglong Liu📧](https://xlliu-beihang.github.io/), [Dacheng Tao](https://scholar.google.com/citations?user=RwlJNLcAAAAJ&hl=en)

  (\* は同等の貢献を示し、📧 は対応する著者を示します。)

  <div align=center>
   <img src="./imgs/best_practice.png" alt="comp" width="800" />
  </div>

  校正コスト、推論効率、および量子化精度を考慮して、量子化技術をモジュール化し、公平にベンチマークしました。多様なモデルとデータセットでの約600の実験が、校正データ、アルゴリズムパイプライン、および量子化構成の選択に関する3つの洞察を提供します。これらの洞察に基づいて、LLM PTQパイプラインのベストプラクティスが設計され、さまざまなシナリオで最高の精度と効率のパフォーマンスバランスを実現します。

* **2024年3月7日:** 🚀 強力で効率的なLLM圧縮ツールの量子化部分をリリースしました。注目すべきは、ベンチマーク論文が近日公開予定です😊。

## ハイライト機能

- LLMs（例：Llama2-70B、OPT-175B）を量子化し、1つのA100/H100/H800 GPUでPPLを評価します💥。
- ユーザーが選択できる最先端の圧縮アルゴリズムが[元のリポジトリと一致](benchmark/align.md)し、ユーザーは1つのLLMで複数のアルゴリズムを順次使用できます💥。
- 特定の圧縮アルゴリズムでツールによってエクスポートされた変換モデル（[構成](#構成)の`quant`部分の`save_trans`モード）は、複数のバックエンド（例：[Lightllm](https://github.com/ModelTC/lightllm)、[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)）によって単純な量子化を行い、特定の圧縮アルゴリズムで最適化されたモデルを取得できます。対応するバックエンドが推論できます💥。
- 浅いメモリフットプリントを持つ圧縮モデル（[構成](#構成)の`quant`部分の`save_lightllm`モード）は、[Lightllm](https://github.com/ModelTC/lightllm)によって直接推論できます💥。

## 使用方法

1. このリポジトリをクローンし、パッケージをインストールします：

   ```shell
   # パッケージをインストール
   cd llmc
   pip install -r requirements.txt
   ```

2. モデルとデータを準備します。

   ```shell
   # huggingfaceからLLMをダウンロードした後、次のように校正データと評価データを準備します：
   cd tools
   python download_calib_dataset.py --save_path [校正データパス]
   python download_eval_dataset.py --save_path [評価データパス]
   ```

3. アルゴリズムを選択してモデルを量子化します：

   ```shell
   # これはAwqに関する例です：
   cd scripts
   # bashファイル内のllmcのパス``llmc_path``を変更します。``llmc/configs/quantization/Awq/``に配置された構成の1つを選択してモデルを量子化するか、run_awq_llama.shの``--config``引数を変更して提供された構成を使用します。
   bash run_awq_llama.sh
   ```

## 構成

ユーザーが構成を設計するのを支援するために、`llmc/configs/`の下に提供されているすべての構成のいくつかの一般的な構成を説明します：

- `model`:

  ```yaml
  model:
      # ``llmc/models/*.py``のクラス名に置き換えます。
      type: Llama
      # モデルのパスに置き換えます。
      path: model path
      torch_dtype: auto
  ```

- `calib`:

  ```yaml
  # 注意：一部のアルゴリズムには``calib``が必要ありません。例：naive... したがって、この部分を削除できます。
  calib:
      # 以前にダウンロードした校正データ名に置き換えます。例：pileval、c4、wikitext2、またはptb。
      name: pileval
      download: False
      # 以前にダウンロードした校正データの1つのパスに置き換えます。例：pileval、c4、wikitext2、またはptb。
      path: calib data path
      n_samples: 128
      bs: -1
      seq_len: 512
      # ``llmc/data/dataset/specified_preproc.py``の関数名に置き換えます。
      preproc: general
      seed: *seed
  ```

- `eval`:

  ```yaml
  # 事前トレーニング/変換/偽量子化モデルのPPLを評価したい場合。
  eval:
      # 事前トレーニング、変換、偽量子化モデルを評価し、評価したい位置を設定できます。
      eval_pos: [pretrain, transformed, fake_quant]
      # 以前にダウンロードした評価データの名前に置き換えます。例：c4、wikitext2、ptb、または[c4, wikitext2]。
      name: wikitext2
      download: False
      path: eval data path
      # 70Bモデルの評価の場合、bsを20に設定し、inference_per_blockをTrueに設定できます。
      # 7B / 13Bモデルの評価の場合、bsを1に設定し、inference_per_blockをFalseに設定できます。
      bs: 1
      inference_per_block: False
      seq_len: 2048
  ```

- `save`:

  ```yaml
  save:
      # ``save_trans``がTrueの場合、変換モデル（例：パラメータが変更されたモデル）をエクスポートしたいことを意味します。パフォーマンスと構造は元のモデルと同じであり、ユーザーは単純な量子化を使用して、特定のアルゴリズムで量子化されたモデルと同じパフォーマンスを得ることができます。
      save_trans: False
      # ``save_lightllm``または ``save_trtllm`` がTrueの場合、実際の量子化モデル（例：低ビットの重みと重みおよびアクティベーションの量子化パラメータ）をエクスポートしたいことを意味します。
      save_lightllm: False
      # ``save_fake``がTrueの場合、偽量子化モデル（例：量子化解除された重みとアクティベーションの量子化パラメータ）をエクスポートしたいことを意味します。
      save_fake: False
      save_path: ./save
  ```

- `quant`:

  ```yaml
  quant:
      # ``llmc/compression/quantization/*.py``のクラス名に置き換えます。
      method: OmniQuant
      # 重みのみの量子化には``act``部分がありません。
      weight:
          bit: 8
          symmetric: True
          # 量子化の粒度：per_channel、per_tensor、per_head（推奨されません）。
          granularity: per_channel
          group_size: -1
          # 校正アルゴリズム：learnble、mse、およびminmax（デフォルト）。
          calib_algo: learnable
          # ストレートスルー推定を使用します。これは、学習可能な校正アルゴリズムに必要です。
          ste: True
      act:
          bit: 8
          symmetric: True
          # 量子化の粒度：per_token、per_tensor
          granularity: per_token
          ste: True
          # 静的量子化（校正中の量子化）または動的量子化（推論中の量子化）。
          static: True
      # この部分は特定のアルゴリズム用に設計されており、提供されているものを参考にして独自のアルゴリズムを設計できます。
      special:
          let: True
          lwc_lr: 0.01
          let_lr: 0.005
          use_shift: False
          alpha: 0.5
          deactive_amp: True
          epochs: 20
          wd: 0
      # quant_outがTrueの場合、前の量子化ブロックの出力を次のブロックの校正データとして使用します。
      quant_out: True
  ```

## サポートされているモデルリスト

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

`llmc/models/*.py`の下のファイルを参照して、独自のモデルタイプを追加できます。

## サポートされているアルゴリズムリスト

### 量子化

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

### 剪定

✅ Naive(Magnitude)

✅ [Wanda](https://arxiv.org/abs/2306.11695)

✅ [ShortGPT](https://arxiv.org/abs/2403.03853)

## 謝辞

以下のリポジトリを参考にしてコードを開発しました：

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

## スター履歴

[![スター履歴チャート](https://api.star-history.com/svg?repos=ModelTC/llmc&type=Timeline)](https://star-history.com/#ModelTC/llmc&Timeline)

## 引用

LLM-QBench論文/llmcツールキットが研究に役立つまたは関連している場合は、論文を引用してください：

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
