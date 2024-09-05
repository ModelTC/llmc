# 模型精度测试V2

模型精度测试V1中提到的精度测试方式，流程上不够简洁，我们倾听了社区开发者的声音，开发了模型精度测试V2

在V2版本中，我们不再需要使用推理引擎起服务，也不需要再去拆成多段流程进行测试。

我们的目标是，将下游精度测试等价于PPL测试，运行一个llmc的程序，会在执行完算法之后，直接进行ppl测试，同时也会直接进行对应的下游精度测试。

完成上述的目标，我们只需要在已有的config中，添加一个opencompass的设置


```
base:
    seed: &seed 42
model:
    type: Llama
    path: model path
    torch_dtype: auto
calib:
    name: pileval
    download: False
    path: calib data path
    n_samples: 128
    bs: -1
    seq_len: 512
    preproc: pileval_awq
    seed: *seed
eval:
    eval_pos: [pretrain, fake_quant]
    name: wikitext2
    download: False
    path: eval data path
    bs: 1
    seq_len: 2048
quant:
    method: Awq
    weight:
        bit: 4
        symmetric: False
        granularity: per_group
        group_size: 128
    special:
        weight_clip: False
save:
    save_trans: True
    save_path: ./save
opencompass:
    cfg_path: opencompass config path
    max_num_workers: max num works
    output_path: ./oc_output
```

<font color=792ee5> opencompass下的cfg_path，需要指向一个opencompass的config路径 </font>

我们在[这里](https://github.com/ModelTC/llmc/tree/main/configs/opencompass)分别给出了base模型和chat模型的关于human-eval测试的config，作为给大家的参考。

需要注意的是[opencompass自带的config](https://github.com/ModelTC/opencompass/blob/opencompass-llmc/configs/models/hf_llama/hf_llama3_8b.py)中，需要有path这个key，而这里我们不需要这个key，因为llmc会默认模型的路径在trans的save路径。

当然，因为需要trans的save路径，所以想测试opencompass，就需要设置save_trans为True

<font color=792ee5> opencompass下的max_num_workers，表示最大的推理实例数 </font>

假设模型是在单卡上跑的，那么max_num_workers就是表示，要起max_num_workers个推理实例，即占用了max_num_workers张卡。

假设模型是在多卡上跑的，即参考下面的多卡并行测试，举例如果模型是在2张卡上进行推理，那么max_num_workers就是表示，要起max_num_workers个推理实例，即占用了2*max_num_workers张卡。

综上，所需占用的卡数 = PP数 * max_num_workers

如果所需占用的卡数超过实际中的卡数，那么就会有worker排队情况。

max_num_workers不仅会起多个推理实例，还会把每个数据集进行切分成max_num_workers份，可以理解成是数据并行。

所以：最佳的设置方案就是，让所需占用的卡数=实际可用的卡数。

比如：

在一个8卡机器上，某个模型，用单卡跑，则max_num_workers=8

在一个8卡机器上，某个模型，用四卡跑，则max_num_workers=2

我们尽量让PP数降低，让max_num_workers提高。因为PP并行会变慢，PP仅用在模型实在跑不了的情况，比如70B模型，单卡跑不了，我们就可以设置PP=4，用4个80G显存的卡去跑。

<font color=792ee5> opencompass下的output_path，是设置opencompass的评测日志的输出目录 </font>

在该日志目录中，opencompass会输出推理和评测的日志，推理的具体结果，评测最终的精度等。

在运行llmc程序之前，还需要安装做了[llmc适配的opencompass](https://github.com/ModelTC/opencompass/tree/opencompass-llmc)

```
git clone https://github.com/ModelTC/opencompass.git -b opencompass-llmc
cd opencompass
pip install -v -e .
pip install human-eval
```

根据opencompass的[文档](https://opencompass.readthedocs.io/zh-cn/latest/get_started/installation.html#id2)，做好数据集的准备，将数据集，放在你执行命令的当前目录

最后你就可以像运行一个正常的llmc程序一样，载入上述的config，进行模型压缩和精度测试

## 多卡并行测试

如果模型太大，单卡评测放不下，需要使用多卡评测精度，我们支持在运行opencompass时使用pipeline parallel，即PP并行。

你需要做的仅仅就是：

1. 确定哪些卡是可用的，在你的运行脚本最前面，添加到CUDA_VISIBLE_DEVICES中

2. 修改opencompass下的cfg_path指向的文件，将里面的num_gpus设置成你需要的数量
