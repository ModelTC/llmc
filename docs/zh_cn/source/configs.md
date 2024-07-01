# 设计自定义配置

所有的配置均可以在[这里](https://github.com/ModelTC/llmc/tree/main/configs)找到

```
base:
    seed: &seed 42 # 设置随机种子
model:
    type: Llama # 模型的类型
    path: model path # 模型的路径
    tokenizer_mode: fast # 模型的tokenizer类型
    torch_dtype: auto # 模型的dtype
calib:
    name: pileval # 校准数据集名
    download: False # 校准数据集是否在线下载
    path: calib data path # 校准数据集路径
    n_samples: 512 # 校准数据集的数量
    bs: 1 # 校准数据集的batch size
    seq_len: 512 # 校准数据集的长度
    preproc: pileval_smooth # 校准数据集的预处理方式
    seed: *seed # 校准数据集的随机种子
eval:
    eval_pos: [pretrain, transformed, fake_quant] # 评测的位点
    name: wikitext2 # 评测数据集的名字
    download: False # 评测数据集是否在线下载
    path: eval data path # 评测数据集的路径
    bs: 1 # 评测数据集的batch size
    seq_len: 2048 # 评测数据集的长度
quant:
    method: SmoothQuant # 压缩方法
    weight:
        bit: 8 # 权重的量化bit数
        symmetric: True # 权重量化是否是对称量化
        granularity: per_channel # 权重量化的粒度
    act:
        bit: 8 # 激活的量化bit数
        symmetric: True # 激活量化是否是对称量化
        granularity: per_token # 激活量化的粒度
save:
    save_fp: False # 是否保存调整之后的模型
    save_path: ./save # 保存路径
```
