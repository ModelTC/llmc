# 自定义校准数据集

llmc目前支持以下几种校准数据集

1. pileval

2. wikitext2

3. c4

4. ptb

5. custom

其中custom表示使用用户自定义的校准数据集。某些特定场景下的专有模型，量化的时候的校准数据使用该场景下的数据更为合适。下面是一个配置的例子。

```
calib:
    name: custom
    download: False
    load_from_txt: True
    path: 自定义数据集，以txt为后缀结尾
    n_samples: 128
    bs: -1
    seq_len: 512
    preproc: random_truncate_txt
    seed: *seed
```

用户可以将一条一条数据文本，写到txt文件里面，每一行代表一条文本数据，使用上述的配置，可以实现自定义数据集的校准。
