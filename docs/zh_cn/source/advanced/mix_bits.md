# 层间混合比特量化

llmc目前支持了层间混合比特量化，可以实现任意程度的混合。

以下是一些设置样例：

1. 模型整体实现4bit weight-only量化，对所有的down_proj实现8bit weight-only量化

```
quant:
    method: Awq
    weight:
        bit: 4
        symmetric: False
        granularity: per_group
        group_size: 128
    mix_bits:
        setting_0:
            layer_name: [down_proj]
            do_quant: True
            weight:
                bit: 8
                symmetric: False
                granularity: per_group
                group_size: 128
```

2. 模型整体实现4bit weight-only量化，对第0,1,2,3,28,29,30,31个block内的down_proj实现8bit weight-only量化，对所有的o_proj不量化

```
quant:
    method: Awq
    weight:
        bit: 4
        symmetric: False
        granularity: per_group
        group_size: 128
    mix_bits:
        setting_0:
            layer_name: [down_proj#0-1-2-3-28-29-30-31]
            do_quant: True
            weight:
                bit: 8
                symmetric: False
                granularity: per_group
                group_size: 128
        setting_1:
            layer_name: [o_proj]
            do_quant: False
```

3. 模型整体实现W4A4量化，对所有的down_proj实现W8A8量化

```
quant:
    method: Awq
    weight:
        bit: 4
        symmetric: False
        granularity: per_channel
    act:
        bit: 4
        symmetric: False
        granularity: per_token
    mix_bits:
        setting_0:
            layer_name: [down_proj]
            do_quant: True
            weight:
                bit: 8
                symmetric: False
                granularity: per_channel
            act:
                bit: 8
                symmetric: False
                granularity: per_token
```

4. 一个足够混乱的设置，可能没有现实意义

```
quant:
    method: Awq
    weight:
        bit: 4
        symmetric: False
        granularity: per_channel
    act:
        bit: 4
        symmetric: False
        granularity: per_token
    mix_bits:
        setting_0:
            layer_name: [down_proj#0-1-8-15]
            do_quant: True
            weight:
                bit: 8
                symmetric: False
                granularity: per_channel
            act:
                bit: 8
                symmetric: False
                granularity: per_token
        setting_1:
            layer_name: [down_proj#2-6-4-11, o_proj#2-7]
            do_quant: False
        setting_2:
            layer_name: [down_proj#27]
            do_quant: True
            weight:
                bit: 6
                symmetric: False
                granularity: per_channel
            act:
                bit: 6
                symmetric: False
                granularity: per_token
        setting_3:
            layer_name: [down_proj#13-21]
            do_quant: True
            weight:
                bit: 4
                symmetric: False
                granularity: per_channel
```
