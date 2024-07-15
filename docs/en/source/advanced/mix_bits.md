# Layerwise mixed bits quantization

llmc currently supports layerwise mixed bit quantization, which can achieve any degree of mixing.

Here are some sample settings:

1. The model as a whole implements 4-bit weight-only quantification, and all down_proj implements 8-bit weight-only quantification.

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

2. The model as a whole implements 4-bit weight-only quantization, 8-bit weight-only quantification is implemented for down_proj in the 0, 1, 2, 3, 28, 29, 30, and 31 blocks, and all o_proj are not quantified.

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

3. The model as a whole implements W4A4 quantification, and all down_proj implements W8A8 quantification.

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

4. A mixing enough config that it may not make practical sense.

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
