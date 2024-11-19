# VLM quant and img-txt datatsets

llmc目前支持对VLM模型使用图像-文本数据集进行校准并量化

## VLM quant
当前支持的模型如下：
1. llava

2. intervl2

3. llama3.2

4. qwen2vl

更多的vlm正在实现中

下面是一个配置的例子

```yaml
model:
    type: Llava
    path: model path
    tokenizer_mode: slow
    torch_dtype: auto
calib:
    name: vlm_datastes
    type: img_txt
    download: False
    path: datastes path
    n_samples: 32
    bs: 1
    seq_len: 512
    preproc: vlm_general
    padding: True
    seed: *seed
```

## img-txt datatsets
img-txt 数据集格式如下：
```
img_txt-datasets/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   ├── image3.jpg
│   └── ... (other images)
└── img_qa.json
```

img_qa.json 格式示例：
```json
[
    {
        "img": "images/00000000.jpg",
        "question": "Is this picture captured in a place of ice floe? Please answer yes or no.",
        "answer": "Yes"
    },
    {
        "img": "images/00000000.jpg",
        "question": "Is this picture captured in a place of closet? Please answer yes or no.",
        "answer": "No"
    },
]
```
"answer" 可以不需要

img-txt数据集中可以存在仅有文本的校准数据（当前llama3.2除外）