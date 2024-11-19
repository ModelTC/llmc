# VLM Quantization and Image-Text Datasets

llmc currently supports calibrating and quantizing VLM models using image-text datasets.

## VLM Quantization
The currently supported models are:  
1. llava  
2. intervl2  
3. llama3.2  
4. qwen2vl  

More VLM models are under development.

Here is an example configuration:

```yaml
model:
    type: Llava
    path: model path
    tokenizer_mode: slow
    torch_dtype: auto
calib:
    name: vlm_datasets
    type: img_txt
    download: False
    path: datasets path
    n_samples: 32
    bs: 1
    seq_len: 512
    preproc: vlm_general
    padding: True
    seed: *seed
```

## img-txt datatsets
The format of the img-txt dataset is as follows:
```
img_txt-datasets/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   ├── image3.jpg
│   └── ... (other images)
└── img_qa.json
```

Example format of img_qa.json:
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
The "answer" field is optional.
The img-txt dataset can include calibration data that contains only text (except for llama3.2).