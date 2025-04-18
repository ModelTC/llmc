# VLM Quantization and custom_mm Datasets

llmc currently supports calibrating and quantizing VLM models using custom_mm datasets.

## VLM Quantization
The currently supported models are:  
1. llava  
2. intervl2  
3. llama3.2  
4. qwen2vl  

More VLM models are under development.

Here is an example configuration. You can refer to the [Calibration Dataset Template](https://github.com/user-attachments/files/18433608/general_custom_data_examples.zip) on GitHub.:

```yaml
model:
    type: Llava
    path: model path
    tokenizer_mode: slow
    torch_dtype: auto
calib:
    name: custom_mm
    download: False
    path: calib data path
    apply_chat_template: True
    add_answer: True # Defalut is False. If set it to Ture, calib data will add answers.
    n_samples: 8
    bs: -1
    seq_len: 512
    padding: True
```

## custom_mm datatsets
The format of the custom_mm dataset is as follows:
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
        "image": "images/0a3035bfca2ab920.jpg",
        "question": "Is this an image of Ortigia? Please answer yes or no.",
        "answer": "Yes"
    },
    {
        "image": "images/0a3035bfca2ab920.jpg",
        "question": "Is this an image of Montmayeur castle? Please answer yes or no.",
        "answer": "No"
    },
    {
        "image": "images/0ab2ed007db301d5.jpg",
        "question": "Is this a picture of Highgate Cemetery? Please answer yes or no.",
        "answer": "Yes"
    }
]
```
The "answer" field is optional.
The custom_mm dataset can include calibration data that contains only text (except for llama3.2).

## VLM Evaluation

LLMC integrates [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) for evaluations on various downstream datasets. In the config's eval section, the type should be specified as "vqa", and the downstream evaluation datasets in the name should follow the standards set by lmms-eval.

```
eval:
    type: vqa
    name: [mme] # vqav2, gqa, vizwiz_vqa, scienceqa, textvqa
```
