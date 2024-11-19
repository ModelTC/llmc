# Vit quant and img datatsets

llmc currently supports the use of image datasets for calibration and quantification of Vit models

## Vit quant

Here is an example configuration:

```yaml
model:
    type: Vit
    path: /models/vit-base-patch16-224
    torch_dtype: auto
calib:
    name: imagenet
    type: img
    download: False
    path: img calib datasets path
    n_samples: 32
    bs: 1
    seq_len: 512      # Useless arguments for vit
    preproc: img_general
    seed: *seed
eval:
    eval_pos: [pretrain, fake_quant]
    name: imagenet
    type: acc         # acc: accracy 
    download: False
    path: img datasets path
    seq_len: 2048     # Useless arguments for vit
    bs: 1
    inference_per_block: False
    eval_token_consist: False
```

## img datatsets
IMG dataset format requirements: There are images in the IMG dataset directory

The format of the img dataset is as follows:
```
images/
├── image1.jpg
├── image2.jpg
├── image3.jpg
└── ... (other images)
```
