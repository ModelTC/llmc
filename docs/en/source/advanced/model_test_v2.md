# Model accuracy test V2

In the accuracy testing of Model accuracy test V1, the process was not streamlined enough. We listened to feedback from the community developers and developed Model Accuracy Test V2.

In the V2 version, we no longer need to use an inference engine to start a service, nor do we need to break the testing into multiple steps.

Our goal is to make downstream accuracy testing equivalent to PPL testing. Running a program from llmc will, after completing the algorithm execution, directly conduct PPL testing and simultaneously perform the corresponding downstream accuracy testing.

To achieve the above goals, we only need to add an opencompass setting in the existing configuration.


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
    output_path: ./oc_output
```

The cfg_path under opencompass needs to point to a configuration path for opencompass.

[Here](https://github.com/ModelTC/llmc/tree/main/configs/opencompass), we have provided the configurations for both the base model and the chat model regarding the human-eval test as a reference for everyone.

It is important to note that [the configuration provided by opencompass](https://github.com/ModelTC/opencompass/blob/opencompass-llmc/configs/models/hf_llama/hf_llama3_8b.py) needs to have the path key. However, in this case, we do not need this key because llmc will default to using the model path in the save path of trans

Of course, since the save path of trans model is required, you need to set save_trans to True if you want to test in opencompass.

The output_path under opencompass is used to set the output directory for the evaluation logs of opencompass.

Before running the llmc program, you also need to install the version of [opencompass](https://github.com/ModelTC/opencompass/tree/opencompass-llmc) that has been adapted for llmc.

```
git clone https://github.com/ModelTC/opencompass.git -b opencompass-llmc
cd opencompass
pip install -v -e .
pip install human-eval
```

According to the opencompass [documentation](https://opencompass.readthedocs.io/en/latest/get_started/installation.html#dataset-preparation), prepare the dataset and place it in the current directory where you execute the command.

Finally, you can load the above configuration and perform model compression and accuracy testing just like running a regular llmc program.

Note: If the model is too large to fit on a single GPU for evaluation, and multi-GPU evaluation is needed, we support using pipeline parallelism when running opencompass.

What you need to do is identify which GPUs are available, add them to CUDA_VISIBLE_DEVICES at the beginning of your run script, and then modify the file pointed to by cfg_path under opencompass, setting the num_gpus to the desired number.
