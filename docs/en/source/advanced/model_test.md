# Model accuracy test

## Accuracy test pipeline

LLMC supports basic PPL (Perplexity) evaluation, but more downstream task evaluations are not supported by LLMC itself.

It is common practice to use evaluation tools to directly test the inference of the model, including but not limited to:

1. [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)

2. [opencompass](https://github.com/open-compass/opencompass)

However, this evaluation method is not efficient, so we recommend using the inference engine evaluation tool to separate the model accuracy evaluation, the model is inferred by the inference engine, and served in the form of an API, and the evaluation tool evaluates the API. This approach has the following benefits:


1. Using an efficient inference engine for model inference can speed up the entire evaluation process

2. The reasoning of the model and the evaluation of the model are separated, and each is responsible for its own professional affairs, and the code structure is clearer

3. Using the inference engine to infer a model is more in line with the actual deployment scenario and easier to align with the accuracy of the actual deployment of the model

We recommend and introduce the compression-deployment-evaluation process using the following model: **LLMC compression-lightllm inference-opencompass evaluation**


Here are the links to the relevant tools:

1. llmc, Large language Model Compression Tool, [(GitHub)(https://github.com/ModelTC/llmc), [Doc](https://llmc-zhcn.readthedocs.io/en/latest/)]

2. Lightllm, Large language Model Inference Engine, [[GitHub](https://github.com/ModelTC/lightllm)]

3. OpenCompass, Large language Model Evaluation Tool, [[GitHub]((https://github.com/open-compass/opencompass)), [Doc](https://opencompass.readthedocs.io/zh-cn/latest/)]

## Use of the lightLLM inference engine

The official [lightllm](https://github.com/ModelTC/llmc) repository has more detailed documentation, but here is a simple and quick start

<font color=792ee5> start a service of a float model </font>

**install lightllm**

```
git clone https://github.com/ModelTC/lightllm.git
cd lightllm
pip install -v -e .
```

**start a service**

```
python -m lightllm.server.api_server --model_dir # model path       \
                                     --host 0.0.0.0                 \
                                     --port 1030                    \
                                     --nccl_port 2066               \
                                     --max_req_input_len 6144       \
                                     --max_req_total_len 8192       \
                                     --tp 2                         \
                                     --trust_remote_code            \
                                     --max_total_token_num 120000
```

The above command will serve a 2-card on port 1030 of the machine

The above commands can be set by the number of tp, and TensorParallel inference can be performed on tp cards, which is suitable for inference of larger models.

The max_total_token_num in the above command will affect the throughput performance during the test, and can be set according to the lightllm [documentation](https://github.com/ModelTC/lightllm/blob/main/docs/ApiServerArgs.md). As long as the gpu memory is not exploded, the larger the setting, the better.

If you want to set up multiple lightllm services on the same machine, you need to reset the port and nccl_port above without conflicts.

<font color=792ee5> Simple testing of the service </font>

Execute the following python script

```
import requests
import json

url = 'http://localhost:1030/generate'
headers = {'Content-Type': 'application/json'}
data = {
    'inputs': 'What is AI?',
    "parameters": {
        'do_sample': False,
        'ignore_eos': False,
        'max_new_tokens': 128,
    }
}
response = requests.post(url, headers=headers, data=json.dumps(data))
if response.status_code == 200:
    print(response.json())
else:
    print('Error:', response.status_code, response.text)
```

If the above script returns normally, the service is normal

<font color=792ee5> start a service of a quantization model </font>

```
python -m lightllm.server.api_server --model_dir 模型路径            \
                                     --host 0.0.0.0                 \
                                     --port 1030                    \
                                     --nccl_port 2066               \
                                     --max_req_input_len 6144       \
                                     --max_req_total_len 8192       \
                                     --tp 2                         \
                                     --trust_remote_code            \
                                     --max_total_token_num 120000   \
                                     --mode triton_w4a16
```

Added to the command `--mode triton_w4a16`, indicates that the naive quantization of w4a16 was used

After the service is started, you also need to verify whether the service is normal

The model path used by the above command is the original pre-trained model and has not been adjusted by the llmc. You can follow the LLMC documentation, open the save_trans, save a modified model, and then run the naive quantization service command described above.

## Use of the opencompass evaluation tool

The official [opencompass](https://github.com/open-compass/opencompass) repository has more detailed documentation, but here is a simple and quick start

**install opencompass**

```
git clone https://github.com/open-compass/opencompass.git
cd opencompass
pip install -v -e .
```

**Modify the config**

The config file is [here](https://github.com/open-compass/opencompass/blob/main/configs/eval_lightllm.py), this configuration file is used by OpenCompass to evaluate the accuracy of Lightllm's API service, and it should be noted that the port inside it url should be consistent with the above Lightllm service port


For the selection of the evaluation dataset, you need to modify this part of the code

```
with read_base():
    from .summarizers.leaderboard import summarizer
    from .datasets.humaneval.deprecated_humaneval_gen_a82cae import humaneval_datasets
```

The above code snippet, which represents the test humaneval dataset, can be found here for more dataset testing support

**Dataset download**

It is necessary to prepare the best dataset according to the OpenCompass [documentation](https://opencompass.readthedocs.io/en/latest/get_started/installation.html#dataset-preparation).

**Run accuracy tests**

After modifying the above configuration file, you can run the following command
```
python run.py configs/eval_lightllm.py
```
When the model has completed the inference and metric calculations, we can get the evaluation results of the model. The output folder will be generated in the current directory, the logs subfolder will record the logs in the evaluation, and the summary subfile will record the accuracy of the measured data set

## FAQ

**<font color=red> Q1 </font>** 

What does the dataset configuration file in OpenCompass mean when the same dataset has different suffixes?

**<font color=green> Solution </font>** 

Different suffixes represent different prompt templates, and for detailed OpenCompass questions, please refer to the OpenCompass documentation

**<font color=red> Q2 </font>** 

The test accuracy of the Humaneval of the LLAMA model is too low

**<font color=green> Solution </font>** 

You may need to delete the \n at the end of each entry in the Humaneval jsonl file in the dataset provided by OpenCompass and retest it

**<font color=red> Q3 </font>** 

The test is still not fast enough

**<font color=green> Solution </font>** 

You can consider whether the max_total_token_num parameter settings are reasonable when starting the lightllm service, and if the setting is too small, the test concurrency will be low

