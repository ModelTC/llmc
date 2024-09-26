import random

import torch

from llmc.utils.registry_factory import PREPROC_REGISTRY


@PREPROC_REGISTRY
def wikitext2_gptq(calib_dataset, tokenizer, n_samples, seq_len):
    trainenc = tokenizer('\n\n'.join(calib_dataset['text']), return_tensors='pt')
    samples = []
    for _ in range(n_samples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seq_len - 1)
        j = i + seq_len
        inp = trainenc.input_ids[:, i:j]
        samples.append(inp)
    return samples


@PREPROC_REGISTRY
def ptb_gptq(calib_dataset, tokenizer, n_samples, seq_len):
    trainenc = tokenizer(' '.join(calib_dataset['sentence']), return_tensors='pt')
    samples = []
    for _ in range(n_samples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seq_len - 1)
        j = i + seq_len
        inp = trainenc.input_ids[:, i:j]
        samples.append(inp)
    return samples


@PREPROC_REGISTRY
def c4_gptq(calib_dataset, tokenizer, n_samples, seq_len):
    samples = []
    for _ in range(n_samples):
        while True:
            i = random.randint(0, len(calib_dataset) - 1)
            trainenc = tokenizer(calib_dataset[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seq_len:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seq_len - 1)
        j = i + seq_len
        inp = trainenc.input_ids[:, i:j]
        samples.append(inp)
    return samples


@PREPROC_REGISTRY
def pileval_awq(calib_dataset, tokenizer, n_samples, seq_len):
    dataset = calib_dataset.shuffle(seed=42)
    samples = []
    n_run = 0
    for data in dataset:
        line = data['text']
        line = line.strip()
        line_encoded = tokenizer.encode(line)
        if len(line_encoded) > seq_len:
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break
    samples = torch.cat(samples, dim=1)
    n_split = samples.shape[1] // seq_len
    samples = [samples[:, i * seq_len: (i + 1) * seq_len] for i in range(n_split)]
    return samples


@PREPROC_REGISTRY
def pileval_smooth(calib_dataset, tokenizer, n_samples, seq_len):
    dataset = calib_dataset.shuffle(seed=42)
    samples = []
    n_run = 0
    for data in dataset:
        line = data['text']
        trainenc = tokenizer(
            line, return_tensors='pt', max_length=seq_len, truncation=True
        )
        line_encoded = trainenc.input_ids
        samples.append(line_encoded)
        n_run += 1
        if n_run == n_samples:
            break
    return samples


@PREPROC_REGISTRY
def pileval_omni(calib_dataset, tokenizer, n_samples, seq_len):
    trainenc = tokenizer('\n\n'.join(calib_dataset['text'][:1000]), return_tensors='pt')
    samples = []
    for _ in range(n_samples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seq_len - 1)
        j = i + seq_len
        inp = trainenc.input_ids[:, i:j]
        samples.append(inp)
    return samples, None


@PREPROC_REGISTRY
def vlm_native(calib_dataset, processor, n_samples):
    random.shuffle(calib_dataset)
    samples = {
        'prompts': [],
        'raw_images': []
    }
    n_run = 0
    for data in calib_dataset:
        prompt, raw_image = data
        samples['prompts'].append(prompt)
        samples['raw_images'].append(raw_image)
        n_run += 1
        if n_run == n_samples:
            break
    return samples


@PREPROC_REGISTRY
def vlm_divide_equal(calib_dataset, processor, n_samples):
    samples_native = vlm_native(calib_dataset, processor, n_samples)
    inputs = processor('\n\n'.join(samples_native['prompts']), return_tensors='pt')
    samples = {
        'prompts': [],
        'raw_images': []
    }
    total_len = inputs.input_ids.shape[1]
    seq_len = total_len // n_samples
    for i in range(n_samples):
        s = i * seq_len
        e = (i + 1) * seq_len
        token_ids = inputs.input_ids[:, s:e]
        prompt = processor.decode(token_ids.squeeze(), skip_special_tokens=True)
        prompt = prompt.replace('USER:', 'USER: <image>')
        samples['prompts'].append(prompt)
        samples['raw_images'].append(samples_native['raw_images'][i])
    return samples


@PREPROC_REGISTRY
def vlm_clip_min(calib_dataset, processor, n_samples):
    samples_native = vlm_native(calib_dataset, processor, n_samples)
    samples = {
        'prompts': [],
        'raw_images': []
    }
    trainenc = [
        processor(prompt, return_tensors='pt')
        for prompt in samples_native['prompts']
    ]
    min_len = min(enc.input_ids.shape[1] for enc in trainenc)
    for i in range(n_samples):
        token_ids = trainenc[i].input_ids[:, :min_len]
        prompt = processor.decode(token_ids.squeeze(), skip_special_tokens=True)
        prompt = prompt.replace('USER:', 'USER: <image>')
        samples['prompts'].append(prompt)
        samples['raw_images'].append(samples_native['raw_images'][i])
    return samples


@PREPROC_REGISTRY
def random_truncate_txt(calib_dataset, tokenizer, n_samples, seq_len):
    random.shuffle(calib_dataset)
    trainenc = tokenizer('\n\n'.join(calib_dataset), return_tensors='pt')
    samples = []
    for _ in range(n_samples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seq_len - 1)
        j = i + seq_len
        inp = trainenc.input_ids[:, i:j]
        samples.append(inp)
    return samples
