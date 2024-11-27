import json
import os
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
    return samples


@PREPROC_REGISTRY
def vlm_general(calib_dataset, n_samples):
    img_qa_json = os.path.join(calib_dataset, 'img_qa.json')
    fp = open(img_qa_json)
    img_qas = json.load(fp)
    for idx in range(len(img_qas)):
        if 'img' in img_qas[idx]:
            if isinstance(img_qas[idx]['img'], list):
                for img_idx in range(len(img_qas[idx]['img'])):
                    img_qas[idx]['img'][img_idx] = os.path.join(calib_dataset, img_qas[idx]['img'][img_idx]) # noqa
            else:
                img_qas[idx]['img'] = os.path.join(calib_dataset, img_qas[idx]['img'])
        else:
            img_qas[idx]['img'] = None
        if 'answer' not in img_qas[idx]:
            img_qas[idx]['answer'] = ''
    random.shuffle(img_qas)
    if len(img_qas) > n_samples:
        img_qas = img_qas[:n_samples]
    return img_qas


@PREPROC_REGISTRY
def alm_general(calib_dataset, n_samples):
    audio_qa_json = os.path.join(calib_dataset, 'audio_qa.json')
    fp = open(audio_qa_json)
    audio_qas = json.load(fp)
    for idx in range(len(audio_qas)):
        if 'audio' in audio_qas[idx]:
            if isinstance(audio_qas[idx]['audio'], list):
                for audio_idx in range(len(audio_qas[idx]['audio'])):
                    audio_qas[idx]['audio'][audio_idx] = os.path.join(calib_dataset, audio_qas[idx]['audio'][audio_idx]) # noqa
            else:
                audio_qas[idx]['audio'] = os.path.join(calib_dataset, audio_qas[idx]['audio'])
        else:
            audio_qas[idx]['audio'] = None
        if 'answer' not in audio_qas[idx]:
            audio_qas[idx]['answer'] = ''
    random.shuffle(audio_qas)
    if len(audio_qas) > n_samples:
        audio_qas = audio_qas[:n_samples]
    return audio_qas


@PREPROC_REGISTRY
def avlm_general(calib_dataset, n_samples):
    audio_img_qa_json = os.path.join(calib_dataset, 'audio_img_qa.json')
    fp = open(audio_img_qa_json)
    audio_img_qas = json.load(fp)
    for idx in range(len(audio_img_qas)):
        if 'audio' in audio_img_qas[idx]:
            if isinstance(audio_img_qas[idx]['audio'], list):
                for audio_idx in range(len(audio_img_qas[idx]['audio'])):
                    audio_img_qas[idx]['audio'][audio_idx] = os.path.join(calib_dataset, audio_img_qas[idx]['audio'][audio_idx]) # noqa
            else:
                audio_img_qas[idx]['audio'] = os.path.join(calib_dataset, audio_img_qas[idx]['audio']) # noqa
        else:
            audio_img_qas[idx]['audio'] = None
        if 'img' in audio_img_qas[idx]:
            if isinstance(audio_img_qas[idx]['img'], list):
                for img_idx in range(len(audio_img_qas[idx]['img'])):
                    audio_img_qas[idx]['img'][img_idx] = os.path.join(calib_dataset, audio_img_qas[idx]['img'][img_idx]) # noqa
            else:
                audio_img_qas[idx]['img'] = os.path.join(calib_dataset, audio_img_qas[idx]['img'])
        else:
            audio_img_qas[idx]['img'] = None
        if 'question' not in audio_img_qas[idx]:
            audio_img_qas[idx]['question'] = ''
        if 'answer' not in audio_img_qas[idx]:
            audio_img_qas[idx]['answer'] = ''
    random.shuffle(audio_img_qas)
    if len(audio_img_qas) > n_samples:
        audio_img_qas = audio_img_qas[:n_samples]
    return audio_img_qas


@PREPROC_REGISTRY
def img_general(calib_dataset, tokenizer, batch_process, n_samples):
    random.shuffle(calib_dataset)
    if len(calib_dataset) > n_samples:
        calib_dataset = calib_dataset[:n_samples]
    samples = batch_process(calib_dataset)
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


@PREPROC_REGISTRY
def original_txt(calib_dataset, tokenizer, n_samples, seq_len=None):
    random.shuffle(calib_dataset)
    n_samples = min(n_samples, len(calib_dataset))
    samples = []
    for i in range(n_samples):
        trainenc = tokenizer(calib_dataset[i], return_tensors='pt')
        inp = trainenc.input_ids
        samples.append(inp)
    return samples


@PREPROC_REGISTRY
def ultrachat_general(calib_dataset, tokenizer, n_samples, seq_len):
    calib_dataset = calib_dataset.shuffle(seed=42).select(range(n_samples))
    texts = []
    samples = []
    for example in calib_dataset:
        text = tokenizer.apply_chat_template(
            example['messages'],
            tokenize=False,
        )
        texts.append(text)

    for i in range(n_samples):
        trainenc = tokenizer(
            texts[i],
            padding=False,
            max_length=seq_len,
            truncation=True,
            add_special_tokens=False,
            return_tensors='pt'
        )
        inp = trainenc.input_ids
        samples.append(inp)
    return samples
