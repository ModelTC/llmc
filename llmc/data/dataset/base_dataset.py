import json
import os
from abc import ABCMeta

import torch
from datasets import load_dataset, load_from_disk
from loguru import logger
from PIL import Image
from torch.nn import functional as F

from .specified_preproc import PREPROC_REGISTRY


class BaseDataset(metaclass=ABCMeta):
    def __init__(self, tokenizer, calib_cfg, batch_process=None):
        # calib_cfg
        logger.info(f'calib_cfg : {calib_cfg}')
        self.tokenizer = tokenizer
        self.batch_process = batch_process
        self.calib_dataset_name = calib_cfg['name']
        self.calib_dataset_type = calib_cfg.get('type', 'txt')
        self.padding = calib_cfg.get('padding', False)
        if self.calib_dataset_name == 'ultrachat':
            assert self.padding
        self.download = calib_cfg['download']
        self.load_from_txt = calib_cfg.get('load_from_txt', False)
        self.calib_dataset_path = calib_cfg.get('path', None)
        self.n_samples = calib_cfg['n_samples']
        self.calib_bs = calib_cfg['bs']
        self.seq_len = calib_cfg.get('seq_len', None)
        self.preproc = calib_cfg['preproc']
        if self.calib_dataset_name == 'ultrachat':
            assert self.preproc == 'ultrachat_general'
        if self.preproc == 'original_txt':
            assert self.seq_len is None
        self.seed = calib_cfg['seed']
        self.dataset_key = {
            'pileval': 'text',
            'c4': 'text',
            'wikitext2': 'text',
            'ptb': 'sentence',
        }
        if self.calib_dataset_name in self.dataset_key:
            self.key = self.dataset_key[self.calib_dataset_name]
        self.build_calib_dataset()

    def build_calib_dataset(self):
        if self.calib_dataset_type == 'txt':
            if self.download:
                if self.calib_dataset_name == 'pileval':
                    self.calib_dataset = load_dataset(
                        'mit-han-lab/pile-val-backup', split='validation'
                    )
                elif self.calib_dataset_name == 'c4':
                    self.calib_dataset = load_dataset(
                        'allenai/c4',
                        data_files={'train': 'en/c4-train.00000-of-01024.json.gz'},
                        split='train',
                    )
                elif self.calib_dataset_name == 'wikitext2':
                    self.calib_dataset = load_dataset(
                        'wikitext', 'wikitext-2-raw-v1', split='train'
                    )
                elif self.calib_dataset_name == 'ptb':
                    self.calib_dataset = load_dataset(
                        'ptb_text_only', 'penn_treebank', split='train'
                    )
                elif self.calib_dataset_name == 'ultrachat':
                    self.calib_dataset = load_dataset(
                        'HuggingFaceH4/ultrachat_200k', split='train_sft'
                    )
                else:
                    raise Exception(f'Not support {self.calib_dataset_name} dataset.')
            else:
                if not self.load_from_txt:
                    # Need to pre-download the dataset.
                    self.calib_dataset = load_from_disk(self.calib_dataset_path)
                else:
                    """Load dataset from your custom txt file.

                    Each line in the txt file represents one input text data.
                    """
                    assert self.calib_dataset_path.endswith('.txt')
                    logger.info(f'calib_dataset_path: {self.calib_dataset_path}')
                    with open(self.calib_dataset_path, 'r') as fp:
                        lines = fp.readlines()
                    self.calib_dataset = []
                    for line in lines:
                        self.calib_dataset.append(line.strip())
        elif self.calib_dataset_type == 'img_txt':
            logger.info(f'calib_dataset_path: {self.calib_dataset_path}')
            self.calib_dataset = self.calib_dataset_path
        elif self.calib_dataset_type == 'audio_txt':
            logger.info(f'calib_dataset_path: {self.calib_dataset_path}')
            self.calib_dataset = self.calib_dataset_path
        elif self.calib_dataset_type == 'audio_img_txt':
            logger.info(f'calib_dataset_path: {self.calib_dataset_path}')
            self.calib_dataset = self.calib_dataset_path
        elif self.calib_dataset_type == 'img':
            self.calib_dataset = []
            logger.info(f'calib_dataset_path: {self.calib_dataset_path}')
            for root, _, files in os.walk(self.calib_dataset_path):
                for name in files:
                    if name.endswith(('.jpg', '.png', '.JPEG')):
                        img_path = os.path.join(root, name)
                        raw_image = Image.open(img_path).convert('RGB')
                        self.calib_dataset.append(raw_image)
        else:
            raise ValueError(f'Unsupported data type: {self.calib_dataset_type}')

    def get_calib_samples(self):
        if self.preproc == 'general':
            samples = self.general_preproc(
                self.calib_dataset, self.tokenizer, self.n_samples, self.seq_len
            )
        elif self.preproc.startswith(('vlm_', 'alm_', 'avlm_', 'img_')):
            preproc = PREPROC_REGISTRY[self.preproc]
            samples = preproc(
                self.calib_dataset,
                self.n_samples
            )
        else:
            preproc = PREPROC_REGISTRY[self.preproc]
            samples = preproc(
                self.calib_dataset, self.tokenizer,
                self.n_samples, self.seq_len
            )
        return samples

    def get_pad_setting(self, length):
        if self.tokenizer.padding_side == 'left':
            return [length, 0]
        elif self.tokenizer.padding_side == 'right':
            return [0, length]
        else:
            raise Exception(f'Not support padding_side: {self.tokenizer.padding_side}.')

    def txt_group_samples_with_mask(self, samples):
        calib_samples = []
        input_ids = []
        attention_mask = []
        pad_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        if self.calib_bs < 0:
            samples_len = [sample.shape[-1] for sample in samples]
            max_len = max(samples_len)
            samples_tmp = []
            attention_mask_tmp = []
            for sample in samples:
                samples_tmp.append(
                    F.pad(
                        sample,
                        self.get_pad_setting(max_len - sample.shape[-1]),
                        value=pad_token_id
                    )
                )
                attention_mask_tmp.append(
                    F.pad(
                        torch.ones(1, sample.shape[-1], dtype=torch.int64),
                        self.get_pad_setting(max_len - sample.shape[-1]),
                        value=0
                    )
                )
            batch_input_ids = torch.cat(samples_tmp, dim=0)
            batch_attention_mask = torch.cat(attention_mask_tmp, dim=0)
            calib_samples.append(
                {'input_ids': batch_input_ids, 'attention_mask': batch_attention_mask}
            )
        elif self.calib_bs == 1:
            input_ids = samples
            attention_mask = [torch.ones(1, sample.shape[-1], dtype=torch.int64) for sample in samples] # noqa
            for i in range(len(samples)):
                calib_samples.append(
                    {'input_ids': input_ids[i], 'attention_mask': attention_mask[i]}
                )
        elif self.calib_bs > 1:
            for i in range(0, len(samples), self.calib_bs):
                start = i
                end = min(i + self.calib_bs, len(samples))
                batch_samples = samples[start:end]
                batch_samples_len = [sample.shape[-1] for sample in batch_samples]
                batch_max_len = max(batch_samples_len)
                samples_tmp = []
                attention_mask_tmp = []
                for sample in batch_samples:
                    samples_tmp.append(
                        F.pad(
                            sample,
                            self.get_pad_setting(batch_max_len - sample.shape[-1]),
                            value=pad_token_id
                        )
                    )
                    attention_mask_tmp.append(
                        F.pad(
                            torch.ones(1, sample.shape[-1], dtype=torch.int64),
                            self.get_pad_setting(batch_max_len - sample.shape[-1]),
                            value=0
                        )
                    )
                batch_input_ids = torch.cat(samples_tmp, dim=0)
                batch_attention_mask = torch.cat(attention_mask_tmp, dim=0)
                calib_samples.append(
                    {
                        'input_ids': batch_input_ids,
                        'attention_mask': batch_attention_mask
                    }
                )
        return calib_samples

    def txt_group_samples_wo_mask(self, samples):  # without mask
        calib_samples = []
        if self.calib_bs < 0:
            batch = torch.cat(samples, dim=0)
            calib_samples.append({'input_ids': batch})
        elif self.calib_bs == 1:
            for i in range(len(samples)):
                calib_samples.append({'input_ids': samples[i]})
        elif self.calib_bs > 1:
            for i in range(0, len(samples), self.calib_bs):
                start = i
                end = min(i + self.calib_bs, len(samples))
                batch = samples[start:end]
                batch = torch.cat(batch, dim=0)
                calib_samples.append({'input_ids': batch})
        return calib_samples

    def img_txt_group_samples_with_mask(self, samples):
        calib_samples = []
        if self.calib_bs < 0:
            calib_samples.append(self.batch_process(samples, calib_or_eval='calib'))
        elif self.calib_bs == 1:
            calib_samples = [self.batch_process([sample], calib_or_eval='calib') for sample in samples] # noqa
        elif self.calib_bs > 1:
            for i in range(0, len(samples), self.calib_bs):
                start = i
                end = min(i + self.calib_bs, len(samples))
                batch = samples[start:end]
                calib_samples.append(self.batch_process(batch, calib_or_eval='calib'))
        return calib_samples

    def audio_txt_group_samples_with_mask(self, samples):
        calib_samples = []
        if self.calib_bs < 0:
            calib_samples.append(self.batch_process(samples, calib_or_eval='calib'))
        elif self.calib_bs == 1:
            calib_samples = [self.batch_process([sample], calib_or_eval='calib') for sample in samples] # noqa
        elif self.calib_bs > 1:
            for i in range(0, len(samples), self.calib_bs):
                start = i
                end = min(i + self.calib_bs, len(samples))
                batch = samples[start:end]
                calib_samples.append(self.batch_process(batch, calib_or_eval='calib'))
        return calib_samples

    def audio_img_txt_group_samples_with_mask(self, samples):
        calib_samples = []
        if self.calib_bs < 0:
            calib_samples.append(self.batch_process(samples, calib_or_eval='calib'))
        elif self.calib_bs == 1:
            calib_samples = [self.batch_process([sample], calib_or_eval='calib') for sample in samples] # noqa
        elif self.calib_bs > 1:
            for i in range(0, len(samples), self.calib_bs):
                start = i
                end = min(i + self.calib_bs, len(samples))
                batch = samples[start:end]
                calib_samples.append(self.batch_process(batch, calib_or_eval='calib'))
        return calib_samples

    def img_group_samples_wo_mask(self, samples):  # without mask
        calib_samples = []
        if self.calib_bs < 0:
            batch = {'pixel_values': torch.cat([sample['pixel_values']
                                                for sample in samples], dim=0)}
            calib_samples.append(batch)
        elif self.calib_bs == 1:
            calib_samples = samples
        elif self.calib_bs > 1:
            for i in range(0, len(samples), self.calib_bs):
                start = i
                end = min(i + self.calib_bs, len(samples))
                batch = samples[start:end]
                batch = {'pixel_values': torch.cat([sample['pixel_values']
                                                    for sample in batch], dim=0)}
                calib_samples.append(batch)
        return calib_samples

    def get_calib_dataset(self):
        samples = self.get_calib_samples()
        if self.calib_dataset_type in ['txt', 'img', 'img_txt', 'audio_txt']:
            logger.info(f'len(samples) all : {len(samples)}')
            samples = samples[int(os.environ['RANK'])::int(os.environ['WORLD_SIZE'])]
            logger.info(f'len(samples) rank : {len(samples)}')
        calib_samples = []
        if self.calib_dataset_type == 'txt':
            if self.padding:
                calib_samples = self.txt_group_samples_with_mask(samples)
            else:
                calib_samples = self.txt_group_samples_wo_mask(samples)
        elif self.calib_dataset_type == 'img':
            calib_samples = self.img_group_samples_wo_mask(samples)
        elif self.calib_dataset_type == 'img_txt':
            calib_samples = self.img_txt_group_samples_with_mask(samples)
        elif self.calib_dataset_type == 'audio_txt':
            calib_samples = self.audio_txt_group_samples_with_mask(samples)
        elif self.calib_dataset_type == 'audio_img_txt':
            calib_samples = self.audio_img_txt_group_samples_with_mask(samples)
        logger.info(f'len(calib_samples) : {len(calib_samples)}')
        if self.padding:
            padding_mask = [calib_sample['attention_mask'] for calib_sample in calib_samples] # noqa
        else:
            padding_mask = None
        return calib_samples, padding_mask

    def general_preproc(self, calib_dataset, tokenizer, n_samples, seq_len):
        dataset = calib_dataset.shuffle(seed=self.seed)
        samples = []
        n_run = 0
        for data in dataset:
            line = data[self.key]
            trainenc = tokenizer(
                line, return_tensors='pt', max_length=seq_len, truncation=True
            )
            line_encoded = trainenc.input_ids
            if line_encoded.shape[1] < seq_len:
                continue
            samples.append(line_encoded)
            n_run += 1
            if n_run == n_samples:
                break
        return samples
