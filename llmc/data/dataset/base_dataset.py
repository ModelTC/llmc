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
        self.padding = calib_cfg.get('padding', False)
        if self.calib_dataset_name == 'ultrachat':
            assert self.padding
        self.download = calib_cfg['download']
        self.load_from_txt = calib_cfg.get('load_from_txt', False)
        self.calib_dataset_path = calib_cfg.get('path', None)
        self.apply_chat_template = calib_cfg.get('apply_chat_template', False)
        self.n_samples = calib_cfg['n_samples']
        self.calib_bs = calib_cfg['bs']
        self.seq_len = calib_cfg.get('seq_len', None)
        self.preproc = calib_cfg.get('preproc', False)
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
            if self.calib_dataset_name == 'custom_txt' or self.calib_dataset_name == 'custom_mm':
                self.calib_dataset = self.get_cutomdata(self.calib_dataset_path)
            else:
                self.calib_dataset = load_from_disk(self.calib_dataset_path)

    def get_calib_samples(self):
        if self.calib_dataset_name == 'custom_txt' or self.calib_dataset_name == 'custom_mm':
            samples = self.calib_dataset
        else:
            preproc = PREPROC_REGISTRY[self.preproc]
            preproc_param_dict = {
                'calib_dataset': self.calib_dataset,
                'tokenizer': self.tokenizer,
                'n_samples': self.n_samples,
                'seq_len': self.seq_len
            }
            if self.preproc == 'txt_general_preproc':
                preproc_param_dict['key'] = self.key
            samples = preproc(**preproc_param_dict)
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

    def get_calib_model_inputs(self, samples):
        if not self.padding:
            assert not self.calib_dataset_name == 'custom_mm'
            if self.calib_dataset_name == 'custom_txt':
                txts = self.batch_process(
                    samples,
                    calib_or_eval='calib',
                    apply_chat_template=self.apply_chat_template,
                    return_inputs=False
                )
            else:
                txts = self.calib_dataset
            preproc = PREPROC_REGISTRY[self.preproc]
            preproc_param_dict = {
                'calib_dataset': txts,
                'tokenizer': self.tokenizer,
                'n_samples': self.n_samples,
                'seq_len': self.seq_len
            }
            if self.preproc == 'txt_general_preproc':
                preproc_param_dict['key'] = self.key
            samples = preproc(**preproc_param_dict)
            calib_model_inputs = []
            if self.calib_bs < 0:
                batch = torch.cat(samples, dim=0)
                calib_model_inputs.append({'input_ids': batch})
            elif self.calib_bs == 1:
                for i in range(len(samples)):
                    calib_model_inputs.append({'input_ids': samples[i]})
            elif self.calib_bs > 1:
                for i in range(0, len(samples), self.calib_bs):
                    start = i
                    end = min(i + self.calib_bs, len(samples))
                    batch = samples[start:end]
                    batch = torch.cat(batch, dim=0)
                    calib_model_inputs.append({'input_ids': batch})
        else:
            assert self.calib_dataset_name == 'custom_txt' or self.calib_dataset_name == 'custom_mm'
            calib_model_inputs = []
            if self.calib_bs < 0:
                calib_model_inputs.append(
                    self.batch_process(
                        samples,
                        calib_or_eval='calib',
                        apply_chat_template=self.apply_chat_template
                    )
                )
            elif self.calib_bs == 1:
                calib_model_inputs = [self.batch_process([sample], calib_or_eval='calib', apply_chat_template=self.apply_chat_template) for sample in samples] # noqa
            elif self.calib_bs > 1:
                for i in range(0, len(samples), self.calib_bs):
                    start = i
                    end = min(i + self.calib_bs, len(samples))
                    batch = samples[start:end]
                    calib_model_inputs.append(
                        self.batch_process(
                            batch,
                            calib_or_eval='calib',
                            apply_chat_template=self.apply_chat_template
                        )
                    )
        return calib_model_inputs

    def get_calib_dataset(self):
        samples = self.calib_dataset[int(os.environ['RANK'])::int(os.environ['WORLD_SIZE'])]
        logger.info(f'len(samples) rank : {len(samples)}')

        calib_model_inputs = self.get_calib_model_inputs(samples)
        logger.info(f'len(calib_model_inputs) : {len(calib_model_inputs)}')
        if self.padding:
            padding_mask = [calib_model_input['attention_mask'] for calib_model_input in calib_model_inputs] # noqa
        else:
            padding_mask = None
        return calib_model_inputs, padding_mask

    def get_cutomdata(self, custom_dataset):
        audio_img_qa_json = os.path.join(custom_dataset, 'samples.json')
        fp = open(audio_img_qa_json)
        custom_data_samples = json.load(fp)
        for idx in range(len(custom_data_samples)):
            if 'audio' in custom_data_samples[idx]:
                if isinstance(custom_data_samples[idx]['audio'], list):
                    for audio_idx in range(len(custom_data_samples[idx]['audio'])):
                        custom_data_samples[idx]['audio'][audio_idx] = os.path.join(
                            custom_dataset, custom_data_samples[idx]['audio'][audio_idx]
                        )
                else:
                    custom_data_samples[idx]['audio'] = os.path.join(
                        custom_dataset, custom_data_samples[idx]['audio']
                    )
            else:
                custom_data_samples[idx]['audio'] = None
            if 'image' in custom_data_samples[idx]:
                if isinstance(custom_data_samples[idx]['image'], list):
                    for img_idx in range(len(custom_data_samples[idx]['image'])):
                        custom_data_samples[idx]['image'][img_idx] = os.path.join(
                            custom_dataset, custom_data_samples[idx]['image'][img_idx]
                        )
                else:
                    custom_data_samples[idx]['image'] = os.path.join(
                        custom_dataset, custom_data_samples[idx]['image']
                    )
            else:
                custom_data_samples[idx]['image'] = None
            if 'question' not in custom_data_samples[idx]:
                custom_data_samples[idx]['question'] = ''
            if 'answer' not in custom_data_samples[idx]:
                custom_data_samples[idx]['answer'] = ''
        return custom_data_samples
