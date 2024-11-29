import gc
import os
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
from datasets import load_dataset, load_from_disk
from human_eval.data import read_problems
from loguru import logger


class BaseEval:
    def __init__(self, tokenizer, config):
        self.tokenizer = tokenizer
        # eval_cfg
        eval_cfg = config.eval
        self.model_type = config.model.type
        logger.info(f'eval_cfg : {eval_cfg}')
        self.dataset = eval_cfg['name']
        assert self.dataset in [
            'wikitext2',
            'c4',
            'ptb',
            'custom',
            'human_eval'
        ], 'Ppl eval only support wikitext2, c4, ptb, human_eval dataset now.'
        self.seq_len = eval_cfg.get('seq_len', None)
        self.bs = eval_cfg['bs']
        self.path = eval_cfg.get('path', None)
        self.download = eval_cfg.get('download', False)
        self.load_from_txt = eval_cfg.get('load_from_txt', False)
        self.inference_per_block = eval_cfg.get('inference_per_block', False)
        self.testenc = self.build_data()
        self.res_path = eval_cfg.get('res_path', None)
        assert self.dataset in ['human_eval'] and self.res_path is not None
        os.makedirs(self.res_path, exist_ok=True)
        self.format_tabs = eval_cfg.get('format_tabs', False)

    @torch.no_grad()
    def build_data(self):
        # load data
        if self.dataset == 'human_eval':
            testenc = read_problems()
        else:
            if self.download:
                if self.dataset == 'wikitext2':
                    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
                elif self.dataset == 'c4':
                    testdata = load_dataset(
                        'allenai/c4',
                        data_files={
                            'validation': 'en/c4-validation.00000-of-00008.json.gz'
                        },
                        split='validation',
                    )
                elif self.dataset == 'ptb':
                    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test')
            else:
                if not self.load_from_txt:
                    assert self.path, 'Please set path in eval_cfg.'
                    testdata = load_from_disk(self.path)
                else:
                    """Load dataset from your custom txt file.

                    Each line in the txt file represents one input text data.
                    """
                    assert self.path.endswith('.txt')
                    logger.info(f'eval dataset path: {self.path}')
                    with open(self.path, 'r') as fp:
                        lines = fp.readlines()
                    testdata = []
                    for line in lines:
                        testdata.append(line.strip())
            # encode data
            if self.dataset == 'wikitext2':
                testenc = self.tokenizer('\n\n'.join(testdata['text']), return_tensors='pt')
            elif self.dataset == 'c4':
                testenc = self.tokenizer(
                    ' '.join(testdata[:1100]['text']), return_tensors='pt'
                )
                testenc.input_ids = testenc.input_ids[:, : (256 * self.seq_len)]
            elif self.dataset == 'ptb':
                testenc = self.tokenizer(
                    ' '.join(testdata['sentence']), return_tensors='pt'
                )
            elif self.dataset == 'custom':
                testenc = self.tokenizer(
                    '\n'.join(testdata), return_tensors='pt'
                )
        return testenc

    @torch.no_grad()
    def forward_pre_hook(self, m, x):
        m.cuda()

    @torch.no_grad()
    def forward_hook(self, m, x, y):
        with ThreadPoolExecutor() as executor:
            executor.submit(self.load_layer_to_cpu, m)

    @torch.no_grad()
    def load_layer_to_cpu(self, m):
        m.cpu()

    def register_hooks(self, model):
        handles = []
        for layer in model.get_blocks():
            handles.append(layer.register_forward_pre_hook(self.forward_pre_hook))
        for layer in model.get_blocks():
            handles.append(layer.register_forward_hook(self.forward_hook))
        for layer in model.get_layers_except_blocks():
            layer.cuda()
        return handles

    @torch.no_grad()
    def eval(self, model_llmc, model_org=None, eval_pos=None):
        handles, handles_org = [], []
        if self.inference_per_block:
            handles = self.register_hooks(model_llmc)
        else:
            model_llmc.model.cuda()
        model_llmc.model.eval()

        if model_org is not None:
            if self.inference_per_block:
                handles_org = self.register_hooks(model_org)
            else:
                model_org.model.cuda()

            model_org.model.eval()

        eval_res = self.eval_func(model_org,
                                  model_llmc,
                                  self.testenc,
                                  self.seq_len,
                                  self.bs,
                                  eval_pos)
        if self.inference_per_block:
            for h in handles + handles_org:
                h.remove()

        model_llmc.model.cpu()
        if model_org is not None:
            model_org.model.cpu()

        gc.collect()
        torch.cuda.empty_cache()
        return eval_res

    def post_process(self, testenc):
        pass
