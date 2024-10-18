import gc
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
from datasets import load_dataset, load_from_disk
from loguru import logger


class TokenConsistencyEval:
    def __init__(self, tokenizer, eval_cfg):
        self.tokenizer = tokenizer
        # eval_cfg
        logger.info(f'eval_cfg : {eval_cfg}')
        self.dataset = eval_cfg['name']
        assert self.dataset in [
            'wikitext2',
            'c4',
            'ptb',
        ], 'Token consistency eval only supports wikitext2, c4, ptb datasets now.'
        self.seq_len = eval_cfg['seq_len']
        self.bs = eval_cfg['bs']
        self.path = eval_cfg.get('path', None)
        self.download = eval_cfg['download']
        self.inference_per_block = eval_cfg.get('inference_per_block', False)
        self.testenc = self.build_data()

    @torch.no_grad()
    def build_data(self):
        # load data
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
            assert self.path, 'Please set path in eval_cfg.'
            testdata = load_from_disk(self.path)

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
        return testenc

    @torch.no_grad()
    def eval(self, model_llmc_1, model_llmc_2):
        model1 = model_llmc_1.get_model()
        model2 = model_llmc_2.get_model()

        if self.inference_per_block:
            handles1 = []
            handles2 = []
            for layer in model_llmc_1.get_blocks():
                handles1.append(layer.register_forward_pre_hook(self.forward_pre_hook))
                handles1.append(layer.register_forward_hook(self.forward_hook))
            for layer in model_llmc_2.get_blocks():
                handles2.append(layer.register_forward_pre_hook(self.forward_pre_hook))
                handles2.append(layer.register_forward_hook(self.forward_hook))
            for layer in model_llmc_1.get_layers_except_blocks():
                layer.cuda()
            for layer in model_llmc_2.get_layers_except_blocks():
                layer.cuda()
        else:
            model1.cuda()
            model2.cuda()

        model1.eval()
        model2.eval()

        consistency = self.eval_token_consistency(model1, model2, self.testenc, self.seq_len, self.bs)

        if self.inference_per_block:
            for h in handles1 + handles2:
                h.remove()

        model1.cpu()
        model2.cpu()
        gc.collect()
        torch.cuda.empty_cache()
        return consistency

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

    @torch.no_grad()
    def eval_token_consistency(self, model1, model2, testenc, seq_len, bs):
        testenc = testenc.input_ids
        nsamples = testenc.numel() // seq_len

        consistent_tokens = 0
        total_tokens = 0

        # Loop through each batch
        for i in range(0, nsamples, bs):
            logger.info(f'index : {(i + 1) // bs}/{nsamples // bs}')
            # Calculate end index
            j = min(i + bs, nsamples)

            # Prepare inputs and move to gpu
            inputs = testenc[:, (i * seq_len): (j * seq_len)].cuda()
            inputs = inputs.reshape(j - i, seq_len)

            # Forward pass through the models
            logits1 = model1(inputs).logits
            logits2 = model2(inputs).logits

            # Get predicted tokens
            preds1 = torch.argmax(logits1, dim=-1)
            preds2 = torch.argmax(logits2, dim=-1)

            # Compare tokens for consistency
            consistent_tokens += (preds1 == preds2).sum().item()
            total_tokens += preds1.numel()

        # Calculate consistency ratio
        consistency_ratio = consistent_tokens / total_tokens

        # Empty CUDA cache to save memory
        testenc.cpu()
        torch.cuda.empty_cache()

        return consistency_ratio


if __name__ == '__main__':
    import sys

    sys.path.append('../../')
    import argparse

    from llmc.data import BaseTokenizer
    from llmc.models import Llama
    from llmc.utils.registry_factory import MODEL_REGISTRY

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type_1', type=str, required=True)
    parser.add_argument('--model_path_1', type=str, required=True)
    parser.add_argument('--model_type_2', type=str, required=True)
    parser.add_argument('--model_path_2', type=str, required=True)
    args = parser.parse_args()

    tokenizer = BaseTokenizer(args.model_path_1)
    model1 = MODEL_REGISTRY[args.model_type_1](args.model_path_1, 'auto')
    model2 = MODEL_REGISTRY[args.model_type_2](args.model_path_2, 'auto')

    # Llama2-70B config example
    eval_cfg = {
        'name': 'wikitext2',
        'seq_len': 2048,
        'bs': 20,
        'download': False,
        'path': 'data_path',
        'inference_per_block': True,
    }
    token_consistency_eval = TokenConsistencyEval(tokenizer.get_tokenizer(), eval_cfg)

    consistency_ratio = token_consistency_eval.eval(model1, model2)
    logger.info(f'Token consistency ratio: {consistency_ratio}')
