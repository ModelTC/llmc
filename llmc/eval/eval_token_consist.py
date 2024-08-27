import gc
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
from datasets import load_dataset, load_from_disk
from loguru import logger
from .eval_base import BaseEval

class TokenConsistencyEval(BaseEval):

    @torch.no_grad()
    def eval_func(self, model1, model2):
        self.testenc = self.testenc.input_ids
        nsamples = self.testenc.numel() // self.seq_len

        consistent_tokens = 0
        total_tokens = 0

        # Loop through each batch
        for i in range(0, nsamples, self.bs):
            logger.info(f'index : {(i + 1) // self.bs}/{nsamples // self.bs}')
            # Calculate end index
            j = min(i + self.bs, nsamples)

            # Prepare inputs and move to gpu
            inputs = self.testenc[:, (i * self.seq_len): (j * self.seq_len)].cuda()
            inputs = inputs.reshape(j - i, self.seq_len)

            # Forward pass through the models
            logits1 = model1(inputs).logits
            logits2 = model2(inputs).logits

            # Get predicted tokens
            preds1 = torch.argmax(logits1, dim=-1)
            preds2 = torch.argmax(logits2, dim=-1)

            consistent_tokens += (preds1 == preds2).sum().item()
            total_tokens += preds1.numel()

        # Calculate consistency ratio
        consistency_ratio = consistent_tokens / total_tokens

        # Empty CUDA cache to save memory
        self.testenc.cpu()
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

    tokenizer = BaseTokenizer(args.model_path_1, tokenizer_mode="slow")
    model1 = MODEL_REGISTRY[args.model_type_1](args.model_path_1, 'auto')
    model2 = MODEL_REGISTRY[args.model_type_2](args.model_path_2, 'auto')

    eval_cfg = {
        'name': 'wikitext2',
        'seq_len': 2048,
        'bs': 1,
        'download': False,
        'path': "data_path",
        'inference_per_block': False,
    }
    token_consistency_eval = TokenConsistencyEval(tokenizer.get_tokenizer(), eval_cfg)

    consistency_ratio = token_consistency_eval.eval(model1, model2)
    logger.info(f'Token consistency ratio: {consistency_ratio}')