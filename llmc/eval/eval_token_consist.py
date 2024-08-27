import gc
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
from datasets import load_dataset, load_from_disk
from loguru import logger

from .eval_base import BaseEval


class TokenConsistencyEval(BaseEval):

    @torch.no_grad()
    def eval_func(self, org_model, model, testenc, seq_len, bs):
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
            logits1 = org_model(inputs).logits
            logits2 = model(inputs).logits

            # Get predicted tokens
            preds1 = torch.argmax(logits1, dim=-1)
            preds2 = torch.argmax(logits2, dim=-1)

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

    tokenizer = BaseTokenizer(args.model_path_1, tokenizer_mode='slow')
    org_model = MODEL_REGISTRY[args.model_type_1](args.model_path_1, 'auto')
    model = MODEL_REGISTRY[args.model_type_2](args.model_path_2, 'auto')

    eval_cfg = {
        'name': 'wikitext2',
        'seq_len': 2048,
        'bs': 1,
        'download': False,
        'path': 'data_path',
        'inference_per_block': False,
    }
    token_consistency_eval = TokenConsistencyEval(tokenizer.get_tokenizer(), eval_cfg)

    consistency_ratio = token_consistency_eval.eval(org_model, model)
    logger.info(f'Token consistency ratio: {consistency_ratio}')
