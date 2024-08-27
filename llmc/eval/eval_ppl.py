import gc
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
from datasets import load_dataset, load_from_disk
from loguru import logger

from .eval_base import BaseEval


class PerplexityEval(BaseEval):

    @torch.no_grad()
    def eval_func(self, org_model, model, testenc, seq_len, bs):
        testenc = testenc.input_ids
        nsamples = testenc.numel() // seq_len

        nlls = []

        # Loop through each batch
        for i in range(0, nsamples, bs):
            logger.info(f'index : {(i + 1) // bs}/{nsamples // bs}')
            # Calculate end index
            j = min(i + bs, nsamples)

            # Prepare inputs and move to gpu
            inputs = testenc[:, (i * seq_len): (j * seq_len)].cuda()
            inputs = inputs.reshape(j - i, seq_len)

            # Forward pass through the model
            lm_logits = model(inputs).logits

            # Shift logits and labels for next token prediction
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = inputs[:, 1:]

            # Compute loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
            )

            # Calculate negative log likelihood
            neg_log_likelihood = loss.float() * seq_len * (j - i)

            # Append to list of negative log likelihoods
            nlls.append(neg_log_likelihood)

        # Compute perplexity
        ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seq_len))

        # Empty CUDA cache to save memory
        testenc.cpu()
        torch.cuda.empty_cache()

        return ppl.item()


if __name__ == '__main__':
    import sys

    sys.path.append('../../')
    import argparse

    from llmc.data import BaseTokenizer
    from llmc.models import Llama
    from llmc.utils.registry_factory import MODEL_REGISTRY

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    args = parser.parse_args()

    tokenizer = BaseTokenizer(args.model_path, tokenizer_mode='fast')
    model = MODEL_REGISTRY[args.model_type](args.model_path, 'auto')

    # Llama2-70B config example
    eval_cfg = {
        'name': 'wikitext2',
        'seq_len': 2048,
        'bs': 20,
        'download': False,
        'path': 'data_path',
        'inference_per_block': True,
    }
    ppl_eval = PerplexityEval(tokenizer.get_tokenizer(), eval_cfg)

    ppl = ppl_eval.eval(model)
    logger.info(f'ppl : {ppl}')
