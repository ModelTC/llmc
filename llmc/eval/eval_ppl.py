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
            lm_logits = model.model(inputs).logits
            model.reset_kv()

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
