import torch
from loguru import logger

from llmc.utils.registry_factory import MODEL_REGISTRY

from .eval_base import BaseEval


class TokenConsistencyEval(BaseEval):

    @torch.no_grad()
    def eval_func(self, model, testenc, seq_len, bs, eval_pos):
        handles_origin = []
        model_origin = MODEL_REGISTRY[self.config.model.type](self.config)
        if self.inference_per_block:
            handles_origin = self.register_hooks(model_origin)
        else:
            if model_origin.mm_model:
                model_origin.mm_model.cuda()
            else:
                model_origin.model.cuda()

        if model_origin.mm_model:
            model_origin.mm_model.eval()
        else:
            model_origin.model.eval()

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
            logits1 = model_origin.model(inputs).logits
            logits2 = model.model(inputs).logits
            model.reset_kv()

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

        if model_origin.mm_model:
            model_origin.mm_model.cpu()
        else:
            model_origin.model.cpu()

        if self.inference_per_block:
            for h in handles_origin:
                h.remove()

        return consistency_ratio
