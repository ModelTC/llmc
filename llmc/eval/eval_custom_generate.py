import glob
import os

import torch
from human_eval.data import stream_jsonl, write_jsonl
from human_eval.evaluation import evaluate_functional_correctness
from loguru import logger
from tqdm import tqdm

from .eval_base import BaseEval


class CustomGenerate(BaseEval):
    def __init__(self, model, config):
        super().__init__(model, config)
        self.max_new_tokens = self.eval_cfg.get('max_new_tokens', 32)

    @torch.no_grad()
    def eval_func(self, model, testenc, seq_len, bs, eval_pos):
        responses = []
        for data in testenc:
            data = {
                k: (v.cuda() if torch.is_tensor(v) else v)
                for k, v in data.items()
            }
            if model.mm_model:
                generated_ids = model.mm_model.generate(
                    **data,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False
                )
            else:
                generated_ids = model.model.generate(
                    **data,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False
                )
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
            responses.append(response)
        responses = self.flatten_2d_to_1d(responses)
        assert len(responses) == len(self.testdata)
        logger.info('CustomGenerate Results:')
        for index in range(len(responses)):
            print('*' * 10)
            print(f'test data: {self.testdata[index]}')
            print(f'model response: {responses[index]}')
            print()

        for data in testenc:
            data = {
                k: (v.cpu() if torch.is_tensor(v) else v)
                for k, v in data.items()
            }
        torch.cuda.empty_cache()

        return 'custom gen done.'

    def flatten_2d_to_1d(self, two_d_list):
        return [item for sublist in two_d_list for item in sublist]
