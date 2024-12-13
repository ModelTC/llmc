import glob
import os

import torch
from human_eval.data import stream_jsonl, write_jsonl
from human_eval.evaluation import evaluate_functional_correctness
from loguru import logger
from tqdm import tqdm

from .eval_base import BaseEval


class HumanEval(BaseEval):
    def __init__(self, model, config):
        super().__init__(model, config)
        self.res_path = self.eval_cfg.get('res_path', None)
        assert self.res_path is not None
        os.makedirs(self.res_path, exist_ok=True)
        self.format_tabs = self.eval_cfg.get('format_tabs', False)
        self.instruction = self.eval_cfg.get('instruction',
                                             'Complete the following Python code:')
        self.add_chat_temp = self.eval_cfg.get('add_chat_temp', False)

    @torch.no_grad()
    def eval_func(self, model, testenc, seq_len, bs, eval_pos):
        samples = []
        pbar = tqdm(total=len(testenc) * bs, dynamic_ncols=True, position=0, desc='Evaluating')

        for task_id in testenc:
            if self.format_tabs:
                prompt = testenc[task_id]['prompt'].replace('    ', '\t')
            else:
                prompt = testenc[task_id]['prompt']
            prompt = self.gen_prompt(prompt)
            batch_completions = self.generate_batch_completion(
                model, prompt, bs
            )

            for sample in batch_completions:
                result = dict(
                    task_id=task_id,
                    completion=sample,
                )
                samples += [result]

            pbar.update(bs)

        pbar.close()

        self.output_dir = os.path.join(self.res_path, eval_pos)

        os.makedirs(self.output_dir, exist_ok=True)
        out_path = os.path.join(self.output_dir, 'eval.jsonl')
        write_jsonl(out_path, samples)

        res = self.post_process(testenc)
        return res

    def gen_prompt(self, prompt):
        prompt = self.instruction + '\n' + prompt
        if self.model_type in ['Starcoder']:
            prompt = '<fim_prefix>' + prompt + '<fim_suffix><fim_middle>'

        if self.add_chat_temp:
            chat_prompt = [{'role': 'user', 'content': prompt}]
            chat_prompt = self.tokenizer.apply_chat_template(
                chat_prompt,
                tokenize=False,
            )
            return chat_prompt

        return prompt

    @torch.no_grad()
    def generated(
        self,
        model,
        inputs,
        max_new_tokens=512,
        temperature=0.2,
        top_p=0.95,
        do_sample=True,
    ):

        if hasattr(self.tokenizer, 'pad_token_id'):
            pad_token_id = self.tokenizer.pad_token_id
        else:
            pad_token_id = self.tokenizer.eos_token_id

        generated_ids = model.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=pad_token_id,
            use_cache=True,
        )
        return generated_ids

    @torch.no_grad()
    def generate_batch_completion(self, model, prompt, bs):
        input_batch = [prompt for _ in range(bs)]
        inputs = self.tokenizer(input_batch, return_tensors='pt').to(model.model.device)
        input_ids_cutoff = inputs.input_ids.size(dim=1)

        generated_ids = self.generated(model, inputs)
        model.reset_kv()

        batch_completions = self.tokenizer.batch_decode(
            [ids[input_ids_cutoff:] for ids in generated_ids],
            skip_special_tokens=True,
        )

        return [
            self.filter_code(self.fix_indents(completion))
            for completion in batch_completions
        ]

    @torch.no_grad()
    def post_process(self, testenc):
        files = sorted(glob.glob(os.path.join(self.output_dir, 'eval.jsonl')))
        logger.info(f'{len(files)} files in {self.output_dir}')
        output = []

        for code_file in tqdm(files, total=len(files)):
            codes = [c for c in stream_jsonl(code_file)]
            output += codes

        out_path = os.path.join(self.output_dir, 'processed.jsonl')
        logger.info(f'save to {out_path}')
        write_jsonl(out_path, output)
        res = self.entry_point(out_path)
        return res

    @torch.no_grad()
    def filter_code(self, completion):
        completion = completion.lstrip('\n')
        return completion.split('\n\n')[0]

    @torch.no_grad()
    def fix_indents(self, text):
        return text.replace('\t', '    ')

    @torch.no_grad()
    def entry_point(self, sample_file):
        results = evaluate_functional_correctness(sample_file)
        return results
