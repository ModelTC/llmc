import functools
import gc
from collections import defaultdict

import torch
from loguru import logger

from llmc.utils import copy_files

from ..blockwise_optimization import BlockwiseOpt
from .sparse import Sparser


class BaseBlockwiseSparsification(BlockwiseOpt):
    def __init__(self, model, sparsity_config, input, config):
        super().__init__(model, sparsity_config, input, config)
        self.set_sparsity_config()

    def block_init(self, block):
        pass

    def set_sparsity_config(self):
        if 'sparsity_out' in self.sparsity_config and self.sparsity_config[
            'sparsity_out'
        ]:
            self.sparsity_out = True
        else:
            self.sparsity_out = False
        logger.info(f'use sparsity_out {self.sparsity_out}')

        self.sparser = Sparser(self.sparsity_config['weight'])

    def block_forward(self, block, input_data=None):
        output = []
        if input_data is None:
            input_data = self.input['data']

        for i in range(len(input_data)):
            input_data[i] = input_data[i].to(device=next(block.parameters()).device)
            if 'attention_mask' in self.input[
                'kwargs'
            ][i] and self.input['kwargs'][i]['attention_mask'] is not None:
                self.input['kwargs'][i]['attention_mask'] = self.input['kwargs'][i][
                    'attention_mask'
                ].cuda()
            with torch.no_grad():
                out = block(input_data[i], **self.input['kwargs'][i])[0]
                output.append(out)
        return output

    def block_opt(self, block):
        block = block.cuda()
        named_linears = self.model.get_block_linears(block)
        logger.info(f'named_linears: {named_linears}')
        input_feat = defaultdict(list)
        handles = []
        self.block_init(block)

        for name in named_linears:
            handles.append(
                named_linears[name].register_forward_hook(
                    functools.partial(
                        self.cache_input_hook, name=name, feat_dict=input_feat
                    )
                )
            )

        if not self.sparsity_out:
            self.input['data'] = self.block_forward(block)
        else:
            self.block_forward(block)
        for h in handles:
            h.remove()
        torch.cuda.empty_cache()

        self.block_transform(block, input_feat, self.input['kwargs'])

        if self.sparsity_out:
            self.input['data'] = self.block_forward(block)

        block = block.cpu()
        del input_feat
        gc.collect()
        torch.cuda.empty_cache()

    def block_transform(self, block, input_feat, block_kwargs):
        logger.info(f'Start transform the {self.block_idx+1}-th block')
        subsets = self.model.get_subsets_in_block(block)
        for index, subset in enumerate(subsets):
            if not self.filter_subset(subset):
                continue
            # logger.info(f"subset: {subset}")
            prev_op = subset['prev_op']
            layers_dict = subset['layers']
            input_name = subset['input'][0]
            inspect_module = subset['inspect']
            inspect_has_kwargs = subset['has_kwargs']
            subset_kwargs = block_kwargs if inspect_has_kwargs else {}
            self.subset_transform(
                layers_dict,
                input_feat,
                prev_op,
                input_name,
                inspect_module,
                subset_kwargs
            )
        logger.info(f'End transform the {self.block_idx+1}-th block')

    def filter_subset(self, subset):
        return True

    @torch.no_grad()
    def deploy(self, deploy_format):
        logger.info('-- deploy_sparsity_model start --')
        logger.info(f'sparsity_config : {self.sparsity_config}')

        logger.info('-- deploy_sparsity_model done --')

    @torch.no_grad()
    def copy_tokenizer(self, path):
        for substring in self.config.save.get('tokenizer_file_substring', ['token']):
            copy_files(self.config.model.path, path, substring)
        logger.info('copy tokenizer done --')

    @torch.no_grad()
    def save_model(self, path):
        if self.config.model.type == 'Llava':
            self.model.llava_model.language_model = self.model.get_model()
            self.model.llava_model.save_pretrained(path)
            logger.info('save model done --')
            self.copy_tokenizer(path)
            copy_files(self.config.model.path, path, 'preprocessor_config')
        else:
            self.model.get_model().save_pretrained(path)
            logger.info('save model done --')
            self.copy_tokenizer(path)
