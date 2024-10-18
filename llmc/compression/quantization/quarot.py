import gc

import torch
import torch.nn as nn
from loguru import logger

from llmc.utils.registry_factory import ALGO_REGISTRY

from .base_blockwise_quantization import BaseBlockwiseQuantization
from .hadamard_utils import apply_exact_had_to_linear, random_hadamard_matrix
from .module_utils import (_LLMC_LN_TYPES_, _TRANSFORMERS_LN_TYPES_,
                           LlmcRMSNorm, RotateLinear)


@ALGO_REGISTRY
class Quarot(BaseBlockwiseQuantization):
    def __init__(self, model, quant_config, input, config):
        super().__init__(model, quant_config, input, config)
        self.dev = torch.device('cuda')
        self.add_quant_config()
        self.preprocess()

    def preprocess(self):
        assert self.config['model']['type'] in ['Opt', 'Llama']
        # if self.config["model"]["type"] in ["Opt"]:
        self.remove_mean_from_embed()

        self.Q = self.get_orthogonal_matrix()
        self.rotate_embeddings(self.Q)

        pre_head_ln = self.model.get_pre_head_layernorm_layers()[0]
        self.fuse_ln_fcs(pre_head_ln, self.model.get_head_layers())

        self.model.replace_module_subset(
            LlmcRMSNorm,
            self.model.model,
            {'layers': {'model.norm': pre_head_ln}},
            None,
            {},
        )

        self.rotate_head(self.Q)

        gc.collect()
        torch.cuda.empty_cache()

    def a_rot(self, act, module, a_rotater):
        return a_rotater.rotate(act)

    @torch.no_grad()
    def add_quant_config(self):
        self.rotate_mode = self.quant_config['special']['rotate_mode']

    def get_orthogonal_matrix(self):
        if self.rotate_mode == 'random':
            return random_orthogonal_matrix(self.hidden_size, self.dev)
        elif self.rotate_mode == 'hadamard':
            return random_hadamard_matrix(self.hidden_size, self.dev)
        else:
            raise ValueError(f'Unsupported mode {self.mode}')

    def block_transform(self, block, ):
        logger.info(f'Start transform the {self.block_idx+1}-th block')

        if self.online_rotate:
            self.replace_rotate_fcs(block)
        subsets = self.model.get_subsets_in_block(block)
        for index, subset in enumerate(subsets):
            self.subset_transform(block, subset)

        self.model.replace_module_block(LlmcRMSNorm, block, self.block_idx, {})

        logger.info(f'block:{block}')
        logger.info(f'End transform the {self.block_idx+1}-th block')

    @torch.no_grad()
    def subset_transform(self, block, subset):
        prev_op = subset['prev_op']
        layers_dict = subset['layers']
        assert (
            len(prev_op) == 1
        ), 'Only support single prev_op. If multi prev_ops, code need to be updated.'

        layers = list(layers_dict.values())

        if isinstance(prev_op[0], tuple(_LLMC_LN_TYPES_ + _TRANSFORMERS_LN_TYPES_)):
            self.fuse_ln_fcs(prev_op[0], layers)
            self.rotate_pre_layers(layers, self.Q)
        else:
            if self.config['model']['type'] in ['Opt']:
                self.bake_mean_into_linear(layers[0])

            if 'is_mlp' in subset and subset['is_mlp']:
                self.rotate_post_layers(
                    layers, self.Q, exact_had=True if self.online_rotate else False
                )
            else:
                self.rotate_post_layers(layers, self.Q, exact_had=False)
                if self.online_rotate:
                    R2 = None
                    apply_exact_had_to_linear(
                        prev_op[0], had_dim=self.head_dim, output=True, R2=R2
                    )
                    apply_exact_had_to_linear(layers[0], had_dim=-1, output=False, R2=R2)
