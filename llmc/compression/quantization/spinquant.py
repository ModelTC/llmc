import gc
from functools import partial

import torch
import torch.nn as nn
from loguru import logger

from llmc.utils.registry_factory import ALGO_REGISTRY

from .base_blockwise_quantization import BaseBlockwiseQuantization
from .hadamard_utils import apply_exact_had_to_linear, random_hadamard_matrix
from .module_utils import *
from .module_utils import (_LLMC_LN_TYPES_, _TRANSFORMERS_LN_TYPES_,
                           EffcientFakeQuantLinear, FakeQuantLinear,
                           LlmcRMSNorm, OriginEmbedding, OriginFloatLinear,
                           RotateEmbedding, RotateLinear)
from .rotate_utils import ActRotater, RotateModule, WeightRotater


@ALGO_REGISTRY
class SpinQuant(BaseBlockwiseQuantization):
    def __init__(self, model, quant_config, input, config):
        super().__init__(model, quant_config, input, config)
        self.dev = torch.device('cuda')
        self.add_quant_config()
        self.preprocess()

    def add_quant_config(self):
        self.rotate_mode = self.quant_config['special']['rotate_mode']
        self.weight_rotate = True
        self.w_rotater = WeightRotater(weight_rotate_func=self.rotate_weight, dev=self.dev)
        # self.o_proj_group_quant = self.quant_config['special']['o_proj_group_quant']

    def preprocess(self):
        for m in self.model.model.parameters():
            m.requires_grad = False

        assert self.config['model']['type'] in ['Opt', 'Llama']
        # if self.config["model"]["type"] in ["Opt"]:
        self.remove_mean_from_embed()

        Q1 = self.get_orthogonal_matrix(self.hidden_size)
        self.model.model.Q1 = RotateModule(Q1)

        self.register_embed_spin_parameters()

        pre_head_ln = self.model.get_pre_head_layernorm_layers()[0]
        self.fuse_ln_fcs(pre_head_ln, self.model.get_head_layers())

        self.model.replace_module_subset(
            LlmcRMSNorm,
            self.model.model,
            {'layers': {'model.norm': pre_head_ln}},
            None,
            {},
        )
        self.register_lmhead_spin_parameters()

        gc.collect()
        torch.cuda.empty_cache()

    def get_trainable_params(self):
        trainable_parameters = []
        for n, m in self.model.model.named_parameters():
            if 'Q1' in n or 'Q2' in n:
                trainable_parameters.append(m)
        return trainable_parameters

    def a_rot(self, act, module, a_rotater):
        return a_rotater.rotate(act)

    def w_rot(self, module, w_rotater, args):
        return w_rotater.rotate(module.weight, module.bias, args['Q1'], args['Q2'], args['transpose'])

    def w_qdq_tmp(self, module, wquantizer):
        args = {'lowbound_factor': None, 'upbound_factor': None}
        if hasattr(module, 'buf_lowbound_factor'):
            args['lowbound_factor'] = module.buf_lowbound_factor
        if hasattr(module, 'buf_upbound_factor'):
            args['upbound_factor'] = module.buf_upbound_factor

        return wquantizer.fake_quant_weight_dynamic(module.tmp_weight, args)

    def register_embed_spin_parameters(self):
        embedding_layer = self.model.get_embed_layers()[0]
        args = {}
        args['Q1'] = self.model.model.Q1
        args['Q2'] = None
        args['transpose'] = False
        params_dict = self.get_replacement_params(mode='rotate', w_only=self.w_only, name=None, args=args)
        params_dict.pop('a_rot')
        self.model.replace_module_subset(
            RotateEmbedding,
            self.model.model,
            {'layers': {'model.embed_tokens': embedding_layer}},
            None,
            params_dict
        )
        self.model.find_embed_layers()

    def register_lmhead_spin_parameters(self):
        lm_head_layer = self.model.get_head_layers()[0]
        args = {}
        args['Q1'] = self.model.model.Q1
        args['Q2'] = None
        args['transpose'] = False
        params_dict = self.get_replacement_params(mode='rotate', w_only=self.w_only, name=None, args=args)
        self.model.replace_module_subset(
            RotateLinear,
            self.model.model,
            {'layers': {'lm_head': lm_head_layer}},
            None,
            params_dict
        )

    def apply_fc_rotate_weight(self):
        for idx, block in enumerate(self.blocks):
            logger.info(f'Start apply {idx}-th block rotate weights')
            for name, module in block.named_modules():
                if isinstance(module, (RotateLinear, FakeQuantLinear)):
                    weight, bias = module._rotate_weight()
                    module.weight, module.bias = weight, bias
            logger.info(f'End apply {idx}-th block rotate weights')

    def apply_embedding_rotate_weight(self):

        embedding_layer = self.model.get_embed_layers()[0]
        if isinstance(embedding_layer, RotateEmbedding):
            weight = embedding_layer._rotate_weight()
            embedding_layer.weight.data = weight
            self.model.replace_module_subset(
                OriginEmbedding,
                self.model.model,
                {'layers': {'model.embed_tokens': embedding_layer}},
                None,
                {}
            )

    def apply_lmhead_rotate_weight(self):
        lm_head_layer = self.model.get_head_layers()[0]
        if isinstance(lm_head_layer, RotateLinear):
            weight, bias = lm_head_layer._rotate_weight()
            lm_head_layer.weight, lm_head_layer.bias = weight, bias
            self.model.replace_module_subset(
                OriginFloatLinear,
                self.model.model,
                {'layers': {'lm_head': lm_head_layer}},
                None,
                {}
            )


    def get_orthogonal_matrix(self, size):
        if self.rotate_mode == 'random':
            return random_orthogonal_matrix(size, self.dev)
        elif self.rotate_mode == 'hadamard':
            return random_hadamard_matrix(size, self.dev)
        else:
            raise ValueError(f'Unsupported mode {self.mode}')

    def block_transform(self, block):
        logger.info(f'Start transform the {self.block_idx+1}-th block')

        subsets = self.model.get_subsets_in_block(block)
        for index, subset in enumerate(subsets):
            self.subset_transform(block, subset)

        self.model.replace_module_block(LlmcRMSNorm, block, self.block_idx, {})

        logger.info(f'block:{block}')
        logger.info(f'End transform the {self.block_idx+1}-th block')

    def subset_transform(self, block, subset):
        prev_op = subset['prev_op']
        layers_dict = subset['layers']
        assert (
            len(prev_op) == 1
        ), 'Only support single prev_op. If multi prev_ops, code need to be updated.'

        layers = list(layers_dict.values())

        if isinstance(prev_op[0], tuple(_LLMC_LN_TYPES_ + _TRANSFORMERS_LN_TYPES_)):
            self.fuse_ln_fcs(prev_op[0], layers)
            for n in layers_dict.keys():
                m = layers_dict[n]
                self.replace_rotate_fc(block, n, m, Q1=self.model.model.Q1, Q2=None, transpose=False)
            if 'is_mlp' not in subset or not subset['is_mlp']:
                Q2 = self.get_orthogonal_matrix(self.hidden_size // self.num_heads)
                subset['inspect'].Q2 = RotateModule(Q2)

        else:
            if self.config['model']['type'] in ['Opt']:
                self.bake_mean_into_linear(layers[0])

            n = list(layers_dict.keys())[0]
            m = layers[0]
            if 'is_mlp' in subset and subset['is_mlp']:
                if self.online_rotate:
                    apply_exact_had_to_linear(m, had_dim=-1, output=False)
                self.replace_rotate_fc(block, n, m, Q1=self.model.model.Q1, Q2=None, transpose=True)
            else:
                self.replace_rotate_fc(block, n, m, Q1=self.model.model.Q1, Q2=block.self_attn.Q2, transpose=True)
                self.replace_rotate_fc(block, 'self_attn.v_proj', prev_op[0], Q1=self.model.model.Q1, Q2=block.self_attn.Q2, transpose=False)

    def apply_rotate_weight(self):
        self.apply_embedding_rotate_weight()
        self.apply_lmhead_rotate_weight()
        self.apply_fc_rotate_weight()

    def deploy(self, quant_format):
        if quant_format == 'train_rotate_quant':
            logger.info(f'-- deploy_{quant_format}_model start --')
            logger.info(f'quant_config : {self.quant_config}')
            logger.info(self.model.model)

            params_dict = {}
            params_dict['w_qdq'] = partial(self.w_qdq_tmp, wquantizer=self.wquantizer)
            params_dict['a_qdq'] = (
                partial(self.a_qdq, aquantizer=self.aquantizer)
                if not self.w_only
                else None
            )
            self.model.replace_module_all(
                FakeQuantLinear, params_dict
            )

            logger.info(f'-- deploy_{quant_format}_model done --')
            logger.info(f'-- strat train rotation--')
        else:
            self.apply_rotate_weight()
            super().deploy(quant_format)
