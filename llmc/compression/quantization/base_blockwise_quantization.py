import functools
import gc
import json
import os
from collections import defaultdict
from functools import partial

import torch
import torch.distributed as dist
import torch.nn as nn
from loguru import logger

from llmc.utils import copy_files

from ..blockwise_optimization import BlockwiseOpt
from .hadamard_utils import apply_exact_had_to_linear, get_hadK
from .module_utils import (_LLMC_LINEAR_TYPES_, _LLMC_LN_TYPES_,
                           _REALQUANT_LINEAR_MAP_, _TRANSFORMERS_LINEAR_TYPES_,
                           _TRANSFORMERS_LN_TYPES_, EffcientFakeQuantLinear,
                           FakeQuantLinear, LlmcActFn, LlmcViTSelfAttention,
                           OriginFloatLinear, RotateLinear)
from .quant import FloatQuantizer, IntegerQuantizer
from .utils import check_do_quant, check_w_only, get_aquantizer, get_wquantizer


class BaseBlockwiseQuantization(BlockwiseOpt):
    def __init__(self, model, quant_config, input, padding_mask, config):
        super().__init__(model, quant_config, input, padding_mask, config)
        self.set_quant_config()

    def w_qdq(self, module, wquantizer):
        args = {'lowbound_factor': None, 'upbound_factor': None}
        if hasattr(module, 'buf_lowbound_factor'):
            args['lowbound_factor'] = module.buf_lowbound_factor
        if hasattr(module, 'buf_upbound_factor'):
            args['upbound_factor'] = module.buf_upbound_factor

        return wquantizer.fake_quant_weight_dynamic(module.weight, args)

    def w_q(self, module, wquantizer):
        return wquantizer.real_quant_weight_dynamic(module.weight.data)

    def a_qdq(self, act, module, aquantizer, input_index=0):
        if self.act_static:
            args = {
                'scales': (
                    getattr(module, f'buf_act_scales_{input_index}', None)
                ),
                'zeros': (
                    getattr(module, f'buf_act_zeros_{input_index}', None)
                ),
                'qmax': (
                    getattr(module, f'buf_act_qmax_{input_index}', None)
                ),
                'qmin': (
                    getattr(module, f'buf_act_qmin_{input_index}', None)
                )
            }
            return aquantizer.fake_quant_act_static(act, args)
        else:
            return aquantizer.fake_quant_act_dynamic(act)

    def logit(self, x):
        return torch.log(x / (1 - x))

    def get_replacement_params(self, mode='fake_quant', w_only=False, name=None):
        params_dict = {}
        if mode == 'fake_quant':
            if not self.mix_bits:
                params_dict['a_qdq'] = (
                    partial(self.a_qdq, aquantizer=self.aquantizer)
                    if not w_only
                    else None
                )
                params_dict['w_qdq'] = partial(self.w_qdq, wquantizer=self.wquantizer)
            else:
                params_dict['mix_bits'] = True
                params_dict['a_qdq'] = self.a_qdq
                params_dict['w_qdq'] = self.w_qdq
                params_dict['mix_bits_map'] = self.mix_bits_map
                params_dict['quantizer_mix_bits'] = self.quantizer_mix_bits
                params_dict['wquantizer_default'] = self.wquantizer
                params_dict['aquantizer_default'] = self.aquantizer
                params_dict['w_only_default'] = w_only

        elif mode in _REALQUANT_LINEAR_MAP_.keys():
            params_dict['w_q'] = partial(self.w_q, wquantizer=self.wquantizer)
            params_dict['quant_config'] = self.quant_config

        elif mode == 'online_rotate':
            had_K, K = get_hadK(
                self.intermediate_size if 'down_proj' in name else self.num_heads
            )
            params_dict = {
                'had_K': had_K,
                'K': K,
                'online_full_had': 'down_proj' in name,
                'online_partial_had': 'o_proj' in name,
                'had_dim': (
                    None if 'down_proj' in name else self.hidden_size // self.num_heads
                ),
                'fp32_had': self.fp32_had,
            }

        elif mode == 'quant_attn':
            params_dict = {
                'matmul_a1_qdq': partial(self.a_qdq, aquantizer=self.aquantizer, input_index=0),
                'matmul_a2_qdq': partial(self.a_qdq, aquantizer=self.aquantizer, input_index=1),
                'softmax_a_qdq': partial(self.a_qdq, aquantizer=self.aquantizer)
                if self.quant_softmax else None
            }
        elif mode == 'quant_act_fn':
            params_dict = {
                'a_qdq': partial(self.a_qdq, aquantizer=self.aquantizer)
            }

        return params_dict

    def alloc_bits(self, mix_bits_settings):
        for i in range(len(mix_bits_settings)):
            mix_bits_setting = mix_bits_settings[f'setting_{i}']
            if mix_bits_setting['do_quant']:
                wquantizer_mix_bits = self.quant_module(**mix_bits_setting['weight'])
                if 'act' in mix_bits_setting:
                    w_only_mix_bits = False
                    aquantizer_mix_bits = self.quant_module(**mix_bits_setting['act'])
                else:
                    w_only_mix_bits = True
                self.quantizer_mix_bits.append(
                    {
                        'layer_name': mix_bits_setting['layer_name'],
                        'do_quant': mix_bits_setting['do_quant'],
                        'w_only_mix_bits': w_only_mix_bits,
                        'wquantizer': wquantizer_mix_bits,
                        'aquantizer': (
                            aquantizer_mix_bits if not w_only_mix_bits else None
                        ),
                    }
                )
            else:
                self.quantizer_mix_bits.append(
                    {
                        'layer_name': mix_bits_setting['layer_name'],
                        'do_quant': mix_bits_setting['do_quant'],
                    }
                )

        for i in range(len(self.quantizer_mix_bits)):
            logger.info(f'quantizer_mix_bits {i} : {self.quantizer_mix_bits[i]}')
            layer_name = self.quantizer_mix_bits[i]['layer_name']
            for name in layer_name:
                n_layeridx = name.split('#')
                assert (
                    len(n_layeridx) == 1 or len(n_layeridx) == 2
                ), 'layer_name in mix_bits must be name#1-3-4 or name.'
                if len(n_layeridx) == 2:
                    n = n_layeridx[0]
                    layeridx = n_layeridx[1].split('-')
                    layeridx = [int(idx) for idx in layeridx]
                else:
                    n = n_layeridx[0]
                    layeridx = 'all'
                if layeridx == 'all':
                    for k in range(self.num_blocks):
                        self.mix_bits_map[k][n] = i
                else:
                    for k in layeridx:
                        self.mix_bits_map[k][n] = i

    def set_quant_config(self):
        self.mix_bits = 'mix_bits' in self.quant_config
        self.mix_bits_map = [{} for _ in range(self.num_blocks)]
        self.quantizer_mix_bits = []

        self.quant_out = self.quant_config.get('quant_out', False)
        self.tp = self.quant_config.get('tp', 1)
        self.quant_config['weight']['tp'] = self.tp

        # select quant module
        self.quant_type = self.quant_config.get('quant_type', 'int_quant')
        if self.quant_type == 'int_quant':
            self.quant_module = IntegerQuantizer
        else:
            self.quant_module = FloatQuantizer
        logger.info(f'The used Quant Module is {self.quant_module}')

        # set weight quant config
        self.wquantizer = self.quant_module(**self.quant_config['weight'])

        # set act quant config
        if 'act' in self.quant_config:
            self.w_only = False
            self.quant_config['act']['tp'] = self.tp
            self.aquantizer = self.quant_module(**self.quant_config['act'])
            self.act_static = self.quant_config['act'].get('static', False)
            if self.act_static:
                assert self.quant_config['act']['granularity'] == 'per_tensor', \
                    'Only support per_tensor static quant'
            self.quant_attn = self.quant_config['act'].get('quant_attn', False)
            if self.quant_attn:
                assert self.config['model']['type'] in ['Vit']
                self.quant_softmax = self.quant_config['act'].get('quant_softmax', False)
            self.quant_act_fn = self.quant_config['act'].get('quant_act_fn', False)
        else:
            self.w_only = True
            self.aquantizer = None
            self.act_static = False
            self.quant_attn = False
            self.quant_softmax = False
            self.quant_act_fn = False

        # set mix-bits quant config
        if self.mix_bits:
            mix_bits_settings = self.quant_config['mix_bits']
            logger.info(f'mix_bits_settings number: {len(mix_bits_settings)}')
            logger.info(
                f'mix_bits_settings:\n'
                f'{json.dumps(mix_bits_settings, ensure_ascii=False, indent=4)}'
            )
            self.alloc_bits(mix_bits_settings)

            logger.info(
                f'self.mix_bits_map:\n'
                f'{json.dumps(self.mix_bits_map, ensure_ascii=False, indent=4)}'
            )

        # set special quant config
        special_config = self.quant_config.get('special', {})
        self.true_sequential = special_config.get('true_sequential', True)
        self.weight_clip = special_config.get('weight_clip', True)
        self.save_scale = special_config.get('save_scale', False)
        self.save_clip = special_config.get('save_clip', False)
        self.clip_version = special_config.get('clip_version', 'v1')
        self.clip_sym = special_config.get('clip_sym', self.wquantizer.sym)
        self.clip_all = special_config.get('clip_all', False)

        if self.save_scale:
            self.scale_path = special_config['scale_path']
            self.act_scales = {}

        if self.save_clip:
            self.clip_path = special_config['clip_path']
            self.weight_clips = {}

        if self.clip_version == 'v2':
            assert self.wquantizer.calib_algo == 'learnable'

        # set online-rotation config
        self.online_rotate = special_config.get('online_rotate', False)
        if self.online_rotate:
            assert self.config['model']['type'] in ['Opt', 'Llama']

        self.hidden_size = self.model.model_config.hidden_size
        if self.online_rotate:
            self.num_heads = self.model.model_config.num_attention_heads
            self.head_dim = self.hidden_size // self.num_heads
            self.intermediate_size = self.model.model_config.intermediate_size
            self.fp32_had = special_config.get('fp32_had', False)

    def replace_rotate_linears(self, block):
        for n, m in block.named_modules():
            if isinstance(m, nn.Linear) and ('down_proj' in n
                                             or 'o_proj' in n
                                             or 'fc2' in n
                                             or 'out_proj' in n):
                subset = {'layers': {n: m}}
                self.model.replace_module_subset(
                    RotateLinear,
                    block,
                    subset,
                    None,
                    self.get_replacement_params(
                        mode='online_rotate', w_only=self.w_only, name=n
                    ),
                )

    def replace_act_fn(self, block, extra_modules):
        act_fn_dict = self.model.get_act_fn_in_block(block)
        layers_dict = {'layers': act_fn_dict}
        self.model.replace_module_subset(
            LlmcActFn,
            block,
            layers_dict,
            self.block_idx,
            self.get_replacement_params(
                mode='quant_act_fn', w_only=self.w_only, name=None
            ),
        )
        extra_modules.update(act_fn_dict)

    def replace_attention(self, block, extra_modules):
        attn_layers_dict = self.model.get_attn_in_block(block)
        layers_dict = {'layers': attn_layers_dict}
        self.model.replace_module_subset(
            LlmcViTSelfAttention,
            block,
            layers_dict,
            self.block_idx,
            self.get_replacement_params(
                mode='quant_attn', w_only=self.w_only, name=None
            ),
        )

        matmul_modules = self.model.get_matmul_in_block(block)
        softmax_modules = self.model.get_softmax_in_block(block) if self.quant_softmax else {}
        extra_modules.update(matmul_modules)
        extra_modules.update(softmax_modules)

    @torch.no_grad()
    def collect_block_qparams(self, block):
        named_linears = self.model.get_block_linears(block)
        for n, m in named_linears.items():
            args = {}
            if hasattr(m, 'buf_lowbound_factor'):
                args['lowbound_factor'] = m.buf_lowbound_factor
            if hasattr(m, 'buf_upbound_factor'):
                args['upbound_factor'] = m.buf_upbound_factor
            (
                tensor,
                scales,
                zeros,
                max_int,
                min_int,
            ) = self.wquantizer.get_tensor_qparams(m.weight.data, args=args)
            m.register_buffer('buf_scales', scales)
            m.register_buffer('buf_zeros', zeros)
            m.register_buffer('buf_qmax', torch.tensor(max_int).to(self.dev))
            m.register_buffer('buf_qmin', torch.tensor(min_int).to(self.dev))

    def block_forward(self, block, input_data=None):
        output = []

        if input_data is None:
            input_data = self.input['data']

        for i in range(len(input_data)):
            input_data[i] = input_data[i].to(device=next(block.parameters()).device)
            keys_to_device = ['attention_mask', 'cross_attention_mask', 'cross_attention_states']
            for key in keys_to_device:
                if (
                    key in self.input['kwargs'][i]
                    and self.input['kwargs'][i][key] is not None
                ):
                    self.input['kwargs'][i][key] = \
                        self.input['kwargs'][i][key].to(device=next(block.parameters()).device)
            with torch.no_grad():
                out = block(input_data[i], **self.input['kwargs'][i])[0]
                output.append(out)
        return output

    def block_opt(self, block):
        block = block.cuda()
        named_linears = self.model.get_block_linears(block)
        extra_modules = self.model.get_extra_modules(block)
        self.extra_module_name = list(extra_modules.keys())[0]

        if self.quant_attn:
            self.replace_attention(block, extra_modules)
        if self.quant_act_fn:
            self.replace_act_fn(block, extra_modules)

        input_feat_modules = {
            k: v for d in [named_linears, extra_modules] for k, v in d.items()
        }
        logger.info(f': {input_feat_modules}')
        input_feat = defaultdict(list)

        handles = self.register_hooks(input_feat_modules, input_feat)

        self.block_init(block)

        self.run(block, input_feat, handles)

        block = block.cpu()
        del input_feat
        gc.collect()
        torch.cuda.empty_cache()

    def register_hooks(self, input_feat_modules, input_feat):
        handles = []
        if not self.data_free:
            for name in input_feat_modules:
                handles.append(
                    input_feat_modules[name].register_forward_hook(
                        functools.partial(
                            self.cache_input_hook, name=name, feat_dict=input_feat
                        )
                    )
                )
        return handles

    def run(self, block, input_feat, handles):
        if not self.data_free:
            if self.quant_out:
                self.block_forward(block)
            else:
                self.input['data'] = self.block_forward(block)

            for h in handles:
                h.remove()
            torch.cuda.empty_cache()

            self.block_transform(block, input_feat, self.input['kwargs'])
        else:
            self.block_transform(block)

        if not self.data_free and self.quant_out:
            self.model.replace_module_block(
                FakeQuantLinear,
                block,
                self.block_idx,
                self.get_replacement_params(
                    mode='fake_quant', w_only=self.w_only, name=None
                ),
            )
            self.set_non_linear_mode('fake_quant', block, False)
            self.input['data'] = self.block_forward(block)

    def block_transform(self, block, input_feat, block_kwargs):
        logger.info(f'Start transform the {self.block_idx}-th block')
        subsets = self.model.get_subsets_in_block(block)

        if self.act_static:
            self.register_non_linear_qparams(block, input_feat)
            self.register_except_subsets_qparams(block, input_feat)

        self.set_non_linear_mode('fake_quant', block, False)

        for index, subset in enumerate(subsets):
            logger.info(f'subset: {subset}')
            prev_op = subset['prev_op']
            layers_dict = subset['layers']
            input_name = subset['input'][0]
            inspect_module = subset['inspect']
            inspect_has_kwargs = subset['has_kwargs']
            if inspect_has_kwargs:
                if 'sub_keys' in subset:
                    subset_kwargs = [{k: block_kwargs[0][v] for k, v in subset['sub_keys'].items()}]
                else:
                    subset_kwargs = block_kwargs
            else:
                subset_kwargs = {}
            self.subset_transform(
                layers_dict,
                input_feat,
                prev_op,
                input_name,
                inspect_module,
                subset_kwargs,
            )
            if self.act_static:
                self.register_act_qparams(layers_dict, input_feat[input_name])

            if self.true_sequential and index != len(subsets) - 1:
                next_subset = subsets[index + 1]
                input_feat_subset = self.rehook_next_subset(block,
                                                            subset,
                                                            next_subset)
                input_feat.update(input_feat_subset)

        self.set_non_linear_mode('fake_quant', block, True)
        logger.info(f'End transform the {self.block_idx}-th block')

    def rehook_next_subset(self, block, subset, next_subset):
        self.subset_init(next_subset)

        layers_except_subsets = self.model.get_linears_except_subsets(block)
        if (
            layers_except_subsets
            and not isinstance(
                layers_except_subsets[list(layers_except_subsets.keys())[0]],
                FakeQuantLinear
            )
        ):
            self.model.replace_module_subset(
                FakeQuantLinear,
                block,
                {'layers': layers_except_subsets},
                self.block_idx,
                self.get_replacement_params(
                    mode='fake_quant', w_only=self.w_only, name=None
                ),
            )

        self.model.replace_module_subset(
            FakeQuantLinear,
            block,
            subset,
            self.block_idx,
            self.get_replacement_params(
                mode='fake_quant', w_only=self.w_only, name=None
            ),
        )

        input_feat_subset = defaultdict(list)
        input_feat_modules = next_subset['layers']
        handles = self.register_hooks(input_feat_modules, input_feat_subset)

        self.block_forward(block)
        for h in handles:
            h.remove()

        return input_feat_subset

    def layer_init(self, layer):
        pass

    def subset_init(self, subset):
        pass

    def block_init(self, block):
        pass

    def collect_layers_weights(self, layers, tensor_parallelize_style=None):
        weights = []
        for _m in layers:
            weights.append(_m.weight)
        return weights

    def moving_avg_range(self, act_tensors):
        avg_min_vals, avg_max_vals = [], []
        if isinstance(act_tensors[0], tuple):
            def transform_multiple_inputs(pairs):
                inputs_lists = zip(*pairs)
                transformed = [list(zip(inputs, inputs)) for inputs in inputs_lists]
                return [item for sublist in transformed for item in sublist]
            act_tensors = transform_multiple_inputs(act_tensors)
        else:
            if len(act_tensors) == 1:
                tensor_list = [act_tensors[0][i] for i in range(act_tensors[0].size(0))]
                act_tensors[0] = tensor_list
            else:
                act_tensors = [act_tensors]

        for tensors in act_tensors:
            avg_min_val, avg_max_val = None, None
            for tensor in tensors:
                tensor = self.aquantizer.reshape_tensor(tensor)
                tensor_range = self.aquantizer.get_tensor_range(tensor)
                min_val, max_val = tensor_range[0], tensor_range[1]
                if min_val is None:
                    avg_min_val = None
                else:
                    if avg_min_val is None:
                        avg_min_val = min_val / len(tensors)
                    else:
                        avg_min_val += min_val / len(tensors)
                if max_val is None:
                    avg_max_val = None
                else:
                    if avg_max_val is None:
                        avg_max_val = max_val / len(tensors)
                    else:
                        avg_max_val += max_val / len(tensors)
            avg_min_vals.append(avg_min_val)
            avg_max_vals.append(avg_max_val)

        return avg_min_vals, avg_max_vals

    @torch.no_grad()
    def register_except_subsets_qparams(self, block, input_feat):
        layers_dict = self.model.get_linears_except_subsets(block)
        for name, layer in layers_dict.items():
            self.register_act_qparams({name: layer}, input_feat[name])

    @torch.no_grad()
    def register_non_linear_qparams(self, block, input_feat):
        layer_types = [
            ('quant_attn', self.model.get_matmul_in_block),
            ('quant_softmax', self.model.get_softmax_in_block, 'quant_attn'),
            ('quant_act_fn', self.model.get_act_fn_in_block)
        ]

        for mode, layer_func, *dependency in layer_types:
            if getattr(self, mode, True) and all(getattr(self, dep, True) for dep in dependency):
                layers_dict = layer_func(block)
                for name, layer in layers_dict.items():
                    self.register_act_qparams({name: layer}, input_feat[name])

    @torch.no_grad()
    def register_act_qparams(self, layers_dict, act_tensors):
        avg_min_vals, avg_max_vals = self.moving_avg_range(act_tensors)
        for i in range(len(avg_min_vals)):
            avg_min_val, avg_max_val = avg_min_vals[i], avg_max_vals[i]
            scales, zeros, qmax, qmin = self.aquantizer.get_qparams(
                (avg_min_val, avg_max_val), avg_min_val.device
            )
            logger.info(f'avg_min_val:{avg_min_val}')
            logger.info(f'avg_max_val:{avg_max_val}')
            for name in layers_dict:
                layers_dict[name].register_buffer(f'buf_act_scales_{i}', scales)
                layers_dict[name].register_buffer(f'buf_act_zeros_{i}', zeros)
                layers_dict[name].register_buffer(f'buf_act_qmin_{i}', qmin)
                layers_dict[name].register_buffer(f'buf_act_qmax_{i}', qmax)
                logger.info(f'name: {name}')
                logger.info(f'buf_act_scales_{i} : {scales}')
                logger.info(f'buf_act_zeros_{i} : {zeros}')

    @torch.no_grad()
    def apply_scale(self, scales, prev_op, layers):
        assert (
            len(prev_op) == 1
        ), 'Only support single prev_op. If multi prev_ops, code need to be updated.'
        if isinstance(
            prev_op[0], tuple(_LLMC_LINEAR_TYPES_ + _TRANSFORMERS_LINEAR_TYPES_)
        ):
            assert len(layers) == 1
            logger.info('apply scale between fc and fc')
            self.scale_fc_fc(prev_op[0], layers[0], scales)
        elif isinstance(prev_op[0], tuple(_LLMC_LN_TYPES_ + _TRANSFORMERS_LN_TYPES_)):
            logger.info('apply scale between ln and fc')
            self.scale_ln_fcs(prev_op[0], layers, scales)
        else:
            raise NotImplementedError(f'prev_op {type(prev_op[0])} not supported yet!')

    @torch.no_grad()
    def apply_shift(self, shifts, prev_op, layers):
        if shifts is None:
            return

        assert (
            len(prev_op) == 1
        ), 'Only support single prev_op. If multi prev_ops, code need to be updated.'
        if isinstance(
            prev_op[0], tuple(_LLMC_LINEAR_TYPES_ + _TRANSFORMERS_LINEAR_TYPES_)
        ):
            assert len(layers) == 1
            self.shift_fc_fc(prev_op[0], layers[0], shifts)
        elif isinstance(prev_op[0], tuple(_LLMC_LN_TYPES_ + _TRANSFORMERS_LN_TYPES_)):
            self.shift_ln_fcs(prev_op[0], layers, shifts)
        else:
            raise NotImplementedError(f'prev_op {type(prev_op[0])} not supported yet!')

    @torch.no_grad()
    def scale_fc_fc(self, fc1, fc2, scales):
        scales = scales.to(fc1.weight.device)
        if fc1.out_features == fc2.in_features * 3:
            num_heads = self.model.get_num_attention_heads()
            fc1.weight.t_()
            org_shape = fc1.weight.shape
            fc1.weight.data = fc1.weight.data.reshape(org_shape[0] * num_heads, 3, -1)
            value = fc1.weight.data[:, 2, :].reshape(org_shape[0], -1)
            fc1.weight.data[:, 2, :] = value.div(scales.view(-1)).reshape(
                fc1.weight[:, 2, :].shape
            )
            fc1.weight.data = fc1.weight.data.reshape(org_shape).t_()
            if hasattr(fc1, 'bias') and fc1.bias is not None:
                fc1.bias.data = fc1.bias.data.reshape(num_heads, 3, -1)

                value = fc1.bias.data[:, 2, :].reshape(-1)

                fc1.bias.data[:, 2, :] = value.div(scales.view(-1)).reshape(
                    fc1.bias[:, 2, :].shape
                )
                fc1.bias.data = fc1.bias.data.reshape(-1)
        else:
            assert fc1.out_features == fc2.in_features

            if hasattr(fc1, 'bias') and fc1.bias is not None:
                fc1.bias.div_(scales.view(-1))

            fc1.weight.div_(scales.view(-1, 1))

        fc2.weight.mul_(scales.view(1, -1))

    @torch.no_grad()
    def shift_fc_fc(self, fc1, fc2, shifts):
        if fc1.out_features == fc2.in_features * 3:
            num_heads = self.model.get_model_config().to_dict().get('n_head', None)
            if hasattr(fc1, 'bias') and fc1.bias is not None:
                fc1.bias.data = fc1.bias.data.reshape(num_heads, 3, -1)

                value = fc1.bias.data[:, 2, :].reshape(-1)
                fc1.bias.data[:, 2, :] = (value - shifts).reshape(
                    fc1.bias[:, 2, :].shape
                )
                fc1.bias.data = fc1.bias.data.reshape(-1)
        else:
            assert fc1.out_features == fc2.in_features

            if hasattr(fc1, 'bias') and fc1.bias is not None:
                fc1.bias.sub_(shifts)

        if hasattr(fc2, 'bias') and fc2.bias is not None:
            fc2.bias.add_(fc2.weight @ shifts)
        else:
            if hasattr(self, 'use_shift') and self.use_shift:
                del fc2.bias
                fc2.register_buffer('bias', fc2.weight @ shifts)

    @torch.no_grad()
    def shift_ln_fcs(self, ln, fcs, shifts):
        if not isinstance(fcs, list):
            fcs = [fcs]

        if self.model.has_bias():
            ln.bias.sub_(shifts)

        for fc in fcs:
            if self.model.has_bias():
                fc.bias.add_(fc.weight @ shifts)
            else:
                if hasattr(self, 'use_shift') and self.use_shift:
                    del fc.bias
                    fc.register_buffer('bias', fc.weight @ shifts)

        for p in ln.parameters():
            assert torch.isnan(p).sum() == 0
        for fc in fcs:
            for p in fc.parameters():
                assert torch.isnan(p).sum() == 0

    @torch.no_grad()
    def scale_ln_fcs(self, ln, fcs, scales):
        if not isinstance(fcs, list):
            fcs = [fcs]
        scales = scales.to(ln.weight.device)
        ln.weight.div_(scales)

        if hasattr(ln, 'bias') and ln.bias is not None:
            ln.bias.div_(scales)

        for fc in fcs:
            fc.weight.mul_(scales.view(1, -1))

        for p in ln.parameters():
            assert torch.isnan(p).sum() == 0
        for fc in fcs:
            for p in fc.parameters():
                assert torch.isnan(p).sum() == 0

    @torch.no_grad()
    def auto_clip(self, block, input_feat, n_sample_token):
        # auto clip
        for n, m in block.named_modules():
            if not check_do_quant(
                self.block_idx, n, self.mix_bits_map, self.quantizer_mix_bits
            ):
                logger.info(
                    f'This layer {n} in {self.block_idx}-th block is set to float.'
                    f'No need to clip this layer.'
                )
                continue
            if isinstance(m, tuple(_LLMC_LINEAR_TYPES_ + _TRANSFORMERS_LINEAR_TYPES_)):
                m = m.cuda()
                if any([_ in n for _ in ['q_', 'k_', 'query', 'key', 'Wqkv']]):
                    if self.clip_version == 'v2':
                        m.register_buffer('buf_upbound_factor', None)
                        m.register_buffer('buf_lowbound_factor', None)
                    continue
                logger.info(f'clip layer: {n}')

                if len(input_feat[n]) != 1:
                    inputs = [torch.cat(input_feat[n])]
                else:
                    inputs = input_feat[n]

                max_val, min_val = self.auto_clip_layer(
                    n,
                    m.weight,
                    inputs,
                    n_sample_token=n_sample_token,
                )

                dist.all_reduce(max_val, op=dist.ReduceOp.SUM)
                max_val /= int(os.environ['WORLD_SIZE'])

                dist.all_reduce(min_val, op=dist.ReduceOp.SUM)
                min_val /= int(os.environ['WORLD_SIZE'])

                self.apply_clip(m, min_val, max_val, n)

    @torch.no_grad()
    def apply_clip(self, layer, min_val, max_val, layer_name):
        if self.clip_version == 'v1':
            max_val = max_val.to(layer.weight.device)
            org_shape = layer.weight.shape
            try:
                layer.weight.data = layer.weight.data.reshape(*max_val.shape[:2], -1)
            except RuntimeError:
                layer.weight.data = self.wquantizer.reshape_tensor(layer.weight.data)
                layer.weight.data = layer.weight.data.reshape(*max_val.shape[:2], -1)
            if self.clip_sym:
                min_val = -max_val

            layer.weight.data = torch.clamp(layer.weight.data, min_val, max_val)
            try:
                layer.weight.data = layer.weight.data.reshape(org_shape)
            except RuntimeError:
                layer.weight.data = self.wquantizer \
                    .restore_tensor(layer.weight.data, org_shape)
        elif self.clip_version == 'v2':
            up_factor, low_factor = self.get_clip_factor(
                layer, min_val, max_val, layer_name
            )
            layer.register_buffer('buf_upbound_factor', up_factor)
            layer.register_buffer('buf_lowbound_factor', low_factor)
            if self.save_clip:
                if self.block_idx not in self.weight_clips:
                    self.weight_clips[self.block_idx] = dict()
                n = f'{layer_name}.weight_quantizer.'
                self.weight_clips[self.block_idx][
                    n + 'upbound_factor'
                ] = up_factor.cpu()
                if low_factor is not None:
                    self.weight_clips[self.block_idx][
                        n + 'lowbound_factor'
                    ] = low_factor.cpu()
                else:
                    self.weight_clips[self.block_idx][
                        n + 'lowbound_factor'
                    ] = None
        else:
            raise Exception('Not support other clip version')

    def get_clip_factor(self, layer, min_val, max_val, layer_name):
        wquantizer = get_wquantizer(
            self.block_idx,
            layer_name,
            self.mix_bits_map,
            self.quantizer_mix_bits,
            self.wquantizer,
        )
        org_min_val, org_max_val = wquantizer.get_minmax_range(
            wquantizer.reshape_tensor(layer.weight.data)
        )
        org_val_shape = org_max_val.shape

        if self.clip_sym:
            abs_max_val = torch.max(org_max_val.abs(), org_min_val.abs())
            abs_max_val = abs_max_val.clamp(min=1e-5)
            abs_max_val = abs_max_val.reshape(*max_val.shape[:2], -1)
            up_factor = self.logit((max_val / abs_max_val))
            up_factor = up_factor.reshape(org_val_shape)
            low_factor = None
        else:
            org_max_val = org_max_val.reshape(*max_val.shape[:2], -1)

            up_factor = self.logit((max_val / org_max_val))
            up_factor = up_factor.reshape(org_val_shape)

            org_min_val = org_min_val.reshape(*min_val.shape[:2], -1)
            low_factor = self.logit((min_val / org_min_val))
            low_factor = low_factor.reshape(org_val_shape)

        return up_factor, low_factor

    @torch.no_grad()
    def auto_clip_layer(
        self,
        layer_name,
        w,
        input,
        n_grid=20,
        max_shrink=0.5,
        n_sample_token=512,
        eps=0.0,
    ):
        assert w.dim() == 2

        wquantizer = get_wquantizer(
            self.block_idx,
            layer_name,
            self.mix_bits_map,
            self.quantizer_mix_bits,
            self.wquantizer,
        )
        if wquantizer.granularity == 'per_group':
            group_size = wquantizer.group_size
        else:
            group_size = w.shape[1]

        try:
            w = w.reshape(w.shape[0], 1, -1, group_size)
        except RuntimeError:
            w = self.wquantizer.reshape_tensor(w)
            w = w.reshape(w.shape[0], 1, -1, group_size)
        oc_batch_size = 256 if w.shape[0] % 256 == 0 else 64  # prevent OOM
        assert w.shape[0] % oc_batch_size == 0

        w_all = w
        best_max_val_all = []
        best_min_val_all = []

        for i_b in range(w.shape[0] // oc_batch_size):
            w = w_all[i_b * oc_batch_size: (i_b + 1) * oc_batch_size]

            if self.clip_sym:
                org_max_val = w.abs().amax(dim=-1, keepdim=True)
            else:
                org_max_val = w.amax(dim=-1, keepdim=True)

            org_min_val = w.amin(dim=-1, keepdim=True)

            best_max_val = org_max_val.clone()
            best_min_val = org_min_val.clone()
            min_errs = torch.ones_like(org_max_val) * 1e9
            org_out_dict = {}
            for i_s in range(int(max_shrink * n_grid)):
                if i_s == 0:
                    if self.clip_version == 'v2' and not check_w_only(
                        self.block_idx,
                        layer_name,
                        self.mix_bits_map,
                        self.quantizer_mix_bits,
                        self.w_only,
                    ):
                        i_s += eps
                err_mean = 0
                for i in range(len(input)):
                    input[i] = input[i].to(w.device)
                    x = input[i]
                    x = x.view(-1, x.shape[-1])
                    if self.padding_mask and self.padding_mask[i].numel() == x.shape[0]:
                        mask_tmp = self.padding_mask[i].flatten()
                        x = x[mask_tmp.bool()]
                    try:
                        x = x.reshape(1, x.shape[0], -1, group_size)
                    except RuntimeError:
                        x = self.wquantizer.reshape_tensor(x)
                        x = x.reshape(1, x.shape[0], -1, group_size)
                    if n_sample_token is None:
                        n_sample_token = min(x.shape[1], 512)
                    x = x[:, 0:: x.shape[1] // n_sample_token]
                    if i in org_out_dict:
                        org_out = org_out_dict[i]
                    else:
                        org_out = (x * w).sum(dim=-1)
                        org_out_dict[i] = org_out

                    max_val = org_max_val * (1 - i_s / n_grid)

                    if self.clip_sym:
                        min_val = -max_val
                    else:
                        min_val = org_min_val * (1 - i_s / n_grid)

                    if self.clip_version == 'v1':
                        cur_w = torch.clamp(w, min_val, max_val)
                        q_w = wquantizer.fake_quant_weight_dynamic(cur_w)
                    elif self.clip_version == 'v2':
                        low_factor = self.logit((min_val / org_min_val))
                        up_factor = self.logit((max_val / org_max_val))
                        tensor_range = wquantizer.get_learnable_range(
                            w, low_factor, up_factor
                        )

                        scales, zeros, qmax, qmin = wquantizer.get_qparams(
                            tensor_range, w.device
                        )
                        args = {}
                        args['scales'] = scales
                        args['zeros'] = zeros
                        args['qmax'] = qmax
                        args['qmin'] = qmin
                        q_w = wquantizer.fake_quant_weight_static(w, args)
                    else:
                        raise Exception('Not support other clip version')

                    if not check_w_only(
                        self.block_idx,
                        layer_name,
                        self.mix_bits_map,
                        self.quantizer_mix_bits,
                        self.w_only,
                    ):
                        q_x = get_aquantizer(
                            self.block_idx,
                            layer_name,
                            self.mix_bits_map,
                            self.quantizer_mix_bits,
                            self.aquantizer,
                        ).fake_quant_act_dynamic(x)
                    else:
                        q_x = x

                    cur_out = (q_x * q_w).sum(dim=-1)

                    # co, 1, n_group, 1
                    err = (cur_out - org_out).pow(2).mean(dim=1).view(min_errs.shape)
                    err_mean += err

                    if self.clip_version == 'v1':
                        del cur_w
                    del cur_out

                err_mean /= len(input)
                cur_best_idx = err_mean < min_errs

                min_errs[cur_best_idx] = err_mean[cur_best_idx]
                best_max_val[cur_best_idx] = max_val[cur_best_idx]
                best_min_val[cur_best_idx] = min_val[cur_best_idx]

            best_max_val_all.append(best_max_val)
            best_min_val_all.append(best_min_val)

        best_max_val = torch.cat(best_max_val_all, dim=0)
        best_min_val = torch.cat(best_min_val_all, dim=0)

        del org_out
        del org_out_dict
        gc.collect()
        torch.cuda.empty_cache()
        return best_max_val.squeeze(1), best_min_val.squeeze(1)

    def rotate_pre_layers(self, pre_layers, Q):
        for layer in pre_layers:
            dtype = layer.weight.dtype
            device = layer.weight.data.device
            W = layer.weight.data.to(device=device, dtype=torch.float64)
            layer.weight.data = torch.matmul(W, Q).to(device='cpu', dtype=dtype)

    def rotate_post_layers(self, post_layers, Q, exact_had=False):
        for layer in post_layers:
            dtype = layer.weight.dtype
            device = layer.weight.data.device
            W = layer.weight.data.to(device=device, dtype=torch.float64)
            layer.weight.data = torch.matmul(Q.T, W).to(device='cpu', dtype=dtype)

            if exact_had and self.online_rotate:
                apply_exact_had_to_linear(layer, had_dim=-1, output=False)

            if hasattr(layer, 'bias') and layer.bias is not None:
                b = layer.bias.data.to(device=device, dtype=torch.float64)
                layer.bias.data = torch.matmul(Q.T, b).to(device='cpu', dtype=dtype)

    def rotate_embeddings(self, Q):
        embeddings = self.model.get_embed_layers()
        assert len(embeddings) == 1
        for layer in embeddings:
            dtype = layer.weight.data.dtype
            W = layer.weight.data.to(device=self.dev, dtype=torch.float64)
            layer.weight.data = torch.matmul(W, Q).to(device='cpu', dtype=dtype)

    def rotate_head(self, Q):
        heads = self.model.get_head_layers()
        for layer in heads:
            dtype = layer.weight.data.dtype
            W = layer.weight.data.to(device=self.dev, dtype=torch.float64)
            layer.weight.data = torch.matmul(W, Q).to(device='cpu', dtype=dtype)

    def fuse_ln_fcs(self, ln, fcs):
        for fc in fcs:
            fc_dtype = fc.weight.dtype
            W = fc.weight.data.double()
            fc.weight.data = (W * ln.weight.double()).to(fc_dtype)
            if hasattr(ln, 'bias') and ln.bias is not None:
                if fc.bias is None:
                    fc.bias = torch.nn.Parameter(
                        torch.zeros(fc.out_features, dtype=torch.float64)
                    )
                fc.bias.data = fc.bias.data.double().to(device=W.device) \
                    + torch.matmul(W, ln.bias.double())
                fc.bias.data = fc.bias.data.to(fc_dtype)

    def remove_mean_from_embed(self):
        embeddings = self.model.get_embed_layers()
        for layer in embeddings:
            W = layer.weight.data.double()
            layer.weight.data = (W - W.mean(dim=-1, keepdim=True)).to(
                layer.weight.data.dtype
            )

    def bake_mean_into_fc(self, fc):
        fc_dtype = fc.weight.dtype
        W_ = fc.weight.data.double()
        fc.weight.data = W_ - W_.mean(dim=-2, keepdim=True)
        fc.weight.data = fc.weight.data.to(fc_dtype)
        if hasattr(fc, 'bias') and fc.bias is not None:
            b_ = fc.bias.data.double()
            fc.bias.data = b_ - b_.mean()
            fc.bias.data = fc.bias.data.to(fc_dtype)

    @torch.no_grad()
    def update_input_feat(self, scale, input_feat, layers_dict):
        for layer_name in layers_dict:
            for i in range(len(input_feat[layer_name])):
                inp = input_feat[layer_name][i]
                inp.div_(scale.view(1, -1).to(inp.device))

    @torch.no_grad()
    def set_non_linear_mode(self, quant_format, module, mode):
        assert mode in [True, False]
        if quant_format != 'fake_quant':
            return
        for name, m in module.named_modules():
            if getattr(m, 'calib', None) is not None:
                m.calib = mode
                # logger.info(f'{m} has set calibration mode {mode}.')

    @torch.no_grad()
    def deploy(self, quant_format, keep_device=False):
        logger.info(f'-- deploy_{quant_format}_model start --')
        logger.info(f'quant_config : {self.quant_config}')

        module_mapping = {
            'origin_float': OriginFloatLinear,
            'fake_quant': EffcientFakeQuantLinear
        }
        module_mapping.update(_REALQUANT_LINEAR_MAP_)

        if quant_format not in module_mapping:
            raise NotImplementedError(
                f"Quant format '{quant_format}' is not implemented."
            )

        module = module_mapping[quant_format]
        self.model.replace_module_all(
            module,
            self.get_replacement_params(mode=quant_format, w_only=self.w_only),
            keep_device=keep_device
        )
        self.set_non_linear_mode(quant_format, self.model.model, False)

        logger.info(f'-- deploy_{quant_format}_model done --')

    @torch.no_grad()
    def copy_tokenizer(self, path):
        for substring in self.config.save.get('tokenizer_file_substring',
                                              ['token', 'merges', 'vocab']):
            copy_files(self.config.model.path, path, substring)
        logger.info('copy tokenizer done --')

    @torch.no_grad()
    def contiguous_params(self):
        for name, param in self.model.model.named_parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()

        for name, param in self.model.model.named_buffers():
            if not param.is_contiguous():
                param.data = param.data.contiguous()

    @torch.no_grad()
    def save_model(self, path):
        if int(os.environ['RANK']) != 0:
            return
        self.contiguous_params()
        if self.config.model.type in ['Llava', 'InternVL2']:
            self.model.vlm_model.language_model = self.model.get_model()
            self.model.vlm_model.save_pretrained(path)
            logger.info('save model done --')
            self.copy_tokenizer(path)
            copy_files(self.config.model.path, path, 'preprocessor_config')
        elif self.config.model.type in ['InternOmni']:
            self.model.avlm_model.language_model = self.model.get_model()
            self.model.avlm_model.save_pretrained(path)
            logger.info('save model done --')
            self.copy_tokenizer(path)
            copy_files(self.config.model.path, path, 'preprocessor_config')
        else:
            self.model.get_model().save_pretrained(path)
            logger.info('save model done --')
            self.copy_tokenizer(path)
