import copy
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
from llmc.utils.registry_factory import KV_REGISTRY

from ..blockwise_optimization import BlockwiseOpt
from .auto_clip import AutoClipper
from .hadamard_utils import apply_exact_had_to_linear, get_hadK
from .module_utils import (_LLMC_ATTN_MAP_, _LLMC_LINEAR_TYPES_,
                           _LLMC_LN_TYPES_, _REALQUANT_LINEAR_MAP_,
                           _TRANSFORMERS_LINEAR_TYPES_,
                           _TRANSFORMERS_LN_TYPES_, EffcientFakeQuantLinear,
                           FakeQuantLinear, LlmcActFn, OriginFloatLinear,
                           RotateLinear)
from .quant import FloatQuantizer, IntegerQuantizer
from .utils import check_do_quant, check_w_only, get_aquantizer, get_wquantizer


class BaseBlockwiseQuantization(BlockwiseOpt):
    def __init__(
        self, model, quant_config, input, padding_mask, config, modality='language'
    ):
        super().__init__(model, quant_config, input, padding_mask, config, modality)
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
                'scales': (getattr(module, f'buf_act_scales_{input_index}', None)),
                'zeros': (getattr(module, f'buf_act_zeros_{input_index}', None)),
                'qmax': (getattr(module, f'buf_act_qmax_{input_index}', None)),
                'qmin': (getattr(module, f'buf_act_qmin_{input_index}', None)),
            }
            return aquantizer.fake_quant_act_static(act, args)
        else:
            return aquantizer.fake_quant_act_dynamic(act)

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
                'matmul_a1_qdq': partial(
                    self.a_qdq, aquantizer=self.aquantizer, input_index=0
                ),
                'matmul_a2_qdq': partial(
                    self.a_qdq, aquantizer=self.aquantizer, input_index=1
                ),
                'softmax_a_qdq': (
                    partial(self.a_qdq, aquantizer=self.aquantizer)
                    if self.quant_softmax
                    else None
                ),
            }

        elif mode == 'quant_act_fn':
            params_dict = {'a_qdq': partial(self.a_qdq, aquantizer=self.aquantizer)}

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
        self.quant_type = self.quant_config.get('quant_type', 'int-quant')
        if self.quant_type == 'int-quant':
            self.quant_module = IntegerQuantizer
        elif self.quant_type == 'float-quant':
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
                assert (
                    self.quant_config['act']['granularity'] == 'per_tensor'
                ), 'Only support per_tensor static quant'
            self.quant_attn = self.quant_config['act'].get('quant_attn', False)
            if self.quant_attn:
                assert self.config['model']['type'] in ['Vit', 'DeepseekV2']
                self.quant_softmax = self.quant_config['act'].get(
                    'quant_softmax', False
                )
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

        # set kv cache quant config
        if 'kvcache' in self.quant_config:
            self.quant_config['kvcache']['static'] = self.act_static
            self.kv_module = KV_REGISTRY[self.quant_config['kvcache']['method']](
                self.quant_type, self.quant_config['kvcache'],
                self.model.model_config.num_hidden_layers, self.config.calib.n_samples,
                self.config.calib.bs
            )
            self.quant_kvcache = True
            self.model.kvcache_buffer.append(self.kv_module)
        else:
            self.quant_kvcache = False

        # set special quant config
        special_config = self.quant_config.get('special', {})
        self.true_sequential = special_config.get('true_sequential', False)

        # set weight clip config
        self.weight_clip = special_config.get('weight_clip', False)
        if self.weight_clip:
            self.save_clip = special_config.get('save_clip', False)
            if self.save_clip:
                self.clip_path = special_config['clip_path']
            self.clip_version = special_config.get('clip_version', 'v1')
            if self.clip_version == 'v2':
                assert self.wquantizer.calib_algo == 'learnable'
            clip_sym = special_config.get('clip_sym', self.wquantizer.sym)
            self.auto_clipper = AutoClipper(
                w_only=self.w_only,
                mix_bits_map=self.mix_bits_map,
                quantizer_mix_bits=self.quantizer_mix_bits,
                wquantizer=self.wquantizer,
                aquantizer=self.aquantizer,
                clip_version=self.clip_version,
                clip_sym=clip_sym,
                save_clip=self.save_clip,
                padding_mask=self.padding_mask,
            )

        # set transformation config
        self.save_scale = special_config.get('save_scale', False)
        if self.save_scale:
            self.scale_path = special_config['scale_path']
            self.act_scales = {}

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

        self.quant_objects = self.quant_config.get('quant_objects', ['language'])
        logger.info(f'self.quant_objects : {self.quant_objects}')

    def replace_rotate_linears(self, block):
        for n, m in block.named_modules():
            if isinstance(m, nn.Linear) and (
                'down_proj' in n or 'o_proj' in n or 'fc2' in n or 'out_proj' in n
            ):
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
        attn_module = _LLMC_ATTN_MAP_[self.config['model']['type']]
        self.model.replace_module_subset(
            attn_module,
            block,
            layers_dict,
            self.block_idx,
            self.get_replacement_params(
                mode='quant_attn', w_only=self.w_only, name=None
            ),
        )

        matmul_modules = self.model.get_matmul_in_block(block)
        softmax_modules = (
            self.model.get_softmax_in_block(block) if self.quant_softmax else {}
        )
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
            for k in self.input['kwargs'][i]:
                if torch.is_tensor(self.input['kwargs'][i][k]):
                    self.input['kwargs'][i][k] = self.input['kwargs'][i][k].to(
                        device=next(block.parameters()).device
                    )  # noqa
                if isinstance(self.input['kwargs'][i][k], tuple):
                    self.input['kwargs'][i][k] = tuple(
                        tmp.to(device=next(block.parameters()).device)
                        for tmp in self.input['kwargs'][i][k]
                    )  # noqa
            with torch.no_grad():
                out = block(input_data[i], **self.input['kwargs'][i])
                if isinstance(out, tuple):
                    out = out[0]
                output.append(out)
        return output

    def block_opt(self, block):

        if self.quant_kvcache:
            self.register_kv_cache(block)

        block = block.cuda()
        named_linears = self.model.get_block_linears(block)
        extra_modules = self.model.get_extra_modules(block)

        if self.quant_attn:
            self.replace_attention(block, extra_modules)
        if self.quant_act_fn:
            self.replace_act_fn(block, extra_modules)

        input_feat_modules = {
            k: v for d in [named_linears, extra_modules] for k, v in d.items()
        }
        logger.info(f'input_feat_modules: {input_feat_modules}')
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
        torch.cuda.empty_cache()

    def block_transform(self, block, input_feat, block_kwargs):
        logger.info(f'Start transform the {self.block_idx}-th block')
        subsets = (
            self.model.get_subsets_in_block(block)
            if self.modality == 'language'
            else self.model.get_vision_subsets_in_block(block)
        )

        if self.act_static:
            self.register_non_linear_qparams(block, input_feat)

        self.set_non_linear_mode('fake_quant', block, False)

        for index, subset in enumerate(subsets):
            logger.info(f'subset: {subset}')
            layers_dict = subset['layers']
            input_name = subset['input'][0]
            inspect_has_kwargs = subset['has_kwargs']
            if inspect_has_kwargs:
                if 'sub_keys' in subset:
                    subset_kwargs = [
                        {k: block_kwargs[0][v] for k, v in subset['sub_keys'].items()}
                    ]
                else:
                    subset_kwargs = block_kwargs
            else:
                subset_kwargs = {}
            self.subset_transform(
                subset,
                input_feat,
                subset_kwargs,
            )
            if self.act_static:
                input_tensors = copy.deepcopy(input_feat[input_name])
                self.register_act_qparams(layers_dict, input_tensors)
                del input_tensors

            if self.true_sequential and index != len(subsets) - 1:
                next_subset = subsets[index + 1]
                input_feat_subset = self.rehook_next_subset(block, subset, next_subset)
                input_feat.update(input_feat_subset)

        self.set_non_linear_mode('fake_quant', block, True)
        logger.info(f'End transform the {self.block_idx}-th block')

    def rehook_next_subset(self, block, subset, next_subset):
        self.subset_init(next_subset)
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

    def collect_layers_weights(self, layers, tensor_parallelize_style=None):
        weights = []
        for _m in layers:
            weights.append(_m.weight)
        return weights

    @torch.no_grad()
    def register_kv_cache(self, block):
        attn_layers_dict = self.model.get_attn_in_block(block)
        attn_layer = attn_layers_dict[list(attn_layers_dict.keys())[0]]
        setattr(attn_layer, 'kvcache', self.kv_module)
        attn_layer.register_forward_pre_hook(
            self.kv_cache_input_hook(), with_kwargs=True
        )

    @torch.no_grad()
    def register_non_linear_qparams(self, block, input_feat):
        layer_types = [
            ('quant_attn', self.model.get_matmul_in_block),
            ('quant_softmax', self.model.get_softmax_in_block, 'quant_attn'),
            ('quant_act_fn', self.model.get_act_fn_in_block),
        ]

        for mode, layer_func, *dependency in layer_types:
            if getattr(self, mode, True) and all(
                getattr(self, dep, True) for dep in dependency
            ):
                layers_dict = layer_func(block)
                for name, layer in layers_dict.items():
                    input_tensors = copy.deepcopy(input_feat[name])
                    self.register_act_qparams({name: layer}, input_tensors)
                    del input_tensors

    @torch.no_grad()
    def register_act_qparams(self, layers_dict, act_tensors):
        scales_list, zeros_list, qmin_list, qmax_list = (
            self.aquantizer.get_batch_tensors_qparams(act_tensors)
        )
        world_size = int(os.environ['WORLD_SIZE'])

        for i, (scales, zeros, qmin, qmax) in enumerate(
            zip(scales_list, zeros_list, qmin_list, qmax_list)
        ):
            scales = scales.cuda()
            dist.all_reduce(scales, op=dist.ReduceOp.SUM)
            scales = scales / world_size

            for name, layer in layers_dict.items():
                if not isinstance(
                    layer, tuple(_LLMC_LINEAR_TYPES_ + _TRANSFORMERS_LINEAR_TYPES_)
                ):
                    continue
                layer.register_buffer(f'buf_act_scales_{i}', scales)
                layer.register_buffer(f'buf_act_zeros_{i}', zeros.cuda())
                layer.register_buffer(f'buf_act_qmin_{i}', qmin.cuda())
                layer.register_buffer(f'buf_act_qmax_{i}', qmax.cuda())

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
            logger.info('fc1.out_features == fc2.in_features * 3')
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
        elif fc1.out_features == fc2.in_features * 2:
            logger.info('fc1.out_features == fc2.in_features * 2')
            fc1.weight.data[fc1.weight.data.shape[0] // 2:].div_(scales.view(-1, 1))
            if hasattr(fc1, 'bias') and fc1.bias is not None:
                fc1.bias.data[fc1.bias.data.shape[0] // 2:].div_(scales.view(-1))
        elif fc1.out_features == fc2.in_features:
            logger.info('fc1.out_features == fc2.in_features')
            assert fc1.out_features == fc2.in_features

            if hasattr(fc1, 'bias') and fc1.bias is not None:
                fc1.bias.div_(scales.view(-1))

            fc1.weight.div_(scales.view(-1, 1))
        else:
            logger.error(f'fc1.out_features: {fc1.out_features}')
            logger.error(f'fc2.in_features: {fc2.in_features}')
            raise Exception('Can not scale this fc-fc.')

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
                fc.bias.data = fc.bias.data.double().to(device=W.device) + torch.matmul(
                    W, ln.bias.double()
                )
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
            if 'kvcache' in name:
                continue
            if getattr(m, 'calib', None) is not None:
                m.calib = mode

    @torch.no_grad()
    def deploy(self, quant_format, keep_device=False):
        logger.info(f'-- deploy_{quant_format}_model start --')
        logger.info(f'quant_config : {self.quant_config}')

        module_mapping = {
            'origin_float': OriginFloatLinear,
            'fake_quant': EffcientFakeQuantLinear,
        }
        module_mapping.update(_REALQUANT_LINEAR_MAP_)

        if quant_format not in module_mapping:
            raise NotImplementedError(
                f"Quant format '{quant_format}' is not implemented."
            )

        module = module_mapping[quant_format]
        if 'vision' in self.quant_objects:
            self.model.replace_vision_module_all(
                module,
                self.get_replacement_params(mode=quant_format, w_only=self.w_only),
                keep_device=keep_device,
            )
        if 'language' in self.quant_objects:
            self.model.replace_language_module_all(
                module,
                self.get_replacement_params(mode=quant_format, w_only=self.w_only),
                keep_device=keep_device,
            )
        self.set_non_linear_mode(quant_format, self.model.model, False)

        if self.quant_kvcache:
            if quant_format == 'transformed':
                self.kv_module.transformed = True
            elif quant_format == 'fake_quant':
                self.kv_module.transformed = False
                if self.act_static:
                    self.kv_module.calib = False

        if self.model.vlm_model is not None:
            logger.info(f'Now, the vlm_model is: {self.model.vlm_model}')

        logger.info(f'-- deploy_{quant_format}_model done --')

    @torch.no_grad()
    def copy_tokenizer(self, path):
        for substring in self.config.save.get(
            'tokenizer_file_substring', ['token', 'merges', 'vocab']
        ):
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
        if self.config.model.type in ['Llava', 'InternVL2', 'Mllama', 'Qwen2vl']:
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
