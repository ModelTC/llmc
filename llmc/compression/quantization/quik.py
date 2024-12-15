import functools
import gc

import torch
import torch.nn as nn
from tqdm import tqdm

from llmc.utils.registry_factory import ALGO_REGISTRY

from .base_blockwise_quantization import BaseBlockwiseQuantization


@ALGO_REGISTRY
class QUIK(BaseBlockwiseQuantization):
    def __init__(self, model, quant_config, input, padding_mask, config):
        super().__init__(model, quant_config, input, padding_mask, config)
        self.add_quant_config()

    def add_quant_config(self):
        self.prefix = self.model.block_name_prefix
        self.fp_relative = self.quant_config['special']['fp_relative']
        self.fp_features = self.quant_config['special']['fp_features']
        self.fp_threshold = self.quant_config['special']['fp_threshold']
        if 'last_fc_bit' in self.quant_config:
            self.last_fc_bit = self.quant_config['special']['last_fc_bit']
        self.act_scales = self.get_act_scale_shift(stat='scales')
        self.int_ids = {}
        self.fp_ids = {}

    def get_act_scale_shift(self, stat='scales'):
        self.model.model.eval()

        act_stat = {}

        def get_tensor_scale(name, tensor):
            hidden_dim = tensor.shape[-1]
            tensor = tensor.view(-1, hidden_dim).abs().detach()
            comming_max = torch.max(tensor, dim=0)[0].float().cpu()
            if name in act_stat:
                act_stat[name] = torch.max(act_stat[name], comming_max)
            else:
                act_stat[name] = comming_max

        def get_tensor_shift(name, tensor):
            hidden_dim = tensor.shape[-1]
            tensor = tensor.view(-1, hidden_dim).detach()
            comming_max = torch.max(tensor, dim=0)[0].float().cpu()
            comming_min = torch.min(tensor, dim=0)[0].float().cpu()
            if name in act_stat:
                act_stat[name] = 0.99 * act_stat[name] + 0.01 * (
                    (comming_max + comming_min) / 2
                )
            else:
                act_stat[name] = (comming_max + comming_min) / 2

        def stat_input_hook(m, x, y, name):
            if isinstance(x, tuple):
                x = x[0]
            if stat == 'scales':
                get_tensor_scale(name, x)
            elif stat == 'shifts':
                get_tensor_shift(name, x)

        hooks = []
        for name, m in self.model.model.named_modules():
            if isinstance(m, nn.Linear):
                hooks.append(
                    m.register_forward_hook(
                        functools.partial(stat_input_hook, name=name)
                    )
                )

        with torch.no_grad():
            for i in tqdm(range(len(self.blocks))):
                block = self.blocks[i]
                block.cuda()
                if i == 0:
                    fp_inps = self.block_forward(block)
                else:
                    fp_inps = self.block_forward(block, fp_inps)

                block.cpu()

        for h in hooks:
            h.remove()
        gc.collect()
        torch.cuda.empty_cache()

        return act_stat

    def block_opt(self, block):
        layers_dict = self.model.get_block_linears(block)
        for n, m in layers_dict.items():
            layer_name = f'{self.prefix}.{self.block_idx}.{n}'

            if self.fp_relative:
                outlier_num = (
                    int(block.in_features / self.model.model.config.hidden_size)
                    * self.fp_features
                )
            else:
                outlier_num = self.fp_features

            layer_scales = None
            if outlier_num > 0:
                layer_scales = self.act_scales[layer_name]
                max_val = layer_scales.abs().max()

                fp_threshold = self.fp_threshold
                if hasattr(self, 'last_fc_bit'):
                    if 'dense_4h_to_h' in n or 'down_proj' in n:
                        fp_threshold = self.fp_threshold * 2
                        m.register_buffer(
                            'buf_current_bit', torch.tensor(self.last_fc_bit)
                        )

                if max_val <= fp_threshold:
                    outlier_num = 0
                    layer_scales = None

            int_indices = torch.sort(layer_scales)[1][:-outlier_num]
            fp_indices = torch.sort(layer_scales)[1][-outlier_num:]

            m.register_buffer('buf_int_ids', int_indices)
            m.register_buffer('buf_fp_ids', fp_indices)
            del self.act_scales[layer_name]

    def w_qdq(self, module, wquantizer):
        weight = module.weight
        args = {}
        args['int_indices'] = module.buf_int_ids
        args['fp_indices'] = module.buf_fp_ids

        if hasattr(module, 'buf_current_bit'):
            args['current_bit'] = module.buf_current_bit

        weight = self.wquantizer.fake_quant_weight_dynamic(weight, args)

        return weight

    def a_qdq(self, act, module, aquantizer):
        args = {}
        args['int_indices'] = module.buf_int_ids
        args['fp_indices'] = module.buf_fp_ids

        if hasattr(module, 'buf_current_bit'):
            args['current_bit'] = module.buf_current_bit

        act = self.aquantizer.fake_quant_act_dynamic(act, args)

        return act
