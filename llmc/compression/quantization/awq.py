import gc
import os

import torch
import torch.distributed as dist
import torch.nn as nn
from loguru import logger

from llmc.utils.registry_factory import ALGO_REGISTRY

from .base_blockwise_quantization import BaseBlockwiseQuantization
from .utils import is_fp8_supported_gpu

if is_fp8_supported_gpu():
    from .kernel import weight_cast_to_bf16, weight_cast_to_fp8
    logger.info('import kernel successful.')
else:
    from .quant import weight_cast_to_bf16, weight_cast_to_fp8
    logger.info('import quant successful.')

from .module_utils import (_LLMC_LINEAR_TYPES_, _LLMC_LN_TYPES_,
                           _TRANSFORMERS_LINEAR_TYPES_,
                           _TRANSFORMERS_LN_TYPES_, FakeQuantLinear)
from .utils import check_do_quant, check_w_only, get_aquantizer, get_wquantizer


@ALGO_REGISTRY
class Awq(BaseBlockwiseQuantization):
    def __init__(self, model, quant_config, input, padding_mask, config):
        super().__init__(model, quant_config, input, padding_mask, config)
        special_config = self.quant_config.get('special', {})
        self.trans = special_config.get('trans', True)
        self.trans_version = special_config.get('trans_version', 'v2')
        self.save_scale = special_config.get('save_scale', False)
        self.awq_bs = special_config.get('awq_bs', None)
        self.save_mem = special_config.get('save_mem', True)

    @torch.no_grad()
    def scaling_weight(self, w, scales, is_gqa):
        if is_gqa:
            scales_tmp = self.repeat_gqa_scales(scales)
        else:
            scales_tmp = scales
        w.mul_(scales_tmp.view(1, -1))
        return w

    def get_weight_scale(self, layers_dict):
        layers = list(layers_dict.values())
        total_scale = None
        first_layer_name = list(layers_dict.keys())[0]

        wquantizer = get_wquantizer(
            self.block_idx,
            first_layer_name,
            self.mix_bits_map,
            self.quantizer_mix_bits,
            self.wquantizer,
        )

        for idx, _m in enumerate(layers):
            if _m.weight.data.dtype == torch.float8_e4m3fn:
                weight = weight_cast_to_bf16(_m.weight.data,
                                             _m.weight_scale_inv.data).to(torch.bfloat16)
            else:
                weight = _m.weight.data.clone()
            org_shape = weight.shape
            reshaped = wquantizer.reshape_tensor(weight)
            abs_weights = reshaped.abs()
            max_vals = abs_weights.amax(dim=1, keepdim=True)
            layer_scale = abs_weights.div_(max_vals)
            layer_scale = layer_scale.view(org_shape)
            if total_scale is None:
                total_scale = layer_scale.mean(0)
            else:
                total_scale.add_(layer_scale.mean(0))
            del weight, reshaped, abs_weights, max_vals, layer_scale
            torch.cuda.empty_cache()

        return total_scale.div_(len(layers))

    def get_act_scale(self, x):
        if x.shape[0] == self._bs:
            return x.abs().view(-1, x.shape[-1]).mean(0)
        else:
            batch_means = []
            b_num = x.shape[0] // self._bs
            for num in range(b_num):
                batch_x = x[num * self._bs:(num + 1) * self._bs]
                batch_mean = batch_x.abs().view(-1, batch_x.shape[-1]).mean(0)
                batch_means.append(batch_mean)
            final_mean = sum(batch_means) / len(batch_means)
            return final_mean

    @torch.no_grad()
    def get_scales(self, prev_op, x, w_max, is_gqa, ratio):
        if is_gqa:
            x_tmp = prev_op(x)
            w_tmp = self.get_weight_scale({'prev_op': prev_op})
        else:
            x_tmp = x
            w_tmp = w_max

        x_tmp = self.get_act_scale(x_tmp)

        if self.trans_version == 'v1' and not is_gqa:
            scales = (
                (x_tmp.pow(ratio) / w_tmp.pow(1 - ratio))
                .clamp(min=1e-4)
                .view(-1)
            )
        elif self.trans_version == 'v2' or is_gqa:
            scales = x_tmp.pow(ratio).clamp(min=1e-4).view(-1)

        scales = scales / (scales.max() * scales.min()).sqrt()
        return scales

    def inspect_module_forward(self, x, inspect_module, kwargs):
        if self._bs == x.shape[0]:
            with torch.no_grad():
                out = inspect_module(x, **kwargs)
                if isinstance(out, tuple):
                    out = out[0]
            return out
        else:
            outs = []
            b_num = x.shape[0] // self._bs
            for num in range(b_num):
                _x = x[num * self._bs:(num + 1) * self._bs]
                out = inspect_module(_x, **kwargs)
                if isinstance(out, tuple):
                    out = out[0]
                outs.append(out)
            return torch.cat(outs, dim=0)

    @torch.no_grad()
    def get_original_out(self, x, inspect_module, subset_kwargs):
        with torch.no_grad():
            org_out = self.inspect_module_forward(x, inspect_module, subset_kwargs)
        return org_out

    def calculate_loss(self, org_out, out):
        if out.shape[0] == self._bs:
            return (org_out - out).float().pow(2).mean().item()
        else:
            total_loss = 0.0
            b_num = org_out.shape[0] // self._bs
            for num in range(b_num):
                _org_out = org_out[num * self._bs:(num + 1) * self._bs]
                _out = out[num * self._bs:(num + 1) * self._bs]
                single_loss = (_org_out - _out).float().pow(2).mean().item()
                total_loss += single_loss
            return total_loss / b_num

    def fake_quantize_weight(self, fc, scales, is_gqa, layer_name):
        if fc.weight.data.dtype == torch.float8_e4m3fn:
            tmp_weight_data = weight_cast_to_bf16(fc.weight.data,
                                                  fc.weight_scale_inv.data).to(torch.bfloat16)
        else:
            tmp_weight_data = fc.weight.data

        tmp_weight_data = self.scaling_weight(tmp_weight_data, scales, is_gqa)
        tmp_weight_data = get_wquantizer(
            self.block_idx,
            layer_name,
            self.mix_bits_map,
            self.quantizer_mix_bits,
            self.wquantizer,
        ).fake_quant_weight_dynamic(tmp_weight_data)

        if fc.weight.data.dtype == torch.float8_e4m3fn:
            fc.weight.data, fc.weight_scale_inv.data = weight_cast_to_fp8(tmp_weight_data)
        else:
            fc.weight.data = tmp_weight_data

        return fc.weight

    def fake_quantize_input(self, x_tmp, layers_dict):
        if self._bs == x_tmp.shape[0]:
            x_tmp = get_aquantizer(
                self.block_idx,
                list(layers_dict.keys())[0],
                self.mix_bits_map,
                self.quantizer_mix_bits,
                self.aquantizer,
            ).fake_quant_act_dynamic(x_tmp)
        else:
            outs = []
            for i in range(x_tmp.shape[0]):
                _x = x_tmp[i]
                _x = get_aquantizer(
                    self.block_idx,
                    list(layers_dict.keys())[0],
                    self.mix_bits_map,
                    self.quantizer_mix_bits,
                    self.aquantizer,
                ).fake_quant_act_dynamic(_x)
                outs.append(_x)
            x_tmp = torch.stack(outs)
        return x_tmp

    @torch.no_grad()
    def search_scale_subset(
        self,
        prev_op,
        layers_dict,
        input,
        inspect_module,
        is_gqa,
        subset_kwargs
    ):

        if self.awq_bs is None:
            self._bs = input[0].shape[0]
        else:
            self._bs = self.awq_bs

        w_max = self.get_weight_scale(layers_dict)
        # grid search for ratio
        best_error = float('inf')
        best_scales = None
        n_grid = 20
        org_sd = {k: v.cpu() for k, v in inspect_module.state_dict().items()}

        org_out_dict = {}
        for n in range(n_grid):
            loss_mean = 0
            scales_mean = 0
            for i in range(len(input)):
                input[i] = input[i].to(next(inspect_module.parameters()).device)
                x = input[i]
                if isinstance(subset_kwargs, list):
                    kwargs = subset_kwargs[i]
                else:
                    kwargs = subset_kwargs
                if i in org_out_dict:
                    org_out = org_out_dict[i]
                else:
                    org_out = self.get_original_out(x, inspect_module, kwargs)
                    org_out_dict[i] = org_out

                ratio = n * 1 / n_grid
                scales = self.get_scales(prev_op, x, w_max, is_gqa, ratio)
                for layer_name in layers_dict:
                    fc = layers_dict[layer_name]
                    fc.weight = self.fake_quantize_weight(fc, scales, is_gqa, layer_name)

                x_tmp = self.scaling_input(x, scales, is_gqa)

                if not check_w_only(
                    self.block_idx,
                    list(layers_dict.keys())[0],
                    self.mix_bits_map,
                    self.quantizer_mix_bits,
                    self.w_only,
                ):
                    x_tmp = self.fake_quantize_input(x_tmp, layers_dict)

                out = self.inspect_module_forward(x_tmp, inspect_module, kwargs)

                if self.padding_mask and org_out.shape[1] == self.padding_mask[i].shape[-1]:
                    org_out = org_out * self.padding_mask[i].unsqueeze(dim=-1).to(org_out.device)  # noqa
                    out = out * self.padding_mask[i].unsqueeze(dim=-1).to(out.device)

                loss = self.calculate_loss(org_out, out)

                if len(input) == 1:
                    n_samples = x.shape[0]
                else:
                    n_samples = self.n_samples

                loss_mean += x.shape[0] * 1.0 / n_samples * loss
                scales_mean += x.shape[0] * 1.0 / n_samples * scales
                inspect_module.load_state_dict(org_sd)
                is_best = loss_mean < best_error
                if is_best:
                    best_error = loss_mean
                    best_scales = scales_mean
                if self.save_mem:
                    del org_out
                    del out
                    gc.collect()
                    torch.cuda.empty_cache()

        # Synchronize across ranks
        best_error_tensor = torch.tensor([best_error], device='cuda')
        dist.all_reduce(best_error_tensor, op=dist.ReduceOp.MIN)
        global_best_error = best_error_tensor.item()

        # Identify the rank with the minimum loss
        global_best_rank = torch.tensor([dist.get_rank()
                                        if abs(best_error - global_best_error) < 1e-5
                                        else -1],
                                        device='cuda')
        dist.all_reduce(global_best_rank, op=dist.ReduceOp.MAX)
        global_best_rank = global_best_rank.item()

        # Broadcast the best scales from the rank with the minimum loss to all ranks
        if dist.get_rank() == global_best_rank:
            dist.broadcast(best_scales, src=global_best_rank)
        else:
            best_scales = torch.zeros_like(best_scales, device='cuda')
            dist.broadcast(best_scales, src=global_best_rank)

        del org_out_dict
        gc.collect()
        torch.cuda.empty_cache()
        return best_scales

    @torch.no_grad()
    def block_transform(self, block, input_feat, block_kwargs):
        if self.trans:
            super().block_transform(block, input_feat, block_kwargs)

        if self.weight_clip:
            logger.info('auto_clip start')
            logger.info(f'clip version: {self.clip_version}')
            self.auto_clipper.run(
                block,
                self.block_idx,
                input_feat,
                n_sample_token=self.config.calib.get('seq_len', None)
            )
            logger.info('auto_clip finished')
        else:
            logger.info('disable weight clip')

    @torch.no_grad()
    def subset_transform(
        self,
        subset,
        input_feat,
        subset_kwargs,
    ):
        layers_dict = subset['layers']
        prev_op = subset['prev_op']
        input_name = subset['input'][0]
        inspect_module = subset['inspect']
        do_trans = subset.get('do_trans', True)
        if not do_trans:
            logger.info('do_trans is set to False. Do not transform this subset.')
            return

        if not check_do_quant(
            self.block_idx,
            list(layers_dict.keys())[0],
            self.mix_bits_map,
            self.quantizer_mix_bits,
        ):
            logger.info(
                'This subset is set to float. No need to transform this subset.'
            )
            return
        if self.config['model']['type'] == 'Starcoder':
            if isinstance(prev_op[0], (nn.Linear, FakeQuantLinear)):
                logger.info('Do not transform this subset.')
                return

        assert (
            len(prev_op) in (0, 1)
        ), 'Only support single prev_op. If multi prev_ops, code need to be updated.'

        if len(prev_op) == 0 or (len(prev_op) == 1 and prev_op[0] is None):
            logger.info('Cannot apply scale. Do not transform this subset.')
            return

        if isinstance(
            prev_op[0],
            tuple(
                _LLMC_LN_TYPES_ +
                _TRANSFORMERS_LN_TYPES_ +
                _LLMC_LINEAR_TYPES_ +
                _TRANSFORMERS_LINEAR_TYPES_
            ),
        ):
            layers = list(layers_dict.values())

            if (
                isinstance(prev_op[0], (nn.Linear, FakeQuantLinear))
                and prev_op[0].out_features != layers[0].in_features * 3
                and prev_op[0].out_features != layers[0].in_features * 2
                and prev_op[0].out_features != layers[0].in_features
            ):

                if self.has_gqa and self.do_gqa_trans:
                    is_gqa = True
                    input_keys = list(input_feat.keys())
                    input_name = input_keys[input_keys.index(input_name) - 1]
                else:
                    logger.info('Cannot apply scale. Do not transform this subset.')
                    return
            else:
                is_gqa = False

            scale = self.search_scale_subset(
                prev_op[0],
                layers_dict,
                input_feat[input_name],
                inspect_module,
                is_gqa,
                subset_kwargs
            )

            self.apply_scale(scale, prev_op, layers)
            self.update_input_feat(scale, input_feat, layers_dict, is_gqa)

            if self.save_scale:
                for n in layers_dict:
                    layer_name = f'{self.model.block_name_prefix}.{self.block_idx}.{n}'
                    self.act_scales[layer_name] = scale
        else:
            logger.info('Do not transform this subset.')
