import functools
import gc
from collections import defaultdict

import torch
import torch.nn as nn
from loguru import logger

from llmc.utils.registry_factory import ALGO_REGISTRY

from .base_blockwise_quantization import BaseBlockwiseQuantization
from .module_utils import (_LLMC_LINEAR_TYPES_, _LLMC_LN_TYPES_,
                           _TRANSFORMERS_LINEAR_TYPES_,
                           _TRANSFORMERS_LN_TYPES_, FakeQuantLinear,
                           OriginFloatLinear)
from .utils import is_fp8_supported_gpu

if is_fp8_supported_gpu():
    from .fp8_kernel import weight_cast_to_bf16, weight_cast_to_fp8
    logger.info('import fp8_kernel successful.')


@ALGO_REGISTRY
class OsPlus(BaseBlockwiseQuantization):
    def __init__(self, model, quant_config, input, padding_mask, config):
        torch.set_grad_enabled(False)
        super().__init__(model, quant_config, input, padding_mask, config)

    @torch.no_grad()
    def filter_subset(self, prev_op):
        if isinstance(prev_op[0], tuple(_LLMC_LN_TYPES_ + _TRANSFORMERS_LN_TYPES_)):
            return True
        else:
            return False

    @torch.no_grad()
    def get_original_out(self, x, inspect_module, subset_kwargs):
        with torch.no_grad():
            org_out = inspect_module(x, **subset_kwargs)
            if isinstance(org_out, tuple):
                org_out = org_out[0]
        return org_out

    @torch.no_grad()
    def search_scale_shift_subset(
        self, layers, input_feats, inspect_module, subset_kwargs
    ):
        org_sd = {k: v.cpu() for k, v in inspect_module.state_dict().items()}
        org_out_dict = {}

        for i in range(len(input_feats)):
            input_feats[i] = input_feats[i].to(next(inspect_module.parameters()).device)
            x = input_feats[i]

            if self.model.has_bias():
                if x.dim() == 3:
                    cmx = torch.amax(x, dim=(0, 1))
                    cmn = torch.amin(x, dim=(0, 1))
                elif x.dim() == 2:
                    cmx = torch.amax(x, dim=0)
                    cmn = torch.amin(x, dim=0)
                shift = (cmx + cmn) / 2
            else:
                shift = None

            if isinstance(subset_kwargs, list):
                kwargs = subset_kwargs[i]
            else:
                kwargs = subset_kwargs

            if len(input_feats) == 1:
                org_out = self.get_original_out(x, inspect_module, kwargs)
            else:
                if i in org_out_dict:
                    org_out = org_out_dict[i]
                else:
                    org_out = self.get_original_out(x, inspect_module, kwargs)
                    org_out_dict[i] = org_out

            if self.model.has_bias():
                x_shift = x - shift
            else:
                x_shift = x.clone()

            if x.dim() == 3:
                cmx = torch.amax(x_shift, dim=(0, 1))
                cmn = torch.amin(x_shift, dim=(0, 1))
            elif x.dim() == 2:
                cmx = torch.amax(x_shift, dim=0)
                cmn = torch.amin(x_shift, dim=0)
            amx = max(
                x_shift.max(), torch.tensor(0.0, dtype=x_shift.dtype).to(x_shift.device)
            )
            amn = min(
                x_shift.min(), torch.tensor(0.0, dtype=x_shift.dtype).to(x_shift.device)
            )

            if torch.isnan(amx):
                num = 100
            else:
                num = max(100, int(amx / 0.5))

            best_loss = None
            bounds = (1.0, max(-amn.item(), amx.item()))
            step = (bounds[1] - bounds[0]) / num

            best_min_range = -bounds[1]
            best_max_range = bounds[1]
            st = bounds[1]
            cnt = 0
            while st >= bounds[0]:
                min_range = torch.tensor(-st, dtype=x_shift.dtype).to(x_shift.device)
                max_range = torch.tensor(st, dtype=x_shift.dtype).to(x_shift.device)

                mx_scale = torch.where(
                    cmx > max_range,
                    cmx / max_range,
                    torch.tensor(1.0, dtype=x_shift.dtype).to(x_shift.device),
                )
                mn_scale = torch.where(
                    cmn < min_range,
                    cmn / min_range,
                    torch.tensor(1.0, dtype=x_shift.dtype).to(x_shift.device),
                )
                cur_scale = torch.max(mx_scale, mn_scale)

                for fc in layers:
                    if self.model.has_bias():
                        fc.bias.data += shift @ fc.weight.data.T

                    if fc.weight.data.dtype == torch.float8_e4m3fn:
                        fp8_scale = fc.weight_scale_inv.data
                        tmp_weight_data = weight_cast_to_bf16(fc.weight.data,
                                                              fp8_scale).to(torch.bfloat16)
                        tmp_fp8_scale = self.scaling_fp8_scale(fp8_scale,
                                                               cur_scale,
                                                               is_pre_layer=False)
                    else:
                        tmp_weight_data = fc.weight.data

                    tmp_weight_data.mul_(cur_scale.view(1, -1))
                    tmp_weight_data = self.wquantizer.fake_quant_weight_dynamic(
                        tmp_weight_data
                    )

                    if fc.weight.data.dtype == torch.float8_e4m3fn:
                        fc.weight.data = weight_cast_to_fp8(tmp_weight_data, tmp_fp8_scale)
                        fc.weight_scale_inv.data = tmp_fp8_scale
                    else:
                        fc.weight.data = tmp_weight_data

                x_shift_tmp = x_shift / cur_scale.view(1, -1)
                q_x = self.aquantizer.fake_quant_act_dynamic(x_shift_tmp)

                out = inspect_module(q_x, **kwargs)
                if isinstance(out, tuple):
                    out = out[0]

                loss = (org_out - out).pow(2).sum(-1).mean()

                if best_loss is None or best_loss > loss:
                    best_loss = loss
                    best_min_range = -st
                    best_max_range = st
                cnt += 1
                st -= step
                inspect_module.load_state_dict(org_sd)

            best_min_range = torch.tensor(best_min_range, dtype=x_shift.dtype).to(
                x_shift.device
            )
            best_max_range = torch.tensor(best_max_range, dtype=x_shift.dtype).to(
                x_shift.device
            )

            mn_scale = torch.where(
                cmn < best_min_range,
                cmn / best_min_range,
                torch.tensor(1.0, dtype=x_shift.dtype).to(x_shift.device),
            )
            mx_scale = torch.where(
                cmx > best_max_range,
                cmx / best_max_range,
                torch.tensor(1.0, dtype=x_shift.dtype).to(x_shift.device),
            )

            best_scale = torch.max(mx_scale, mn_scale)

            del org_out_dict
            gc.collect()
            torch.cuda.empty_cache()
            return best_scale, shift

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

        assert (
            len(prev_op) == 1
        ), 'Only support single prev_op. If multi prev_ops, code need to be updated.'

        layers = list(layers_dict.values())
        if (
            isinstance(
                prev_op[0], tuple(_LLMC_LINEAR_TYPES_ + _TRANSFORMERS_LINEAR_TYPES_)
            )
            and prev_op[0].out_features != layers[0].in_features * 3
            and prev_op[0].out_features != layers[0].in_features
        ):
            logger.info('Cannot apply scale. Do not transform this subset.')
            return

        if not self.filter_subset(prev_op):
            logger.info('Do not transform this subset.')
            return

        scale, shift = self.search_scale_shift_subset(
            layers, input_feat[input_name], inspect_module, subset_kwargs
        )
        self.apply_shift(shift, prev_op, layers)
        self.apply_scale(scale, prev_op, layers)
        if self.act_static:
            self.update_input_feat(scale, input_feat, layers_dict)
