import gc
import os

import torch
import torch.distributed as dist
from loguru import logger

from .module_utils import _LLMC_LINEAR_TYPES_, _TRANSFORMERS_LINEAR_TYPES_
from .utils import check_do_quant, check_w_only, get_aquantizer, get_wquantizer


class AutoClipper:
    def __init__(
        self,
        w_only,
        mix_bits_map,
        quantizer_mix_bits,
        wquantizer,
        aquantizer,
        clip_version,
        clip_sym,
        save_clip,
        padding_mask,
    ):
        self.mix_bits_map = mix_bits_map
        self.quantizer_mix_bits = quantizer_mix_bits
        self.wquantizer = wquantizer
        self.aquantizer = aquantizer
        self.clip_version = clip_version
        self.clip_sym = clip_sym
        self.save_clip = save_clip
        self.padding_mask = padding_mask
        self.weight_clips = {}
        self.w_only = w_only
        self.logit = lambda x: torch.log(x / (1 - x))

    @torch.no_grad()
    def run(self, block, block_idx, input_feat, n_sample_token):
        for n, m in block.named_modules():
            if not check_do_quant(
                block_idx, n, self.mix_bits_map, self.quantizer_mix_bits
            ):
                logger.info(
                    f'This layer {n} in {block_idx}-th block is set to float.'
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
                inputs = (
                    [torch.cat(input_feat[n])]
                    if len(input_feat[n]) != 1
                    else input_feat[n]
                )
                max_val, min_val = self.auto_clip_layer(
                    block_idx, n, m.weight, inputs, n_sample_token=n_sample_token
                )

                dist.all_reduce(max_val, op=dist.ReduceOp.SUM)
                max_val /= int(os.environ['WORLD_SIZE'])

                dist.all_reduce(min_val, op=dist.ReduceOp.SUM)
                min_val /= int(os.environ['WORLD_SIZE'])

                self.apply_clip(block_idx, m, min_val, max_val, n)

    @torch.no_grad()
    def auto_clip_layer(
        self,
        block_idx,
        layer_name,
        w,
        inputs,
        n_grid=20,
        max_shrink=0.5,
        n_sample_token=512,
        eps=0.0,
    ):

        assert w.dim() == 2

        wquantizer = get_wquantizer(
            block_idx,
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
        best_max_val_all, best_min_val_all = [], []
        for i_b in range(w.shape[0] // oc_batch_size):
            w = w_all[i_b * oc_batch_size:(i_b + 1) * oc_batch_size]

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
                        block_idx,
                        layer_name,
                        self.mix_bits_map,
                        self.quantizer_mix_bits,
                        self.w_only,
                    ):
                        i_s += eps
                err_mean = 0
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].to(w.device)
                    x = inputs[i]
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
                    x = x[:, 0::x.shape[1] // n_sample_token]

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

                    q_w = self.fake_quantize_weight(
                        w, min_val, max_val, org_min_val, org_max_val
                    )
                    q_x = self.fake_quantize_input(block_idx, x, layer_name)

                    cur_out = (q_x * q_w).sum(dim=-1)

                    # co, 1, n_group, 1
                    err = (cur_out - org_out).pow(2).mean(dim=1).view(min_errs.shape)
                    err_mean += err

                    del cur_out

                err_mean /= len(inputs)
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

    @torch.no_grad()
    def apply_clip(self, block_idx, layer, min_val, max_val, layer_name):
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
                layer.weight.data = self.wquantizer.restore_tensor(
                    layer.weight.data, org_shape
                )
        elif self.clip_version == 'v2':
            up_factor, low_factor = self.get_clip_factor(
                block_idx, layer, min_val, max_val, layer_name
            )
            layer.register_buffer('buf_upbound_factor', up_factor)
            layer.register_buffer('buf_lowbound_factor', low_factor)
            if self.save_clip:
                if block_idx not in self.weight_clips:
                    self.weight_clips[block_idx] = dict()
                n = f'{layer_name}.weight_quantizer.'
                self.weight_clips[block_idx][n + 'upbound_factor'] = up_factor.cpu()
                if low_factor is not None:
                    self.weight_clips[block_idx][
                        n + 'lowbound_factor'
                    ] = low_factor.cpu()
                else:
                    self.weight_clips[block_idx][n + 'lowbound_factor'] = None
        else:
            raise Exception('Not support other clip version')

    def get_clip_factor(self, block_idx, layer, min_val, max_val, layer_name):
        wquantizer = get_wquantizer(
            block_idx,
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

    def fake_quantize_weight(self, w, min_val, max_val, org_min_val, org_max_val):
        if self.clip_version == 'v1':
            cur_w = torch.clamp(w, min_val, max_val)
            q_w = self.wquantizer.fake_quant_weight_dynamic(cur_w)
        elif self.clip_version == 'v2':
            low_factor = self.logit((min_val / org_min_val))
            up_factor = self.logit((max_val / org_max_val))
            tensor_range = self.wquantizer.get_learnable_range(w, low_factor, up_factor)

            scales, zeros, qmax, qmin = self.wquantizer.get_qparams(
                tensor_range, w.device
            )
            args = {'scales': scales, 'zeros': zeros, 'qmax': qmax, 'qmin': qmin}
            q_w = self.wquantizer.fake_quant_weight_static(w, args)
        else:
            raise Exception('Not support other clip version')
        return q_w

    def fake_quantize_input(self, block_idx, x, layer_name):
        if not check_w_only(
            block_idx,
            layer_name,
            self.mix_bits_map,
            self.quantizer_mix_bits,
            self.w_only,
        ):
            q_x = get_aquantizer(
                block_idx,
                layer_name,
                self.mix_bits_map,
                self.quantizer_mix_bits,
                self.aquantizer,
            ).fake_quant_act_dynamic(x)
        else:
            q_x = x
        return q_x
