import gc

import torch
import torch.nn as nn
from loguru import logger

from llmc.utils.registry_factory import ALGO_REGISTRY

from .base_blockwise_quantization import BaseBlockwiseQuantization
from .module_utils import _LLMC_LN_TYPES_, _TRANSFORMERS_LN_TYPES_
from .quant import IntegerQuantizer


@ALGO_REGISTRY
class DGQ(BaseBlockwiseQuantization):
    def __init__(self, model, quant_config, input, padding_mask, config):
        super().__init__(model, quant_config, input, padding_mask, config)
        self.model_dtype = next(self.model.model.parameters()).dtype

    def w_qdq(self, module, wquantizer):
        scales = module.buf_scales
        zeros = module.buf_zeros
        scale8 = module.buf_scale8
        s = (scales * scale8.reshape(-1, 1)).reshape(-1, 1)
        int_max = torch.round(127 / scales)
        upper = torch.clamp(zeros + int_max, max=15.0).reshape(-1, 1)
        lower = torch.clamp(zeros - int_max, min=0.0).reshape(-1, 1)
        args = {}
        args['scales'] = s.reshape(-1, 1)
        args['zeros'] = zeros.reshape(-1, 1)
        args['qmax'] = upper
        args['qmin'] = lower
        # logger.info(f"s.shape : {s.shape}")
        # logger.info(f"scales.shape : {scales.shape}")
        # logger.info(f"zeros.shape : {zeros.shape}")
        # logger.info(f"upper.shape : {upper.shape}")
        # logger.info(f"lower.shape : {lower.shape}")
        return self.wquantizer_w4.fake_quant_weight_static(module.weight.data, args)

    def set_quant_config(self):
        logger.info(f'self.quant_config : {self.quant_config}')
        if 'quant_out' in self.quant_config and self.quant_config['quant_out']:
            self.quant_out = True
        else:
            self.quant_out = False
        self.quant_type = self.quant_config.get('quant_type', 'int_quant')
        assert self.quant_type != 'float_quant', 'DGQ do not support Float quant now.'
        # set weight quant config
        self.wquantizer_w4 = IntegerQuantizer(**self.quant_config['weight']['w_1'])
        perchannel_setting = {
            'bit': self.quant_config['weight']['w_1']['bit'],
            'symmetric': self.quant_config['weight']['w_1']['symmetric'],
            'granularity': 'per_channel',
        }
        self.wquantizer_w4_perchannel = IntegerQuantizer(**perchannel_setting)
        self.wquantizer_w8 = IntegerQuantizer(**self.quant_config['weight']['w_2'])

        # set act quant config
        if 'act' in self.quant_config and self.quant_config['act'] is not None:
            self.w_only = False
            self.aquantizer = IntegerQuantizer(**self.quant_config['act'])
        else:
            self.w_only = True

    @torch.no_grad()
    def get_weight_scale(self, layers):
        weights = self.collect_layers_weights(layers)
        scale = torch.cat(
            [fc.abs().max(dim=0, keepdim=True)[0] for fc in weights], dim=0
        )
        scale = scale.max(dim=0)[0].clamp(min=1e-5)
        del weights
        gc.collect()
        torch.cuda.empty_cache()
        return scale

    @torch.no_grad()
    def get_act_scale(self, tensors):
        scale_max = None
        for x in tensors:
            x = x.cuda()
            x = x.abs().view(-1, x.shape[-1])
            comming_max = torch.max(x, dim=0)[0].float()
            if scale_max is not None:
                scale_max = torch.max(scale_max, comming_max)
            else:
                scale_max = comming_max
            x = x.cpu()
        return scale_max

    @torch.no_grad()
    def search_scale_subset(self, layers, tensors):
        w_max = self.get_weight_scale(layers)
        x_max = self.get_act_scale(tensors)
        x_max = x_max.to(dtype=w_max.dtype, device=w_max.device)
        scale = (x_max.pow(0.5) / w_max.pow(0.5)).clamp(min=1e-5)
        return scale

    @torch.no_grad()
    def smoothquant_transform(self, prev_op, layers, tensors):
        scale = self.search_scale_subset(layers, tensors)
        self.apply_scale(scale, prev_op, layers)

    @torch.no_grad()
    def smooth_llama_mlp(self, upp, downp, act_scales):
        device, dtype = downp.weight.device, downp.weight.dtype

        # downp_scales = downp.weight.abs().max(dim=0)[0].cuda().float().clamp(min=1e-5)

        maxsv, inds = act_scales.sort()
        basl = int(len(act_scales) * 0.005 + 1.5)  # hyperparameter
        baseline = maxsv[-basl]
        if baseline < 1e-4:
            return
        scales = act_scales / baseline
        scales[act_scales <= baseline] = 1.0
        # downp_m = downp_scales[inds[-basl:]]
        # downp_redu = 50 * downp_scales.max() / downp_m
        scales[inds[-basl:]] = scales[inds[-basl:]]
        # print(scales.max())

        act_scales /= scales
        scales = scales.to(device=device, dtype=dtype)
        logger.info(f'scales.device : {scales.device}')
        # gatep.weight.div_(scales)
        upp.weight.data.div_(scales.view(-1, 1))

        if hasattr(upp, 'bias') and upp.bias is not None:
            upp.bias.div_(scales)
        downp.weight.data.mul_(scales.view(1, -1))

    @torch.no_grad()
    def search_scale_zero_layer(self, layer, input_feat):
        w4_group_size = self.wquantizer_w4.group_size
        weight_tmp = layer.weight.data.clone()
        org_w_shape = weight_tmp.shape
        device = weight_tmp.device
        dtype = weight_tmp.dtype
        w_out_channels, w_in_channels = weight_tmp.shape
        input_feat = input_feat.to(device)
        input_feat = input_feat.squeeze()
        assert w_in_channels % w4_group_size == 0
        best_scales = torch.ones(
            [w_out_channels, w_in_channels // w4_group_size],
            dtype=self.model_dtype,
            device=device,
        )
        best_zeros = torch.ones(
            [w_out_channels, w_in_channels // w4_group_size],
            dtype=self.model_dtype,
            device=device,
        )
        for group_index in range(w_in_channels // w4_group_size):
            inp_LxG = input_feat[
                :, group_index * w4_group_size: (group_index + 1) * w4_group_size
            ]
            weight_OxG = weight_tmp[
                :, group_index * w4_group_size: (group_index + 1) * w4_group_size
            ]
            """
            For each pair of (inp_LxG weight_OxG),
            we can all consider it as per channel quantization.
            Let's consider weight as
            the transpose matrix of the weight in PyTorch's linear layer.

            output = input x weight

            input => [L * in]
            weight => [in * out]

            Split each input channel according to groups.
            input => (in/G) * [L * G]
            weight => (in/G) * [G * out]

            [L * G] x [G * out] is per channel quantization.
            The scale shape is [out * 1].
            input x weight is per group quantization.
            The scale shape is [out * (in/G)].
            """
            org_out_LxO = inp_LxG @ (weight_OxG.t())
            grid = 20
            best_loss = torch.full(
                [weight_OxG.shape[0]], float('inf'), device=device, dtype=dtype
            )
            w_max = weight_OxG.amax(dim=-1, keepdim=True)
            w_min = weight_OxG.amin(dim=-1, keepdim=True)
            for i in range(grid):
                ratio = 1.02 - (i + 1) / grid * 0.22
                weight_OxG = weight_OxG.clamp(w_min * ratio, w_max * ratio)
                (
                    _,
                    scales,
                    zeros,
                    qmax,
                    qmin,
                ) = self.wquantizer_w4_perchannel.get_tensor_qparams(weight_OxG)
                # Perchannel do not need reshape and restore tensor.
                weight_OxG_fq = self.wquantizer_w4_perchannel.quant_dequant(
                    weight_OxG, scales, zeros, qmax, qmin
                )
                if not self.w_only:
                    inp_LxG_fq = self.a_qdq(inp_LxG)
                else:
                    inp_LxG_fq = inp_LxG
                out_LxO = inp_LxG_fq @ (weight_OxG_fq.t())
                loss = (org_out_LxO - out_LxO).squeeze().pow(2).mean(dim=0).view(-1)

                best_idx = best_loss > loss
                best_loss[best_idx] = loss[best_idx]
                best_scales[:, group_index][best_idx] = scales.view(-1)[best_idx]
                best_zeros[:, group_index][best_idx] = zeros.view(-1)[best_idx]

        grid = 80
        org_out = input_feat @ weight_tmp.t()
        best_loss = torch.full(
            [w_out_channels], float('inf'), device=device, dtype=dtype
        )
        best_scale8 = torch.zeros(
            (w_out_channels,), dtype=self.model_dtype, device=device
        )
        for i in range(grid):
            ratio = 1.02 - (i + 1) / grid * 0.82
            w_max = weight_tmp.abs().amax(dim=-1, keepdim=True)
            (
                _,
                qscales_8,
                zeros,
                qmax,
                qmin,
            ) = self.wquantizer_w8.get_tensor_qparams(
                weight_tmp.clamp(-w_max * ratio, w_max * ratio)
            )
            qscale = torch.round(best_scales / qscales_8).clamp(min=1.0)
            int_max = torch.round(127 / qscales_8)
            upper = torch.clamp(best_zeros + int_max, max=15.0).reshape(-1, 1)
            lower = torch.clamp(best_zeros - int_max, min=0.0).reshape(-1, 1)
            qscale_q = (qscale * qscales_8).reshape(-1, 1)

            weight_tmp_fq = self.wquantizer_w4.reshape_tensor(weight_tmp)
            weight_tmp_fq = self.wquantizer_w4.quant_dequant(
                weight_tmp_fq, qscale_q, best_zeros.reshape(-1, 1), upper, lower
            )
            weight_tmp_fq = self.wquantizer_w4.restore_tensor(
                weight_tmp_fq, org_w_shape
            )

            if not self.w_only:
                input_feat_fq = self.a_qdq(input_feat)
            else:
                input_feat_fq = input_feat

            out = input_feat_fq @ (weight_tmp_fq.t())
            loss = (org_out - out).abs().pow(2).mean(dim=0).view(-1)
            best_idx = (best_loss > loss).view(-1)
            best_loss[best_idx] = loss[best_idx]
            best_scale8[best_idx] = qscales_8[best_idx].view(-1)

        best_scales = torch.round(best_scales / best_scale8.view(-1, 1)).clamp(min=1.0)
        return best_scales, best_zeros, best_scale8

    @torch.no_grad()
    def search_scale_zero_subset(self, layers_dict, input_feat):
        logger.info(f'layers_dict : {layers_dict}')
        for layer_name in layers_dict:
            logger.info(f'search for : {layer_name}')
            best_scales, best_zeros, best_scale8 = self.search_scale_zero_layer(
                layers_dict[layer_name], input_feat
            )
            # logger.info(f"best_scales : {best_scales}, {best_scales.shape}")
            # logger.info(f"best_zeros : {best_zeros}, {best_zeros.shape}")
            # logger.info(f"best_scale8 : {best_scale8}, {best_scale8.shape}")
            layers_dict[layer_name].register_buffer('buf_scales', best_scales)
            layers_dict[layer_name].register_buffer('buf_zeros', best_zeros)
            layers_dict[layer_name].register_buffer('buf_scale8', best_scale8)

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

        layers = list(layers_dict.values())
        if isinstance(prev_op[0], tuple(_LLMC_LN_TYPES_ + _TRANSFORMERS_LN_TYPES_)):
            self.smoothquant_transform(prev_op, layers, input_feat[input_name])
        # For llama model down proj
        if 'mlp.down_proj' in layers_dict:
            scale = self.search_scale_subset(layers, input_feat[input_name])
            self.smooth_llama_mlp(prev_op[0], layers[0], scale)
        self.search_scale_zero_subset(layers_dict, input_feat[input_name][0])
