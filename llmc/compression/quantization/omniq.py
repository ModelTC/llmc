import torch
import torch.nn as nn
import functools
import gc
import pdb
import math
import copy
import random
import numpy as np

from math import inf
from loguru import logger
from tqdm import tqdm
from contextlib import nullcontext

from .base_blockwise_quantization import BaseBlockwiseQuantization
from .module_utils import (
    FakeQuantLinear,
    LlmcLayerNorm,
    LlmcLlamaRMSNorm,
    LlmcMistralRMSNorm,
)
from .train_utils import NativeScalerWithGradNormCount, TruncateFunction, LossFunction
from llmc.utils.registry_factory import ALGO_REGISTRY


@ALGO_REGISTRY
class OmniQuant(BaseBlockwiseQuantization):
    def __init__(self, model, quant_config, input, config):
        super().__init__(model, quant_config, input, config)
        self.add_quant_config()

        if self.config["model"]["type"] not in ["Llama", "Opt", "Falcon", "Mistral"] and self.let:
            raise ValueError("Only support for opt/llama/Llama-2/falcon/Mistral now")
        elif self.config["model"]["type"] in ("Llama", "Mistral"):
            self.attention_mask = self.input["kwargs"][0]["attention_mask"]
            self.position_ids = self.input["kwargs"][0]["position_ids"]
        else:
            self.attention_mask = self.input["kwargs"][0]["attention_mask"]
            self.position_ids = None

        if self.deactive_amp:
            self.batch_mask = self.attention_mask.repeat(
                self.input["data"][0].shape[0], 1, 1, 1
            ).cuda()
        else:
            self.batch_mask = (
                self.attention_mask.repeat(self.input["data"][i].shape[0], 1, 1, 1)
                .float()
                .cuda()
            )
        self.dev = torch.device("cuda")
        self.model_dtype = next(self.model.model.parameters()).dtype

    def add_quant_config(self):
        self.prefix = self.model.block_name_prefix
        self.loss_func = LossFunction(method="mse")
        self.deactive_amp = self.quant_config["special"]["deactive_amp"]
        self.wd = self.quant_config["special"]["wd"]

        if self.deactive_amp:
            self.dtype = torch.float
            self.traincast = nullcontext
        else:
            self.dtype = self.model_dtype
            self.traincast = torch.cuda.amp.autocast

        self.epochs = self.quant_config["special"]["epochs"]
        self.aug_loss = self.quant_config["special"]["aug_loss"]

        self.lwc = self.quant_config["special"]["lwc"]
        if self.lwc:
            self.lwc_lr = self.quant_config["special"]["lwc_lr"]

        self.let = self.quant_config["special"]["let"]
        if self.let:
            if self.config["model"]["type"] == "Falcon":
                raise ValueError("Falcon not yet support let")

            self.let_lr = self.quant_config["special"]["let_lr"]
            self.use_shift = self.quant_config["special"]["use_shift"]

            if self.use_shift and not self.model.has_bias():
                raise ValueError("Don't support no bias model use shift")

            self.alpha = self.quant_config["special"]["alpha"]
            self.act_scales = self.get_act_scale_shift(stat="scales")
            if self.use_shift:
                self.act_shifts = self.get_act_scale_shift(stat="shifts")
        else:
            self.use_shift = False

        if self.epochs > 0:
            assert self.lwc or self.let

    def block_forward(self, block, input_data=None):
        output = []

        if input_data is None:
            input_data = self.input["data"]

        for i in range(len(input_data)):
            input_data[i] = input_data[i].to(device=next(block.parameters()).device)
            if "attention_mask" in self.input["kwargs"][i]:
                self.input["kwargs"][i]["attention_mask"] = self.input["kwargs"][i][
                    "attention_mask"
                ].cuda()
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    out = block(input_data[i], **self.input["kwargs"][i])[0]
                    output.append(out)
        return output

    def get_original_out(self, block, idx):
        if idx == 0:
            self.ori_out = self.block_forward(block)
            if self.aug_loss:
                self.ori_out2 = self.ori_out
        else:
            self.ori_out = self.block_forward(block, self.ori_out)
            if self.aug_loss:
                self.ori_out2 = self.block_forward(block)

    def block_transform(self, block, input_feat, idx, block_kwargs):
        logger.info(f"Start transform the {idx+1}-th block")

        with torch.no_grad():
            block.float()

        for i in range(len(self.input["data"])):
            self.input["data"][i] = self.input["data"][i].to(self.dtype)

        self.get_original_out(block, idx)
        self.register_omni_parameters(block, idx)
        self.omni_train(block, idx)

        if self.let:
            subsets = self.model.get_subsets_in_block(block)
            for index, subset in enumerate(subsets):
                prev_op = subset["prev_op"]
                layers_dict = subset["layers"]
                self.subset_transform(block, layers_dict, prev_op)

        self.clear_tmp(block)

        logger.info(f"End transform the {idx+1}-th block")

    def omni_train(self, block, idx):
        if self.let and self.lwc:
            optimizer = torch.optim.AdamW(
                [
                    {"params": self.get_lwc_parameters(block), "lr": self.lwc_lr},
                    {"params": self.get_let_parameters(block), "lr": self.let_lr},
                ],
                weight_decay=self.wd,
            )
        elif self.lwc:
            optimizer = torch.optim.AdamW(
                [{"params": self.get_lwc_parameters(block), "lr": self.lwc_lr}],
                weight_decay=self.wd,
            )
        elif self.let:
            optimizer = torch.optim.AdamW(
                [{"params": self.get_let_parameters(block), "lr": self.let_lr}],
                weight_decay=self.wd,
            )

        loss_scaler = NativeScalerWithGradNormCount()

        for epochs in range(self.epochs):
            loss_list = []
            norm_list = []

            for i in range(len(self.input["data"])):
                with self.traincast():
                    if self.let:
                        self.smooth_weight_tmp(block)

                    if self.position_ids is not None:
                        quant_out = block(
                            self.input["data"][i],
                            attention_mask=self.batch_mask,
                            position_ids=self.position_ids,
                        )[0]
                    else:
                        quant_out = block(
                            self.input["data"][i], attention_mask=self.batch_mask
                        )[0]

                    loss = self.loss_func(self.ori_out[i], quant_out)
                    if self.aug_loss:
                        loss += self.loss_func(self.ori_out2[i], quant_out)

                if not math.isfinite(loss.item()):
                    logger.info("Loss is NAN, stopping training")
                    pdb.set_trace()

                loss_list.append(loss.data)
                optimizer.zero_grad()
                norm = loss_scaler(
                    loss, optimizer, parameters=self.get_omni_parameters(block)
                )
                norm_list.append(norm.data)

            loss_mean = torch.stack(loss_list).mean()
            norm_mean = torch.stack(norm_list).mean()
            logger.info(f"block {idx} iter {epochs} loss:{loss_mean} norm:{norm_mean} ")

        del optimizer

    def subset_transform(self, block, layers_dict, prev_op):
        layers = list(layers_dict.values())

        if (
            isinstance(prev_op[0], (nn.Linear, FakeQuantLinear))
            and prev_op[0].out_features != layers[0].in_features
        ):
            logger.info("Cannot apply scale. Do not transform this subset.")
            return

        scale, shift = self.search_scale_shift_subset(block, layers_dict)

        if len(scale):
            if len(shift) and shift[0] is not None:
                self.apply_shift(shift[0], prev_op, layers)
            scale = scale[0]
            scale.data = self.truncate(scale)
            self.apply_scale(scale, prev_op, layers)
        else:
            self.smooth_q_k_inplace(block)

    def search_scale_shift_subset(self, block, layers_dict):
        scale = []
        shift = []
        for name, module in block.named_parameters():
            if name.endswith("scale"):
                for n in layers_dict:
                    for key in self.model.pairs.keys():
                        if key in n and self.model.pairs[key] in name:
                            scale.append(module)
            if name.endswith("shift"):
                for n in layers_dict:
                    for key in self.model.pairs.keys():
                        if key in n and self.model.pairs[key] in name:
                            shift.append(module)
        return scale, shift

    def register_omni_parameters(self, block, idx):
        params_dict = {}
        module = FakeQuantLinear
        params_dict["a_qdq"] = self.a_qdq if not self.w_only else None
        params_dict["w_qdq"] = self.w_qdq
        self.model.replace_module_block(module, block, idx, params_dict)
        if self.lwc:
            self.register_lwc_parameters(block)
        if self.let:
            self.register_let_parameters(block, idx)

    def register_lwc_parameters(self, block, init_value=4.0):
        for n, m in block.named_modules():
            if isinstance(m, FakeQuantLinear):
                if self.wquantizer.granularity == "per_group":
                    dim = int(
                        m.weight.data.shape[0]
                        * math.ceil(m.weight.data.shape[1] / self.wquantizer.group_size)
                    )
                else:
                    dim = m.weight.data.shape[0]
                up_param = nn.Parameter(
                    torch.ones(
                        (dim, 1),
                        device=self.dev,
                        dtype=self.dtype,
                    )
                    * init_value
                )
                low_param = nn.Parameter(
                    torch.ones(
                        (dim, 1),
                        device=self.dev,
                        dtype=self.dtype,
                    )
                    * init_value
                )
                m.register_parameter("buf_upbound_factor", up_param)
                m.register_parameter("buf_lowbound_factor", low_param)
                m.dynamic_quant_weight = True

    def register_let_parameters(self, block, idx):
        block.register_parameter(
            "qkt_smooth_scale",
            nn.Parameter(
                torch.ones(
                    block.self_attn.q_proj.out_features,
                    device=self.dev,
                    dtype=self.dtype,
                )
            ),
        )
        self.replace_layer_norms(block, idx)

        for n, m in block.named_modules():
            if isinstance(m, FakeQuantLinear):
                for key in self.model.pairs.keys():
                    if key in n:
                        scale, shift = self.get_weight_scale_shift(m, idx, n)
                        if shift is not None:
                            block.register_parameter(
                                f"{self.model.pairs[key]}_smooth_shift",
                                nn.Parameter(shift),
                            )
                        else:
                            block.register_buffer(
                                f"{self.model.pairs[key]}_smooth_shift", None
                            )
                        block.register_parameter(
                            f"{self.model.pairs[key]}_smooth_scale", nn.Parameter(scale)
                        )

                m.dynamic_quant_weight = False
                m.dynamic_quant_tmp_weight = True

    def replace_layer_norms(self, block, idx):
        if self.config["model"]["type"] == "Mistral":
            self.model.replace_module_block(LlmcMistralRMSNorm, block, idx, {})
        elif self.config["model"]["type"] == "Llama":
            self.model.replace_module_block(LlmcLlamaRMSNorm, block, idx, {})
        else:
            self.model.replace_module_block(LlmcLayerNorm, block, idx, {})

    def get_layer_norms(self, block):
        layer_norms = []
        for n, m in block.named_modules():
            if isinstance(m, (LlmcLayerNorm, LlmcLlamaRMSNorm, LlmcMistralRMSNorm)):
                layer_norms.append(m)
        return layer_norms

    def get_lwc_parameters(self, block):
        params = []
        for n, m in block.named_parameters():
            if n.find("bound_factor") > -1:
                params.append(m)
        return iter(params)

    def get_let_parameters(self, block):
        params = []
        template = "smooth" if self.use_shift else "smooth_scale"
        for n, m in block.named_parameters():
            if n.find(template) > -1:
                params.append(m)
        return iter(params)

    def get_omni_parameters(self, block):
        params = []
        template = "smooth" if self.use_shift else "smooth_scale"
        for n, m in block.named_parameters():
            if n.find("bound_factor") > -1 or n.find(template) > -1:
                params.append(m)
        return iter(params)

    def get_act_scale_shift(self, stat="scales"):
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
            if stat == "scales":
                get_tensor_scale(name, x)
            elif stat == "shifts":
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

    def get_weight_scale_shift(self, layer, layer_idx, name):
        act = (
            self.act_scales[f"{self.prefix}.{layer_idx}.{name}"]
            .to(
                device=self.dev,
                dtype=self.dtype,
            )
            .clamp(min=1e-5)
        )

        weight = layer.weight.data.max(dim=0)[0].clamp(min=1e-5)
        scale = (act.pow(self.alpha) / weight.half().pow(1 - self.alpha)).clamp(
            min=1e-5
        )

        if self.use_shift:
            shift = self.act_shifts[f"{self.prefix}.{layer_idx}.{name}"].to(
                device=self.dev,
                dtype=self.dtype,
            )
        else:
            shift = None

        return scale, shift

    def truncate(self, num, thre=1e-2):
        return TruncateFunction.apply(num, thre)

    def clear_tmp(self, block):
        for n, m in block.named_modules():
            if isinstance(m, FakeQuantLinear):
                del m.tmp_weight
                del m.tmp_bias
                m.dynamic_quant_weight = False
                m.dynamic_quant_tmp_weight = False
                if self.lwc:
                    m.buf_upbound_factor.requires_grad = False
                    m.buf_lowbound_factor.requires_grad = False

    def smooth_weight_tmp(self, block):
        subsets = self.model.get_subsets_in_block(block)
        with torch.no_grad():
            for n, m in block.named_parameters():
                if "smooth_scale" in n:
                    m.data = self.truncate(m)

        layer_norms = self.get_layer_norms(block)

        qkv_layers = [subsets[0]["layers"][name] for name in subsets[0]["layers"]]

        self.smooth_ln_fcs_tmp(
            layer_norms[0],
            qkv_layers,
            block.qkv_smooth_scale,
            block.qkv_smooth_shift,
        )
        self.smooth_ln_fcs_tmp(
            layer_norms[1],
            [subsets[2]["layers"][name] for name in subsets[2]["layers"]],
            block.fc1_smooth_scale,
            block.fc1_smooth_shift,
        )
        self.smooth_fc_fc_tmp(
            subsets[1]["prev_op"][0],
            subsets[1]["inspect"],
            block.out_smooth_scale,
            block.out_smooth_shift,
        )
        self.smooth_q_k_tmp(qkv_layers[0], qkv_layers[1], block.qkt_smooth_scale)
        subsets[3]["inspect"].tmp_weight = subsets[3]["inspect"].weight

        for name, module in block.named_modules():
            if isinstance(module, FakeQuantLinear):
                if not hasattr(module, "tmp_bias"):
                    module.tmp_bias = module.bias

    def smooth_ln_fcs_tmp(self, ln, fcs, scales, shifts):
        ln.use_tmp_parameter = True
        if not isinstance(fcs, list):
            fcs = [fcs]

        if shifts is not None:
            if hasattr(ln, "bias") and ln.bias is not None:
                ln.tmp_bias = (ln.bias - shifts) / scales
            else:
                ln.tmp_bias = (-1 * shifts) / scales

        ln.tmp_weight = ln.weight / scales

        for fc in fcs:
            if shifts is not None:
                if hasattr(fc, "bias") and fc.bias is not None:
                    fc.tmp_bias = fc.bias + fc.weight @ shifts
                else:
                    fc.tmp_bias = fc.weight @ shifts
            fc.tmp_weight = fc.weight * scales.view(1, -1)

    def smooth_fc_fc_tmp(self, fc1, fc2, scales, shifts):
        if fc1.out_features != fc2.in_features:
            fc1.tmp_weight = fc1.weight
            fc2.tmp_weight = fc2.weight
            return

        if hasattr(fc1, "tmp_weight"):
            if hasattr(fc1, "tmp_bias") and fc1.tmp_bias is not None:
                if shifts is not None:
                    fc1.tmp_bias = fc1.tmp_bias - shifts
                    fc1.tmp_bias = fc1.tmp_bias / scales.view(-1)
            fc1.tmp_weight = fc1.tmp_weight / scales.view(-1, 1)
        else:
            if hasattr(fc1, "bias") and fc1.bias is not None:
                fc1.tmp_bias = fc1.bias / scales.view(-1)
            fc1.tmp_weight = fc1.weight / scales.view(-1, 1)

        if shifts is not None:
            if hasattr(fc2, "bias") and fc2.bias is not None:
                fc2.tmp_bias = fc2.bias + fc2.weight @ shifts
            else:
                fc2.tmp_bias = fc2.weight @ shifts

        fc2.tmp_weight = fc2.weight * scales.view(1, -1)

    def smooth_q_k_tmp(self, q_proj, k_proj, scales):
        if q_proj.tmp_weight.shape != k_proj.tmp_weight.shape:
            return

        q_proj.tmp_weight = q_proj.tmp_weight / scales.view(-1, 1)
        if hasattr(q_proj, "tmp_bias") and q_proj.tmp_bias is not None:
            q_proj.tmp_bias = q_proj.tmp_bias / scales.view(-1)
        k_proj.tmp_weight = k_proj.tmp_weight * scales.view(-1, 1)
        if hasattr(k_proj, "tmp_bias") and k_proj.tmp_bias is not None:
            k_proj.tmp_bias = k_proj.tmp_bias * scales.view(-1)

    def smooth_q_k_inplace(self, block):
        if block.self_attn.q_proj.weight.shape != block.self_attn.k_proj.weight.shape:
            return

        scales = block.qkt_smooth_scale
        scales.data = self.truncate(scales)
        block.self_attn.q_proj.weight.div_(scales.view(-1, 1))
        if block.self_attn.q_proj.bias is not None:
            block.self_attn.q_proj.bias.div_(scales.view(-1))
        block.self_attn.k_proj.weight.mul_(scales.view(-1, 1))
        if block.self_attn.k_proj.bias is not None:
            block.self_attn.k_proj.bias.mul_(scales.view(-1))
        for name, module in block.named_modules():
            if isinstance(
                module, (LlmcLayerNorm, LlmcLlamaRMSNorm, LlmcMistralRMSNorm)
            ):
                module.use_tmp_parameter = False

    def w_qdq(self, module):
        args = {}
        if hasattr(module, "buf_upbound_factor"):
            args["lowbound_factor"] = module.buf_lowbound_factor
            args["upbound_factor"] = module.buf_upbound_factor

        if module.dynamic_quant_weight:
            return self.wquantizer.fake_quant_weight_dynamic(module.weight, args)

        elif module.dynamic_quant_tmp_weight:
            return self.wquantizer.fake_quant_weight_dynamic(module.tmp_weight, args)
        else:
            return self.wquantizer.fake_quant_weight_dynamic(module.weight, args)

    def deploy(self, quant_format):
        super().deploy(quant_format)
        self.model.convert_dtype(self.model_dtype)

    def save_model(self, path):
        self.model.convert_dtype(self.model_dtype)
        super().save_model(path)

    def block_cvt(self, block, idx):
        for n, m in block.named_modules():
            if isinstance(m, FakeQuantLinear):
                tensor = self.wquantizer.reshape_tensor(m.weight.data)
                shape = m.weight.data.shape
                min_val, max_val = self.wquantizer.get_learnable_range(
                    tensor,
                    m.buf_lowbound_factor,
                    m.buf_upbound_factor,
                )
                if self.wquantizer.sym:
                    max_val = torch.max(max_val.abs(), min_val.abs())
                    min_val = -max_val

                clip_range_min = min_val
                clip_range_max = max_val

                tensor = tensor.clamp(
                    clip_range_min.to(m.weight.data.device),
                    clip_range_max.to(m.weight.data.device),
                )
                m.weight.data = self.wquantizer.restore_tensor(tensor, shape)

                del m.buf_upbound_factor
                del m.buf_lowbound_factor