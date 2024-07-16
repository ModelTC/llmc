from loguru import logger
import torch
import torch.nn as nn
import gc
import functools
import json
from collections import defaultdict
from functools import partial
from llmc.utils import copy_files
from ..blockwise_optimization import BlockwiseOpt
from .module_utils import _LLMC_LN_TYPES_, _TRANSFORMERS_LN_TYPES_
from .module_utils import _LLMC_LINEAR_TYPES_, _TRANSFORMERS_LINEAR_TYPES_
from .module_utils import (
    FakeQuantLinear,
    EffcientFakeQuantLinear,
    RealQuantLinear,
    OriginFloatLinear,
)
from .quant import Quantizer
from .hadamard_utils import apply_exact_had_to_linear
from .utils import get_wquantizer, get_aquantizer, check_do_quant, check_w_only


class BaseBlockwiseQuantization(BlockwiseOpt):
    def __init__(self, model, quant_config, input, config):
        super().__init__(model, quant_config, input, config)
        self.set_quant_config()

    def w_qdq(self, module, wquantizer):
        args = {"lowbound_factor": None, "upbound_factor": None}
        if hasattr(module, "buf_lowbound_factor"):
            args["lowbound_factor"] = module.buf_lowbound_factor
        if hasattr(module, "buf_upbound_factor"):
            args["upbound_factor"] = module.buf_upbound_factor

        return wquantizer.fake_quant_weight_dynamic(module.weight, args)

    def w_q(self, module):
        return self.wquantizer.real_quant_weight_dynamic(module.weight.data)

    def a_qdq(self, act, module, aquantizer):
        return aquantizer.fake_quant_act_dynamic(act)

    def logit(self, x):
        return torch.log(x / (1 - x))

    def set_quant_config(self):
        self.mix_bits = False
        self.mix_bits_map = [{} for _ in range(self.num_blocks)]
        self.quantizer_mix_bits = []
        if "quant_out" in self.quant_config and self.quant_config["quant_out"]:
            self.quant_out = True
        else:
            self.quant_out = False

        # set weight quant config
        self.wquantizer = Quantizer(**self.quant_config["weight"])

        # set act quant config
        if "act" in self.quant_config:
            self.w_only = False
            self.aquantizer = Quantizer(**self.quant_config["act"])
        else:
            self.w_only = True
            self.aquantizer = None

        logger.info(f"self.model.model_config : {self.model.model_config}")

        if "mix_bits" in self.quant_config:
            self.mix_bits = True
            mix_bits_settings = self.quant_config["mix_bits"]
            logger.info(f"mix_bits_settings number: {len(mix_bits_settings)}")
            logger.info(
                f"mix_bits_settings:\n{json.dumps(mix_bits_settings, ensure_ascii=False, indent=4)}"
            )
            for i in range(len(mix_bits_settings)):
                mix_bits_setting = mix_bits_settings[f"setting_{i}"]
                if mix_bits_setting["do_quant"]:
                    wquantizer_mix_bits = Quantizer(**mix_bits_setting["weight"])
                    if "act" in mix_bits_setting:
                        w_only_mix_bits = False
                        aquantizer_mix_bits = Quantizer(**mix_bits_setting["act"])
                    else:
                        w_only_mix_bits = True
                    self.quantizer_mix_bits.append(
                        {
                            "layer_name": mix_bits_setting["layer_name"],
                            "do_quant": mix_bits_setting["do_quant"],
                            "w_only_mix_bits": w_only_mix_bits,
                            "wquantizer": wquantizer_mix_bits,
                            "aquantizer": aquantizer_mix_bits
                            if not w_only_mix_bits
                            else None,
                        }
                    )
                else:
                    self.quantizer_mix_bits.append(
                        {
                            "layer_name": mix_bits_setting["layer_name"],
                            "do_quant": mix_bits_setting["do_quant"],
                        }
                    )
        for i in range(len(self.quantizer_mix_bits)):
            logger.info(f"quantizer_mix_bits {i} : {self.quantizer_mix_bits[i]}")
            layer_name = self.quantizer_mix_bits[i]["layer_name"]
            for name in layer_name:
                n_layeridx = name.split("#")
                assert (
                    len(n_layeridx) == 1 or len(n_layeridx) == 2
                ), "layer_name in mix_bits must be name#1-3-4 or name."
                if len(n_layeridx) == 2:
                    n = n_layeridx[0]
                    layeridx = n_layeridx[1].split("-")
                    layeridx = [int(idx) for idx in layeridx]
                else:
                    n = n_layeridx[0]
                    layeridx = "all"
                if layeridx == "all":
                    for k in range(self.num_blocks):
                        self.mix_bits_map[k][n] = i
                else:
                    for k in layeridx:
                        self.mix_bits_map[k][n] = i

        logger.info(
            f"self.mix_bits_map:\n{json.dumps(self.mix_bits_map, ensure_ascii=False, indent=4)}"
        )

        # set special quant config
        if (
            "special" in self.quant_config
            and "weight_clip" in self.quant_config["special"]
        ):
            self.weight_clip = self.quant_config["special"]["weight_clip"]
        else:
            self.weight_clip = True

        if (
            "special" in self.quant_config
            and "save_scale" in self.quant_config["special"]
        ):
            self.save_scale = self.quant_config["special"]["save_scale"]
            self.scale_path = self.quant_config["special"]["scale_path"]
            self.act_scales = {}
        else:
            self.save_scale = False

        if (
            "special" in self.quant_config
            and "save_clip" in self.quant_config["special"]
        ):
            self.save_clip = self.quant_config["special"]["save_clip"]
            self.clip_path = self.quant_config["special"]["clip_path"]
            self.weight_clips = {}
        else:
            self.save_clip = False

        if (
            "special" in self.quant_config
            and "clip_version" in self.quant_config["special"]
        ):
            self.clip_version = self.quant_config["special"]["clip_version"]
        else:
            self.clip_version = "v1"

        if (
            "special" in self.quant_config
            and "clip_sym" in self.quant_config["special"]
        ):
            self.clip_sym = self.quant_config["special"]["clip_sym"]
        else:
            self.clip_sym = self.wquantizer.sym

        if self.clip_version == "v2":
            assert self.wquantizer.calib_algo == "learnable"

    def block_forward(self, block, input_data=None):
        output = []

        if input_data is None:
            input_data = self.input["data"]

        for i in range(len(input_data)):
            input_data[i] = input_data[i].to(device=next(block.parameters()).device)
            if (
                "attention_mask" in self.input["kwargs"][i]
                and self.input["kwargs"][i]["attention_mask"] is not None
            ):
                self.input["kwargs"][i]["attention_mask"] = self.input["kwargs"][i][
                    "attention_mask"
                ].cuda()
            with torch.no_grad():
                out = block(input_data[i], **self.input["kwargs"][i])[0]
                output.append(out)
        return output

    def block_opt(self, block):
        block = block.cuda()
        named_linears = self.model.get_block_linears(block)
        logger.info(f"named_linears: {named_linears}")
        input_feat = defaultdict(list)
        handles = []
        self.block_init(block)

        for name in named_linears:
            handles.append(
                named_linears[name].register_forward_hook(
                    functools.partial(
                        self.cache_input_hook, name=name, feat_dict=input_feat
                    )
                )
            )

        if not self.quant_out:
            self.input["data"] = self.block_forward(block)
        else:
            self.block_forward(block)

        for h in handles:
            h.remove()
        torch.cuda.empty_cache()

        self.block_transform(block, input_feat, self.input["kwargs"])

        if self.quant_out:
            params_dict = {}
            params_dict["a_qdq"] = self.a_qdq if not self.w_only else None
            params_dict["w_qdq"] = self.w_qdq
            self.model.replace_module_block(
                FakeQuantLinear, block, self.block_idx, params_dict
            )
            self.input["data"] = self.block_forward(block)

        block = block.cpu()
        del input_feat
        gc.collect()
        torch.cuda.empty_cache()

    def block_transform(self, block, input_feat, block_kwargs):
        logger.info(f"Start transform the {self.block_idx}-th block")
        subsets = self.model.get_subsets_in_block(block)
        for index, subset in enumerate(subsets):
            if not self.filter_subset(subset):
                continue
            logger.info(f"subset: {subset}")
            prev_op = subset["prev_op"]
            layers_dict = subset["layers"]
            input_name = subset["input"][0]
            inspect_module = subset["inspect"]
            inspect_has_kwargs = subset["has_kwargs"]
            subset_kwargs = block_kwargs if inspect_has_kwargs else {}
            self.subset_transform(
                layers_dict,
                input_feat,
                prev_op,
                input_name,
                inspect_module,
                subset_kwargs,
            )
        logger.info(f"End transform the {self.block_idx}-th block")

    def block_init(self, block):
        pass

    @torch.no_grad()
    def filter_subset(self, subset):
        return True

    def collect_layers_weights(self, layers):
        weights = []
        for _m in layers:
            weights.append(_m.weight)
        return weights

    @torch.no_grad()
    def apply_scale(self, scales, prev_op, layers):
        assert (
            len(prev_op) == 1
        ), "Only support single prev_op. If multi prev_ops, code need to be updated."
        if isinstance(
            prev_op[0], tuple(_LLMC_LINEAR_TYPES_ + _TRANSFORMERS_LINEAR_TYPES_)
        ):
            assert len(layers) == 1
            logger.info("apply scale between fc and fc")
            self.scale_fc_fc(prev_op[0], layers[0], scales)
        elif isinstance(prev_op[0], tuple(_LLMC_LN_TYPES_ + _TRANSFORMERS_LN_TYPES_)):
            logger.info("apply scale between ln and fc")
            self.scale_ln_fcs(prev_op[0], layers, scales)
        else:
            raise NotImplementedError(f"prev_op {type(prev_op[0])} not supported yet!")

    @torch.no_grad()
    def apply_shift(self, shifts, prev_op, layers):
        if shifts is None:
            return

        assert (
            len(prev_op) == 1
        ), "Only support single prev_op. If multi prev_ops, code need to be updated."
        if isinstance(
            prev_op[0], tuple(_LLMC_LINEAR_TYPES_ + _TRANSFORMERS_LINEAR_TYPES_)
        ):
            assert len(layers) == 1
            self.shift_fc_fc(prev_op[0], layers[0], shifts)
        elif isinstance(prev_op[0], tuple(_LLMC_LN_TYPES_ + _TRANSFORMERS_LN_TYPES_)):
            self.shift_ln_fcs(prev_op[0], layers, shifts)
        else:
            raise NotImplementedError(f"prev_op {type(prev_op[0])} not supported yet!")

    @torch.no_grad()
    def scale_fc_fc(self, fc1, fc2, scales):
        scales = scales.to(fc1.weight.device)
        if fc1.out_features == fc2.in_features * 3:
            num_heads = self.model.get_model_config().to_dict().get("n_head", None)
            fc1.weight.t_()
            org_shape = fc1.weight.shape
            fc1.weight.data = fc1.weight.data.reshape(org_shape[0] * num_heads, 3, -1)
            value = fc1.weight.data[:, 2, :].reshape(org_shape[0], -1)
            fc1.weight.data[:, 2, :] = value.div(scales.view(-1)).reshape(
                fc1.weight[:, 2, :].shape
            )
            fc1.weight.data = fc1.weight.data.reshape(org_shape).t_()
            if hasattr(fc1, "bias") and fc1.bias is not None:
                fc1.bias.data = fc1.bias.data.reshape(num_heads, 3, -1)

                value = fc1.bias.data[:, 2, :].reshape(-1)

                fc1.bias.data[:, 2, :] = value.div(scales.view(-1)).reshape(
                    fc1.bias[:, 2, :].shape
                )
                fc1.bias.data = fc1.bias.data.reshape(-1)
        else:
            assert fc1.out_features == fc2.in_features

            if hasattr(fc1, "bias") and fc1.bias is not None:
                fc1.bias.div_(scales.view(-1))

            fc1.weight.div_(scales.view(-1, 1))

        fc2.weight.mul_(scales.view(1, -1))

    @torch.no_grad()
    def shift_fc_fc(self, fc1, fc2, shifts):
        if fc1.out_features == fc2.in_features * 3:
            num_heads = self.model.get_model_config().to_dict().get("n_head", None)
            if hasattr(fc1, "bias") and fc1.bias is not None:
                fc1.bias.data = fc1.bias.data.reshape(num_heads, 3, -1)

                value = fc1.bias.data[:, 2, :].reshape(-1)
                fc1.bias.data[:, 2, :] = (value - shifts).reshape(
                    fc1.bias[:, 2, :].shape
                )
                fc1.bias.data = fc1.bias.data.reshape(-1)
        else:
            assert fc1.out_features == fc2.in_features

            if hasattr(fc1, "bias") and fc1.bias is not None:
                fc1.bias.sub_(shifts)

        if hasattr(fc2, "bias") and fc2.bias is not None:
            fc2.bias.add_(fc2.weight @ shifts)
        else:
            if hasattr(self, "use_shift") and self.use_shift:
                del fc2.bias
                fc2.register_buffer("bias", fc2.weight @ shifts)

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
                if hasattr(self, "use_shift") and self.use_shift:
                    del fc.bias
                    fc.register_buffer("bias", fc.weight @ shifts)

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

        if self.model.has_bias():
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
                    f"This layer {n} in {self.block_idx}-th block is set to float. No need to clip this layer."
                )
                continue
            if isinstance(m, tuple(_LLMC_LINEAR_TYPES_ + _TRANSFORMERS_LINEAR_TYPES_)):
                m = m.cuda()
                if any([_ in n for _ in ["q_", "k_", "query", "key", "Wqkv"]]):
                    if self.clip_version == "v2":
                        m.register_buffer("buf_upbound_factor", None)
                        m.register_buffer("buf_lowbound_factor", None)
                    continue
                logger.info(f"clip layer: {n}")

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

                self.apply_clip(m, min_val, max_val, n)

    @torch.no_grad()
    def apply_clip(self, layer, min_val, max_val, layer_name):
        if self.clip_version == "v1":
            max_val = max_val.to(layer.weight.device)
            org_shape = layer.weight.shape
            layer.weight.data = layer.weight.data.reshape(*max_val.shape[:2], -1)
            if self.clip_sym:
                min_val = -max_val

            layer.weight.data = torch.clamp(layer.weight.data, min_val, max_val)
            layer.weight.data = layer.weight.data.reshape(org_shape)
        elif self.clip_version == "v2":
            up_factor, low_factor = self.get_clip_factor(
                layer, min_val, max_val, layer_name
            )
            layer.register_buffer("buf_upbound_factor", up_factor)
            layer.register_buffer("buf_lowbound_factor", low_factor)
            if self.save_clip:
                layer_name = (
                    f"{self.model.block_name_prefix}.{self.block_idx}.{layer_name}"
                )
                self.weight_clips[layer_name] = {"up_factor": None, "low_factor": None}
                self.weight_clips[layer_name]["up_factor"] = up_factor.cpu()
                if low_factor is not None:
                    self.weight_clips[layer_name]["low_factor"] = low_factor.cpu()
        else:
            raise Exception(f"Not support other clip version")

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
        if wquantizer.granularity == "per_group":
            group_size = wquantizer.group_size
        else:
            group_size = w.shape[1]

        w = w.reshape(w.shape[0], 1, -1, group_size)
        oc_batch_size = 256 if w.shape[0] % 256 == 0 else 64  # prevent OOM
        assert w.shape[0] % oc_batch_size == 0

        w_all = w
        best_max_val_all = []
        best_min_val_all = []

        for i_b in range(w.shape[0] // oc_batch_size):
            w = w_all[i_b * oc_batch_size : (i_b + 1) * oc_batch_size]

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
                    if self.clip_version == "v2" and not check_w_only(
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
                    x = x.reshape(1, x.shape[0], -1, group_size)
                    x = x[:, 0 :: x.shape[1] // n_sample_token]
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

                    if self.clip_version == "v1":
                        cur_w = torch.clamp(w, min_val, max_val)
                        q_w = wquantizer.fake_quant_weight_dynamic(cur_w)
                    elif self.clip_version == "v2":
                        low_factor = self.logit((min_val / org_min_val))
                        up_factor = self.logit((max_val / org_max_val))
                        tensor_range = wquantizer.get_learnable_range(
                            w, low_factor, up_factor
                        )

                        scales, zeros, max_int, min_int = wquantizer.get_qparams(
                            tensor_range, w.device
                        )
                        args = {}
                        args["scales"] = scales
                        args["zeros"] = zeros
                        args["max_int"] = max_int
                        args["min_int"] = min_int
                        q_w = wquantizer.fake_quant_weight_static(w, args)
                    else:
                        raise Exception(f"Not support other clip version")

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

                    if self.clip_version == "v1":
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
            layer.weight.data = torch.matmul(W, Q).to(device="cpu", dtype=dtype)

    def rotate_post_layers(self, post_layers, Q, exact_had=False):
        for layer in post_layers:
            dtype = layer.weight.dtype
            device = layer.weight.data.device
            W = layer.weight.data.to(device=device, dtype=torch.float64)
            layer.weight.data = torch.matmul(Q.T, W).to(device="cpu", dtype=dtype)

            if exact_had and self.online_rote:
                apply_exact_had_to_linear(layer, had_dim=-1, output=False)

            if hasattr(layer, "bias") and layer.bias is not None:
                b = layer.bias.data.to(device=device, dtype=torch.float64)
                layer.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)

    def rotate_embeddings(self, Q):
        embeddings = self.model.get_embed_layers()
        assert len(embeddings) == 1
        for layer in embeddings:
            dtype = layer.weight.data.dtype
            W = layer.weight.data.to(device=self.dev, dtype=torch.float64)
            layer.weight.data = torch.matmul(W, Q).to(device="cpu", dtype=dtype)

    def rotate_head(self, Q):
        heads = self.model.get_head_layers()
        for layer in heads:
            dtype = layer.weight.data.dtype
            W = layer.weight.data.to(device=self.dev, dtype=torch.float64)
            layer.weight.data = torch.matmul(W, Q).to(device="cpu", dtype=dtype)

    def fuse_ln_fcs(self, ln, fcs):
        for fc in fcs:
            fc_dtype = fc.weight.dtype
            W = fc.weight.data.double()
            fc.weight.data = (W * ln.weight.double()).to(fc_dtype)
            if hasattr(ln, "bias") and ln.bias is not None:
                if fc.bias is None:
                    fc.bias = torch.nn.Parameter(
                        torch.zeros(fc.out_features, dtype=torch.float64)
                    )
                fc.bias.data = fc.bias.data.double() + torch.matmul(W, ln.bias.double())
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
        if hasattr(fc, "bias") and fc.bias is not None:
            b_ = fc.bias.data.double()
            fc.bias.data = b_ - b_.mean()
            fc.bias.data = fc.bias.data.to(fc_dtype)

    @torch.no_grad()
    def deploy(self, quant_format):
        logger.info(f"-- deploy_{quant_format}_model start --")
        logger.info(f"quant_config : {self.quant_config}")
        params_dict = {}
        if quant_format == "fake_quant":
            module = EffcientFakeQuantLinear
            if not self.mix_bits:
                params_dict["mix_bits"] = False
                params_dict["a_qdq"] = (
                    partial(self.a_qdq, aquantizer=self.aquantizer)
                    if not self.w_only
                    else None
                )
                params_dict["w_qdq"] = partial(self.w_qdq, wquantizer=self.wquantizer)
            else:
                params_dict["mix_bits"] = True
                params_dict["a_qdq"] = self.a_qdq
                params_dict["w_qdq"] = self.w_qdq
                params_dict["mix_bits_map"] = self.mix_bits_map
                params_dict["quantizer_mix_bits"] = self.quantizer_mix_bits
                params_dict["wquantizer_default"] = self.wquantizer
                params_dict["aquantizer_default"] = self.aquantizer
                params_dict["w_only_default"] = self.w_only
        elif quant_format == "real_quant":
            module = RealQuantLinear
            params_dict["w_q"] = self.w_q
            params_dict["quant_config"] = self.quant_config
        elif quant_format == "origin_float":
            module = OriginFloatLinear
            params_dict = {}
        else:
            raise NotImplementedError
        self.model.replace_module_all(module, params_dict)
        logger.info(f"-- deploy_{quant_format}_model done --")

    @torch.no_grad()
    def copy_tokenizer(self, path):
        for substring in self.config.save.get("tokenizer_file_substring", ["token"]):
            copy_files(self.config.model.path, path, substring)
        logger.info(f"copy tokenizer done --")

    @torch.no_grad()
    def save_model(self, path):
        if self.config.model.type == "Llava":
            self.model.llava_model.language_model = self.model.get_model()
            self.model.llava_model.save_pretrained(path)
            logger.info(f"save model done --")
            self.copy_tokenizer(path)
            copy_files(self.config.model.path, path, "preprocessor_config")
        else:
            self.model.get_model().save_pretrained(path)
            logger.info(f"save model done --")
            self.copy_tokenizer(path)

    def cleanup_memory(self, verbos=True):
        """Run GC and clear GPU memory."""
        import gc
        import inspect

        caller_name = ""
        try:
            caller_name = f" (from {inspect.stack()[1].function})"
        except (ValueError, KeyError):
            pass

        def total_reserved_mem() -> int:
            return sum(
                torch.cuda.memory_reserved(device=i)
                for i in range(torch.cuda.device_count())
            )

        memory_before = total_reserved_mem()

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            memory_after = total_reserved_mem()
            if verbos:
                logger.info(
                    f"GPU memory{caller_name}: {memory_before / (1024 ** 3):.2f} -> {memory_after / (1024 ** 3):.2f} GB"
                    f" ({(memory_after - memory_before) / (1024 ** 3):.2f} GB)"
                )
