from loguru import logger
import torch
import torch.nn as nn
import gc
import functools
from collections import defaultdict
from ..blockwise_optimization import BlockwiseOpt
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers.models.mistral.modeling_mistral import MistralRMSNorm
from .module_utils import (
    FakeQuantLinear,
    EffcientFakeQuantLinear,
    RealQuantLinear,
    OriginFloatLinear,
    LlmcLayerNorm,
    LlmcLlamaRMSNorm,
    LlmcMistralRMSNorm,
)
from .quant import Quantizer


class BaseBlockwiseQuantization(BlockwiseOpt):
    def __init__(self, model, quant_config, input, config):
        super().__init__(model, quant_config, input, config)
        self.set_quant_config()

    def w_qdq(self, module):
        return self.wquantizer.fake_quant_weight_dynamic(module.weight.data)

    def w_qdq_naive(self, module):
        return self.wquantizer.fake_quant_weight_dynamic(module.weight.data)

    def w_q(self, module):
        return self.wquantizer.fake_quant_weight_dynamic(module.weight.data)

    def a_qdq(self, act, module=None):
        return self.aquantizer.fake_quant_act_dynamic(act)

    def set_quant_config(self):
        if "quant_out" in self.quant_config and self.quant_config["quant_out"]:
            self.quant_out = True
        else:
            self.quant_out = False

        # set weight quant config
        self.wquantizer = Quantizer(**self.quant_config["weight"])

        # set act quant config
        if "act" in self.quant_config and self.quant_config["act"] is not None:
            self.w_only = False
            self.aquantizer = Quantizer(**self.quant_config["act"])
        else:
            self.w_only = True

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
                out = block(input_data[i], **self.input["kwargs"][i])[0]
                output.append(out)
        return output

    def block_opt(self, block, idx):
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

        self.block_transform(block, input_feat, idx, self.input["kwargs"])

        if self.quant_out:
            params_dict = {}
            params_dict["a_qdq"] = self.a_qdq if not self.w_only else None
            params_dict["w_qdq"] = self.w_qdq
            self.model.replace_module_block(FakeQuantLinear, block, idx, params_dict)
            self.input["data"] = self.block_forward(block)

        block = block.cpu()
        del input_feat
        gc.collect()
        torch.cuda.empty_cache()

    def block_transform(self, block, input_feat, idx, block_kwargs):
        logger.info(f"Start transform the {idx+1}-th block")
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
        logger.info(f"End transform the {idx+1}-th block")

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
        if isinstance(prev_op[0], (nn.Linear, FakeQuantLinear)):
            assert len(layers) == 1
            self.scale_fc_fc(prev_op[0], layers[0], scales)
        elif isinstance(
            prev_op[0],
            (
                nn.LayerNorm,
                LlamaRMSNorm,
                LlmcLayerNorm,
                LlmcLlamaRMSNorm,
                MistralRMSNorm,
                LlmcMistralRMSNorm,
            ),
        ):
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
        if isinstance(prev_op[0], (nn.Linear, FakeQuantLinear)):
            assert len(layers) == 1
            self.shift_fc_fc(prev_op[0], layers[0], shifts)
        elif isinstance(
            prev_op[0],
            (
                nn.LayerNorm,
                LlamaRMSNorm,
                LlmcLayerNorm,
                LlmcLlamaRMSNorm,
                MistralRMSNorm,
                LlmcMistralRMSNorm,
            ),
        ):
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
    def deploy(self, quant_format):
        logger.info(f"-- deploy_{quant_format}_model start --")
        logger.info(f"quant_config : {self.quant_config}")

        params_dict = {}

        if quant_format == "fake_quant":
            module = EffcientFakeQuantLinear
            params_dict["a_qdq"] = self.a_qdq if not self.w_only else None
            if self.config is not None:
                if "cvt" in self.config and self.config.get("cvt", True):
                    params_dict["w_qdq"] = self.w_qdq_naive
                else:
                    params_dict["w_qdq"] = self.w_qdq
            else:
                params_dict["w_qdq"] = self.w_qdq
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
    def save_model(self, path):
        self.model.get_model().save_pretrained(path)