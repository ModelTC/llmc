import torch
import torch.nn as nn
from loguru import logger
import gc
from .module_utils import _LLMC_LN_TYPES_, _TRANSFORMERS_LN_TYPES_
from .module_utils import _LLMC_LINEAR_TYPES_, _TRANSFORMERS_LINEAR_TYPES_
from .module_utils import FakeQuantLinear
from .base_blockwise_quantization import BaseBlockwiseQuantization
from .utils import get_wquantizer, get_aquantizer, check_do_quant, check_w_only
from llmc.utils.registry_factory import ALGO_REGISTRY


@ALGO_REGISTRY
class Awq(BaseBlockwiseQuantization):
    def __init__(self, model, quant_config, input, config):
        super().__init__(model, quant_config, input, config)
        if "special" in self.quant_config and "trans" in self.quant_config["special"]:
            self.trans = self.quant_config["special"]["trans"]
        else:
            self.trans = True

        if (
            "special" in self.quant_config
            and "trans_version" in self.quant_config["special"]
        ):
            self.trans_version = self.quant_config["special"]["trans_version"]
        else:
            self.trans_version = "v2"
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
        else:
            self.save_scale = False

    @torch.no_grad()
    def get_weight_scale(self, layers_dict):
        layers = list(layers_dict.values())
        weights = self.collect_layers_weights(layers)
        weights = torch.cat(weights, dim=0)
        org_shape = weights.shape
        wquantizer = get_wquantizer(
            self.block_idx,
            list(layers_dict.keys())[0],
            self.mix_bits_map,
            self.quantizer_mix_bits,
            self.wquantizer,
        )
        weights = wquantizer.reshape_tensor(weights)
        scale = weights.abs() / weights.abs().amax(dim=1, keepdim=True)
        scale = scale.view(org_shape)
        scale = scale.mean(0)
        del weights
        gc.collect()
        torch.cuda.empty_cache()
        return scale

    @torch.no_grad()
    def get_act_scale(self, x):
        return x.abs().view(-1, x.shape[-1]).mean(0)

    @torch.no_grad()
    def get_original_out(self, x, inspect_module, subset_kwargs):
        with torch.no_grad():
            org_out = inspect_module(x, **subset_kwargs)
            if isinstance(org_out, tuple):
                org_out = org_out[0]
        return org_out

    @torch.no_grad()
    def search_scale_subset(self, layers_dict, input, inspect_module, subset_kwargs):
        w_max = self.get_weight_scale(layers_dict)
        # grid search for ratio
        best_error = float("inf")
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
                x_max = self.get_act_scale(x)

                ratio = n * 1 / n_grid
                if self.trans_version == "v1":
                    scales = (
                        (x_max.pow(ratio) / w_max.pow(1 - ratio))
                        .clamp(min=1e-4)
                        .view(-1)
                    )
                elif self.trans_version == "v2":
                    scales = x_max.pow(ratio).clamp(min=1e-4).view(-1)
                scales = scales / (scales.max() * scales.min()).sqrt()
                for layer_name in layers_dict:
                    fc = layers_dict[layer_name]
                    fc.weight.mul_(scales.view(1, -1))

                    fc.weight.data = get_wquantizer(
                        self.block_idx,
                        layer_name,
                        self.mix_bits_map,
                        self.quantizer_mix_bits,
                        self.wquantizer,
                    ).fake_quant_weight_dynamic(fc.weight.data)

                x_tmp = x / scales.view(1, -1)
                if not check_w_only(
                    self.block_idx,
                    list(layers_dict.keys())[0],
                    self.mix_bits_map,
                    self.quantizer_mix_bits,
                    self.w_only,
                ):
                    x_tmp = get_aquantizer(
                        self.block_idx,
                        list(layers_dict.keys())[0],
                        self.mix_bits_map,
                        self.quantizer_mix_bits,
                        self.aquantizer,
                    ).fake_quant_act_dynamic(x_tmp)

                out = inspect_module(x_tmp, **kwargs)

                if isinstance(out, tuple):
                    out = out[0]

                loss = (org_out - out).float().pow(2).mean().item()
                loss_mean += x.shape[0] * 1.0 / self.n_samples * loss
                scales_mean += x.shape[0] * 1.0 / self.n_samples * scales
                inspect_module.load_state_dict(org_sd)
            is_best = loss_mean < best_error
            if is_best:
                best_error = loss_mean
                best_scales = scales_mean
        best_scales = best_scales.view(-1)
        del org_out_dict
        gc.collect()
        torch.cuda.empty_cache()
        return best_scales

    @torch.no_grad()
    def update_input_feat(self, scale, input_feat, layers_dict):
        for layer_name in layers_dict:
            for i in range(len(input_feat[layer_name])):
                inp = input_feat[layer_name][i]
                inp.div_(scale.view(1, -1).to(inp.device))

    @torch.no_grad()
    def block_transform(self, block, input_feat, block_kwargs):
        if self.trans:
            super().block_transform(block, input_feat, block_kwargs)

        if self.weight_clip:
            logger.info(f"auto_clip start")
            logger.info(f"clip version: {self.clip_version}")
            self.auto_clip(block, input_feat, n_sample_token=self.config.calib.seq_len)
            logger.info(f"auto_clip finished")
        else:
            logger.info(f"disable weight clip")

    @torch.no_grad()
    def subset_transform(
        self,
        layers_dict,
        input_feat,
        prev_op,
        input_name,
        inspect_module,
        subset_kwargs,
    ):
        if not check_do_quant(
            self.block_idx,
            list(layers_dict.keys())[0],
            self.mix_bits_map,
            self.quantizer_mix_bits,
        ):
            logger.info(
                "This subset is set to float. No need to transform this subset."
            )
            return
        if self.config["model"]["type"] == "Starcoder":
            if isinstance(prev_op[0], (nn.Linear, FakeQuantLinear)):
                logger.info("Do not transform this subset.")
                return

        assert (
            len(prev_op) == 1
        ), "Only support single prev_op. If multi prev_ops, code need to be updated."

        if isinstance(
            prev_op[0],
            tuple(
                _LLMC_LN_TYPES_
                + _TRANSFORMERS_LN_TYPES_
                + _LLMC_LINEAR_TYPES_
                + _TRANSFORMERS_LINEAR_TYPES_
            ),
        ):
            layers = list(layers_dict.values())

            if (
                isinstance(prev_op[0], (nn.Linear, FakeQuantLinear))
                and prev_op[0].out_features != layers[0].in_features * 3
                and prev_op[0].out_features != layers[0].in_features
            ):
                logger.info("Cannot apply scale. Do not transform this subset.")
                return

            scale = self.search_scale_subset(
                layers_dict, input_feat[input_name], inspect_module, subset_kwargs
            )

            self.apply_scale(scale, prev_op, layers)
            self.update_input_feat(scale, input_feat, layers_dict)

            if self.save_scale:
                for n in layers_dict:
                    layer_name = f"{self.model.block_name_prefix}.{self.block_idx}.{n}"
                    self.act_scales[layer_name] = scale
        else:
            logger.info("Do not transform this subset.")
