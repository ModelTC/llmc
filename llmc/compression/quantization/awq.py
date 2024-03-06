import torch
import torch.nn as nn
from loguru import logger
import gc
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers.models.mistral.modeling_mistral import MistralRMSNorm
from .base_blockwise_quantization import BaseBlockwiseQuantization
from llmc.utils.registry_factory import ALGO_REGISTRY
from .module_utils import FakeQuantLinear


@ALGO_REGISTRY
class Awq(BaseBlockwiseQuantization):
    def __init__(self, model, quant_config, input, config):
        super().__init__(model, quant_config, input, config)
        if "special" in self.quant_config and "version" in self.quant_config["special"]:
            self.version = self.quant_config["special"]["version"]
        else:
            self.version = "v2"
        if "special" in self.quant_config and "weight_clip" in self.quant_config["special"]:
            self.weight_clip = self.quant_config["special"]["weight_clip"]
        else:
            self.weight_clip = True

    @torch.no_grad()
    def get_weight_scale(self, layers):
        weights = self.collect_layers_weights(layers)
        weights = torch.cat(weights, dim=0)
        org_shape = weights.shape
        weights = self.wquantizer.reshape_tensor(weights)
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
    def search_scale_subset(self, layers, input, inspect_module, subset_kwargs):
        w_max = self.get_weight_scale(layers)
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
                if self.version == "v1":
                    scales = (
                        (x_max.pow(ratio) / w_max.pow(1 - ratio))
                        .clamp(min=1e-4)
                        .view(-1)
                    )
                elif self.version == "v2":
                    scales = x_max.pow(ratio).clamp(min=1e-4).view(-1)
                scales = scales / (scales.max() * scales.min()).sqrt()
                for fc in layers:
                    fc.weight.mul_(scales.view(1, -1))

                    fc.weight.data = self.wquantizer.fake_quant_weight_dynamic(
                        fc.weight.data
                    )

                x_tmp = x / scales.view(1, -1)
                if not self.w_only:
                    x_tmp = self.aquantizer.fake_quant_act_dynamic(x_tmp)

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
    def auto_clip_layer(
        self,
        w,
        input,
        n_grid=20,
        max_shrink=0.5,
        n_sample_token=512,
    ):
        assert w.dim() == 2

        if self.wquantizer.granularity == "per_group":
            group_size = self.wquantizer.group_size
        else:
            group_size = w.shape[1]

        w = w.reshape(w.shape[0], 1, -1, group_size)
        oc_batch_size = 256 if w.shape[0] % 256 == 0 else 64  # prevent OOM
        assert w.shape[0] % oc_batch_size == 0

        w_all = w
        best_max_val_all = []

        for i_b in range(w.shape[0] // oc_batch_size):
            w = w_all[i_b * oc_batch_size : (i_b + 1) * oc_batch_size]
            org_max_val = w.abs().amax(dim=-1, keepdim=True)  # co, 1, n_group, 1

            best_max_val = org_max_val.clone()
            min_errs = torch.ones_like(org_max_val) * 1e9
            org_out_dict = {}
            for i_s in range(int(max_shrink * n_grid)):
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
                        org_out = (x * w).sum(dim=-1)  # co, n_token, n_group
                        org_out_dict[i] = org_out
                    max_val = org_max_val * (1 - i_s / n_grid)
                    min_val = -max_val
                    cur_w = torch.clamp(w, min_val, max_val)

                    q_w = self.wquantizer.fake_quant_weight_dynamic(cur_w)

                    cur_out = (x * q_w).sum(dim=-1)

                    # co, 1, n_group, 1
                    err = (cur_out - org_out).pow(2).mean(dim=1).view(min_errs.shape)
                    err_mean += err
                    del cur_w
                    del cur_out
                err_mean /= len(input)
                cur_best_idx = err_mean < min_errs
                min_errs[cur_best_idx] = err_mean[cur_best_idx]
                best_max_val[cur_best_idx] = max_val[cur_best_idx]
            best_max_val_all.append(best_max_val)
        best_max_val = torch.cat(best_max_val_all, dim=0)

        del org_out
        del org_out_dict
        gc.collect()
        torch.cuda.empty_cache()
        return best_max_val.squeeze(1)

    @torch.no_grad()
    def block_transform(self, block, input_feat, idx, block_kwargs):
        super().block_transform(block, input_feat, idx, block_kwargs)
        if self.weight_clip:
            logger.info(f"auto_clip start")
            self.auto_clip(block, input_feat, n_sample_token=self.config.calib.seq_len)
            logger.info(f"auto_clip finished")
        else:
            logger.info(f"disable weight clip")

    @torch.no_grad()
    def auto_clip(self, block, input_feat, n_sample_token):
        # auto clip
        named_linears = self.model.get_block_linears(block)
        for name in named_linears:
            if any([_ in name for _ in ["q_", "k_", "query", "key", "Wqkv"]]):
                continue
            logger.info(f"clip layer: {name}")
            named_linears[name].cuda()
            max_val = self.auto_clip_layer(
                named_linears[name].weight,
                input_feat[name],
                n_sample_token=n_sample_token,
            )
            self.apply_clip(named_linears[name], max_val)

    def apply_clip(self, layer, max_val):
        max_val = max_val.to(layer.weight.device)
        org_shape = layer.weight.shape
        layer.weight.data = layer.weight.data.reshape(*max_val.shape[:2], -1)
        layer.weight.data = torch.clamp(layer.weight.data, -max_val, max_val)
        layer.weight.data = layer.weight.data.reshape(org_shape)

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

        if self.config["model"]["type"] == 'Starcoder':
            if isinstance(prev_op[0], (nn.Linear, FakeQuantLinear)):
                logger.info("Do not transform this subset.")
                return

        assert (
            len(prev_op) == 1
        ), "Only support single prev_op. If multi prev_ops, code need to be updated."
        if isinstance(
            prev_op[0],
            (nn.Linear, nn.LayerNorm, LlamaRMSNorm, FakeQuantLinear, MistralRMSNorm),
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
                layers, input_feat[input_name], inspect_module, subset_kwargs
            )
            self.apply_scale(scale, prev_op, layers)
            self.update_input_feat(scale, input_feat, layers_dict)
        else:
            logger.info("Do not transform this subset.")
