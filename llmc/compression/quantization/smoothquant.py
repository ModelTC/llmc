import torch
import gc
import torch.nn as nn
from .module_utils import _LLMC_LN_TYPES_, _TRANSFORMERS_LN_TYPES_
from .base_blockwise_quantization import BaseBlockwiseQuantization
from llmc.utils.registry_factory import ALGO_REGISTRY


@ALGO_REGISTRY
class SmoothQuant(BaseBlockwiseQuantization):
    def __init__(self, model, quant_config, input, config):
        super().__init__(model, quant_config, input, config)

    @torch.no_grad()
    def filter_subset(self, subset):
        prev_op = subset["prev_op"]
        if isinstance(prev_op[0], tuple(_LLMC_LN_TYPES_ + _TRANSFORMERS_LN_TYPES_)):
            return True
        else:
            return False

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
    def subset_transform(
        self,
        layers_dict,
        input_feat,
        prev_op,
        input_name,
        inspect_module,
        subset_kwargs,
    ):
        layers = list(layers_dict.values())
        scale = self.search_scale_subset(layers, input_feat[input_name])
        self.apply_scale(scale, prev_op, layers)
