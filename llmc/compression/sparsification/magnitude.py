import torch
from loguru import logger

from llmc.utils.registry_factory import ALGO_REGISTRY

from .base_blockwise_sparsification import BaseBlockwiseSparsification


@ALGO_REGISTRY
class Magnitude(BaseBlockwiseSparsification):
    def __init__(self, model, sparsity_config, input, config):
        super().__init__(model, sparsity_config, input, config)

    @torch.no_grad()
    def subset_transform(
        self,
        layers_dict,
        input_feat,
        prev_op,
        input_name,
        inspect_module,
        subset_kwargs
    ):
        layers = list(layers_dict.values())
        for layer in layers:
            W = layer.weight.data
            W_metric = torch.abs(W)
            thresh = torch.sort(W_metric.flatten().cuda())[0][
                int(W.numel() * self.sparser.sparsity)
            ].cpu()
            W_mask = W_metric <= thresh
        W[W_mask] = 0
