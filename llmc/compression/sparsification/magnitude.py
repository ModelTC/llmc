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
            if self.sparser.prunen != 0:
                W_mask = (torch.zeros_like(W)==1)
                for input_idx in range(W_metric.shape[1]):
                    if input_idx % self.sparser.prunem == 0:
                        tmp = W_metric[:, input_idx:(input_idx+self.sparser.prunem)].float()
                        W_mask.scatter_(1, input_idx+torch.topk(tmp, self.sparser.prunen, dim=1, largest=False)[1], True)
            else:
                thresh = torch.sort(W_metric.flatten().cuda())[0][
                    int(W.numel() * self.sparser.sparsity)
                ].cpu()
                W_mask = W_metric <= thresh
        W[W_mask] = 0
