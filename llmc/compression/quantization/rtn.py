import torch
from loguru import logger

from llmc.utils.registry_factory import ALGO_REGISTRY

from .base_blockwise_quantization import BaseBlockwiseQuantization


@ALGO_REGISTRY
class RTN(BaseBlockwiseQuantization):
    def __init__(self, model, quant_config, input, padding_mask, config):
        super().__init__(model, quant_config, input, padding_mask, config)

    @torch.no_grad()
    def block_opt(self, block, *opt_kwargs):
        if self.quant_kvcache:
            self.register_kv_cache(block)
        if self.act_static:
            super().block_opt(block, *opt_kwargs)

    @torch.no_grad()
    def subset_transform(
        self,
        subset,
        input_feat,
        subset_kwargs,
    ):
        pass
