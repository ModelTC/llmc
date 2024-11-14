import torch
from loguru import logger

from llmc.utils.registry_factory import ALGO_REGISTRY

from .base_blockwise_quantization import BaseBlockwiseQuantization


@ALGO_REGISTRY
class RTN(BaseBlockwiseQuantization):
    def __init__(self, model, quant_config, input, padding_mask, config, modality='language'):
        super().__init__(model, quant_config, input, padding_mask, config, modality)

    # @torch.no_grad()
    # def block_opt(self, *opt_kwargs):
    #     if self.act_static:
    #         super().block_opt(*opt_kwargs)

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
        pass
