import functools
import gc
from collections import defaultdict

import torch
from loguru import logger

from llmc.utils.registry_factory import ALGO_REGISTRY, TOKEN_REDUCTION_REGISTRY

from ..blockwise_optimization import BlockwiseOpt


@ALGO_REGISTRY
class TokenReduction(BlockwiseOpt):
    def __init__(self, model, sparsity_config, input, padding_mask, config):
        super().__init__(model, sparsity_config, input, padding_mask, config)
        self.register_reduction_modules()

    def register_reduction_modules(self):
        TOKEN_REDUCTION_REGISTRY[self.sparsity_config['special']['method']](
            self.sparsity_config, self.model, self.blocks
        )

    def block_opt(self, block):
        pass

    @torch.no_grad()
    def deploy(self, deploy_format):
        logger.info('-- deploy_token_reduction_model start --')
        logger.info(f'sparsity_config : {self.sparsity_config}')
        logger.info('-- deploy_token_reduction_model done --')
