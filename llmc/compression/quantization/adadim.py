import torch
from loguru import logger

from llmc.utils.registry_factory import ALGO_REGISTRY

from .base_blockwise_quantization import BaseBlockwiseQuantization
from .module_utils import FakeQuantLinear


@ALGO_REGISTRY
class AdaDim(BaseBlockwiseQuantization):
    def __init__(self, model, quant_config, input, config, modality='language'):
        super().__init__(model, quant_config, input, config, modality)

    def get_layer_out(self, x, layer):
        with torch.no_grad():
            org_out = layer(x)
            if isinstance(org_out, tuple):
                org_out = org_out[0]
        return org_out

    def search_dim_subset(self, layers_dict, input):
        for name in layers_dict:
            layer = layers_dict[name]

            loss_dict = {}
            for dim in ['oc', 'ic']:
                loss_mean = 0

                weight = layer.weight.data.clone()

                q_weight = self.wquantizer.fake_quant_weight_dynamic(
                    weight, {'dim': dim}
                )

                for i in range(len(input)):
                    input[i] = input[i].to(layer.weight.data.device)
                    x = input[i]

                    layer.weight.data = weight
                    org_out = self.get_layer_out(x, layer)

                    layer.weight.data = q_weight
                    out = self.get_layer_out(x, layer)

                    loss = (org_out - out).float().pow(2).mean().item()
                    loss_mean += x.shape[0] * 1.0 / self.n_samples * loss

                loss_dict[dim] = loss_mean
                layer.weight.data = weight

            if loss_dict['ic'] < loss_dict['oc']:
                layer.register_buffer('buf_qdim', torch.tensor(0))
                logger.info(f'Suggest layer {name} use per-input channel quant')
            else:
                layer.register_buffer('buf_qdim', torch.tensor(1))
                logger.info(f'Suggest layer {name} use per-output channel quant')

    def block_transform(self, block, input_feat, block_kwargs):
        logger.info(f'Start transform the {self.block_idx}-th block')
        subsets = self.model.get_subsets_in_block(block)
        for index, subset in enumerate(subsets):
            logger.info(f'subset: {subset}')
            layers_dict = subset['layers']
            input_name = subset['input'][0]

            self.search_dim_subset(layers_dict, input_feat[input_name])

            self.model.replace_module_subset(
                FakeQuantLinear,
                block,
                subset,
                self.block_idx,
                self.get_replacement_params(
                    mode='fake_quant', w_only=self.w_only, name=None
                ),
            )

        logger.info(f'End transform the {self.block_idx}-th block')

    def w_qdq(self, module, wquantizer):
        weight = module.weight
        args = {}
        args['dim'] = 'ic' if module.buf_qdim == 0 else 'oc'

        weight = self.wquantizer.fake_quant_weight_dynamic(weight, args)

        return weight
