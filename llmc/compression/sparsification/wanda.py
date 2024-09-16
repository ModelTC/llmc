import torch
import torch.nn as nn
from loguru import logger

from llmc.utils.registry_factory import ALGO_REGISTRY

from .base_blockwise_sparsification import BaseBlockwiseSparsification


@ALGO_REGISTRY
class Wanda(BaseBlockwiseSparsification):
    def __init__(self, model, sparsity_config, input, config):
        super().__init__(model, sparsity_config, input, config)

    @torch.no_grad()
    def get_row_scale(self, layer, act, scaler_row):
        if isinstance(act, list):
            act = torch.cat(act, dim=0)
        if len(act.shape) == 2:
            act = act.unsqueeze(0)
        if isinstance(layer, nn.Linear):
            if len(act.shape) == 3:
                act = act.reshape((-1, act.shape[-1]))
            act = act.t()

        act = act.type(torch.float32).to(scaler_row.device)
        scaler_row += torch.norm(act, p=2, dim=1) ** 2
        return scaler_row

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
        for layer in layers:
            columns = layer.weight.data.shape[1]
            scaler_row = torch.zeros((columns), device=layer.weight.device)
            nsamples = 0
            for batch_idx in range(len(input_feat[input_name])):
                scaler_row = self.get_row_scale(
                    layer, input_feat[input_name][batch_idx], scaler_row
                )
                nsamples += input_feat[input_name][batch_idx].shape[0]
            scaler_row /= nsamples
            W_metric = torch.abs(layer.weight.data) * torch.sqrt(
                scaler_row.reshape((1, -1))
            )
            W_mask = (
                torch.zeros_like(W_metric) == 1
            )  # initialize a mask to be all False

            if self.sparser.prunen != 0:
                # semi-structured n:m sparsity
                for input_idx in range(W_metric.shape[1]):
                    if input_idx % self.sparser.prunem == 0:
                        tmp = W_metric[
                            :, input_idx: (input_idx + self.sparser.prunem)
                        ].float()
                        W_mask.scatter_(
                            1,
                            input_idx
                            + torch.topk(
                                tmp, self.sparser.prunen, dim=1, largest=False
                            )[1],
                            True,
                        )
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)
                indices = sort_res[1][
                    :, : int(W_metric.shape[1] * self.sparser.sparsity)
                ]
                W_mask.scatter_(1, indices, True)
            layer.weight.data[W_mask] = 0  # set weights to zero
