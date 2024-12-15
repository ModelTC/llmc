import gc

import torch
import torch.nn as nn
from loguru import logger

from llmc.utils.registry_factory import ALGO_REGISTRY

from .base_blockwise_quantization import BaseBlockwiseQuantization


@ALGO_REGISTRY
class HQQ(BaseBlockwiseQuantization):
    def __init__(self, model, quant_config, input, padding_mask, config):
        super().__init__(model, quant_config, input, padding_mask, config)
        self.add_quant_config()

    @torch.no_grad()
    def add_quant_config(self):
        self.lp_norm = self.quant_config['special']['lp_norm']
        self.beta = self.quant_config['special']['beta']
        self.kappa = self.quant_config['special']['kappa']
        self.iters = self.quant_config['special']['iters']
        self.axis = self.quant_config['special']['axis']
        if self.lp_norm == 1:
            self.shrink_op = lambda x, beta: torch.sign(x) * torch.nn.functional.relu(
                torch.abs(x) - 1.0 / self.beta
            )
        else:
            self.shrink_op = lambda x, beta, p=self.lp_norm: torch.sign(
                x
            ) * torch.nn.functional.relu(
                torch.abs(x) - (1.0 / self.beta) * torch.pow(torch.abs(x), p - 1)
            )

    @torch.no_grad()
    def optimize_weights_proximal(self, W_f, scales, zeros, qmax, qmin):
        best_error = 1e4
        current_beta = self.beta
        current_kappa = self.kappa
        scales = 1 / scales
        for i in range(self.iters):
            W_q = torch.round(W_f * scales + zeros).clamp(qmin, qmax)
            W_r = (W_q - zeros) / scales
            W_e = self.shrink_op(W_f - W_r, current_beta)

            zeros = torch.mean(W_q - (W_f - W_e) * scales, axis=-1, keepdim=True)
            current_beta *= current_kappa
            current_error = float(torch.abs(W_f - W_r).mean())

            logger.info(f'iter : {i}, error : {current_error}')

            if current_error < best_error:
                best_error = current_error
            else:
                break

        torch.cuda.empty_cache()
        scales = 1 / scales

        return scales, zeros

    @torch.no_grad()
    def block_opt(self, block):
        block = block.cuda()
        named_linears = self.model.get_block_linears(block)
        logger.info(f'named_linears: {named_linears}')

        for name in named_linears:
            logger.info(f'Optimize weights proximal of {name}')
            layer = named_linears[name]

            tensor = layer.weight.data.float()
            if self.axis == 0:
                tensor = tensor.T
            (
                tensor,
                org_scales,
                org_zeros,
                qmax,
                qmin,
            ) = self.wquantizer.get_tensor_qparams(tensor)

            best_scales, best_zeros = self.optimize_weights_proximal(
                tensor, org_scales, org_zeros, qmax, qmin
            )
            layer.register_buffer('buf_scales', best_scales)
            layer.register_buffer('buf_zeros', best_zeros)
            layer.register_buffer('buf_qmax', torch.tensor(qmax))
            layer.register_buffer('buf_qmin', torch.tensor(qmin))

        block = block.cpu()
        gc.collect()
        torch.cuda.empty_cache()

    def w_qdq(self, module, wquantizer):
        args = {}
        if self.axis == 0:
            args['dim'] = 'ic'
        args['scales'] = module.buf_scales
        args['zeros'] = module.buf_zeros
        args['qmax'] = module.buf_qmax
        args['qmin'] = module.buf_qmin

        return wquantizer.fake_quant_weight_static(module.weight, args)
