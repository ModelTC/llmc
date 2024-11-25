import copy
import functools
import math
from abc import ABCMeta, abstractmethod
from collections import defaultdict

import torch
import torch.nn as nn
import transformers
from loguru import logger

from llmc.utils.registry_factory import ALGO_REGISTRY

from .base_blockwise_quantization import BaseBlockwiseQuantization
from .module_utils import (_LLMC_LINEAR_TYPES_, _TRANSFORMERS_LINEAR_TYPES_,
                           FakeQuantLinear, RotateLinear)


@ALGO_REGISTRY
class GPTQ(BaseBlockwiseQuantization):
    def __init__(
        self, model, quant_config, input, padding_mask, config, modality='language'
    ):
        super().__init__(model, quant_config, input, padding_mask, config, modality)
        self.dev = torch.device('cuda')
        self.model_dtype = next(self.model.model.parameters()).dtype
        self.add_quant_config()
        self.layers_cache = {}
        self.collect_model_qparams()

    @torch.no_grad()
    def add_quant_config(self):
        self.prefix = self.model.block_name_prefix
        special_config = self.quant_config['special']

        self.true_sequential = special_config['true_sequential']
        self.static_groups = special_config['static_groups']
        self.actorder = special_config['actorder']
        self.percdamp = special_config['percdamp']
        self.blocksize = special_config['blocksize']

        self.owq = special_config.get('owq', False)

        if self.owq:
            self.n_outs = special_config['n_outs']
            self.static_groups = False
            self.actorder = False

        self.need_perm = (
            self.wquantizer.granularity == 'per_group'
            and not self.static_groups
            and self.actorder
        ) or self.owq

    def hessian_sorting(self, name):
        H = self.layers_cache[name]['H']

        if not self.owq:
            if self.actorder:
                self.perm = torch.argsort(torch.diag(H), descending=True)
            return

        temp_mask = torch.full([self.columns], True, device=self.dev)
        H_diag = torch.diag(H)
        descending_ids = torch.argsort(H_diag, descending=True)
        temp_mask[descending_ids[: self.n_out]] = False

        if self.actorder:
            perm = torch.cat(
                [descending_ids[self.n_out:], descending_ids[:self.self.n_out]]
            )
        else:
            perm = torch.cat(
                [
                    torch.arange(self.columns, device=self.dev)[temp_mask],
                    descending_ids[: self.n_out],
                ]
            )

        self.perm = perm

    @torch.no_grad()
    def block_transform(self, block, input_feat, block_kwargs):
        if self.online_rotate:
            self.replace_rotate_linears(block)
        if self.owq and not hasattr(self, 'n_out_dict'):
            named_linears = self.model.get_block_linears(block)
            self.n_out_dict = {}
            for i, name in enumerate(named_linears.keys()):
                self.n_out_dict[name] = self.n_outs[i]
        super().block_transform(block, input_feat, block_kwargs)

    @torch.no_grad()
    def subset_transform(
        self,
        subset,
        input_feat,
        subset_kwargs,
    ):
        layers_dict = subset['layers']
        for name in layers_dict:
            layer = layers_dict[name]
            if not isinstance(
                layer, tuple(_LLMC_LINEAR_TYPES_ + _TRANSFORMERS_LINEAR_TYPES_)
            ):
                continue
            self.layer_transform(layer, name)
            self.free(name)

    @torch.no_grad()
    def layer_transform(self, layer, name):
        self.initialize_qparams_and_prepare_weights(layer, name)
        W, H = self.process_hessian_and_weights(layer, name)
        self.update_layer_with_transformed_weights(layer, W, H, name)

    def initialize_qparams_and_prepare_weights(self, layer, name):
        self.qparams = {}
        self.columns = self.layers_cache[name]['columns']
        self.n_out = self.n_out_dict[name] if self.owq else 0
        self.n_nonout = self.columns - self.n_out

        if self.actorder or self.owq:
            self.hessian_sorting(name)

    def process_hessian_and_weights(self, layer, name):
        W = layer.weight.data.clone()
        if isinstance(layer, nn.Conv2d):
            W = W.flatten(1)
        elif isinstance(layer, transformers.Conv1D):
            W = W.t()

        W = W.float()
        H = self.layers_cache[name]['H']
        del self.layers_cache[name]['H']

        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        if not self.ready():
            if self.wquantizer.granularity == 'per_group':
                self.groups = []
                self.search_group_qparams(layer)
            else:
                self.search_layer_qparams(layer)

        if self.actorder or self.owq:
            W = W[:, self.perm]
            H = H[self.perm][:, self.perm]
            self.invperm = torch.argsort(self.perm)

            layer.register_buffer('buf_perm', self.perm)
            layer.register_buffer('buf_invperm', self.invperm)

            if self.owq:
                layer.register_buffer('buf_n_nonout', torch.tensor(self.n_nonout))
                if self.wquantizer.granularity == 'per_channel':
                    _, layer.buf_scales, layer.buf_zeros, _, _ = (
                        self.wquantizer.get_tensor_qparams(W[:, : self.n_nonout])
                    )
                    self.qparams['scale'], self.qparams['zero'] = (
                        layer.buf_scales,
                        layer.buf_zeros,
                    )

        damp = self.percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)

        return W, H

    def update_layer_with_transformed_weights(self, layer, W, H, name):
        Losses = torch.zeros_like(W)
        tmp = torch.zeros_like(W)

        self.weight_transform(W, H, Losses, tmp)
        torch.cuda.synchronize()
        logger.info(f'error {torch.sum(Losses).item()}')

        if self.actorder or self.owq:
            tmp[:, self.n_nonout:] = W[:, self.n_nonout:]
            tmp = tmp[:, self.invperm]

        if isinstance(layer, transformers.Conv1D):
            tmp = tmp.t()

        layer.weight.data = tmp.reshape(layer.weight.shape)

        if self.wquantizer.granularity == 'per_group' and not self.static_groups:
            self.update_model_qparams(layer)

    @torch.no_grad()
    def weight_transform(self, W, Hinv, Losses, tmp):
        for i1 in range(0, self.n_nonout, self.blocksize):
            i2 = min(i1 + self.blocksize, self.n_nonout)
            count = i2 - i1
            W1, Hinv1 = W[:, i1:i2].clone(), Hinv[i1:i2, i1:i2]
            tmp1, Err1, Losses1 = (
                torch.zeros_like(W1),
                torch.zeros_like(W1),
                torch.zeros_like(W1),
            )

            for i in range(count):
                w, d = W1[:, i], Hinv1[i, i]
                if self.wquantizer.granularity == 'per_group':
                    idx = i1 + i
                    if not self.static_groups:
                        if (i1 + i) % self.wquantizer.group_size == 0:
                            column_tensors = W[
                                :,
                                (i1 + i): min(
                                    (i1 + i + self.wquantizer.group_size),
                                    (self.columns - self.n_out),
                                ),
                            ]
                            self.search_column_qparams(column_tensors, idx)
                    else:
                        if self.actorder:
                            idx = self.perm[idx]
                        self.qparams = self.groups[idx // self.wquantizer.group_size]

                q = self.wquantizer.quant_dequant(
                    w.unsqueeze(1),
                    self.qparams['scale'],
                    self.qparams['zero'],
                    self.qparams['qmax'],
                    self.qparams['qmin'],
                ).squeeze(1)

                tmp1[:, i] = w
                Losses1[:, i] = ((w - q) ** 2) / (2 * d**2)
                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            tmp[:, i1:i2], Losses[:, i1:i2] = tmp1, Losses1
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

    @torch.no_grad()
    def cache_input_hook(self, m, inp, out, name, feat_dict):
        if isinstance(m, tuple(_LLMC_LINEAR_TYPES_ + _TRANSFORMERS_LINEAR_TYPES_)):
            self.add_batch(self.named_layers[name], name, inp[0].data, out.data)
        if self.act_static:
            super().cache_input_hook(m, inp, out, name, feat_dict)

    @torch.no_grad()
    def add_batch(self, layer, name, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(
            layer, (FakeQuantLinear, nn.Linear, transformers.Conv1D, RotateLinear)
        ):
            if isinstance(layer, RotateLinear):
                # online rotate
                inp = layer.rotater.rotate(inp)
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if isinstance(layer, nn.Conv2d):
            unfold = nn.Unfold(
                layer.kernel_size,
                dilation=layer.dilation,
                padding=layer.padding,
                stride=layer.stride,
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)

        self.layers_cache[name]['H'] *= self.layers_cache[name]['nsamples'] / (
            self.layers_cache[name]['nsamples'] + tmp
        )
        self.layers_cache[name]['nsamples'] += tmp
        inp = math.sqrt(2 / self.layers_cache[name]['nsamples']) * inp.float()
        self.layers_cache[name]['H'] += inp.matmul(inp.t())

    @torch.no_grad()
    def layer_init(self, layer, name):
        W = layer.weight.data.clone()
        if isinstance(layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(layer, transformers.Conv1D):
            W = W.t()
        self.layers_cache[name]['H'] = torch.zeros(
            (W.shape[1], W.shape[1]), device=self.dev
        )
        self.layers_cache[name]['nsamples'] = 0
        self.layers_cache[name]['columns'] = W.shape[1]

    @torch.no_grad()
    def subset_init(self, subset):
        self.named_layers = subset['layers']
        for name in self.named_layers:
            self.layers_cache[name] = {}
            self.layer_init(self.named_layers[name], name)

    @torch.no_grad()
    def block_init(self, block):
        self.named_layers = self.model.get_block_linears(block)
        for name in self.named_layers:
            self.layers_cache[name] = {}
            self.layer_init(self.named_layers[name], name)

    @torch.no_grad()
    def collect_model_qparams(self):
        for i in range(len(self.blocks)):
            block = self.blocks[i]
            block = block.cuda()
            self.collect_block_qparams(block)
            block = block.cpu()

    @torch.no_grad()
    def split_qparams(self, qparams):
        group_qparams = []
        group_num = math.ceil(self.columns / self.wquantizer.group_size)
        qparams = qparams.reshape(math.ceil(qparams.shape[0] / group_num), -1)
        qparams = qparams.t()
        group_qparams = list(torch.split(qparams, 1, dim=0))
        for i in range(len(group_qparams)):
            group_qparams[i] = group_qparams[i].reshape(-1, 1)
        return group_qparams

    @torch.no_grad()
    def merge_qparams(self, qparams):
        if isinstance(qparams, int):
            return qparams
        if self.wquantizer.granularity == 'per_head':
            head_size = self.rows // self.head_num
            qparams = qparams.t()
            qparams = qparams.repeat(head_size, 1)
            qparams = qparams.t()
            qparams = qparams.reshape(-1, 1)
        elif self.wquantizer.granularity == 'per_group':
            qparams = torch.stack(qparams, dim=1)
            qparams = qparams.reshape(-1, 1)
        return qparams

    @torch.no_grad()
    def search_column_qparams(self, c_tensor, idx):
        _, scale, zero, qmax, qmin = self.wquantizer.get_tensor_qparams(c_tensor)
        self.qparams['scale'] = scale
        self.qparams['zero'] = zero
        self.qparams['qmax'] = qmax
        self.qparams['qmin'] = qmin
        qparams = copy.deepcopy(self.qparams)
        self.groups[idx // self.wquantizer.group_size] = qparams

    @torch.no_grad()
    def search_layer_qparams(self, layer):
        scales = layer.buf_scales
        zeros = layer.buf_zeros
        scales = self.merge_qparams(scales)
        if not self.wquantizer.sym:
            zeros = self.merge_qparams(zeros)
        self.qparams['scale'], self.qparams['zero'] = scales, zeros
        self.qparams['qmax'] = layer.buf_qmax
        self.qparams['qmin'] = layer.buf_qmin

    @torch.no_grad()
    def search_group_qparams(self, layer):
        scales = layer.buf_scales
        zeros = layer.buf_zeros
        self.group_scales = self.split_qparams(scales)
        if not self.wquantizer.sym:
            self.group_zeros = self.split_qparams(zeros)
        for i in range(len(self.group_scales)):
            qparams = {}
            qparams['scale'] = self.group_scales[i]
            if not self.wquantizer.sym:
                qparams['zero'] = self.group_zeros[i]
            else:
                qparams['zero'] = torch.tensor(0.0)
            qparams['qmax'] = layer.buf_qmax
            qparams['qmin'] = layer.buf_qmin
            self.groups.append(qparams)

    @torch.no_grad()
    def update_model_qparams(self, layer):
        _scales = []
        _zeros = []
        for g in self.groups:
            _scales.append(g['scale'])
            _zeros.append(g['zero'])
        scales = self.merge_qparams(_scales)
        layer.buf_scales = copy.deepcopy(scales)

        if not self.wquantizer.sym:
            zeros = self.merge_qparams(_zeros)
            layer.buf_zeros = copy.deepcopy(zeros)

    @torch.no_grad()
    def w_q(self, module, wquantizer):
        weight = module.weight.data
        args = {}
        args['scales'] = module.buf_scales
        args['zeros'] = module.buf_zeros
        args['qmax'] = module.buf_qmax
        args['qmin'] = module.buf_qmin
        args['scales'] = args['scales'].to(self.model_dtype)

        weight, scales, zeros = wquantizer.real_quant_weight_static(weight, args)
        return weight, scales, zeros

    @torch.no_grad()
    def w_qdq(self, module, wquantizer):
        weight = module.weight
        if self.need_perm:
            perm = module.buf_perm
            weight = module.weight[:, perm]

        args = {}
        args['scales'] = module.buf_scales
        if hasattr(module, 'buf_zeros'):
            args['zeros'] = module.buf_zeros
        else:
            args['zeros'] = None
        args['qmax'] = module.buf_qmax
        args['qmin'] = module.buf_qmin

        if self.owq:
            fp_weight = weight[:, module.buf_n_nonout:]

        weight = wquantizer.fake_quant_weight_static(weight, args).to(self.model_dtype)

        if self.owq:
            weight[:, module.buf_n_nonout:] = fp_weight.to(self.model_dtype)

        if self.need_perm:
            invperm = module.buf_invperm
            weight = weight[:, invperm]

        return weight

    @torch.no_grad()
    def deploy(self, quant_format):
        if quant_format not in ['fake_quant', 'origin_float']:
            assert not self.need_perm
        super().deploy(quant_format)
        self.model.convert_dtype(self.model_dtype)

    @torch.no_grad()
    def save_model(self, path):
        self.model.convert_dtype(self.model_dtype)
        super().save_model(path)

    @torch.no_grad()
    def free(self, name):
        self.H = None
        self.Losses = None
        self.Trace = None
        del self.layers_cache[name]
        torch.cuda.empty_cache()

    @torch.no_grad()
    def ready(self):
        if 'scale' not in self.qparams:
            return False
        return torch.all(self.qparams['scale'] != 0)
