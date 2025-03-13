import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

from .quant import FloatQuantizer
from .utils import is_fp8_supported_gpu

if is_fp8_supported_gpu():
    from .fp8_kernel import act_quant, fp8_gemm, weight_cast_to_bf16
    USE_FP8GEMM_TRITON_KERNEL = True
    logger.info('import fp8_kernel successful.')
else:
    USE_FP8GEMM_TRITON_KERNEL = False
    from .quant import weight_cast_to_bf16

try:
    import fast_hadamard_transform

    from .hadamard_utils import matmul_hadU_cuda
except Exception:
    logger.warning(
        'fast_hadamard_transform not installed. '
        'If you need it, please install it firstly.'
    )

from .utils import calculate_zeros_width


def block_wise_fp8_forward_func(x, w, w_scale, block_size, bias):
    x, scale = act_quant(x, block_size)
    y = fp8_gemm(x, scale, w, w_scale).to(torch.bfloat16)
    if bias is not None:
        y += bias
    return y


class LlmcFp8Linear(nn.Module):
    def __init__(self, in_features, out_features, bias, block_size=128):
        super().__init__()
        self.block_size = block_size
        self.in_features = in_features
        self.out_features = out_features
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

        # Init empty weight and scale
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.float8_e4m3fn)
        )
        scale_out_features = (out_features + block_size - 1) // block_size
        scale_in_features = (in_features + block_size - 1) // block_size
        self.weight_scale_inv = nn.Parameter(
            torch.empty(scale_out_features, scale_in_features, dtype=torch.float32)
        )

    def forward(self, x):
        if self.weight.data.dtype == torch.float8_e4m3fn:
            if USE_FP8GEMM_TRITON_KERNEL:
                y = block_wise_fp8_forward_func(
                    x, self.weight, self.weight_scale_inv, self.block_size, self.bias
                )
                return y
            else:
                self.weight.data \
                    = weight_cast_to_bf16(self.weight.data,
                                          self.weight_scale_inv.data).to(torch.bfloat16)
        y = torch.functional.F.linear(x, self.weight, self.bias)
        return y

    @classmethod
    @torch.no_grad()
    def new(cls, module):
        in_features = module.in_features
        out_features = module.out_features
        bias = module.bias
        new_module = cls(in_features, out_features, bias)
        return new_module

    def __repr__(self):
        return (
            'LlmcFp8Linear('
            + f'in_features={self.in_features}, '
            + f'out_features={self.out_features}, '
            + f'bias={self.bias is not None}, '
            + f'weight_shape={self.weight.shape}, '
            + f'weight_dtype={self.weight.dtype}, '
            # + f"scales_shape={self.weight_scale_inv.shape}, "
            # + f"scales_dtype={self.weight_scale_inv.dtype}, "
            + f'use_fp8gemm_triton_kernel={USE_FP8GEMM_TRITON_KERNEL})'
        )


class LlmcActFn(nn.Module):
    def __init__(self, module, a_qdq) -> None:
        super().__init__()
        self.act_fn = module
        self.a_qdq = a_qdq
        self.calib = True

    def forward(self, x):
        if self.a_qdq is not None and not self.calib:
            x = self.a_qdq(x, self)
        x = self.act_fn(x)
        return x

    @classmethod
    @torch.no_grad()
    def new(cls, module, a_qdq):
        new_module = cls(module, a_qdq)
        return new_module

    def disable_calib(self):
        self.calib = False

    def __repr__(self):
        return f'LlmcActFn(calib={self.calib})'


class RectifiedSigmoid(nn.Module):
    def __init__(self, gamma, zeta):
        super(RectifiedSigmoid, self).__init__()
        self.gamma = gamma
        self.zeta = zeta

    def forward(self, x):
        return torch.clamp(
            torch.sigmoid(x) * (self.zeta - self.gamma) + self.gamma, 0, 1
        )

    def inverse(self, y):
        """return x that satisfies y = RectifiedSigmoid(x)"""
        return -torch.log((self.zeta - self.gamma) / (y - self.gamma) - 1)


class LlmcLayerNorm(nn.Module):
    def __init__(self, weight, bias, eps, normalized_shape, elementwise_affine):
        super().__init__()
        self.register_buffer('weight', weight)
        if bias is not None:
            self.register_buffer('bias', bias)
        else:
            self.bias = None
        self.eps = eps
        self.norm_func = nn.functional.layer_norm
        self.normalized_shape = normalized_shape
        self.elementwise_affine = elementwise_affine
        self.use_tmp_parameter = False

    def forward(self, x):
        if self.use_tmp_parameter:
            weight = self.tmp_weight
            bias = self.tmp_bias
        else:
            weight = self.weight
            bias = self.bias
        out = self.norm_func(x, self.normalized_shape, weight, bias, eps=self.eps)
        return out

    @classmethod
    @torch.no_grad()
    def new(cls, module):
        weight = module.weight.data
        if module.bias is not None:
            bias = module.bias.data
        else:
            bias = None
        eps = module.eps
        normalized_shape = module.normalized_shape
        elementwise_affine = module.elementwise_affine

        new_module = cls(weight, bias, eps, normalized_shape, elementwise_affine)

        return new_module

    def __repr__(self):
        return (
            f'LlmcLayerNorm({self.normalized_shape},'
            f'eps={self.eps},'
            f'elementwise_affine={self.elementwise_affine})'
        )


class LlmcLlamaRMSNorm(nn.Module):
    def __init__(self, weight, eps=1e-6):
        super().__init__()
        self.register_buffer('weight', weight)
        self.bias = None
        self.variance_epsilon = eps
        self.use_tmp_parameter = False

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        if self.use_tmp_parameter:
            weight = self.tmp_weight
            bias = self.tmp_bias if hasattr(self, 'tmp_bias') else None
        else:
            weight = self.weight
            bias = self.bias if hasattr(self, 'bias') else None

        return (
            (weight * hidden_states + bias).to(input_dtype)
            if bias is not None
            else (weight * hidden_states).to(input_dtype)
        )

    @classmethod
    @torch.no_grad()
    def new(cls, module):
        weight = module.weight.data
        eps = module.variance_epsilon
        new_module = cls(weight, eps)
        return new_module

    def __repr__(self):
        return 'LlmcLlamaRMSNorm()'


class LlmcRMSNorm(nn.Module):
    def __init__(self, weight, eps=1e-6):
        super().__init__()
        self.variance_epsilon = eps
        self.weight = nn.Parameter(torch.ones_like(weight))

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return hidden_states.to(input_dtype)

    @classmethod
    @torch.no_grad()
    def new(cls, module):
        if hasattr(module, 'eps'):
            eps = module.eps
        else:
            eps = module.variance_epsilon
        weight = module.weight
        new_module = cls(weight, eps)
        return new_module

    def __repr__(self):
        return 'LlmcRMSNorm()'


class LlmcQwen2RMSNorm(LlmcLlamaRMSNorm):
    def __init__(self, weight, eps=1e-6):
        super().__init__(weight, eps)

    def __repr__(self):
        return 'LlmcQwen2RMSNorm()'


class LlmcMixtralRMSNorm(LlmcLlamaRMSNorm):
    def __init__(self, weight, eps=1e-6):
        super().__init__(weight, eps)

    def __repr__(self):
        return 'LlmcMixtralRMSNorm()'


class LlmcMistralRMSNorm(LlmcLlamaRMSNorm):
    def __init__(self, weight, eps=1e-6):
        super().__init__(weight, eps)

    def __repr__(self):
        return 'LlmcMistralRMSNorm()'


class LlmcInternLM2RMSNorm(LlmcLlamaRMSNorm):
    def __init__(self, weight, eps=1e-6):
        super().__init__(weight, eps)

    def __repr__(self):
        return 'LlmcInternLM2RMSNorm()'


class LlmcGemma2RMSNorm(LlmcLlamaRMSNorm):
    def __init__(self, weight, eps=1e-6):
        super().__init__(weight, eps)

    def __repr__(self):
        return 'LlmcGemma2RMSNorm()'


class LlmcMiniCPMRMSNorm(LlmcLlamaRMSNorm):
    def __init__(self, weight, eps=1e-6):
        super().__init__(weight, eps)

    def __repr__(self):
        return 'LlmcMiniCPMRMSNorm()'


class OriginFloatLinear(nn.Module):
    def __init__(self, weight, bias, ori_module):
        super().__init__()
        self.register_buffer('weight', weight)
        if bias is not None:
            self.register_buffer('bias', bias)
        else:
            self.bias = None

        for name, buf in ori_module.named_buffers():
            if name.startswith('buf_'):
                self.register_buffer(name, buf.data)
        if hasattr(self, 'buf_rotate') and self.buf_rotate:
            self.rotater = ori_module.rotater
        else:
            self.buf_rotate = False

        if self.weight.data.dtype == torch.float8_e4m3fn:
            self.fp8_forward = True
            self.weight_scale_inv = ori_module.weight_scale_inv
            self.block_size = 128
        else:
            self.fp8_forward = False

    @torch.no_grad()
    def forward(self, x):
        if hasattr(self, 'buf_rotate') and self.buf_rotate:
            x = self.rotater.rotate(x)
        if self.fp8_forward:
            y = block_wise_fp8_forward_func(
                x, self.weight, self.weight_scale_inv, self.block_size, self.bias
            )
        else:
            y = torch.functional.F.linear(x, self.weight, self.bias)
        return y

    @classmethod
    @torch.no_grad()
    def new(cls, module):
        if isinstance(module, nn.Linear):
            return module

        weight = module.weight.data
        if module.bias is not None:
            bias = module.bias.data
        else:
            bias = None

        new_module = cls(weight, bias, module)

        new_module.in_features = module.in_features
        new_module.out_features = module.out_features
        return new_module

    def __repr__(self):
        return (
            f'OriginFloatLinear(in_features={self.in_features},'
            f'out_features={self.out_features},'
            f'online_rotate={self.buf_rotate},'
            f'fp8_forward={self.fp8_forward},'
            f'bias={self.bias is not None})'
        )


class Rotater:
    def __init__(
        self, online_full_had, online_partial_had, fp32_had, K, had_K=None, had_dim=None
    ):
        self.online_full_had = online_full_had
        self.online_partial_had = online_partial_had
        self.fp32_had = fp32_had
        self.K = K
        self.had_K = had_K
        self.had_dim = had_dim

    def rotate(self, x):
        x_dtype = x.dtype

        if self.online_full_had:
            if self.fp32_had:
                x = matmul_hadU_cuda(x.float(), self.had_K, self.K).to(x_dtype)
            else:
                x = matmul_hadU_cuda(x, self.had_K, self.K)

        elif self.online_partial_had:
            if self.fp32_had:
                x = x.float()
            init_shape = x.shape
            if self.K == 1:
                x = fast_hadamard_transform.hadamard_transform(
                    x.reshape(
                        -1, init_shape[-1] // self.had_dim, self.had_dim
                    ).transpose(1, 2),
                    scale=1 / math.sqrt(init_shape[-1] // self.had_dim),
                ).transpose(1, 2)
            else:
                self.had_K = self.had_K.to(x.device)

                x = (
                    self.had_K.to(x.dtype)
                    @ x.reshape(-1, init_shape[-1] // self.had_dim, self.had_dim)
                ) / math.sqrt(init_shape[-1] // self.had_dim)

            if self.fp32_had:
                x = x.to(x_dtype)
            x = x.reshape(init_shape)

        return x


class RotateLinear(nn.Module):
    def __init__(
        self,
        weight,
        bias,
        ori_module,
        online_full_had,
        online_partial_had,
        fp32_had,
        K,
        had_K,
        had_dim,
    ):
        super().__init__()
        self.register_buffer('weight', weight)
        if bias is not None:
            self.register_buffer('bias', bias)
        else:
            self.bias = None

        for name, buf in ori_module.named_buffers():
            if name.startswith('buf_'):
                self.register_buffer(name, buf.data)

        self.rotater = Rotater(
            online_full_had, online_partial_had, fp32_had, K, had_K, had_dim
        )
        self.register_buffer('buf_rotate', torch.tensor(True))

    def forward(self, x):
        x = self.rotater.rotate(x)
        x = torch.functional.F.linear(x, self.weight, self.bias)

        return x

    @classmethod
    @torch.no_grad()
    def new(
        cls, module, online_full_had, online_partial_had, fp32_had, K, had_K, had_dim
    ):
        weight = module.weight.data
        if module.bias is not None:
            bias = module.bias.data
        else:
            bias = None

        new_module = cls(
            weight,
            bias,
            ori_module=module,
            online_full_had=online_full_had,
            online_partial_had=online_partial_had,
            fp32_had=fp32_had,
            K=K,
            had_K=had_K,
            had_dim=had_dim,
        )

        new_module.in_features = module.in_features
        new_module.out_features = module.out_features
        return new_module

    @classmethod
    def get_func_name(cls, any_callable):
        if isinstance(any_callable, partial):
            return any_callable.func.__name__
        return any_callable.__name__

    def register_activation_parameters(self, named_parameters):
        pass

    def __repr__(self):
        return (
            f'RotateLinear(in_features={self.in_features},'
            f'out_features={self.out_features},'
            f'bias={self.bias is not None},'
            f'online_rotate={self.buf_rotate})'
        )


class FakeQuantLinear(nn.Module):
    def __init__(self, weight, bias, ori_module, w_qdq, a_qdq):
        super().__init__()
        self.register_buffer('weight', weight)
        if bias is not None:
            self.register_buffer('bias', bias)
        else:
            self.bias = None
        self.a_qdq = a_qdq
        self.w_qdq = w_qdq

        for name, buf in ori_module.named_buffers():
            if name.startswith('buf_'):
                self.register_buffer(name, buf.data)
        for name, buf in ori_module.named_parameters():
            if name.startswith('buf_'):
                self.register_buffer(name, buf.data)

        if hasattr(self, 'buf_rotate') and self.buf_rotate:
            self.rotater = ori_module.rotater
        else:
            self.buf_rotate = False

        if self.weight.data.dtype == torch.float8_e4m3fn:
            self.fp8_forward = True
            self.weight_scale_inv = ori_module.weight_scale_inv
            self.block_size = 128
        else:
            self.fp8_forward = False

        self.dynamic_quant_weight = False
        self.dynamic_quant_tmp_weight = False

    def forward(self, x):
        if hasattr(self, 'buf_rotate') and self.buf_rotate:
            x = self.rotater.rotate(x)

        if self.a_qdq is not None:
            x = self.a_qdq(x, self)

        if not hasattr(self, 'tmp_weight'):
            tmp_weight = self.w_qdq(self)
            self.register_buffer('tmp_weight', tmp_weight, persistent=False)
            self.tmp_bias = self.bias

        elif self.dynamic_quant_weight:
            self.tmp_weight = self.w_qdq(self)
            self.tmp_bias = self.bias

        elif self.dynamic_quant_tmp_weight:
            self.tmp_weight = self.w_qdq(self)

        if self.fp8_forward:
            y = block_wise_fp8_forward_func(
                x, self.weight, self.weight_scale_inv, self.block_size, self.bias
            )
        else:
            y = torch.functional.F.linear(x, self.tmp_weight, self.tmp_bias)
        return y

    @classmethod
    @torch.no_grad()
    def new(cls, module, w_qdq, a_qdq):
        weight = module.weight.data
        if hasattr(module, 'bias') and module.bias is not None:
            bias = module.bias.data
        else:
            bias = None

        new_module = cls(weight, bias, ori_module=module, w_qdq=w_qdq, a_qdq=a_qdq)

        new_module.in_features = module.in_features
        new_module.out_features = module.out_features
        new_module.w_qdq_name = cls.get_func_name(w_qdq)
        new_module.a_qdq_name = (
            cls.get_func_name(a_qdq) if a_qdq is not None else 'None'
        )
        return new_module

    @classmethod
    def get_func_name(cls, any_callable):
        if isinstance(any_callable, partial):
            return any_callable.func.__name__
        return any_callable.__name__

    def __repr__(self):
        return (
            f'FakeQuantLinear(in_features={self.in_features},'
            f'out_features={self.out_features}, bias={self.bias is not None},'
            f'weight_quant={self.w_qdq_name},'
            f'act_quant={self.a_qdq_name},'
            f'online_rotate={self.buf_rotate})'
        )


class EffcientFakeQuantLinear(nn.Module):
    def __init__(self, weight, bias, ori_module, a_qdq):
        super().__init__()
        self.register_buffer('weight', weight)
        if bias is not None:
            self.register_buffer('bias', bias)
        else:
            self.bias = None
        self.a_qdq = a_qdq

        for name, buf in ori_module.named_buffers():
            if name.startswith('buf_'):
                self.register_buffer(name, buf.data)
        if hasattr(self, 'buf_rotate') and self.buf_rotate:
            self.rotater = ori_module.rotater
        else:
            self.buf_rotate = False

        if self.weight.data.dtype == torch.float8_e4m3fn:
            self.fp8_forward = True
            self.weight_scale_inv = ori_module.weight_scale_inv
            self.block_size = 128
        else:
            self.fp8_forward = False

    @torch.no_grad()
    def forward(self, x):
        if hasattr(self, 'buf_rotate') and self.buf_rotate:
            x = self.rotater.rotate(x)

        if self.a_qdq is not None:
            x = self.a_qdq(x, self)

        if self.fp8_forward:
            y = block_wise_fp8_forward_func(
                x, self.weight, self.weight_scale_inv, self.block_size, self.bias
            )
        else:
            y = torch.functional.F.linear(x, self.weight, self.bias)
        return y

    @classmethod
    @torch.no_grad()
    def new(cls, module, w_qdq, a_qdq, debug_print={}):
        weight = w_qdq(module)

        if module.bias is not None:
            bias = module.bias.data
        else:
            bias = None

        new_module = cls(weight, bias, ori_module=module, a_qdq=a_qdq)

        new_module.in_features = module.in_features
        new_module.out_features = module.out_features
        new_module.w_qdq_name = cls.get_func_name(w_qdq)
        new_module.a_qdq_name = (
            cls.get_func_name(a_qdq) if a_qdq is not None else 'None'
        )
        new_module.debug_print = debug_print
        return new_module

    @classmethod
    def get_func_name(cls, any_callable):
        if isinstance(any_callable, partial):
            return any_callable.func.__name__
        return any_callable.__name__

    def __repr__(self):
        return (
            f'EffcientFakeQuantLinear(in_features={self.in_features},'
            f'out_features={self.out_features},'
            f'bias={self.bias is not None},'
            f'weight_quant={self.w_qdq_name},'
            f'act_quant={self.a_qdq_name},'
            f'online_rotate={self.buf_rotate},'
            f'fp8_forward={self.fp8_forward},'
            f'debug_print={self.debug_print})'
        )


class VllmRealQuantLinear(nn.Module):
    def __init__(self, weight, bias, scales, input_scale, need_pack, scales_name):
        super().__init__()
        weight_name = 'weight_packed' if need_pack else 'weight'
        self.register_buffer(weight_name, weight)

        (
            self.register_buffer('bias', bias)
            if bias is not None
            else setattr(self, 'bias', None)
        )

        self.register_buffer(scales_name, scales)
        self.register_buffer('input_scale', input_scale)

    @torch.no_grad()
    def forward(self, x):
        raise NotImplementedError

    @classmethod
    @torch.no_grad()
    def new(cls, module, w_q, quant_config):
        weight, scales = cls.quant_pack(module, w_q, quant_config)
        if hasattr(module, 'buf_act_scales_0'):
            input_scale = module.buf_act_scales_0
        else:
            input_scale = None
        if (
            'act' in quant_config
            and quant_config.act.get('static', False)
            and quant_config.get('quant_type', 'int-quant') == 'int-quant'
        ):
            input_scale = input_scale.unsqueeze(0)

        if module.bias is not None:
            bias = module.bias.data
        else:
            bias = None

        need_pack = quant_config['weight'].get('need_pack', False)

        if quant_config['weight']['granularity'] == 'per_block':
            scales_name = 'weight_scale_inv'
        else:
            scales_name = 'weight_scale'

        new_module = cls(weight, bias, scales, input_scale, need_pack, scales_name)
        new_module.in_features = module.in_features
        new_module.out_features = module.out_features
        new_module.weight_shape = weight.shape
        new_module.weight_dtype = weight.dtype
        new_module.scales_shape = scales.shape
        new_module.scales_dtype = scales.dtype

        new_module.zeros_shape = None
        new_module.zeros_dtype = None

        return new_module

    @classmethod
    @torch.no_grad()
    def quant_pack(cls, module, w_q, quant_config):
        if module.weight.data.dtype == torch.float8_e4m3fn:
            module.weight.data = weight_cast_to_bf16(
                module.weight.data, module.weight_scale_inv.data
            ).to(torch.bfloat16)
        weight, scales, zeros = w_q(module)
        need_pack = quant_config['weight'].get('need_pack', False)
        if need_pack:
            weight, scales = cls.pack(weight, scales, quant_config)
        return weight, scales

    @classmethod
    @torch.no_grad()
    def pack(self, weight, scales, quant_config):

        # Packs a tensor of quantized weights stored in int8 into int32s with padding
        num_bits = quant_config['weight']['bit']

        # convert to unsigned for packing
        offset = pow(2, num_bits) // 2
        weight = (weight + offset).to(torch.uint8)
        weight = weight.cpu().numpy().astype(np.uint32)
        pack_factor = 32 // num_bits

        # pad input tensor and initialize packed output
        packed_size = math.ceil(weight.shape[1] / pack_factor)
        packed = np.zeros((weight.shape[0], packed_size), dtype=np.uint32)
        padding = packed.shape[1] * pack_factor - weight.shape[1]
        weight = np.pad(weight, pad_width=[(0, 0), (0, padding)], constant_values=0)

        # pack values
        for i in range(pack_factor):
            packed |= weight[:, i::pack_factor] << num_bits * i

        packed = np.ascontiguousarray(packed).view(np.int32)
        int_weight = torch.from_numpy(packed).cuda()
        del weight, packed
        return int_weight, scales.to(torch.float16)

    def __repr__(self):
        return (
            'VllmRealQuantLinear('
            + f'in_features={self.in_features}, '
            + f'out_features={self.out_features}, '
            + f'bias={self.bias is not None}, '
            + f'weight_shape={self.weight_shape}, '
            + f'weight_dtype={self.weight_dtype}, '
            + f'scales_shape={self.scales_shape}, '
            + f'scales_dtype={self.scales_dtype}, '
            + f'zeros_shape={self.zeros_shape}, '
            + f'zeros_dtype={self.zeros_dtype})'
        )


class LightllmRealQuantLinear(VllmRealQuantLinear):
    def __init__(self, weight, bias, scales, input_scale, need_pack):
        super().__init__(weight, bias, scales, input_scale, need_pack)

    def __repr__(self):
        return (
            'LightllmRealQuantLinear('
            + f'in_features={self.in_features}, '
            + f'out_features={self.out_features}, '
            + f'bias={self.bias is not None}, '
            + f'weight_shape={self.weight_shape}, '
            + f'weight_dtype={self.weight_dtype}, '
            + f'scales_shape={self.scales_shape}, '
            + f'scales_dtype={self.scales_dtype}, '
            + f'zeros_shape={self.zeros_shape}, '
            + f'zeros_dtype={self.zeros_dtype})'
        )


class SglRealQuantLinear(VllmRealQuantLinear):
    def __init__(self, weight, bias, scales, input_scale, need_pack):
        super().__init__(weight, bias, scales, input_scale, need_pack)

    def __repr__(self):
        return (
            'SglRealQuantLinear('
            + f'in_features={self.in_features}, '
            + f'out_features={self.out_features}, '
            + f'bias={self.bias is not None}, '
            + f'weight_shape={self.weight_shape}, '
            + f'weight_dtype={self.weight_dtype}, '
            + f'scales_shape={self.scales_shape}, '
            + f'scales_dtype={self.scales_dtype}, '
            + f'zeros_shape={self.zeros_shape}, '
            + f'zeros_dtype={self.zeros_dtype})'
        )


class AutoawqRealQuantLinear(nn.Module):
    def __init__(self, weight, bias, scales, zeros):
        super().__init__()
        self.register_buffer('qweight', weight)

        (
            self.register_buffer('bias', bias)
            if bias is not None
            else setattr(self, 'bias', None)
        )

        self.register_buffer('scales', scales)

        (
            self.register_buffer('qzeros', zeros)
            if zeros is not None
            else setattr(self, 'qzeros', None)
        )

    @torch.no_grad()
    def forward(self, x):
        raise NotImplementedError

    @classmethod
    @torch.no_grad()
    def new(cls, module, w_q, quant_config):
        weight, scales, zeros = cls.quant_pack(module, w_q, quant_config)
        if module.bias is not None:
            bias = module.bias.data
        else:
            bias = None

        new_module = cls(weight, bias, scales, zeros)
        new_module.in_features = module.in_features
        new_module.out_features = module.out_features
        new_module.weight_shape = weight.shape
        new_module.weight_dtype = weight.dtype
        new_module.scales_shape = scales.shape
        new_module.scales_dtype = scales.dtype

        if zeros is not None:
            new_module.zeros_shape = zeros.shape
            new_module.zeros_dtype = zeros.dtype
        else:
            new_module.zeros_shape = None
            new_module.zeros_dtype = None

        return new_module

    @classmethod
    @torch.no_grad()
    def quant_pack(cls, module, w_q, quant_config):
        if module.weight.data.dtype == torch.float8_e4m3fn:
            module.weight.data = weight_cast_to_bf16(
                module.weight.data, module.weight_scale_inv.data
            ).to(torch.bfloat16)
        weight, scales, zeros = w_q(module)
        pack_version = quant_config['weight']['pack_version']
        if pack_version == 'gemm_pack':
            int_weight, scales, int_zeros = cls.gemm_pack(
                weight, scales, zeros, quant_config
            )
        elif pack_version == 'gemv_pack':
            int_weight, scales, int_zeros = cls.gemv_pack(
                module, weight, scales, zeros, quant_config
            )
        return int_weight, scales, int_zeros

    @classmethod
    @torch.no_grad()
    def gemm_pack(self, weight, scales, zeros, quant_config):

        if zeros is not None:
            zeros = zeros.t().contiguous()
        scales = scales.t().contiguous()
        weight = weight.t().contiguous()

        bit = quant_config['weight']['bit']
        pack_num = 32 // bit

        int_weight = torch.zeros(
            (weight.shape[0], weight.shape[1] // 32 * bit),
            dtype=torch.int32,
            device=weight.device,
        )

        for col in range(weight.shape[1] // pack_num):
            if bit == 4:
                order_map = [0, 2, 4, 6, 1, 3, 5, 7]
            else:
                raise NotImplementedError('Only 4-bit are supported for now.')
            for i in range(pack_num):
                int_weight_col = weight[:, col * pack_num + order_map[i]]
                int_weight[:, col] |= int_weight_col << (i * bit)

        if zeros is not None:
            int_zeros = torch.zeros(
                (zeros.shape[0], zeros.shape[1] // 32 * bit),
                dtype=torch.int32,
                device=zeros.device,
            )

            for col in range(zeros.shape[1] // pack_num):
                if bit == 4:
                    order_map = [0, 2, 4, 6, 1, 3, 5, 7]
                else:
                    raise NotImplementedError('Only 4-bit are supported for now.')
                for i in range(pack_num):
                    intzero_col = zeros[:, col * pack_num + order_map[i]]
                    int_zeros[:, col] |= intzero_col << (i * bit)
        else:
            int_zeros = None
        del weight
        return int_weight, scales, int_zeros

    @classmethod
    @torch.no_grad()
    def gemv_pack(self, module, weight, scales, zeros, quant_config):

        bit = quant_config['weight']['bit']
        group_size = quant_config['weight']['group_size']
        pack_num = 32 // bit

        q_scales = torch.zeros(
            (
                scales.shape[0],
                calculate_zeros_width(module.in_features, group_size) * pack_num,
            ),
            dtype=torch.float16,
            device=scales.device,
        )
        q_scales[:, : scales.shape[1]] = scales

        int_weight = torch.zeros(
            (weight.shape[0], weight.shape[1] // 32 * bit),
            dtype=torch.int32,
            device=weight.device,
        )

        for col in range(weight.shape[1] // pack_num):
            if bit == 4:
                order_map = [0, 1, 2, 3, 4, 5, 6, 7]
            else:
                raise NotImplementedError('Only 4-bit are supported for now.')
            for i in range(pack_num):
                int_weight_col = weight[:, col * pack_num + order_map[i]]
                int_weight[:, col] |= int_weight_col << (i * bit)

        if zeros is not None:
            int_zeros = torch.zeros(
                (zeros.shape[0], calculate_zeros_width(module.in_features, group_size)),
                dtype=torch.int32,
                device=zeros.device,
            )

            for col in range(zeros.shape[1] // pack_num):
                if bit == 4:
                    order_map = [0, 1, 2, 3, 4, 5, 6, 7]
                else:
                    raise NotImplementedError('Only 4-bit are supported for now.')
                for i in range(pack_num):
                    if col * pack_num + order_map[i] >= zeros.shape[1]:
                        continue
                    int_zero_col = zeros[:, col * pack_num + order_map[i]]
                    int_zeros[:, col] |= int_zero_col << (i * bit)
        else:
            int_zeros = None

        return int_weight, q_scales, int_zeros

    def __repr__(self):
        return (
            'AutoawqRealQuantLinear('
            + f'in_features={self.in_features}, '
            + f'out_features={self.out_features}, '
            + f'bias={self.bias is not None}, '
            + f'weight_shape={self.weight_shape}, '
            + f'weight_dtype={self.weight_dtype}, '
            + f'scales_shape={self.scales_shape}, '
            + f'scales_dtype={self.scales_dtype}, '
            + f'zeros_shape={self.zeros_shape}, '
            + f'zeros_dtype={self.zeros_dtype})'
        )


class MlcllmRealQuantLinear(AutoawqRealQuantLinear):
    def __init__(self, weight, bias, scales, zeros):
        super().__init__(weight, bias, scales, zeros)

    def __repr__(self):
        return (
            'MlcllmRealQuantLinear('
            + f'in_features={self.in_features}, '
            + f'out_features={self.out_features}, '
            + f'bias={self.bias is not None}, '
            + f'weight_shape={self.weight_shape}, '
            + f'weight_dtype={self.weight_dtype}, '
            + f'scales_shape={self.scales_shape}, '
            + f'scales_dtype={self.scales_dtype}, '
            + f'zeros_shape={self.zeros_shape}, '
            + f'zeros_dtype={self.zeros_dtype})'
        )


_TRANSFORMERS_LN_TYPES_ = ALL_LAYERNORM_LAYERS
_TRANSFORMERS_LINEAR_TYPES_ = [nn.Linear]

_MODEL_LN_TYPES_PAIRS_ = {
    'Llama': LlmcLlamaRMSNorm,
    'Llava': LlmcLlamaRMSNorm,
    'Mistral': LlmcMistralRMSNorm,
    'Mixtral': LlmcMixtralRMSNorm,
    'Interlm2': LlmcInternLM2RMSNorm,
    'Qwen2': LlmcQwen2RMSNorm,
    'Gemma2': LlmcGemma2RMSNorm,
    'MiniCPM': LlmcMiniCPMRMSNorm,
    'Starcoder': LlmcLayerNorm,
    'Opt': LlmcLayerNorm,
    'Bloom': LlmcLayerNorm,
}


_LLMC_LN_TYPES_ = [
    LlmcLayerNorm,
    LlmcLlamaRMSNorm,
    LlmcRMSNorm,
    LlmcQwen2RMSNorm,
    LlmcMistralRMSNorm,
    LlmcMixtralRMSNorm,
    LlmcInternLM2RMSNorm,
    LlmcGemma2RMSNorm,
    LlmcMiniCPMRMSNorm,
]


_LLMC_LINEAR_TYPES_ = [
    LlmcFp8Linear,
    OriginFloatLinear,
    RotateLinear,
    FakeQuantLinear,
    EffcientFakeQuantLinear,
    VllmRealQuantLinear,
    SglRealQuantLinear,
    AutoawqRealQuantLinear,
    MlcllmRealQuantLinear,
    LightllmRealQuantLinear,
]

_REALQUANT_LINEAR_MAP_ = {
    'vllm_quant': VllmRealQuantLinear,
    'lightllm_quant': LightllmRealQuantLinear,
    'sgl_quant': SglRealQuantLinear,
    'autoawq_quant': AutoawqRealQuantLinear,
    'mlcllm_quant': MlcllmRealQuantLinear,
}
