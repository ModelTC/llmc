import torch
import torch.nn as nn
import gc
import math
from functools import partial
from loguru import logger
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers.models.mistral.modeling_mistral import MistralRMSNorm
from transformers.models.mixtral.modeling_mixtral import MixtralRMSNorm
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

try:
    import fast_hadamard_transform
    from .hadamard_utils import matmul_hadU_cuda
except:
    logger.info(
        "fast_hadamard_transform not installed. If you need it, please install it firstly."
    )


class LlmcLayerNorm(nn.Module):
    def __init__(self, weight, bias, eps, normalized_shape, elementwise_affine):
        super().__init__()
        self.register_buffer("weight", weight)
        if bias is not None:
            self.register_buffer("bias", bias)
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
        return f"LlmcLayerNorm({self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine})"


class LlmcLlamaRMSNorm(nn.Module):
    def __init__(self, weight, eps=1e-6):
        super().__init__()
        self.register_buffer("weight", weight)
        self.bias = None
        self.variance_epsilon = eps
        self.use_tmp_parameter = False

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        if self.use_tmp_parameter:
            weight = self.tmp_weight
            bias = self.tmp_bias if hasattr(self, "tmp_bias") else None
        else:
            weight = self.weight
            bias = self.bias if hasattr(self, "bias") else None

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
        return f"LlmcLlamaRMSNorm()"


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
        eps = module.variance_epsilon
        weight = module.weight
        new_module = cls(weight, eps)
        return new_module

    def __repr__(self):
        return f"LlmcRMSNorm()"


class LlmcQwen2RMSNorm(LlmcLlamaRMSNorm):
    def __init__(self, weight, eps=1e-6):
        super().__init__(weight, eps)

    def __repr__(self):
        return f"LlmcQwen2RMSNorm()"


class LlmcMixtralRMSNorm(LlmcLlamaRMSNorm):
    def __init__(self, weight, eps=1e-6):
        super().__init__(weight, eps)

    def __repr__(self):
        return f"LlmcMixtralRMSNorm()"


class LlmcMistralRMSNorm(LlmcLlamaRMSNorm):
    def __init__(self, weight, eps=1e-6):
        super().__init__(weight, eps)

    def __repr__(self):
        return f"LlmcMistralRMSNorm()"


class LlmcInternLM2RMSNorm(LlmcLlamaRMSNorm):
    def __init__(self, weight, eps=1e-6):
        super().__init__(weight, eps)

    def __repr__(self):
        return f"LlmcInternLM2RMSNorm()"


class OriginFloatLinear(nn.Module):
    def __init__(self, weight, bias, ori_module):
        super().__init__()
        self.register_buffer("weight", weight)
        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None

        for name, buf in ori_module.named_buffers():
            if name.startswith("buf_"):
                self.register_buffer(name, buf.data)
        if hasattr(self, "buf_rotate") and self.buf_rotate:
            self.rotater = ori_module.rotater

    @torch.no_grad()
    def forward(self, x):
        if hasattr(self, "buf_rotate") and self.buf_rotate:
            x = self.rotater.rotate(x)
        x = torch.functional.F.linear(x, self.weight, self.bias)
        return x

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
        return f"OriginFloatLinear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})"


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
        self.register_buffer("weight", weight)
        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None

        for name, buf in ori_module.named_buffers():
            if name.startswith("buf_"):
                self.register_buffer(name, buf.data)

        self.rotater = Rotater(
            online_full_had, online_partial_had, fp32_had, K, had_K, had_dim
        )
        self.register_buffer("buf_rotate", torch.tensor(True))

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
        return f"RotateLinear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, online_rotate={self.buf_rotate})"


class FakeQuantLinear(nn.Module):
    def __init__(self, weight, bias, ori_module, w_qdq, a_qdq):
        super().__init__()
        self.register_buffer("weight", weight)
        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None
        self.a_qdq = a_qdq
        self.w_qdq = w_qdq

        for name, buf in ori_module.named_buffers():
            if name.startswith("buf_"):
                self.register_buffer(name, buf.data)

        if hasattr(self, "buf_rotate") and self.buf_rotate:
            self.rotater = ori_module.rotater
        else:
            self.buf_rotate = False

        self.dynamic_quant_weight = False
        self.dynamic_quant_tmp_weight = False

    def forward(self, x):
        if hasattr(self, "buf_rotate") and self.buf_rotate:
            x = self.rotater.rotate(x)

        if self.a_qdq is not None:
            x = self.a_qdq(x, self)

        if not hasattr(self, "tmp_weight"):
            tmp_weight = self.w_qdq(self)
            self.register_buffer("tmp_weight", tmp_weight, persistent=False)
            self.tmp_bias = self.bias

        elif self.dynamic_quant_weight:
            self.tmp_weight = self.w_qdq(self)
            self.tmp_bias = self.bias

        elif self.dynamic_quant_tmp_weight:
            self.tmp_weight = self.w_qdq(self)

        x = torch.functional.F.linear(x, self.tmp_weight, self.tmp_bias)

        return x

    @classmethod
    @torch.no_grad()
    def new(cls, module, w_qdq, a_qdq):
        weight = module.weight.data
        if module.bias is not None:
            bias = module.bias.data
        else:
            bias = None

        new_module = cls(weight, bias, ori_module=module, w_qdq=w_qdq, a_qdq=a_qdq)

        new_module.in_features = module.in_features
        new_module.out_features = module.out_features
        new_module.w_qdq_name = cls.get_func_name(w_qdq)
        new_module.a_qdq_name = (
            cls.get_func_name(a_qdq) if a_qdq is not None else "None"
        )
        return new_module

    @classmethod
    def get_func_name(cls, any_callable):
        if isinstance(any_callable, partial):
            return any_callable.func.__name__
        return any_callable.__name__

    def register_activation_parameters(self, named_parameters):
        pass

    def __repr__(self):
        return f"FakeQuantLinear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, weight_quant={self.w_qdq_name}, act_quant={self.a_qdq_name}, online_rotate={self.buf_rotate})"


class EffcientFakeQuantLinear(nn.Module):
    def __init__(self, weight, bias, ori_module, a_qdq):
        super().__init__()
        self.register_buffer("weight", weight)
        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None
        self.a_qdq = a_qdq

        for name, buf in ori_module.named_buffers():
            if name.startswith("buf_"):
                self.register_buffer(name, buf.data)

        if hasattr(self, "buf_rotate") and self.buf_rotate:
            self.rotater = ori_module.rotater
        else:
            self.buf_rotate = False

    @torch.no_grad()
    def forward(self, x):
        if hasattr(self, "buf_rotate") and self.buf_rotate:
            x = self.rotater.rotate(x)

        if self.a_qdq is not None:
            x = self.a_qdq(x, self)
        x = torch.functional.F.linear(x, self.weight, self.bias)
        return x

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
            cls.get_func_name(a_qdq) if a_qdq is not None else "None"
        )
        new_module.debug_print = debug_print
        return new_module

    @classmethod
    def get_func_name(cls, any_callable):
        if isinstance(any_callable, partial):
            return any_callable.func.__name__
        return any_callable.__name__

    def __repr__(self):
        return f"EffcientFakeQuantLinear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, weight_quant={self.w_qdq_name}, act_quant={self.a_qdq_name}, online_rotate={self.buf_rotate}, debug_print={self.debug_print})"


class RealQuantLinear(nn.Module):
    def __init__(self, weight, bias, scales, zeros):
        super().__init__()
        self.register_buffer("weight", weight)
        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None
        self.register_buffer("scales", scales)

        if zeros is not None:
            self.register_buffer("zeros", zeros)
        else:
            self.zero = None

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
        weight, scales, zeros = w_q(module)
        weight, scales, zeros = cls.pack(weight, scales, zeros, quant_config)
        return weight, scales, zeros

    @classmethod
    @torch.no_grad()
    def pack(self, weight, scales, zeros, quant_config):
        if quant_config["weight"]["bit"] == 8:
            if zeros is not None:
                zeros = zeros.view(weight.shape[0], -1)
            scales = scales.view(weight.shape[0], -1)
            return weight, scales, zeros

        h1, h2 = weight.shape
        # pack 8 int4 in an int32 number, pack 16 int2 in an int32 number.
        bit = quant_config["weight"]["bit"]
        tmp = 32 // bit

        if (
            quant_config["weight"]["group_size"] != -1
            and quant_config["weight"]["granularity"] == "per_group"
        ):
            group_size = quant_config["weight"]["group_size"]
        else:
            group_size = h2

        assert h1 % tmp == 0 and h2 % tmp == 0, "H1 {} H2 {}".format(h1, h2)
        assert h2 % group_size == 0, "H1 {} H2 {}".format(h1, h2)

        weight = weight.cuda()
        int_weight = torch.empty(h1, h2 // tmp).to(torch.int32).cuda()
        # Weight pack in row.
        for pack in range(0, h2, tmp):
            for i in range(tmp):
                int_weight[:, pack // tmp] += weight[:, pack + i] << (i * bit)
        weight = weight.cpu()
        int_weight = int_weight.cpu()
        del weight

        if zeros is not None:
            zeros = zeros.cuda()
            int_zeros = torch.zeros(h1 // tmp, h2 // group_size).to(torch.int32).cuda()
            zeros = zeros.view(h1, -1)
            # zero point pack in col.
            for pack in range(0, h1, tmp):
                for i in range(tmp):
                    int_zeros[pack // tmp, :] += zeros[pack + i, :] << (i * bit)
            zeros = zeros.cpu()
            int_zeros = int_zeros.cpu()
            del zeros
        else:
            int_zeros = None

        gc.collect()
        torch.cuda.empty_cache()

        scales = scales.view(h1, -1)
        return int_weight, scales, int_zeros

    def __repr__(self):
        return (
            f"RealQuantLinear("
            + f"in_features={self.in_features}, "
            + f"out_features={self.out_features}, "
            + f"bias={self.bias is not None}, "
            + f"weight_shape={self.weight_shape}, "
            + f"weight_dtype={self.weight_dtype}, "
            + f"scales_shape={self.scales_shape}, "
            + f"scales_dtype={self.scales_dtype}, "
            + f"zeros_shape={self.zeros_shape}, "
            + f"zeros_dtype={self.zeros_dtype})"
        )


_TRANSFORMERS_LN_TYPES_ = ALL_LAYERNORM_LAYERS + [
    MistralRMSNorm,
    MixtralRMSNorm,
    Qwen2RMSNorm,
    LlamaRMSNorm,
    nn.LayerNorm,
]
_TRANSFORMERS_LINEAR_TYPES_ = [nn.Linear]

_MODEL_LN_TYPES_PAIRS_ = {
    "Llama": LlmcLlamaRMSNorm,
    "Llava": LlmcLlamaRMSNorm,
    "Mistral": LlmcMistralRMSNorm,
    "Mixtral": LlmcMixtralRMSNorm,
    "Interlm2": LlmcInternLM2RMSNorm,
    "Qwen2": LlmcQwen2RMSNorm,
    "Starcoder": LlmcLayerNorm,
    "Opt": LlmcLayerNorm,
    "Bloom": LlmcLayerNorm,
}


_LLMC_LN_TYPES_ = [
    LlmcLayerNorm,
    LlmcLlamaRMSNorm,
    LlmcRMSNorm,
    LlmcQwen2RMSNorm,
    LlmcMistralRMSNorm,
    LlmcMixtralRMSNorm,
    LlmcInternLM2RMSNorm,
]


_LLMC_LINEAR_TYPES_ = [
    OriginFloatLinear,
    RotateLinear,
    FakeQuantLinear,
    EffcientFakeQuantLinear,
    RealQuantLinear,
]
