import gc
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers.models.mistral.modeling_mistral import MistralRMSNorm
from transformers.models.mixtral.modeling_mixtral import MixtralRMSNorm
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS


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
        return (
            f"LlmcLayerNorm({self.normalized_shape},"
            f"eps={self.eps},"
            f"elementwise_affine={self.elementwise_affine})"
        )


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
        return "LlmcLlamaRMSNorm()"


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
        return "LlmcRMSNorm()"


class LlmcQwen2RMSNorm(LlmcLlamaRMSNorm):
    def __init__(self, weight, eps=1e-6):
        super().__init__(weight, eps)

    def __repr__(self):
        return "LlmcQwen2RMSNorm()"


class LlmcMixtralRMSNorm(LlmcLlamaRMSNorm):
    def __init__(self, weight, eps=1e-6):
        super().__init__(weight, eps)

    def __repr__(self):
        return "LlmcMixtralRMSNorm()"


class LlmcMistralRMSNorm(LlmcLlamaRMSNorm):
    def __init__(self, weight, eps=1e-6):
        super().__init__(weight, eps)

    def __repr__(self):
        return "LlmcMistralRMSNorm()"


class LlmcInternLM2RMSNorm(LlmcLlamaRMSNorm):
    def __init__(self, weight, eps=1e-6):
        super().__init__(weight, eps)

    def __repr__(self):
        return "LlmcInternLM2RMSNorm()"


class OriginEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx,
                 max_norm, norm_type, scale_grad_by_freq,
                 sparse, weight):
        super(OriginEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        self.weight = weight

    def forward(self, input):
        return F.embedding(
            input, self.weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)
        
    @classmethod
    @torch.no_grad()
    def new(cls, module):

        num_embeddings = module.num_embeddings
        embedding_dim = module.embedding_dim
        padding_idx = module.padding_idx
        max_norm = module.max_norm
        norm_type = module.norm_type
        scale_grad_by_freq = module.scale_grad_by_freq
        sparse = module.sparse
        weight = module.weight

        new_module = cls(num_embeddings, embedding_dim, padding_idx,
                 max_norm, norm_type, scale_grad_by_freq,
                 sparse, weight)
        return new_module

    def __repr__(self):
        return (
            f"OriginEmbedding({self.num_embeddings}, "
            f"{self.embedding_dim}, "
            f"padding_idx={self.padding_idx}),"
        )


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

        if getattr(self, "buf_a_rotate", False):
            self.a_rot = ori_module.a_rot

    @torch.no_grad()
    def forward(self, x):
        if hasattr(self, "a_rot"):
            x = self.a_rot(x, self)
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
        return (
            f"OriginFloatLinear(in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"buf_a_rotate={self.buf_a_rotate}, "
            f"bias={self.bias is not None})"
        )


class RotateEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx,
                 max_norm, norm_type, scale_grad_by_freq,
                 sparse, weight, w_rot):
        super(RotateEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        self.weight = weight
        self.bias = None
        self.w_rot = w_rot

    def forward(self, input):
    
        tmp_weight = self._rotate_weight()

        return F.embedding(
            input, tmp_weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)
    
    def _rotate_weight(self):
        if self.w_rot is not None:
            tmp_weight, _ = self.w_rot(self)
        else:
            tmp_weight = self.weight
        return tmp_weight
        
    @classmethod
    @torch.no_grad()
    def new(cls, module, w_rot):

        num_embeddings = module.num_embeddings
        embedding_dim = module.embedding_dim
        padding_idx = module.padding_idx
        max_norm = module.max_norm
        norm_type = module.norm_type
        scale_grad_by_freq = module.scale_grad_by_freq
        sparse = module.sparse
        weight = module.weight

        new_module = cls(num_embeddings, embedding_dim, padding_idx,
                 max_norm, norm_type, scale_grad_by_freq,
                 sparse, weight, w_rot)
        return new_module

    def __repr__(self):
        return (
            f"RotateEmbedding({self.num_embeddings}, "
            f"{self.embedding_dim}, "
            f"w_rotate={self.w_rot is not None}, "
            f"padding_idx={self.padding_idx})"
        )


class RotateLinear(nn.Module):
    def __init__(self, weight, bias, ori_module, w_rot, a_rot):
        super().__init__()
        self.register_buffer("weight", weight)
        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None

        for name, buf in ori_module.named_buffers():
            if name.startswith("buf_"):
                self.register_buffer(name, buf.data)

        self.w_rot = w_rot
        self.a_rot = a_rot

        self.register_buffer("buf_w_rotate", torch.tensor(w_rot is not None))
        self.register_buffer("buf_a_rotate", torch.tensor(a_rot is not None))

    def forward(self, x):

        if self.buf_a_rotate:
            x = self.a_rot(x, self)

        if self.buf_w_rotate:
            tmp_weight, tmp_bias = self._rotate_weight()
            self.register_buffer("tmp_weight", tmp_weight, persistent=False)
            self.register_buffer("tmp_bias", tmp_bias, persistent=False)

        weight = getattr(self, "tmp_weight", self.weight)
        bias = getattr(self, "tmp_bias", self.bias)
        x = torch.functional.F.linear(x, weight, bias)
        return x
    
    def _rotate_weight(self):
        tmp_weight, tmp_bias = self.w_rot(self)
        return tmp_weight, tmp_bias

    @classmethod
    @torch.no_grad()
    def new(cls, module, w_rot, a_rot):
        weight = module.weight.data
        if module.bias is not None:
            bias = module.bias.data
        else:
            bias = None

        new_module = cls(
            weight,
            bias,
            ori_module=module,
            w_rot=w_rot,
            a_rot=a_rot
        )
        new_module.in_features = module.in_features
        new_module.out_features = module.out_features
        return new_module

    @classmethod
    def get_func_name(cls, any_callable):
        if isinstance(any_callable, partial):
            return any_callable.func.__name__
        return any_callable.__name__

    def __repr__(self):
        return (
            f"RotateLinear(in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}, "
            f"w_rotate={self.buf_w_rotate}, "
            f"a_rotate={self.buf_a_rotate})"
        )


class FakeQuantLinear(nn.Module):
    def __init__(self, weight, bias, ori_module, w_qdq, a_qdq, w_rot, a_rot):
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

        if getattr(self, "buf_w_rotate", False):
            self.w_rot = w_rot
        if getattr(self, "buf_a_rotate", False):
            self.a_rot = a_rot

        self.dynamic_quant_weight = False
        self.dynamic_quant_tmp_weight = False

    def forward(self, x):
        if hasattr(self, "a_rot"):
            x = self.a_rot(x, self)

        if self.a_qdq is not None:
            x = self.a_qdq(x, self)

        if hasattr(self, "w_rot") and self.w_rot is not None:
            tmp_weight, tmp_bias = self._rotate_weight()
            self.register_buffer("tmp_weight", tmp_weight, persistent=False)
            self.register_buffer("tmp_bias", tmp_bias, persistent=False)
            self.tmp_weight = self.w_qdq(self)

        else:
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

    def _rotate_weight(self):
        tmp_weight, tmp_bias = self.w_rot(self)
        return tmp_weight, tmp_bias

    @classmethod
    @torch.no_grad()
    def new(cls, module, w_qdq, a_qdq):
        weight = module.weight.data
        if module.bias is not None:
            bias = module.bias.data
        else:
            bias = None
        

        new_module = cls(weight, bias, ori_module=module, w_qdq=w_qdq, a_qdq=a_qdq, w_rot=module.w_rot, a_rot=module.a_rot)

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
        return (
            f"FakeQuantLinear(in_features={self.in_features},"
            f"out_features={self.out_features}, bias={self.bias is not None},"
            f"weight_quant={self.w_qdq_name}, "
            f"act_quant={self.a_qdq_name}, "
            f"w_rotate={self.buf_w_rotate}, "
            f"a_rotate={self.buf_a_rotate},"
        )


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

        if getattr(self, "buf_a_rotate", False):
            self.a_rot = ori_module.a_rot

    @torch.no_grad()
    def forward(self, x):
        if hasattr(self, "a_rot"):
            x = self.a_rot(x, self)

        if self.a_qdq is not None:
            x = self.a_qdq(x, self)
        x = torch.functional.F.linear(x, self.weight, self.bias)
        return x

    @classmethod
    @torch.no_grad()
    def new(cls, module, w_qdq, a_qdq, debug_print={}):

        if hasattr(module, "w_rot") and module.w_rot is not None:
            weight, bias = module.w_rot(module)

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
        return (
            f"EffcientFakeQuantLinear(in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}, "
            f"weight_quant={self.w_qdq_name}, "
            f"act_quant={self.a_qdq_name}, "
            f"debug_print={self.debug_print})"
        )


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
            "RealQuantLinear("
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
