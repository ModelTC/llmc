import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

try:
    import fast_hadamard_transform

    from .hadamard_utils import matmul_hadU_cuda
except Exception:
    logger.warning(
        'fast_hadamard_transform not installed. '
        'If you need it, please install it firstly.'
    )

from .utils import calculate_zeros_width


class LlmcMatmul(nn.Module):
    def __init__(self, a1_qdq=None, a2_qdq=None):
        super().__init__()
        self.a1_qdq = a1_qdq
        self.a2_qdq = a2_qdq
        self.calib = True

    def forward(self, x1, x2):
        if self.a1_qdq is not None and not self.calib:
            x1 = self.a1_qdq(x1, self)
        if self.a2_qdq is not None and not self.calib:
            x2 = self.a2_qdq(x2, self)
        out = torch.matmul(x1, x2)
        return out

    def __repr__(self):
        return f'LlmcMatmul(calib={self.calib})'


class LlmcSoftmax(nn.Module):
    def __init__(self, a_qdq=None):
        super().__init__()
        self.a_qdq = a_qdq
        self.calib = True

    def forward(self, x, dim=-1, dtype=None):
        if self.a_qdq is not None and not self.calib:
            x = self.a_qdq(x, self)
        out = nn.functional.softmax(x, dim=dim, dtype=dtype)
        return out

    def __repr__(self):
        return f'LlmcSoftmax(calib={self.calib})'


class LlmcViTSelfAttention(nn.Module):
    def __init__(
        self,
        query,
        key,
        value,
        num_attention_heads,
        attention_head_size,
        all_head_size,
        dropout,
        matmul_a1_qdq,
        matmul_a2_qdq,
        softmax_a_qdq,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = attention_head_size
        self.all_head_size = all_head_size
        self.query = query
        self.key = key
        self.value = value

        self.dropout = dropout

        self.matmul_1 = LlmcMatmul(matmul_a1_qdq, matmul_a2_qdq)
        self.matmul_2 = LlmcMatmul(matmul_a1_qdq, matmul_a2_qdq)
        self.softmax = LlmcSoftmax(softmax_a_qdq)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, head_mask=None, output_attentions=False):
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        attention_scores = self.matmul_1(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = self.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = self.matmul_2(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )

        return outputs

    @classmethod
    @torch.no_grad()
    def new(cls, module, matmul_a1_qdq=None, matmul_a2_qdq=None, softmax_a_qdq=None):
        query, key, value = module.query, module.key, module.value
        num_attention_heads = module.num_attention_heads
        attention_head_size = module.attention_head_size
        all_head_size = module.all_head_size
        dropout = module.dropout
        new_module = cls(
            query,
            key,
            value,
            num_attention_heads,
            attention_head_size,
            all_head_size,
            dropout,
            matmul_a1_qdq,
            matmul_a2_qdq,
            softmax_a_qdq,
        )
        return new_module

    def __repr__(self):
        return (
            f'LlmcViTSelfAttention(\n'
            f'  (query): {self.query}\n'
            f'  (key): {self.key}\n'
            f'  (value): {self.value}\n'
            f'  (dropout): {self.dropout}\n'
            f'  (matmul_1): {self.matmul_1}\n'
            f'  (matmul_2): {self.matmul_2}\n'
            f'  (softmax): {self.softmax}\n'
            f')'
        )


class LlmcDeepseekAttention(nn.Module):
    def __init__(
        self,
        config,
        layer_idx,
        attention_dropout,
        hidden_size,
        num_heads,
        max_position_embeddings,
        rope_theta,
        q_lora_rank,
        qk_rope_head_dim,
        kv_lora_rank,
        v_head_dim,
        qk_nope_head_dim,
        q_head_dim,
        is_causal,
        q_proj,
        q_a_proj,
        q_a_layernorm,
        q_b_proj,
        kv_a_proj_with_mqa,
        kv_a_layernorm,
        kv_b_proj,
        o_proj,
        rotary_emb,
        softmax_scale,
        matmul_a1_qdq,
        matmul_a2_qdq,
        softmax_a_qdq,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attention_dropout = attention_dropout
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.q_head_dim = q_head_dim
        self.is_causal = is_causal
        self.q_proj = q_proj
        self.q_a_proj = q_a_proj
        self.q_a_layernorm = q_a_layernorm
        self.q_b_proj = q_b_proj
        self.kv_a_proj_with_mqa = kv_a_proj_with_mqa
        self.kv_a_layernorm = kv_a_layernorm
        self.kv_b_proj = kv_b_proj
        self.o_proj = o_proj
        self.rotary_emb = rotary_emb
        self.softmax_scale = softmax_scale
        self.matmul_1 = LlmcMatmul(matmul_a1_qdq, matmul_a2_qdq)
        self.matmul_2 = LlmcMatmul(matmul_a1_qdq, matmul_a2_qdq)
        self.softmax = LlmcSoftmax(softmax_a_qdq)

    def _shape(self, tensor, seq_len, bsz):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.v_head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, q, k, cos, sin, position_ids, unsqueeze_dim=1):
        cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        sin = sin[position_ids].unsqueeze(unsqueeze_dim)

        b, h, s, d = q.shape
        q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

        b, h, s, d = k.shape
        k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed

    @classmethod
    @torch.no_grad()
    def new(cls, module, matmul_a1_qdq=None, matmul_a2_qdq=None, softmax_a_qdq=None):

        config = module.config
        layer_idx = module.layer_idx

        attention_dropout = module.config.attention_dropout
        hidden_size = module.config.hidden_size
        num_heads = module.config.num_attention_heads

        max_position_embeddings = module.config.max_position_embeddings
        rope_theta = module.config.rope_theta
        q_lora_rank = module.config.q_lora_rank
        qk_rope_head_dim = module.config.qk_rope_head_dim
        kv_lora_rank = module.config.kv_lora_rank
        v_head_dim = module.config.v_head_dim
        qk_nope_head_dim = module.config.qk_nope_head_dim
        q_head_dim = module.q_head_dim
        is_causal = module.is_causal

        if q_lora_rank is None:
            q_proj = module.q_proj
            q_a_proj = None
            q_a_layernorm = None
            q_b_proj = None
        else:
            q_proj = None
            q_a_proj = module.q_a_proj
            q_a_layernorm = module.q_a_layernorm
            q_b_proj = module.q_b_proj

        kv_a_proj_with_mqa = module.kv_a_proj_with_mqa
        kv_a_layernorm = module.kv_a_layernorm
        kv_b_proj = module.kv_b_proj

        o_proj = module.o_proj
        rotary_emb = module.rotary_emb

        softmax_scale = module.softmax_scale

        new_module = cls(
            config=config,
            layer_idx=layer_idx,
            attention_dropout=attention_dropout,
            hidden_size=hidden_size,
            num_heads=num_heads,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            q_lora_rank=q_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            kv_lora_rank=kv_lora_rank,
            v_head_dim=v_head_dim,
            qk_nope_head_dim=qk_nope_head_dim,
            q_head_dim=q_head_dim,
            is_causal=is_causal,
            q_proj=q_proj,
            q_a_proj=q_a_proj,
            q_a_layernorm=q_a_layernorm,
            q_b_proj=q_b_proj,
            kv_a_proj_with_mqa=kv_a_proj_with_mqa,
            kv_a_layernorm=kv_a_layernorm,
            kv_b_proj=kv_b_proj,
            o_proj=o_proj,
            rotary_emb=rotary_emb,
            softmax_scale=softmax_scale,
            matmul_a1_qdq=matmul_a1_qdq,
            matmul_a2_qdq=matmul_a2_qdq,
            softmax_a_qdq=softmax_a_qdq,
        )

        return new_module

    def forward(
        self,
        hidden_states,
        attention_mask,
        position_ids,
        past_key_value,
        output_attentions,
        use_cache,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()
        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))

        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        kv = (
            self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
            .view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            .transpose(1, 2)
        )

        k_nope, value_states = torch.split(
            kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )
        kv_seq_len = value_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        q_pe, k_pe = self.apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

        query_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
        query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
        query_states[:, :, :, self.qk_nope_head_dim:] = q_pe

        key_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
        key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
        key_states[:, :, :, self.qk_nope_head_dim:] = k_pe
        if past_key_value is not None:
            cache_kwargs = {'sin': sin, 'cos': cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        attn_weights = (
            self.matmul_1(query_states, key_states.transpose(2, 3)) * self.softmax_scale
        )

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f'Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)},'
                f'but is {attn_weights.size()}'
            )
        assert attention_mask is not None
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f'Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)},'
                    f'but is {attention_mask.size()}'
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = self.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        attn_output = self.matmul_2(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.v_head_dim):
            raise ValueError(
                f'`attn_output` should be of size {(bsz, self.num_heads, q_len, self.v_head_dim)},'
                f' but is {attn_output.size()}'
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


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

    @torch.no_grad()
    def forward(self, x, dtype=None):
        if hasattr(self, 'buf_rotate') and self.buf_rotate:
            x = self.rotater.rotate(x)

        org_dtype = self.weight.data.dtype
        if dtype is not None:
            self.convert_dtype(dtype)

        x = torch.functional.F.linear(x, self.weight, self.bias)
        self.convert_dtype(org_dtype)
        return x

    def convert_dtype(self, dtype):
        self.weight.data = self.weight.data.to(dtype)
        if self.bias is not None:
            self.bias.data = self.bias.data.to(dtype)

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

        self.dynamic_quant_weight = False
        self.dynamic_quant_tmp_weight = False

    def forward(self, x, dtype=None):
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

        org_dtype = self.tmp_weight.data.dtype
        if dtype is not None:
            self.convert_dtype(dtype)

        x = torch.functional.F.linear(x, self.tmp_weight, self.tmp_bias)

        self.convert_dtype(org_dtype)
        return x

    def convert_dtype(self, dtype):
        self.tmp_weight.data = self.tmp_weight.data.to(dtype)
        if self.tmp_bias is not None:
            self.tmp_bias.data = self.tmp_bias.data.to(dtype)

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

    @torch.no_grad()
    def forward(self, x, dtype=None):
        if hasattr(self, 'buf_rotate') and self.buf_rotate:
            x = self.rotater.rotate(x)

        if self.a_qdq is not None:
            x = self.a_qdq(x, self)

        org_dtype = self.weight.data.dtype
        if dtype is not None:
            self.convert_dtype(dtype)

        x = torch.functional.F.linear(x, self.weight, self.bias)
        self.convert_dtype(org_dtype)
        return x

    def convert_dtype(self, dtype):
        self.weight.data = self.weight.data.to(dtype)
        if self.bias is not None:
            self.bias.data = self.bias.data.to(dtype)

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
            f'debug_print={self.debug_print})'
        )


class VllmRealQuantLinear(nn.Module):
    def __init__(self, weight, bias, scales, input_scale, need_pack):
        super().__init__()
        weight_name = 'weight_packed' if need_pack else 'weight'
        self.register_buffer(weight_name, weight)

        (
            self.register_buffer('bias', bias)
            if bias is not None
            else setattr(self, 'bias', None)
        )

        self.register_buffer('weight_scale', scales)
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
        if module.bias is not None:
            bias = module.bias.data
        else:
            bias = None

        need_pack = quant_config['weight'].get('need_pack', False)
        new_module = cls(weight, bias, scales, input_scale, need_pack)
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
        weight, scales, zeros = w_q(module)
        need_pack = quant_config['weight'].get('need_pack', False)
        if need_pack:
            weight, scales = cls.pack(weight, scales, quant_config)
        return weight, scales

    @classmethod
    @torch.no_grad()
    def pack(self, weight, scales, quant_config):

        # Packs a tensor of quantized weights stored in int8 into int32s with padding
        scales = scales.to(torch.float16)
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
        int_weight = torch.from_numpy(packed)
        return int_weight, scales

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


class SglRealQuantLinear(VllmRealQuantLinear):
    def __init__(self, weight, bias, scales, input_scale, need_pack):
        super().__init__(weight, bias, scales, need_pack)

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
    OriginFloatLinear,
    RotateLinear,
    FakeQuantLinear,
    EffcientFakeQuantLinear,
    VllmRealQuantLinear,
    SglRealQuantLinear,
    AutoawqRealQuantLinear,
    MlcllmRealQuantLinear,
]

_LLMC_ATTN_MAP_ = {'Vit': LlmcViTSelfAttention, 'DeepseekV2': LlmcDeepseekAttention}

_REALQUANT_LINEAR_MAP_ = {
    'vllm_quant': VllmRealQuantLinear,
    'sgl_quant': SglRealQuantLinear,
    'autoawq_quant': AutoawqRealQuantLinear,
    'mlcllm_quant': MlcllmRealQuantLinear,
}
