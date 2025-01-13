import torch
import torch.nn as nn
from loguru import logger
from transformers.models.llama.modeling_llama import (apply_rotary_pos_emb,
                                                      repeat_kv)


def eager_attention_forward(
    module,
    query,
    key,
    value,
    attention_mask,
    scaling,
    dropout,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class ShadowKVAttention(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.config = module.config
        self.layer_idx = module.layer_idx
        self.head_dim = module.head_dim
        self.num_key_value_groups = module.num_key_value_groups
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = module.attention_dropout
        self.is_causal = True

        self.q_proj = module.q_proj
        self.k_proj = module.k_proj
        self.v_proj = module.v_proj
        self.o_proj = module.o_proj

    def forward(
        self,
        hidden_states,
        position_embeddings,
        position_ids,
        attention_mask,
        past_key_value,
        output_attentions,
        use_cache,
        cache_position,
        retrieval_position_ids=None,
        cos_sin_cache=None,
        **kwargs,
    ):

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        if past_key_value is not None and past_key_value.prefill:
            past_key_value.get_svd(key_states, layer_idx=self.layer_idx)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            key_states, value_states = \
                past_key_value.update(key_states,
                                      value_states,
                                      self.layer_idx,
                                      retrieval_position_ids,
                                      cos_sin_cache)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2)
        # bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == 'cuda' and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels
        # via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options.
        # An inline conditional prevents dynamic shapes from compiling.
        is_causal = True if causal_mask is None and q_len > 1 else False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value

    @classmethod
    @torch.no_grad()
    def new(cls, module):
        new_module = cls(module)
        return new_module

    def __repr__(self):
        return (
            f'ShadowKVAttention(\n'
            f'  (q_proj): {self.q_proj}\n'
            f'  (k_proj): {self.k_proj}\n'
            f'  (v_proj): {self.v_proj}\n'
            f'  (o_proj): {self.o_proj}\n'
            f'  (kvcache): {self.kvcache}\n'
            f')'
        )


_LLMC_ATTN_MAP_ = {'ShadowKV': {'Llama': ShadowKVAttention}}
