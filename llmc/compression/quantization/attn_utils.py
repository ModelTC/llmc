import math

import torch
import torch.nn as nn


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


_LLMC_ATTN_MAP_ = {'Vit': LlmcViTSelfAttention, 'DeepseekV2': LlmcDeepseekAttention}
