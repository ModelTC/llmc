import math
import types
from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers.models.clip.modeling_clip import CLIPEncoderLayer
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLVisionBlock

from llmc.utils.registry_factory import TOKEN_REDUCTION_REGISTRY

from .token_reduction_module import TokenReductionModule


@TOKEN_REDUCTION_REGISTRY.register('ToMe')
class ToMe(TokenReductionModule):
    def __init__(self, config, model, blocks):
        super().__init__(config, model, blocks)
        self.add_sparse_config()
        self.patch_layer()

    def add_sparse_config(self):
        special_config = self.config.get('special', {})
        r_param = special_config.get('r', 0)
        if isinstance(r_param, int) or isinstance(r_param, float):
            self.r = [max(int(r_param), 0)] * len(self.blocks)
        elif isinstance(r_param, (tuple, list)):
            if len(r_param) == 2:
                start_r, step_r = r_param
                self.r = [max(int(start_r + i * step_r), 0) for i in range(len(self.blocks))]
            else:
                self.r = [0] * len(self.blocks)
                for i, val in enumerate(r_param):
                    if i < len(self.blocks):
                        self.r[i] = max(int(val), 0)
        else:
            raise ValueError('Invalid r format. Expected int or (start, step) tuple.')

        self.model.model.parameters = special_config

    def patch_layer(self):
        for idx, block in enumerate(self.blocks):
            if self.r[idx] > 0:
                block.r = self.r[idx]
                if isinstance(block, CLIPEncoderLayer):  # llava
                    block.self_attn.original_forward = block.self_attn.forward
                    block.self_attn.forward = types.MethodType(
                        tome_CLIPSdpaAttention_forward,
                        block.self_attn
                    )
                    block.original_forward = block.forward
                    block.forward = types.MethodType(
                        tome_CLIPEncoderLayer_forward,
                        block
                    )
                # elif isinstance(block, Qwen2VLVisionBlock): # qwenvl
                #     block.self_attn.original_forward = block.self_attn.forward
                #     block.self_attn.forward = types.MethodType(
                #         tome_VisionSdpaAttention_forward,
                #         block.self_attn
                #     )
                #     block.original_forward = block.forward
                #     block.forward = types.MethodType(
                #         tome_Qwen2VLVisionBlock_forward,
                #         block
                #     )
                # else:   # intervl2   token 剪枝数量有要求
                #     block.attn.original_naive_attn_forward = block.attn._naive_attn
                #     block.attn._naive_attn  = types.MethodType(tome_naive_attn, block.attn)
                #     block.attn.original_flash_attn_forward = block.attn._flash_attn
                #     block.attn._flash_attn  = types.MethodType(tome_flash_attn, block.attn)
                #     block.attn.original_forward = block.attn.forward
                #     block.attn.forward = types.MethodType(
                #         tome_InternAttention_forward,
                #         block.attn
                #     )
                #     block.original_forward = block.forward
                #     block.forward = types.MethodType(
                #         tome_InternVisionEncoderLayer_forward,
                #         block
                #     )


def do_nothing(x, mode=None):
    return x


def bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
    class_token: bool = False,
    distill_token: bool = False,
) -> Tuple[Callable, Callable]:
    """Applies ToMe with a balanced matching set (50%, 50%).

    Input size is [batch, tokens, channels].
    r indicates the number of tokens to remove (max 50% of tokens).

    Extra args:
     - class_token: Whether or not there's a class token.
     - distill_token: Whether or not there's also a distillation token.

    When enabled, the class token and distillation tokens won't get merged.
    """
    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1

    # We can only reduce by a maximum of 50% tokens
    t = metric.shape[1]
    r = min(r, (t - protected) // 2)

    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)

        if class_token:
            scores[..., 0, :] = -math.inf
        if distill_token:
            scores[..., :, 0] = -math.inf

        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

        if class_token:
            # Sort to ensure the class token is at the start
            unm_idx = unm_idx.sort(dim=1)[0]

    def merge(x: torch.Tensor, mode='mean') -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        if distill_token:
            return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
        else:
            return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

        out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)

        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)

        return out

    return merge, unmerge


def merge_wavg(
    merge: Callable, x: torch.Tensor, size: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies the merge function by taking a weighted average based on token
    size.

    Returns the merged tensor and the new token sizes.
    """
    if size is None:
        size = torch.ones_like(x[..., 0, None])

    x = merge(x * size, mode='sum')
    size = merge(size, mode='sum')

    x = x / size
    return x, size


def tome_CLIPSdpaAttention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    causal_attention_mask: Optional[torch.Tensor] = None,
    output_attentions: Optional[bool] = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

    from packaging import version
    parsed_torch_version_base = version.parse(version.parse(torch.__version__).base_version)
    is_torch_greater_or_equal_than_2_2 = parsed_torch_version_base >= version.parse('2.2')

    if output_attentions:
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )

    # CLIP text model uses both `causal_attention_mask` and `attention_mask`
    if attention_mask is not None and causal_attention_mask is not None:
        attn_mask = attention_mask + causal_attention_mask
    elif causal_attention_mask is not None:
        attn_mask = causal_attention_mask
    else:
        attn_mask = attention_mask

    bsz, tgt_len, embed_dim = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)

    if all([
        not is_torch_greater_or_equal_than_2_2,
        query_states.device.type == 'cuda',
        attn_mask is not None
    ]):
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    # CLIP text model uses both `causal_attention_mask` and `attention_mask` sequentially.
    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=attn_mask,
        dropout_p=self.dropout if self.training else 0.0,
        scale=self.scale,
    )

    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

    attn_output = self.out_proj(attn_output)

    return attn_output, None, key_states.mean(1)


def tome_CLIPEncoderLayer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    causal_attention_mask: torch.Tensor,
    output_attentions: Optional[bool] = False,
) -> Tuple[torch.FloatTensor]:

    residual = hidden_states

    hidden_states = self.layer_norm1(hidden_states)
    hidden_states, attn_weights, key_mean = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        causal_attention_mask=causal_attention_mask,
        output_attentions=output_attentions,
    )
    hidden_states = residual + hidden_states

    # ToMe
    merge, _ = bipartite_soft_matching(
        key_mean,
        self.r,
        True
    )
    hidden_states, _ = merge_wavg(merge, hidden_states)

    residual = hidden_states
    hidden_states = self.layer_norm2(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (attn_weights,)

    return outputs


# def tome_VisionSdpaAttention_forward(
#     self, hidden_states: torch.Tensor,
#     cu_seqlens: torch.Tensor,
#     rotary_pos_emb: torch.Tensor = None
# ) -> torch.Tensor:
#     from transformers.models.qwen2_vl.modeling_qwen2_vl import \
#         apply_rotary_pos_emb_vision
#     seq_length = hidden_states.shape[0]
#     q, k, v = self.qkv(hidden_states) \
#         .reshape(seq_length, 3, self.num_heads, -1) \
#         .permute(1, 0, 2, 3) \
#         .unbind(0)
#     q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
#     k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)

#     attention_mask = torch.zeros([1, seq_length, seq_length], device=q.device, dtype=torch.bool)
#     for i in range(1, len(cu_seqlens)):
#         start, end = cu_seqlens[i - 1], cu_seqlens[i]
#         attention_mask[..., start:end, start:end] = True
#     q = q.transpose(0, 1)
#     k = k.transpose(0, 1)
#     v = v.transpose(0, 1)
#     attn_output = F.scaled_dot_product_attention(q, k, v, attention_mask, dropout_p=0.0)
#     attn_output = attn_output.transpose(0, 1)
#     attn_output = attn_output.reshape(seq_length, -1)
#     attn_output = self.proj(attn_output)
#     return attn_output, k.mean(1)


# def tome_Qwen2VLVisionBlock_forward(
#     self, hidden_states, cu_seqlens, rotary_pos_emb
# ) -> torch.Tensor:
#     residual = hidden_states
#     hidden_states, key_mean = self.attn(
#         self.norm1(hidden_states), cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb
#     )
#     hidden_states = residual + hidden_states

#     # ToMe
#     merge, _ = bipartite_soft_matching(
#         key_mean,
#         self.r,
#         True
#     )
#     hidden_states, _ = merge_wavg(merge, hidden_states)
#     hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
#     return hidden_states

# def tome_naive_attn(self, x):
#     B, N, C = x.shape
#     qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#     q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

#     if self.qk_normalization:
#         B_, H_, N_, D_ = q.shape
#         q = self.q_norm(q.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)
#         k = self.k_norm(k.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)

#     attn = ((q * self.scale) @ k.transpose(-2, -1))
#     attn = attn.softmax(dim=-1)
#     attn = self.attn_drop(attn)

#     x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#     x = self.proj(x)
#     x = self.proj_drop(x)
#     return x, k.mean(1)

# def tome_flash_attn(self, x, key_padding_mask=None, need_weights=False):
#     from einops import rearrange
#     qkv = self.qkv(x)
#     qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.num_heads)

#     if self.qk_normalization:
#         q, k, v = qkv.unbind(2)
#         q = self.q_norm(q.flatten(-2, -1)).view(q.shape)
#         k = self.k_norm(k.flatten(-2, -1)).view(k.shape)
#         qkv = torch.stack([q, k, v], dim=2)

#     context, _ = self.inner_attn(
#         qkv, key_padding_mask=key_padding_mask, need_weights=need_weights, causal=False
#     )
#     outs = self.proj(rearrange(context, 'b s h d -> b s (h d)'))
#     outs = self.proj_drop(outs)
#     return outs, k.mean(1)


# def tome_InternAttention_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
#     if self.use_flash_attn:
#         x, key_mean = self._flash_attn(hidden_states)
#     else:
#         x, key_mean = self._naive_attn(hidden_states)
#     return x, key_mean


# def tome_InternVisionEncoderLayer_forward(
#         self,
#         hidden_states: torch.Tensor,
# ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor], Optional[Tuple[torch.FloatTensor]]]:

#     residual = hidden_states
#     x_attn, key_mean = self.attn(self.norm1(hidden_states))
#     hidden_states = residual + self.drop_path1(x_attn * self.ls1)

#     merge, _ = bipartite_soft_matching(
#         key_mean,
#         self.r,
#         True
#     )
#     hidden_states, _ = merge_wavg(merge, hidden_states)

#     residual = hidden_states
#     hidden_states = self.drop_path2(self.mlp(self.norm2(hidden_states)) * self.ls2)
#     hidden_states = residual + hidden_states

#     return hidden_states
