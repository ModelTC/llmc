import functools

import einops as ein
import torch

from llmc.utils.registry_factory import TOKEN_REDUCTION_REGISTRY

from .token_reduction_module import TokenReductionModule
from .utils import prefill_wrapper, prefill_wrapper_model


@TOKEN_REDUCTION_REGISTRY.register('SparseVLM')
class SparseVLM(TokenReductionModule):
    def __init__(self, config, model, blocks):
        super().__init__(config, model, blocks)
        self.add_sparse_config()
        self.register_reduction_modules()

    def add_sparse_config(self):
        special_config = self.config.get('special', {})

        self.pruning_loc = special_config.get('pruning_loc', [2, 6, 15])
        special_config['retained_tokens'] = special_config.get('retained_tokens', 192)
        special_config['init_token_total_shape'] = special_config.get('init_token_total_shape', 668)
        special_config['generate_process_count'] = 0
        special_config['pre_prompt_length_list'] = []
        special_config['token_length_list'] = []
        special_config['image_shape'] = self.model.pruning_config['image_token_length']
        special_config['image_token_index'] = self.model.pruning_config['image_token_index']
        self.model.model.parameters = special_config

    def register_reduction_modules(self):
        @prefill_wrapper
        def input_hook(module, input_args, pruning_pars):
            input_ids = input_args[0]
            pre_prompt_length_list = []
            token_length_list = []
            IMAGE_TOKEN_INDEX = pruning_pars['image_token_index']

            # find the position of the first image token
            for seq in input_ids:
                image_token_index = (
                    seq == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]
                if len(image_token_index) > 0:
                    pre_prompt_length_list.append(image_token_index[0].item())
                else:
                    pre_prompt_length_list.append(0)
                token_length_list.append(seq.shape[0])

            pruning_pars['pre_prompt_length_list'] = pre_prompt_length_list
            pruning_pars['token_length_list'] = token_length_list

            return input_args

        @prefill_wrapper_model
        def register_module_pars(module, args, kwargs, pruning_pars):
            pre_prompt_length_list = pruning_pars['pre_prompt_length_list']
            inputs_embeds = kwargs['inputs_embeds']
            if inputs_embeds is None:
                inputs_embeds = self.embed_tokens(kwargs['input_ids'])
            hidden_states = inputs_embeds  # shape: (B, L, C)

            pruning_pars['B'], L, _ = hidden_states.shape
            B = pruning_pars['B']
            init_n = pruning_pars['init_token_total_shape'] + \
                pruning_pars['generate_process_count']    # 668
            pruning_pars['prev_decision'] = torch.ones(
                B, init_n, 1, dtype=hidden_states.dtype, device=hidden_states.device)
            pruning_pars['policy'] = torch.ones(
                B, init_n, 1, dtype=hidden_states.dtype, device=hidden_states.device)

            pruning_pars['v_token_start'] = pre_prompt_length_list[0] if len(
                pre_prompt_length_list) != 0 else 0  # 35
            v_token_start = pruning_pars['v_token_start']
            pruning_pars['text_token_start'] = pruning_pars['v_token_start'] + \
                pruning_pars['image_shape']  # 35 + 576 = 611
            text_token_start = pruning_pars['text_token_start']
            pruning_pars['v_token_num'] = pruning_pars['image_shape']  # 576

            if (len(pre_prompt_length_list) != 0 and hidden_states.shape[1] != 1):
                v_t = hidden_states[:, v_token_start: text_token_start, :]
                t_t = hidden_states[:, text_token_start:, :]
                m_v_t = v_t @ t_t.transpose(1, 2)  # [1, 576, 53]
                m_v_t = m_v_t.softmax(2).mean(1)  # [1, 53]
                pruning_pars['t_token_idx'] = torch.where(m_v_t > m_v_t.mean())

            return args, kwargs

        def update_output_attentions_hook(module, args, kwargs, pruning_pars, layer_idx):
            kwargs['output_attentions'] = True
            if layer_idx != self.pruning_loc[0]:
                kwargs['position_ids'] = pruning_pars['position_ids']
                kwargs['cache_position'] = pruning_pars['cache_position']
                kwargs['position_embeddings'] = pruning_pars['position_embeddings']
            return args, kwargs

        @prefill_wrapper
        def decoder_attn_hook(module, inputs, kwargs, layer_outputs, pruning_pars, layer_idx):

            attn_logits = layer_outputs[1]
            v_token_start = pruning_pars['v_token_start']
            text_token_start = pruning_pars['text_token_start']
            t_token_idx = pruning_pars['t_token_idx']
            v_token_num = pruning_pars['v_token_num']
            retained_tokens = pruning_pars['retained_tokens']
            B = pruning_pars['B']
            pre_prompt_length_list = pruning_pars['pre_prompt_length_list']
            image_shape = pruning_pars['image_shape']
            if layer_idx == self.pruning_loc[0]:
                position_ids = kwargs['position_ids']
                pruning_pars['position_ids'] = position_ids
            else:
                position_ids = pruning_pars['position_ids']

            hidden_states = inputs[0]  # [B, L, D]
            pre_prompt_length_list = pruning_pars['pre_prompt_length_list']
            image_shape = pruning_pars['image_shape']

            pred_score_vis, s_flag, relation_vis_text = attn_postprocess_topk(
                attn_logits,
                v_token_start,
                v_token_num,
                text_token_start,
                t_token_idx,
                layer_idx,
                retained_tokens
            )

            policy = torch.ones(B, hidden_states.shape[1], dtype=hidden_states.dtype,
                                device=hidden_states.device)
            policy[:, v_token_start:text_token_start] = \
                pred_score_vis.type(dtype=hidden_states.dtype)

            for batch in range(len(pre_prompt_length_list)):
                # keep pre prompt
                prompt_length = pre_prompt_length_list[batch]
                policy[batch, :prompt_length] = 1
                # keep question
                text_token_start = prompt_length + image_shape
                policy[batch, text_token_start:] = 1

            total_sparse_token_idx = torch.where(policy == 0)[1].unsqueeze(0)

            # merge and cluster
            if s_flag and total_sparse_token_idx.shape[1] > 0:
                total_sparse_token_idx = torch.where(policy == 0)[1].unsqueeze(0)
                total_sparse_token = batch_index_select(layer_outputs[0], total_sparse_token_idx)

                merge_token_idx_stage1 = torch.where(pred_score_vis == 0)[1]
                merge_token_stage1 = relation_vis_text[0][merge_token_idx_stage1]
                merge_token_num_stage1 = int(merge_token_idx_stage1.shape[0] * 0.3) + 1  # Top 30%
                merge_token_stage2_idx = merge_token_stage1.topk(merge_token_num_stage1)[1]

                merge_token_stage2 = total_sparse_token[:, merge_token_stage2_idx, :]
                cluster_num = int(merge_token_stage2.shape[1] / 10) + 1
                if cluster_num == 0:
                    cluster_num = merge_token_stage2.shape[1]

                merge_sparse_token = cluster_and_merge(merge_token_stage2, cluster_num)

                select_token_idx = torch.where(policy == 1)[1].unsqueeze(0)
                select_token = batch_index_select(layer_outputs[0], select_token_idx)
                select_vis_token_num = pred_score_vis.sum()

                select_and_merge_token = torch.cat(
                    (
                        select_token[:, :v_token_start +
                                     select_vis_token_num, :],
                        merge_sparse_token,
                        select_token[:, v_token_start +
                                     select_vis_token_num:, :]
                    ),
                    dim=1
                )
                layer_outputs = (select_and_merge_token, layer_outputs[1])
                position_ids = position_ids[:, :len(select_token_idx[0]) + cluster_num]
                # prev_decision = policy
                v_token_num = pred_score_vis.sum() + cluster_num
                text_token_start = v_token_start + v_token_num
            else:
                select_token_idx = torch.where(policy == 1)[1].unsqueeze(0)
                layer_outputs = (batch_index_select(layer_outputs[0], select_token_idx),
                                 layer_outputs[1])
                position_ids = position_ids[:, :len(select_token_idx[0])]
                # prev_decision = policy
                v_token_num = pred_score_vis.sum()
                text_token_start = v_token_start + v_token_num

            new_output = layer_outputs
            # hidden_states = layer_outputs[0]
            cache_position = position_ids.detach().clone()

            pruning_pars['v_token_num'] = v_token_num
            pruning_pars['text_token_start'] = text_token_start
            pruning_pars['position_ids'] = position_ids
            pruning_pars['cache_position'] = cache_position
            pruning_pars['position_embeddings'] = None

            return new_output

        @prefill_wrapper
        def read_parameter_hook(module, args, kwargs, pruning_pars):
            kwargs['position_ids'] = pruning_pars['position_ids']
            kwargs['cache_position'] = pruning_pars['cache_position']
            kwargs['position_embeddings'] = pruning_pars['position_embeddings']

            return args, kwargs

        self.model.embed_tokens.register_forward_pre_hook(
            functools.partial(
                input_hook,
                pruning_pars=self.model.model.parameters
            )
        )

        self.model.model.register_forward_pre_hook(
            functools.partial(
                register_module_pars,
                pruning_pars=self.model.model.parameters),
            with_kwargs=True
        )

        sorted_pruning_locs = sorted(self.pruning_loc)
        total_layers = len(self.blocks)

        for block_idx in range(sorted_pruning_locs[0], total_layers):
            if block_idx in sorted_pruning_locs:
                self.blocks[block_idx].register_forward_pre_hook(
                    functools.partial(
                        update_output_attentions_hook,
                        pruning_pars=self.model.model.parameters,
                        layer_idx=block_idx,
                    ),
                    with_kwargs=True
                )
                self.blocks[block_idx].register_forward_hook(
                    functools.partial(
                        decoder_attn_hook,
                        pruning_pars=self.model.model.parameters,
                        layer_idx=block_idx,
                    ),
                    with_kwargs=True
                )
            else:
                self.blocks[block_idx].register_forward_pre_hook(
                    functools.partial(
                        read_parameter_hook,
                        pruning_pars=self.model.model.parameters
                    ),
                    with_kwargs=True
                )


layer_dict = {2: 0, 6: 1, 15: 2}

sparse_token_list_192 = [300, 200, 110]       # 2*576  4*300 10*200  16*110
sparse_token_list_128 = [303, 110, 36]
sparse_token_list_64 = [66, 30, 17]

sparse_token_dict = {
    192: sparse_token_list_192,
    128: sparse_token_list_128,
    64: sparse_token_list_64
}


def attn_postprocess_topk(
        self_attn_weights,
        v_token_start,
        v_token_num,
        text_token_start,
        t_token_idx,
        layer_idx,
        retained_tokens):
    '''
    self_attn_weights: [B, H, L, L]
    '''
    self_attn_weights = self_attn_weights.mean(1)  # B, L[Q], L[K]

    t_token_idx = t_token_idx[1] + text_token_start
    relation_vis_text = self_attn_weights[:, t_token_idx,
                                          v_token_start: v_token_start + v_token_num]  # B, L2, L1

    relation_vis_text = relation_vis_text.mean(1)  # B, L1

    relation_vis = relation_vis_text
    s_flag = True       # s_flag controls whether token merge is needed.

    sparse_token_list = sparse_token_dict[retained_tokens]

    if v_token_num != 0:
        mask = torch.zeros_like(relation_vis, dtype=bool)
        _, indices = torch.topk(relation_vis, min(
            sparse_token_list[layer_dict[layer_idx]], v_token_num - 1), dim=1)
        mask[0][indices] = 1
    else:
        mask = torch.ones_like(relation_vis_text, dtype=bool)
        s_flag = False
    return mask, s_flag, relation_vis_text


def batch_index_select(x, idx):
    if len(x.size()) == 4:
        B, H, N, C = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long,
                              device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B * N, H, C)[idx.reshape(-1)].reshape(B, H, N_new, C)
        return out
    elif len(x.size()) == 3:
        # in this condition
        B, N, C = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long,
                              device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B * N, C)[idx.reshape(-1)].reshape(B, N_new, C)
        return out
    elif len(x.size()) == 2:
        B, N = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long,
                              device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B * N)[idx.reshape(-1)].reshape(B, N_new)
        return out
    else:
        raise NotImplementedError


def index_points(points, idx):
    """Sample features following the index.
    Returns:
        new_points:, indexed points data, [B, S, C]

    Args:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(
        device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def cluster_and_merge(x, cluster_num):

    B, N, C = x.shape

    x1 = ein.rearrange(x, 'b l r -> b l () r')
    x2 = ein.rearrange(x, 'b l r -> b () l r')
    distance = (x1 - x2).norm(dim=-1, p=2)
    dist_matrix = distance / (C ** 0.5)
    # get local density
    dist_nearest, index_nearest = torch.topk(
        dist_matrix, k=cluster_num, dim=-1, largest=False)
    density = (-(dist_nearest ** 2).mean(dim=-1)).exp()
    # add a little noise to ensure no tokens have the same density.
    density = density + torch.rand(
        density.shape, device=density.device, dtype=density.dtype) * 1e-6

    # get distance indicator
    mask = density[:, None, :] > density[:, :, None]
    mask = mask.type(x.dtype)
    dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
    dist, _ = (dist_matrix * mask +
               dist_max * (1 - mask)).min(dim=-1)

    # select clustering center according to score
    score = dist * density
    _, index_down = torch.topk(score, k=cluster_num, dim=-1)

    # assign tokens to the nearest center
    dist_matrix = index_points(dist_matrix, index_down)

    idx_cluster = dist_matrix.argmin(dim=1)

    # make sure cluster center merge to itself
    idx_batch = torch.arange(B, device=x.device)[
        :, None].expand(B, cluster_num)
    idx_tmp = torch.arange(cluster_num, device=x.device)[
        None, :].expand(B, cluster_num)
    idx_cluster[idx_batch.reshape(-1),
                index_down.reshape(-1)] = idx_tmp.reshape(-1)

    # merge tokens

    B, N, C = x.shape
    # device = dist_matrix.device
    # idx_token = torch.arange(N)[None, :].repeat(B, 1).to(device)
    # agg_weight = x.new_ones(B, N, 1)

    token_weight = x.new_ones(B, N, 1)
    # self_attn_weights = self_attn_weights.mean(1)
    # token_weight = self_attn_weights.sum(dim=1).exp().unsqueeze(2)
    # B_weight,N_weigh,C_weight = token_weight.shape
    # token_weight = token_weight.reshape(B_weight*N_weigh, C_weight)
    # [sparse_token_idx.reshape(-1)].reshape(B, N, 1)

    idx_batch = torch.arange(B, device=x.device)[:, None]
    idx = idx_cluster + idx_batch * cluster_num

    all_weight = token_weight.new_zeros(B * cluster_num, 1)
    all_weight.index_add_(dim=0, index=idx.reshape(B * N),
                          source=token_weight.reshape(B * N, 1))
    all_weight = all_weight + 1e-6
    norm_weight = token_weight / all_weight[idx]

    # average token features
    x_merged = x.new_zeros(B * cluster_num, C)
    source = x * norm_weight
    x_merged.index_add_(dim=0, index=idx.reshape(B * N),
                        source=source.reshape(B * N, C).type(x.dtype))
    x_merged = x_merged.reshape(B, cluster_num, C)

    return x_merged
