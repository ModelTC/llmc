import functools
import math

import torch
from torch import nn
from transformers.modeling_attn_mask_utils import \
    _prepare_4d_causal_attention_mask
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from llmc.utils.registry_factory import TOKEN_REDUCTION_REGISTRY

from .token_reduction_module import TokenReductionModule
from .utils import prefill_wrapper


@TOKEN_REDUCTION_REGISTRY.register('PyramidDrop')
class PyramidDrop(TokenReductionModule):
    def __init__(self, config, model, blocks):
        super().__init__(config, model, blocks)
        self.add_sparse_config()
        self.register_reduction_modules()

    def add_sparse_config(self):

        self.pruning_loc = self.special_config['layer_list']
        image_token_ratio_list = self.special_config['image_token_ratio_list']
        image_token_ratio_list.insert(0, 1.0)
        self.special_config['image_token_ratio_list'] = image_token_ratio_list
        self.special_config['tokenizer_padding_side'] = getattr(
            self.model.vlm_model.language_model.model.config,
            'tokenizer_padding_side',
            'right',
        )

        self.model.model.parameters = self.special_config

    def register_reduction_modules(self):
        @prefill_wrapper
        def pruning_hook(module, args, kwargs, pruning_pars, cur_num, layer_idx):

            if layer_idx == self.pruning_loc[0]:
                position_ids = kwargs['position_ids']
                attention_mask = kwargs['attention_mask']
                position_embeddings = kwargs['position_embeddings']
            else:
                attention_mask = pruning_pars['attention_mask']
                position_ids = pruning_pars['position_ids']
                position_embeddings = pruning_pars['position_embeddings']

            features = args[0]
            _position_ids = position_ids
            _attention_mask = attention_mask
            prompt_len = pruning_pars['prompt_len']
            image_tokens_list = pruning_pars['image_tokens']
            image_token_posi = pruning_pars['image_token_posi']
            image_token_ratio_list = pruning_pars['image_token_ratio_list']

            # for decoding stage
            if features.shape[1] == 1:
                return args, kwargs

            if position_ids is None:
                position_ids = torch.arange(
                    0, features.shape[1], dtype=torch.long, device=features.device
                ).unsqueeze(0)

            if pruning_pars['tokenizer_padding_side'] == 'right':

                batch_size = features.shape[0]
                image_tokens = [
                    int(cur_image_token * image_token_ratio_list[cur_num])
                    for cur_image_token in image_tokens_list
                ]
                keep_length = [
                    int(cur_image_token * image_token_ratio_list[cur_num + 1])
                    for cur_image_token in image_tokens_list
                ]

                features_list = []
                attention_mask_list = []

                if attention_mask is None:
                    attention_mask = torch.ones(
                        (batch_size, features.shape[1]),
                        dtype=torch.bool,
                        device=features.device,
                    )
                else:
                    attention_mask = attention_mask.bool()

                # obtain query_states and key_states to calculate attention map
                hidden_states = features.clone().detach()
                self_attn = module.self_attn
                hidden_states = module.input_layernorm(hidden_states)

                num_heads = self_attn.num_heads
                num_key_value_heads = self_attn.num_key_value_heads
                head_dim = self_attn.head_dim

                bsz, q_len, _ = hidden_states.size()

                query_states = self_attn.q_proj(hidden_states)
                key_states = self_attn.k_proj(hidden_states)
                value_states = self_attn.v_proj(hidden_states)

                query_states = query_states.view(
                    bsz, q_len, num_heads, head_dim
                ).transpose(1, 2)
                key_states = key_states.view(
                    bsz, q_len, num_key_value_heads, head_dim
                ).transpose(1, 2)
                value_states = value_states.view(
                    bsz, q_len, num_key_value_heads, head_dim
                ).transpose(1, 2)

                if position_embeddings is None:
                    cos, sin = self_attn.rotary_emb(value_states, position_ids)
                else:
                    cos, sin = position_embeddings

                query_states, key_states = apply_rotary_pos_emb(
                    query_states, key_states, cos, sin
                )

                # attention_mask
                eager_attention_mask = _prepare_4d_causal_attention_mask(
                    attention_mask,
                    (batch_size, q_len),
                    hidden_states,
                    past_key_values_length=0,
                ).to(device=query_states.device)

                # take valid features
                features = [
                    cur_features[cur_attention_mask]
                    for cur_features, cur_attention_mask in zip(
                        features, attention_mask
                    )
                ]
                attention_mask = [
                    cur_attention_mask[cur_attention_mask]
                    for cur_attention_mask, cur_attention_mask in zip(
                        attention_mask, attention_mask
                    )
                ]

                # rank & drop
                for i in range(batch_size):
                    image_index = image_token_posi[i]
                    if image_index == -1:
                        cur_input_embeds = features[i]
                        features_list.append(cur_input_embeds)
                        attention_mask_list.append(attention_mask[i])
                        continue

                    # obtain current states
                    cur_key_states = key_states[i]
                    cur_query_states = query_states[i]
                    cur_eager_attention_mask = eager_attention_mask[i]

                    prompt_total_len = prompt_len[i] + image_tokens[i]
                    text_query_states = cur_query_states[
                        :, prompt_total_len - 1, :
                    ].unsqueeze(1)
                    text_eager_attention_mask = cur_eager_attention_mask[
                        :, prompt_total_len - 1, :
                    ].unsqueeze(1)

                    # calculate attention map
                    attn_weights = torch.matmul(
                        text_query_states, cur_key_states.transpose(1, 2)
                    ) / math.sqrt(
                        head_dim
                    )  # (num_head, text_token,seq_len)
                    attn_weights = attn_weights + text_eager_attention_mask
                    attn_weights = nn.functional.softmax(
                        attn_weights, dim=-1, dtype=torch.float32
                    ).to(
                        query_states.dtype
                    )  # (num_head, text_token,seq_len)

                    attention_avg_head = torch.mean(
                        attn_weights, dim=0
                    )  # ave across heads
                    attention_avg_head = attention_avg_head[
                        :, image_index: image_index + image_tokens[i]
                    ]  # select image token as keys
                    attention_avg_text = torch.mean(attention_avg_head, dim=0)  # (576)

                    # rank and drop by attention score
                    top_rank_index = attention_avg_text.topk(keep_length[i]).indices
                    top_rank_index = top_rank_index + image_index
                    top_rank_index = top_rank_index.sort().values

                    start_index = image_index + image_tokens[i]
                    new_input_embeds = torch.cat(
                        [
                            features[i][:image_index, :],
                            features[i][top_rank_index, :],
                            features[i][start_index:, :],
                        ],
                        dim=0,
                    )
                    new_attention_mask = torch.cat(
                        [
                            attention_mask[i][:image_index],
                            attention_mask[i][top_rank_index],
                            attention_mask[i][start_index:],
                        ],
                        dim=0,
                    )

                    features_list.append(new_input_embeds)
                    attention_mask_list.append(new_attention_mask)

                # Truncate sequences to max length as image embeddings can make the sequence longer
                tokenizer_model_max_length = getattr(
                    self.model.vlm_model.language_model.model.config,
                    'tokenizer_model_max_length',
                    2048,
                )
                if tokenizer_model_max_length is not None:
                    new_input_embeds = [
                        x[:tokenizer_model_max_length] for x in features_list
                    ]
                    new_attention_mask = [
                        x[:tokenizer_model_max_length] for x in attention_mask_list
                    ]

                max_len = max(x.shape[0] for x in new_input_embeds)

                # padding the sequences to form batch
                embeds_padded = []
                attention_mask_padded = []
                position_ids = torch.zeros(
                    (batch_size, max_len),
                    dtype=position_ids.dtype,
                    device=position_ids.device,
                )
                for i, cur_new_embed in enumerate(new_input_embeds):
                    cur_len_emb = cur_new_embed.shape[0]
                    dif = max_len - cur_len_emb  # padding to longest seq

                    cur_new_embed = torch.cat(
                        [
                            cur_new_embed,
                            torch.zeros(
                                (dif, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                        ],
                        dim=0,
                    )
                    cur_attention_mask = new_attention_mask[i]
                    cur_attention_mask = torch.cat(
                        [
                            cur_attention_mask,
                            torch.full(
                                (dif,),
                                False,
                                dtype=cur_attention_mask.dtype,
                                device=cur_attention_mask.device,
                            ),
                        ],
                        dim=0,
                    )

                    embeds_padded.append(cur_new_embed)

                    attention_mask_padded.append(cur_attention_mask)

                    cur_len = new_attention_mask[i].sum().item()
                    position_ids[i, :cur_len] = torch.arange(
                        0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                    )

                new_input_embeds = torch.stack(embeds_padded, dim=0)
                new_input_embeds = new_input_embeds.to(features[0].dtype)

                new_attention_mask = torch.stack(attention_mask_padded, dim=0)

                if _position_ids is None:
                    position_ids = None

                if _attention_mask is None:
                    new_attention_mask = None
                else:
                    new_attention_mask = new_attention_mask.to(
                        dtype=_attention_mask.dtype
                    )

                kwargs['attention_mask'] = new_attention_mask
                kwargs['position_ids'] = position_ids
                kwargs['position_embeddings'] = None
                pruning_pars['attention_mask'] = new_attention_mask
                pruning_pars['position_ids'] = position_ids
                pruning_pars['position_embeddings'] = None

                return (new_input_embeds,), kwargs

        @prefill_wrapper
        def input_hook(module, input_args, pruning_pars):

            input_ids = input_args[0]
            pre_prompt_length_list = []
            image_token_posi = []
            vision_tokens = []
            VISION_TOKEN_INDEX = pruning_pars['vision_token_index']

            # find the position of the first image token
            for seq in input_ids:
                image_token_idxs = (seq == VISION_TOKEN_INDEX).nonzero(as_tuple=True)[0]
                vision_tokens.append(pruning_pars['vision_token_length'])
                image_token_posi.append(image_token_idxs[0].item())
                pre_prompt_length_list.append(seq.shape[0] - image_token_idxs.shape[0])

            pruning_pars['prompt_len'] = pre_prompt_length_list
            pruning_pars['image_token_posi'] = image_token_posi
            pruning_pars['image_tokens'] = vision_tokens

            return input_args

        @prefill_wrapper
        def read_parameter_hook(module, args, kwargs, pruning_pars):
            kwargs['attention_mask'] = pruning_pars['attention_mask']
            # kwargs['cache_position'] = pruning_pars['cache_position']
            kwargs['position_ids'] = pruning_pars['position_ids']
            kwargs['position_embeddings'] = pruning_pars['position_embeddings']

            return args, kwargs

        self.model.embed_tokens.register_forward_pre_hook(
            functools.partial(input_hook, pruning_pars=self.model.model.parameters)
        )

        for layer_idx in range(self.pruning_loc[0], len(self.blocks)):
            if layer_idx in self.pruning_loc:
                stage = self.pruning_loc.index(layer_idx)
                self.blocks[layer_idx].register_forward_pre_hook(
                    functools.partial(
                        pruning_hook,
                        pruning_pars=self.model.model.parameters,
                        cur_num=stage,
                        layer_idx=layer_idx,
                    ),
                    with_kwargs=True,
                )
            else:
                self.blocks[layer_idx].register_forward_pre_hook(
                    functools.partial(
                        read_parameter_hook, pruning_pars=self.model.model.parameters
                    ),
                    with_kwargs=True,
                )
