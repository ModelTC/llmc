import torch
from loguru import logger
from transformers import DynamicCache

from llmc.utils.registry_factory import KV_REGISTRY

from .quant import FloatQuantizer, IntegerQuantizer


@KV_REGISTRY.register('Naive')
class NaiveQuantKVCache(DynamicCache):
    def __init__(self, quant_type, kvquant_cfg, num_hidden_layers, num_samples=128, bsz=1):
        super().__init__()

        assert kvquant_cfg.granularity in ['per_token', 'per_tensor', 'per_group']
        self.num_hidden_layers, self.num_samples, self.bsz = (
            num_hidden_layers,
            num_samples,
            bsz,
        )
        if quant_type == 'int-quant':
            self.kvquantizer = IntegerQuantizer(**kvquant_cfg)
        elif quant_type == 'float-quant':
            self.kvquantizer = FloatQuantizer(**kvquant_cfg)

        self.kvquant_cfg = kvquant_cfg
        self.static = kvquant_cfg.get('static', False)
        self._quantized_key_cache = []
        self._quantized_value_cache = []
        self.use_org_kv = False

        if self.static:
            self._reset_buffers()
            self.calib_key_cache = [
                [] for i in range(self.num_hidden_layers)
            ]
            self.calib_value_cache = [
                [] for i in range(self.num_hidden_layers)
            ]
            self.calib = True
        else:
            self.calib = False

    def update(
        self,
        key_states,
        value_states,
        layer_idx,
        cache_kwargs,
    ):
        if self.use_org_kv:
            return super().update(key_states, value_states, layer_idx, cache_kwargs)
        elif self.static and self.calib:
            self._calibration(layer_idx, key_states, value_states)
            keys_to_return, values_to_return = key_states, value_states
        else:
            if layer_idx == 0:
                self._seen_tokens += key_states.shape[-2]

            if len(self._quantized_key_cache) <= layer_idx:
                # Prefill
                q_keys = self._quantize(key_states.contiguous(), layer_idx, is_key=True)
                q_values = self._quantize(
                    value_states.contiguous(), layer_idx, is_key=False
                )
                self._quantized_key_cache.append(q_keys)
                self._quantized_value_cache.append(q_values)
                keys_to_return = self._dequantize(q_keys)
                values_to_return = self._dequantize(q_values)
            else:
                # Decode
                dequant_key = self._dequantize(self._quantized_key_cache[layer_idx])
                dequant_value = self._dequantize(self._quantized_value_cache[layer_idx])
                keys_to_return = [dequant_key, key_states]
                values_to_return = [dequant_value, value_states]

                keys_to_return = torch.cat(keys_to_return, dim=-2)
                values_to_return = torch.cat(values_to_return, dim=-2)

                self._quantized_key_cache[layer_idx] = self._quantize(
                    keys_to_return.contiguous(), layer_idx, is_key=True
                )
                self._quantized_value_cache[layer_idx] = self._quantize(
                    values_to_return.contiguous(), layer_idx, is_key=False
                )

        return keys_to_return, values_to_return

    def _check_pass_all_calib_data(self, layer_idx):
        return (
            self.bsz == 1 and len(self.calib_value_cache[layer_idx]) == self.num_samples
        ) or (
            self.bsz == -1
            and self.calib_value_cache[layer_idx][0].shape[0] == self.num_samples
        )

    def _calibration(self, layer_idx, key_states, value_states):
        # Calibration data can be provided through the prompt or
        # the preprocessed decode data.
        # Therefore, calibration occurs only during the prefill stage.
        self.calib_key_cache[layer_idx].append(key_states)
        self.calib_value_cache[layer_idx].append(value_states)

        if self._check_pass_all_calib_data(layer_idx):
            # Get and store calibration parameters for keys and values
            for data, buffer, scale_buffer, zero_buffer, qmin_buffer, qmax_buffer in [
                (
                    self.calib_key_cache[layer_idx],
                    self.k_scales_buffer,
                    self.k_scales_buffer,
                    self.k_zeros_buffer,
                    self.k_qmin_buffer,
                    self.k_qmax_buffer,
                ),
                (
                    self.calib_value_cache[layer_idx],
                    self.v_scales_buffer,
                    self.v_scales_buffer,
                    self.v_zeros_buffer,
                    self.v_qmin_buffer,
                    self.v_qmax_buffer,
                ),
            ]:
                scales, zeros, qmin, qmax = self.get_qparams(data)
                (
                    scale_buffer[layer_idx],
                    zero_buffer[layer_idx],
                    qmin_buffer[layer_idx],
                    qmax_buffer[layer_idx],
                ) = (scales, zeros, qmin, qmax)

            # Clear the calibration caches
            self.calib_key_cache[layer_idx].clear()
            self.calib_value_cache[layer_idx].clear()

    def _quantize(self, tensor, layer_idx, is_key):
        org_shape = tensor.shape
        tensor = self.kvquantizer.reshape_tensor(tensor)

        if self.static:
            scales = (
                self.k_scales_buffer[layer_idx]
                if is_key
                else self.v_scales_buffer[layer_idx]
            )
            zeros = (
                self.k_zeros_buffer[layer_idx]
                if is_key
                else self.v_zeros_buffer[layer_idx]
            )
            qmax = (
                self.k_qmax_buffer[layer_idx]
                if is_key
                else self.v_qmax_buffer[layer_idx]
            )
            qmin = (
                self.k_qmin_buffer[layer_idx]
                if is_key
                else self.v_qmin_buffer[layer_idx]
            )
        else:
            tensor_range = self.kvquantizer.get_tensor_range(tensor, {})
            scales, zeros, qmax, qmin = self.kvquantizer.get_qparams(
                tensor_range, tensor.device
            )

        q_tensor = self.kvquantizer.quant(tensor, scales, zeros, qmax, qmin)
        q_tensor = self.kvquantizer.restore_tensor(q_tensor, org_shape)

        q_tensors = {
            'q_tensor': q_tensor,
            'scales': scales,
            'zeros': zeros,
        }

        return q_tensors

    def _dequantize(self, q_tensors):
        q_tensor = q_tensors['q_tensor']
        scales = q_tensors['scales']
        zeros = q_tensors['zeros']
        org_shape = q_tensor.shape
        q_tensor = self.kvquantizer.reshape_tensor(q_tensor)
        qdq_tensor = self.kvquantizer.dequant(q_tensor, scales, zeros)
        qdq_tensor = self.kvquantizer.restore_tensor(qdq_tensor, org_shape)
        return qdq_tensor

    def _reset_buffers(self):
        self.k_scales_buffer = [torch.zeros(0)] * self.num_hidden_layers
        self.k_zeros_buffer = [torch.zeros(0)] * self.num_hidden_layers
        self.k_qmin_buffer = [0] * self.num_hidden_layers
        self.k_qmax_buffer = [0] * self.num_hidden_layers

        self.v_scales_buffer = [torch.zeros(0)] * self.num_hidden_layers
        self.v_zeros_buffer = [torch.zeros(0)] * self.num_hidden_layers
        self.v_qmin_buffer = [0] * self.num_hidden_layers
        self.v_qmax_buffer = [0] * self.num_hidden_layers

    def _reset_states(self):
        self._quantized_key_cache = []
        self._quantized_value_cache = []
        self.key_cache = []
        self.value_cache = []
        self._seen_tokens = 0

    def get_qparams(self, tensor):
        scales_list, zeros_list, qmin_list, qmax_list = (
            self.kvquantizer.get_batch_tensors_qparams(tensor)
        )
        scales, zeros, qmin, qmax = (
            scales_list[0],
            zeros_list[0],
            qmin_list[0],
            qmax_list[0],
        )
        return scales, zeros, qmin, qmax

    def get_seq_length(self, layer_idx=0):
        if self.use_org_kv:
            return super().get_seq_length()
        if len(self._quantized_key_cache) <= layer_idx:
            return 0
        return self._seen_tokens if layer_idx == 0 else self._seen_tokens - 1


@KV_REGISTRY.register('Kivi')
class KiviQuantKVCache(NaiveQuantKVCache):
    def __init__(self, quant_type, kvquant_cfg, num_hidden_layers, num_samples=128, bsz=1):
        super().__init__(quant_type, kvquant_cfg, num_hidden_layers, num_samples, bsz)
        assert not self.static, 'Only support dynamic quantization for KIVI'
        self.residual_length = kvquant_cfg.get('residual_length', 128)

    def update(
        self,
        key_states,
        value_states,
        layer_idx,
        cache_kwargs,
    ):
        if self.use_org_kv:
            return super().update(key_states, value_states, layer_idx, cache_kwargs)
        else:
            if layer_idx == 0:
                self._seen_tokens += key_states.shape[-2]

            if len(self.key_cache) <= layer_idx:
                self._quantized_key_cache.append(self._quantize(key_states.contiguous(),
                                                                layer_idx,
                                                                is_key=True))
                self._quantized_value_cache.append(self._quantize(value_states.contiguous(),
                                                                  layer_idx,
                                                                  is_key=False))
                self.key_cache.append(torch.zeros(0,
                                                  dtype=key_states.dtype,
                                                  device=key_states.device))
                self.value_cache.append(torch.zeros(0,
                                                    dtype=key_states.dtype,
                                                    device=key_states.device))
                keys_to_return, values_to_return = key_states, value_states
            else:
                dequant_key = self._dequantize(self._quantized_key_cache[layer_idx])
                dequant_value = self._dequantize(self._quantized_value_cache[layer_idx])
                keys_to_return = [dequant_key, self.key_cache[layer_idx], key_states]
                values_to_return = [dequant_value, self.value_cache[layer_idx], value_states]

                keys_to_return = torch.cat(keys_to_return, dim=-2)
                values_to_return = torch.cat(values_to_return, dim=-2)
                if (
                    self.key_cache[layer_idx].dim() == 4
                    and self.key_cache[layer_idx].shape[-2] + 1 >= self.residual_length
                ):
                    self._quantized_key_cache[layer_idx] = \
                        self._quantize(keys_to_return.contiguous(), layer_idx, is_key=True)
                    self._quantized_value_cache[layer_idx] = self._quantize(
                        values_to_return.contiguous(), layer_idx, is_key=False
                    )
                    self.key_cache[layer_idx] = torch.zeros(0,
                                                            dtype=key_states.dtype,
                                                            device=key_states.device)
                    self.value_cache[layer_idx] = torch.zeros(0,
                                                              dtype=key_states.dtype,
                                                              device=key_states.device)
                else:
                    self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states],
                                                          dim=-2)
                    self.value_cache[layer_idx] = \
                        torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

            return keys_to_return, values_to_return


@KV_REGISTRY.register('Sink')
class SinkQuantKVCache(NaiveQuantKVCache):
    def __init__(
        self,
        quant_type,
        kvquant_cfg,
        num_hidden_layers,
        window_length,
        num_sink_tokens,
        num_samples=128,
        bsz=1
    ):
        super().__init__(quant_type, kvquant_cfg, num_hidden_layers, num_samples, bsz)
        assert not self.static, 'Only support dynamic quantization for Sink'
        self.window_length = window_length
        self.num_sink_tokens = num_sink_tokens
        self.cos_sin_rerotation_cache = {}
        self._cos_cache = None
        self._sin_cache = None

    @staticmethod
    def _rotate_half(x):
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_key_rotary_pos_emb(
        self, key_states, cos, sin
    ):
        rotated_key_states = (key_states * cos) + (self._rotate_half(key_states) * sin)
        return rotated_key_states

    def _get_rerotation_cos_sin(
        self, key_states, cos, sin
    ):
        if key_states.shape[-2] not in self.cos_sin_rerotation_cache:
            # Upcast to float32 temporarily for better accuracy
            cos = cos.to(torch.float32)
            sin = sin.to(torch.float32)

            original_cos = cos[self.num_sink_tokens + key_states.shape[-2]:]
            shifted_cos = cos[self.num_sink_tokens:-key_states.shape[-2]]
            original_sin = sin[self.num_sink_tokens + key_states.shape[-2]:]
            shifted_sin = sin[self.num_sink_tokens:-key_states.shape[-2]]
            rerotation_cos = original_cos * shifted_cos + original_sin * shifted_sin
            rerotation_sin = -original_sin * shifted_cos + original_cos * shifted_sin

            self.cos_sin_rerotation_cache[key_states.shape[-2]] = (
                rerotation_cos.to(key_states.dtype).unsqueeze(0),
                rerotation_sin.to(key_states.dtype).unsqueeze(0),
            )
        return self.cos_sin_rerotation_cache[key_states.shape[-2]]

    def get_seq_length(self, layer_idx=0):
        """Returns the sequence length of the cached states.

        A layer index can be optionally passed.
        """
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_max_cache_shape(self):
        """Returns the maximum sequence length of the cache object, in case of
        SinkCache it is the window length."""
        return self.window_length

    def update(
        self,
        key_states,
        value_states,
        layer_idx,
        cache_kwargs,
    ):

        if self.use_org_kv:
            return super().update(key_states, value_states, layer_idx, cache_kwargs)
        else:
            sin = cache_kwargs.get('sin')
            cos = cache_kwargs.get('cos')
            partial_rotation_size = cache_kwargs.get('partial_rotation_size')
            using_rope = cos is not None and sin is not None

            if layer_idx == 0:
                self._seen_tokens += key_states.shape[-2]

            if using_rope and layer_idx == 0:

                if cos.dim() == 2:
                    self._cos_cache = cos
                    self._sin_cache = sin
                else:
                    if self._cos_cache is None:
                        self._cos_cache = cos[0, ...]
                        self._sin_cache = sin[0, ...]
                    elif self._cos_cache.shape[0] < self.window_length:
                        self._cos_cache = torch.cat([self._cos_cache, cos[0, ...]], dim=0)
                        self._sin_cache = torch.cat([self._sin_cache, sin[0, ...]], dim=0)

            # [bsz, num_heads, seq_len, head_dim]
            if len(self.key_cache) <= layer_idx:
                # Empty cache
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)

            elif key_states.shape[-2] + self.get_seq_length(layer_idx) < self.window_length:
                # Growing cache
                self.key_cache[layer_idx] = \
                    torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
                self.value_cache[layer_idx] = \
                    torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

            else:
                # Shifting cache
                keys_to_keep = self.key_cache[layer_idx][
                    :, :, -self.window_length + self.num_sink_tokens + key_states.shape[-2]:
                ]

                if using_rope:
                    rerotation_cos, rerotation_sin = self._get_rerotation_cos_sin(
                        key_states,
                        self._cos_cache[: self.window_length],
                        self._sin_cache[: self.window_length]
                    )
                    if partial_rotation_size is not None:
                        keys_to_keep, keys_pass = (
                            keys_to_keep[..., :partial_rotation_size],
                            keys_to_keep[..., partial_rotation_size:],
                        )
                    keys_to_keep = self._apply_key_rotary_pos_emb(keys_to_keep,
                                                                  rerotation_cos,
                                                                  rerotation_sin)
                    if partial_rotation_size is not None:
                        keys_to_keep = torch.cat((keys_to_keep, keys_pass), dim=-1)

                # Concatenate sink tokens, shifted & rotated tokens (if needed), and new tokens
                sink_keys = self.key_cache[layer_idx][:, :, : self.num_sink_tokens]

                dq_keys_to_keep = self._dequantize(self._quantize(keys_to_keep.contiguous(),
                                                                  layer_idx,
                                                                  is_key=True))
                dq_keys = self._dequantize(self._quantize(key_states.contiguous(),
                                                          layer_idx,
                                                          is_key=True))

                self.key_cache[layer_idx] = torch.cat([sink_keys, dq_keys_to_keep, dq_keys], dim=-2)

                sink_values = self.value_cache[layer_idx][:, :, : self.num_sink_tokens]
                values_to_keep = self.value_cache[layer_idx][
                    :, :, -self.window_length + self.num_sink_tokens + value_states.shape[-2]:
                ]
                dq_values_to_keep = self._dequantize(self._quantize(values_to_keep.contiguous(),
                                                                    layer_idx,
                                                                    is_key=True))
                dq_values = self._dequantize(self._quantize(value_states.contiguous(),
                                                            layer_idx,
                                                            is_key=True))

                self.value_cache[layer_idx] = torch.cat([sink_values,
                                                         dq_values_to_keep,
                                                         dq_values], dim=-2)

            return self.key_cache[layer_idx], self.value_cache[layer_idx]
