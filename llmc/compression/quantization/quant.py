import torch
from loguru import logger
from torch import nn


class BaseQuantizer(object):
    def __init__(self, bit, symmetric, granularity, **kwargs):
        self.bit = bit
        self.sym = symmetric
        self.granularity = granularity
        self.kwargs = kwargs

        self.calib_algo = self.kwargs.get('calib_algo', 'minmax')

        if self.granularity == 'per_group':
            self.group_size = self.kwargs['group_size']
        elif self.granularity == 'per_head':
            self.head_num = self.kwargs['head_num']

        self.mse_b_num = self.kwargs.get('mse_b_num', 1)

        if self.kwargs.get('ste', False):
            self.round_func = lambda x: (x.round() - x).detach() + x
        else:
            self.round_func = torch.round
        if 'ste_all' in self.kwargs and self.kwargs['ste_all']:
            self.round_func = torch.round
            self.ste_all = True
        else:
            self.ste_all = False

        self.round_zp = self.kwargs.get('round_zp', True)
        self.sigmoid = torch.nn.Sigmoid()

    def get_tensor_range(self, tensor, args={}):
        if self.calib_algo == 'minmax':
            return self.get_minmax_range(tensor)
        elif self.calib_algo == 'mse':
            return self.get_mse_range(tensor)
        elif self.calib_algo == 'learnable':
            return self.get_learnable_range(tensor, **args)
        else:
            raise ValueError(f'Unsupported calibration algorithm: {self.calib_algo}')

    def get_running_tensor_range(self, act_tensors, alpha, args):
        assert len(act_tensors) > 0, (
            'Calibration data is insufficient. Please provide more data to ensure '
            'all experts in the MOE receive an adequate number of tokens.'
        )

        runing_min_vals, runing_max_vals = [], []
        if isinstance(act_tensors[0], tuple):
            # Handle multiple inputs by stacking tensors.
            unzipped_inputs = zip(*act_tensors)
            act_tensors = [torch.stack(tensor_list) for tensor_list in unzipped_inputs]
        else:
            if len(act_tensors) == 1:
                # Handle batch-size=-1 case.
                tensor_list = [act_tensors[0][i] for i in range(act_tensors[0].size(0))]
                act_tensors[0] = tensor_list
            else:
                act_tensors = [act_tensors]

        for tensors in act_tensors:
            runing_min_val, runing_max_val = None, None
            for tensor in tensors:
                tensor = self.reshape_tensor(tensor)
                tensor_range = self.get_tensor_range(tensor, args)
                min_val, max_val = tensor_range[0], tensor_range[1]

                if runing_min_val is None or runing_max_val is None:
                    runing_min_val = min_val
                    runing_max_val = max_val
                else:
                    runing_min_val = runing_min_val + alpha * (
                        min_val - runing_min_val
                    )
                    runing_max_val = runing_max_val + alpha * (
                        max_val - runing_max_val
                    )
            runing_min_vals.append(runing_min_val)
            runing_max_vals.append(runing_max_val)

        return runing_min_vals, runing_max_vals

    def get_minmax_range(self, tensor):
        if self.granularity == 'per_tensor':
            max_val = torch.max(tensor)
            min_val = torch.min(tensor)
        else:
            max_val = tensor.amax(dim=-1, keepdim=True)
            min_val = tensor.amin(dim=-1, keepdim=True)

        return (min_val, max_val)

    def get_mse_range(self, tensor, grid=100, norm=2.4, maxshrink=0.8, bs=256):

        assert (
            self.mse_b_num >= 1 and tensor.shape[0] % self.mse_b_num == 0
        ), 'Batch number must be divisible by tensor.shape[0],'
        bs = tensor.shape[0] // self.mse_b_num
        tensor = tensor.float()
        min_val, max_val = self.get_minmax_range(tensor)

        dev = tensor.device

        for b_num in range(self.mse_b_num):
            _tensor = tensor[b_num * bs:(b_num + 1) * bs, :]
            _min_val, _max_val = (
                min_val[b_num * bs:(b_num + 1) * bs, :],
                max_val[b_num * bs:(b_num + 1) * bs, :],
            )

            best = torch.full([_tensor.shape[0]], float('inf'), device=dev)

            best_min_val, best_max_val = _min_val, _max_val

            for i in range(int(maxshrink * grid)):
                p = 1 - i / grid

                xmin = p * _min_val
                xmax = p * _max_val

                if self.quant_type == 'float-quant' and not self.use_qtorch:
                    clip_tensor, scales = self.get_float_qparams(
                        _tensor, (xmin, xmax), dev
                    )
                    zeros, qmin, qmax = 0, None, None
                    q_tensor = self.quant_dequant(
                        clip_tensor, scales, zeros, qmax, qmin
                    )

                else:
                    scales, zeros, qmax, qmin = self.get_qparams((xmin, xmax), dev)
                    q_tensor = self.quant_dequant(_tensor, scales, zeros, qmax, qmin)

                q_tensor -= _tensor
                q_tensor.abs_()
                q_tensor.pow_(norm)
                err = torch.sum(q_tensor, 1)

                tmp = err < best

                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    best_min_val[tmp] = xmin[tmp]
                    best_max_val[tmp] = xmax[tmp]

            (
                min_val[b_num * bs:(b_num + 1) * bs, :],
                max_val[b_num * bs:(b_num + 1) * bs, :],
            ) = (best_min_val, best_max_val)

        return (min_val, max_val)

    def get_learnable_range(self, tensor, lowbound_factor=None, upbound_factor=None):
        min_val, max_val = self.get_minmax_range(tensor)
        if self.sym:
            if upbound_factor is not None:
                abs_max = torch.max(max_val.abs(), min_val.abs())
                abs_max = abs_max.clamp(min=1e-5)
                abs_max = self.sigmoid(upbound_factor) * abs_max
                min_val = -abs_max
                max_val = abs_max
        else:
            if upbound_factor is not None and lowbound_factor is not None:
                min_val = self.sigmoid(lowbound_factor) * min_val
                max_val = self.sigmoid(upbound_factor) * max_val

        return (min_val, max_val)

    def get_qparams(self, tensor_range, device):
        min_val, max_val = tensor_range[0], tensor_range[1]
        qmin = self.qmin.to(device)
        qmax = self.qmax.to(device)
        if self.sym:
            abs_max = torch.max(max_val.abs(), min_val.abs())
            abs_max = abs_max.clamp(min=1e-5)
            scales = abs_max / qmax
            zeros = torch.tensor(0.0)
        else:
            scales = (max_val - min_val).clamp(min=1e-5) / (qmax - qmin)
            zeros = (qmin - torch.round(min_val / scales)).clamp(qmin, qmax)
            if not self.round_zp:
                zeros = qmin - (min_val / scales)
        return scales, zeros, qmax, qmin

    def get_tensor_qparams(self, tensor, args={}):
        tensor = self.reshape_tensor(tensor)
        tensor_range = self.get_tensor_range(tensor, args)
        scales, zeros, qmax, qmin = self.get_qparams(tensor_range, tensor.device)
        return tensor, scales, zeros, qmax, qmin

    def get_batch_tensors_qparams(self, act_tensors, alpha=0.01, args={}):
        scales_list, zeros_list, qmin_list, qmax_list = [], [], [], []
        runing_min_vals, runing_max_vals = self.get_running_tensor_range(
            act_tensors, alpha, args
        )
        for i in range(len(runing_min_vals)):
            runing_min_val, runing_max_val = runing_min_vals[i], runing_max_vals[i]
            scales, zeros, qmax, qmin = self.get_qparams(
                (runing_min_val, runing_max_val), runing_min_val.device
            )
            scales_list.append(scales)
            zeros_list.append(zeros)
            qmin_list.append(qmin)
            qmax_list.append(qmax)

        return scales_list, zeros_list, qmin_list, qmax_list

    def reshape_tensor(self, tensor, allow_padding=False):
        if self.granularity == 'per_group':
            if tensor.shape[-1] >= self.group_size:
                if tensor.shape[-1] % self.group_size == 0:
                    t = tensor.reshape(-1, self.group_size)
                elif allow_padding:
                    deficiency = self.group_size - tensor.shape[1] % self.group_size
                    prefix = tensor.shape[:-1]
                    pad_zeros = torch.zeros(
                        (*prefix, deficiency), device=tensor.device, dtype=tensor.dtype
                    )
                    t = torch.cat((tensor, pad_zeros), dim=-1).reshape(
                        -1, self.group_size
                    )
                else:
                    raise ValueError(
                        f'Dimension {tensor.shape[-1]} '
                        f'not divisible by group size {self.group_size}'
                    )
            else:
                t = tensor
        elif self.granularity == 'per_head':
            t = tensor.reshape(self.head_num, -1)
        else:
            t = tensor
        return t

    def restore_tensor(self, tensor, shape):
        if tensor.shape == shape:
            t = tensor
        else:
            try:
                t = tensor.reshape(shape)
            except RuntimeError:
                deficiency = self.group_size - shape[1] % self.group_size
                t = tensor.reshape(*shape[:-1], -1)[..., :-deficiency]
        return t


class IntegerQuantizer(BaseQuantizer):
    def __init__(self, bit, symmetric, granularity, **kwargs):
        super().__init__(bit, symmetric, granularity, **kwargs)
        self.quant_type = 'int-quant'
        if 'int_range' in self.kwargs:
            self.qmin = self.kwargs['int_range'][0]
            self.qmax = self.kwargs['int_range'][1]
        else:
            if self.sym:
                self.qmin = -(2 ** (self.bit - 1))
                self.qmax = 2 ** (self.bit - 1) - 1
            else:
                self.qmin = 0.0
                self.qmax = 2**self.bit - 1

        self.qmin = torch.tensor(self.qmin)
        self.qmax = torch.tensor(self.qmax)

    def quant(self, tensor, scales, zeros, qmax, qmin):
        if self.round_zp:
            tensor = torch.clamp(self.round_func(tensor / scales) + zeros, qmin, qmax)
        else:
            tensor = torch.clamp(
                self.round_func(tensor / scales.clamp_min(1e-9) + zeros),
                qmin,
                qmax,
            )
        return tensor

    def dequant(self, tensor, scales, zeros):
        tensor = (tensor - zeros) * scales
        return tensor

    def quant_dequant(self, tensor, scales, zeros, qmax, qmin, output_scale_factor=1):
        tensor = self.quant(tensor, scales, zeros, qmax, qmin)
        tensor = self.dequant(tensor, scales * output_scale_factor, zeros)
        return tensor

    def fake_quant_act_static(self, act, args={}):
        if 'int_indices' in args:
            q_act = act[:, :, args['int_indices']]
            fp_act = act[:, :, args['fp_indices']]
        else:
            q_act = act

        if 'current_bit' in args:
            org_bit = self.bit
            self.bit = args['current_bit']

        org_act_shape = q_act.shape
        org_act_dtype = q_act.dtype

        scales, zeros, qmax, qmin = (
            args['scales'],
            args['zeros'],
            args['qmax'],
            args['qmin'],
        )
        q_act = self.reshape_tensor(q_act)
        q_act = self.quant_dequant(q_act, scales, zeros, qmax, qmin)
        q_act = self.restore_tensor(q_act, org_act_shape).to(org_act_dtype)

        if 'current_bit' in args:
            self.bit = org_bit

        if 'int_indices' in args:
            mix_act = torch.zeros_like(act)
            mix_act[:, :, args['int_indices']] = q_act
            mix_act[:, :, args['fp_indices']] = fp_act
            return mix_act

        return q_act

    def fake_quant_act_dynamic(self, act, args={}):
        if 'int_indices' in args:
            q_act = act[:, :, args['int_indices']]
            fp_act = act[:, :, args['fp_indices']]
        else:
            q_act = act

        if 'current_bit' in args:
            org_bit = self.bit
            self.bit = args['current_bit']

        org_act_shape = q_act.shape
        org_act_dtype = q_act.dtype

        q_act, scales, zeros, qmax, qmin = self.get_tensor_qparams(q_act, args)
        q_act = self.quant_dequant(q_act, scales, zeros, qmax, qmin)

        q_act = self.restore_tensor(q_act, org_act_shape).to(org_act_dtype)

        if 'current_bit' in args:
            self.bit = org_bit

        if 'int_indices' in args:
            mix_act = torch.zeros_like(act)
            mix_act[:, :, args['int_indices']] = q_act
            mix_act[:, :, args['fp_indices']] = fp_act
            return mix_act
        if self.ste_all:
            return (q_act - act).detach() + act
        return q_act

    def fake_quant_weight_static(self, weight, args):
        if 'int_indices' in args:
            if self.granularity == 'per_group':
                assert len(args['int_indices']) % self.group_size == 0
            q_weight = weight[:, args['int_indices']]
            fp_weight = weight[:, args['fp_indices']]

        elif 'dim' in args and 'ic' in args['dim']:
            q_weight = weight.T
        else:
            q_weight = weight

        if 'rounding' in args:
            org_round_func = self.round_func
            self.round_func = lambda x: torch.floor(x) + args['rounding']

        org_w_shape = q_weight.shape
        org_w_dtype = q_weight.dtype
        scales, zeros, qmax, qmin = (
            args['scales'],
            args['zeros'],
            args['qmax'],
            args['qmin'],
        )
        output_scale_factor = (
            args['output_scale_factor'] if 'output_scale_factor' in args else 1
        )

        q_weight = self.reshape_tensor(q_weight)
        q_weight = self.quant_dequant(
            q_weight, scales, zeros, qmax, qmin, output_scale_factor
        )
        q_weight = self.restore_tensor(q_weight, org_w_shape).to(org_w_dtype)

        if 'int_indices' in args:
            mix_weight = torch.zeros_like(weight)
            mix_weight[:, args['int_indices']] = q_weight
            mix_weight[:, args['fp_indices']] = fp_weight
            return mix_weight

        elif 'dim' in args and 'ic' in args['dim']:
            q_weight = q_weight.T

        if 'rounding' in args:
            self.round_func = org_round_func

        return q_weight

    def fake_quant_weight_dynamic(self, weight, args={}):
        if 'int_indices' in args:
            if self.granularity == 'per_group':
                assert len(args['int_indices']) % self.group_size == 0
            q_weight = weight[:, args['int_indices']]
            fp_weight = weight[:, args['fp_indices']]

        elif 'dim' in args and 'ic' in args['dim']:
            q_weight = weight.T
        else:
            q_weight = weight

        if 'current_bit' in args:
            org_bit = self.bit
            self.bit = args['current_bit']

        org_w_shape = q_weight.shape
        org_w_dtype = q_weight.dtype

        q_weight, scales, zeros, qmax, qmin = self.get_tensor_qparams(q_weight, args)
        q_weight = self.quant_dequant(q_weight, scales, zeros, qmax, qmin)

        q_weight = self.restore_tensor(q_weight, org_w_shape).to(org_w_dtype)

        if 'current_bit' in args:
            self.bit = org_bit

        if 'int_indices' in args:
            mix_weight = torch.zeros_like(weight)
            mix_weight[:, args['int_indices']] = q_weight
            mix_weight[:, args['fp_indices']] = fp_weight
            return mix_weight

        elif 'dim' in args and 'ic' in args['dim']:
            q_weight = q_weight.T

        return q_weight

    def real_quant_weight_static(self, weight, args):
        org_w_shape = weight.shape
        if 'output_scale_factor' in args:
            output_scale_factor = args['output_scale_factor']
            del args['output_scale_factor']
        else:
            output_scale_factor = 1
        scales, zeros, qmax, qmin = (
            args['scales'],
            args['zeros'],
            args['qmax'],
            args['qmin'],
        )
        weight = self.reshape_tensor(weight)
        weight = self.quant(weight, scales, zeros, qmax, qmin)
        weight = self.restore_tensor(weight, org_w_shape)

        scales = scales * output_scale_factor

        if self.bit == 8:
            if self.qmin != 0:
                dtype = torch.int8
            else:
                dtype = torch.uint8
        else:
            dtype = torch.int32
        weight = weight.to(dtype)
        if not self.sym and self.round_zp:
            zeros = zeros.to(dtype)
        elif self.sym:
            zeros = None

        if self.granularity == 'per_tensor':
            qparams_shape = 1
        else:
            qparams_shape = (weight.shape[0], -1)

        if zeros is not None:
            zeros = zeros.view(qparams_shape)
        scales = scales.view(qparams_shape)

        return weight, scales, zeros

    def real_quant_weight_dynamic(self, weight, args={}):
        org_w_shape = weight.shape
        if 'output_scale_factor' in args:
            output_scale_factor = args['output_scale_factor']
            del args['output_scale_factor']
        else:
            output_scale_factor = 1
        weight, scales, zeros, qmax, qmin = self.get_tensor_qparams(weight, args)
        weight = self.quant(weight, scales, zeros, qmax, qmin)
        weight = self.restore_tensor(weight, org_w_shape)

        scales = scales * output_scale_factor

        if self.bit == 8:
            if self.qmin != 0:
                dtype = torch.int8
            else:
                dtype = torch.uint8
        else:
            dtype = torch.int32
        weight = weight.to(dtype)
        if not self.sym and self.round_zp:
            zeros = zeros.to(dtype)
        elif self.sym:
            zeros = None

        if self.granularity == 'per_tensor':
            qparams_shape = 1
        else:
            qparams_shape = (weight.shape[0], -1)

        if zeros is not None:
            zeros = zeros.view(qparams_shape)
        scales = scales.view(qparams_shape)

        return weight, scales, zeros

    def __repr__(self):
        return (
            f'IntegerQuantizer(bit={self.bit}, sym={self.sym},'
            f'granularity={self.granularity},'
            f'kwargs={self.kwargs}, qmin={self.qmin}, qmax={self.qmax})'
        )


class FloatQuantizer(BaseQuantizer):
    def __init__(self, bit, symmetric, granularity, **kwargs):
        super().__init__(bit, symmetric, granularity, **kwargs)
        self.sym = True
        self.quant_type = 'float-quant'
        self.e_bits = int(self.bit[1])
        self.m_bits = int(self.bit[-1])
        self.sign_bits = 1
        self.num_bits = self.e_bits + self.m_bits + self.sign_bits
        self.default_bias = 2 ** (self.e_bits - 1)

        self.use_qtorch = self.kwargs.get('use_qtorch')
        if self.use_qtorch:
            try:
                from qtorch.quant import float_quantize
            except ImportError:
                logger.error('qtorch not found, please install qtorch.')
                raise ImportError('Please install qtorch (pip install qtorch).')

            self.float_quantize = float_quantize

            if 'float_range' in self.kwargs:
                self.qmin, self.qmax = self.kwargs['float_range']
            else:
                bit_ranges = {
                    ('e4m3', 8): torch.float8_e4m3fn,
                    ('e5m2', 8): torch.float8_e5m2,
                    ('e3m2', 6): (-28, 28),
                    ('e4m7', 12): (-510, 510),
                    ('e2m1', 4): (-6, 6),
                }

                key = (self.bit, self.num_bits)
                if key in bit_ranges:
                    if isinstance(bit_ranges[key], tuple):
                        self.qmin, self.qmax = bit_ranges[key]
                    else:
                        finfo = torch.finfo(bit_ranges[key])
                        self.qmin, self.qmax = finfo.min, finfo.max
                else:
                    raise NotImplementedError(
                        'Only 4, 6, 8, and \
                                                12-bit quantization is supported.'
                    )
            self.qmax = torch.tensor(self.qmax)
            self.qmin = torch.tensor(self.qmin)

    def get_float_qparams(self, tensor, tensor_range, device):
        min_val, max_val = tensor_range[0], tensor_range[1]
        maxval = torch.max(max_val, -min_val)

        e_bits = torch.tensor(self.e_bits, dtype=torch.float32).cuda()
        m_bits = torch.tensor(self.m_bits, dtype=torch.float32).cuda()

        if maxval.shape[0] != 1 and len(maxval.shape) != len(tensor.shape):
            maxval = maxval.view([-1] + [1] * (len(tensor.shape) - 1))

        if e_bits >= 5:
            maxval = maxval.to(dtype=torch.float32)

        bias = 2**e_bits - torch.log2(maxval) + torch.log2(2 - 2 ** (-m_bits)) - 1

        xc = torch.min(torch.max(tensor, -maxval), maxval)

        log_scales = torch.clamp(
            (torch.floor(torch.log2(torch.abs(xc)) + bias)).detach(), 1.0
        )
        scales = 2.0 ** (log_scales - m_bits - bias)

        return xc, scales

    def get_tensor_qparams(self, tensor, args={}):
        tensor = self.reshape_tensor(tensor)
        tensor_range = self.get_tensor_range(tensor, args)
        if self.use_qtorch:
            scales, zeros, qmax, qmin = self.get_qparams(tensor_range, tensor.device)
        else:
            tensor, scales = self.get_float_qparams(tensor, tensor_range, tensor.device)
            zeros, qmin, qmax = torch.tensor(0), None, None

        return tensor, scales, zeros, qmax, qmin

    def quant(self, tensor, scales, zeros, qmax, qmin):
        scales[scales == 0] = 1
        scaled_tensor = tensor / scales + zeros
        if self.use_qtorch:
            org_dtype = scaled_tensor.dtype
            q_tensor = self.float_quantize(
                scaled_tensor.float(), self.e_bits, self.m_bits, rounding='nearest'
            )
            q_tensor.to(org_dtype)
        else:
            q_tensor = self.round_func(scaled_tensor)
        return q_tensor

    def dequant(self, tensor, scales, zeros):
        tensor = (tensor - zeros) * scales
        return tensor

    def quant_dequant(self, tensor, scales, zeros, qmax, qmin):
        tensor = self.quant(tensor, scales, zeros, qmax, qmin)
        tensor = self.dequant(tensor, scales, zeros)
        return tensor

    def fake_quant_act_static(self, act, args={}):
        q_act = act
        org_act_shape = q_act.shape
        org_act_dtype = q_act.dtype

        scales, zeros, qmax, qmin = (
            args['scales'],
            args['zeros'],
            args['qmax'],
            args['qmin'],
        )
        q_act = self.reshape_tensor(q_act)
        q_act = self.quant_dequant(q_act, scales, zeros, qmax, qmin)
        q_act = self.restore_tensor(q_act, org_act_shape).to(org_act_dtype)

        return q_act

    def fake_quant_act_dynamic(self, act, args={}):
        q_act = act
        org_act_shape = q_act.shape
        org_act_dtype = q_act.dtype

        q_act, scales, zeros, qmax, qmin = self.get_tensor_qparams(q_act, args)
        q_act = self.quant_dequant(q_act, scales, zeros, qmax, qmin)

        q_act = self.restore_tensor(q_act, org_act_shape).to(org_act_dtype)
        return q_act

    def fake_quant_weight_static(self, weight, args):

        if 'dim' in args and 'ic' in args['dim']:
            q_weight = weight.T
        else:
            q_weight = weight

        if 'rounding' in args:
            org_round_func = self.round_func
            self.round_func = lambda x: torch.floor(x) + args['rounding']

        org_w_shape = q_weight.shape
        org_w_dtype = q_weight.dtype
        scales, zeros, qmax, qmin = (
            args['scales'],
            args['zeros'],
            args['qmax'],
            args['qmin'],
        )
        q_weight = self.reshape_tensor(q_weight)
        q_weight = self.quant_dequant(q_weight, scales, zeros, qmax, qmin)
        q_weight = self.restore_tensor(q_weight, org_w_shape).to(org_w_dtype)

        if 'dim' in args and 'ic' in args['dim']:
            q_weight = q_weight.T

        if 'rounding' in args:
            self.round_func = org_round_func

        return q_weight

    def fake_quant_weight_dynamic(self, weight, args={}):

        if 'dim' in args and 'ic' in args['dim']:
            q_weight = weight.T
        else:
            q_weight = weight

        org_w_shape = q_weight.shape
        org_w_dtype = q_weight.dtype

        q_weight, scales, zeros, qmax, qmin = self.get_tensor_qparams(q_weight, args)
        q_weight = self.quant_dequant(q_weight, scales, zeros, qmax, qmin)
        q_weight = self.restore_tensor(q_weight, org_w_shape).to(org_w_dtype)

        if 'dim' in args and 'ic' in args['dim']:
            q_weight = q_weight.T

        return q_weight

    def real_quant_weight_static(self, weight, args):
        assert self.bit in ['e4m3', 'e5m2'], 'Only FP8 E4M3 and E5M2 support real quant'
        dtype = torch.float8_e4m3fn if self.e_bits == 4 else torch.float8_e5m2

        org_w_shape = weight.shape
        if 'output_scale_factor' in args:
            output_scale_factor = args['output_scale_factor']
            del args['output_scale_factor']
        else:
            output_scale_factor = 1
        scales, zeros, qmax, qmin = (
            args['scales'],
            args['zeros'],
            args['qmax'],
            args['qmin'],
        )
        weight = self.reshape_tensor(weight)
        weight = self.quant(weight, scales, zeros, qmax, qmin)
        weight = self.restore_tensor(weight, org_w_shape)

        scales = scales * output_scale_factor

        weight = weight.to(dtype)
        zeros = None
        if self.granularity == 'per_tensor':
            qparams_shape = 1
        else:
            qparams_shape = (weight.shape[0], -1)

        scales = scales.view(qparams_shape)
        return weight, scales, zeros

    def real_quant_weight_dynamic(self, weight, args={}):
        assert self.bit in ['e4m3', 'e5m2'], 'Only FP8 E4M3 and E5M2 support real quant'
        dtype = torch.float8_e4m3fn if self.e_bits == 4 else torch.float8_e5m2

        org_w_shape = weight.shape
        if 'output_scale_factor' in args:
            output_scale_factor = args['output_scale_factor']
            del args['output_scale_factor']
        else:
            output_scale_factor = 1
        weight, scales, zeros, qmax, qmin = self.get_tensor_qparams(weight, args)
        weight = self.quant(weight, scales, zeros, qmax, qmin)
        weight = self.restore_tensor(weight, org_w_shape)

        scales = scales * output_scale_factor

        weight = weight.to(dtype)
        zeros = None
        if self.granularity == 'per_tensor':
            qparams_shape = 1
        else:
            qparams_shape = (weight.shape[0], -1)

        scales = scales.view(qparams_shape)
        return weight, scales, zeros

    def __repr__(self):
        return (
            f'FloatQuantizer(bit={self.bit},'
            f'e_bits={self.e_bits}, m_bits={self.m_bits},'
            f'granularity={self.granularity},'
            f'kwargs={self.kwargs}, qmin={self.qmin}, qmax={self.qmax})'
        )
