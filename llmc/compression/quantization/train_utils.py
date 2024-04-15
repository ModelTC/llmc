import torch
import torch.nn as nn

import sys
import os
import time
from math import inf
from loguru import logger


class TruncateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold):
        truncated_tensor = input.clone()
        truncated_tensor[truncated_tensor.abs() < threshold] = (
            truncated_tensor[truncated_tensor.abs() < threshold].sign() * threshold
        )
        return truncated_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None


class LossFunction:
    def __init__(self, method="mse", reduction="mean", dim=0):
        self.method = method
        self.reduction = reduction
        self.dim = dim

    def __call__(self, f_out, q_out):
        # MSE Loss
        if self.method == "mse":
            mse_loss = nn.MSELoss(reduction=self.reduction)
            return mse_loss(f_out, q_out)

        # Distribution Loss
        elif self.method == "dist":
            mse_loss = nn.MSELoss(reduction=self.reduction)

            channel_num = f_out.shape[-1]
            f_out = f_out.reshape(-1, channel_num)
            q_out = q_out.reshape(-1, channel_num)

            mean_error = mse_loss(f_out.mean(dim=self.dim), q_out.mean(dim=self.dim))
            std_error = mse_loss(f_out.std(dim=self.dim), q_out.std(dim=self.dim))
            return mean_error + std_error

        # KL divergence Loss
        elif self.method == "kl":
            kl_loss = nn.KLDivLoss(reduction=self.reduction)
            return kl_loss(f_out, q_out)


class NativeScalerWithGradNormCount:
    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(
        self,
        loss,
        optimizer,
        clip_grad=None,
        parameters=None,
        create_graph=False,
        update_grad=True,
        retain_graph=False,
    ):
        self._scaler.scale(loss).backward(
            create_graph=create_graph, retain_graph=retain_graph
        )
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = self.ampscaler_get_grad_norm(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def ampscaler_get_grad_norm(self, parameters, norm_type=2.0):
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        norm_type = float(norm_type)
        if len(parameters) == 0:
            return torch.tensor(0.0)
        device = parameters[0].grad.device
        if norm_type == inf:
            total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
        else:
            total_norm = torch.norm(
                torch.stack(
                    [
                        torch.norm(p.grad.detach(), norm_type).to(device)
                        for p in parameters
                    ]
                ),
                norm_type,
            )
        return total_norm
