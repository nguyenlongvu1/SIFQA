from __future__ import annotations

import torch


class _GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambd * grad_output, None


def grl(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return _GRL.apply(x, lambd)
