import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg_custom.registry import MODELS


@MODELS.register_module()
class SILogLoss(nn.Module):  # pixel-wise loss function used in AdaBins
    def __init__(self,
                 loss_weight: float = 1.0,
                 loss_name: str = 'loss_silog'):
        super(SILogLoss, self).__init__()
        self.loss_weight = loss_weight
        self._loss_name = loss_name

    def forward(self, input, target, mask=None, interpolate=True):
        if interpolate:
            input = nn.functional.interpolate(input, target.shape[-2:], mode='bilinear', align_corners=True)

        if mask is not None:
            input = input[mask]
            target = target[mask]
        alpha = 1e-3
        g = torch.log(alpha + input) - torch.log(alpha + target)
        # n, c, h, w = g.shape
        # norm = 1/(h*w)
        # Dg = norm * torch.sum(g**2) - (0.85/(norm**2)) * (torch.sum(g))**2

        Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
        return 10 * torch.sqrt(Dg) * self.loss_weight

    @property
    def loss_name(self):
        return self._loss_name