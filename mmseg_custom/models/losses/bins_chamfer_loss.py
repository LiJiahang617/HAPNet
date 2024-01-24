import torch
import torch.nn as nn


from mmseg_custom.registry import MODELS
from pytorch3d.loss import chamfer_distance
from torch.nn.utils.rnn import pad_sequence


@MODELS.register_module()
class BinsChamferLoss(nn.Module):  # Bin centers regularizer used in AdaBins paper
    def __init__(self,
        loss_weight: float = 1.0,
        loss_name : str = 'loss_chamfer'):
        super().__init__()
        self.loss_weight = loss_weight
        self._loss_name = loss_name

    def forward(self, bins, target_depth_maps):
        bin_centers = 0.5 * (bins[:, 1:] + bins[:, :-1])
        n, p = bin_centers.shape
        input_points = bin_centers.view(n, p, 1)  # .shape = n, p, 1
        # n, c, h, w = target_depth_maps.shape

        target_points = target_depth_maps.flatten(1)  # n, hwc
        mask = target_points.ge(1e-3)  # only valid ground truth points
        target_points = [p[m] for p, m in zip(target_points, mask)]
        target_lengths = torch.Tensor([len(t) for t in target_points]).long().to(target_depth_maps.device)
        target_points = pad_sequence(target_points, batch_first=True).unsqueeze(2)  # .shape = n, T, 1

        loss, _ = chamfer_distance(x=input_points, y=target_points, y_lengths=target_lengths)
        return loss * self.loss_weight

    @property
    def loss_name(self):
        return self._loss_name