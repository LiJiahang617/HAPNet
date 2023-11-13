# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch.nn as nn
from mmcv_custom.cnn import ConvModule, build_conv_layer
from mmengine_custom.model import BaseModule
from mmengine_custom.model.weight_init import (constant_init, kaiming_init,
                                        normal_init, update_init_info,
                                        xavier_init)
from torch import Tensor
from torch.nn import init

from mmseg_custom.registry import MODELS
from mmseg_custom.models.decode_heads.decode_head import BaseDecodeHead


def generation_init_weights(module, init_type='normal', init_gain=0.02):
    """Default initialization of network weights for image generation.

    By default, we use normal init, but xavier and kaiming might work
    better for some applications.

    Args:
        module (nn.Module): Module to be initialized.
        init_type (str): The name of an initialization method:
            normal | xavier | kaiming | orthogonal. Default: 'normal'.
        init_gain (float): Scaling factor for normal, xavier and
            orthogonal. Default: 0.02.
    """

    def init_func(m):
        """Initialization function.

        Args:
            m (nn.Module): Module to be initialized.
        """
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1
                                     or classname.find('Linear') != -1):
            if init_type == 'normal':
                normal_init(m, 0.0, init_gain)
            elif init_type == 'xavier':
                xavier_init(m, gain=init_gain, distribution='normal')
            elif init_type == 'kaiming':
                kaiming_init(
                    m,
                    a=0,
                    mode='fan_in',
                    nonlinearity='leaky_relu',
                    distribution='normal')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight, gain=init_gain)
                init.constant_(m.bias.data, 0.0)
            else:
                raise NotImplementedError(
                    f"Initialization method '{init_type}' is not implemented")
            init_info = (f'Initialize {m.__class__.__name__} by \'init_type\' '
                         f'{init_type}.')
        elif classname.find('BatchNorm2d') != -1:
            # BatchNorm Layer's weight is not a matrix;
            # only normal distribution applies.
            normal_init(m, 1.0, init_gain)
            init_info = (f'{m.__class__.__name__} is BatchNorm2d, initialize '
                         'by Norm initialization with mean=1, '
                         f'std={init_gain}')

        if hasattr(m, '_params_init_info'):
            update_init_info(module, init_info)

    module.apply(init_func)


@MODELS.register_module()
class PatchDiscriminator(BaseDecodeHead):
    """A PatchGAN discriminator.

    Args:
        in_channels (int): Number of channels in input images.
        base_channels (int): Number of channels at the first conv layer.
            Default: 64.
        num_conv (int): Number of stacked intermediate convs (excluding input
            and output conv). Default: 3.
        norm_cfg (dict): Config dict to build norm layer. Default:
            `dict(type='BN')`.
        init_cfg (dict): Config dict for initialization.
            `type`: The name of our initialization method. Default: 'normal'.
            `gain`: Scaling factor for normal, xavier and orthogonal.
            Default: 0.02.
    """

    def __init__(self,
                 in_channels: int,
                 base_channels: int = 64,
                 num_conv: int = 3,
                 norm_cfg: dict = dict(type='BN'),
                 init_cfg: Optional[dict] = dict(type='normal', gain=0.02)):
        super().__init__(init_cfg=init_cfg)
        assert isinstance(norm_cfg, dict), ("'norm_cfg' should be dict, but"
                                            f'got {type(norm_cfg)}')
        assert 'type' in norm_cfg, "'norm_cfg' must have key 'type'"
        # We use norm layers in the patch discriminator.
        # Only for IN, use bias since it does not have affine parameters.
        use_bias = norm_cfg['type'] == 'IN'

        kernel_size = 4
        padding = 1

        # input layer
        sequence = [
            ConvModule(
                in_channels=in_channels,
                out_channels=base_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=padding,
                bias=True,
                norm_cfg=None,
                act_cfg=dict(type='LeakyReLU', negative_slope=0.2))
        ]

        # stacked intermediate layers,
        # gradually increasing the number of filters
        multiple_now = 1
        multiple_prev = 1
        for n in range(1, num_conv):
            multiple_prev = multiple_now
            multiple_now = min(2**n, 8)
            sequence += [
                ConvModule(
                    in_channels=base_channels * multiple_prev,
                    out_channels=base_channels * multiple_now,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=padding,
                    bias=use_bias,
                    norm_cfg=norm_cfg,
                    act_cfg=dict(type='LeakyReLU', negative_slope=0.2))
            ]
        multiple_prev = multiple_now
        multiple_now = min(2**num_conv, 8)
        sequence += [
            ConvModule(
                in_channels=base_channels * multiple_prev,
                out_channels=base_channels * multiple_now,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                bias=use_bias,
                norm_cfg=norm_cfg,
                act_cfg=dict(type='LeakyReLU', negative_slope=0.2))
        ]

        # output one-channel prediction map
        sequence += [
            build_conv_layer(
                dict(type='Conv2d'),
                base_channels * multiple_now,
                1,
                kernel_size=kernel_size,
                stride=1,
                padding=padding)
        ]

        self.model = nn.Sequential(*sequence)
        self.init_type = 'normal' if init_cfg is None else init_cfg.get(
            'type', 'normal')
        self.init_gain = 0.02 if init_cfg is None else init_cfg.get(
            'gain', 0.02)

    def forward(self, x: Tensor) -> Tensor:
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        return self.model(x)

    def init_weights(self) -> None:
        """Initialize weights for the model.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Default: None.
        """
        if self.init_cfg is None and self.init_cfg['type'] == 'Pretrained':
            super().init_weights()
            return
        generation_init_weights(
            self, init_type=self.init_type, init_gain=self.init_gain)
        self._is_init = True
