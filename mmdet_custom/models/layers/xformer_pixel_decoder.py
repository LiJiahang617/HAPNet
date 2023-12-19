# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv_custom.cnn import Conv2d, ConvModule
from mmcv_custom.cnn.bricks.transformer import MultiScaleDeformableAttention
from mmengine_custom.model import (BaseModule, ModuleList, caffe2_xavier_init,
                                   normal_init, xavier_init)
from torch import Tensor

from mmdet_custom.registry import MODELS
from mmdet_custom.utils import ConfigType, OptMultiConfig
from ..task_modules.prior_generators import MlvlPointGenerator
from .positional_encoding import SinePositionalEncoding
from .transformer import Mask2FormerTransformerEncoder


@MODELS.register_module()
class XFormerPixelDecoder(BaseModule):
    """Pixel decoder with multi-scale deformable attention in XFormer.

    Args:
        in_channels (list[int] | tuple[int]): Number of channels in the
            input feature maps.
        strides (list[int] | tuple[int]): Output strides of feature from
            backbone.
        feat_channels (int): Number of channels for feature.
        out_channels (int): Number of channels for output.
        num_outs (int): Number of output scales.
        norm_cfg (:obj:`ConfigDict` or dict): Config for normalization.
            Defaults to dict(type='GN', num_groups=32).
        act_cfg (:obj:`ConfigDict` or dict): Config for activation.
            Defaults to dict(type='ReLU').
        encoder (:obj:`ConfigDict` or dict): Config for transformer
            encoder. Defaults to None.
        positional_encoding (:obj:`ConfigDict` or dict): Config for
            transformer encoder position encoding. Defaults to
            dict(num_feats=128, normalize=True).
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict], optional): Initialization config dict. Defaults to None.
    """

    def __init__(self,
                 in_channels: Union[List[int],
                                    Tuple[int]] = [256, 512, 1024, 2048],
                 strides: Union[List[int], Tuple[int]] = [4, 8, 16, 32],
                 feat_channels: int = 256,
                 out_channels: int = 256,
                 num_outs: int = 3,
                 norm_cfg: ConfigType = dict(type='GN', num_groups=32),
                 act_cfg: ConfigType = dict(type='ReLU'),
                 encoder: ConfigType = None,
                 positional_encoding: ConfigType = dict(
                     num_feats=128, normalize=True),
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.strides = strides
        self.num_input_levels = len(in_channels)
        self.num_encoder_levels = \
            encoder.layer_cfg.self_attn_cfg.num_levels
        assert self.num_encoder_levels >= 1, \
            'num_levels in attn_cfgs must be at least one'
        input_conv_list = []
        # from top to down (low to high resolution)
        for i in range(self.num_input_levels - 1,
                       self.num_input_levels - self.num_encoder_levels - 1,
                       -1):
            input_conv = ConvModule(
                in_channels[i],
                feat_channels,
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=None,
                bias=True)
            input_conv_list.append(input_conv)
        self.input_convs = ModuleList(input_conv_list)
        # encoder is a standard Transformer self_attn encoder stacked-layer
        self.encoder = Mask2FormerTransformerEncoder(**encoder)
        self.postional_encoding = SinePositionalEncoding(**positional_encoding)
        # high resolution to low resolution
        self.level_encoding = nn.Embedding(self.num_encoder_levels,
                                           feat_channels)

        # fpn-like structure
        self.lateral_convs = ModuleList()
        self.output_convs = ModuleList()
        self.use_bias = norm_cfg is None
        # from top to down (low to high resolution)
        # fpn for the rest features that didn't pass in encoder
        for i in range(self.num_input_levels - self.num_encoder_levels - 1, -1,
                       -1):
            lateral_conv = ConvModule(
                in_channels[i],
                feat_channels,
                kernel_size=1,
                bias=self.use_bias,
                norm_cfg=norm_cfg,
                act_cfg=None)
            output_conv = ConvModule(
                feat_channels,
                feat_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=self.use_bias,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            self.lateral_convs.append(lateral_conv)
            self.output_convs.append(output_conv)

        self.mask_feature = Conv2d(
            feat_channels, out_channels, kernel_size=1, stride=1, padding=0)

        self.num_outs = num_outs
        self.point_generator = MlvlPointGenerator(strides)

    def init_weights(self) -> None:
        """Initialize weights."""
        for i in range(0, self.num_encoder_levels):
            xavier_init(
                self.input_convs[i].conv,
                gain=1,
                bias=0,
                distribution='uniform')

        for i in range(0, self.num_input_levels - self.num_encoder_levels):
            caffe2_xavier_init(self.lateral_convs[i].conv, bias=0)
            caffe2_xavier_init(self.output_convs[i].conv, bias=0)

        caffe2_xavier_init(self.mask_feature, bias=0)

        normal_init(self.level_encoding, mean=0, std=1)
        for p in self.encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

        # init_weights defined in MultiScaleDeformableAttention
        for m in self.encoder.layers.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()

    def forward(self, feats: List[Tensor]) -> Tuple[Tensor, Tensor]:

        feats_RGB = []
        feats_X = []
        for feat in feats:
            feat_RGB, feat_X = torch.chunk(feat, 2, dim=1)
            feats_RGB.append(feat_RGB)
            feats_X.append(feat_X)

