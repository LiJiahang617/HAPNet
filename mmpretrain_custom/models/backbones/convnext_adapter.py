# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial
from itertools import chain
from typing import Sequence
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv_custom.cnn.bricks import DropPath
from mmcv_custom.ops import MultiScaleDeformableAttention
from mmengine_custom.model import BaseModule, ModuleList, Sequential
from mmengine_custom.logging import print_log
from mmengine_custom.model import constant_init, kaiming_init, normal_init
from mmengine_custom.runner import CheckpointLoader

from mmpretrain_custom.registry import MODELS
from ..utils import GRN, build_norm_layer
from .base_backbone import BaseBackbone


class ConvNeXtBlock(BaseModule):
    """ConvNeXt Block.

    Args:
        in_channels (int): The number of input channels.
        dw_conv_cfg (dict): Config of depthwise convolution.
            Defaults to ``dict(kernel_size=7, padding=3)``.
        norm_cfg (dict): The config dict for norm layers.
            Defaults to ``dict(type='LN2d', eps=1e-6)``.
        act_cfg (dict): The config dict for activation between pointwise
            convolution. Defaults to ``dict(type='GELU')``.
        mlp_ratio (float): The expansion ratio in both pointwise convolution.
            Defaults to 4.
        linear_pw_conv (bool): Whether to use linear layer to do pointwise
            convolution. More details can be found in the note.
            Defaults to True.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        layer_scale_init_value (float): Init value for Layer Scale.
            Defaults to 1e-6.

    Note:
        There are two equivalent implementations:

        1. DwConv -> LayerNorm -> 1x1 Conv -> GELU -> 1x1 Conv;
           all outputs are in (N, C, H, W).
        2. DwConv -> LayerNorm -> Permute to (N, H, W, C) -> Linear -> GELU
           -> Linear; Permute back

        As default, we use the second to align with the official repository.
        And it may be slightly faster.
    """

    def Ï__init__(self,
                 in_channels,
                 dw_conv_cfg=dict(kernel_size=7, padding=3),
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 mlp_ratio=4.,
                 linear_pw_conv=True,
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 use_grn=False,
                 with_cp=False):
        super().__init__()
        self.with_cp = with_cp

        self.depthwise_conv = nn.Conv2d(
            in_channels, in_channels, groups=in_channels, **dw_conv_cfg)

        self.linear_pw_conv = linear_pw_conv
        self.norm = build_norm_layer(norm_cfg, in_channels)

        mid_channels = int(mlp_ratio * in_channels)
        if self.linear_pw_conv:
            # Use linear layer to do pointwise conv.
            pw_conv = nn.Linear
        else:
            pw_conv = partial(nn.Conv2d, kernel_size=1)

        self.pointwise_conv1 = pw_conv(in_channels, mid_channels)
        self.act = MODELS.build(act_cfg)
        self.pointwise_conv2 = pw_conv(mid_channels, in_channels)

        if use_grn:
            self.grn = GRN(mid_channels)
        else:
            self.grn = None

        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones((in_channels)),
            requires_grad=True) if layer_scale_init_value > 0 else None

        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):

        def _inner_forward(x):
            shortcut = x
            x = self.depthwise_conv(x)

            if self.linear_pw_conv:
                x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
                x = self.norm(x, data_format='channel_last')
                x = self.pointwise_conv1(x)
                x = self.act(x)
                if self.grn is not None:
                    x = self.grn(x, data_format='channel_last')
                x = self.pointwise_conv2(x)
                x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
            else:
                x = self.norm(x, data_format='channel_first')
                x = self.pointwise_conv1(x)
                x = self.act(x)

                if self.grn is not None:
                    x = self.grn(x, data_format='channel_first')
                x = self.pointwise_conv2(x)

            if self.gamma is not None:
                x = x.mul(self.gamma.view(1, -1, 1, 1))

            x = shortcut + self.drop_path(x)
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


def get_reference_points(spatial_shapes, device):
    reference_points_list = []
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
            torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
        ref_y = ref_y.reshape(-1)[None] / H_
        ref_x = ref_x.reshape(-1)[None] / W_
        ref = torch.stack((ref_x, ref_y), -1)
        reference_points_list.append(ref)
    reference_points = torch.cat(reference_points_list, 1)
    reference_points = reference_points[:, :, None]
    return reference_points

def deform_inputs(x):
    bs, c, h, w = x.shape
    # stage 0
    spatial_shapes = torch.as_tensor([(h // 8, w // 8),
                                      (h // 16, w // 16),
                                      (h // 32, w // 32)],
                                     dtype=torch.long, device=x.device)
    level_start_index = torch.cat((spatial_shapes.new_zeros(
        (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    reference_points = get_reference_points([(h // 4, w // 4)], x.device)
    deform_inputs0_1 = [reference_points, spatial_shapes, level_start_index]

    spatial_shapes = torch.as_tensor([(h // 4, w // 4)], dtype=torch.long, device=x.device)
    level_start_index = torch.cat((spatial_shapes.new_zeros(
        (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    reference_points = get_reference_points([(h // 8, w // 8),
                                             (h // 16, w // 16),
                                             (h // 32, w // 32)], x.device)
    deform_inputs0_2 = [reference_points, spatial_shapes, level_start_index]
    # stage 1
    spatial_shapes = torch.as_tensor([(h // 8, w // 8),
                                      (h // 16, w // 16),
                                      (h // 32, w // 32)],
                                     dtype=torch.long, device=x.device)
    level_start_index = torch.cat((spatial_shapes.new_zeros(
        (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    reference_points = get_reference_points([(h // 8, w // 8)], x.device)
    deform_inputs1_1 = [reference_points, spatial_shapes, level_start_index]

    spatial_shapes = torch.as_tensor([(h // 8, w // 8)], dtype=torch.long, device=x.device)
    level_start_index = torch.cat((spatial_shapes.new_zeros(
        (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    reference_points = get_reference_points([(h // 8, w // 8),
                                             (h // 16, w // 16),
                                             (h // 32, w // 32)], x.device)
    deform_inputs1_2 = [reference_points, spatial_shapes, level_start_index]
    # stage 2
    spatial_shapes = torch.as_tensor([(h // 8, w // 8),
                                      (h // 16, w // 16),
                                      (h // 32, w // 32)],
                                     dtype=torch.long, device=x.device)
    level_start_index = torch.cat((spatial_shapes.new_zeros(
        (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    reference_points = get_reference_points([(h // 16, w // 16)], x.device)
    deform_inputs2_1 = [reference_points, spatial_shapes, level_start_index]

    spatial_shapes = torch.as_tensor([(h // 16, w // 16)], dtype=torch.long, device=x.device)
    level_start_index = torch.cat((spatial_shapes.new_zeros(
        (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    reference_points = get_reference_points([(h // 8, w // 8),
                                             (h // 16, w // 16),
                                             (h // 32, w // 32)], x.device)
    deform_inputs2_2 = [reference_points, spatial_shapes, level_start_index]
    # stage 3
    spatial_shapes = torch.as_tensor([(h // 8, w // 8),
                                      (h // 16, w // 16),
                                      (h // 32, w // 32)],
                                     dtype=torch.long, device=x.device)
    level_start_index = torch.cat((spatial_shapes.new_zeros(
        (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    reference_points = get_reference_points([(h // 32, w // 32)], x.device)
    deform_inputs3_1 = [reference_points, spatial_shapes, level_start_index]

    spatial_shapes = torch.as_tensor([(h // 32, w // 32)], dtype=torch.long, device=x.device)
    level_start_index = torch.cat((spatial_shapes.new_zeros(
        (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    reference_points = get_reference_points([(h // 8, w // 8),
                                             (h // 16, w // 16),
                                             (h // 32, w // 32)], x.device)
    deform_inputs3_2 = [reference_points, spatial_shapes, level_start_index]

    return (deform_inputs0_1, deform_inputs0_2, deform_inputs1_1, deform_inputs1_2,
            deform_inputs2_1, deform_inputs2_2, deform_inputs3_1, deform_inputs3_2)


class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        n = N // 21
        x1 = x[:, 0:16 * n, :].transpose(1, 2).view(B, C, H * 2, W * 2).contiguous()
        x2 = x[:, 16 * n:20 * n, :].transpose(1, 2).view(B, C, H, W).contiguous()
        x3 = x[:, 20 * n:, :].transpose(1, 2).view(B, C, H // 2, W // 2).contiguous()
        x1 = self.dwconv(x1).flatten(2).transpose(1, 2)
        x2 = self.dwconv(x2).flatten(2).transpose(1, 2)
        x3 = self.dwconv(x3).flatten(2).transpose(1, 2)
        x = torch.cat([x1, x2, x3], dim=1)
        return x


class Extractor(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
                 with_cffn=True, cffn_ratio=0.25, drop=0., drop_path=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), with_cp=False):
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = MultiScaleDeformableAttention(embed_dims=dim, num_levels=n_levels, num_heads=num_heads,
                                 num_points=n_points, value_proj_ratio=deform_ratio)
        self.with_cffn = with_cffn
        self.with_cp = with_cp
        if with_cffn:
            self.ffn = ConvFFN(in_features=dim, hidden_features=int(dim * cffn_ratio), drop=drop)
            self.ffn_norm = norm_layer(dim)
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index, H, W):

        def _inner_forward(query, feat):

            attn = self.attn(self.query_norm(query), reference_points,
                             self.feat_norm(feat), spatial_shapes,
                             level_start_index, None)
            query = query + attn

            if self.with_cffn:
                query = query + self.drop_path(self.ffn(self.ffn_norm(query), H, W))
            return query

        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat)
        else:
            query = _inner_forward(query, feat)

        return query


class Injector(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values=0., with_cp=False):
        super().__init__()
        self.with_cp = with_cp
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = MultiScaleDeformableAttention(d_model=dim, n_levels=n_levels, n_heads=num_heads,
                                 n_points=n_points, ratio=deform_ratio)
        self.gamma = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index):

        def _inner_forward(query, feat):

            attn = self.attn(self.query_norm(query), reference_points,
                             self.feat_norm(feat), spatial_shapes,
                             level_start_index, None)
            return query + self.gamma * attn

        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat)
        else:
            query = _inner_forward(query, feat)

        return query


class InteractionBlock(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 drop=0., drop_path=0., with_cffn=True, cffn_ratio=0.25, init_values=0.,
                 deform_ratio=1.0, extra_extractor=False, with_cp=False):
        super().__init__()

        self.injector = Injector(dim=dim, n_levels=3, num_heads=num_heads, init_values=init_values,
                                 n_points=n_points, norm_layer=norm_layer, deform_ratio=deform_ratio,
                                 with_cp=with_cp)
        self.extractor = Extractor(dim=dim, n_levels=1, num_heads=num_heads, n_points=n_points,
                                   norm_layer=norm_layer, deform_ratio=deform_ratio, with_cffn=with_cffn,
                                   cffn_ratio=cffn_ratio, drop=drop, drop_path=drop_path, with_cp=with_cp)
        if extra_extractor:
            self.extra_extractors = nn.Sequential(*[
                Extractor(dim=dim, num_heads=num_heads, n_points=n_points, norm_layer=norm_layer,
                          with_cffn=with_cffn, cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                          drop=drop, drop_path=drop_path, with_cp=with_cp)
                for _ in range(2)
            ])
        else:
            self.extra_extractors = None

    def forward(self, x, c, blocks, deform_inputs1, deform_inputs2, H, W):
        x = self.injector(query=x, reference_points=deform_inputs1[0],
                          feat=c, spatial_shapes=deform_inputs1[1],
                          level_start_index=deform_inputs1[2])
        for idx, blk in enumerate(blocks):
            x = blk(x, H, W)
        c = self.extractor(query=c, reference_points=deform_inputs2[0],
                           feat=x, spatial_shapes=deform_inputs2[1],
                           level_start_index=deform_inputs2[2], H=H, W=W)
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(query=c, reference_points=deform_inputs2[0],
                              feat=x, spatial_shapes=deform_inputs2[1],
                              level_start_index=deform_inputs2[2], H=H, W=W)
        return x, c


@MODELS.register_module()
class TwinConvNeXt(BaseBackbone):
    """TwinConvNeXt backbone.

    A PyTorch implementation of `A ConvNet for the 2020s
    <https://arxiv.org/abs/2201.03545>`_ and
    `ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders
    <http://arxiv.org/abs/2301.00808>`_

    Modified from the `official repo
    <https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py>`_
    and `timm
    <https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/convnext.py>`_.

    To use ConvNeXt v2, please set ``use_grn=True`` and ``layer_scale_init_value=0.``.

    Args:
        arch (str | dict): The model's architecture. If string, it should be
            one of architecture in ``ConvNeXt.arch_settings``. And if dict, it
            should include the following two keys:

            - depths (list[int]): Number of blocks at each stage.
            - channels (list[int]): The number of channels at each stage.

            Defaults to 'tiny'.
        in_channels (int): Number of input image channels. Defaults to 3.
        stem_patch_size (int): The size of one patch in the stem layer.
            Defaults to 4.
        norm_cfg (dict): The config dict for norm layers.
            Defaults to ``dict(type='LN2d', eps=1e-6)``.
        act_cfg (dict): The config dict for activation between pointwise
            convolution. Defaults to ``dict(type='GELU')``.
        linear_pw_conv (bool): Whether to use linear layer to do pointwise
            convolution. Defaults to True.
        use_grn (bool): Whether to add Global Response Normalization in the
            blocks. Defaults to False.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        layer_scale_init_value (float): Init value for Layer Scale.
            Defaults to 1e-6.
        out_indices (Sequence | int): Output from which stages.
            Defaults to -1, means the last stage.
        frozen_stages (int): Stages to be frozen (all param fixed).
            Defaults to 0, which means not freezing any parameters.
        gap_before_final_norm (bool): Whether to globally average the feature
            map before the final norm layer. In the official repo, it's only
            used in classification task. Defaults to True.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Defaults to False.
        init_cfg (dict, optional): Initialization config dict
    """  # noqa: E501
    arch_settings = {
        'atto': {
            'depths': [2, 2, 6, 2],
            'channels': [40, 80, 160, 320]
        },
        'femto': {
            'depths': [2, 2, 6, 2],
            'channels': [48, 96, 192, 384]
        },
        'pico': {
            'depths': [2, 2, 6, 2],
            'channels': [64, 128, 256, 512]
        },
        'nano': {
            'depths': [2, 2, 8, 2],
            'channels': [80, 160, 320, 640]
        },
        'tiny': {
            'depths': [3, 3, 9, 3],
            'channels': [96, 192, 384, 768]
        },
        'small': {
            'depths': [3, 3, 27, 3],
            'channels': [96, 192, 384, 768]
        },
        'base': {
            'depths': [3, 3, 27, 3],
            'channels': [128, 256, 512, 1024]
        },
        'large': {
            'depths': [3, 3, 27, 3],
            'channels': [192, 384, 768, 1536]
        },
        'xlarge': {
            'depths': [3, 3, 27, 3],
            'channels': [256, 512, 1024, 2048]
        },
        'huge': {
            'depths': [3, 3, 27, 3],
            'channels': [352, 704, 1408, 2816]
        }
    }

    def __init__(self,
                 arch='tiny',
                 in_channels=3,
                 stem_patch_size=4,
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 linear_pw_conv=True,
                 use_grn=False,
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 out_indices=-1,
                 frozen_stages=0,
                 gap_before_final_norm=True,
                 with_cp=False,
                 embed_dims=256,
                 deform_num_heads=6,
                 with_cffn=True,
                 add_both_feature=True,
                 cffn_ratio=0.25,
                 deform_ratio=1.0,
                 init_values=0.,
                 num_points=4,
                 init_cfg=[
                     dict(
                         type='TruncNormal',
                         layer=['Conv2d', 'Linear'],
                         std=.02,
                         bias=0.),
                     dict(
                         type='Constant', layer=['LayerNorm'], val=1.,
                         bias=0.),
                 ]):
        super().__init__(init_cfg=init_cfg)

        if isinstance(arch, str):
            assert arch in self.arch_settings, \
                f'Unavailable arch, please choose from ' \
                f'({set(self.arch_settings)}) or pass a dict.'
            arch = self.arch_settings[arch]
        elif isinstance(arch, dict):
            assert 'depths' in arch and 'channels' in arch, \
                f'The arch dict must have "depths" and "channels", ' \
                f'but got {list(arch.keys())}.'

        self.depths = arch['depths']
        self.channels = arch['channels']
        self.embed_dims = embed_dims
        assert (isinstance(self.depths, Sequence)
                and isinstance(self.channels, Sequence)
                and len(self.depths) == len(self.channels)), \
            f'The "depths" ({self.depths}) and "channels" ({self.channels}) ' \
            'should be both sequence with the same length.'

        self.num_stages = len(self.depths)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = 4 + index
                assert out_indices[i] >= 0, f'Invalid out_indices {index}'
        self.out_indices = out_indices

        self.frozen_stages = frozen_stages
        self.gap_before_final_norm = gap_before_final_norm

        # stochastic depth decay rule
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(self.depths))
        ]
        block_idx = 0

        # rgb encoder
        self.downsample_layers_x = ModuleList()
        stem_x = nn.Sequential(
            nn.Conv2d(
                in_channels,
                self.channels[0],
                kernel_size=stem_patch_size,
                stride=stem_patch_size),
            build_norm_layer(norm_cfg, self.channels[0]),
        )
        self.downsample_layers_x.append(stem_x)
        # another encoder
        self.downsample_layers_y = ModuleList()
        stem_y = nn.Sequential(
            nn.Conv2d(
                in_channels,
                self.channels[0],
                kernel_size=stem_patch_size,
                stride=stem_patch_size),
            build_norm_layer(norm_cfg, self.channels[0]),
        )
        self.downsample_layers_y.append(stem_y)
        # 4 feature resolution stages, each consisting of multiple residual
        # blocks
        self.stages_x = nn.ModuleList()
        self.stages_y = nn.ModuleList()

        for i in range(self.num_stages):
            depth = self.depths[i]
            channels = self.channels[i]

            if i >= 1:
                # rgb encoder
                downsample_layer_x = nn.Sequential(
                    build_norm_layer(norm_cfg, self.channels[i - 1]),
                    nn.Conv2d(
                        self.channels[i - 1],
                        channels,
                        kernel_size=2,
                        stride=2),
                )
                self.downsample_layers_x.append(downsample_layer_x)
                # another encoder
                downsample_layer_y = nn.Sequential(
                    build_norm_layer(norm_cfg, self.channels[i - 1]),
                    nn.Conv2d(
                        self.channels[i - 1],
                        channels,
                        kernel_size=2,
                        stride=2),
                )
                self.downsample_layers_y.append(downsample_layer_y)
            # rgb encoder
            stage_x = Sequential(*[
                ConvNeXtBlock(
                    in_channels=channels,
                    drop_path_rate=dpr[block_idx + j],
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    linear_pw_conv=linear_pw_conv,
                    layer_scale_init_value=layer_scale_init_value,
                    use_grn=use_grn,
                    with_cp=with_cp) for j in range(depth)
            ])
            # another encoder
            stage_y = Sequential(*[
                ConvNeXtBlock(
                    in_channels=channels,
                    drop_path_rate=dpr[block_idx + j],
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    linear_pw_conv=linear_pw_conv,
                    layer_scale_init_value=layer_scale_init_value,
                    use_grn=use_grn,
                    with_cp=with_cp) for j in range(depth)
            ])
            block_idx += depth

            self.stages_x.append(stage_x)
            self.stages_y.append(stage_y)

            if i in self.out_indices:
                norm_layer_x = build_norm_layer(norm_cfg, channels)
                norm_layer_y = build_norm_layer(norm_cfg, channels)
                self.add_module(f'norm_x{i}', norm_layer_x)
                self.add_module(f'norm_y{i}', norm_layer_y)
        self.reduces_rgb = nn.ModuleList()
        self.reduces_x = nn.ModuleList()
        for i in range(self.num_stages):
            channels = self.channels[i]
            reduce_block_rgb = nn.Conv2d(channels, embed_dims, kernel_size=1, stride=1, padding=0, bias=True)
            reduce_block_x = nn.Conv2d(channels, embed_dims, kernel_size=1, stride=1, padding=0, bias=True)
            self.reduces_rgb.append(reduce_block_rgb)
            self.reduces_x.append(reduce_block_x)

        self.interactions = nn.Sequential(*[InteractionBlock(dim=embed_dims, num_heads=deform_num_heads,
                                                             n_points=num_points, init_values=init_values,
                                                             drop_path=drop_path_rate, norm_layer=build_norm_layer(norm_cfg, self.channels[i]),
                                                             with_cffn=with_cffn,cffn_ratio=cffn_ratio,
                                                             deform_ratio=deform_ratio, extra_extractor=
                                                             True if i == len(self.channels)-1 else False,
                                                             with_cp=with_cp)
                                            for i in range(len(self.channels))
                                            ])
        self._freeze_stages()

    def init_weights(self):
        if self.init_cfg is None:
            print_log(f'No pre-trained weights for '
                      f'{self.__class__.__name__}, '
                      f'training start from scratch')
            for name, m in self.named_modules():
                if name.endswith('_x') or name.endswith('_y'):
                    if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                        kaiming_init(m)
                    elif isinstance(m, nn.BatchNorm2d):
                        constant_init(m, 1)
                    elif isinstance(m, nn.Linear):
                        normal_init(m, std=.01)
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            ckpt = CheckpointLoader.load_checkpoint(
                self.init_cfg['checkpoint'], logger=None, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt

            state_dict = OrderedDict()
            for k, v in _state_dict.items():
                if k.startswith('backbone.'):
                    state_dict[k[9:]] = v
                else:
                    state_dict[k] = v

            # strip prefix of state_dict
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}

            state_dict_x = OrderedDict()
            state_dict_y = OrderedDict()

            for k, v in state_dict.items():
                # find the first '.'
                first_dot_pos = k.find('.')
                if first_dot_pos != -1:
                    # insert '_x' or '_y' before the first '.'
                    new_key_x = k[:first_dot_pos] + '_x' + k[first_dot_pos:]
                    new_key_y = k[:first_dot_pos] + '_y' + k[first_dot_pos:]
                else:
                    # append '_x' or '_y' if there is no '.'
                    new_key_x = k + '_x'
                    new_key_y = k + '_y'

                state_dict_x[new_key_x] = v
                state_dict_y[new_key_y] = v

            # load state_dict
            self.load_state_dict(state_dict_x, strict=False)
            self.load_state_dict(state_dict_y, strict=False)

    def forward(self, x):
        # fetch inputs and split it to 2 modalities
        x, y = torch.split(x, (3, 3), dim=1)
        # x modality encoder forward
        outs_y = []
        for i, stage_y in enumerate(self.stages_y):
            y = self.downsample_layers_y[i](y)
            y = stage_y(y)
            # all stages have same feat channels
            y = self.reduces_x[i](y)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm_y{i}')
                if self.gap_before_final_norm:
                    gap = y.mean([-2, -1], keepdim=True)
                    outs_y.append(norm_layer(gap).flatten(1))
                else:
                    outs_y.append(norm_layer(y))

        # rgb+x fusion forward
        outs_x = []
        (deform_inputs0_1, deform_inputs0_2, deform_inputs1_1, deform_inputs1_2,
         deform_inputs2_1, deform_inputs2_2, deform_inputs3_1, deform_inputs3_2) = deform_inputs(x)
        # stage 0
        x = self.downsample_layers_x[0](x)

        x = self.stages_x[0](x)
        norm_layer = getattr(self, f'norm_x{0}')
        outs_x.append(norm_layer(x))











        # concat rgb encoder and ano encoder together and feed them to following parts
        outs = [torch.cat((out_x, out_y), dim=1) for out_x, out_y in zip(outs_x, outs_y)]

        return outs

    def _freeze_stages(self):
        for i in range(self.frozen_stages):
            downsample_layer_x = self.downsample_layers_x[i]
            downsample_layer_y = self.downsample_layers_y[i]
            stage_x = self.stages_x[i]
            stage_y = self.stages_y[i]
            reduce_block_rgb = self.reduces_rgb[i]
            reduce_block_x = self.reduces_x[i]
            downsample_layer_x.eval()
            downsample_layer_y.eval()
            stage_x.eval()
            stage_y.eval()
            reduce_block_rgb.eval()
            reduce_block_x.eval()
            for param in chain(downsample_layer_x.parameters(),
                               stage_x.parameters(), reduce_block_rgb.parameters()):
                param.requires_grad = False
            for param in chain(downsample_layer_y.parameters(),
                               stage_y.parameters(), reduce_block_x.parameters()):
                param.requires_grad = False

    def train(self, mode=True):
        super(TwinConvNeXt, self).train(mode)
        self._freeze_stages()
