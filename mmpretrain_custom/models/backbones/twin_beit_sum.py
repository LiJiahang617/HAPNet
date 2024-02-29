# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial
import copy
from collections import OrderedDict
import warnings
import math
import os

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_, normal_
from mmcv_custom.cnn.bricks import DropPath
from mmcv_custom.ops import MultiScaleDeformableAttnFunction
from mmcv_custom.cnn import Conv2d, ConvModule
from mmengine_custom.model import BaseModule, ModuleList, Sequential
from mmengine_custom.model.weight_init import (constant_init, trunc_normal_,
                                               trunc_normal_init)
from mmengine_custom.logging import print_log
from mmengine_custom.model import constant_init, kaiming_init, normal_init
from mmengine_custom.runner import CheckpointLoader

from mmpretrain_custom.registry import MODELS
from ..utils import GRN, build_norm_layer
from .base_backbone import BaseBackbone
from .dual_beit import DualBEiT


# ----------------------------------------for featmap visualization----------------------------------------
def save_feature_maps(feature, stage, save_dir='path_to_save', cmap='viridis'):
    """
    保存特征图到PNG图像
    :param feature: 特征张量，形状为 [batch_size, channels, height, width]
    :param stage: 特征所在的stage，用于文件命名
    :param save_dir: 保存图像的路径
    :param cmap: 使用的颜色映射
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    batch_size, channels, _, _ = feature.shape
    for b in range(batch_size):
        for c in range(channels):
            # 提取单个特征图
            fmap = feature[b, c].cpu().detach().numpy()
            # 标准化特征图到0-1
            fmap_norm = (fmap - np.min(fmap)) / (np.max(fmap) - np.min(fmap))
            # 转换为0-255的uint8
            fmap_scaled = (fmap_norm * 255).astype(np.uint8)
            # 保存特征图
            file_name = f'stage{stage}_b{b}_c{c}.png'
            plt.imsave(os.path.join(save_dir, file_name), fmap_scaled, cmap=cmap)
            print(f"Feature map saved: {file_name}")

def save_selected_feature_maps(features, selected_stages, save_dir='path_to_save', cmap='viridis'):
    """
    保存选定stages的特征图到PNG图像
    :param features: 包含所有stages特征的列表
    :param selected_stages: 一个包含所选stages编号的列表
    :param save_dir: 保存图像的路径
    :param cmap: 使用的颜色映射
    """
    for stage, feature in enumerate(features, start=1):
        if stage in selected_stages:
            save_feature_maps(feature, stage, save_dir, cmap)
# ----------------------------------------for featmap visualization----------------------------------------


class InteractionBlockWithCls(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, cls, blocks, H, W):

        x = torch.cat((cls, x), dim=1)

        for idx, blk in enumerate(blocks):
            x = blk(x, H, W)
        cls, x = x[:, :1, ], x[:, 1:, ]

        return x, cls


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError('invalid input for _is_power_of_2: {} (type: {})'.format(n, type(n)))
    return (n & (n - 1) == 0) and n != 0


@MODELS.register_module()
class TwinBeiTSum(DualBEiT):

    def __init__(self, pretrain_size=224, n_points=4, deform_num_heads=6,
                 init_values=0., cffn_ratio=0.25, deform_ratio=1.0, with_cffn=True,
                 interaction_indexes=None, with_cp=False,
                 arch='small', x_modality_encoder= ..., *args, **kwargs):

        super().__init__(init_values=init_values, with_cp=with_cp, *args, **kwargs)

        # self.num_classes = 80
        # self.cls_token = None
        self.num_block = len(self.blocks_x)
        self.pretrain_size = (pretrain_size, pretrain_size)
        self.flags = [i for i in range(-1, self.num_block, self.num_block // 4)][1:]
        self.interaction_indexes = interaction_indexes
        embed_dim = self.embed_dim

        print('Use embed dims of ----->>>', embed_dim, '<<<-----')
        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))
        x_modality_encoder_ = copy.deepcopy(x_modality_encoder)
        """ 
        config could be update here if necessary using:
        x_modality_encoder_.update(
                                    in_channels=in_channels,
                                    feat_channels=feat_channels,
                                    out_channels=out_channels)
        """
        # self.x_modality_encoder = MODELS.build(x_modality_encoder_)
        print('Note, this is ablation study for adapter-twin_vit-b_sum!!!')
        self.interactions_x = nn.Sequential(*[
            InteractionBlockWithCls()
            for i in range(len(interaction_indexes))
        ])
        self.interactions_y = nn.Sequential(*[
            InteractionBlockWithCls()
            for i in range(len(interaction_indexes))
        ])

        self.up = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
        self.norm1 = nn.SyncBatchNorm(embed_dim)
        self.norm2 = nn.SyncBatchNorm(embed_dim)
        self.norm3 = nn.SyncBatchNorm(embed_dim)
        self.norm4 = nn.SyncBatchNorm(embed_dim)

        # FIXME: self.cls_token need to be added, also in forawrd func
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.up.apply(self._init_weights)
        # self.x_modality_encoder.init_weights()
        self.interactions_x.apply(self._init_weights)
        self.interactions_y.apply(self._init_weights)

        trunc_normal_(self.cls_token, std=.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _get_pos_embed(self, pos_embed, H, W):
        pos_embed = pos_embed.reshape(
            1, self.pretrain_size[0] // 16, self.pretrain_size[1] // 16, -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False). \
            reshape(1, -1, H * W).permute(0, 2, 1)
        return pos_embed


    def forward(self, inputs):
        # fetch inputs and split it to 2 modalities: rgb:x
        x, y = torch.split(inputs, (3, 3), dim=1)

        # Patch Embedding forward
        x, H, W = self.patch_embed_x(x)
        y, H, W = self.patch_embed_y(y)
        bs, n, dim = y.shape
        # class token for vit
        cls_x = self.cls_token.clone().expand(bs, -1, -1)
        cls_y = self.cls_token.clone().expand(bs, -1, -1)
        if self.pos_embed is not None:
            pos_embed = self._get_pos_embed(self.pos_embed, H, W)
            x = x + pos_embed
            y = y + pos_embed
        x = self.pos_drop(x)
        y = self.pos_drop(y)

        # Interaction
        outs_y = list()
        for i, layer_y in enumerate(self.interactions_y):
            indexes = self.interaction_indexes[i]
            y, cls_y = layer_y(y, cls_y, self.blocks_y[indexes[0]:indexes[-1] + 1],
                              H, W)
            # need different context level features for segmentation
            outs_y.append(y.transpose(1, 2).view(bs, dim, H, W).contiguous())
        outs_x = list()
        for i, layer_x in enumerate(self.interactions_x):
            indexes = self.interaction_indexes[i]
            x, cls_x = layer_x(x, cls_x, self.blocks_x[indexes[0]:indexes[-1] + 1],
                              H, W)
            # need different context level features for segmentation
            outs_x.append(x.transpose(1, 2).view(bs, dim, H, W).contiguous())

        y1, y2, y3, y4 = outs_y
        y1 = F.interpolate(y1, scale_factor=4, mode='bilinear', align_corners=False)
        y2 = F.interpolate(y2, scale_factor=2, mode='bilinear', align_corners=False)
        y4 = F.interpolate(y4, scale_factor=0.5, mode='bilinear', align_corners=False)

        x1, x2, x3, x4 = outs_x
        x1 = F.interpolate(x1, scale_factor=4, mode='bilinear', align_corners=False)
        x2 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)
        x4 = F.interpolate(x4, scale_factor=0.5, mode='bilinear', align_corners=False)

        c1, c2, c3, c4 = x1 + y1, x2 + y2, x3 + y3, x4 + y4

        # Final Norm
        f1 = self.norm1(c1)
        f2 = self.norm2(c2)
        f3 = self.norm3(c3)
        f4 = self.norm4(c4)
        # # -------------------------------------------for visualzation-------------------------------------------
        # # 定义要保存的stages
        # selected_stages = [1]  # 保存第1个stage的特征图
        # save_selected_feature_maps([f1, f2, f3, f4], selected_stages, 'your_save_directory', 'viridis')
        # # -------------------------------------------for visualzation-------------------------------------------
        return [f1, f2, f3, f4]