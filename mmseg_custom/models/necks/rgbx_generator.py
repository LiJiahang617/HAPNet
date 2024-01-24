from typing import List, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmengine_custom.model import constant_init, kaiming_init
from mmengine_custom.model import BaseModule, ModuleList

from mmseg_custom.registry import MODELS
from mmseg_custom.models.decode_heads.decode_head import BaseDecodeHead
from mmseg_custom.structures.seg_data_sample import SegDataSample
from mmseg_custom.utils import ConfigType, SampleList


class TransConvBnLeakyRelu2d(BaseModule):
    # deconvolution
    # batch normalization
    # Lrelu
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0):
        super(TransConvBnLeakyRelu2d, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                       padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.2, inplace=True)

    def init_weights(self) -> None:
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight.data)

    def forward(self, x):
        return self.leaky(self.bn(self.conv(x)))


class ConvBnrelu2d(BaseModule):
    # convolution
    # batch normalization
    # relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1, bias=False):
        super(ConvBnrelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn   = nn.BatchNorm2d(out_channels)

    def init_weights(self) -> None:
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class Conv2d(BaseModule):
    # convolution
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1, bias=False):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                              dilation=dilation, groups=groups, bias=bias)

    def init_weights(self) -> None:
        """Initialize weights."""
        nn.init.xavier_uniform_(self.conv.weight.data)

    def forward(self, x):
        return self.conv(x)


@MODELS.register_module()
class RGBXGenerator(BaseModule):
    def __init__(self,
                 in_channels: List[int],
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform'),
                 **kwargs):
        super().__init__(init_cfg)
        # fake X generator
        self.fake_X_Transupsample1 = TransConvBnLeakyRelu2d(in_channels[-1], in_channels[-2])
        self.fake_X_conv1 = ConvBnrelu2d(in_channels[-2], in_channels[-2])
        self.fake_X_Transupsample2 = TransConvBnLeakyRelu2d(in_channels[-2], in_channels[-3])
        self.fake_X_conv2 = ConvBnrelu2d(in_channels[-3], in_channels[-3])
        self.fake_X_Transupsample3 = TransConvBnLeakyRelu2d(in_channels[-3], in_channels[-4])
        self.fake_X_conv3 = ConvBnrelu2d(in_channels[-4], in_channels[-4])
        self.fake_X_Transupsample4 = TransConvBnLeakyRelu2d(in_channels[-4], in_channels[-4])
        self.fake_X_conv4 = ConvBnrelu2d(in_channels[-4], in_channels[-4])
        self.fake_X_Transupsample5 = TransConvBnLeakyRelu2d(in_channels[-4], in_channels[-4])
        # TODO: maybe bugs here, just one channel is enough
        self.fake_X_last = Conv2d(in_channels[-4], 3)
        # fake RGB generator
        self.fake_RGB_Transupsample1 = TransConvBnLeakyRelu2d(in_channels[-1], in_channels[-2])
        self.fake_RGB_conv1 = ConvBnrelu2d(in_channels[-2], in_channels[-2])
        self.fake_RGB_Transupsample2 = TransConvBnLeakyRelu2d(in_channels[-2], in_channels[-3])
        self.fake_RGB_conv2 = ConvBnrelu2d(in_channels[-3], in_channels[-3])
        self.fake_RGB_Transupsample3 = TransConvBnLeakyRelu2d(in_channels[-3], in_channels[-4])
        self.fake_RGB_conv3 = ConvBnrelu2d(in_channels[-4], in_channels[-4])
        self.fake_RGB_Transupsample4 = TransConvBnLeakyRelu2d(in_channels[-4], in_channels[-4])
        self.fake_RGB_conv4 = ConvBnrelu2d(in_channels[-4], in_channels[-4])
        self.fake_RGB_Transupsample5 = TransConvBnLeakyRelu2d(in_channels[-4], in_channels[-4])
        self.fake_RGB_last = Conv2d(in_channels[-4], 3)

    def forward(self, feats: List[Tensor]):
        # split multiscale features
        feats_RGB = []
        feats_X = []
        for feat in feats:
            feat_RGB, feat_X = torch.chunk(feat, 2, dim=1)
            feats_RGB.append(feat_RGB)
            feats_X.append(feat_X)
        # image translation RGB-->fake X
        RGB_feats = self.fake_X_Transupsample1(feats_RGB[-1])
        RGB_feats = self.fake_X_conv1(RGB_feats+feats_RGB[-2])
        RGB_feats = self.fake_X_Transupsample2(RGB_feats)
        RGB_feats = self.fake_X_conv2(RGB_feats+feats_RGB[-3])
        RGB_feats = self.fake_X_Transupsample3(RGB_feats)
        RGB_feats = self.fake_X_conv3(RGB_feats+feats_RGB[-4])
        RGB_feats = self.fake_X_Transupsample4(RGB_feats)
        RGB_feats = self.fake_X_conv4(RGB_feats)
        RGB_feats = self.fake_X_Transupsample5(RGB_feats)
        RGB_feats = self.fake_X_last(RGB_feats)
        fake_X = torch.sigmoid(RGB_feats)
        # image translation X-->fake RGB
        X_feats = self.fake_RGB_Transupsample1(feats_X[-1])
        X_feats = self.fake_RGB_conv1(X_feats+feats_X[-2])
        X_feats = self.fake_RGB_Transupsample2(X_feats)
        X_feats = self.fake_RGB_conv2(X_feats+feats_X[-3])
        X_feats = self.fake_RGB_Transupsample3(X_feats)
        X_feats = self.fake_RGB_conv3(X_feats+feats_X[-4])
        X_feats = self.fake_RGB_Transupsample4(X_feats)
        X_feats = self.fake_RGB_conv4(X_feats)
        X_feats =self.fake_RGB_Transupsample5(X_feats)
        X_feats = self.fake_RGB_last(X_feats)
        fake_RGB = torch.sigmoid(X_feats)

        output = dict(fake_X=fake_X, fake_RGB=fake_RGB)

        return output