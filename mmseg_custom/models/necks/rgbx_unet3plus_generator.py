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


class upsample_layer(BaseModule):
    def __init__(self, in_ch, out_ch, up_scale=2):
        super(upsample_layer, self).__init__()
        self.up_scale = up_scale
        self.up = nn.Upsample(scale_factor=self.up_scale, mode='bilinear', align_corners=True)
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.up(x)
        x = self.conv1(x)
        x = self.bn1(x)
        output = self.activation(x)
        return output


@MODELS.register_module()
class UNet3plusGenerators(BaseModule):
    def __init__(self, in_channels, out_channels=3) -> None:
        # in_channels needs 5 elements
        super(UNet3plusGenerators, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up_rgb_conv = nn.Conv2d(in_channels=in_channels[1], out_channels=in_channels[0], kernel_size=1)
        self.up_x_conv = nn.Conv2d(in_channels=in_channels[1], out_channels=in_channels[0], kernel_size=1)

        self.CatChannels = in_channels[0]
        self.CatBlocks = 5
        self.UpChannels = self.CatChannels * self.CatBlocks

        #------------------------------RGB generator----------------------------------
        '''stage 4d RGB'''
        # h1->320*320, hd4->40*40, Pooling 8 times
        self.h1_PT_hd4_R = nn.MaxPool2d(8, 8, ceil_mode=True)
        self.h1_PT_hd4_conv_R = nn.Conv2d(in_channels[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd4_bn_R = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd4_relu_R = nn.ReLU(inplace=True)

        # h2->160*160, hd4->40*40, Pooling 4 times
        self.h2_PT_hd4_R = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h2_PT_hd4_conv_R = nn.Conv2d(in_channels[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd4_bn_R = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd4_relu_R = nn.ReLU(inplace=True)

        # h3->80*80, hd4->40*40, Pooling 2 times
        self.h3_PT_hd4_R = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h3_PT_hd4_conv_R = nn.Conv2d(in_channels[2], self.CatChannels, 3, padding=1)
        self.h3_PT_hd4_bn_R = nn.BatchNorm2d(self.CatChannels)
        self.h3_PT_hd4_relu_R = nn.ReLU(inplace=True)

        # h4->40*40, hd4->40*40, Concatenation
        self.h4_Cat_hd4_conv_R = nn.Conv2d(in_channels[3], self.CatChannels, 3, padding=1)
        self.h4_Cat_hd4_bn_R = nn.BatchNorm2d(self.CatChannels)
        self.h4_Cat_hd4_relu_R = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->40*40, Upsample 2 times
        self.hd5_UT_hd4_R = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd5_UT_hd4_conv_R = nn.Conv2d(in_channels[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd4_bn_R = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd4_relu_R = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)
        self.conv4d_1_R = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn4d_1_R = nn.BatchNorm2d(self.UpChannels)
        self.relu4d_1_R = nn.ReLU(inplace=True)

        '''stage 3d RGB'''
        # h1->320*320, hd3->80*80, Pooling 4 times
        self.h1_PT_hd3_R = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h1_PT_hd3_conv_R = nn.Conv2d(in_channels[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd3_bn_R = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd3_relu_R = nn.ReLU(inplace=True)

        # h2->160*160, hd3->80*80, Pooling 2 times
        self.h2_PT_hd3_R = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h2_PT_hd3_conv_R = nn.Conv2d(in_channels[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd3_bn_R = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd3_relu_R = nn.ReLU(inplace=True)

        # h3->80*80, hd3->80*80, Concatenation
        self.h3_Cat_hd3_conv_R = nn.Conv2d(in_channels[2], self.CatChannels, 3, padding=1)
        self.h3_Cat_hd3_bn_R = nn.BatchNorm2d(self.CatChannels)
        self.h3_Cat_hd3_relu_R = nn.ReLU(inplace=True)

        # hd4->40*40, hd4->80*80, Upsample 2 times
        self.hd4_UT_hd3_R = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd4_UT_hd3_conv_R = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd3_bn_R = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd3_relu_R = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->80*80, Upsample 4 times
        self.hd5_UT_hd3_R = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd5_UT_hd3_conv_R = nn.Conv2d(in_channels[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd3_bn_R = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd3_relu_R = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3)
        self.conv3d_1_R = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn3d_1_R = nn.BatchNorm2d(self.UpChannels)
        self.relu3d_1_R = nn.ReLU(inplace=True)

        '''stage 2d RGB '''
        # h1->320*320, hd2->160*160, Pooling 2 times
        self.h1_PT_hd2_R = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h1_PT_hd2_conv_R = nn.Conv2d(in_channels[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd2_bn_R = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd2_relu_R = nn.ReLU(inplace=True)

        # h2->160*160, hd2->160*160, Concatenation
        self.h2_Cat_hd2_conv_R = nn.Conv2d(in_channels[1], self.CatChannels, 3, padding=1)
        self.h2_Cat_hd2_bn_R = nn.BatchNorm2d(self.CatChannels)
        self.h2_Cat_hd2_relu_R = nn.ReLU(inplace=True)

        # hd3->80*80, hd2->160*160, Upsample 2 times
        self.hd3_UT_hd2_R = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd3_UT_hd2_conv_R = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd2_bn_R = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd2_relu_R = nn.ReLU(inplace=True)

        # hd4->40*40, hd2->160*160, Upsample 4 times
        self.hd4_UT_hd2_R = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd4_UT_hd2_conv_R = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd2_bn_R = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd2_relu_R = nn.ReLU(inplace=True)

        # hd5->20*20, hd2->160*160, Upsample 8 times
        self.hd5_UT_hd2_R = nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
        self.hd5_UT_hd2_conv_R = nn.Conv2d(in_channels[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd2_bn_R = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd2_relu_R = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2)
        self.conv2d_1_R = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn2d_1_R = nn.BatchNorm2d(self.UpChannels)
        self.relu2d_1_R = nn.ReLU(inplace=True)

        '''stage 1d RGB'''
        # h1->320*320, hd1->320*320, Concatenation
        self.h1_Cat_hd1_conv_R = nn.Conv2d(in_channels[0], self.CatChannels, 3, padding=1)
        self.h1_Cat_hd1_bn_R = nn.BatchNorm2d(self.CatChannels)
        self.h1_Cat_hd1_relu_R = nn.ReLU(inplace=True)

        # hd2->160*160, hd1->320*320, Upsample 2 times
        self.hd2_UT_hd1_R = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd2_UT_hd1_conv_R = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd2_UT_hd1_bn_R = nn.BatchNorm2d(self.CatChannels)
        self.hd2_UT_hd1_relu_R = nn.ReLU(inplace=True)

        # hd3->80*80, hd1->320*320, Upsample 4 times
        self.hd3_UT_hd1_R = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd3_UT_hd1_conv_R = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd1_bn_R = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd1_relu_R = nn.ReLU(inplace=True)

        # hd4->40*40, hd1->320*320, Upsample 8 times
        self.hd4_UT_hd1_R = nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
        self.hd4_UT_hd1_conv_R = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd1_bn_R = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd1_relu_R = nn.ReLU(inplace=True)

        # hd5->20*20, hd1->320*320, Upsample 16 times
        self.hd5_UT_hd1_R = nn.Upsample(scale_factor=16, mode='bilinear')  # 14*14
        self.hd5_UT_hd1_conv_R = nn.Conv2d(in_channels[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd1_bn_R = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd1_relu_R = nn.ReLU(inplace=True)

        # fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)
        self.conv1d_1_R = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn1d_1_R = nn.BatchNorm2d(self.UpChannels)
        self.relu1d_1_R = nn.ReLU(inplace=True)

        # ------------------------------X generator----------------------------------
        '''stage 4d X'''
        # h1->320*320, hd4->40*40, Pooling 8 times
        self.h1_PT_hd4_X = nn.MaxPool2d(8, 8, ceil_mode=True)
        self.h1_PT_hd4_conv_X = nn.Conv2d(in_channels[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd4_bn_X = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd4_relu_X = nn.ReLU(inplace=True)

        # h2->160*160, hd4->40*40, Pooling 4 times
        self.h2_PT_hd4_X = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h2_PT_hd4_conv_X = nn.Conv2d(in_channels[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd4_bn_X = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd4_relu_X = nn.ReLU(inplace=True)

        # h3->80*80, hd4->40*40, Pooling 2 times
        self.h3_PT_hd4_X = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h3_PT_hd4_conv_X = nn.Conv2d(in_channels[2], self.CatChannels, 3, padding=1)
        self.h3_PT_hd4_bn_X = nn.BatchNorm2d(self.CatChannels)
        self.h3_PT_hd4_relu_X = nn.ReLU(inplace=True)

        # h4->40*40, hd4->40*40, Concatenation
        self.h4_Cat_hd4_conv_X = nn.Conv2d(in_channels[3], self.CatChannels, 3, padding=1)
        self.h4_Cat_hd4_bn_X = nn.BatchNorm2d(self.CatChannels)
        self.h4_Cat_hd4_relu_X = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->40*40, Upsample 2 times
        self.hd5_UT_hd4_X = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd5_UT_hd4_conv_X = nn.Conv2d(in_channels[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd4_bn_X = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd4_relu_X = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)
        self.conv4d_1_X = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn4d_1_X = nn.BatchNorm2d(self.UpChannels)
        self.relu4d_1_X = nn.ReLU(inplace=True)

        '''stage 3d X'''
        # h1->320*320, hd3->80*80, Pooling 4 times
        self.h1_PT_hd3_X = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h1_PT_hd3_conv_X = nn.Conv2d(in_channels[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd3_bn_X = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd3_relu_X = nn.ReLU(inplace=True)

        # h2->160*160, hd3->80*80, Pooling 2 times
        self.h2_PT_hd3_X = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h2_PT_hd3_conv_X = nn.Conv2d(in_channels[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd3_bn_X = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd3_relu_X = nn.ReLU(inplace=True)

        # h3->80*80, hd3->80*80, Concatenation
        self.h3_Cat_hd3_conv_X = nn.Conv2d(in_channels[2], self.CatChannels, 3, padding=1)
        self.h3_Cat_hd3_bn_X = nn.BatchNorm2d(self.CatChannels)
        self.h3_Cat_hd3_relu_X = nn.ReLU(inplace=True)

        # hd4->40*40, hd4->80*80, Upsample 2 times
        self.hd4_UT_hd3_X = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd4_UT_hd3_conv_X = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd3_bn_X = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd3_relu_X = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->80*80, Upsample 4 times
        self.hd5_UT_hd3_X = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd5_UT_hd3_conv_X = nn.Conv2d(in_channels[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd3_bn_X = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd3_relu_X = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3)
        self.conv3d_1_X = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn3d_1_X = nn.BatchNorm2d(self.UpChannels)
        self.relu3d_1_X = nn.ReLU(inplace=True)

        '''stage 2d X '''
        # h1->320*320, hd2->160*160, Pooling 2 times
        self.h1_PT_hd2_X = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h1_PT_hd2_conv_X = nn.Conv2d(in_channels[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd2_bn_X = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd2_relu_X = nn.ReLU(inplace=True)

        # h2->160*160, hd2->160*160, Concatenation
        self.h2_Cat_hd2_conv_X = nn.Conv2d(in_channels[1], self.CatChannels, 3, padding=1)
        self.h2_Cat_hd2_bn_X = nn.BatchNorm2d(self.CatChannels)
        self.h2_Cat_hd2_relu_X = nn.ReLU(inplace=True)

        # hd3->80*80, hd2->160*160, Upsample 2 times
        self.hd3_UT_hd2_X = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd3_UT_hd2_conv_X = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd2_bn_X = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd2_relu_X = nn.ReLU(inplace=True)

        # hd4->40*40, hd2->160*160, Upsample 4 times
        self.hd4_UT_hd2_X = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd4_UT_hd2_conv_X = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd2_bn_X = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd2_relu_X = nn.ReLU(inplace=True)

        # hd5->20*20, hd2->160*160, Upsample 8 times
        self.hd5_UT_hd2_X = nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
        self.hd5_UT_hd2_conv_X = nn.Conv2d(in_channels[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd2_bn_X = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd2_relu_X = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2)
        self.conv2d_1_X = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn2d_1_X = nn.BatchNorm2d(self.UpChannels)
        self.relu2d_1_X = nn.ReLU(inplace=True)

        '''stage 1d X'''
        # h1->320*320, hd1->320*320, Concatenation
        self.h1_Cat_hd1_conv_X = nn.Conv2d(in_channels[0], self.CatChannels, 3, padding=1)
        self.h1_Cat_hd1_bn_X = nn.BatchNorm2d(self.CatChannels)
        self.h1_Cat_hd1_relu_X = nn.ReLU(inplace=True)

        # hd2->160*160, hd1->320*320, Upsample 2 times
        self.hd2_UT_hd1_X = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd2_UT_hd1_conv_X = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd2_UT_hd1_bn_X = nn.BatchNorm2d(self.CatChannels)
        self.hd2_UT_hd1_relu_X = nn.ReLU(inplace=True)

        # hd3->80*80, hd1->320*320, Upsample 4 times
        self.hd3_UT_hd1_X = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd3_UT_hd1_conv_X = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd1_bn_X = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd1_relu_X = nn.ReLU(inplace=True)

        # hd4->40*40, hd1->320*320, Upsample 8 times
        self.hd4_UT_hd1_X = nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
        self.hd4_UT_hd1_conv_X = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd1_bn_X = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd1_relu_X = nn.ReLU(inplace=True)

        # hd5->20*20, hd1->320*320, Upsample 16 times
        self.hd5_UT_hd1_X = nn.Upsample(scale_factor=16, mode='bilinear')  # 14*14
        self.hd5_UT_hd1_conv_X = nn.Conv2d(in_channels[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd1_bn_X = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd1_relu_X = nn.ReLU(inplace=True)

        # fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)
        self.conv1d_1_X = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn1d_1_X = nn.BatchNorm2d(self.UpChannels)
        self.relu1d_1_X = nn.ReLU(inplace=True)

        # final generation
        self.final_R = upsample_layer(5 * in_channels[0], out_channels)
        self.final_X = upsample_layer(5 * in_channels[0], out_channels)

    def forward(self, feats: List[Tensor]):
        feats_RGB = []
        feats_X = []
        for feat in feats:
            feat_RGB, feat_X = torch.chunk(feat, 2, dim=1)
            feats_RGB.append(feat_RGB)
            feats_X.append(feat_X)
        # upsample RGB_0 for UNet3+ generator decoder
        feat_max_RGB = feats_RGB[0]
        upsampled_feat_RGB = F.interpolate(feat_max_RGB, scale_factor=2, mode='bilinear', align_corners=False)
        upsampled_feat_RGB = self.up_rgb_conv(upsampled_feat_RGB)
        feats_RGB.insert(0, upsampled_feat_RGB)
        assert len(feats_RGB) == 5, 'Upsampling RGB features failed! Please check your generators'
        # upsample X_0 for UNet3+ generator decoder
        feat_max_X = feats_X[0]
        upsampled_feat_X = F.interpolate(feat_max_X, scale_factor=2, mode='bilinear', align_corners=False)
        upsampled_feat_X = self.up_x_conv(upsampled_feat_X)
        feats_X.insert(0, upsampled_feat_X)
        assert len(feats_X) == 5, 'Upsampling X features failed! Please check your generators'

        # image translation RGB --> fake X
        h1_r, h2_r, h3_r, h4_r, hd5_r = feats_RGB

        h1_r_PT_hd4_r = self.h1_PT_hd4_relu_R(self.h1_PT_hd4_bn_R(self.h1_PT_hd4_conv_R(self.h1_PT_hd4_R(h1_r))))
        h2_r_PT_hd4_r = self.h2_PT_hd4_relu_R(self.h2_PT_hd4_bn_R(self.h2_PT_hd4_conv_R(self.h2_PT_hd4_R(h2_r))))
        h3_r_PT_hd4_r = self.h3_PT_hd4_relu_R(self.h3_PT_hd4_bn_R(self.h3_PT_hd4_conv_R(self.h3_PT_hd4_R(h3_r))))
        h4_r_Cat_hd4_r = self.h4_Cat_hd4_relu_R(self.h4_Cat_hd4_bn_R(self.h4_Cat_hd4_conv_R(h4_r)))
        hd5_r_UT_hd4_r = self.hd5_UT_hd4_relu_R(self.hd5_UT_hd4_bn_R(self.hd5_UT_hd4_conv_R(self.hd5_UT_hd4_R(hd5_r))))
        hd4_r = self.relu4d_1_R(self.bn4d_1_R(self.conv4d_1_R(
            torch.cat((h1_r_PT_hd4_r, h2_r_PT_hd4_r, h3_r_PT_hd4_r, h4_r_Cat_hd4_r, hd5_r_UT_hd4_r), 1))))  # hd4->40*40*UpChannels

        h1_r_PT_hd3_r = self.h1_PT_hd3_relu_R(self.h1_PT_hd3_bn_R(self.h1_PT_hd3_conv_R(self.h1_PT_hd3_R(h1_r))))
        h2_r_PT_hd3_r = self.h2_PT_hd3_relu_R(self.h2_PT_hd3_bn_R(self.h2_PT_hd3_conv_R(self.h2_PT_hd3_R(h2_r))))
        h3_r_Cat_hd3_r = self.h3_Cat_hd3_relu_R(self.h3_Cat_hd3_bn_R(self.h3_Cat_hd3_conv_R(h3_r)))
        hd4_r_UT_hd3_r = self.hd4_UT_hd3_relu_R(self.hd4_UT_hd3_bn_R(self.hd4_UT_hd3_conv_R(self.hd4_UT_hd3_R(hd4_r))))
        hd5_r_UT_hd3_r = self.hd5_UT_hd3_relu_R(self.hd5_UT_hd3_bn_R(self.hd5_UT_hd3_conv_R(self.hd5_UT_hd3_R(hd5_r))))
        hd3_r = self.relu3d_1_R(self.bn3d_1_R(self.conv3d_1_R(
            torch.cat((h1_r_PT_hd3_r, h2_r_PT_hd3_r, h3_r_Cat_hd3_r, hd4_r_UT_hd3_r, hd5_r_UT_hd3_r), 1))))  # hd3->80*80*UpChannels

        h1_r_PT_hd2_r = self.h1_PT_hd2_relu_R(self.h1_PT_hd2_bn_R(self.h1_PT_hd2_conv_R(self.h1_PT_hd2_R(h1_r))))
        h2_r_Cat_hd2_r = self.h2_Cat_hd2_relu_R(self.h2_Cat_hd2_bn_R(self.h2_Cat_hd2_conv_R(h2_r)))
        hd3_r_UT_hd2_r = self.hd3_UT_hd2_relu_R(self.hd3_UT_hd2_bn_R(self.hd3_UT_hd2_conv_R(self.hd3_UT_hd2_R(hd3_r))))
        hd4_r_UT_hd2_r = self.hd4_UT_hd2_relu_R(self.hd4_UT_hd2_bn_R(self.hd4_UT_hd2_conv_R(self.hd4_UT_hd2_R(hd4_r))))
        hd5_r_UT_hd2_r = self.hd5_UT_hd2_relu_R(self.hd5_UT_hd2_bn_R(self.hd5_UT_hd2_conv_R(self.hd5_UT_hd2_R(hd5_r))))
        hd2_r = self.relu2d_1_R(self.bn2d_1_R(self.conv2d_1_R(
            torch.cat((h1_r_PT_hd2_r, h2_r_Cat_hd2_r, hd3_r_UT_hd2_r, hd4_r_UT_hd2_r, hd5_r_UT_hd2_r), 1))))  # hd2->160*160*UpChannels

        h1_r_Cat_hd1_r = self.h1_Cat_hd1_relu_R(self.h1_Cat_hd1_bn_R(self.h1_Cat_hd1_conv_R(h1_r)))
        hd2_r_UT_hd1_r = self.hd2_UT_hd1_relu_R(self.hd2_UT_hd1_bn_R(self.hd2_UT_hd1_conv_R(self.hd2_UT_hd1_R(hd2_r))))
        hd3_r_UT_hd1_r = self.hd3_UT_hd1_relu_R(self.hd3_UT_hd1_bn_R(self.hd3_UT_hd1_conv_R(self.hd3_UT_hd1_R(hd3_r))))
        hd4_r_UT_hd1_r = self.hd4_UT_hd1_relu_R(self.hd4_UT_hd1_bn_R(self.hd4_UT_hd1_conv_R(self.hd4_UT_hd1_R(hd4_r))))
        hd5_r_UT_hd1_r = self.hd5_UT_hd1_relu_R(self.hd5_UT_hd1_bn_R(self.hd5_UT_hd1_conv_R(self.hd5_UT_hd1_R(hd5_r))))
        hd1_r = self.relu1d_1_R(self.bn1d_1_R(self.conv1d_1_R(
            torch.cat((h1_r_Cat_hd1_r, hd2_r_UT_hd1_r, hd3_r_UT_hd1_r, hd4_r_UT_hd1_r, hd5_r_UT_hd1_r), 1))))  # hd1->320*320*UpChannels

        fake_X = self.final_R(hd1_r)
        fake_X = torch.sigmoid(fake_X)
        # image translation X --> fake RGB
        h1_x, h2_x, h3_x, h4_x, hd5_x = feats_X

        h1_x_PT_hd4_x = self.h1_PT_hd4_relu_X(self.h1_PT_hd4_bn_X(self.h1_PT_hd4_conv_X(self.h1_PT_hd4_X(h1_x))))
        h2_x_PT_hd4_x = self.h2_PT_hd4_relu_X(self.h2_PT_hd4_bn_X(self.h2_PT_hd4_conv_X(self.h2_PT_hd4_X(h2_x))))
        h3_x_PT_hd4_x = self.h3_PT_hd4_relu_X(self.h3_PT_hd4_bn_X(self.h3_PT_hd4_conv_X(self.h3_PT_hd4_X(h3_x))))
        h4_x_Cat_hd4_x = self.h4_Cat_hd4_relu_X(self.h4_Cat_hd4_bn_X(self.h4_Cat_hd4_conv_X(h4_x)))
        hd5_x_UT_hd4_x = self.hd5_UT_hd4_relu_X(self.hd5_UT_hd4_bn_X(self.hd5_UT_hd4_conv_X(self.hd5_UT_hd4_X(hd5_x))))
        hd4_x = self.relu4d_1_X(self.bn4d_1_X(self.conv4d_1_X(
            torch.cat((h1_x_PT_hd4_x, h2_x_PT_hd4_x, h3_x_PT_hd4_x, h4_x_Cat_hd4_x, hd5_x_UT_hd4_x),
                      1))))  # hd4->40*40*UpChannels

        h1_x_PT_hd3_x = self.h1_PT_hd3_relu_X(self.h1_PT_hd3_bn_X(self.h1_PT_hd3_conv_X(self.h1_PT_hd3_X(h1_x))))
        h2_x_PT_hd3_x = self.h2_PT_hd3_relu_X(self.h2_PT_hd3_bn_X(self.h2_PT_hd3_conv_X(self.h2_PT_hd3_X(h2_x))))
        h3_x_Cat_hd3_x = self.h3_Cat_hd3_relu_X(self.h3_Cat_hd3_bn_X(self.h3_Cat_hd3_conv_X(h3_x)))
        hd4_x_UT_hd3_x = self.hd4_UT_hd3_relu_X(self.hd4_UT_hd3_bn_X(self.hd4_UT_hd3_conv_X(self.hd4_UT_hd3_X(hd4_x))))
        hd5_x_UT_hd3_x = self.hd5_UT_hd3_relu_X(self.hd5_UT_hd3_bn_X(self.hd5_UT_hd3_conv_X(self.hd5_UT_hd3_X(hd5_x))))
        hd3_x = self.relu3d_1_X(self.bn3d_1_X(self.conv3d_1_X(
            torch.cat((h1_x_PT_hd3_x, h2_x_PT_hd3_x, h3_x_Cat_hd3_x, hd4_x_UT_hd3_x, hd5_x_UT_hd3_x),
                      1))))  # hd3->80*80*UpChannels

        h1_x_PT_hd2_x = self.h1_PT_hd2_relu_X(self.h1_PT_hd2_bn_X(self.h1_PT_hd2_conv_X(self.h1_PT_hd2_X(h1_x))))
        h2_x_Cat_hd2_x = self.h2_Cat_hd2_relu_X(self.h2_Cat_hd2_bn_X(self.h2_Cat_hd2_conv_X(h2_x)))
        hd3_x_UT_hd2_x = self.hd3_UT_hd2_relu_X(self.hd3_UT_hd2_bn_X(self.hd3_UT_hd2_conv_X(self.hd3_UT_hd2_X(hd3_x))))
        hd4_x_UT_hd2_x = self.hd4_UT_hd2_relu_X(self.hd4_UT_hd2_bn_X(self.hd4_UT_hd2_conv_X(self.hd4_UT_hd2_X(hd4_x))))
        hd5_x_UT_hd2_x = self.hd5_UT_hd2_relu_X(self.hd5_UT_hd2_bn_X(self.hd5_UT_hd2_conv_X(self.hd5_UT_hd2_X(hd5_x))))
        hd2_x = self.relu2d_1_X(self.bn2d_1_X(self.conv2d_1_X(
            torch.cat((h1_x_PT_hd2_x, h2_x_Cat_hd2_x, hd3_x_UT_hd2_x, hd4_x_UT_hd2_x, hd5_x_UT_hd2_x),
                      1))))  # hd2->160*160*UpChannels

        h1_x_Cat_hd1_x = self.h1_Cat_hd1_relu_X(self.h1_Cat_hd1_bn_X(self.h1_Cat_hd1_conv_X(h1_x)))
        hd2_x_UT_hd1_x = self.hd2_UT_hd1_relu_X(self.hd2_UT_hd1_bn_X(self.hd2_UT_hd1_conv_X(self.hd2_UT_hd1_X(hd2_x))))
        hd3_x_UT_hd1_x = self.hd3_UT_hd1_relu_X(self.hd3_UT_hd1_bn_X(self.hd3_UT_hd1_conv_X(self.hd3_UT_hd1_X(hd3_x))))
        hd4_x_UT_hd1_x = self.hd4_UT_hd1_relu_X(self.hd4_UT_hd1_bn_X(self.hd4_UT_hd1_conv_X(self.hd4_UT_hd1_X(hd4_x))))
        hd5_x_UT_hd1_x = self.hd5_UT_hd1_relu_X(self.hd5_UT_hd1_bn_X(self.hd5_UT_hd1_conv_X(self.hd5_UT_hd1_X(hd5_x))))
        hd1_x = self.relu1d_1_X(self.bn1d_1_X(self.conv1d_1_X(
            torch.cat((h1_x_Cat_hd1_x, hd2_x_UT_hd1_x, hd3_x_UT_hd1_x, hd4_x_UT_hd1_x, hd5_x_UT_hd1_x),
                      1))))  # hd1->320*320*UpChannels
        
        fake_RGB = self.final_X(hd1_x)
        fake_RGB = torch.sigmoid(fake_RGB)
        # pack output dict
        outputs = dict(fake_X=fake_X, fake_RGB=fake_RGB)
        
        return outputs