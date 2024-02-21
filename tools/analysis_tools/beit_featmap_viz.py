# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
from typing import Type, Union, Sequence
from collections import defaultdict
import os

import matplotlib.pyplot as plt
import numpy as np
import mmcv
import torch
import torch.nn as nn

from mmengine_custom.dataset import Compose
from mmengine_custom.model import revert_sync_batchnorm
from mmengine_custom.structures import PixelData
from mmseg_custom.apis import init_model
from mmseg_custom.structures import SegDataSample
from mmseg_custom.utils import register_all_modules
from mmseg_custom.visualization import SegLocalVisualizer
from mmseg_custom.models import BaseSegmentor
from mmseg_custom.utils import SampleList


def _preprare_mulitmodal_data(imgs, anos, model):

    cfg = model.cfg
    for t in cfg.test_pipeline:
        if t.get('type') in ['LoadAnnotations', 'LoadCarlaAnnotations']:
            cfg.test_pipeline.remove(t)

    is_batch = True
    if not isinstance(imgs, (list, tuple)):
        imgs = [imgs]
        anos = [anos]
        is_batch = False

    if isinstance(imgs[0], np.ndarray):
        cfg.test_pipeline[0]['type'] = 'LoadImageFromNDArray'

    # TODO: Consider using the singleton pattern to avoid building
    # a pipeline for each inference
    pipeline = Compose(cfg.test_pipeline)

    data = defaultdict(list)
    for (img, ano) in zip(imgs, anos):
        if isinstance(img, np.ndarray) and isinstance(ano, np.ndarray):
            data_ = dict(img=img)
            data_['ano'] = ano
        else:
            data_ = dict(img_path=img)
            data_['ano_path'] = ano
        data_ = pipeline(data_)
        data['inputs'].append(data_['inputs'])
        data['data_samples'].append(data_['data_samples'])

    return data, is_batch

ImageType = Union[str, np.ndarray, Sequence[str], Sequence[np.ndarray]]

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
    assert batch_size == 1, 'only support single image featmap inference now!'
    for b in range(batch_size):
        for c in range(channels):
            # 提取单个特征图
            fmap = feature[b, c].cpu().detach().numpy()
            # 标准化特征图到0-1
            fmap_norm = (fmap - np.min(fmap)) / (np.max(fmap) - np.min(fmap))
            # 转换为0-255的uint8
            fmap_scaled = (fmap_norm * 255).astype(np.uint8)
            # 保存特征图
            file_name = f'stage{stage}_c{c}.png'
            plt.imsave(os.path.join(save_dir, file_name), fmap_scaled, cmap=cmap)
            print(f"Feature map saved: {file_name}")
        avg_feat = feature[b].mean(0).unsqueeze(0)
        print(avg_feat.shape)
        for channel_index in range(avg_feat.size(0)):
            channel = avg_feat[channel_index]
            min_, max_ = channel.min(), channel.max()
            channel = (channel - min_) / (max_ - min_)  # normalize
            channel = channel.cpu().numpy()
            plt.imsave(os.path.join(save_dir, f'stage{stage}_mean.png'), channel, cmap=cmap)

def inference_multimodel_featmap(img: ImageType, ano: ImageType,
                         model):
    """Inference image(s) with the segmentor.

    Args:
        model (nn.Module): The loaded segmentor.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        :obj:`SegDataSample` or list[:obj:`SegDataSample`]:
        If imgs is a list or tuple, the same length list type results
        will be returned, otherwise return the segmentation results directly.
    """
    # prepare data
    data, is_batch = _preprare_mulitmodal_data(img, ano, model)
    selected_stage = 0
    # forward the model
    with torch.no_grad():
        results = model.predict_featmap_step(data, selected_stage)

    save_feature_maps(results, selected_stage, 'work_dirs/beit_stage1_demo', 'viridis')
    # return results if is_batch else results[0]


def main():
    parser = ArgumentParser(
        description='Draw the Feature Map During Inference')
    parser.add_argument('img', help='Image file')
    parser.add_argument('ano', help='Another file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--gt_mask', default=None, help='Path of gt mask file')
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument(
        '--title', default='result', help='The image identifier.')
    args = parser.parse_args()

    register_all_modules()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    if args.device == 'cpu':
        model = revert_sync_batchnorm(model)

    inference_multimodel_featmap(args.img, args.ano, model)


if __name__ == '__main__':
    main()
