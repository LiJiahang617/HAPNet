# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
from typing import Type, Union, Sequence
from collections import defaultdict

import numpy as np
import mmcv_custom
import torch
import torch.nn as nn

from mmengine_custom.dataset import Compose
from mmengine_custom.model import revert_sync_batchnorm
from mmengine_custom.structures import PixelData
from mmseg_custom.apis import inference_model, init_model
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

def inference_multimodel(img: ImageType, ano: ImageType,
                         model: BaseSegmentor) -> Union[SegDataSample, SampleList]:
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

    # forward the model
    with torch.no_grad():
        results = model.test_step(data)

    return results if is_batch else results[0]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def main():
    parser = ArgumentParser(
        description='Draw the Feature Map During Inference')
    parser.add_argument('img', help='Image file')
    parser.add_argument('ano', help='Another file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()
    register_all_modules()
    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    if args.device == 'cpu':
        model = revert_sync_batchnorm(model)

    # print(f'{args.config} has {count_parameters(model):,} trainable parameters')
    # print('FLOPs = ' + str((count_parameters(model)/1000 ** 3) + 'G'))
    print('Params = ' + str(count_parameters(model) / 1000 ** 2) + 'M')



if __name__ == '__main__':
    main()
