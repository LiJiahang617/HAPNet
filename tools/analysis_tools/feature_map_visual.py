# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
from typing import Type, Union, Sequence
from collections import defaultdict

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


class Recorder:
    """record the forward output feature map and save to data_buffer."""

    def __init__(self) -> None:
        self.data_buffer = list()

    def __enter__(self, ):
        self._data_buffer = list()

    def record_data_hook(self, model: nn.Module, input: Type, output: Type):
        if len(output.shape) == 3:
            if output.shape[1] == 14080:
                h, w = (88,160)
                output = output.reshape(-1, output.shape[2], h, w)
            elif output.shape[1] == 220:
                h, w = (11, 20)
                output = output.reshape(-1, output.shape[2], h, w)
            elif output.shape[1] == 3520:
                h, w = (44, 80)
                output = output.reshape(-1, output.shape[2], h, w)
            elif output.shape[1] == 880:
                h, w = (22, 40)
                output = output.reshape(-1, output.shape[2], h, w)
            self.data_buffer.append(output)
        else:
            self.data_buffer.append(output)

    def __exit__(self, *args, **kwargs):
        pass


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

# alpha is suitible to set as 0.8
def visualize(args, model, recorder, result, source):
    seg_visualizer = SegLocalVisualizer(
        vis_backends=[dict(type='LocalVisBackend')],
        save_dir='predict_demo',
        alpha=0.8)
    seg_visualizer.dataset_meta = dict(
        classes=model.dataset_meta['classes'],
        palette=model.dataset_meta['palette'])
    image = mmcv.imread(args.img, 'color', channel_order='rgb')

    # add predict result to visualizer
    seg_visualizer.add_datasample(
        name='carla',
        image=image,
        data_sample=result,
        draw_gt=False,
        draw_pred=True,
        wait_time=0,
        out_file=None,
        show=False)

    # add feature map to visualizer
    module_list = list(source.keys())
    for i in range(len(recorder.data_buffer)):
        module = module_list[i]
        feature = recorder.data_buffer[i][0]  # remove the batch
        drawn_img = seg_visualizer.draw_featmap(
            feature, image, channel_reduction=None, topk=8, arrangement=(4, 2))
        # seg_visualizer.show(drawn_img)
        """
        default to save image using:
        Visualizer.add_image --> Localvisbackend.add_image, actually using cv2.imwrite()
        self.save_dir + self._img_save_dir + save_file_name
        save_dir is defined in init Localvisualizer, which could be inherited by its backends as default.
        _img_save_dir is defined by visbackend default
        users should modify the save_file_name
        """

        seg_visualizer.add_image(f'{module}', drawn_img)

    if args.gt_mask:
        sem_seg = mmcv.imread(args.gt_mask, 'unchanged')
        sem_seg = torch.from_numpy(sem_seg)
        gt_mask = dict(data=sem_seg)
        gt_mask = PixelData(**gt_mask)
        data_sample = SegDataSample()
        data_sample.gt_sem_seg = gt_mask

        seg_visualizer.add_datasample(
            name='gt_mask',
            image=image,
            data_sample=data_sample,
            draw_gt=True,
            draw_pred=False,
            wait_time=0,
            out_file=None,
            show=False)

    seg_visualizer.add_image('image', image)


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

    # show all named module in the model and use it in source list below
    for name, module in model.named_modules():
        print(name)

    source = [
              # 'backbone.norm_x0',
              # 'backbone.norm_y0',
              # 'backbone.norm_x3',
              'backbone.norm1'
    ]
    source = dict.fromkeys(source)

    count = 0
    recorder = Recorder()
    # registry the forward hook
    for name, module in model.named_modules():
        if name in source:
            count += 1
            module.register_forward_hook(recorder.record_data_hook)
            if count == len(source):
                break

    with recorder:
        # test a single image, and record feature map to data_buffer
        result = inference_multimodel(args.img, args.ano, model)

    visualize(args, model, recorder, result, source)


if __name__ == '__main__':
    main()
