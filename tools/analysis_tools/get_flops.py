# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import tempfile
from pathlib import Path
from functools import partial

import torch
import numpy as np
from mmengine_custom import Config, DictAction
from mmengine_custom.logging import MMLogger
from mmengine_custom.model import revert_sync_batchnorm
from mmengine_custom.registry import init_default_scope
from mmengine_custom.runner import Runner

from mmseg_custom.models import BaseSegmentor
from mmseg_custom.registry import MODELS
from mmseg_custom.structures import SegDataSample

try:
    from mmengine_custom.analysis import get_model_complexity_info
    from mmengine_custom.analysis.print_helper import _format_size
except ImportError:
    raise ImportError('Please upgrade mmengine >= 0.6.0 to use this script.')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Get the FLOPs of a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--num-images',
        type=int,
        default=100,
        help='num images of calculate model flops')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def inference(args: argparse.Namespace, logger: MMLogger) -> dict:
    config_name = Path(args.config)

    if not config_name.exists():
        logger.error(f'Config file {config_name} does not exist')

    cfg: Config = Config.fromfile(config_name)
    cfg.val_dataloader.batch_size = 1
    cfg.work_dir = tempfile.TemporaryDirectory().name

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    init_default_scope(cfg.get('scope', 'mmseg_custom'))

    result = {}
    avg_flops = []
    data_loader = Runner.build_dataloader(cfg.val_dataloader)
    model: BaseSegmentor = MODELS.build(cfg.model)
    if hasattr(model, 'auxiliary_head'):
        model.auxiliary_head = None
    if torch.cuda.is_available():
        model.cuda()
    model = revert_sync_batchnorm(model)
    model.eval()
    _forward = model.forward
    for idx, data_batch in enumerate(data_loader):
        if idx == args.num_images:
            break
        data = model.data_preprocessor(data_batch)
        result['ori_shape'] = data['data_samples'][0].ori_shape
        result['pad_shape'] = data['data_samples'][0].img_shape
        if hasattr(data['data_samples'][0], 'batch_input_shape'):
            result['pad_shape'] = data['data_samples'][0].batch_input_shape
        model.forward = partial(_forward, data_samples=data['data_samples'])
        # if inputs is specified, the input_shape could be ``None``
        outputs = get_model_complexity_info(
            model,
            None,
            inputs=data['inputs'],
            show_table=False,
            show_arch=False)
        avg_flops.append(outputs['flops'])
        params = outputs['params']
        result['compute_type'] = 'dataloader: load a picture from the dataset'
    del data_loader
    # TODO: check if the result of Mask2Former is correct (contain deformable attention cuda opts)
    mean_flops = _format_size(int(np.average(avg_flops)))
    params = _format_size(params)
    result['flops'] = mean_flops
    result['params'] = params

    return result


def main():

    args = parse_args()
    logger = MMLogger.get_instance(name='MMLogger')

    result = inference(args, logger)
    split_line = '=' * 30
    ori_shape = result['ori_shape']
    pad_shape = result['pad_shape']
    flops = result['flops']
    params = result['params']
    compute_type = result['compute_type']

    if pad_shape != ori_shape:
        print(f'{split_line}\nUse size divisor set input shape '
              f'from {ori_shape} to {pad_shape}')
    print(f'{split_line}\nCompute type: {compute_type}\n'
          f'Input shape: {pad_shape}\nFlops: {flops}\n'
          f'Params: {params}\n{split_line}')
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify '
          'that the flops computation is correct.')


if __name__ == '__main__':
    main()