import numpy as np

import mmcv
import os.path as osp
import torch

from mmengine_custom.structures import PixelData

from mmseg.structures import SegDataSample
from mmseg.visualization import SegLocalVisualizer

# choose type of dataset for demo
dataset_type = 'carla'
# need to modify here
save_dir = '/home/ljh/Desktop/Workspace/mmsegmentation/gt_demo'
# need to modify here
image = mmcv.imread(
    '/home/ljh/Desktop/Workspace/mmsegmentation/data/carla_v3/images/testing/9_798018147.png',
    'color', channel_order='rgb')
# need to modify here
sem_seg = mmcv.imread(
    '/home/ljh/Desktop/Workspace/mmsegmentation/data/carla_v3/annotations/testing/9_798018147.png',
    'unchanged')
seg_local_visualizer = SegLocalVisualizer(
        vis_backends=[dict(type='LocalVisBackend')],
        save_dir=save_dir,
        alpha=0.8)

if dataset_type == 'cityscapes':
    sem_seg = torch.from_numpy(sem_seg)
    gt_sem_seg_data = dict(data=sem_seg)
    gt_sem_seg = PixelData(**gt_sem_seg_data)
    data_sample = SegDataSample()
    data_sample.gt_sem_seg = gt_sem_seg

    seg_local_visualizer.dataset_meta = dict(
        classes=('road', 'sidewalk', 'building', 'wall', 'fence',
                 'pole', 'traffic light', 'traffic sign',
                 'vegetation', 'terrain', 'sky', 'person', 'rider',
                 'car', 'truck', 'bus', 'train', 'motorcycle',
                 'bicycle'),
        palette=[[128, 64, 128], [244, 35, 232], [70, 70, 70],
                 [102, 102, 156], [190, 153, 153], [153, 153, 153],
                 [250, 170, 30], [220, 220, 0], [107, 142, 35],
                 [152, 251, 152], [70, 130, 180], [220, 20, 60],
                 [255, 0, 0], [0, 0, 142], [0, 0, 70],
                 [0, 60, 100], [0, 80, 100], [0, 0, 230],
                 [119, 11, 32]])
    # 当`show=True`时，直接显示结果，
    # 当 `show=False`时，结果将保存在本地文件夹中。
    name = 'cityscapes'
    seg_local_visualizer.add_datasample(name, image,
                                        data_sample, draw_gt=True, draw_pred=False, show=False)

# carla does not contain direct labels, only has
# rgb_visual_annotations, so transform it to true label
elif dataset_type in ['kitti', 'carla']:
    if dataset_type == 'carla':
        label = np.zeros_like(sem_seg[:, :, 0])
        label[sem_seg[:,:,2]==128] = 1
        label[sem_seg[:,:,1]==255] = 2
    elif dataset_type == 'kitti':
        label = np.zeros_like(sem_seg[:, :, 0])
        label[(sem_seg[:, :, 0] == 255) & (sem_seg[:, :, 2] == 255)] = 0

    label = torch.from_numpy(label)
    gt_sem_seg_data = dict(data=label)
    gt_sem_seg = PixelData(**gt_sem_seg_data)
    data_sample = SegDataSample()
    data_sample.gt_sem_seg = gt_sem_seg

    seg_local_visualizer.dataset_meta = dict(
        classes=('ignore', 'road', 'pothole'),
        palette=[[0, 0, 0], [128, 64, 128], [0, 255, 0]])
    # 当`show=True`时，直接显示结果，
    # 当 `show=False`时，结果将保存在本地文件夹中。
    name = 'kitti_carla'
    seg_local_visualizer.add_datasample(name, image,
                                        data_sample, draw_gt=True, draw_pred=False, show=False)