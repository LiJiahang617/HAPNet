dataset_type = 'MMCityscapesDataset'
data_root = '/media/ljh/data/Cityscapes'
sample_scale = (1024, 512)
train_pipeline = [
    dict(
        type='LoadCityscapesImageFromFile', to_float32=True,
        modality='normal'),
    dict(type='StackByChannel', keys=('img', 'ano')),
    dict(type='LoadCityscapesAnnotations', reduce_zero_label=False),
    dict(type='Resize', scale=(1024, 512)),
    dict(type='PackSegInputs')
]
val_pipeline = [
    dict(
        type='LoadCityscapesImageFromFile', to_float32=True,
        modality='normal'),
    dict(type='StackByChannel', keys=('img', 'ano')),
    dict(type='Resize', scale=(1024, 512)),
    dict(type='LoadCityscapesAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(
        type='LoadCityscapesImageFromFile', to_float32=True,
        modality='normal'),
    dict(type='StackByChannel', keys=('img', 'ano')),
    dict(type='Resize', scale=(1024, 512)),
    dict(type='PackSegInputs')
]
train_dataloader = dict(
    batch_size=1,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='MMCityscapesDataset',
        data_root='/media/ljh/data/Cityscapes',
        reduce_zero_label=False,
        modality='normal',
        ano_suffix='_normal.png',
        data_prefix=dict(
            img_path='images/train',
            disp_path='disp/train',
            normal_path='sne/train',
            seg_map_path='annotations/train'),
        pipeline=[
            dict(
                type='LoadCityscapesImageFromFile',
                to_float32=True,
                modality='normal'),
            dict(type='StackByChannel', keys=('img', 'ano')),
            dict(type='LoadCityscapesAnnotations', reduce_zero_label=False),
            dict(type='Resize', scale=(1024, 512)),
            dict(type='PackSegInputs')
        ]))
val_dataloader = dict(
    batch_size=1,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='MMCityscapesDataset',
        data_root='/media/ljh/data/Cityscapes',
        reduce_zero_label=False,
        modality='normal',
        ano_suffix='_normal.png',
        data_prefix=dict(
            img_path='images/val',
            disp_path='disp/val',
            normal_path='sne/val',
            seg_map_path='annotations/val'),
        pipeline=[
            dict(
                type='LoadCityscapesImageFromFile',
                to_float32=True,
                modality='normal'),
            dict(type='StackByChannel', keys=('img', 'ano')),
            dict(type='Resize', scale=(1024, 512)),
            dict(type='LoadCityscapesAnnotations', reduce_zero_label=False),
            dict(type='PackSegInputs')
        ]))
test_dataloader = dict(
    batch_size=1,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='MMCityscapesDataset',
        data_root='/media/ljh/data/Cityscapes',
        reduce_zero_label=False,
        modality='normal',
        ano_suffix='_normal.png',
        data_prefix=dict(
            img_path='images/test',
            disp_path='disp/test',
            normal_path='sne/test',
            seg_map_path='annotations/test'),
        pipeline=[
            dict(
                type='LoadCityscapesImageFromFile',
                to_float32=True,
                modality='normal'),
            dict(type='StackByChannel', keys=('img', 'ano')),
            dict(type='Resize', scale=(1024, 512)),
            dict(type='PackSegInputs')
        ]))
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mFscore'])
test_evaluator = dict(
    type='IoUMetric',
    iou_metrics=['mIoU', 'mFscore'],
    format_only=True,
    output_dir='work_dirs/format_results')
pretrained = '/home/ljh/Desktop/TIV/Workspace/RoadFormer/pretrain/mit_b0_20220624-7e0fe6dd.pth'
crop_size = (512, 1024)
norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[0, 0, 0, 0, 0, 0],
    std=[1, 1, 1, 1, 1, 1],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=(512, 1024))
num_classes = 19
model = dict(
    type='EncoderDecoder',
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[0, 0, 0, 0, 0, 0],
        std=[1, 1, 1, 1, 1, 1],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
        size=(512, 1024)),
    backbone=dict(
        type='TwinMixVisionTransformer',
        in_channels=3,
        embed_dims=32,
        num_stages=4,
        num_layers=[2, 2, 2, 2],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            '/home/ljh/Desktop/TIV/Workspace/RoadFormer/pretrain/mit_b0_20220624-7e0fe6dd.pth'
        )),
    decode_head=dict(
        type='AllmlpHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
optimizer = dict(
    type='AdamW', lr=6e-05, weight_decay=0.01, eps=1e-08, betas=(0.9, 0.999))
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=6e-05,
        weight_decay=0.01,
        eps=1e-08,
        betas=(0.9, 0.999)),
    paramwise_cfg=dict(
        custom_keys=dict(
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
            head=dict(lr_mult=10.0))),
    clip_grad=dict(max_norm=5.0))
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-06, by_epoch=False, begin=0,
        end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False)
]
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=160000, val_interval=16000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=16000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', interval=100, draw=False))
default_scope = 'mmseg_custom'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer')
log_processor = dict(
    window_size=10, by_epoch=True, custom_cfg=None, num_digits=4)
log_level = 'INFO'
load_from = None
resume = False
tta_model = dict(type='SegTTAModel')
launcher = 'none'
work_dir = './work_dirs/mit-b0_mlp_cityscapes-512x1024'
