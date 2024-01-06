# dataset settings
dataset_type = 'MMMFDataset'
data_root = '/remote-home/jhli/TIV/TIV/data/MF_RGBT'

# vit-adapter needs square, so crop must has h==w
crop_size = (480, 480) # h, w
img_size = (480, 640) # h, w

train_pipeline = [
    # modality value must be modified
    dict(type='LoadMFImageFromFile', to_float32=True, modality='thermal'),
    dict(type='StackByChannel', keys=('img', 'ano')),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='RandomResize', scale=(640, 480),
         ratio_range=(0.5, 2.0), keep_ratio=True),  # Note: w, h instead of h, w
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs')
]
val_pipeline = [
    # modality value must be modified
    dict(type='LoadMFImageFromFile', to_float32=True, modality='thermal'),
    dict(type='StackByChannel', keys=('img', 'ano')),
    dict(
        type='Resize',
        scale=(640, 480), keep_ratio=True),  # Note: w, h instead of h, w
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs')
]
test_pipeline = [
    # modality value must be modified
    dict(type='LoadMFImageFromFile', to_float32=True, modality='thermal'),
    dict(type='StackByChannel', keys=('img', 'ano')),
    dict(type='Resize', scale=(640, 480), keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs')
]
# tta settings: Note: val will not use this strategy
img_ratios = [1.0, 1.25, 1.5]  # 多尺度预测缩放比例
tta_pipeline = [  # 多尺度测试
    dict(type='LoadMFImageFromFile', to_float32=True, modality='thermal'),
    dict(type='StackByChannel', keys=('img', 'ano')),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
        ])
]

train_dataloader = dict(
    batch_size=5,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        reduce_zero_label=False,
        # have to modify next 2 properties at the same time
        modality='thermal',
        data_prefix=dict(
            img_path='images/train',
            thermal_path='thermal/train',
            seg_map_path='labels/train'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        reduce_zero_label=False,
        # have to modify next 2 properties at the same time
        modality='thermal',
        data_prefix=dict(
            img_path='images/test',
            thermal_path='thermal/test',
            seg_map_path='labels/test'),
        pipeline=val_pipeline))
test_dataloader = dict(
    batch_size=1,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        reduce_zero_label=False,
        # have to modify next 2 properties at the same time
        modality='thermal',
        data_prefix=dict(
            img_path='images/test',
            thermal_path='thermal/test',
            seg_map_path='labels/test'),
        pipeline=test_pipeline))

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mFscore'])
test_evaluator = val_evaluator

convnext_pretrained = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-small_3rdparty_32xb128-noema_in1k_20220301-303e75e3.pth'
beit_pretrained = '/remote-home/jhli/TIV/TIV/pretrained/beitv2_base_patch16_224_pt1k_ft21k.pth'


data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[0.485, 0.456, 0.406, 0, 0, 0], # depth images in NYU has 3 channels
    std=[0.229, 0.224, 0.225, 1, 1, 1],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size,
    # if you want to use tta, then you should give test_cfg or test data won't be padding, and cause errors!
    # test_cfg=dict(size=crop_size)
    )
num_classes = 9

norm_cfg = dict(type='SyncBN', requires_grad=True)
# model setting
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='mmpretrain_custom.BEiTAdapter_rgbxsum',
        pretrained=beit_pretrained,
        img_size=480,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        use_abs_pos_emb=False,
        use_rel_pos_bias=True,
        init_values=1e-6,
        drop_path_rate=0.2,
        n_points=4,
        deform_num_heads=12,
        cffn_ratio=0.25,
        deform_ratio=0.5,
        with_cp=True,  # set with_cp=True to save memory
        interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]],
        # this param is to link with x_modality_encoder
        arch='small',
        x_modality_encoder=dict(
            type='mmpretrain_custom.ShareSumConvNeXt',
            arch='small',
            out_indices=[0, 1, 2, 3],
            drop_path_rate=0.3,
            layer_scale_init_value=1.0,
            gap_before_final_norm=False,
            init_cfg=dict(
                type='Pretrained', checkpoint=convnext_pretrained,
                prefix='backbone.'))),
    decode_head=dict(
        type='UPerHead',
        in_channels=[768, 768, 768, 768],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=768,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=768,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(320, 320))) #h,w

# optimizer
optimizer = dict(
    type='AdamW', lr=2e-5, weight_decay=0.05, eps=1e-8, betas=(0.9, 0.999))
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg=dict(vit_num_layers=12, decay_rate=0.95, x_encoder_num_layers=12),
    clip_grad=dict(max_norm=5.0))

# learning policy
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0,
        power=0.9,
        begin=0,
        end=200,
        by_epoch=True)
]

# training schedule for 160k
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=200, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=True),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', by_epoch=True, interval=100,
        save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', interval=1, draw=False))

# Runtime configs
default_scope = 'mmseg_custom'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
vis_backends = [dict(type='LocalVisBackend'),
                dict(type='WandbVisBackend', init_kwargs=dict(project="ECCV-MFNet-encoder-ablation", name="beit-adapter-b_upernet_layer_decay_constructor_lr2e-5")),
]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(window_size=10, by_epoch=True, custom_cfg=None, num_digits=4)
log_level = 'INFO'
load_from = None
resume = False

tta_model = dict(type='SegTTAModel')
