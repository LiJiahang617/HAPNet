# dataset settings
dataset_type = 'MMNYUDataset'
data_root = '/media/ljh/Kobe24/NYU_oridepth'
sample_scale = (640, 480)

train_pipeline = [
    # modality value must be modified
    dict(type='LoadNYUImageFromFile', to_float32=True, modality='HHA'),
    dict(type='StackByChannel', keys=('img', 'ano')),
    dict(type='LoadNYUAnnotations', reduce_zero_label=True),
    dict(
        type='RandomChoiceResize',
        scales=[int(640 * x * 0.1) for x in range(5, 20)],
        resize_type='ResizeShortestEdge',
        max_size=1280),  # Note: w, h instead of h, w
    dict(type='RandomCrop', crop_size=(480, 640), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs')
]
val_pipeline = [
    # modality value must be modified
    dict(type='LoadNYUImageFromFile', to_float32=True, modality='HHA'),
    dict(type='StackByChannel', keys=('img', 'ano')),
    dict(
        type='Resize',
        scale=sample_scale, keep_ratio=True),  # Note: w, h instead of h, w
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadNYUAnnotations', reduce_zero_label=True),
    dict(type='PackSegInputs')
]
test_pipeline = [
    # modality value must be modified
    dict(type='LoadNYUImageFromFile', to_float32=True, modality='HHA'),
    dict(type='StackByChannel', keys=('img', 'ano')),
    dict(type='Resize', scale=sample_scale, keep_ratio=True),
    dict(type='LoadNYUAnnotations', reduce_zero_label=True),
    dict(type='PackSegInputs')
]
train_dataloader = dict(
    batch_size=4,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        reduce_zero_label=True,
        # have to modify next 2 properties at the same time
        modality='HHA',
        ano_suffix='.jpg',
        data_prefix=dict(
            img_path='images/train',
            depth_path='depth/train',
            hha_path='HHA/train',
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
        reduce_zero_label=True,
        # have to modify next 2 properties at the same time
        modality='HHA',
        ano_suffix='.jpg',
        data_prefix=dict(
            img_path='images/test',
            depth_path='depth/test',
            hha_path='HHA/test',
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
        reduce_zero_label=True,
        # have to modify next 2 properties at the same time
        modality='HHA',
        ano_suffix='.jpg',
        data_prefix=dict(
            img_path='images/test',
            depth_path='depth/test',
            hha_path='HHA/test',
            seg_map_path='labels/test'),
        pipeline=test_pipeline))

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mFscore'])
test_evaluator = val_evaluator


