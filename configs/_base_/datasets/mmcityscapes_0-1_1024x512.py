# dataset settings
dataset_type = 'MMCityscapesDataset'
data_root = '/media/ljh/data/Cityscapes'
sample_scale = (1024, 512)

train_pipeline = [
    # modality value must be modified
    dict(type='LoadCityscapesImageFromFile', to_float32=True, modality='normal'),
    dict(type='StackByChannel', keys=('img', 'ano')),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(
        type='RandomChoiceResize',
        scales=[int(1024 * x * 0.1) for x in range(5, 21)],
        resize_type='ResizeShortestEdge',
        max_size=4096),  # Note: w, h instead of h, w
    dict(type='RandomCrop', crop_size=(512, 1024), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs')
]
val_pipeline = [
    # modality value must be modified
    dict(type='LoadCityscapesImageFromFile', to_float32=True, modality='normal'),
    dict(type='StackByChannel', keys=('img', 'ano')),
    dict(
        type='Resize',
        scale=sample_scale, keep_ratio=True),  # Note: w, h instead of h, w
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs')
]
test_pipeline = [
    # modality value must be modified
    dict(type='LoadCityscapesImageFromFile', to_float32=True, modality='normal'),
    dict(type='StackByChannel', keys=('img', 'ano')),
    dict(type='Resize', scale=sample_scale, keep_ratio=True),
    dict(type='PackSegInputs')
]
train_dataloader = dict(
    batch_size=2,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        reduce_zero_label=False,
        # have to modify next 2 properties at the same time
        modality='normal',
        ano_suffix='_normal.png',
        data_prefix=dict(
            img_path='images/train',
            disp_path='disp/train',
            normal_path='sne/train',
            seg_map_path='annotations/train'),
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
        modality='normal',
        ano_suffix='_normal.png',
        data_prefix=dict(
            img_path='images/val',
            disp_path='disp/val',
            normal_path='sne/val',
            seg_map_path='annotations/val'),
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
        modality='normal',
        ano_suffix='_normal.png',
        data_prefix=dict(
            img_path='images/test',
            disp_path='disp/test',
            normal_path='sne/test',
            seg_map_path='annotations/test'),
        pipeline=test_pipeline))

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mFscore'])
test_evaluator = dict(
    type='IoUMetric',
    iou_metrics=['mIoU', 'mFscore'],
    format_only=True,
    output_dir='work_dirs/format_results')

