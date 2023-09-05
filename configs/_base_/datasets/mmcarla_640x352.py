# dataset settings
dataset_type = 'MMCarlaDataset'
data_root = '/home/ljh/Desktop/Workspace/mmsegmentation/data/carla_v3'
train_pipeline = [
    dict(type='LoadMultimodalImageFromFile', to_float32=True, modality='normal'),  # modality value must be modified
    dict(type='StackByChannel', keys=('img', 'ano')),
    dict(type='LoadCarlaAnnotations', reduce_zero_label=False),
    dict(
        type='Resize',
        scale=(640, 352)),  # Note: w, h instead of h, w
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadMultimodalImageFromFile', to_float32=True, modality='normal'),  # modality value must be modified
    dict(type='StackByChannel', keys=('img', 'ano')),
    dict(type='Resize', scale=(640, 352)),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadCarlaAnnotations', reduce_zero_label=False),
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
        img_suffix='.png',
        modality='normal',
        data_prefix=dict(
            img_path='images/training',
            depth_path='depth/training',
            disp_path='disparity/training',
            tdisp_path='tdisp/training', # had an issue in tdisp data, solve it in future
            normal_path='normal/training',
            seg_map_path='annotations/training'),
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
        img_suffix='.png',
        modality='normal',
        data_prefix=dict(
            img_path='images/validation',
            depth_path='depth/validation',
            disp_path='disparity/validation',
            tdisp_path='tdisp/validation',  # had an issue in tdisp data, solve it in future
            normal_path='normal/validation',
            seg_map_path='annotations/validation'),
        pipeline=test_pipeline))
test_dataloader = dict(
    batch_size=1,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        reduce_zero_label=False,
        img_suffix='.png',
        modality='normal',
        data_prefix=dict(
            img_path='images/testing',
            depth_path='depth/testing',
            disp_path='disparity/testing',
            tdisp_path='tdisp/testing',  # had an issue in tdisp data, solve it in future
            normal_path='normal/testing',
            seg_map_path='annotations/testing'),
        pipeline=test_pipeline))

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mFscore'])
test_evaluator = val_evaluator
