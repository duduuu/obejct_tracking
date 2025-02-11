img_scale = (540, 960)
model = dict(
    data_preprocessor=dict(
        type='TrackDataPreprocessor',
        pad_size_divisor=32,
        batch_augments=[
            dict(
                type='mmdet.BatchSyncRandomResize',
                random_size_range=(576, 1024),
                size_divisor=32,
                interval=10)
        ]),
    detector=dict(
        _scope_='mmdet',
        type='YOLOX',
        backbone=dict(
            type='CSPDarknet', deepen_factor=1.33, widen_factor=1.25),
        neck=dict(
            type='YOLOXPAFPN',
            in_channels=[320, 640, 1280],
            out_channels=320,
            num_csp_blocks=4),
        bbox_head=dict(
            type='YOLOXHead',
            num_classes=1,
            in_channels=320,
            feat_channels=320),
        train_cfg=dict(
            assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
        test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.7)),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_x_8x8_300e_coco/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth'
        )),
    type='ByteTrack',
    motion=dict(type='KalmanFilter'),
    tracker=dict(
        type='ByteTracker',
        obj_score_thrs=dict(high=0.6, low=0.1),
        init_track_thr=0.7,
        weight_iou_with_det_scores=True,
        match_iou_thrs=dict(high=0.1, low=0.5, tentative=0.3),
        num_frames_retain=30))
dataset_type = 'BaseVideoDataset'
data_root = '/workspace/data/01_data'
train_pipeline = [
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(
        type='mmdet.Resize',
        scale=(540, 960),
        keep_ratio=True,
        clip_object_border=False),
    dict(
        type='mmdet.Pad',
        size_divisor=32,
        pad_val=dict(img=(114.0, 114.0, 114.0)))
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='mmdet.Resize', scale=(540, 960), keep_ratio=True),
    dict(
        type='mmdet.Pad',
        size_divisor=32,
        pad_val=dict(img=(114.0, 114.0, 114.0)))
]
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='BaseVideoDataset',
        data_root='/workspace/data/01_data_ir',
        visibility_thr=-1,
        ann_file='annotations/half-train_cocoformat.json',
        data_prefix=dict(img_path='train'),
        metainfo=dict(CLASSES=('pedestrian', )),
        pipeline=[
            dict(type='mmdet.RandomFlip', prob=0.5),
            dict(
                type='mmdet.Resize',
                scale=(540, 960),
                keep_ratio=True,
                clip_object_border=False),
            dict(
                type='mmdet.Pad',
                size_divisor=32,
                pad_val=dict(img=(114.0, 114.0, 114.0)))
        ]))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BaseVideoDataset',
        data_root='/workspace/data/01_data_ir',
        ann_file='annotations/half-val_cocoformat.json',
        data_prefix=dict(img_path='train'),
        ref_img_sampler=None,
        load_as_video=True,
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='mmdet.Resize', scale=(540, 960), keep_ratio=True),
            dict(
                type='mmdet.Pad',
                size_divisor=32,
                pad_val=dict(img=(114.0, 114.0, 114.0)))
        ]))
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BaseVideoDataset',
        data_root='/workspace/data/01_data_ir',
        ann_file='annotations/half-val_cocoformat.json',
        data_prefix=dict(img_path='train'),
        ref_img_sampler=None,
        load_as_video=True,
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='mmdet.Resize', scale=(540, 960), keep_ratio=True),
            dict(
                type='mmdet.Pad',
                size_divisor=32,
                pad_val=dict(img=(114.0, 114.0, 114.0)))
        ]))
val_evaluator = dict(type='MOTChallengeMetrics', metric=['Identity'])
test_evaluator = dict(type='MOTChallengeMetrics', metric=['Identity'])
default_scope = 'mmtrack'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='TrackVisualizationHook', draw=False))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='TrackLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer')
log_level = 'INFO'
load_from = None
resume = True
batch_size = 4
auto_scale_lr = dict(enable=True, base_batch_size=4)
lr = 0.004
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(
        type='SGD', lr=0.004, momentum=0.9, weight_decay=0.0005,
        nesterov=True),
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0),
    loss_scale='dynamic')
total_epochs = 80
num_last_epochs = 10
resume_from = None
interval = 5
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=80, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
param_scheduler = [
    dict(
        type='mmdet.QuadraticWarmupLR',
        by_epoch=True,
        begin=0,
        end=1,
        convert_to_iter_based=True),
    dict(
        type='mmdet.CosineAnnealingLR',
        eta_min=0.0002,
        begin=1,
        T_max=70,
        end=70,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(type='mmdet.ConstantLR', by_epoch=True, factor=1, begin=70, end=80)
]
custom_hooks = [
    dict(type='YOLOXModeSwitchHook', num_last_epochs=10, priority=48),
    dict(type='mmdet.SyncNormHook', priority=48),
    dict(
        type='mmdet.EMAHook',
        ema_type='mmdet.ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        priority=49)
]
launcher = 'none'
randomness = dict(seed=777, deterministic=True)
gpu_ids = range(0, 1)
work_dir = './work_dirs/bytetrack_yolox_x_8xb4-80e_maicon-ir-cocovidhalftrain_test-mot17halfval'
