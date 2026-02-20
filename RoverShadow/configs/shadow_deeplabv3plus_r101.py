crop_size = (
    512,
    512,
)

custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        "rovershadow.losses",
    ],
)

data_preprocessor = dict(
    _scope_="mmseg",
    bgr_to_rgb=True,
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    pad_val=0,
    seg_pad_val=255,
    size=(
        512,
        512,
    ),
    size_divisor=None,
    std=[
        58.395,
        57.12,
        57.375,
    ],
    type="SegDataPreProcessor",
)

data_root = "data/public/Rover_Shadow_Public_Dataset"

dataset_type = "BaseSegDataset"

default_hooks = dict(
    checkpoint=dict(
        _scope_="mmseg",
        by_epoch=False,
        interval=1000,
        max_keep_ckpts=12,
        type="CheckpointHook",
    ),
    logger=dict(
        _scope_="mmseg", interval=50, log_metric_by_epoch=False, type="LoggerHook"
    ),
    param_scheduler=dict(_scope_="mmseg", type="ParamSchedulerHook"),
    sampler_seed=dict(_scope_="mmseg", type="DistSamplerSeedHook"),
    timer=dict(_scope_="mmseg", type="IterTimerHook"),
    visualization=dict(_scope_="mmseg", type="SegVisualizationHook"),
)

default_scope = "mmseg"

device = "cuda"

env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend="nccl"),
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
)

launcher = "none"

load_from = None

log_level = "INFO"

log_processor = dict(by_epoch=False)

metainfo = dict(
    classes=(
        "background",
        "shadow",
    ),
    palette=[
        (
            0,
            0,
            0,
        ),
        (
            255,
            255,
            255,
        ),
    ],
)

model = dict(
    _scope_="mmseg",
    auxiliary_head=dict(
        align_corners=False,
        channels=256,
        concat_input=False,
        dropout_ratio=0.1,
        in_channels=1024,
        in_index=2,
        loss_decode=dict(
            avg_non_ignore=True,
            class_weight=[
                1.3,
                1.0,
            ],
            loss_weight=0.4,
            type="SafeCrossEntropyLoss",
            use_sigmoid=False,
        ),
        norm_cfg=dict(requires_grad=True, type="SyncBN"),
        num_classes=2,
        num_convs=1,
        type="FCNHead",
    ),
    backbone=dict(
        contract_dilation=True,
        depth=101,
        dilations=(
            1,
            1,
            2,
            4,
        ),
        norm_cfg=dict(requires_grad=True, type="SyncBN"),
        norm_eval=False,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        strides=(
            1,
            2,
            1,
            1,
        ),
        style="pytorch",
        type="ResNetV1c",
    ),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_val=0,
        seg_pad_val=255,
        size=(
            512,
            512,
        ),
        size_divisor=None,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type="SegDataPreProcessor",
    ),
    decode_head=dict(
        align_corners=False,
        c1_channels=48,
        c1_in_channels=256,
        channels=512,
        dilations=(
            1,
            12,
            24,
            36,
        ),
        dropout_ratio=0.1,
        in_channels=2048,
        in_index=3,
        loss_decode=[
            dict(
                avg_non_ignore=True,
                class_weight=[
                    1.3,
                    1.0,
                ],
                loss_weight=1.0,
                type="SafeCrossEntropyLoss",
                use_sigmoid=False,
            ),
            dict(alpha=0.7, beta=0.3, loss_weight=0.4, type="TverskyLoss"),
            dict(
                ignore_index=255,
                loss_weight=0.15,
                shadow_class=1,
                type="ShadowFalsePositiveLoss",
            ),
        ],
        norm_cfg=dict(requires_grad=True, type="SyncBN"),
        num_classes=2,
        type="DepthwiseSeparableASPPHead",
    ),
    pretrained="open-mmlab://resnet101_v1c",
    test_cfg=dict(mode="whole"),
    train_cfg=dict(),
    type="EncoderDecoder",
)

norm_cfg = dict(_scope_="mmseg", requires_grad=True, type="SyncBN")

num_classes = 2

optim_wrapper = dict(
    _scope_="mmseg",
    clip_grad=None,
    optimizer=dict(lr=0.005, momentum=0.9, type="SGD", weight_decay=0.0005),
    type="OptimWrapper",
)

optimizer = dict(
    _scope_="mmseg", lr=0.005, momentum=0.9, type="SGD", weight_decay=0.0005
)

param_scheduler = [
    dict(begin=0, by_epoch=False, end=8000, eta_min=0.0001, power=0.9, type="PolyLR"),
]

randomness = dict(deterministic=False, seed=42)

resume = False

test_cfg = dict(_scope_="mmseg", type="TestLoop")

test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(img_path="ShadowImages/val", seg_map_path="ShadowMasks/val"),
        data_root="data/public/Rover_Shadow_Public_Dataset",
        metainfo=dict(
            classes=(
                "background",
                "shadow",
            ),
            palette=[
                (
                    0,
                    0,
                    0,
                ),
                (
                    255,
                    255,
                    255,
                ),
            ],
        ),
        pipeline=[
            dict(type="LoadImageFromFile"),
            dict(type="LoadAnnotations"),
            dict(type="PackSegInputs"),
        ],
        type="BaseSegDataset",
    ),
    num_workers=0,
    persistent_workers=False,
    sampler=dict(shuffle=False, type="DefaultSampler"),
)

test_evaluator = dict(
    iou_metrics=[
        "mIoU",
    ],
    type="IoUMetric",
)

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]

train_cfg = dict(
    _scope_="mmseg", max_iters=8000, type="IterBasedTrainLoop", val_interval=1000
)

train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        data_prefix=dict(
            img_path="ShadowImages/train", seg_map_path="ShadowMasks/train"
        ),
        data_root="data/public/Rover_Shadow_Public_Dataset",
        metainfo=dict(
            classes=(
                "background",
                "shadow",
            ),
            palette=[
                (
                    0,
                    0,
                    0,
                ),
                (
                    255,
                    255,
                    255,
                ),
            ],
        ),
        pipeline=[
            dict(type="LoadImageFromFile"),
            dict(type="LoadAnnotations"),
            dict(
                keep_ratio=True,
                ratio_range=(
                    0.5,
                    2.0,
                ),
                scale=(
                    1024,
                    1024,
                ),
                type="RandomResize",
            ),
            dict(
                cat_max_ratio=0.85,
                crop_size=(
                    512,
                    512,
                ),
                type="RandomCrop",
            ),
            dict(prob=0.5, type="RandomFlip"),
            dict(degree=20, prob=0.3, type="RandomRotate"),
            dict(
                prob=0.8,
                transforms=[
                    dict(type="PhotoMetricDistortion"),
                ],
                type="RandomApply",
            ),
            dict(
                transforms=[
                    [
                        dict(
                            prob=0.5,
                            transforms=[
                                dict(
                                    clip_limit=2.0,
                                    tile_grid_size=(
                                        8,
                                        8,
                                    ),
                                    type="CLAHE",
                                ),
                            ],
                            type="RandomApply",
                        ),
                    ],
                    [
                        dict(
                            prob=0.5,
                            transforms=[
                                dict(gamma=0.7, type="AdjustGamma"),
                            ],
                            type="RandomApply",
                        ),
                    ],
                    [
                        dict(
                            prob=0.3,
                            transforms=[
                                dict(gamma=1.5, type="AdjustGamma"),
                            ],
                            type="RandomApply",
                        ),
                    ],
                ],
                type="RandomChoice",
            ),
            dict(
                cutout_ratio=[
                    (
                        0.05,
                        0.05,
                    ),
                    (
                        0.1,
                        0.1,
                    ),
                ],
                n_holes=(
                    1,
                    2,
                ),
                prob=0.15,
                seg_fill_in=0,
                type="RandomCutOut",
            ),
            dict(type="PackSegInputs"),
        ],
        type="BaseSegDataset",
    ),
    num_workers=0,
    persistent_workers=False,
    sampler=dict(shuffle=True, type="InfiniteSampler"),
)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(
        keep_ratio=True,
        ratio_range=(
            0.5,
            2.0,
        ),
        scale=(
            1024,
            1024,
        ),
        type="RandomResize",
    ),
    dict(
        cat_max_ratio=0.85,
        crop_size=(
            512,
            512,
        ),
        type="RandomCrop",
    ),
    dict(prob=0.5, type="RandomFlip"),
    dict(degree=20, prob=0.3, type="RandomRotate"),
    dict(
        prob=0.8,
        transforms=[
            dict(type="PhotoMetricDistortion"),
        ],
        type="RandomApply",
    ),
    dict(
        transforms=[
            [
                dict(
                    prob=0.5,
                    transforms=[
                        dict(
                            clip_limit=2.0,
                            tile_grid_size=(
                                8,
                                8,
                            ),
                            type="CLAHE",
                        ),
                    ],
                    type="RandomApply",
                ),
            ],
            [
                dict(
                    prob=0.5,
                    transforms=[
                        dict(gamma=0.7, type="AdjustGamma"),
                    ],
                    type="RandomApply",
                ),
            ],
            [
                dict(
                    prob=0.3,
                    transforms=[
                        dict(gamma=1.5, type="AdjustGamma"),
                    ],
                    type="RandomApply",
                ),
            ],
        ],
        type="RandomChoice",
    ),
    dict(
        cutout_ratio=[
            (
                0.05,
                0.05,
            ),
            (
                0.1,
                0.1,
            ),
        ],
        n_holes=(
            1,
            2,
        ),
        prob=0.15,
        seg_fill_in=0,
        type="RandomCutOut",
    ),
    dict(type="PackSegInputs"),
]

tta_model = dict(_scope_="mmseg", type="SegTTAModel")

val_cfg = dict(_scope_="mmseg", type="ValLoop")

val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(img_path="ShadowImages/val", seg_map_path="ShadowMasks/val"),
        data_root="data/public/Rover_Shadow_Public_Dataset",
        metainfo=dict(
            classes=(
                "background",
                "shadow",
            ),
            palette=[
                (
                    0,
                    0,
                    0,
                ),
                (
                    255,
                    255,
                    255,
                ),
            ],
        ),
        pipeline=[
            dict(type="LoadImageFromFile"),
            dict(type="LoadAnnotations"),
            dict(type="PackSegInputs"),
        ],
        type="BaseSegDataset",
    ),
    num_workers=0,
    persistent_workers=False,
    sampler=dict(shuffle=False, type="DefaultSampler"),
)

val_evaluator = dict(
    iou_metrics=[
        "mIoU",
    ],
    type="IoUMetric",
)

val_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]

vis_backends = [
    dict(_scope_="mmseg", type="LocalVisBackend"),
]

visualizer = dict(
    _scope_="mmseg",
    name="visualizer",
    type="SegLocalVisualizer",
    vis_backends=[
        dict(type="LocalVisBackend"),
    ],
)

work_dir = "work_dirs/phase3_r101_candidate_8000"
