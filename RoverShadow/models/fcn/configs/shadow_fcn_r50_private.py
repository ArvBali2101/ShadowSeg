_base_ = [
    "mmseg::_base_/models/fcn_r50-d8.py",
    "mmseg::_base_/default_runtime.py",
    "mmseg::_base_/schedules/schedule_20k.py",
]

param_scheduler = None


train_dataloader = None

train_cfg = None

optim_wrapper = None

param_scheduler = None

val_dataloader = None

val_cfg = None

val_evaluator = None

num_classes = 2

metainfo = dict(classes=("background", "shadow"), palette=[(0, 0, 0), (255, 255, 255)])

data_preprocessor = dict(
    type="SegDataPreProcessor",
    bgr_to_rgb=True,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    pad_val=0,
    seg_pad_val=255,
    size=(512, 512),
    size_divisor=None,
)

model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=num_classes),
    auxiliary_head=dict(num_classes=num_classes),
)

dataset_type = "BaseSegDataset"

data_root = "data/private/LunarShadowDataset"

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]


test_dataloader = dict(
    batch_size=1,
    num_workers=0,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        img_suffix=".png",
        seg_map_suffix=".png",
        data_prefix=dict(img_path="ShadowImages", seg_map_path="ShadowMasks"),
        pipeline=test_pipeline,
    ),
)

test_evaluator = dict(type="IoUMetric", iou_metrics=["mIoU"])
