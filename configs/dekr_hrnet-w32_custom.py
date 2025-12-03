# ============================================================================
# DEKR HRNet-W32 自定义训练配置文件
# 
# 基于: dekr_hrnet-w32_8xb10-140e_coco-512x512.py
# 用于在自定义数据集上训练DEKR自底向上姿态估计模型
# ============================================================================

_base_ = ['/home/satuo/code/Train_Parallel_Model/configs/_base_/default_runtime.py']

# ============================================================================
# 训练配置
# ============================================================================
train_cfg = dict(max_epochs=140, val_interval=10)

# ============================================================================
# 优化器配置
# ============================================================================
optim_wrapper = dict(optimizer=dict(
    type='Adam',
    lr=1e-3,  # 学习率，如果batch_size改变，建议按比例调整
))

# ============================================================================
# 学习率调度策略
# ============================================================================
param_scheduler = [
    dict(
        type='LinearLR', begin=0, end=500, start_factor=0.001,
        by_epoch=False),  # warm-up
    dict(
        type='MultiStepLR',
        begin=0,
        end=140,
        milestones=[90, 120],  # 在第90和120个epoch降低学习率
        gamma=0.1,
        by_epoch=True)
]

# 根据实际训练batch size自动缩放学习率
auto_scale_lr = dict(base_batch_size=80)

# ============================================================================
# Hooks配置
# ============================================================================
default_hooks = dict(checkpoint=dict(save_best='coco/AP', rule='greater'))

# ============================================================================
# Codec设置（用于编码/解码关键点和分组信息）
# ============================================================================
codec = dict(
    type='SPR',
    input_size=(512, 512),  # 输入图像尺寸，如果GPU内存不足可以减小（如384x384）
    heatmap_size=(128, 128),
    sigma=(4, 2),
    minimal_diagonal_length=32**0.5,
    generate_keypoint_heatmaps=True,
    decode_max_instances=30)  # 最多检测30个人，根据场景调整

# ============================================================================
# 模型配置
# ============================================================================
model = dict(
    type='BottomupPoseEstimator',  # 自底向上姿态估计器
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='HRNet',
        in_channels=3,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(32, 64)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(32, 64, 128)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(32, 64, 128, 256),
                multiscale_output=True)),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmpose/'
            'pretrain_models/hrnet_w32-36af842e.pth'),
    ),
    neck=dict(
        type='FeatureMapProcessor',
        concat=True,
    ),
    head=dict(
        type='DEKRHead',
        in_channels=480,
        num_keypoints=21,  # 关键点数量：COCO标准17个 + 新增4个（left_heel, right_heel, left_foot, right_foot）
        heatmap_loss=dict(type='KeypointMSELoss', use_target_weight=True),
        displacement_loss=dict(
            type='SoftWeightSmoothL1Loss',
            use_target_weight=True,
            supervise_empty=False,
            beta=1 / 9,
            loss_weight=0.002,
        ),
        decoder=codec,
        # ====================================================================
        # 注意：rescore_cfg是为标准COCO数据集设计的
        # 如果使用自定义数据集，建议注释掉或删除以下部分
        # ====================================================================
        # rescore_cfg=dict(
        #     in_channels=74,
        #     norm_indexes=(5, 6),
        #     init_cfg=dict(
        #         type='Pretrained',
        #         checkpoint='https://download.openmmlab.com/mmpose/'
        #         'pretrain_models/kpt_rescore_coco-33d58c5c.pth')),
    ),
    test_cfg=dict(
        multiscale_test=False,
        flip_test=True,
        nms_dist_thr=0.05,
        shift_heatmap=True,
        align_corners=False))

# 当使用rescore net时启用DDP训练
find_unused_parameters = True

# ============================================================================
# 数据集配置
# ============================================================================
dataset_type = 'CocoDataset'
data_mode = 'bottomup'  # 自底向上数据模式

# ============================================================================
# 数据路径配置
# ============================================================================
# 修改为您的数据根目录
# 建议使用绝对路径，例如：
# data_root = '/home/satuo/code/Train_Parallel_Model/data/coco/'
data_root = 'data/coco/'

# ============================================================================
# 数据变换管道（Pipeline）
# ============================================================================
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='BottomupRandomAffine', input_size=codec['input_size']),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='BottomupGetHeatmapMask'),
    dict(type='PackPoseInputs'),
]

val_pipeline = [
    dict(type='LoadImage'),
    dict(
        type='BottomupResize',
        input_size=codec['input_size'],
        size_factor=32,
        resize_mode='expand'),
    dict(
        type='PackPoseInputs',
        meta_keys=('id', 'img_id', 'img_path', 'crowd_index', 'ori_shape',
                   'img_shape', 'input_size', 'input_center', 'input_scale',
                   'flip', 'flip_direction', 'flip_indices', 'raw_ann_info',
                   'skeleton_links'))
]

# ============================================================================
# 数据加载器配置
# ============================================================================
train_dataloader = dict(
    batch_size=10,  # 批次大小，根据GPU内存调整（如果OOM，可以减小到4或8）
    num_workers=2,  # 数据加载线程数，根据CPU核心数调整
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        # 修改为您的训练集标注文件路径
        ann_file='annotations/person_keypoints_train2017.json',
        # 修改为您的训练图像目录
        data_prefix=dict(img='train2017/'),
        pipeline=train_pipeline,
    ))

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        # 修改为您的验证集标注文件路径
        ann_file='annotations/person_keypoints_val2017.json',
        # 修改为您的验证图像目录
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=val_pipeline,
    ))

test_dataloader = val_dataloader

# ============================================================================
# 评估器配置
# ============================================================================
val_evaluator = dict(
    type='CocoMetric',
    # 修改为您的验证集标注文件路径（使用绝对路径或相对于data_root的路径）
    ann_file=data_root + 'annotations/person_keypoints_val2017.json',
    nms_mode='none',
    score_mode='keypoint',
)

test_evaluator = val_evaluator

# ============================================================================
# 配置说明
# ============================================================================
# 
# 1. 数据路径修改：
#    - 修改 data_root 为您的数据根目录
#    - 修改 train_dataloader.dataset.ann_file 和 data_prefix
#    - 修改 val_dataloader.dataset.ann_file 和 data_prefix
#    - 修改 val_evaluator.ann_file
#
# 2. 训练参数调整：
#    - batch_size: 根据GPU内存调整（如果OOM，减小此值）
#    - lr: 如果batch_size改变，按比例调整学习率
#    - max_epochs: 根据数据集大小调整训练轮数
#    - input_size: 如果GPU内存不足，可以减小（如384x384）
#
# 3. 关键点数量：
#    - 如果使用不同的关键点定义，修改 num_keypoints
#    - 同时需要修改转换脚本中的 STANDARD_KEYPOINT_ORDER
#
# 4. rescore_cfg：
#    - 如果使用自定义数据集，建议注释掉或删除 rescore_cfg
#    - 这是为标准COCO数据集设计的，可能不适合自定义数据集
#
# ============================================================================

