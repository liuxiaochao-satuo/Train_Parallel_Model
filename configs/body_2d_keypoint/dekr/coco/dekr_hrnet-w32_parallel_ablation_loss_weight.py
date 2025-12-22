_base_ = ['dekr_hrnet-w32_parallel_rtx4500.py']

# 确保CocoParallelDataset被导入和注册
custom_imports = dict(
    imports=['mmpose.datasets.datasets.body.coco_parallel_dataset'],
    allow_failed_imports=False)

# ============================================================================
# 消融实验配置1：仅使用损失权重（Loss Weighting Only）
# ============================================================================
# 
# 实验设置：
# - 使用group_id_weight={1: 2.0}增加group_id=1的损失权重
# - 不使用加权采样（使用默认的DefaultSampler）
# - 在pipeline中使用ApplyGroupWeight transform
# 
# 注意：codec 和 val_pipeline 从 _base_ 配置继承
# ============================================================================

# 更新数据路径 - 使用data盘的绝对路径
# 实际数据集路径：/data/lxc/datasets/coco_paralel
data_root = '/data/lxc/datasets/coco_paralel'

# Codec配置 - 从_base_继承，需要显式定义以便在pipeline中使用
codec = dict(
    type='SPR',
    input_size=(512, 512),
    heatmap_size=(128, 128),
    sigma=(4, 2),
    minimal_diagonal_length=32**0.5,
    generate_keypoint_heatmaps=True,
    decode_max_instances=30)

# 验证pipeline - 从_base_继承
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

# 更新训练pipeline，添加ApplyGroupWeight
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='BottomupRandomAffine', input_size=codec['input_size']),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='BottomupGetHeatmapMask'),
    dict(type='ApplyGroupWeight'),  # 应用group_id损失权重
    dict(type='PackPoseInputs'),
]

# ============================================================================
# 数据加载器配置 - 仅使用损失权重
# ============================================================================
train_dataloader = dict(
    batch_size=12,
    num_workers=8,
    persistent_workers=True,
    # 使用默认的DefaultSampler（不使用加权采样）
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CocoParallelDataset',
        data_root=data_root,
        data_mode='bottomup',
        ann_file='annotations_id/person_keypoints_train_parallel.json',
        data_prefix=dict(img='images/'),
        # 仅使用损失权重，不使用加权采样
        group_id_weight={1: 2.0},  # group_id=1的损失权重为2倍
        pipeline=train_pipeline,
    ))

val_dataloader = dict(
    batch_size=1,  # DEKRHead只支持batch_size=1的预测
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type='CocoParallelDataset',
        data_root=data_root,
        data_mode='bottomup',
        ann_file='annotations_id/person_keypoints_val_parallel.json',
        data_prefix=dict(img='images/'),
        group_id_weight={},  # 验证时不使用权重
        test_mode=True,
        pipeline=val_pipeline,
    ))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + '/annotations_id/person_keypoints_val_parallel.json',
    nms_mode='none',
    score_mode='keypoint',
    # 注意：outfile_prefix会在运行时由Runner自动解析work_dir
    # 如果不设置，keypoints会保存到临时目录并在评估后清理
    # 为了进行group_id分析，需要在运行时设置outfile_prefix
)

test_evaluator = val_evaluator

