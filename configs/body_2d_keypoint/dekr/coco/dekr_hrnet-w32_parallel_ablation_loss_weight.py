_base_ = ['dekr_hrnet-w32_parallel_rtx4500.py']

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

# 更新数据路径
data_root = 'data/coco_parallel'

# 更新训练pipeline，添加ApplyGroupWeight
# codec 从 _base_ 继承
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='BottomupRandomAffine', input_size=codec['input_size']),  # noqa: F821
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='GenerateTarget', encoder=codec),  # noqa: F821
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
    batch_size=4,
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
        pipeline=val_pipeline,  # noqa: F821
    ))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + '/annotations_id/person_keypoints_val_parallel.json',
    nms_mode='none',
    score_mode='keypoint',
    # 注意：预测结果会自动保存到 work_dir/predictions/results.keypoints.json
    # 无需手动设置outfile_prefix，除非需要自定义路径
)

test_evaluator = val_evaluator

