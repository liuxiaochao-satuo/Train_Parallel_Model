_base_ = ['dekr_hrnet-w32_parallel_rtx4500.py']

# ============================================================================
# 消融实验配置2：仅使用加权采样（Weighted Sampling Only）
# ============================================================================
# 
# 实验设置：
# - 使用WeightedGroupSampler增加group_id=1的采样频率
# - 不使用损失权重（group_id_weight为空）
# - 不在pipeline中使用ApplyGroupWeight transform
# 
# 注意：codec 和 val_pipeline 从 _base_ 配置继承
# ============================================================================

# 更新数据路径
data_root = 'data/coco_parallel'

# 训练pipeline（不使用ApplyGroupWeight）
# codec 从 _base_ 继承
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='BottomupRandomAffine', input_size=codec['input_size']),  # noqa: F821
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='GenerateTarget', encoder=codec),  # noqa: F821
    dict(type='BottomupGetHeatmapMask'),
    # 不使用ApplyGroupWeight
    dict(type='PackPoseInputs'),
]

# ============================================================================
# 数据加载器配置 - 仅使用加权采样
# ============================================================================
train_dataloader = dict(
    batch_size=12,
    num_workers=8,
    persistent_workers=True,
    # 使用WeightedGroupSampler进行加权采样
    sampler=dict(
        type='WeightedGroupSampler',
        group_id_weights={1: 2.0},  # group_id=1的采样频率为2倍
        replacement=True,  # 允许重复采样
    ),
    dataset=dict(
        type='CocoParallelDataset',
        data_root=data_root,
        data_mode='bottomup',
        ann_file='annotations_id/person_keypoints_train_parallel.json',
        data_prefix=dict(img='images/'),
        # 不使用损失权重
        group_id_weight={},  # 空字典，不使用损失权重
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
        group_id_weight={},
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

