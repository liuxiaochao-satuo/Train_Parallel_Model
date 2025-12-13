# -*- coding: utf-8 -*-
"""
使用加权采样（Weighted Sampling）的配置示例
"""

# 方案2：使用加权采样替代损失权重
train_dataloader = dict(
    batch_size=12,
    num_workers=8,
    persistent_workers=True,
    # 使用WeightedGroupSampler替代DefaultSampler
    sampler=dict(
        type='WeightedGroupSampler',
        group_id_weights={1: 2.0},  # group_id=1的采样频率为2倍
        replacement=True,  # 允许重复采样
    ),
    dataset=dict(
        type='CocoParallelDataset',
        data_root='data/coco_parallel',
        data_mode='bottomup',
        ann_file='annotations_id/person_keypoints_train_parallel.json',
        data_prefix=dict(img='images/'),
        # 不需要group_id_weight，因为使用采样而不是损失权重
        pipeline=train_pipeline,
    ))

# 方案A：结合使用损失权重和加权采样（推荐）
train_dataloader = dict(
    batch_size=12,
    num_workers=8,
    persistent_workers=True,
    # 使用加权采样
    sampler=dict(
        type='WeightedGroupSampler',
        group_id_weights={1: 1.5},  # 采样频率1.5倍
        replacement=True,
    ),
    dataset=dict(
        type='CocoParallelDataset',
        data_root='data/coco_parallel',
        data_mode='bottomup',
        ann_file='annotations_id/person_keypoints_train_parallel.json',
        data_prefix=dict(img='images/'),
        # 同时使用损失权重
        group_id_weight={1: 1.5},  # 损失权重1.5倍
        pipeline=[
            dict(type='LoadImage'),
            dict(type='BottomupRandomAffine', input_size=codec['input_size']),
            dict(type='RandomFlip', direction='horizontal'),
            dict(type='GenerateTarget', encoder=codec),
            dict(type='BottomupGetHeatmapMask'),
            dict(type='ApplyGroupWeight'),  # 应用损失权重
            dict(type='PackPoseInputs'),
        ],
    ))

