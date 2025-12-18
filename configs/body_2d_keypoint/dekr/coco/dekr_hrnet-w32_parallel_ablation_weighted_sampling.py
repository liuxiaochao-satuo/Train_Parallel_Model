_base_ = ['dekr_hrnet-w32_parallel_rtx4500.py']

# 确保CocoParallelDataset被导入和注册
custom_imports = dict(
    imports=['mmpose.datasets.datasets.body.coco_parallel_dataset'],
    allow_failed_imports=False)

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

# 训练pipeline（不使用ApplyGroupWeight）
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='BottomupRandomAffine', input_size=codec['input_size']),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='GenerateTarget', encoder=codec),
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
    # 注意：WeightedGroupSampler不接受shuffle参数，只接受group_id_weights和replacement
    sampler=dict(
        type='WeightedGroupSampler',
        group_id_weights={1: 2.0},  # group_id=1的采样频率为2倍
        replacement=True,  # 允许重复采样
        # 不包含shuffle参数，因为WeightedGroupSampler不支持
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
        group_id_weight={},
        test_mode=True,
        pipeline=val_pipeline,
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

