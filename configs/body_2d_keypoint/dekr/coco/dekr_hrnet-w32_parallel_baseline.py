_base_ = ['dekr_hrnet-w32_parallel_rtx4500.py']

# 确保CocoParallelDataset被导入和注册
custom_imports = dict(
    imports=['mmpose.datasets.datasets.body.coco_parallel_dataset'],
    allow_failed_imports=False)

# ============================================================================
# Baseline配置：不使用任何消融实验方法
# ============================================================================
# 
# 实验设置：
# - 不使用损失权重（group_id_weight为空）
# - 不使用加权采样（使用默认的DefaultSampler）
# - 不在pipeline中使用ApplyGroupWeight transform
# - 使用预训练模型（HRNet backbone的预训练权重）
# 
# 注意：这是真正的baseline，用于对比消融实验的效果
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
    dict(type='PackPoseInputs'),
]

# ============================================================================
# 数据加载器配置 - Baseline（不使用任何消融方法）
# ============================================================================
train_dataloader = dict(
    batch_size=2,  # 使用与基础配置相同的batch_size，避免OOM
    num_workers=4,  # 使用与基础配置相同的num_workers
    persistent_workers=True,
    # 使用默认的DefaultSampler（不使用加权采样）
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CocoParallelDataset',
        data_root=data_root,
        data_mode='bottomup',
        ann_file='annotations_id/person_keypoints_train_parallel.json',
        data_prefix=dict(img='images/'),
        # Baseline：不使用损失权重
        group_id_weight={},  # 空字典，不使用任何权重
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

