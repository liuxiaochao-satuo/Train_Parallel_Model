# Group_id=1（运动员）效果分析说明

## 分析脚本工作原理

### 1. `evaluate_by_group_id.py` 核心流程

#### 步骤1：加载标注并创建group_id映射
```python
# 从COCO标注文件中提取group_id信息
coco_gt = COCO(ann_file)
ann_id_to_group_id = {}
for ann_id, ann in coco_gt.anns.items():
    group_id = ann.get('group_id')
    if group_id is not None:
        ann_id_to_group_id[ann_id] = group_id
```

#### 步骤2：过滤指定group_id的数据
```python
# 获取group_id=1的所有annotation IDs（运动员）
target_ann_ids = [
    ann_id for ann_id in coco_gt.getAnnIds()
    if ann_id_to_group_id.get(ann_id) == 1
]

# 获取对应的image IDs
target_img_ids = set()
for ann_id in target_ann_ids:
    ann = coco_gt.anns[ann_id]
    target_img_ids.add(ann['image_id'])
```

#### 步骤3：创建过滤后的COCO对象
```python
filtered_gt = {
    'images': [img for img in coco_gt.dataset['images'] 
               if img['id'] in target_img_ids],
    'annotations': [ann for ann in coco_gt.dataset['annotations'] 
                   if ann['id'] in target_ann_ids],
    'categories': coco_gt.dataset['categories'],
}
```

#### 步骤4：过滤预测结果并评估
```python
# 只保留目标图像的预测
filtered_pred = [
    pred for pred in pred_data
    if pred['image_id'] in target_img_ids
]

# 使用COCOeval进行评估
coco_eval = COCOeval(filtered_coco_gt, filtered_coco_dt, 'keypoints')
coco_eval.evaluate()
coco_eval.accumulate()
```

#### 步骤5：重点展示group_id=1的结果
```python
# 重点对比group_id=1（运动员）
if 1 in group_ids:
    print("🎯 重点分析：group_id=1（运动员）")
    print(f"{'实验名称':<30s} {'AP':>10s} {'AP50':>10s} {'AP75':>10s} {'AR':>10s}")
    
    for exp_name in experiment_names:
        if exp_name in all_results and 1 in all_results[exp_name]:
            r = all_results[exp_name][1]  # group_id=1的结果
            print(f"{exp_name:<30s} {r['AP']:>10.4f} {r['AP50']:>10.4f} "
                  f"{r['AP75']:>10.4f} {r['AR']:>10.4f}")
    
    # 计算改进幅度
    print("\n相对于第一个实验的改进（group_id=1，运动员）:")
    for exp_name in experiment_names[1:]:
        ap = all_results[exp_name][1]['AP']
        improvement = ap - baseline_ap
        improvement_pct = (improvement / baseline_ap) * 100 if baseline_ap > 0 else 0
        arrow = "↑" if improvement > 0 else "↓"
        print(f"  {exp_name:30s} AP: {ap:.4f} "
              f"({arrow}{abs(improvement):.4f}, {arrow}{abs(improvement_pct):.2f}%)")
```

## 如何体现group_id=1（运动员）的效果

### 1. 单独计算AP和AR
- 脚本会**单独计算**group_id=1的AP、AP50、AP75、AR等指标
- 不包含其他group_id的数据，确保结果只反映运动员的识别精度

### 2. 与其他group_id对比
- 同时计算group_id=2、3、4的指标
- 可以对比不同group_id之间的性能差异

### 3. 实验间对比
- 对比三个实验（损失权重、加权采样、组合）在group_id=1上的表现
- 显示每个实验对运动员识别精度的提升幅度

### 4. 改进幅度计算
- 计算相对于baseline（第一个实验）的绝对提升和百分比提升
- 例如：`AP: 0.7456 (↑0.0222, ↑3.07%)` 表示AP提升了0.0222，相对提升3.07%

## 运行分析脚本

### 前提条件
1. 需要预测结果文件（`.keypoints.json`格式）
2. 预测文件通常在：`work_dir/predictions/results.keypoints.json`

### 生成预测文件
```bash
# 使用修复版本的test脚本
python tools/test_with_fix.py \
    configs/body_2d_keypoint/dekr/coco/dekr_hrnet-w32_parallel_ablation_loss_weight.py \
    work_dirs/ablation_experiments/loss_weight_only/best_coco_AP_epoch_130.pth \
    --work-dir work_dirs/ablation_experiments/loss_weight_only \
    --out work_dirs/ablation_experiments/loss_weight_only/predictions/results.keypoints.json \
    --launcher none
```

### 运行分析
```bash
python tools/evaluate_by_group_id.py \
    --ann-file /data/lxc/datasets/coco_paralel/annotations_id/person_keypoints_val_parallel.json \
    --work-dirs /data/lxc/outputs/train_parallel_model/ablation_experiments/loss_weight_only \
                /data/lxc/outputs/train_parallel_model/ablation_experiments/weighted_sampling_only \
                /data/lxc/outputs/train_parallel_model/ablation_experiments/combined \
    --experiment-names "Loss Weight Only" "Weighted Sampling Only" "Combined" \
    --group-ids 1 2 3 4
```

## 预期输出示例

```
🎯 重点分析：group_id=1（运动员）
================================================================================

实验名称                        AP        AP50       AP75         AR
--------------------------------------------------------------------------------
Loss Weight Only            0.7234    0.8567    0.7891    0.8123
Weighted Sampling Only      0.7312    0.8612    0.7956    0.8198
Combined                    0.7456    0.8723    0.8089    0.8345

相对于第一个实验的改进（group_id=1，运动员）:
  Weighted Sampling Only     AP: 0.7312 (↑0.0078, ↑1.08%)
  Combined                   AP: 0.7456 (↑0.0222, ↑3.07%)
```

## 验证集统计

根据当前验证集：
- **group_id=1（运动员）**: 815 个标注
- group_id=2: 815 个标注
- group_id=3: 47 个标注
- group_id=4: 4 个标注

## 关键指标说明

- **AP (Average Precision)**: 平均精度，主要指标
- **AP50**: OKS阈值0.5时的AP
- **AP75**: OKS阈值0.75时的AP
- **AR (Average Recall)**: 平均召回率

这些指标会**单独计算**group_id=1的数据，确保只反映运动员的识别效果。

