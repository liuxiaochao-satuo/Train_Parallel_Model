#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集划分脚本

将COCO格式的标注文件划分为训练集和验证集
"""

import json
import os
import shutil
import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple

def load_all_coco_files(coco_dir: str) -> Tuple[Dict, List[str]]:
    """加载目录下所有COCO JSON文件并合并"""
    coco_dir = Path(coco_dir)
    json_files = sorted(coco_dir.glob('*.json'))
    
    if not json_files:
        print(f"错误: 在 {coco_dir} 中未找到JSON文件")
        return None, []
    
    print(f"找到 {len(json_files)} 个COCO JSON文件，开始合并...")
    
    # 初始化合并后的COCO结构
    merged_coco = {
        'categories': None,
        'images': [],
        'annotations': []
    }
    
    image_id_map = {}  # {old_image_id: new_image_id}
    ann_id_counter = 0
    img_id_counter = 0
    file_list = []
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                coco_data = json.load(f)
            
            # 保存categories（应该都相同）
            if merged_coco['categories'] is None:
                merged_coco['categories'] = coco_data.get('categories', [])
            
            # 处理images
            for img in coco_data.get('images', []):
                old_img_id = img['id']
                new_img_id = img_id_counter
                image_id_map[(json_file.name, old_img_id)] = new_img_id
                
                img['id'] = new_img_id
                merged_coco['images'].append(img)
                img_id_counter += 1
            
            # 处理annotations
            for ann in coco_data.get('annotations', []):
                old_img_id = ann['image_id']
                new_img_id = image_id_map.get((json_file.name, old_img_id))
                if new_img_id is not None:
                    ann['image_id'] = new_img_id
                    ann['id'] = ann_id_counter
                    merged_coco['annotations'].append(ann)
                    ann_id_counter += 1
            
            file_list.append(json_file.name)
            print(f"  已处理: {json_file.name} (图像: {len(coco_data.get('images', []))}, 标注: {len(coco_data.get('annotations', []))})")
            
        except Exception as e:
            print(f"警告: 处理 {json_file.name} 时出错: {e}")
            continue
    
    print(f"\n合并完成:")
    print(f"  总图像数: {len(merged_coco['images'])}")
    print(f"  总标注数: {len(merged_coco['annotations'])}")
    
    return merged_coco, file_list

def split_dataset(coco_data: Dict, train_ratio: float = 0.8, shuffle: bool = True, seed: int = 42) -> Tuple[Dict, Dict]:
    """划分数据集为训练集和验证集"""
    images = coco_data['images']
    annotations = coco_data['annotations']
    categories = coco_data['categories']
    
    # 打乱图像顺序
    if shuffle:
        random.seed(seed)
        random.shuffle(images)
    
    # 计算划分点
    total_images = len(images)
    train_count = int(total_images * train_ratio)
    
    # 划分图像
    train_images = images[:train_count]
    val_images = images[train_count:]
    
    # 创建图像ID映射
    train_img_ids = {img['id'] for img in train_images}
    val_img_ids = {img['id'] for img in val_images}
    
    # 划分标注
    train_annotations = [ann for ann in annotations if ann['image_id'] in train_img_ids]
    val_annotations = [ann for ann in annotations if ann['image_id'] in val_img_ids]
    
    # 重新分配ID（从0开始）
    train_img_id_map = {old_id: new_id for new_id, old_id in enumerate([img['id'] for img in train_images])}
    val_img_id_map = {old_id: new_id for new_id, old_id in enumerate([img['id'] for img in val_images])}
    
    for img in train_images:
        img['id'] = train_img_id_map[img['id']]
    for img in val_images:
        img['id'] = val_img_id_map[img['id']]
    
    for ann in train_annotations:
        ann['image_id'] = train_img_id_map[ann['image_id']]
    for ann in val_annotations:
        ann['image_id'] = val_img_id_map[ann['image_id']]
    
    # 重新分配annotation ID
    for i, ann in enumerate(train_annotations):
        ann['id'] = i
    for i, ann in enumerate(val_annotations):
        ann['id'] = i
    
    # 构建训练集和验证集COCO格式
    train_coco = {
        'categories': categories,
        'images': train_images,
        'annotations': train_annotations
    }
    
    val_coco = {
        'categories': categories,
        'images': val_images,
        'annotations': val_annotations
    }
    
    return train_coco, val_coco

def copy_images(images: List[Dict], source_dir: str, target_dir: str):
    """复制图像文件到目标目录"""
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    
    copied = 0
    failed = 0
    
    for img in images:
        img_name = img['file_name']
        source_file = source_path / img_name
        target_file = target_path / img_name
        
        try:
            if source_file.exists():
                shutil.copy2(source_file, target_file)
                copied += 1
            else:
                print(f"警告: 图像文件不存在: {source_file}")
                failed += 1
        except Exception as e:
            print(f"警告: 复制 {img_name} 失败: {e}")
            failed += 1
    
    return copied, failed

def main():
    parser = argparse.ArgumentParser(description='划分COCO格式数据集为训练集和验证集')
    parser.add_argument('coco_dir', type=str, help='包含COCO JSON文件的目录')
    parser.add_argument('--image-dir', type=str, required=True,
                        help='图像文件目录')
    parser.add_argument('--output-dir', type=str, default='data/coco',
                        help='输出目录（默认: data/coco）')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help='训练集比例（默认: 0.8，即80%%训练，20%%验证）')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子（默认: 42）')
    parser.add_argument('--no-shuffle', action='store_true',
                        help='不打乱数据顺序')
    
    args = parser.parse_args()
    
    # 检查输入目录
    if not os.path.exists(args.coco_dir):
        print(f"错误: COCO目录不存在: {args.coco_dir}")
        return 1
    
    if not os.path.exists(args.image_dir):
        print(f"错误: 图像目录不存在: {args.image_dir}")
        return 1
    
    # 加载并合并所有COCO文件
    merged_coco, file_list = load_all_coco_files(args.coco_dir)
    if merged_coco is None:
        return 1
    
    # 划分数据集
    print(f"\n划分数据集 (训练集: {args.train_ratio*100:.1f}%, 验证集: {(1-args.train_ratio)*100:.1f}%)...")
    train_coco, val_coco = split_dataset(
        merged_coco,
        train_ratio=args.train_ratio,
        shuffle=not args.no_shuffle,
        seed=args.seed
    )
    
    print(f"\n划分结果:")
    print(f"  训练集: {len(train_coco['images'])} 张图像, {len(train_coco['annotations'])} 个标注")
    print(f"  验证集: {len(val_coco['images'])} 张图像, {len(val_coco['annotations'])} 个标注")
    
    # 创建输出目录
    output_path = Path(args.output_dir)
    annotations_dir = output_path / 'annotations'
    train_dir = output_path / 'train2017'
    val_dir = output_path / 'val2017'
    
    annotations_dir.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存标注文件
    train_ann_file = annotations_dir / 'person_keypoints_train2017.json'
    val_ann_file = annotations_dir / 'person_keypoints_val2017.json'
    
    print(f"\n保存标注文件...")
    with open(train_ann_file, 'w', encoding='utf-8') as f:
        json.dump(train_coco, f, indent=2, ensure_ascii=False)
    print(f"  训练集标注: {train_ann_file}")
    
    with open(val_ann_file, 'w', encoding='utf-8') as f:
        json.dump(val_coco, f, indent=2, ensure_ascii=False)
    print(f"  验证集标注: {val_ann_file}")
    
    # 复制图像文件
    print(f"\n复制图像文件...")
    train_copied, train_failed = copy_images(train_coco['images'], args.image_dir, train_dir)
    print(f"  训练集图像: 成功 {train_copied}, 失败 {train_failed}")
    
    val_copied, val_failed = copy_images(val_coco['images'], args.image_dir, val_dir)
    print(f"  验证集图像: 成功 {val_copied}, 失败 {val_failed}")
    
    print(f"\n✅ 数据集划分完成！")
    print(f"输出目录: {output_path}")
    print(f"\n数据目录结构:")
    print(f"  {output_path}/")
    print(f"    ├── annotations/")
    print(f"    │   ├── person_keypoints_train2017.json")
    print(f"    │   └── person_keypoints_val2017.json")
    print(f"    ├── train2017/")
    print(f"    │   └── (训练图像)")
    print(f"    └── val2017/")
    print(f"        └── (验证图像)")
    
    return 0

if __name__ == '__main__':
    exit(main())

