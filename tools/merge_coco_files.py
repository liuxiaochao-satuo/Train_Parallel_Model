# -*- coding: utf-8 -*-
"""
合并多个COCO格式的标注文件为一个完整的训练/验证集
保留group_id信息
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict


def merge_coco_files(coco_files, output_file, split_ratio=0.8):
    """合并多个COCO文件为一个完整的COCO数据集。
    
    Args:
        coco_files: COCO文件路径列表
        output_file: 输出文件路径
        split_ratio: 训练集比例（如果输出训练集）
    """
    merged_coco = {
        'info': {
            'description': 'Merged COCO Parallel Dataset with group_id',
            'version': '1.0',
        },
        'licenses': [],
        'categories': None,
        'images': [],
        'annotations': []
    }
    
    img_id_counter = 0
    ann_id_counter = 0
    
    # 统计信息
    group_id_stats = defaultdict(int)
    
    for coco_file in coco_files:
        with open(coco_file, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)
        
        # 设置categories（只需要设置一次）
        if merged_coco['categories'] is None and 'categories' in coco_data:
            merged_coco['categories'] = coco_data['categories']
        
        # 处理images
        for img in coco_data.get('images', []):
            old_img_id = img['id']
            img['id'] = img_id_counter
            merged_coco['images'].append(img)
            
            # 处理该图像的所有annotations
            for ann in coco_data.get('annotations', []):
                if ann.get('image_id') == old_img_id:
                    ann_copy = ann.copy()
                    ann_copy['image_id'] = img_id_counter
                    ann_copy['id'] = ann_id_counter
                    ann_id_counter += 1
                    
                    # 统计group_id
                    if 'group_id' in ann_copy:
                        group_id_stats[ann_copy['group_id']] += 1
                    
                    merged_coco['annotations'].append(ann_copy)
            
            img_id_counter += 1
    
    # 保存合并后的文件
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_coco, f, indent=2, ensure_ascii=False)
    
    # 打印统计信息
    print(f'\n合并完成！')
    print(f'输出文件: {output_path}')
    print(f'总图像数: {len(merged_coco["images"])}')
    print(f'总标注数: {len(merged_coco["annotations"])}')
    print(f'\nGroup ID统计:')
    for group_id, count in sorted(group_id_stats.items()):
        print(f'  group_id={group_id}: {count} 个标注')
    
    return merged_coco


def split_dataset(coco_data, train_ratio=0.8, shuffle=True, seed=42):
    """将数据集分割为训练集和验证集。
    
    Args:
        coco_data: COCO格式的数据字典
        train_ratio: 训练集比例
        shuffle: 是否随机打乱
        seed: 随机种子
    """
    import random
    random.seed(seed)
    
    images = coco_data['images']
    if shuffle:
        random.shuffle(images)
    
    split_idx = int(len(images) * train_ratio)
    train_images = images[:split_idx]
    val_images = images[split_idx:]
    
    train_img_ids = {img['id'] for img in train_images}
    val_img_ids = {img['id'] for img in val_images}
    
    train_annotations = [ann for ann in coco_data['annotations'] 
                        if ann['image_id'] in train_img_ids]
    val_annotations = [ann for ann in coco_data['annotations'] 
                      if ann['image_id'] in val_img_ids]
    
    train_coco = {
        'info': coco_data.get('info', {}),
        'licenses': coco_data.get('licenses', []),
        'categories': coco_data['categories'],
        'images': train_images,
        'annotations': train_annotations
    }
    
    val_coco = {
        'info': coco_data.get('info', {}),
        'licenses': coco_data.get('licenses', []),
        'categories': coco_data['categories'],
        'images': val_images,
        'annotations': val_annotations
    }
    
    return train_coco, val_coco


def main():
    parser = argparse.ArgumentParser(
        description='合并多个COCO标注文件并可选地分割为训练/验证集')
    parser.add_argument(
        'input_dir',
        type=str,
        help='包含COCO JSON文件的目录')
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='输出目录（默认：与输入目录相同）')
    parser.add_argument(
        '--split',
        action='store_true',
        help='是否分割为训练集和验证集')
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.8,
        help='训练集比例（默认：0.8）')
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子（默认：42）')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f'错误：输入目录不存在: {input_dir}')
        return
    
    # 查找所有COCO JSON文件
    # 排除已经合并的训练集和验证集文件
    coco_files = list(input_dir.glob('*.json'))
    coco_files = [f for f in coco_files 
                  if 'train' not in f.name.lower() 
                  and 'val' not in f.name.lower()
                  and 'merged' not in f.name.lower()]
    
    if not coco_files:
        print(f'错误：在 {input_dir} 中未找到COCO JSON文件')
        return
    
    print(f'找到 {len(coco_files)} 个COCO文件')
    
    # 确定输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = input_dir
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 合并所有文件
    merged_file = output_dir / 'person_keypoints_merged.json'
    merged_coco = merge_coco_files(coco_files, merged_file)
    
    # 如果需要分割
    if args.split:
        train_coco, val_coco = split_dataset(
            merged_coco, 
            train_ratio=args.train_ratio,
            shuffle=True,
            seed=args.seed
        )
        
        train_file = output_dir / 'person_keypoints_train_parallel.json'
        val_file = output_dir / 'person_keypoints_val_parallel.json'
        
        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump(train_coco, f, indent=2, ensure_ascii=False)
        
        with open(val_file, 'w', encoding='utf-8') as f:
            json.dump(val_coco, f, indent=2, ensure_ascii=False)
        
        print(f'\n已分割数据集:')
        print(f'  训练集: {train_file} ({len(train_coco["images"])} 图像, {len(train_coco["annotations"])} 标注)')
        print(f'  验证集: {val_file} ({len(val_coco["images"])} 图像, {len(val_coco["annotations"])} 标注)')


if __name__ == '__main__':
    main()

