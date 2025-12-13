# -*- coding: utf-8 -*-
"""
从原始labelme标注文件生成annotation_id到group_id的映射文件
用于在训练时针对特定group_id的样本进行精度提升
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict


def extract_group_id_mapping_from_labelme(labelme_file):
    """从单个labelme文件中提取group_id映射"""
    with open(labelme_file, 'r', encoding='utf-8') as f:
        labelme = json.load(f)
    
    # 提取group_id信息
    group_keypoints = defaultdict(dict)
    
    for each_ann in labelme.get('shapes', []):
        if each_ann.get('shape_type') == 'point':
            group_id = each_ann.get('group_id')
            if group_id is not None:
                label = each_ann.get('label')
                group_keypoints[group_id][label] = True
    
    # 返回所有group_id的列表
    return list(group_keypoints.keys())


def generate_mapping_from_coco(coco_file, labelme_dir=None):
    """从COCO标注文件生成映射（如果COCO文件中没有group_id，需要从labelme文件恢复）"""
    with open(coco_file, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    # 如果COCO文件中已经有group_id信息（在annotations的额外字段中）
    mapping = {}
    for ann in coco_data.get('annotations', []):
        ann_id = ann['id']
        # 检查是否有group_id字段
        if 'group_id' in ann:
            mapping[ann_id] = ann['group_id']
        else:
            # 如果没有，尝试从labelme文件恢复
            # 这需要根据文件名匹配
            mapping[ann_id] = None  # 标记为未知
    
    return mapping


def generate_mapping_from_labelme_dir(labelme_dir, coco_ann_file):
    """从labelme目录和COCO标注文件生成完整的映射"""
    labelme_dir = Path(labelme_dir)
    coco_ann_file = Path(coco_ann_file)
    
    # 读取COCO文件获取annotation IDs
    with open(coco_ann_file, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    # 创建image_id到file_name的映射
    img_id_to_file = {}
    for img in coco_data.get('images', []):
        img_id_to_file[img['id']] = img['file_name']
    
    # 创建annotation_id到group_id的映射
    ann_id_to_group_id = {}
    
    # 从每个labelme文件中提取group_id
    labelme_files = list(labelme_dir.glob('*.json'))
    labelme_files = [f for f in labelme_files if 'coco' not in f.name.lower()]
    
    # 创建file_name到labelme文件的映射
    file_to_labelme = {}
    for labelme_file in labelme_files:
        with open(labelme_file, 'r', encoding='utf-8') as f:
            labelme = json.load(f)
        file_name = labelme.get('imagePath', labelme_file.stem + '.jpg')
        # 标准化文件名（去除路径）
        file_name = Path(file_name).name
        file_to_labelme[file_name] = labelme_file
    
    # 处理每个annotation
    group_counter = {}  # 用于跟踪每个图像中的group_id计数
    for ann in coco_data.get('annotations', []):
        ann_id = ann['id']
        img_id = ann['image_id']
        
        if img_id in img_id_to_file:
            file_name = img_id_to_file[img_id]
            if file_name in file_to_labelme:
                labelme_file = file_to_labelme[file_name]
                
                # 从labelme文件中提取group_id
                with open(labelme_file, 'r', encoding='utf-8') as f:
                    labelme = json.load(f)
                
                # 根据annotation的顺序确定group_id
                # 这里假设COCO annotations的顺序与labelme中group_id的顺序一致
                if img_id not in group_counter:
                    group_counter[img_id] = 0
                
                # 从labelme中提取所有group_id
                group_keypoints = defaultdict(dict)
                for shape in labelme.get('shapes', []):
                    if shape.get('shape_type') == 'point':
                        group_id = shape.get('group_id')
                        if group_id is not None:
                            label = shape.get('label')
                            group_keypoints[group_id][label] = True
                
                # 按group_id排序
                sorted_groups = sorted(group_keypoints.keys())
                
                # 根据当前计数获取group_id
                if group_counter[img_id] < len(sorted_groups):
                    group_id = sorted_groups[group_counter[img_id]]
                    ann_id_to_group_id[ann_id] = group_id
                    group_counter[img_id] += 1
                else:
                    ann_id_to_group_id[ann_id] = None
    
    return ann_id_to_group_id


def main():
    parser = argparse.ArgumentParser(
        description='生成annotation_id到group_id的映射文件')
    parser.add_argument(
        '--coco-ann-file',
        type=str,
        required=True,
        help='COCO标注文件路径')
    parser.add_argument(
        '--labelme-dir',
        type=str,
        default=None,
        help='原始labelme标注文件目录（可选，如果COCO文件中没有group_id）')
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='输出映射文件路径（默认：与COCO文件同目录）')
    
    args = parser.parse_args()
    
    coco_ann_file = Path(args.coco_ann_file)
    if not coco_ann_file.exists():
        print(f'错误：COCO标注文件不存在: {coco_ann_file}')
        return
    
    # 生成映射
    if args.labelme_dir:
        print(f'从labelme目录生成映射: {args.labelme_dir}')
        mapping = generate_mapping_from_labelme_dir(args.labelme_dir, coco_ann_file)
    else:
        print('从COCO文件生成映射（如果COCO文件中包含group_id）')
        mapping = generate_mapping_from_coco(coco_ann_file)
    
    # 保存映射文件
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = coco_ann_file.parent / f'{coco_ann_file.stem}_group_id_mapping.json'
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    
    # 统计信息
    total = len(mapping)
    group_1_count = sum(1 for gid in mapping.values() if gid == 1)
    unknown_count = sum(1 for gid in mapping.values() if gid is None)
    
    print(f'\n映射文件已生成: {output_file}')
    print(f'总annotation数: {total}')
    print(f'group_id=1的数量: {group_1_count}')
    print(f'未知group_id的数量: {unknown_count}')
    print(f'group_id=1占比: {group_1_count/total*100:.2f}%' if total > 0 else '')


if __name__ == '__main__':
    main()

