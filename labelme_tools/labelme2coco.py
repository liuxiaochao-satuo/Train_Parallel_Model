# -*- coding: utf-8 -*-
"""
Labelme转COCO格式 - 自底向上姿态估计专用版本
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

# 固定输出路径（可在此修改）
OUTPUT_COCO_FILE = '/home/satuo/code/Train_Parallel_Model/data/coco_annotations_id'

STANDARD_KEYPOINT_ORDER = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_heel", "right_heel", "left_foot", "right_foot",
]

def init_coco_dict():
    coco = {}
    class_list = {
        'supercategory': 'person',
        'id': 1,
        'name': 'person',
        'keypoints': STANDARD_KEYPOINT_ORDER,
        'skeleton': [
            [1, 2], [0, 1], [0, 2], [1, 3], [2, 4],
            [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
            [5, 11], [6, 12], [11, 12],
            [11, 13], [13, 15],
            [12, 14], [14, 16],
            [16, 18], [16, 20], [20, 18],
            [15, 17], [15, 19], [19, 17],
        ]
    }
    coco['categories'] = [class_list]
    coco['images'] = []
    coco['annotations'] = []
    return coco

def calculate_bbox_from_keypoints(keypoints_dict):
    visible_points = []
    for label, point_data in keypoints_dict.items():
        if len(point_data) >= 3 and point_data[2] > 0:
            visible_points.append([point_data[0], point_data[1]])
    
    if len(visible_points) == 0:
        return None
    
    visible_points = np.array(visible_points)
    x_min = float(np.min(visible_points[:, 0]))
    y_min = float(np.min(visible_points[:, 1]))
    x_max = float(np.max(visible_points[:, 0]))
    y_max = float(np.max(visible_points[:, 1]))
    
    padding_ratio = 0.1
    width = x_max - x_min
    height = y_max - y_min
    padding_x = width * padding_ratio
    padding_y = height * padding_ratio
    
    x_min = max(0, x_min - padding_x)
    y_min = max(0, y_min - padding_y)
    x_max = x_max + padding_x
    y_max = y_max + padding_y
    
    return [x_min, y_min, x_max - x_min, y_max - y_min]

def process_single_json(labelme, image_id, ann_id_start):
    coco_annotations = []
    ann_id = ann_id_start
    group_keypoints = defaultdict(dict)
    
    for each_ann in labelme['shapes']:
        if each_ann['shape_type'] == 'point':
            group_id = each_ann.get('group_id')
            if group_id is None:
                print(f"警告：关键点 {each_ann.get('label')} 没有group_id，将被跳过")
                continue
            
            label = each_ann['label']
            x = float(each_ann['points'][0][0])
            y = float(each_ann['points'][0][1])
            
            description = each_ann.get('description', '2')
            try:
                visibility = int(str(description).strip())
                if visibility not in [0, 1, 2]:
                    visibility = 2
            except:
                visibility = 2
            
            group_keypoints[group_id][label] = [x, y, visibility]
    
    for group_id, keypoints_dict in group_keypoints.items():
        bbox_dict = {}
        bbox_dict['category_id'] = 1
        bbox_dict['image_id'] = image_id
        bbox_dict['id'] = ann_id
        ann_id += 1
        
        bbox = calculate_bbox_from_keypoints(keypoints_dict)
        if bbox is None:
            print(f"警告：group_id={group_id} 没有可见的关键点，跳过该annotation")
            continue
        
        bbox_dict['bbox'] = bbox
        bbox_dict['area'] = bbox[2] * bbox[3]
        
        bbox_dict['keypoints'] = []
        num_keypoints = 0
        
        for each_class in STANDARD_KEYPOINT_ORDER:
            if each_class in keypoints_dict:
                point_data = keypoints_dict[each_class]
                bbox_dict['keypoints'].append(point_data[0])
                bbox_dict['keypoints'].append(point_data[1])
                bbox_dict['keypoints'].append(point_data[2])
                if point_data[2] > 0:
                    num_keypoints += 1
            else:
                bbox_dict['keypoints'].append(0)
                bbox_dict['keypoints'].append(0)
                bbox_dict['keypoints'].append(0)
        
        bbox_dict['num_keypoints'] = num_keypoints
        bbox_dict['iscrowd'] = 0
        bbox_dict['segmentation'] = []
        # 保留group_id信息到COCO标注中
        bbox_dict['group_id'] = group_id
        
        coco_annotations.append(bbox_dict)
    
    return coco_annotations, ann_id

def process_json_file(json_path, output_dir):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            labelme = json.load(f)
        
        coco = init_coco_dict()
        
        img_dict = {}
        img_dict['file_name'] = labelme.get('imagePath', Path(json_path).stem + '.jpg')
        img_dict['height'] = labelme.get('imageHeight', 0)
        img_dict['width'] = labelme.get('imageWidth', 0)
        img_dict['id'] = 0
        coco['images'].append(img_dict)
        
        coco_annotations, _ = process_single_json(labelme, 0, 0)
        coco['annotations'].extend(coco_annotations)
        
        output_path = output_dir / Path(json_path).name
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(coco, f, indent=2, ensure_ascii=False)
        
        print(f'{Path(json_path).name} -> {output_path.name} (group_id数量: {len(coco_annotations)})')
        return True
    except Exception as e:
        print(f'错误：处理 {json_path} 时出错: {e}')
        return False

def find_json_files(input_path):
    json_files = []
    input_path = Path(input_path)
    
    if input_path.is_file():
        if input_path.suffix.lower() == '.json':
            json_files.append(input_path)
    elif input_path.is_dir():
        json_files = list(input_path.glob('*.json'))
        json_files = [f for f in json_files if 'coco' not in f.name.lower() and not f.name.endswith('.bak')]
    else:
        print(f'错误：路径不存在: {input_path}')
        sys.exit(1)
    
    return sorted(json_files)

def main():
    parser = argparse.ArgumentParser(description='Labelme转COCO格式')
    parser.add_argument('input', type=str, help='输入：JSON文件路径或包含JSON文件的文件夹路径')
    args = parser.parse_args()
    
    json_files = find_json_files(args.input)
    if not json_files:
        print(f'错误：未找到JSON文件')
        sys.exit(1)
    
    output_dir = Path(OUTPUT_COCO_FILE)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f'找到 {len(json_files)} 个JSON文件，开始转换...')
    print(f'输出目录：{output_dir}\n')
    
    success_count = 0
    for json_file in json_files:
        if process_json_file(json_file, output_dir):
            success_count += 1
    
    print(f'\n转换完成！')
    print(f'成功转换：{success_count}/{len(json_files)} 个文件')
    print(f'输出目录：{output_dir}')

if __name__ == '__main__':
    main()
