# -*- coding: utf-8 -*-
"""
Labelme转COCO格式 - 自底向上姿态估计专用版本

适用场景：
- 只标注了关键点（point），没有标注bounding box（rectangle）
- 使用group_id来标识不同的人
- 用于自底向上（bottom-up）姿态估计模型训练

与原始版本的区别：
1. 不需要rectangle，直接按group_id分组关键点
2. 从关键点自动计算bounding box
3. 从description字段读取可见性（而不是固定为2）
4. 不需要匹配polygon（分割掩码）
"""

import os
import json
import numpy as np
from collections import defaultdict

# ============================================================================
# 配置：关键点定义（21个关键点：COCO 17个 + 新增4个）
# ============================================================================

STANDARD_KEYPOINT_ORDER = [
    "nose",           # 0
    "left_eye",       # 1
    "right_eye",      # 2
    "left_ear",       # 3
    "right_ear",      # 4
    "left_shoulder",  # 5
    "right_shoulder", # 6
    "left_elbow",     # 7
    "right_elbow",    # 8
    "left_wrist",     # 9
    "right_wrist",    # 10
    "left_hip",       # 11
    "right_hip",      # 12
    "left_knee",      # 13
    "right_knee",     # 14
    "left_ankle",     # 15
    "right_ankle",    # 16
    "left_heel",      # 17 (新增)
    "right_heel",     # 18 (新增)
    "left_foot",      # 19 (新增)
    "right_foot",     # 20 (新增)
]

# ============================================================================
# 初始化COCO字典结构
# ============================================================================

coco = {}

# 定义类别信息
class_list = {
    'supercategory': 'person',
    'id': 1,
    'name': 'person',
    'keypoints': STANDARD_KEYPOINT_ORDER,  # 21个关键点
    'skeleton': [
        # COCO标准骨架连接（17个关键点的连接）
        [16, 14], [14, 12], [17, 15], [15, 13],  # 腿部
        [12, 13],  # 髋部
        [6, 12], [7, 13], [6, 7],  # 肩部到髋部
        [6, 8], [7, 9], [8, 10], [9, 11],  # 手臂
        [2, 3], [1, 2], [1, 3],  # 头部
        [2, 4], [3, 5], [4, 6], [5, 7],  # 头部到肩部
        # 新增关键点的连接（可根据需要调整）
        [15, 17], [16, 18], [17, 19], [18, 20]  # 脚部连接
    ]
}

coco['categories'] = [class_list]
coco['images'] = []
coco['annotations'] = []

IMG_ID = 0
ANN_ID = 0

# ============================================================================
# 核心转换函数：处理单个Labelme JSON文件
# ============================================================================

def calculate_bbox_from_keypoints(keypoints_dict):
    """
    从关键点坐标计算bounding box
    
    参数:
        keypoints_dict: {label: [x, y, visibility], ...}
    
    返回:
        bbox: [x, y, width, height] 或 None（如果没有可见的关键点）
    """
    visible_points = []
    
    # 收集所有可见的关键点（visibility > 0）
    for label, point_data in keypoints_dict.items():
        if len(point_data) >= 3 and point_data[2] > 0:  # 可见性 > 0
            visible_points.append([point_data[0], point_data[1]])
    
    if len(visible_points) == 0:
        return None
    
    # 计算边界框
    visible_points = np.array(visible_points)
    x_min = float(np.min(visible_points[:, 0]))
    y_min = float(np.min(visible_points[:, 1]))
    x_max = float(np.max(visible_points[:, 0]))
    y_max = float(np.max(visible_points[:, 1]))
    
    # 添加一些padding（可选，通常添加5-10%的边距）
    padding_ratio = 0.1  # 10%的padding
    width = x_max - x_min
    height = y_max - y_min
    padding_x = width * padding_ratio
    padding_y = height * padding_ratio
    
    x_min = max(0, x_min - padding_x)
    y_min = max(0, y_min - padding_y)
    x_max = x_max + padding_x
    y_max = y_max + padding_y
    
    return [x_min, y_min, x_max - x_min, y_max - y_min]


def process_single_json(labelme, image_id=1):
    """
    处理单个Labelme格式的JSON文件，转换为COCO格式的annotations
    
    参数:
        labelme: Labelme格式的字典数据
        image_id: 当前图像的ID
    
    返回:
        coco_annotations: COCO格式的标注列表
    """
    global ANN_ID
    
    coco_annotations = []
    
    # ========================================================================
    # 步骤1：按group_id分组关键点
    # ========================================================================
    # 在自底向上方法中，group_id标识不同的人
    group_keypoints = defaultdict(dict)  # {group_id: {label: [x, y, v], ...}}
    
    # 遍历所有标注，提取关键点
    for each_ann in labelme['shapes']:
        if each_ann['shape_type'] == 'point':  # 只处理关键点
            # 获取group_id（如果没有，默认为None）
            group_id = each_ann.get('group_id')
            if group_id is None:
                # 如果没有group_id，可以跳过或使用默认值
                print(f"警告：关键点 {each_ann.get('label')} 没有group_id，将被跳过")
                continue
            
            # 获取关键点信息
            label = each_ann['label']
            x = float(each_ann['points'][0][0])
            y = float(each_ann['points'][0][1])
            
            # 读取可见性（从description字段）
            # description: "0"=完全遮挡, "1"=遮挡可推测, "2"=清晰可见
            description = each_ann.get('description', '2')
            try:
                visibility = int(str(description).strip())
                if visibility not in [0, 1, 2]:
                    visibility = 2  # 默认值
            except:
                visibility = 2  # 默认值
            
            # 存储到对应group_id的字典中
            group_keypoints[group_id][label] = [x, y, visibility]
    
    # ========================================================================
    # 步骤2：为每个group_id生成一个annotation
    # ========================================================================
    for group_id, keypoints_dict in group_keypoints.items():
        
        # 初始化annotation字典
        bbox_dict = {}
        bbox_dict['category_id'] = 1
        bbox_dict['image_id'] = image_id
        bbox_dict['id'] = ANN_ID
        ANN_ID += 1
        
        # ====================================================================
        # 步骤3：从关键点计算bounding box
        # ====================================================================
        bbox = calculate_bbox_from_keypoints(keypoints_dict)
        if bbox is None:
            print(f"警告：group_id={group_id} 没有可见的关键点，跳过该annotation")
            continue
        
        bbox_dict['bbox'] = bbox
        bbox_dict['area'] = bbox[2] * bbox[3]  # width * height
        
        # ====================================================================
        # 步骤4：按顺序排列关键点
        # ====================================================================
        bbox_dict['keypoints'] = []
        num_keypoints = 0
        
        for each_class in STANDARD_KEYPOINT_ORDER:
            if each_class in keypoints_dict:
                # 关键点存在
                point_data = keypoints_dict[each_class]
                bbox_dict['keypoints'].append(point_data[0])  # x
                bbox_dict['keypoints'].append(point_data[1])  # y
                bbox_dict['keypoints'].append(point_data[2])  # visibility
                if point_data[2] > 0:  # 统计可见关键点数量
                    num_keypoints += 1
            else:
                # 关键点不存在
                bbox_dict['keypoints'].append(0)
                bbox_dict['keypoints'].append(0)
                bbox_dict['keypoints'].append(0)
        
        bbox_dict['num_keypoints'] = num_keypoints
        
        # ====================================================================
        # 步骤5：其他字段
        # ====================================================================
        bbox_dict['iscrowd'] = 0
        bbox_dict['segmentation'] = []  # 自底向上方法通常不需要分割掩码
        
        coco_annotations.append(bbox_dict)
    
    return coco_annotations


# ============================================================================
# 批量处理所有Labelme JSON文件
# ============================================================================

IMG_ID = 0
ANN_ID = 0

# 遍历当前目录下的所有文件
for labelme_json in os.listdir():
    
    if labelme_json.split('.')[-1] == 'json':
        
        # 跳过输出文件
        if 'coco' in labelme_json.lower() or labelme_json.endswith('.bak'):
            continue
        
        try:
            with open(labelme_json, 'r', encoding='utf-8') as f:
                labelme = json.load(f)
            
            # 提取图像元数据
            img_dict = {}
            img_dict['file_name'] = labelme.get('imagePath', labelme_json.replace('.json', '.jpg'))
            img_dict['height'] = labelme.get('imageHeight', 0)
            img_dict['width'] = labelme.get('imageWidth', 0)
            img_dict['id'] = IMG_ID
            coco['images'].append(img_dict)
            
            # 提取标注信息
            coco_annotations = process_single_json(labelme, image_id=IMG_ID)
            coco['annotations'] += coco_annotations
            
            IMG_ID += 1
            print(f'{labelme_json} 已处理完毕 (group_id数量: {len(coco_annotations)})')
            
        except Exception as e:
            print(f'错误：处理 {labelme_json} 时出错: {e}')
            continue

# ============================================================================
# 保存为COCO格式的JSON文件
# ============================================================================

if not os.path.exists('output_coco'):
    os.mkdir('output_coco')
    print('创建新目录 output_coco')

coco_path = 'output_coco/coco_bottomup.json'

with open(coco_path, 'w', encoding='utf-8') as f:
    json.dump(coco, f, indent=2, ensure_ascii=False)

print(f'\n转换完成！')
print(f'输出文件：{coco_path}')
print(f'总图像数：{len(coco["images"])}')
print(f'总标注数：{len(coco["annotations"])}')

# ============================================================================
# 验证（可选）
# ============================================================================

# 可以使用pycocotools验证COCO格式是否正确
# from pycocotools.coco import COCO
# my_coco = COCO(coco_path)
# print(f'验证通过！包含 {len(my_coco.imgs)} 张图像，{len(my_coco.anns)} 个标注')

