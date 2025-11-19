# -*- coding: utf-8 -*-
"""
Labelme转COCO格式转换脚本 - 详细注释版

本脚本将Labelme格式的标注文件转换为COCO格式
支持：目标检测 + 关键点检测 + 图像分割

转换原理：
1. 以rectangle（矩形框）为中心，每个rectangle生成一个COCO annotation
2. 通过空间位置匹配，将polygon和point关联到对应的rectangle
3. 关键点按照预定义的顺序排列
"""

import os
import json
import numpy as np

# ============================================================================
# 第一部分：初始化COCO字典结构
# ============================================================================

# 创建空的COCO格式字典
coco = {}

# 定义类别信息
# categories是COCO格式中最重要的部分，定义了：
# - 类别名称和ID
# - 关键点的名称和顺序（这个顺序决定了后续keypoints数组的排列）
# - skeleton（骨架连接，用于可视化）
class_list = {
    'supercategory': 'sjb_rect',  # 父类别
    'id': 1,                      # 类别ID
    'name': 'sjb_rect',           # 类别名称
    'keypoints': ['angle_30', 'angle_60', 'angle_90'],  # 关键点名称列表（顺序很重要！）
    'skeleton': [[0,1], [0,2], [1,2]]  # 骨架连接，表示哪些关键点之间有连接
}

# 初始化COCO字典的各个部分
coco['categories'] = []  # 类别列表
coco['categories'].append(class_list)  # 添加类别定义

coco['images'] = []      # 图像信息列表
coco['annotations'] = []  # 标注信息列表

# 全局ID计数器
IMG_ID = 0  # 图像ID，每处理一张图像递增
ANN_ID = 0  # 标注ID，每处理一个annotation递增

# ============================================================================
# 第二部分：核心转换函数
# ============================================================================

def process_single_json(labelme, image_id=1):
    '''
    处理单个Labelme格式的JSON文件，转换为COCO格式的annotations
    
    参数:
        labelme: Labelme格式的字典数据
        image_id: 当前图像的ID
    
    返回:
        coco_annotations: COCO格式的标注列表
    '''
    global ANN_ID  # 使用全局标注ID计数器
    
    coco_annotations = []  # 存储当前图像的所有annotations
    
    # ========================================================================
    # 步骤1：遍历所有标注，找到矩形框（rectangle）
    # ========================================================================
    # 在Labelme中，rectangle代表一个目标对象（如一个人、一个物体）
    # 每个rectangle会生成一个COCO格式的annotation
    for each_ann in labelme['shapes']:  # 遍历该json文件中的所有标注
        
        if each_ann['shape_type'] == 'rectangle':  # 筛选出矩形框
            
            # ================================================================
            # 步骤2：初始化annotation字典
            # ================================================================
            bbox_dict = {}
            bbox_dict['category_id'] = 1  # 类别ID（这里固定为1）
            bbox_dict['segmentation'] = []  # 分割掩码（先初始化为空）
            bbox_dict['iscrowd'] = 0  # 是否拥挤（0=不拥挤，1=拥挤）
            bbox_dict['image_id'] = image_id  # 对应的图像ID
            bbox_dict['id'] = ANN_ID  # 当前annotation的唯一ID
            ANN_ID += 1  # 递增标注ID，确保每个annotation的ID唯一
            
            # ================================================================
            # 步骤3：计算边界框（bbox）
            # ================================================================
            # Labelme的rectangle可能不是标准格式（左上-右下），
            # 需要取min/max确保得到正确的左上角和右下角坐标
            
            # 获取矩形框的两个对角点
            point1 = each_ann['points'][0]  # 第一个点 [x1, y1]
            point2 = each_ann['points'][1]   # 第二个点 [x2, y2]
            
            # 计算左上角坐标（取x和y的最小值）
            bbox_left_top_x = min(int(point1[0]), int(point2[0]))
            bbox_left_top_y = min(int(point1[1]), int(point2[1]))
            
            # 计算右下角坐标（取x和y的最大值）
            bbox_right_bottom_x = max(int(point1[0]), int(point2[0]))
            bbox_right_bottom_y = max(int(point1[1]), int(point2[1]))
            
            # 计算宽度和高度
            bbox_w = bbox_right_bottom_x - bbox_left_top_x
            bbox_h = bbox_right_bottom_y - bbox_left_top_y
            
            # COCO格式的bbox：左上角x、左上角y、宽度、高度
            bbox_dict['bbox'] = [bbox_left_top_x, bbox_left_top_y, bbox_w, bbox_h]
            
            # 计算面积
            bbox_dict['area'] = bbox_w * bbox_h
            
            # ================================================================
            # 步骤4：匹配分割多边形（polygon）
            # ================================================================
            # 在Labelme中，polygon用于标注分割掩码
            # 需要找到属于当前rectangle的polygon
            
            # 重新遍历所有标注，寻找polygon
            for each_ann in labelme['shapes']:  # 遍历所有标注
                
                if each_ann['shape_type'] == 'polygon':  # 筛选出分割多边形
                    
                    # 获取polygon的第一个点的坐标
                    # 通过判断第一个点是否在rectangle内部来匹配
                    first_x = each_ann['points'][0][0]
                    first_y = each_ann['points'][0][1]
                    
                    # 空间匹配判断：polygon的第一个点是否在当前rectangle内部？
                    # 注意：这里使用的是简单的边界框判断，可能不够精确
                    if (first_x > bbox_left_top_x) & \
                       (first_x < bbox_right_bottom_x) & \
                       (first_y < bbox_right_bottom_y) & \
                       (first_y > bbox_left_top_y):
                        
                        # 匹配成功！提取segmentation
                        # Labelme格式：[[x1,y1], [x2,y2], [x3,y3], ...]
                        # COCO格式：[[x1, y1, x2, y2, x3, y3, ...]]（一维数组）
                        
                        # 使用lambda函数将坐标保留两位小数，并展平为一维数组
                        bbox_dict['segmentation'] = list(map(
                            lambda x: list(map(lambda y: round(y, 2), x)), 
                            each_ann['points']
                        ))
                        # 上面的代码等价于：
                        # segmentation = []
                        # for point in each_ann['points']:
                        #     segmentation.append(round(point[0], 2))
                        #     segmentation.append(round(point[1], 2))
                        # bbox_dict['segmentation'] = [segmentation]
            
            # ================================================================
            # 步骤5：匹配关键点（point）
            # ================================================================
            # 在Labelme中，point用于标注关键点
            # 需要找到属于当前rectangle的所有关键点
            
            bbox_keypoints_dict = {}  # 存储匹配到的关键点 {label: [x, y]}
            
            # 重新遍历所有标注，寻找关键点
            for each_ann in labelme['shapes']:  # 遍历所有标注
                
                if each_ann['shape_type'] == 'point':  # 筛选出关键点标注
                    
                    # 获取关键点的坐标和标签
                    x = int(each_ann['points'][0][0])  # 关键点x坐标
                    y = int(each_ann['points'][0][1])  # 关键点y坐标
                    label = each_ann['label']  # 关键点的标签（如'angle_30'）
                    
                    # 空间匹配判断：关键点坐标是否在当前rectangle内部？
                    if (x > bbox_left_top_x) & \
                       (x < bbox_right_bottom_x) & \
                       (y < bbox_right_bottom_y) & \
                       (y > bbox_left_top_y):
                        
                        # 匹配成功！存储到字典中
                        # 使用label作为key，方便后续按顺序排列
                        bbox_keypoints_dict[label] = [x, y]
            
            # 统计关键点数量
            bbox_dict['num_keypoints'] = len(bbox_keypoints_dict)
            
            # ================================================================
            # 步骤6：按类别顺序排列关键点
            # ================================================================
            # 这是最关键的一步！
            # COCO格式要求关键点必须按照categories中定义的顺序排列
            # 格式：[x1, y1, v1, x2, y2, v2, ...]
            # 每3个数字一组：x坐标、y坐标、可见性标志
            
            bbox_dict['keypoints'] = []  # 初始化关键点列表
            
            # 按照categories中定义的keypoints顺序遍历
            for each_class in class_list['keypoints']:  # ['angle_30', 'angle_60', 'angle_90']
                
                if each_class in bbox_keypoints_dict:
                    # 如果该关键点存在，添加坐标和可见性
                    bbox_dict['keypoints'].append(bbox_keypoints_dict[each_class][0])  # x坐标
                    bbox_dict['keypoints'].append(bbox_keypoints_dict[each_class][1])  # y坐标
                    bbox_dict['keypoints'].append(2)  # 可见性标志
                    # 2 = 可见且不遮挡
                    # 1 = 遮挡但可推测
                    # 0 = 不存在或完全不可见
                else:
                    # 如果该关键点不存在，填充0
                    bbox_dict['keypoints'].append(0)  # x坐标 = 0
                    bbox_dict['keypoints'].append(0)  # y坐标 = 0
                    bbox_dict['keypoints'].append(0)  # 可见性 = 0（不存在）
            
            # ================================================================
            # 步骤7：添加到annotations列表
            # ================================================================
            coco_annotations.append(bbox_dict)
    
    return coco_annotations  # 返回当前图像的所有annotations

# ============================================================================
# 第三部分：批量处理所有Labelme JSON文件
# ============================================================================

# 重置ID计数器
IMG_ID = 0
ANN_ID = 0

# 遍历当前目录下的所有文件
for labelme_json in os.listdir():
    
    # 只处理JSON文件
    if labelme_json.split('.')[-1] == 'json':
        
        # 读取Labelme格式的JSON文件
        with open(labelme_json, 'r', encoding='utf-8') as f:
            labelme = json.load(f)
        
        # ====================================================================
        # 提取图像元数据
        # ====================================================================
        img_dict = {}
        img_dict['file_name'] = labelme['imagePath']  # 图像文件名
        img_dict['height'] = labelme['imageHeight']    # 图像高度
        img_dict['width'] = labelme['imageWidth']      # 图像宽度
        img_dict['id'] = IMG_ID                        # 图像ID
        coco['images'].append(img_dict)                 # 添加到images列表
        
        # ====================================================================
        # 提取框和关键点信息
        # ====================================================================
        # 调用核心转换函数，处理当前图像的标注
        coco_annotations = process_single_json(labelme, image_id=IMG_ID)
        
        # 将当前图像的annotations添加到总列表中
        coco['annotations'] += coco_annotations
        
        # 递增图像ID
        IMG_ID += 1
        
        print(labelme_json, '已处理完毕')
    
    else:
        pass  # 跳过非JSON文件

# ============================================================================
# 第四部分：保存为COCO格式的JSON文件
# ============================================================================

# 创建输出目录
if not os.path.exists('output_coco'):
    os.mkdir('output_coco')
    print('创建新目录 output_coco')

# 保存COCO格式的JSON文件
coco_path = 'output_coco/coco_sample.json'

with open(coco_path, 'w') as f:
    json.dump(coco, f, indent=2)  # indent=2 使JSON格式更易读

print(f'转换完成！输出文件：{coco_path}')

# ============================================================================
# 第五部分：验证（可选）
# ============================================================================

# 可以使用pycocotools验证COCO格式是否正确
# from pycocotools.coco import COCO
# my_coco = COCO(coco_path)
# print(f'加载成功！包含 {len(my_coco.imgs)} 张图像，{len(my_coco.anns)} 个标注')

