#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COCO格式验证脚本

用于验证转换后的COCO格式标注文件是否符合训练要求
"""

import json
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

try:
    from pycocotools.coco import COCO
    PYCOCOTOOLS_AVAILABLE = True
except ImportError:
    PYCOCOTOOLS_AVAILABLE = False
    print("警告: pycocotools未安装，部分验证功能将不可用")

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("警告: opencv-python未安装，可视化功能将不可用")


class COCOValidator:
    """COCO格式验证器"""
    
    def __init__(self, coco_file: str, image_dir: Optional[str] = None):
        self.coco_file = coco_file
        self.image_dir = image_dir
        self.errors = []
        self.warnings = []
        self.data = None
        self.stats = {}
        
    def load_json(self) -> bool:
        """加载JSON文件"""
        try:
            with open(self.coco_file, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            return True
        except json.JSONDecodeError as e:
            self.errors.append(f"JSON解析错误: {e}")
            return False
        except FileNotFoundError:
            self.errors.append(f"文件不存在: {self.coco_file}")
            return False
        except Exception as e:
            self.errors.append(f"文件读取错误: {e}")
            return False
    
    def check_structure(self):
        """检查基本结构"""
        if not self.data:
            return
        
        required_keys = ["categories", "images", "annotations"]
        for key in required_keys:
            if key not in self.data:
                self.errors.append(f"缺少必需的字段: {key}")
        
        # 检查数据类型
        if "images" in self.data and not isinstance(self.data["images"], list):
            self.errors.append("images字段必须是列表")
        if "annotations" in self.data and not isinstance(self.data["annotations"], list):
            self.errors.append("annotations字段必须是列表")
        if "categories" in self.data and not isinstance(self.data["categories"], list):
            self.errors.append("categories字段必须是列表")
    
    def check_categories(self):
        """检查类别定义"""
        if not self.data or "categories" not in self.data:
            return
        
        categories = self.data["categories"]
        if len(categories) == 0:
            self.errors.append("categories列表为空")
            return
        
        if len(categories) > 1:
            self.warnings.append(f"发现 {len(categories)} 个类别，通常姿态估计只需要1个类别（person）")
        
        for cat in categories:
            # 检查必需字段
            required_fields = ["id", "name", "keypoints"]
            for field in required_fields:
                if field not in cat:
                    self.errors.append(f"类别缺少必需字段: {field}")
            
            # 检查keypoints
            if "keypoints" in cat:
                keypoints = cat["keypoints"]
                if not isinstance(keypoints, list):
                    self.errors.append("keypoints必须是列表")
                elif len(keypoints) == 0:
                    self.errors.append("keypoints列表为空")
                else:
                    # 记录关键点数量
                    self.stats["num_keypoints"] = len(keypoints)
    
    def check_images(self):
        """检查图像信息"""
        if not self.data or "images" not in self.data:
            return
        
        images = self.data["images"]
        if len(images) == 0:
            self.errors.append("images列表为空")
            return
        
        self.stats["num_images"] = len(images)
        
        # 检查图像ID唯一性
        image_ids = []
        for img in images:
            required_fields = ["id", "file_name"]
            for field in required_fields:
                if field not in img:
                    self.errors.append(f"图像缺少必需字段: {field}")
            
            if "id" in img:
                img_id = img["id"]
                if img_id in image_ids:
                    self.errors.append(f"重复的图像ID: {img_id}")
                image_ids.append(img_id)
            
            # 检查图像文件是否存在
            if "file_name" in img and self.image_dir:
                img_path = os.path.join(self.image_dir, img["file_name"])
                if not os.path.exists(img_path):
                    self.warnings.append(f"图像文件不存在: {img['file_name']}")
    
    def check_annotations(self):
        """检查标注信息"""
        if not self.data or "annotations" not in self.data:
            return
        
        annotations = self.data["annotations"]
        if len(annotations) == 0:
            self.errors.append("annotations列表为空")
            return
        
        self.stats["num_annotations"] = len(annotations)
        
        # 检查必需字段
        required_fields = ["id", "image_id", "category_id", "keypoints", "num_keypoints"]
        keypoint_count_per_image = {}
        
        for ann in annotations:
            for field in required_fields:
                if field not in ann:
                    self.errors.append(f"标注缺少必需字段: {field}")
            
            # 检查keypoints格式
            if "keypoints" in ann:
                keypoints = ann["keypoints"]
                if not isinstance(keypoints, list):
                    self.errors.append("keypoints必须是列表")
                else:
                    # 检查keypoints格式（每3个数字一组）
                    if len(keypoints) % 3 != 0:
                        self.errors.append(
                            f"标注ID {ann.get('id')} 的keypoints长度不是3的倍数"
                        )
                    else:
                        # 检查keypoints长度（应该等于categories中定义的关键点数量 * 3）
                        expected_keypoint_count = self.stats.get("num_keypoints", 0)
                        if expected_keypoint_count > 0:
                            expected_len = expected_keypoint_count * 3
                            if len(keypoints) != expected_len:
                                self.errors.append(
                                    f"标注ID {ann.get('id')} 的keypoints长度不匹配: "
                                    f"期望 {expected_len}（{expected_keypoint_count}个关键点×3），实际 {len(keypoints)}"
                                )
                        
                        # 检查可见性值（应该是0, 1, 或2）
                        for i in range(2, len(keypoints), 3):
                            visibility = keypoints[i]
                            if visibility not in [0, 1, 2]:
                                self.warnings.append(
                                    f"标注ID {ann.get('id')} 的可见性值异常: {visibility}"
                                )
                        
                        # 验证num_keypoints是否正确（应该等于visibility > 0的关键点数量）
                        if "num_keypoints" in ann:
                            visible_count = sum(1 for i in range(2, len(keypoints), 3) if keypoints[i] > 0)
                            if ann["num_keypoints"] != visible_count:
                                self.warnings.append(
                                    f"标注ID {ann.get('id')} 的num_keypoints不匹配: "
                                    f"标注值 {ann['num_keypoints']}，实际可见关键点数 {visible_count}"
                                )
            
            # 检查bbox
            if "bbox" in ann:
                bbox = ann["bbox"]
                if not isinstance(bbox, list) or len(bbox) != 4:
                    self.errors.append(f"标注ID {ann.get('id')} 的bbox格式错误")
                else:
                    x, y, w, h = bbox
                    if w <= 0 or h <= 0:
                        self.warnings.append(
                            f"标注ID {ann.get('id')} 的bbox宽度或高度 <= 0"
                        )
            
            # 统计每个图像的人数
            if "image_id" in ann:
                img_id = ann["image_id"]
                if img_id not in keypoint_count_per_image:
                    keypoint_count_per_image[img_id] = 0
                keypoint_count_per_image[img_id] += 1
        
        # 统计信息
        if keypoint_count_per_image:
            self.stats["avg_persons_per_image"] = sum(keypoint_count_per_image.values()) / len(keypoint_count_per_image)
            self.stats["max_persons_per_image"] = max(keypoint_count_per_image.values())
            self.stats["min_persons_per_image"] = min(keypoint_count_per_image.values())
    
    def check_consistency(self):
        """检查数据一致性"""
        if not self.data:
            return
        
        # 检查image_id是否存在于images中
        if "images" in self.data and "annotations" in self.data:
            image_ids = {img["id"] for img in self.data["images"]}
            for ann in self.data["annotations"]:
                if "image_id" in ann:
                    if ann["image_id"] not in image_ids:
                        self.errors.append(
                            f"标注ID {ann.get('id')} 的image_id {ann['image_id']} 不存在于images中"
                        )
        
        # 检查category_id是否存在于categories中
        if "categories" in self.data and "annotations" in self.data:
            category_ids = {cat["id"] for cat in self.data["categories"]}
            for ann in self.data["annotations"]:
                if "category_id" in ann:
                    if ann["category_id"] not in category_ids:
                        self.errors.append(
                            f"标注ID {ann.get('id')} 的category_id {ann['category_id']} 不存在于categories中"
                        )
    
    def validate_with_pycocotools(self):
        """使用pycocotools验证"""
        if not PYCOCOTOOLS_AVAILABLE:
            return
        
        try:
            coco = COCO(self.coco_file)
            self.stats["pycocotools_images"] = len(coco.imgs)
            self.stats["pycocotools_annotations"] = len(coco.anns)
            self.stats["pycocotools_categories"] = len(coco.cats)
            return True
        except Exception as e:
            self.warnings.append(f"pycocotools验证失败: {e}")
            return False
    
    def validate_all(self) -> Tuple[List[str], List[str], Dict]:
        """执行所有验证"""
        if not self.load_json():
            return self.errors, self.warnings, self.stats
        
        self.check_structure()
        self.check_categories()
        self.check_images()
        self.check_annotations()
        self.check_consistency()
        
        # 使用pycocotools验证（如果可用）
        if PYCOCOTOOLS_AVAILABLE:
            self.validate_with_pycocotools()
        
        return self.errors, self.warnings, self.stats
    
    def validate(self):
        """执行验证，返回错误信息"""
        errors, warnings, stats = self.validate_all()
        return errors, len(errors) == 0


def visualize_coco_annotations(coco_file: str, image_dir: str, output_dir: str = "validation_vis", max_images: int = 10):
    """可视化COCO格式的标注"""
    if not CV2_AVAILABLE:
        print("错误: 需要安装opencv-python才能使用可视化功能")
        print("安装命令: pip install opencv-python")
        return False
    
    try:
        with open(coco_file, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)
    except Exception as e:
        print(f"错误: 无法读取COCO文件: {e}")
        return False
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    images = coco_data.get('images', [])
    annotations = coco_data.get('annotations', [])
    categories = coco_data.get('categories', [])
    
    if not categories:
        print("错误: 未找到categories定义")
        return False
    
    # 获取关键点定义和骨架连接
    keypoints = categories[0].get('keypoints', [])
    skeleton = categories[0].get('skeleton', [])
    
    # 按image_id组织annotations
    anns_by_image = {}
    for ann in annotations:
        img_id = ann.get('image_id')
        if img_id not in anns_by_image:
            anns_by_image[img_id] = []
        anns_by_image[img_id].append(ann)
    
    # 可视化前max_images张图像
    vis_count = 0
    for img_info in images[:max_images]:
        img_id = img_info.get('id')
        if img_id not in anns_by_image:
            continue
        
        img_name = img_info.get('file_name', '')
        img_path = os.path.join(image_dir, img_name) if image_dir else img_name
        
        if not os.path.exists(img_path):
            print(f"警告: 图像文件不存在: {img_path}")
            continue
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"警告: 无法读取图像: {img_path}")
            continue
        
        # 绘制标注
        for ann in anns_by_image[img_id]:
            # 绘制bbox
            bbox = ann.get('bbox', [])
            if len(bbox) == 4:
                x, y, w, h = map(int, bbox)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # 绘制关键点（按索引存储，以便正确绘制骨架）
            keypoints_list = ann.get('keypoints', [])
            kpt_coords = {}  # {keypoint_index: (x, y, visibility)}
            
            num_kpts = len(keypoints_list) // 3
            for i in range(num_kpts):
                idx = i * 3
                x, y, v = keypoints_list[idx], keypoints_list[idx+1], keypoints_list[idx+2]
                kpt_coords[i] = (int(x), int(y), v)
                if v > 0:  # 只绘制可见的关键点
                    color = (0, 0, 255) if v == 2 else (0, 165, 255)  # 红色=清晰可见，橙色=遮挡
                    cv2.circle(img, (int(x), int(y)), 5, color, -1)
            
            # 绘制骨架连接
            for connection in skeleton:
                if len(connection) == 2:
                    idx1, idx2 = connection[0], connection[1]
                    if idx1 in kpt_coords and idx2 in kpt_coords:
                        pt1 = kpt_coords[idx1]
                        pt2 = kpt_coords[idx2]
                        if pt1[2] > 0 and pt2[2] > 0:  # 两个关键点都可见
                            cv2.line(img, pt1[:2], pt2[:2], (255, 0, 0), 2)
        
        # 保存可视化结果
        output_file = output_path / f"vis_{Path(img_name).stem}.jpg"
        cv2.imwrite(str(output_file), img)
        print(f"已保存可视化结果: {output_file}")
        vis_count += 1
    
    print(f"\n可视化完成！共处理 {vis_count} 张图像")
    print(f"输出目录: {output_path}")
    return True


def find_coco_files(input_path):
    """查找COCO格式的JSON文件"""
    input_path = Path(input_path)
    coco_files = []
    
    if input_path.is_file():
        if input_path.suffix.lower() == '.json':
            coco_files.append(input_path)
    elif input_path.is_dir():
        coco_files = list(input_path.glob('*.json'))
        coco_files = sorted(coco_files)
    else:
        return []
    
    return coco_files


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='COCO格式验证工具')
    parser.add_argument('input', type=str, help='COCO格式的JSON标注文件或包含JSON文件的目录')
    parser.add_argument('--image-dir', type=str, default=None, 
                        help='图像文件目录（用于检查图像文件是否存在和可视化）')
    parser.add_argument('--visualize', action='store_true', 
                        help='启用可视化功能（需要opencv-python）')
    parser.add_argument('--output-dir', type=str, default='validation_vis',
                        help='可视化结果输出目录（默认: validation_vis）')
    parser.add_argument('--max-images', type=int, default=10,
                        help='最多可视化的图像数量（默认: 10）')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"错误: 路径不存在: {args.input}")
        sys.exit(1)
    
    # 查找所有COCO文件
    coco_files = find_coco_files(args.input)
    if not coco_files:
        print(f"错误: 未找到JSON文件: {args.input}")
        sys.exit(1)
    
    # 批量验证
    all_errors = []  # [(file_path, errors), ...]
    success_count = 0
    error_count = 0
    
    for coco_file in coco_files:
        validator = COCOValidator(str(coco_file), args.image_dir)
        errors, success = validator.validate()
        
        if success:
            success_count += 1
        else:
            error_count += 1
            all_errors.append((coco_file, errors))
        
        # 执行可视化（如果启用）
        if args.visualize:
            if not args.image_dir:
                print("警告: 可视化需要指定 --image-dir 参数")
            else:
                visualize_coco_annotations(
                    str(coco_file), 
                    args.image_dir, 
                    args.output_dir,
                    args.max_images
                )
    
    # 输出统计结果
    print(f"\n验证完成:")
    print(f"  总文件数: {len(coco_files)}")
    print(f"  正确: {success_count}")
    print(f"  错误: {error_count}")
    
    # 只输出错误信息
    if all_errors:
        print(f"\n错误详情:")
        for file_path, errors in all_errors:
            print(f"\n  {file_path.name}:")
            for error in errors:
                print(f"    - {error}")
    
    sys.exit(0 if error_count == 0 else 1)


if __name__ == "__main__":
    main()

