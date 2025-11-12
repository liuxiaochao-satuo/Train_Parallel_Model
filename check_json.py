#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Labelme标注文件检查工具
检查规则：
1. 每个group_id代表一个人，应该有21个关键点
2. 关键点必须按照标准顺序排列
3. description值必须是0、1或2（0=完全遮挡，1=遮挡可推测，2=清晰可见）
4. 每个group_id内不能有重复的关键点标签
5. 坐标应该在图像范围内
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# 标准关键点顺序（COCO 17个 + 新增4个）
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

EXPECTED_KEYPOINT_COUNT = len(STANDARD_KEYPOINT_ORDER)  # 21
VALID_VISIBILITY = ["0", "1", "2"]


class LabelmeChecker:
    """Labelme标注文件检查器"""
    
    def __init__(self, json_path: str):
        self.json_path = json_path
        self.errors = []
        self.warnings = []
        self.data = None
        
    def load_json(self) -> bool:
        """加载JSON文件"""
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            return True
        except json.JSONDecodeError as e:
            self.errors.append(f"JSON解析错误: {e}")
            return False
        except Exception as e:
            self.errors.append(f"文件读取错误: {e}")
            return False
    
    def check_structure(self):
        """检查JSON基本结构"""
        if not self.data:
            return
        
        required_keys = ["version", "shapes", "imagePath"]
        for key in required_keys:
            if key not in self.data:
                self.errors.append(f"缺少必需的字段: {key}")
        
        if "shapes" not in self.data:
            return
        
        if not isinstance(self.data["shapes"], list):
            self.errors.append("shapes字段必须是列表")
    
    def check_keypoint_format(self, shape: Dict):
        """检查单个关键点的格式"""
        required_fields = ["label", "points", "group_id", "description", "shape_type"]
        for field in required_fields:
            if field not in shape:
                self.errors.append(f"关键点缺少字段: {field}")
                return False
        
        if shape.get("shape_type") != "point":
            self.warnings.append(f"关键点 {shape.get('label')} 的shape_type不是'point'")
        
        points = shape.get("points", [])
        if not isinstance(points, list) or len(points) != 1:
            self.errors.append(f"关键点 {shape.get('label')} 的points格式错误，应该是包含一个点的列表")
            return False
        
        point = points[0]
        if not isinstance(point, list) or len(point) != 2:
            self.errors.append(f"关键点 {shape.get('label')} 的坐标格式错误")
            return False
        
        try:
            x, y = float(point[0]), float(point[1])
            # 检查坐标是否在图像范围内
            if "imageWidth" in self.data and "imageHeight" in self.data:
                width = self.data["imageWidth"]
                height = self.data["imageHeight"]
                if x < 0 or x > width or y < 0 or y > height:
                    self.warnings.append(
                        f"关键点 {shape.get('label')} (group_id={shape.get('group_id')}) "
                        f"坐标 ({x:.1f}, {y:.1f}) 超出图像范围 ({width}x{height})"
                    )
        except (ValueError, TypeError):
            self.errors.append(f"关键点 {shape.get('label')} 的坐标不是有效数字")
            return False
        
        # 检查可见性值
        visibility = str(shape.get("description", "")).strip()
        if visibility not in VALID_VISIBILITY:
            self.errors.append(
                f"关键点 {shape.get('label')} (group_id={shape.get('group_id')}) "
                f"的可见性值 '{visibility}' 无效，应该是 0、1 或 2"
            )
        
        return True
    
    def check_group_keypoints(self):
        """检查每个group_id的关键点"""
        if not self.data or "shapes" not in self.data:
            return
        
        # 按group_id分组
        groups: Dict[int, List[Dict]] = {}
        for shape in self.data["shapes"]:
            if not self.check_keypoint_format(shape):
                continue
            
            group_id = shape.get("group_id")
            if group_id is None:
                self.errors.append(f"关键点 {shape.get('label')} 缺少group_id")
                continue
            
            if group_id not in groups:
                groups[group_id] = []
            groups[group_id].append(shape)
        
        # 检查每个group
        for group_id, shapes in groups.items():
            # 检查关键点数量
            if len(shapes) != EXPECTED_KEYPOINT_COUNT:
                self.errors.append(
                    f"group_id={group_id} 的关键点数量错误: "
                    f"期望 {EXPECTED_KEYPOINT_COUNT} 个，实际 {len(shapes)} 个"
                )
            
            # 检查是否有重复标签
            labels = [s.get("label") for s in shapes]
            seen_labels = set()
            duplicates = []
            for label in labels:
                if label in seen_labels:
                    duplicates.append(label)
                seen_labels.add(label)
            
            if duplicates:
                self.errors.append(
                    f"group_id={group_id} 存在重复的关键点标签: {', '.join(set(duplicates))}"
                )
            
            # 检查关键点顺序和完整性
            label_to_shape = {s.get("label"): s for s in shapes}
            missing_keypoints = []
            out_of_order = []
            
            for idx, expected_label in enumerate(STANDARD_KEYPOINT_ORDER):
                if expected_label not in label_to_shape:
                    missing_keypoints.append(f"{expected_label} (索引 {idx})")
                else:
                    # 检查顺序（在原始列表中的位置）
                    actual_idx = labels.index(expected_label)
                    if actual_idx != idx:
                        out_of_order.append(
                            f"{expected_label}: 期望位置 {idx}, 实际位置 {actual_idx}"
                        )
            
            if missing_keypoints:
                self.errors.append(
                    f"group_id={group_id} 缺少关键点: {', '.join(missing_keypoints)}"
                )
            
            if out_of_order:
                self.warnings.append(
                    f"group_id={group_id} 关键点顺序不正确: {', '.join(out_of_order)}"
                )
            
            # 检查是否有未知的关键点标签
            unknown_labels = [label for label in labels if label not in STANDARD_KEYPOINT_ORDER]
            if unknown_labels:
                self.errors.append(
                    f"group_id={group_id} 包含未知的关键点标签: {', '.join(unknown_labels)}"
                )
    
    def check_all(self) -> Tuple[List[str], List[str]]:
        """执行所有检查"""
        if not self.load_json():
            return self.errors, self.warnings
        
        self.check_structure()
        self.check_group_keypoints()
        
        return self.errors, self.warnings
    
    def print_report(self):
        """打印检查报告"""
        errors, warnings = self.check_all()
        
        print(f"\n{'='*60}")
        print(f"检查文件: {os.path.basename(self.json_path)}")
        print(f"{'='*60}")
        
        if not errors and not warnings:
            print("✓ 检查通过！没有发现错误或警告。")
            return True
        
        if errors:
            print(f"\n❌ 发现 {len(errors)} 个错误:")
            for i, error in enumerate(errors, 1):
                print(f"  {i}. {error}")
        
        if warnings:
            print(f"\n⚠️  发现 {len(warnings)} 个警告:")
            for i, warning in enumerate(warnings, 1):
                print(f"  {i}. {warning}")
        
        print()
        return len(errors) == 0


def check_single_file(json_path: str) -> bool:
    """检查单个文件"""
    checker = LabelmeChecker(json_path)
    return checker.print_report()


def check_directory(directory: str, recursive: bool = False) -> Dict[str, bool]:
    """检查目录中的所有JSON文件"""
    directory = Path(directory)
    if not directory.exists():
        print(f"错误: 目录不存在: {directory}")
        return {}
    
    # 查找所有JSON文件
    pattern = "**/*.json" if recursive else "*.json"
    json_files = list(directory.glob(pattern))
    
    if not json_files:
        print(f"在目录 {directory} 中未找到JSON文件")
        return {}
    
    print(f"找到 {len(json_files)} 个JSON文件，开始检查...\n")
    
    results = {}
    total_errors = 0
    total_warnings = 0
    passed_files = 0
    
    for json_file in sorted(json_files):
        checker = LabelmeChecker(str(json_file))
        errors, warnings = checker.check_all()
        
        results[str(json_file)] = len(errors) == 0
        total_errors += len(errors)
        total_warnings += len(warnings)
        
        if len(errors) == 0:
            passed_files += 1
        
        # 只显示有问题的文件
        if errors or warnings:
            checker.print_report()
    
    # 打印总结
    print(f"\n{'='*60}")
    print("检查总结")
    print(f"{'='*60}")
    print(f"总文件数: {len(json_files)}")
    print(f"通过: {passed_files}")
    print(f"失败: {len(json_files) - passed_files}")
    print(f"总错误数: {total_errors}")
    print(f"总警告数: {total_warnings}")
    print()
    
    return results


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法:")
        print("  python check_json.py <json_file>           # 检查单个文件")
        print("  python check_json.py <directory>           # 检查目录中的所有JSON文件")
        print("  python check_json.py <directory> --recursive  # 递归检查目录")
        sys.exit(1)
    
    path = sys.argv[1]
    recursive = "--recursive" in sys.argv or "-r" in sys.argv
    
    if os.path.isfile(path):
        success = check_single_file(path)
        sys.exit(0 if success else 1)
    elif os.path.isdir(path):
        results = check_directory(path, recursive)
        # 如果有任何文件失败，返回非零退出码
        all_passed = all(results.values())
        sys.exit(0 if all_passed else 1)
    else:
        print(f"错误: 路径不存在: {path}")
        sys.exit(1)


if __name__ == "__main__":
    main()

