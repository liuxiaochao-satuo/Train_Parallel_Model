#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Labelme JSON 排序处理工具

功能：
- 对目录中的所有 Labelme JSON 文件的 `shapes` 进行排序
- 排序规则：先按 `group_id` 从小到大；再按预定义的骨骼关键点顺序
- 原地写回，默认为每个文件创建 .bak 备份

用法：
  python process_labelmejson.py <directory>            # 递归处理目录
  python process_labelmejson.py <directory> --no-rec   # 仅当前目录
  python process_labelmejson.py <directory> --no-bak   # 不生成备份
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# 与 check_json.py 中一致的关键点顺序（COCO 17个 + 新增4个）
STANDARD_KEYPOINT_ORDER: List[str] = [
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

LABEL_TO_INDEX: Dict[str, int] = {name: idx for idx, name in enumerate(STANDARD_KEYPOINT_ORDER)}
UNKNOWN_LABEL_INDEX = 10_000  # 未知标签的排序放到该 group 的末尾


def safe_int(value, default=None):
    try:
        return int(value)
    except Exception:
        return default


def sort_shapes(shapes: List[Dict]) -> List[Dict]:
    """
    严格排序：
    1) 将 shapes 按 group_id 分组；group_id 能转 int 的按数值分组，无法转换或缺失的分到特殊组 (None)
    2) 组间按 group_id 升序，None 组最后
    3) 组内按 STANDARD_KEYPOINT_ORDER 顺序排列；未知标签按字母序追加在末尾
    """
    # 分组
    group_to_shapes: Dict[int, List[Dict]] = {}
    none_group: List[Dict] = []
    for shape in shapes:
        gid_raw = shape.get("group_id")
        gid = safe_int(gid_raw, None)
        if gid is None:
            none_group.append(shape)
        else:
            group_to_shapes.setdefault(gid, []).append(shape)

    # 组间顺序
    ordered_group_ids = sorted(group_to_shapes.keys())

    # 组内排序：标准关键点优先，未知按字母序
    def sort_group(group_shapes: List[Dict]) -> List[Dict]:
        # 稳定地先按是否在标准列表，再按标准索引/标签名称
        def inner_key(s: Dict) -> Tuple[int, int, str]:
            label = str(s.get("label", ""))
            in_std = 0 if label in LABEL_TO_INDEX else 1
            idx = LABEL_TO_INDEX.get(label, UNKNOWN_LABEL_INDEX)
            return (in_std, idx, label)
        return sorted(group_shapes, key=inner_key)

    result: List[Dict] = []
    for gid in ordered_group_ids:
        result.extend(sort_group(group_to_shapes[gid]))
    if none_group:
        # 无 group_id 的放最后，并做同样的组内排序
        result.extend(sort_group(none_group))
    return result


def process_file(json_path: Path, make_backup: bool = True) -> bool:
    """
    处理单个 JSON 文件：排序 shapes 并写回。
    返回值：是否发生了变更（写回）。
    """
    try:
        with json.loads if False else open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"跳过（读取失败）: {json_path.name} - {e}")
        return False

    if not isinstance(data, dict) or "shapes" not in data or not isinstance(data["shapes"], list):
        print(f"跳过（缺少 shapes）: {json_path.name}")
        return False

    original = data["shapes"]
    sorted_shapes = sort_shapes(original)

    # 变更检测：比较 (group_id,label) 序列是否发生变化
    def seq(shapes_list: List[Dict]) -> List[Tuple[str, str]]:
        def gid_to_str(g):
            v = safe_int(g, None)
            return "" if v is None else str(v)
        return [(gid_to_str(s.get("group_id")), str(s.get("label", ""))) for s in shapes_list]
    if seq(original) == seq(sorted_shapes):
        return False

    data["shapes"] = sorted_shapes

    # 备份
    if make_backup:
        backup_path = json_path.with_suffix(json_path.suffix + ".bak")
        try:
            original_text = json_path.read_text(encoding="utf-8")
            backup_path.write_text(original_text, encoding="utf-8")
        except Exception as e:
            print(f"警告：备份失败 {backup_path.name} - {e}")

    # 写回
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"写回失败: {json_path.name} - {e}")
        return False


def process_directory(directory: Path, recursive: bool = True, make_backup: bool = True) -> Tuple[int, int]:
    """
    处理目录中的 JSON 文件
    返回：(处理文件总数, 发生变更的文件数)
    """
    pattern = "**/*.json" if recursive else "*.json"
    json_files = list(directory.glob(pattern))
    if not json_files:
        print(f"在目录 {directory} 中未找到 JSON 文件")
        return (0, 0)

    changed = 0
    for json_file in sorted(json_files):
        did_change = process_file(json_file, make_backup=make_backup)
        if did_change:
            changed += 1
            print(f"已排序: {json_file.name}")
    return (len(json_files), changed)


def main():
    if len(sys.argv) < 2:
        print("用法:")
        print("  python process_labelmejson.py <directory>            # 递归处理目录")
        print("  python process_labelmejson.py <directory> --no-rec   # 仅当前目录")
        print("  python process_labelmejson.py <directory> --no-bak   # 不生成备份")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists() or not path.is_dir():
        print(f"错误：目录不存在或不是目录：{path}")
        sys.exit(1)

    recursive = "--no-rec" not in sys.argv
    make_backup = "--no-bak" not in sys.argv

    total, changed = process_directory(path, recursive=recursive, make_backup=make_backup)
    print()
    print("=" * 60)
    print(f"处理完成：共发现 {total} 个 JSON 文件，其中 {changed} 个文件发生排序变更。")


if __name__ == "__main__":
    main()


