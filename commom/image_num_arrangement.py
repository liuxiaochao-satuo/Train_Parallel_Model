import os
import sys
from pathlib import Path


def list_images_sorted(dir_path):
    """
    列出目录下的图片文件（仅 .jpg/.jpeg），按文件名排序。
    返回绝对路径列表。
    """
    if not os.path.isdir(dir_path):
        return []
    exts = {".jpg", ".jpeg"}
    files = []
    for name in os.listdir(dir_path):
        lower = name.lower()
        suffix = os.path.splitext(lower)[1]
        if suffix in exts:
            files.append(os.path.join(dir_path, name))
    files.sort()
    return files


def two_phase_rename(file_paths, start_index):
    """
    两阶段重命名：
    1) 先改为临时无冲突名称（同目录 .tmp_ 前缀）
    2) 再改为最终 12 位序号 jpg 名称

    返回最终使用到的最后一个索引（包含）。
    """
    # 第一阶段：改为临时名，避免与目标名冲突
    temp_paths = []
    for i, src in enumerate(file_paths):
        src_dir = os.path.dirname(src)
        base = os.path.basename(src)
        temp = os.path.join(src_dir, f".tmp_ren_{i:06d}_{base}")
        # 如已存在同名临时文件，递增尝试
        j = 0
        tmp_candidate = temp
        while os.path.exists(tmp_candidate):
            j += 1
            tmp_candidate = os.path.join(src_dir, f".tmp_ren_{i:06d}_{j:03d}_{base}")
        os.rename(src, tmp_candidate)
        temp_paths.append(tmp_candidate)

    # 第二阶段：改为最终名 12 位序号 .jpg
    index = start_index
    for temp in temp_paths:
        dst_dir = os.path.dirname(temp)
        final_name = f"{index:012d}.jpg"
        final_path = os.path.join(dst_dir, final_name)
        # 如目标已存在，继续递增直到找到空位
        while os.path.exists(final_path):
            index += 1
            final_name = f"{index:012d}.jpg"
            final_path = os.path.join(dst_dir, final_name)
        os.rename(temp, final_path)
        index += 1

    return index - 1


def rename_dataset(root_dir):
    """
    按顺序重命名：
    1) 前摆下 -> 从 000000000000.jpg 开始
    2) 后摆下 -> 延续前摆下最后一个序号 + 1
    3) 全套后摆下 -> 延续后摆下最后一个序号 + 1
    仅处理 .jpg/.jpeg 文件，重命名为 .jpg 扩展名。
    """
    order = ["前摆下", "后摆下", "全套后摆下"]

    current_index = 0
    for folder in order:
        folder_path = os.path.join(root_dir, folder)
        images = list_images_sorted(folder_path)
        if not images:
            # 跳过空目录或不存在的目录
            continue
        last_index = two_phase_rename(images, current_index)
        current_index = last_index + 1


def main():
    if len(sys.argv) < 2:
        print("用法: python image_num_arrangement.py <姿态估计数据集目录>")
        print("例子: python image_num_arrangement.py /path/to/姿态估计数据集")
        sys.exit(1)

    root_dir = sys.argv[1]
    if not os.path.isdir(root_dir):
        print("错误：目录不存在:", root_dir)
        sys.exit(1)

    # 显示即将处理的路径，避免误操作
    print("将处理目录:", root_dir)
    print("子目录顺序: 前摆下 -> 后摆下 -> 全套后摆下")

    rename_dataset(root_dir)

    print("完成：重命名已按顺序执行。")


if __name__ == "__main__":
    main()


