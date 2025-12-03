#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEKR模型训练脚本

用于训练DEKR自底向上姿态估计模型
"""

import argparse
import os
import sys
from pathlib import Path

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练DEKR模型')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='配置文件路径'
    )
    parser.add_argument(
        '--work-dir',
        type=str,
        default='work_dirs/dekr_custom',
        help='工作目录（用于保存训练输出）'
    )
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='使用的GPU数量'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='从checkpoint恢复训练'
    )
    parser.add_argument(
        '--load-from',
        type=str,
        default=None,
        help='加载预训练模型'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='随机种子'
    )
    return parser.parse_args()


def check_environment():
    """检查训练环境"""
    print("=" * 60)
    print("检查训练环境...")
    print("=" * 60)
    
    # 检查MMPose是否安装
    try:
        import mmpose
        print(f"✅ MMPose已安装: {mmpose.__version__}")
    except ImportError:
        print("❌ MMPose未安装，请先安装MMPose")
        print("   安装命令: cd /path/to/mmpose && pip install -e .")
        return False
    
    # 检查PyTorch和CUDA
    try:
        import torch
        print(f"✅ PyTorch已安装: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"✅ CUDA可用: {torch.version.cuda}")
            print(f"   可用GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("⚠️  CUDA不可用，将使用CPU训练（速度会很慢）")
    except ImportError:
        print("❌ PyTorch未安装")
        return False
    
    # 检查pycocotools
    try:
        import pycocotools
        print("✅ pycocotools已安装")
    except ImportError:
        print("⚠️  pycocotools未安装，某些功能可能不可用")
        print("   安装命令: pip install pycocotools")
    
    print()
    return True


def check_config_file(config_path):
    """检查配置文件"""
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        return False
    
    print(f"✅ 配置文件存在: {config_path}")
    
    # 尝试加载配置文件
    try:
        from mmengine import Config
        cfg = Config.fromfile(config_path)
        print(f"✅ 配置文件格式正确")
        print(f"   模型类型: {cfg.model.type}")
        print(f"   数据模式: {cfg.get('data_mode', 'N/A')}")
        return True
    except Exception as e:
        print(f"❌ 配置文件加载失败: {e}")
        return False


def main():
    """主函数"""
    args = parse_args()
    
    print("=" * 60)
    print("DEKR模型训练")
    print("=" * 60)
    print()
    
    # 检查环境
    if not check_environment():
        sys.exit(1)
    
    # 检查配置文件
    if not check_config_file(args.config):
        sys.exit(1)
    
    # 创建工作目录
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    print(f"✅ 工作目录: {work_dir}")
    print()
    
    # 构建训练命令
    if args.gpus > 1:
        # 多GPU训练
        cmd = [
            'bash', 'tools/dist_train.sh',
            args.config,
            str(args.gpus),
            '--work-dir', str(work_dir)
        ]
        if args.resume:
            cmd.extend(['--resume', args.resume])
        if args.load_from:
            cmd.extend(['--load-from', args.load_from])
    else:
        # 单GPU训练
        cmd = [
            'python', 'tools/train.py',
            args.config,
            '--work-dir', str(work_dir)
        ]
        if args.resume:
            cmd.extend(['--resume', args.resume])
        if args.load_from:
            cmd.extend(['--load-from', args.load_from])
        if args.seed:
            cmd.extend(['--seed', str(args.seed)])
    
    print("=" * 60)
    print("开始训练...")
    print("=" * 60)
    print(f"执行命令: {' '.join(cmd)}")
    print()
    
    # 检查是否在mmpose目录中
    if not os.path.exists('tools/train.py'):
        print("⚠️  警告: 未找到 tools/train.py")
        print("   请确保在MMPose根目录下运行此脚本，或者")
        print("   使用绝对路径指向MMPose的tools/train.py")
        print()
        print("   例如:")
        print("   cd /path/to/mmpose")
        print("   python /path/to/Train_Parallel_Model/train_dekr.py --config ...")
        print()
        
        # 尝试查找mmpose
        mmpose_paths = [
            '/home/satuo/mmpose',
            os.path.expanduser('~/mmpose'),
            '/opt/mmpose',
        ]
        
        found = False
        for mmpose_path in mmpose_paths:
            if os.path.exists(os.path.join(mmpose_path, 'tools/train.py')):
                print(f"   找到MMPose: {mmpose_path}")
                print(f"   建议切换到该目录: cd {mmpose_path}")
                found = True
                break
        
        if not found:
            print("   未找到MMPose安装目录，请手动指定")
            sys.exit(1)
    
    # 执行训练命令
    import subprocess
    try:
        result = subprocess.run(cmd, check=False)
        if result.returncode == 0:
            print()
            print("=" * 60)
            print("✅ 训练完成！")
            print("=" * 60)
            print(f"模型保存在: {work_dir}")
            print(f"最佳模型: {work_dir / 'best.pth'}")
            print(f"最新模型: {work_dir / 'latest.pth'}")
        else:
            print()
            print("=" * 60)
            print("❌ 训练失败")
            print("=" * 60)
            sys.exit(result.returncode)
    except KeyboardInterrupt:
        print()
        print("训练被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"训练过程出错: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

