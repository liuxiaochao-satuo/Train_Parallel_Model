#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修复版本的test.py，解决PyTorch weights_only问题
用于运行baseline预测
"""
import sys
import os

# 在导入任何其他模块之前，先修复torch.load
import torch

# 保存原始的torch.load
_original_torch_load = torch.load

# 创建修复版本的torch.load
def patched_torch_load(*args, **kwargs):
    """修复版本的torch.load，强制使用weights_only=False"""
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

# 替换torch.load
torch.load = patched_torch_load

# 现在导入并运行原始的test.py
if __name__ == '__main__':
    # 导入test模块
    from tools import test
    # 运行main函数
    test.main()

