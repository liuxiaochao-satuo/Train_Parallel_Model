#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
临时修复版本的test.py，解决PyTorch weights_only问题
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
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

# 替换torch.load
torch.load = patched_torch_load

# 同时修复torch.serialization模块
import torch.serialization
_original_load = torch.serialization.load
def patched_serialization_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.serialization.load = patched_serialization_load

# 现在导入并运行原始的test.py
from tools.test import main

if __name__ == '__main__':
    main()

