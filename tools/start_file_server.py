#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
启动简单的 HTTP 文件服务器，方便从本地浏览器访问和下载远程文件
使用方法: python tools/start_file_server.py [端口] [目录]
"""

import os
import sys
import http.server
import socketserver
from pathlib import Path

# 默认配置
DEFAULT_PORT = 8000
DEFAULT_DIR = "/data/lxc/outputs/train_parallel_model"

def main():
    # 解析参数
    port = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT
    directory = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_DIR
    
    # 检查目录是否存在
    if not os.path.exists(directory):
        print(f"错误: 目录不存在: {directory}")
        sys.exit(1)
    
    # 切换到目标目录
    os.chdir(directory)
    
    # 创建 HTTP 服务器
    handler = http.server.SimpleHTTPRequestHandler
    
    # 添加 CORS 头，允许跨域访问
    class CORSRequestHandler(http.server.SimpleHTTPRequestHandler):
        def end_headers(self):
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            super().end_headers()
    
    try:
        with socketserver.TCPServer(("", port), CORSRequestHandler) as httpd:
            print("=" * 60)
            print("HTTP 文件服务器已启动")
            print("=" * 60)
            print(f"服务目录: {directory}")
            print(f"访问地址: http://localhost:{port}")
            print(f"          http://0.0.0.0:{port}")
            print("=" * 60)
            print("提示: 在本地浏览器访问上述地址即可浏览和下载文件")
            print("按 Ctrl+C 停止服务器")
            print("=" * 60)
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n服务器已停止")
    except OSError as e:
        if e.errno == 98:  # Address already in use
            print(f"错误: 端口 {port} 已被占用，请使用其他端口")
            print(f"例如: python {sys.argv[0]} {port + 1}")
        else:
            print(f"错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

