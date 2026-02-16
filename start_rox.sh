#!/bin/bash

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "   ROX QUANT 启动程序 (终端版) v13.9"
echo "=========================================="

# 1. 检查 Python 环境
if [ -d ".venv" ]; then
    echo "[1/3] 检测到虚拟环境，正在激活..."
    source .venv/bin/activate
else
    echo "[1/3] 未检测到虚拟环境，尝试使用系统 Python..."
fi

# 2. 检查依赖
echo "[2/3] 正在验证核心依赖..."
python3 -c "import webview; import fastapi; import uvicorn; import akshare" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "错误: 缺少必要依赖。请运行 'pip install -r requirements.txt' 安装依赖。"
    exit 1
fi
echo "依赖检查通过。"

# 3. 启动应用
echo "[3/3] 正在启动 GUI 界面..."
echo "日志将保存至: ~/rox_quant_debug.log"
python3 rox_desktop.py
