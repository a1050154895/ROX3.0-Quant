#!/bin/bash
# ROX 3.0 Quant 一键启动脚本

echo "=========================================="
echo "ROX 3.0 Quant 量化投研平台"
echo "=========================================="
echo ""

# 检查Python版本
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "检测到 Python 版本: $python_version"

# 检查依赖
echo ""
echo "检查依赖包..."
python3 -c "import fastapi" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ 缺少 fastapi，正在安装..."
    python3 -m pip install fastapi uvicorn --quiet
fi

python3 -c "import akshare" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ 缺少 akshare，正在安装..."
    python3 -m pip install akshare --quiet
fi

echo "✅ 依赖检查完成"
echo ""

# 创建必要目录
mkdir -p data/logs
mkdir -p data/cache

# 启动应用
echo "=========================================="
echo "正在启动 ROX 3.0 Quant..."
echo "访问地址: http://127.0.0.1:8099"
echo "API文档: http://127.0.0.1:8099/docs"
echo "=========================================="
echo ""

cd "$(dirname "$0")"

python3 -m uvicorn app.main:app \
    --host 127.0.0.1 \
    --port 8099 \
    --reload \
    --log-level info
