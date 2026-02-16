#!/bin/bash

# ==========================================
# ROX Quant 3.0 启动脚本
# 包含依赖安装和配置检查
# ==========================================

echo "=========================================="
echo "  ROX Quant 3.0 启动脚本"
echo "=========================================="

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR" || { echo "错误: 无法进入项目目录 $SCRIPT_DIR"; exit 1; }

# 检查 Python 是否安装
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到 python3，请先安装 Python 3.9 或更高版本。"
    echo "推荐安装 Python 3.9.x 版本。"
    exit 1
fi

# 检查 Python 版本
PYTHON_VERSION=$(python3 --version | awk '{print $2}')
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 9 ]); then
    echo "警告: Python 版本 $PYTHON_VERSION 可能不兼容。"
    echo "推荐使用 Python 3.9 或更高版本。"
fi

# 检查并创建虚拟环境
if [ ! -d ".venv" ]; then
    echo "创建虚拟环境..."
    python3 -m venv .venv
    if [ $? -ne 0 ]; then
        echo "错误: 创建虚拟环境失败。"
        exit 1
    fi
    echo "虚拟环境创建成功。"
fi

# 激活虚拟环境
echo "激活虚拟环境..."
if [ -f ".venv/bin/activate" ]; then
    source ".venv/bin/activate"
elif [ -f ".venv/Scripts/activate" ]; then
    source ".venv/Scripts/activate"
else
    echo "错误: 找不到虚拟环境激活脚本。"
    exit 1
fi

# 升级 pip
echo "升级 pip..."
pip install --upgrade pip

# 安装依赖
echo "安装依赖..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "警告: 依赖安装失败，可能会影响部分功能。"
    else
        echo "依赖安装成功。"
    fi
else
    echo "警告: 找不到 requirements.txt 文件。"
fi

# 检查 .env 文件
if [ ! -f ".env" ]; then
    echo "创建 .env 文件..."
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo ".env 文件已创建，请根据需要修改配置。"
        echo "重要配置项："
        echo "  - ALLTICK_TOKEN: 实时行情数据令牌"
        echo "  - AI_API_KEY: AI 模型 API 密钥"
        echo "  - SECRET_KEY: JWT 令牌密钥"
        echo ""
        echo "生成 SECRET_KEY 的方法："
        echo "  python -c \"import secrets; print(secrets.token_urlsafe(32))\""
        echo ""
    else
        echo "错误: 找不到 .env.example 文件。"
        exit 1
    fi
fi

# 检查 SECRET_KEY
if grep -q "SECRET_KEY=your_secret_key_here" .env; then
    echo "生成随机 SECRET_KEY..."
    SECRET_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
    sed -i "s/SECRET_KEY=your_secret_key_here/SECRET_KEY=$SECRET_KEY/g" .env
    echo "SECRET_KEY 已生成并更新。"
fi

# 释放端口 8081
PORT=8081
echo "检查端口 $PORT..."
if command -v lsof &> /dev/null; then
    PIDS="$(lsof -nP -iTCP:$PORT -sTCP:LISTEN -t 2>/dev/null || true)"
    if [ -n "$PIDS" ]; then
        echo "端口 $PORT 已被占用，正在尝试释放..."
        kill -TERM $PIDS 2>/dev/null || true
        sleep 1
        PIDS2="$(lsof -nP -iTCP:$PORT -sTCP:LISTEN -t 2>/dev/null || true)"
        if [ -n "$PIDS2" ]; then
            kill -KILL $PIDS2 2>/dev/null || true
            sleep 1
        fi
    fi
fi

# 显示访问地址
echo "=========================================="
echo "  启动 ROX Quant 3.0 服务器"
echo "=========================================="
echo "  访问地址:"
echo "  经典版: http://127.0.0.1:$PORT"
echo "  专业版: http://127.0.0.1:$PORT/pro"
echo "  健康检查: http://127.0.0.1:$PORT/api/system/health"
echo ""
echo "  操作说明:"
echo "  - 按 Ctrl+C 停止服务器"
echo "  - 首次启动可能需要较长时间加载数据"
echo "  - 如有问题，请查看日志输出"
echo "=========================================="
echo ""

# 启动服务器
echo "启动服务器..."
exec python -m uvicorn app.main:app --host 127.0.0.1 --port $PORT --reload
