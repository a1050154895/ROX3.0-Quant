#!/bin/bash

# 获取脚本所在目录的绝对路径，并切换到该目录（确保相对路径正确）
cd "$(dirname "$0")"

echo "=================================================="
echo "   🚀 ROX 3.0 Pro 量化终端 (Mac 启动器)"
echo "   正在初始化环境，请稍候..."
echo "=================================================="

# 检查 Python 环境
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未检测到 Python3 环境！"
    echo "💡 请前往 https://www.python.org/downloads/ 下载并安装 Python 3.9+"
    read -p "按回车键退出..."
    exit 1
fi

# 检查是否存在虚拟环境
if [ ! -d "venv" ]; then
    echo "正在为您创建虚拟环境 (首次运行可能较慢)..."
    python3 -m venv venv
    echo "✅ 虚拟环境创建成功！"
fi

# 激活虚拟环境
source venv/bin/activate

# 安装/更新依赖
echo "正在检查并安装依赖库..."
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 启动浏览器 (延迟3秒，等服务器启动)
(sleep 3 && open "http://localhost:8002") &

# 启动服务器
echo "✅ 环境准备就绪，正在启动 ROX 服务..."
echo "🌐 请在浏览器中访问: http://localhost:8002"
echo "⚠️  注意: 请勿关闭此终端窗口"
echo "--------------------------------------------------"

python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8002 --reload

