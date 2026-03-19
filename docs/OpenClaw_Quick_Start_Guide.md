# OpenClaw快速启动指南

## 概述

本指南帮助您在30分钟内完成OpenClaw与ROX平台的基础集成，包括环境搭建、API测试和消息推送验证。

---

## 前置要求

### 系统要求
- 操作系统：macOS / Linux / Windows (WSL2)
- Node.js：≥ 22.0
- Python：≥ 3.9
- 内存：≥ 4GB
- 存储：≥ 10GB

### 账户要求
- Telegram账户（用于创建Bot）
- OpenAI API密钥 或 Anthropic API密钥

---

## 快速安装（10分钟）

### 步骤1：安装Node.js（如果未安装）

**macOS**：
```bash
# 使用Homebrew安装
brew install node@22

# 或使用nvm安装
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
nvm install 22
nvm use 22
```

**Linux (Ubuntu/Debian)**：
```bash
# 使用NodeSource安装
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
sudo apt-get install -y nodejs

# 验证安装
node --version  # 应显示 v22.x.x
npm --version
```

**Windows (WSL2)**：
```bash
# 在WSL2中运行Ubuntu，然后按照Linux步骤操作
```

### 步骤2：安装OpenClaw

```bash
# 全局安装OpenClaw
npm install -g openclaw@latest

# 验证安装
openclaw --version
```

### 步骤3：运行配置向导

```bash
# 启动配置向导
openclaw onboard --install-daemon
```

向导会引导您完成：
1. Gateway设置
2. 工作空间配置
3. 渠道配置
4. 技能安装

### 步骤4：创建基础配置

```bash
# 创建配置目录
mkdir -p ~/.openclaw

# 创建配置文件
cat > ~/.openclaw/openclaw.json << 'EOF'
{
  "agent": {
    "model": "anthropic/claude-opus-4-6",
    "thinking": "medium"
  },
  "gateway": {
    "port": 18789,
    "bind": "127.0.0.1",
    "verbose": true
  }
}
EOF
```

### 步骤5：启动Gateway

```bash
# 前台运行（用于调试）
openclaw gateway --port 18789 --verbose

# 或后台运行（生产环境）
openclaw gateway --port 18789 --daemon
```

### 步骤6：验证Gateway运行

```bash
# 检查Gateway状态
curl http://localhost:18789/health

# 应返回
# {"status":"ok","version":"2026.3.19"}
```

---

## Telegram Bot配置（5分钟）

### 步骤1：创建Telegram Bot

1. 在Telegram中搜索 `@BotFather`
2. 发送 `/newbot` 命令
3. 按提示设置Bot名称和用户名
4. 保存返回的Bot Token

示例：
```
BotFather: Done! Congratulations on your new bot...
Use this token to access the HTTP API:
1234567890:ABCdefGHIjklMNOpqrsTUVwxyz
```

### 步骤2：更新OpenClaw配置

```bash
# 更新配置文件
cat > ~/.openclaw/openclaw.json << EOF
{
  "agent": {
    "model": "anthropic/claude-opus-4-6"
  },
  "gateway": {
    "port": 18789,
    "bind": "127.0.0.1"
  },
  "channels": {
    "telegram": {
      "botToken": "YOUR_BOT_TOKEN_HERE"
    }
  }
}
EOF

# 重启Gateway
openclaw gateway --restart
```

### 步骤3：获取Chat ID

```bash
# 方法1：使用OpenClaw CLI
openclaw channels telegram chat-id

# 方法2：手动获取
# 1. 向您的Bot发送消息
# 2. 访问 https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates
# 3. 在返回的JSON中找到chat.id
```

---

## ROX集成配置（10分钟）

### 步骤1：安装Python依赖

```bash
cd /Users/mac/Downloads/rox3.0
pip install requests websockets
```

### 步骤2：创建OpenClaw客户端模块

创建文件 `app/services/openclaw_client.py`：

```python
"""
OpenClaw客户端模块
用于ROX平台与OpenClaw Gateway的通信
"""

import requests
import json
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class OpenClawClient:
    """OpenClaw Gateway客户端"""
    
    def __init__(self, gateway_url: str = "http://localhost:18789"):
        self.gateway_url = gateway_url
        self.timeout = 10
    
    def health_check(self) -> bool:
        """检查Gateway健康状态"""
        try:
            response = requests.get(
                f"{self.gateway_url}/health",
                timeout=5
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Gateway健康检查失败: {e}")
            return False
    
    async def send_trade_signal(
        self,
        symbol: str,
        action: str,
        price: float,
        confidence: float,
        reason: str = None
    ) -> Dict:
        """
        发送交易信号
        
        Args:
            symbol: 股票代码
            action: 操作类型 (buy/sell)
            price: 价格
            confidence: 置信度 (0-1)
            reason: 理由
        
        Returns:
            发送结果
        """
        payload = {
            "type": "trade_signal",
            "symbol": symbol,
            "action": action,
            "price": price,
            "confidence": confidence,
            "reason": reason
        }
        
        try:
            response = requests.post(
                f"{self.gateway_url}/api/message/send",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"发送交易信号失败: {e}")
            raise
    
    async def send_telegram_message(
        self,
        chat_id: str,
        message: str,
        parse_mode: str = "Markdown"
    ) -> Dict:
        """
        发送Telegram消息
        
        Args:
            chat_id: Telegram聊天ID
            message: 消息内容
            parse_mode: 解析模式
        
        Returns:
            发送结果
        """
        payload = {
            "channel": "telegram",
            "chat_id": chat_id,
            "message": message,
            "parse_mode": parse_mode
        }
        
        try:
            response = requests.post(
                f"{self.gateway_url}/api/message/send",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"发送Telegram消息失败: {e}")
            raise
    
    async def broadcast_message(
        self,
        message: str,
        channels: List[str] = None,
        recipients: Dict[str, List[str]] = None
    ) -> Dict:
        """
        广播消息到多个渠道
        
        Args:
            message: 消息内容
            channels: 渠道列表
            recipients: 接收者映射
        
        Returns:
            发送结果
        """
        payload = {
            "message": message,
            "channels": channels or ["telegram"],
            "recipients": recipients or {}
        }
        
        try:
            response = requests.post(
                f"{self.gateway_url}/api/message/broadcast",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"广播消息失败: {e}")
            raise

# 全局客户端实例
openclaw_client = OpenClawClient()
```

### 步骤3：创建API端点

创建文件 `app/api/endpoints/openclaw.py`：

```python
"""
OpenClaw集成API端点
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from app.services.openclaw_client import openclaw_client

router = APIRouter()

class TradeSignalRequest(BaseModel):
    symbol: str = Field(..., description="股票代码")
    action: str = Field(..., description="操作类型")
    price: float = Field(..., description="价格")
    confidence: float = Field(..., ge=0, le=1, description="置信度")
    reason: Optional[str] = Field(None, description="理由")

class TelegramMessageRequest(BaseModel):
    chat_id: str = Field(..., description="Telegram聊天ID")
    message: str = Field(..., description="消息内容")
    parse_mode: str = Field("Markdown", description="解析模式")

class BroadcastRequest(BaseModel):
    message: str = Field(..., description="消息内容")
    channels: Optional[List[str]] = Field(None, description="渠道列表")
    recipients: Optional[dict] = Field(None, description="接收者映射")

@router.get("/health")
async def health_check():
    """检查OpenClaw Gateway健康状态"""
    is_healthy = openclaw_client.health_check()
    return {
        "status": "healthy" if is_healthy else "unhealthy",
        "gateway_url": openclaw_client.gateway_url
    }

@router.post("/signal")
async def send_trade_signal(request: TradeSignalRequest):
    """发送交易信号"""
    try:
        result = await openclaw_client.send_trade_signal(
            symbol=request.symbol,
            action=request.action,
            price=request.price,
            confidence=request.confidence,
            reason=request.reason
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/telegram")
async def send_telegram_message(request: TelegramMessageRequest):
    """发送Telegram消息"""
    try:
        result = await openclaw_client.send_telegram_message(
            chat_id=request.chat_id,
            message=request.message,
            parse_mode=request.parse_mode
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/broadcast")
async def broadcast_message(request: BroadcastRequest):
    """广播消息"""
    try:
        result = await openclaw_client.broadcast_message(
            message=request.message,
            channels=request.channels,
            recipients=request.recipients
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### 步骤4：注册API路由

编辑 `app/api/api.py`，添加：

```python
# OpenClaw集成 (新添加)
from app.api.endpoints import openclaw
api_group.include_router(openclaw.router, prefix="/openclaw", tags=["openclaw"])
```

### 步骤5：重启ROX服务

```bash
# 停止现有服务
pkill -f uvicorn

# 重新启动
cd /Users/mac/Downloads/rox3.0
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 功能测试（5分钟）

### 测试1：健康检查

```bash
curl http://localhost:8000/api/openclaw/health
```

预期返回：
```json
{
  "status": "healthy",
  "gateway_url": "http://localhost:18789"
}
```

### 测试2：发送交易信号

```bash
curl -X POST http://localhost:8000/api/openclaw/signal \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "600519",
    "action": "buy",
    "price": 1800.50,
    "confidence": 0.85,
    "reason": "技术面突破关键阻力位"
  }'
```

### 测试3：发送Telegram消息

```bash
curl -X POST http://localhost:8000/api/openclaw/telegram \
  -H "Content-Type: application/json" \
  -d '{
    "chat_id": "YOUR_CHAT_ID",
    "message": "*测试消息*\n\n这是一条来自ROX的测试消息。"
  }'
```

### 测试4：Python测试脚本

创建文件 `test_openclaw_integration.py`：

```python
import asyncio
import requests

async def test_integration():
    print("开始测试OpenClaw集成...")
    
    # 1. 健康检查
    print("\n1. 测试健康检查...")
    response = requests.get("http://localhost:8000/api/openclaw/health")
    print(f"状态: {response.json()}")
    
    # 2. 发送交易信号
    print("\n2. 测试交易信号...")
    signal = {
        "symbol": "600519",
        "action": "buy",
        "price": 1800.50,
        "confidence": 0.85,
        "reason": "技术面突破关键阻力位"
    }
    response = requests.post(
        "http://localhost:8000/api/openclaw/signal",
        json=signal
    )
    print(f"结果: {response.json()}")
    
    # 3. 发送Telegram消息
    print("\n3. 测试Telegram消息...")
    message = {
        "chat_id": "YOUR_CHAT_ID",  # 替换为实际的Chat ID
        "message": "*🚀 交易信号提醒*\n\n股票: 600519\n操作: 买入\n价格: ¥1800.50"
    }
    response = requests.post(
        "http://localhost:8000/api/openclaw/telegram",
        json=message
    )
    print(f"结果: {response.json()}")
    
    print("\n✅ 所有测试完成！")

# 运行测试
asyncio.run(test_integration())
```

运行测试：
```bash
python test_openclaw_integration.py
```

---

## 故障排除

### 问题1：Gateway无法启动

**症状**：
```
Error: listen EADDRINUSE: address already in use :::18789
```

**解决方案**：
```bash
# 查找占用端口的进程
lsof -i :18789

# 杀死进程
kill -9 <PID>

# 重启Gateway
openclaw gateway --port 18789
```

### 问题2：Telegram消息发送失败

**症状**：
```
Error: 404 Not Found
```

**解决方案**：
1. 检查Bot Token是否正确
2. 确认Bot未被禁用
3. 确认Chat ID正确

### 问题3：ROX API调用失败

**症状**：
```
ConnectionError: Connection refused
```

**解决方案**：
1. 确认ROX服务正在运行
2. 检查端口是否正确
3. 检查防火墙设置

### 问题4：消息延迟严重

**解决方案**：
1. 检查网络连接
2. 优化消息队列
3. 增加Gateway资源

---

## 下一步

完成快速启动后，您可以：

1. **阅读完整文档**：
   - [OpenClaw集成可行性报告](./OpenClaw_Integration_Feasibility_Report.md)
   - [OpenClaw集成实施路线图](./OpenClaw_Integration_Roadmap.md)

2. **开始第一阶段实施**：
   - 环境搭建与原型验证
   - API接口设计
   - 基础功能测试

3. **探索高级功能**：
   - 多渠道支持（Slack、Discord）
   - 自然语言交互
   - AI交易员同步

---

## 资源链接

- OpenClaw官方文档：https://docs.openclaw.ai
- OpenClaw GitHub：https://github.com/openclaw/openclaw
- Telegram Bot API：https://core.telegram.org/bots/api
- FastAPI文档：https://fastapi.tiangolo.com

---

**快速启动指南编制**：ROX技术团队  
**创建日期**：2026年3月19日  
**版本**：v1.0
