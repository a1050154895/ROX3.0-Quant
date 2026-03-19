"""
OpenClaw客户端模块
用于ROX平台与OpenClaw Gateway的通信
"""

import requests
import json
from typing import Dict, List, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class OpenClawClient:
    """OpenClaw Gateway客户端"""
    
    def __init__(self, gateway_url: str = "http://localhost:18789"):
        self.gateway_url = gateway_url
        self.timeout = 10
        self.enabled = True
    
    def health_check(self) -> bool:
        """检查Gateway健康状态"""
        if not self.enabled:
            return False
        
        try:
            response = requests.get(
                f"{self.gateway_url}/health",
                timeout=5
            )
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Gateway健康检查失败: {e}")
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
        if not self.enabled:
            return {"status": "disabled", "message": "OpenClaw集成未启用"}
        
        payload = {
            "type": "trade_signal",
            "symbol": symbol,
            "action": action,
            "price": price,
            "confidence": confidence,
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            response = requests.post(
                f"{self.gateway_url}/api/message/send",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            logger.info(f"交易信号发送成功: {symbol} {action}")
            return response.json()
        except Exception as e:
            logger.error(f"发送交易信号失败: {e}")
            return {"status": "error", "message": str(e)}
    
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
        if not self.enabled:
            return {"status": "disabled", "message": "OpenClaw集成未启用"}
        
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
            logger.info(f"Telegram消息发送成功: {chat_id}")
            return response.json()
        except Exception as e:
            logger.error(f"发送Telegram消息失败: {e}")
            return {"status": "error", "message": str(e)}
    
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
        if not self.enabled:
            return {"status": "disabled", "message": "OpenClaw集成未启用"}
        
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
            logger.info(f"消息广播成功")
            return response.json()
        except Exception as e:
            logger.error(f"广播消息失败: {e}")
            return {"status": "error", "message": str(e)}
    
    async def send_trader_update(
        self,
        trader_id: str,
        trader_name: str,
        action: str,
        symbol: Optional[str] = None,
        profit: Optional[float] = None,
        emotion: Optional[float] = None
    ) -> Dict:
        """
        发送AI交易员动态更新
        
        Args:
            trader_id: 交易员ID
            trader_name: 交易员名称
            action: 动作类型
            symbol: 股票代码
            profit: 盈亏
            emotion: 情绪值
        
        Returns:
            发送结果
        """
        if not self.enabled:
            return {"status": "disabled", "message": "OpenClaw集成未启用"}
        
        details = {
            "symbol": symbol,
            "profit": profit,
            "emotion": emotion
        }
        
        payload = {
            "type": "trader_update",
            "trader_id": trader_id,
            "trader_name": trader_name,
            "action": action,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            response = requests.post(
                f"{self.gateway_url}/api/message/send",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            logger.info(f"交易员动态发送成功: {trader_name}")
            return response.json()
        except Exception as e:
            logger.error(f"发送交易员动态失败: {e}")
            return {"status": "error", "message": str(e)}
    
    async def send_risk_alert(
        self,
        alert_level: str,
        title: str,
        message: str,
        details: Dict = None
    ) -> Dict:
        """
        发送风险预警
        
        Args:
            alert_level: 预警级别 (info/warning/critical)
            title: 标题
            message: 消息内容
            details: 详细信息
        
        Returns:
            发送结果
        """
        if not self.enabled:
            return {"status": "disabled", "message": "OpenClaw集成未启用"}
        
        payload = {
            "type": "risk_alert",
            "alert_level": alert_level,
            "title": title,
            "message": message,
            "details": details or {},
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            response = requests.post(
                f"{self.gateway_url}/api/message/send",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            logger.info(f"风险预警发送成功: {alert_level}")
            return response.json()
        except Exception as e:
            logger.error(f"发送风险预警失败: {e}")
            return {"status": "error", "message": str(e)}

# 全局客户端实例
openclaw_client = OpenClawClient()
