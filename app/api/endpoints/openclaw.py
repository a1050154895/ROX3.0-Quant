"""
OpenClaw集成API端点
"""

from fastapi import APIRouter, HTTPException
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

class TraderUpdateRequest(BaseModel):
    trader_id: str = Field(..., description="交易员ID")
    trader_name: str = Field(..., description="交易员名称")
    action: str = Field(..., description="动作类型")
    symbol: Optional[str] = Field(None, description="股票代码")
    profit: Optional[float] = Field(None, description="盈亏")
    emotion: Optional[float] = Field(None, ge=0, le=1, description="情绪值")

class RiskAlertRequest(BaseModel):
    alert_level: str = Field(..., description="预警级别")
    title: str = Field(..., description="标题")
    message: str = Field(..., description="消息内容")
    details: Optional[dict] = Field(None, description="详细信息")

@router.get("/health")
async def health_check():
    """检查OpenClaw Gateway健康状态"""
    is_healthy = openclaw_client.health_check()
    return {
        "status": "healthy" if is_healthy else "unhealthy",
        "gateway_url": openclaw_client.gateway_url,
        "enabled": openclaw_client.enabled
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

@router.post("/trader-update")
async def send_trader_update(request: TraderUpdateRequest):
    """发送交易员动态"""
    try:
        result = await openclaw_client.send_trader_update(
            trader_id=request.trader_id,
            trader_name=request.trader_name,
            action=request.action,
            symbol=request.symbol,
            profit=request.profit,
            emotion=request.emotion
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/risk-alert")
async def send_risk_alert(request: RiskAlertRequest):
    """发送风险预警"""
    try:
        result = await openclaw_client.send_risk_alert(
            alert_level=request.alert_level,
            title=request.title,
            message=request.message,
            details=request.details
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/enable")
async def enable_openclaw():
    """启用OpenClaw集成"""
    openclaw_client.enabled = True
    return {"status": "enabled", "message": "OpenClaw集成已启用"}

@router.post("/disable")
async def disable_openclaw():
    """禁用OpenClaw集成"""
    openclaw_client.enabled = False
    return {"status": "disabled", "message": "OpenClaw集成已禁用"}
