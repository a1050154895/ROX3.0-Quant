"""
Multi-Agent Analysis API Endpoints
提供多智能体分析的 REST API
"""

import logging
from typing import List, Optional
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agents", tags=["Multi-Agent Analysis"])


class StockAnalysisRequest(BaseModel):
    """单股分析请求"""
    stock_code: str
    stock_name: Optional[str] = ""


class BatchAnalysisRequest(BaseModel):
    """批量分析请求"""
    stocks: List[dict]  # [{"code": "600519", "name": "贵州茅台"}, ...]
    max_concurrent: Optional[int] = 3


class TradingAgentsRequest(BaseModel):
    """TradingAgents-CN 分析请求"""
    stock_code: str
    stock_name: Optional[str] = ""
    market: Optional[str] = "cn"
    horizon: Optional[str] = "swing"



async def _analyze_with_local_orchestrator(stock_code: str, stock_name: str = ""):
    """本地多智能体兜底分析。"""
    from app.rox_quant.agents import AgentOrchestrator

    orchestrator = AgentOrchestrator()
    return await orchestrator.analyze_stock(stock_code, stock_name)

@router.post("/analyze/{stock_code}")
async def analyze_stock(stock_code: str, stock_name: str = ""):
    """
    对单只股票进行多智能体分析
    
    Args:
        stock_code: 股票代码
        stock_name: 股票名称（可选）
        
    Returns:
        综合分析结果
    """
    try:
        result = await _analyze_with_local_orchestrator(stock_code, stock_name)
        
        return JSONResponse(result)
        
    except Exception as e:
        logger.error(f"Multi-agent analysis failed: {e}")
        return JSONResponse(
            {"success": False, "error": str(e)},
            status_code=500
        )


@router.post("/analyze")
async def analyze_stock_post(request: StockAnalysisRequest):
    """
    对单只股票进行多智能体分析 (POST 方式)
    """
    return await analyze_stock(request.stock_code, request.stock_name)


@router.post("/batch")
async def batch_analyze(request: BatchAnalysisRequest):
    """
    批量分析多只股票
    
    Args:
        request: 批量分析请求
        
    Returns:
        分析结果列表
    """
    try:
        from app.rox_quant.agents import AgentOrchestrator
        
        orchestrator = AgentOrchestrator()
        results = await orchestrator.analyze_batch(
            request.stocks, 
            request.max_concurrent
        )
        
        return JSONResponse({"success": True, "results": results})
        
    except Exception as e:
        logger.error(f"Batch analysis failed: {e}")
        return JSONResponse(
            {"success": False, "error": str(e)},
            status_code=500
        )


@router.get("/info")
async def get_agents_info():
    """
    获取所有 Agent 的信息
    """
    try:
        from app.rox_quant.agents import AgentOrchestrator
        
        orchestrator = AgentOrchestrator()
        info = orchestrator.get_agent_info()
        
        return JSONResponse({
            "success": True,
            "agents": info,
            "count": len(info),
        })
        
    except Exception as e:
        logger.error(f"Get agents info failed: {e}")
        return JSONResponse(
            {"success": False, "error": str(e)},
            status_code=500
        )


@router.post("/single/{agent_name}/{stock_code}")
async def single_agent_analyze(agent_name: str, stock_code: str, stock_name: str = ""):
    """
    使用单个 Agent 进行分析
    
    Args:
        agent_name: Agent 名称 (technical, market, fundamental, news, risk)
        stock_code: 股票代码
        stock_name: 股票名称（可选）
        
    Returns:
        单个 Agent 的分析结果
    """
    try:
        from app.rox_quant.agents import AgentOrchestrator
        
        orchestrator = AgentOrchestrator()
        result = await orchestrator.analyze_single_agent(
            stock_code, agent_name, stock_name
        )
        
        return JSONResponse(result.to_dict())
        
    except Exception as e:
        logger.error(f"Single agent analysis failed: {e}")
        return JSONResponse(
            {"success": False, "error": str(e)},
            status_code=500
        )


@router.post("/tradingagents/analyze")
async def analyze_with_tradingagents(request: TradingAgentsRequest):
    """优先使用 TradingAgents-CN；失败时可按配置降级到本地分析。"""
    from app.core.config import settings
    from app.services.tradingagents_client import TradingAgentsClient

    try:
        client = TradingAgentsClient()
        result = await client.analyze_stock(
            stock_code=request.stock_code,
            stock_name=request.stock_name or "",
            market=request.market or "cn",
            horizon=request.horizon or "swing",
        )
        return JSONResponse(result)
    except RuntimeError as e:
        if not settings.TRADING_AGENTS_FALLBACK_LOCAL:
            raise HTTPException(status_code=400, detail=str(e))

        logger.warning("TradingAgents-CN 不可用，切换本地分析兜底: %s", e)
        local_result = await _analyze_with_local_orchestrator(
            request.stock_code,
            request.stock_name or "",
        )
        return JSONResponse(
            {
                "success": True,
                "provider": "local-fallback",
                "fallback_reason": str(e),
                "stock_code": request.stock_code,
                "stock_name": request.stock_name or "",
                "result": local_result,
            }
        )
    except Exception as e:
        logger.error("TradingAgents-CN analysis failed: %s", e)
        return JSONResponse(
            {"success": False, "error": str(e)},
            status_code=500,
        )


@router.get("/tradingagents/health")
async def tradingagents_health_check():
    """检查 TradingAgents-CN 可达性与健康状态。"""
    from app.core.config import settings
    from app.services.tradingagents_client import TradingAgentsClient

    client = TradingAgentsClient()
    status = await client.health_check()

    if settings.TRADING_AGENTS_HEALTH_STRICT and not status.get("reachable", False):
        raise HTTPException(status_code=503, detail=status)

    return JSONResponse(status)
