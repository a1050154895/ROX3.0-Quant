from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import logging
from app.auth import get_current_user, User
from app.rox_quant.llm import AIClient
from app.db import get_db, list_prompt_templates, get_prompt_template, save_prompt_template
from app.ai.decision_assistant import AIDecisionAssistant
from app.rox_quant.market_analysis import MarketAnalyzer

router = APIRouter()
logger = logging.getLogger("rox-ai")

# Initialize AI Client
try:
    ai_client = AIClient()
except Exception as e:
    logger.error(f"Failed to initialize AI Client: {e}")
    ai_client = None

# Initialize AI Assistant and Market Analyzer
try:
    ai_assistant = AIDecisionAssistant()
    market_analyzer = MarketAnalyzer()
except Exception as e:
    logger.error(f"Failed to initialize AI components: {e}")
    ai_assistant = None
    market_analyzer = None

class ChatRequest(BaseModel):
    message: str
    context: str = ""
    model: Optional[str] = None
    provider: Optional[str] = None

class AnalysisRequest(BaseModel):
    stock_name: str
    stock_code: str
    price: float
    indicators: Dict[str, Any] = {}
    model: Optional[str] = None
    provider: Optional[str] = None


@router.get("/providers")
async def list_ai_providers():
    """
    返回可用 AI 后端列表（多模型/多平台，参考 go-stock）。
    """
    if not ai_client:
        return {"current": "default", "list": []}
    return ai_client.list_providers()


@router.post("/chat")
async def chat(req: ChatRequest, current_user: User = Depends(get_current_user)):
    """
    AI Chat Endpoint；支持 provider/model 切换。
    """
    if not ai_client:
        return {"response": "AI 服务初始化失败，请检查服务端配置。"}

    try:
        user_context = f"用户: {current_user.username}\n{req.context}"
        response = await ai_client.chat_with_search(
            message=req.message,
            context=user_context,
            model=req.model,
            provider=req.provider,
        )
        return {"response": response}
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        return {"response": "AI 思考过程中发生了错误，请稍后再试。"}


@router.post("/analyze")
async def analyze_stock(req: AnalysisRequest, current_user: User = Depends(get_current_user)):
    """
    Deep Stock Analysis Endpoint；支持 provider/model 切换。
    """
    if not ai_client:
        raise HTTPException(status_code=503, detail="AI Service Unavailable")

    try:
        result = await ai_client.analyze_stock(
            stock_name=req.stock_name,
            stock_code=req.stock_code,
            price=req.price,
            indicators=req.indicators,
            model=req.model,
            provider=req.provider,
        )
        return result
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------- AI 模板（参考 go-stock） ----------
class TemplateCreate(BaseModel):
    name: str
    key: str
    content: str
    scope: str = "user"


@router.get("/templates")
async def api_list_templates(
    scope: Optional[str] = Query(None, description="system | user"),
    current_user: User = Depends(get_current_user),
    conn = Depends(get_db),
):
    """返回 AI 提示词模板列表（可配置分析/选股模板）。"""
    items = list_prompt_templates(conn, user_id=current_user.id, scope=scope)
    return {"items": items}


@router.get("/templates/{key}")
async def api_get_template(
    key: str,
    current_user: User = Depends(get_current_user),
    conn = Depends(get_db),
):
    """按 key 获取单个模板内容。"""
    row = get_prompt_template(conn, key=key, user_id=current_user.id)
    if not row:
        raise HTTPException(status_code=404, detail="模板不存在")
    return {"key": row["key"], "name": row["name"], "content": row["content"], "scope": row["scope"]}


@router.post("/templates")
async def api_create_template(
    req: TemplateCreate,
    current_user: User = Depends(get_current_user),
    conn = Depends(get_db),
):
    """新建用户 AI 提示词模板。"""
    tid = save_prompt_template(conn, current_user.id, req.name, req.key, req.content, req.scope)
    if tid is None:
        raise HTTPException(status_code=500, detail="保存失败")
    return {"id": tid, "key": req.key, "name": req.name}


# ---------- AI 决策助手 ----------
class TradeRecommendationRequest(BaseModel):
    symbol: str
    signals: Dict[str, Any] = {}

class PortfolioRequest(BaseModel):
    positions: List[Dict[str, Any]] = []
    cash: float = 0


@router.post("/decision/trade")
async def get_trade_recommendation(
    req: TradeRecommendationRequest,
    current_user: User = Depends(get_current_user)
):
    """
    获取交易建议
    """
    try:
        recommendation = await ai_assistant.generate_trade_recommendation(
            req.symbol, req.signals
        )
        return recommendation
    except Exception as e:
        logger.error(f"Trade recommendation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/decision/portfolio")
async def get_portfolio_advice(
    req: PortfolioRequest,
    current_user: User = Depends(get_current_user)
):
    """
    获取投资组合建议
    """
    try:
        portfolio = {
            "positions": req.positions,
            "cash": req.cash
        }
        advice = await ai_assistant.get_portfolio_advice(portfolio)
        return advice
    except Exception as e:
        logger.error(f"Portfolio advice failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/decision/market-insights")
async def get_market_insights(
    current_user: User = Depends(get_current_user)
):
    """
    获取市场洞察
    """
    try:
        insights = await ai_assistant.get_market_insights()
        return insights
    except Exception as e:
        logger.error(f"Market insights failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/decision/sector-performance")
async def get_sector_performance(
    current_user: User = Depends(get_current_user)
):
    """
    获取行业表现
    """
    try:
        performance = await market_analyzer.get_sector_performance()
        return performance
    except Exception as e:
        logger.error(f"Sector performance failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/decision/market-opportunities")
async def get_market_opportunities(
    current_user: User = Depends(get_current_user)
):
    """
    获取市场机会
    """
    try:
        opportunities = await market_analyzer.get_market_opportunities()
        return {"opportunities": opportunities}
    except Exception as e:
        logger.error(f"Market opportunities failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/decision/history-analysis")
async def get_recommendation_history(
    current_user: User = Depends(get_current_user)
):
    """
    获取建议历史分析
    """
    try:
        analysis = ai_assistant.analyze_recommendation_history()
        return analysis
    except Exception as e:
        logger.error(f"History analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

