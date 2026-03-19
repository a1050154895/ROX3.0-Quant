from fastapi import APIRouter
from app.api.endpoints import auth, market, trade, analysis, kb, system, ws, ws_enhanced, professional, strategy, ai, stock, philosophy, backtest, tdx

api_router = APIRouter()

# ============ 身份验证路由 (顶级) ============
# /token, /register
api_router.include_router(auth.router, tags=["auth"])

# 用户管理
from app.api.endpoints import users
api_router.include_router(users.router)

# ============ API路由组 (统一 /api 前缀) ============
api_group = APIRouter(prefix="/api")

# AI Chat
api_group.include_router(ai.router, prefix="/ai", tags=["ai"])

# 市场数据相关
api_group.include_router(market.router, tags=["market"])

# 个股诊断相关 (新添加)
api_group.include_router(stock.router, prefix="/stock", tags=["stock"])

# 哲学/方法论相关（矛盾分析、价值规律等）
api_group.include_router(philosophy.router, prefix="/philosophy", tags=["philosophy"])

# 交易相关
api_group.include_router(trade.router, prefix="/trade", tags=["trade"])

# 账户管理相关
from app.api.endpoints import accounts
api_group.include_router(accounts.router, prefix="/accounts", tags=["accounts"])

# 分析相关
api_group.include_router(analysis.router, prefix="/analysis", tags=["analysis"])

# 知识库相关
api_group.include_router(kb.router, prefix="/kb", tags=["kb"])

# 知识管理系统 (新添加)
from app.api.endpoints import knowledge
api_group.include_router(knowledge.router, tags=["knowledge"])

# 聚宽策略系统 (新添加)
from app.api.endpoints import strategies
api_group.include_router(strategies.router)

# 系统相关
api_group.include_router(system.router, prefix="/system", tags=["system"])

# 专业量化系统相关
api_group.include_router(professional.router, prefix="/professional", tags=["professional"])

# 策略构建器相关
api_group.include_router(strategy.router, prefix="/strategy", tags=["strategy"])

# 回测 API（通用 run / 因子分析 / 过拟合检测）
api_group.include_router(backtest.router)

# 通达信插件接口
api_group.include_router(tdx.router, prefix="/tdx", tags=["tdx"])

# 机器学习预测接口 (新添加)
from app.api.endpoints import ml
api_group.include_router(ml.router)

# 模拟交易账户 (新添加)
from app.api.endpoints import portfolio
api_group.include_router(portfolio.router, prefix="/portfolio", tags=["portfolio"])

# 多智能体分析 (新添加)
from app.api.endpoints import agents
api_group.include_router(agents.router, tags=["agents"])

# 策略市场 (新添加)
from app.api.endpoints import marketplace
api_group.include_router(marketplace.router)

# 数据导出 (新添加)
from app.api.endpoints import export
api_group.include_router(export.router, tags=["export"])

# 卢麒元方法论预测系统 (新添加)
from app.api.endpoints import lu_prediction
api_group.include_router(lu_prediction.router, tags=["卢麒元预测"])

# 哲学思想量化系统 (新添加)
from app.api.endpoints import philosophy_prediction
api_group.include_router(philosophy_prediction.router, tags=["哲学思想量化"])

# 东方智慧量化系统 (新添加)
from app.api.endpoints import eastern_wisdom
api_group.include_router(eastern_wisdom.router, tags=["东方智慧量化"])

# 增强版专业系统 (新添加)
from app.api.endpoints import professional_plus
api_group.include_router(professional_plus.router, tags=["增强版专业系统"])

# 价格预警 (新添加)
from app.api.endpoints import alerts
api_group.include_router(alerts.router, tags=["alerts"])

# 数据同步 (新添加)
from app.api.endpoints import sync
api_group.include_router(sync.router)

# 宏观数据 (Phase 6)
from app.api.endpoints import macro
api_group.include_router(macro.router)

# 市场资讯 (Phase 6)
from app.api.endpoints import info
api_group.include_router(info.router)

# 设置相关 (新添加)
from app.api.endpoints import settings
api_group.include_router(settings.router, prefix="/settings", tags=["settings"])

# 卢式作战室 (新添加)
from app.api.endpoints import lu
api_group.include_router(lu.router, tags=["卢式作战室"])

# 交易模拟系统 (新添加)
from app.api.endpoints import trading_simulation
api_group.include_router(trading_simulation.router, prefix="/trading-simulation", tags=["trading-simulation"])

# AI聊天室 (A2A平台)
from app.api.endpoints import ai_chat
api_group.include_router(ai_chat.router, prefix="/ai-chat", tags=["ai-chat"])

# AI评论区 (A2A平台)
from app.api.endpoints import ai_comments
api_group.include_router(ai_comments.router, prefix="/ai-comments", tags=["ai-comments"])

# OpenClaw集成 (AI助手框架)
from app.api.endpoints import openclaw
api_group.include_router(openclaw.router, prefix="/openclaw", tags=["openclaw"])

# 将API组添加到主路由
api_router.include_router(api_group)

# ============ WebSocket路由 (顶级) ============
api_router.include_router(ws.router, tags=["websocket"])
api_router.include_router(ws_enhanced.router, tags=["websocket-enhanced"])
from app.api.endpoints import news, ai_assistant

api_router.include_router(news.router, prefix="/news", tags=["news"])
api_router.include_router(ai_assistant.router, prefix="/ai_assistant", tags=["ai_assistant"])
