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
api_group.include_router(market.router, prefix="/market", tags=["market"])

# 个股诊断相关 (新添加)
api_group.include_router(stock.router, prefix="/stock", tags=["stock"])

# 哲学/方法论相关（矛盾分析、价值规律等）
api_group.include_router(philosophy.router, prefix="/philosophy", tags=["philosophy"])

# 交易相关
api_group.include_router(trade.router, prefix="/trade", tags=["trade"])

# 分析相关
api_group.include_router(analysis.router, prefix="/analysis", tags=["analysis"])

# 知识库相关
api_group.include_router(kb.router, prefix="/kb", tags=["kb"])

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

# 价格预警 (新添加)
from app.api.endpoints import alerts
api_group.include_router(alerts.router, tags=["alerts"])

# 数据同步 (新添加)
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

# 将API组添加到主路由
api_router.include_router(api_group)

# ============ WebSocket路由 (顶级) ============
api_router.include_router(ws.router, tags=["websocket"])
api_router.include_router(ws_enhanced.router, tags=["websocket-enhanced"])
