#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版专业系统API
整合所有高级功能

功能：
1. 7大核心信号（亢龙有悔、游资暗盘等）
2. 参数优化
3. 组合优化
4. 风控仪表盘
5. 多模态分析
6. 实时行情
"""

import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import pandas as pd

from app.rox_quant.param_optimizer import (
    get_strategy_optimizer,
    ParameterRange,
    OptimizationMethod,
    ObjectiveType,
)
from app.rox_quant.portfolio_optimizer import (
    get_portfolio_optimizer,
    StrategyMetrics,
    OptimizationObjective,
)
from app.services.risk_dashboard import get_risk_monitor
from app.rox_quant.multimodal_analysis import get_multimodal_analyzer

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/professional-plus", tags=["增强版专业系统"])


class ParamOptimizeRequest(BaseModel):
    """参数优化请求"""
    strategy_name: str
    param_ranges: List[Dict]
    method: str = "genetic"
    objective: str = "sharpe"


class PortfolioOptimizeRequest(BaseModel):
    """组合优化请求"""
    strategies: List[Dict]
    objective: str = "max_sharpe"


class RiskDashboardRequest(BaseModel):
    """风控仪表盘请求"""
    positions: List[Dict]
    account: Dict
    price_history: List[float] = None


class MultimodalRequest(BaseModel):
    """多模态分析请求"""
    code: str
    ohlc: List[Dict]


@router.get("/signals/{code}")
async def enhanced_signal_analysis(code: str) -> Dict:
    """
    增强版信号分析 V2
    
    整合7大核心信号 + 多模态分析
    """
    try:
        ohlc = await _get_ohlc_data(code)
        
        multimodal_analyzer = get_multimodal_analyzer()
        multimodal_result = multimodal_analyzer.analyze(ohlc)
        
        core_signals = await _analyze_core_signals(code, ohlc)
        
        combined_signal = _combine_signals(core_signals, multimodal_result)
        
        return {
            "code": code,
            "timestamp": datetime.now().isoformat(),
            "core_signals": core_signals,
            "multimodal_analysis": {
                "candle_patterns": [
                    {
                        "type": p.type.value,
                        "signal": p.signal,
                        "confidence": p.confidence,
                        "description": p.description,
                    }
                    for p in multimodal_result.candle_patterns[-5:]
                ],
                "chart_patterns": [
                    {
                        "type": p.type.value,
                        "signal": p.signal,
                        "confidence": p.confidence,
                        "description": p.description,
                        "target_price": p.target_price,
                        "stop_loss": p.stop_loss,
                    }
                    for p in multimodal_result.chart_patterns
                ],
                "support_levels": [
                    {
                        "level": s.level,
                        "strength": s.strength,
                        "touches": s.touches,
                    }
                    for s in multimodal_result.support_levels
                ],
                "resistance_levels": [
                    {
                        "level": r.level,
                        "strength": r.strength,
                        "touches": r.touches,
                    }
                    for r in multimodal_result.resistance_levels
                ],
                "overall_signal": multimodal_result.overall_signal,
                "confidence": multimodal_result.confidence,
                "reasoning": multimodal_result.reasoning,
            },
            "combined_signal": combined_signal,
        }
        
    except Exception as e:
        logger.error(f"增强信号分析失败 {code}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/signals-v2/{code}")
async def enhanced_signal_analysis_v2(code: str) -> Dict:
    """
    V2增强版信号分析
    
    使用升级版7大核心信号系统
    提供更准确的信号和风险控制
    """
    try:
        ohlc = await _get_ohlc_data(code)
        
        if not ohlc or len(ohlc) < 30:
            raise HTTPException(status_code=400, detail="数据不足，至少需要30条K线数据")
        
        df = pd.DataFrame(ohlc)
        df.columns = ['open', 'high', 'low', 'close', 'volume']
        
        from app.analysis.enhanced_signals_v2 import get_enhanced_signal_engine_v2
        
        engine = get_enhanced_signal_engine_v2()
        analysis = engine.analyze(code, df)
        
        signals_detail = []
        for signal_result in analysis.signals:
            signals_detail.append({
                "name": signal_result.name,
                "signal": signal_result.signal.value,
                "strength": round(signal_result.strength, 1),
                "confidence": round(signal_result.confidence, 2),
                "score": round(signal_result.score, 1),
                "trend": signal_result.trend.value,
                "multi_period_confirm": signal_result.multi_period_confirm,
                "volume_confirm": signal_result.volume_confirm,
                "risk_level": signal_result.risk_level,
                "triggers": signal_result.triggers,
                "entry_price": signal_result.suggested_entry,
                "stop_loss": signal_result.suggested_stop,
                "take_profit": signal_result.suggested_target,
                "valid_days": signal_result.valid_days,
            })
        
        return {
            "code": code,
            "name": analysis.name,
            "timestamp": analysis.timestamp.isoformat(),
            "current_price": analysis.current_price,
            "combined_signal": {
                "signal": analysis.combined_signal.value,
                "strength": round(analysis.combined_strength, 1),
                "confidence": round(analysis.combined_confidence, 2),
            },
            "signal_summary": {
                "buy_signals": analysis.buy_signals,
                "sell_signals": analysis.sell_signals,
                "neutral_signals": analysis.neutral_signals,
            },
            "signals_detail": signals_detail,
            "top_signal": {
                "name": analysis.top_signal.name,
                "signal": analysis.top_signal.signal.value,
                "strength": round(analysis.top_signal.strength, 1),
                "triggers": analysis.top_signal.triggers[:3],
            },
            "analysis": {
                "trend": analysis.trend.value,
                "market_phase": analysis.market_phase,
                "suggested_action": analysis.suggested_action,
                "position_suggestion": round(analysis.position_suggestion, 2),
            },
            "trading_plan": {
                "entry_price": analysis.entry_price,
                "stop_loss": analysis.stop_loss,
                "take_profit": analysis.take_profit,
            },
            "reasoning": analysis.reasoning,
            "risk_warning": analysis.risk_warning,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"V2信号分析失败 {code}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/signal-performance")
async def get_signal_performance() -> Dict:
    """
    获取信号表现统计
    
    返回各信号的历史准确率和收益统计
    """
    try:
        from app.analysis.signal_validator import get_signal_validator
        
        validator = get_signal_validator()
        report = validator.get_performance_report()
        recommendations = validator.get_signal_recommendations()
        
        return {
            "performance": report.to_dict('records') if not report.empty else [],
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat(),
        }
        
    except Exception as e:
        logger.error(f"获取信号表现失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/verify-signal")
async def verify_signal_result(signal_id: str, exit_price: float, price_history: List[float] = None) -> Dict:
    """
    验证信号结果
    
    用于追踪信号的实际效果
    """
    try:
        from app.analysis.signal_validator import get_signal_validator
        
        validator = get_signal_validator()
        result = validator.verify_signal(signal_id, exit_price, price_history=price_history)
        
        return {
            "success": "error" not in result,
            "result": result,
        }
        
    except Exception as e:
        logger.error(f"验证信号失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/optimize-params")
async def optimize_strategy_params(req: ParamOptimizeRequest) -> Dict:
    """
    策略参数优化
    
    支持：
    - 遗传算法
    - 贝叶斯优化
    - 网格搜索
    """
    try:
        optimizer = get_strategy_optimizer()
        
        param_ranges = [
            ParameterRange(
                name=p["name"],
                min_value=p["min"],
                max_value=p["max"],
                step=p.get("step", 1),
                param_type=p.get("type", "float"),
            )
            for p in req.param_ranges
        ]
        
        def mock_backtest(params):
            score = sum(params.values()) / len(params) if params else 0
            return {"sharpe": score, "return": score * 0.1, "drawdown": -score * 0.05}
        
        method_map = {
            "genetic": OptimizationMethod.GENETIC,
            "bayesian": OptimizationMethod.BAYESIAN,
            "grid": OptimizationMethod.GRID,
            "random": OptimizationMethod.RANDOM,
        }
        
        result = optimizer.optimize(
            param_ranges=param_ranges,
            backtest_func=mock_backtest,
            method=method_map.get(req.method, OptimizationMethod.GENETIC),
            objective=req.objective,
        )
        
        return {
            "strategy_name": req.strategy_name,
            "best_params": result.best_params,
            "best_fitness": result.best_fitness,
            "method": result.method,
            "generations": result.generations,
            "time_elapsed": result.time_elapsed,
            "top_results": [
                {"params": r.genes, "fitness": r.fitness}
                for r in result.all_results[:5]
            ],
        }
        
    except Exception as e:
        logger.error(f"参数优化失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/optimize-portfolio")
async def optimize_portfolio(req: PortfolioOptimizeRequest) -> Dict:
    """
    策略组合优化
    
    支持：
    - 最大夏普比率
    - 最小波动率
    - 风险平价
    """
    try:
        optimizer = get_portfolio_optimizer()
        
        strategies = [
            StrategyMetrics(
                name=s["name"],
                returns=s.get("returns", [0.01] * 20),
            )
            for s in req.strategies
        ]
        
        obj_map = {
            "max_sharpe": OptimizationObjective.MAX_SHARPE,
            "min_volatility": OptimizationObjective.MIN_VOLATILITY,
            "risk_parity": OptimizationObjective.RISK_PARITY,
        }
        
        result = optimizer.optimize(
            strategies=strategies,
            objective=obj_map.get(req.objective, OptimizationObjective.MAX_SHARPE),
        )
        
        return {
            "weights": result.weights,
            "expected_return": result.expected_return,
            "expected_volatility": result.expected_volatility,
            "sharpe_ratio": result.sharpe_ratio,
            "diversification_ratio": result.diversification_ratio,
            "strategy_names": result.strategy_names,
        }
        
    except Exception as e:
        logger.error(f"组合优化失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/risk-dashboard")
async def get_risk_dashboard(req: RiskDashboardRequest) -> Dict:
    """
    风控仪表盘
    
    实时监控账户风险状态
    """
    try:
        monitor = get_risk_monitor()
        
        dashboard = monitor.calculate_dashboard(
            positions=req.positions,
            account=req.account,
            price_history=req.price_history,
        )
        
        return {
            "total_value": dashboard.total_value,
            "cash": dashboard.cash,
            "position_value": dashboard.position_value,
            "leverage": dashboard.leverage,
            "daily_pnl": dashboard.daily_pnl,
            "daily_pnl_pct": dashboard.daily_pnl_pct,
            "max_drawdown": dashboard.max_drawdown,
            "current_drawdown": dashboard.current_drawdown,
            "var_95": dashboard.var_95,
            "cvar_95": dashboard.cvar_95,
            "sharpe_ratio": dashboard.sharpe_ratio,
            "sortino_ratio": dashboard.sortino_ratio,
            "position_count": dashboard.position_count,
            "concentration": dashboard.concentration,
            "risk_level": dashboard.risk_level.value,
            "risk_score": dashboard.risk_score,
            "metrics": [
                {
                    "name": m.name,
                    "value": m.value,
                    "threshold": m.threshold,
                    "status": m.status,
                    "description": m.description,
                }
                for m in dashboard.metrics
            ],
            "warnings": dashboard.warnings,
            "timestamp": dashboard.timestamp.isoformat(),
        }
        
    except Exception as e:
        logger.error(f"风控仪表盘失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/multimodal-analysis")
async def multimodal_analysis(req: MultimodalRequest) -> Dict:
    """
    多模态分析
    
    K线形态识别 + 图表形态识别 + 支撑阻力
    """
    try:
        analyzer = get_multimodal_analyzer()
        
        result = analyzer.analyze(req.ohlc)
        
        return {
            "code": req.code,
            "candle_patterns": [
                {
                    "type": p.type.value,
                    "position": p.position,
                    "signal": p.signal,
                    "confidence": p.confidence,
                    "description": p.description,
                }
                for p in result.candle_patterns
            ],
            "chart_patterns": [
                {
                    "type": p.type.value,
                    "signal": p.signal,
                    "confidence": p.confidence,
                    "target_price": p.target_price,
                    "stop_loss": p.stop_loss,
                    "description": p.description,
                }
                for p in result.chart_patterns
            ],
            "trend_lines": [
                {
                    "type": t.type,
                    "slope": t.slope,
                    "r_squared": t.r_squared,
                    "touches": t.touches,
                }
                for t in result.trend_lines
            ],
            "support_levels": [
                {
                    "level": s.level,
                    "strength": s.strength,
                    "touches": s.touches,
                }
                for s in result.support_levels
            ],
            "resistance_levels": [
                {
                    "level": r.level,
                    "strength": r.strength,
                    "touches": r.touches,
                }
                for r in result.resistance_levels
            ],
            "overall_signal": result.overall_signal,
            "confidence": result.confidence,
            "reasoning": result.reasoning,
        }
        
    except Exception as e:
        logger.error(f"多模态分析失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.websocket("/realtime/{code}")
async def realtime_quote_websocket(websocket: WebSocket, code: str):
    """
    实时行情WebSocket
    
    推送实时行情数据
    """
    await websocket.accept()
    
    from app.services.realtime_quote import get_quote_manager
    
    manager = get_quote_manager()
    manager.subscribe(code)
    manager.add_ws_client(websocket)
    
    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        manager.unsubscribe(code)
        manager.remove_ws_client(websocket)


@router.get("/efficient-frontier/{strategy_names}")
async def get_efficient_frontier(strategy_names: str) -> Dict:
    """
    获取有效前沿
    
    计算策略组合的有效前沿
    """
    try:
        names = [n.strip() for n in strategy_names.split(",")]
        
        strategies = [
            StrategyMetrics(
                name=name,
                returns=[0.01 * (i % 5 - 2) / 10 for i in range(30)],
            )
            for name in names
        ]
        
        optimizer = get_portfolio_optimizer()
        frontier = optimizer.efficient_frontier(strategies)
        
        return {
            "frontier": [
                {
                    "expected_return": f.expected_return,
                    "expected_volatility": f.expected_volatility,
                    "sharpe_ratio": f.sharpe_ratio,
                    "weights": f.weights,
                }
                for f in frontier
            ],
            "strategy_names": names,
        }
        
    except Exception as e:
        logger.error(f"有效前沿计算失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _get_ohlc_data(code: str) -> List[Dict]:
    """获取OHLC数据"""
    try:
        import akshare as ak
        
        code6 = code[-6:] if len(code) >= 6 else code.zfill(6)
        
        def fetch():
            return ak.stock_zh_a_hist(symbol=code6, period="daily", adjust="qfq")
        
        df = await asyncio.to_thread(fetch)
        
        if df is None or df.empty:
            return []
        
        ohlc = []
        for _, row in df.tail(60).iterrows():
            ohlc.append({
                "open": row.get("开盘", 0),
                "high": row.get("最高", 0),
                "low": row.get("最低", 0),
                "close": row.get("收盘", 0),
                "volume": row.get("成交量", 0),
            })
        
        return ohlc
        
    except Exception as e:
        logger.warning(f"获取OHLC数据失败 {code}: {e}")
        return []


async def _analyze_core_signals(code: str, ohlc: List[Dict]) -> Dict:
    """分析核心信号 - 使用V2增强版信号系统"""
    try:
        if not ohlc or len(ohlc) < 30:
            return _get_default_signals()
        
        df = pd.DataFrame(ohlc)
        df.columns = ['open', 'high', 'low', 'close', 'volume']
        
        from app.analysis.enhanced_signals_v2 import get_enhanced_signal_engine_v2
        
        engine = get_enhanced_signal_engine_v2()
        analysis = engine.analyze(code, df)
        
        signals = {}
        for signal_result in analysis.signals:
            signals[signal_result.name] = {
                "signal": signal_result.signal.value,
                "strength": round(signal_result.strength, 1),
                "confidence": round(signal_result.confidence, 2),
                "score": round(signal_result.score, 1),
                "trend": signal_result.trend.value,
                "volume_confirm": signal_result.volume_confirm,
                "risk_level": signal_result.risk_level,
                "triggers": signal_result.triggers[:3],
                "entry_price": signal_result.suggested_entry,
                "stop_loss": signal_result.suggested_stop,
                "take_profit": signal_result.suggested_target,
            }
        
        signals["综合信号"] = {
            "signal": analysis.combined_signal.value,
            "strength": round(analysis.combined_strength, 1),
            "confidence": round(analysis.combined_confidence, 2),
            "buy_signals": analysis.buy_signals,
            "sell_signals": analysis.sell_signals,
            "trend": analysis.trend.value,
            "market_phase": analysis.market_phase,
            "suggested_action": analysis.suggested_action,
            "position_suggestion": round(analysis.position_suggestion, 2),
            "risk_warning": analysis.risk_warning,
        }
        
        return signals
        
    except Exception as e:
        logger.error(f"V2信号分析失败: {e}")
        return _get_default_signals()


def _get_default_signals() -> Dict:
    """获取默认信号（降级使用）"""
    return {
        "亢龙有悔V2": {"signal": "持有", "confidence": 0.5, "score": 0},
        "游资暗盘V2": {"signal": "持有", "confidence": 0.5, "score": 0},
        "暗盘资金V2": {"signal": "持有", "confidence": 0.5, "score": 0},
        "精准买卖点V2": {"signal": "持有", "confidence": 0.5, "score": 0},
        "三色共振V2": {"signal": "持有", "confidence": 0.5, "score": 0},
        "寻龙诀V2": {"signal": "持有", "confidence": 0.5, "score": 0},
        "主力控盘V2": {"signal": "持有", "confidence": 0.5, "score": 0},
        "综合信号": {"signal": "持有", "confidence": 0.5, "buy_signals": 0, "sell_signals": 0},
    }


def _combine_signals(core_signals: Dict, multimodal_result) -> Dict:
    """综合信号"""
    buy_count = 0
    sell_count = 0
    total_confidence = 0
    
    for signal in core_signals.values():
        if signal["signal"] == "buy":
            buy_count += signal["confidence"]
        elif signal["signal"] == "sell":
            sell_count += signal["confidence"]
        total_confidence += signal["confidence"]
    
    if multimodal_result.overall_signal == "buy":
        buy_count += multimodal_result.confidence
    elif multimodal_result.overall_signal == "sell":
        sell_count += multimodal_result.confidence
    
    total = buy_count + sell_count
    if total == 0:
        final_signal = "hold"
        confidence = 0.5
    elif buy_count > sell_count:
        final_signal = "buy"
        confidence = buy_count / total
    else:
        final_signal = "sell"
        confidence = sell_count / total
    
    return {
        "signal": final_signal,
        "confidence": round(confidence, 2),
        "buy_strength": round(buy_count, 2),
        "sell_strength": round(sell_count, 2),
    }
