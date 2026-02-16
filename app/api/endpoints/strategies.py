#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROX策略API端点
提供策略列表、执行、查询等功能
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import logging
from pathlib import Path

from app.rox_quant.strategy_engine import get_strategy_engine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/strategies", tags=["策略引擎"])

class StrategyExecutionRequest(BaseModel):
    strategy_name: str = Field(..., description="策略名称")
    params: Optional[Dict[str, Any]] = Field(default_factory=dict, description="策略参数")

class StrategyBacktestRequest(BaseModel):
    strategy_name: str = Field(..., description="策略名称")
    params: Optional[Dict[str, Any]] = Field(default_factory=dict, description="策略参数")

class StrategyResponse(BaseModel):
    success: bool
    message: str
    results: Dict[str, Any]

class StrategyInfo(BaseModel):
    name: str
    file: str
    description: str
    size: int
    lines: int
    dependencies: List[str]

@router.get("/list", response_model=List[Dict[str, Any]])
async def list_strategies():
    """获取所有策略列表"""
    try:
        engine = get_strategy_engine()
        strategies = engine.list_strategies()
        return strategies
    except Exception as e:
        logger.error(f"获取策略列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取策略列表失败: {str(e)}")

@router.get("/info/{strategy_name}", response_model=Dict[str, Any])
async def get_strategy_info(strategy_name: str):
    """获取策略详细信息"""
    try:
        engine = get_strategy_engine()
        info = engine.get_strategy_info(strategy_name)
        
        if not info:
            raise HTTPException(status_code=404, detail=f"策略 {strategy_name} 不存在")
        
        return info
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取策略信息失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取策略信息失败: {str(e)}")

@router.post("/execute", response_model=StrategyResponse)
async def execute_strategy(request: StrategyExecutionRequest):
    """执行策略"""
    try:
        engine = get_strategy_engine()
        result = engine.execute_strategy(request.strategy_name, request.params)
        
        return StrategyResponse(
            success=result['success'],
            message=result['message'],
            results=result
        )
    except Exception as e:
        logger.error(f"执行策略失败: {e}")
        raise HTTPException(status_code=500, detail=f"执行策略失败: {str(e)}")

@router.post("/backtest", response_model=StrategyResponse)
async def backtest_strategy(request: StrategyBacktestRequest):
    """回测策略"""
    try:
        engine = get_strategy_engine()
        result = engine.backtest_strategy(request.strategy_name, request.params)
        
        return StrategyResponse(
            success=result['success'],
            message=result.get('message', '回测完成'),
            results=result
        )
    except Exception as e:
        logger.error(f"回测策略失败: {e}")
        raise HTTPException(status_code=500, detail=f"回测策略失败: {str(e)}")

@router.get("/categories")
async def get_strategy_categories():
    """获取策略分类统计"""
    try:
        engine = get_strategy_engine()
        categories = engine.get_strategy_categories()
        
        return {
            'total': sum(categories.values()),
            'categories': categories
        }
    except Exception as e:
        logger.error(f"获取策略分类失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取策略分类失败: {str(e)}")

@router.get("/stats")
async def get_strategy_stats():
    """获取策略统计信息"""
    try:
        engine = get_strategy_engine()
        strategies = engine.list_strategies()
        
        total_size = sum(s['size'] for s in strategies)
        avg_size = total_size / len(strategies) if strategies else 0
        
        return {
            'total_strategies': len(strategies),
            'total_size': total_size,
            'average_size': avg_size,
            'python_files': len(strategies),
            'categories': engine.get_strategy_categories()
        }
    except Exception as e:
        logger.error(f"获取策略统计失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取策略统计失败: {str(e)}")

@router.get("/history")
async def get_execution_history(limit: int = 10):
    """获取执行历史"""
    try:
        engine = get_strategy_engine()
        history = engine.get_execution_history(limit)
        return history
    except Exception as e:
        logger.error(f"获取执行历史失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取执行历史失败: {str(e)}")

@router.delete("/history")
async def clear_execution_history():
    """清空执行历史"""
    try:
        engine = get_strategy_engine()
        engine.clear_execution_history()
        return {"message": "执行历史已清空"}
    except Exception as e:
        logger.error(f"清空执行历史失败: {e}")
        raise HTTPException(status_code=500, detail=f"清空执行历史失败: {str(e)}")

@router.get("/health")
async def strategy_engine_health():
    """策略引擎健康检查"""
    try:
        engine = get_strategy_engine()
        return {
            "status": "healthy",
            "strategies_count": len(engine.list_strategies()),
            "execution_history_count": len(engine.get_execution_history(100))
        }
    except Exception as e:
        logger.error(f"策略引擎健康检查失败: {e}")
        raise HTTPException(status_code=500, detail=f"策略引擎健康检查失败: {str(e)}")