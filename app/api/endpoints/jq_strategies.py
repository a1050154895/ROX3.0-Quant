#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
聚宽策略API端点
提供策略列表、执行、查询等功能
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import logging
from pathlib import Path

from app.rox_quant.jq_adapter import get_jq_adapter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/jq_strategies", tags=["聚宽策略"])

class StrategyExecutionRequest(BaseModel):
    strategy_name: str = Field(..., description="策略名称")
    params: Optional[Dict[str, Any]] = Field(default_factory=dict, description="策略参数")

class StrategyExecutionResponse(BaseModel):
    success: bool
    message: str
    results: Dict[str, Any]

class StrategyInfo(BaseModel):
    name: str
    file: str
    description: str
    size: int
    lines: int

@router.get("/list", response_model=List[Dict[str, Any]])
async def list_strategies():
    """获取所有聚宽策略列表"""
    try:
        adapter = get_jq_adapter()
        strategies = adapter.list_strategies()
        return strategies
    except Exception as e:
        logger.error(f"获取策略列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取策略列表失败: {str(e)}")

@router.get("/info/{strategy_name}", response_model=Dict[str, Any])
async def get_strategy_info(strategy_name: str):
    """获取策略详细信息"""
    try:
        adapter = get_jq_adapter()
        
        strategy_dir = Path(__file__).parent.parent.parent / "rox_quant" / "jq_strategies"
        strategy_file = strategy_dir / f"{strategy_name}.py"
        
        if not strategy_file.exists():
            raise HTTPException(status_code=404, detail=f"策略 {strategy_name} 不存在")
        
        info = adapter.get_strategy_info(str(strategy_file))
        return info
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取策略信息失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取策略信息失败: {str(e)}")

@router.post("/execute", response_model=StrategyExecutionResponse)
async def execute_strategy(request: StrategyExecutionRequest):
    """执行聚宽策略"""
    try:
        adapter = get_jq_adapter()
        
        strategy_dir = Path(__file__).parent.parent.parent / "rox_quant" / "jq_strategies"
        strategy_file = strategy_dir / f"{request.strategy_name}.py"
        
        if not strategy_file.exists():
            raise HTTPException(status_code=404, detail=f"策略 {request.strategy_name} 不存在")
        
        strategy_code = adapter.load_strategy(str(strategy_file))
        if strategy_code is None:
            raise HTTPException(status_code=500, detail="加载策略代码失败")
        
        result = adapter.execute_strategy(strategy_code, request.params)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"执行策略失败: {e}")
        raise HTTPException(status_code=500, detail=f"执行策略失败: {str(e)}")

@router.get("/categories")
async def get_strategy_categories():
    """获取策略分类统计"""
    try:
        adapter = get_jq_adapter()
        strategies = adapter.list_strategies()
        
        categories = {
            '小市值策略': 0,
            'ETF策略': 0,
            '打板策略': 0,
            '机器学习': 0,
            '价值投资': 0,
            '其他': 0
        }
        
        for strategy in strategies:
            name = strategy['name'].lower()
            if '小市值' in name or 'small' in name:
                categories['小市值策略'] += 1
            elif 'etf' in name:
                categories['ETF策略'] += 1
            elif '板' in name or '涨停' in name:
                categories['打板策略'] += 1
            elif '机器学习' in name or 'machine' in name or '学习' in name:
                categories['机器学习'] += 1
            elif '价值' in name or '投资' in name or '股息' in name:
                categories['价值投资'] += 1
            else:
                categories['其他'] += 1
        
        return {
            'total': len(strategies),
            'categories': categories
        }
    except Exception as e:
        logger.error(f"获取策略分类失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取策略分类失败: {str(e)}")

@router.get("/stats")
async def get_strategy_stats():
    """获取策略统计信息"""
    try:
        adapter = get_jq_adapter()
        strategies = adapter.list_strategies()
        
        total_size = sum(s['size'] for s in strategies)
        avg_size = total_size / len(strategies) if strategies else 0
        
        return {
            'total_strategies': len(strategies),
            'total_size': total_size,
            'average_size': avg_size,
            'python_files': len(strategies)
        }
    except Exception as e:
        logger.error(f"获取策略统计失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取策略统计失败: {str(e)}")