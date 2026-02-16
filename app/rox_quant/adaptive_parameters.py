#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自适应参数调整机制

根据市场环境自动调整交易信号的参数，提高信号在不同市场条件下的适应性
"""

import logging
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MarketPhase(Enum):
    """市场阶段"""
    BULL = "牛市"
    BEAR = "熊市"
    CONSOLIDATION = "震荡市"
    VOLATILE = "波动市"


class AdaptiveParameterManager:
    """
    自适应参数管理器
    
    根据市场环境自动调整交易信号的参数
    """
    
    def __init__(self):
        self.market_phase = MarketPhase.CONSOLIDATION
        self.parameter_history: Dict[str, List[Dict]] = {}
        self.adjustment_threshold = 0.1  # 参数调整阈值
        self.max_adjustment = 0.3  # 最大调整幅度
        self.min_history_length = 20  # 最小历史数据长度
    
    def analyze_market_phase(self, df: pd.DataFrame) -> MarketPhase:
        """
        分析当前市场阶段
        
        Args:
            df: 价格数据
        
        Returns:
            市场阶段
        """
        if df.empty or len(df) < self.min_history_length:
            return MarketPhase.CONSOLIDATION
        
        # 计算市场指标
        returns = df['close'].pct_change() * 100
        volatility = returns.std()
        trend = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0] * 100
        
        # 确定市场阶段
        if trend > 10 and volatility < 2:
            return MarketPhase.BULL
        elif trend < -10 and volatility < 2:
            return MarketPhase.BEAR
        elif volatility > 3:
            return MarketPhase.VOLATILE
        else:
            return MarketPhase.CONSOLIDATION
    
    def adjust_parameters(self, signal_name: str, params: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
        """
        调整信号参数
        
        Args:
            signal_name: 信号名称
            params: 当前参数
            df: 价格数据
        
        Returns:
            调整后的参数
        """
        # 分析市场阶段
        current_phase = self.analyze_market_phase(df)
        self.market_phase = current_phase
        
        # 根据市场阶段调整参数
        adjusted_params = self._adjust_based_on_phase(signal_name, params, current_phase)
        
        # 记录参数调整历史
        self._record_parameter_history(signal_name, params, adjusted_params, current_phase)
        
        return adjusted_params
    
    def _adjust_based_on_phase(self, signal_name: str, params: Dict[str, Any], phase: MarketPhase) -> Dict[str, Any]:
        """
        根据市场阶段调整参数
        
        Args:
            signal_name: 信号名称
            params: 当前参数
            phase: 市场阶段
        
        Returns:
            调整后的参数
        """
        adjusted_params = params.copy()
        
        if signal_name == "亢龙有悔":
            if phase == MarketPhase.BULL:
                # 牛市中，放宽买入条件
                adjusted_params['volume_ratio_threshold'] = max(1.5, params.get('volume_ratio_threshold', 1.9) * 0.8)
                adjusted_params['breakout_threshold'] = max(0.01, params.get('breakout_threshold', 0.02) * 0.9)
            elif phase == MarketPhase.BEAR:
                # 熊市中，严格买入条件
                adjusted_params['volume_ratio_threshold'] = min(2.5, params.get('volume_ratio_threshold', 1.9) * 1.2)
                adjusted_params['breakout_threshold'] = min(0.05, params.get('breakout_threshold', 0.02) * 1.3)
            elif phase == MarketPhase.VOLATILE:
                # 波动市中，增加止损
                adjusted_params['stop_loss_pct'] = min(0.15, params.get('stop_loss_pct', 0.1) * 1.2)
        
        elif signal_name == "游资暗盘":
            if phase == MarketPhase.BULL:
                # 牛市中，缩短周期
                adjusted_params['ema_short_period'] = max(2, params.get('ema_short_period', 2))
                adjusted_params['ema_long_period'] = max(30, params.get('ema_long_period', 42) * 0.8)
            elif phase == MarketPhase.BEAR:
                # 熊市中，延长周期
                adjusted_params['ema_short_period'] = max(3, params.get('ema_short_period', 2) * 1.2)
                adjusted_params['ema_long_period'] = min(60, params.get('ema_long_period', 42) * 1.3)
        
        elif signal_name == "精准买卖点":
            if phase == MarketPhase.VOLATILE:
                # 波动市中，增加zigzag阈值
                adjusted_params['zigzag_pct'] = min(8, params.get('zigzag_pct', 5) * 1.3)
            elif phase == MarketPhase.CONSOLIDATION:
                # 震荡市中，减小zigzag阈值
                adjusted_params['zigzag_pct'] = max(3, params.get('zigzag_pct', 5) * 0.8)
        
        elif signal_name == "三色共振":
            if phase == MarketPhase.BULL:
                # 牛市中，缩短周期
                adjusted_params['main_force_period'] = max(20, params.get('main_force_period', 35) * 0.7)
                adjusted_params['hot_money_period'] = max(30, params.get('hot_money_period', 42) * 0.7)
            elif phase == MarketPhase.BEAR:
                # 熊市中，延长周期
                adjusted_params['main_force_period'] = min(50, params.get('main_force_period', 35) * 1.3)
                adjusted_params['hot_money_period'] = min(60, params.get('hot_money_period', 42) * 1.3)
        
        elif signal_name == "寻龙诀":
            if phase == MarketPhase.BULL:
                # 牛市中，降低涨停板确认要求
                adjusted_params['limit_up_threshold'] = max(9.5, params.get('limit_up_threshold', 9.9) * 0.95)
            elif phase == MarketPhase.BEAR:
                # 熊市中，提高涨停板确认要求
                adjusted_params['limit_up_threshold'] = min(9.95, params.get('limit_up_threshold', 9.9) * 1.01)
        
        elif signal_name == "主力控盘":
            if phase == MarketPhase.VOLATILE:
                # 波动市中，提高控盘度阈值
                adjusted_params['control_threshold'] = min(0.8, params.get('control_threshold', 0.6) * 1.2)
            elif phase == MarketPhase.CONSOLIDATION:
                # 震荡市中，降低控盘度阈值
                adjusted_params['control_threshold'] = max(0.4, params.get('control_threshold', 0.6) * 0.8)
        
        return adjusted_params
    
    def _record_parameter_history(self, signal_name: str, original_params: Dict[str, Any], adjusted_params: Dict[str, Any], phase: MarketPhase):
        """
        记录参数调整历史
        
        Args:
            signal_name: 信号名称
            original_params: 原始参数
            adjusted_params: 调整后的参数
            phase: 市场阶段
        """
        if signal_name not in self.parameter_history:
            self.parameter_history[signal_name] = []
        
        self.parameter_history[signal_name].append({
            'timestamp': datetime.now(),
            'market_phase': phase.value,
            'original_params': original_params,
            'adjusted_params': adjusted_params,
            'adjustment_ratio': {k: abs(adjusted_params[k] - original_params.get(k, 0)) / max(1e-6, abs(original_params.get(k, 1))) 
                              for k in adjusted_params if k in original_params}
        })
        
        # 限制历史记录长度
        if len(self.parameter_history[signal_name]) > 100:
            self.parameter_history[signal_name] = self.parameter_history[signal_name][-100:]
    
    def get_parameter_history(self, signal_name: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        获取参数调整历史
        
        Args:
            signal_name: 信号名称
            limit: 限制返回数量
        
        Returns:
            参数调整历史
        """
        if signal_name not in self.parameter_history:
            return []
        
        return self.parameter_history[signal_name][-limit:]
    
    def get_current_market_phase(self) -> MarketPhase:
        """
        获取当前市场阶段
        
        Returns:
            当前市场阶段
        """
        return self.market_phase
    
    def get_adjustment_stats(self, signal_name: str) -> Dict[str, Any]:
        """
        获取参数调整统计信息
        
        Args:
            signal_name: 信号名称
        
        Returns:
            统计信息
        """
        if signal_name not in self.parameter_history:
            return {}
        
        history = self.parameter_history[signal_name]
        if not history:
            return {}
        
        # 计算调整统计
        adjustments = []
        for record in history:
            for param, ratio in record['adjustment_ratio'].items():
                adjustments.append({
                    'param': param,
                    'ratio': ratio,
                    'market_phase': record['market_phase']
                })
        
        # 按参数和市场阶段分组统计
        stats = {}
        for adj in adjustments:
            param = adj['param']
            phase = adj['market_phase']
            ratio = adj['ratio']
            
            if param not in stats:
                stats[param] = {}
            if phase not in stats[param]:
                stats[param][phase] = []
            stats[param][phase].append(ratio)
        
        # 计算平均值
        for param in stats:
            for phase in stats[param]:
                stats[param][phase] = {
                    'count': len(stats[param][phase]),
                    'avg_ratio': np.mean(stats[param][phase]),
                    'max_ratio': np.max(stats[param][phase]),
                    'min_ratio': np.min(stats[param][phase])
                }
        
        return stats


# 全局自适应参数管理器实例
_adaptive_param_manager = None


def get_adaptive_param_manager() -> AdaptiveParameterManager:
    """
    获取自适应参数管理器实例
    
    Returns:
        自适应参数管理器实例
    """
    global _adaptive_param_manager
    if _adaptive_param_manager is None:
        _adaptive_param_manager = AdaptiveParameterManager()
    return _adaptive_param_manager


def adjust_signal_parameters(signal_name: str, params: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
    """
    调整信号参数
    
    Args:
        signal_name: 信号名称
        params: 当前参数
        df: 价格数据
    
    Returns:
        调整后的参数
    """
    manager = get_adaptive_param_manager()
    return manager.adjust_parameters(signal_name, params, df)


def get_market_phase(df: pd.DataFrame) -> MarketPhase:
    """
    获取市场阶段
    
    Args:
        df: 价格数据
    
    Returns:
        市场阶段
    """
    manager = get_adaptive_param_manager()
    return manager.analyze_market_phase(df)


def get_parameter_adjustment_history(signal_name: str, limit: int = 20) -> List[Dict[str, Any]]:
    """
    获取参数调整历史
    
    Args:
        signal_name: 信号名称
        limit: 限制返回数量
    
    Returns:
        参数调整历史
    """
    manager = get_adaptive_param_manager()
    return manager.get_parameter_history(signal_name, limit)


def get_parameter_adjustment_stats(signal_name: str) -> Dict[str, Any]:
    """
    获取参数调整统计信息
    
    Args:
        signal_name: 信号名称
    
    Returns:
        统计信息
    """
    manager = get_adaptive_param_manager()
    return manager.get_adjustment_stats(signal_name)
