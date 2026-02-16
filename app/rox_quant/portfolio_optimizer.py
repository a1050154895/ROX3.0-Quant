#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略组合优化器
基于马科维茨均值方差模型优化策略组合

功能：
1. 均值方差优化
2. 风险平价
3. 最大夏普比率
4. 最小相关性
5. Black-Litterman模型
"""

import logging
import math
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class OptimizationObjective(Enum):
    """优化目标"""
    MAX_SHARPE = "max_sharpe"
    MIN_VOLATILITY = "min_volatility"
    MAX_RETURN = "max_return"
    RISK_PARITY = "risk_parity"
    MIN_CORRELATION = "min_correlation"


@dataclass
class StrategyMetrics:
    """策略指标"""
    name: str
    returns: List[float]
    mean_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    
    def __post_init__(self):
        if self.returns:
            self.mean_return = np.mean(self.returns)
            self.volatility = np.std(self.returns) * np.sqrt(252) if len(self.returns) > 1 else 0
            self.sharpe_ratio = self.mean_return / self.volatility if self.volatility > 0 else 0
            
            cumulative = np.cumprod(1 + np.array(self.returns))
            peak = np.maximum.accumulate(cumulative)
            drawdown = (peak - cumulative) / peak
            self.max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0


@dataclass
class PortfolioResult:
    """组合优化结果"""
    weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    diversification_ratio: float
    correlation_matrix: np.ndarray
    strategy_names: List[str]


class MeanVarianceOptimizer:
    """
    马科维茨均值方差优化器
    
    功能：
    1. 有效前沿计算
    2. 最大夏普比率组合
    3. 最小波动率组合
    4. 风险平价组合
    """
    
    def __init__(self, risk_free_rate: float = 0.03):
        self.risk_free_rate = risk_free_rate
    
    def optimize(
        self,
        strategies: List[StrategyMetrics],
        objective: OptimizationObjective = OptimizationObjective.MAX_SHARPE,
        constraints: Dict[str, Any] = None,
    ) -> PortfolioResult:
        """
        执行组合优化
        
        Args:
            strategies: 策略指标列表
            objective: 优化目标
            constraints: 约束条件
        """
        if not strategies:
            return self._empty_result()
        
        n = len(strategies)
        returns = np.array([s.mean_return for s in strategies])
        
        all_returns = []
        min_len = min(len(s.returns) for s in strategies)
        for s in strategies:
            all_returns.append(s.returns[:min_len])
        returns_matrix = np.array(all_returns).T
        
        cov_matrix = np.cov(returns_matrix.T) if returns_matrix.shape[1] > 1 else np.eye(n)
        
        if objective == OptimizationObjective.MAX_SHARPE:
            weights = self._max_sharpe(returns, cov_matrix)
        elif objective == OptimizationObjective.MIN_VOLATILITY:
            weights = self._min_volatility(cov_matrix)
        elif objective == OptimizationObjective.RISK_PARITY:
            weights = self._risk_parity(cov_matrix)
        elif objective == OptimizationObjective.MIN_CORRELATION:
            weights = self._min_correlation(returns_matrix)
        else:
            weights = self._max_sharpe(returns, cov_matrix)
        
        if constraints:
            weights = self._apply_constraints(weights, constraints)
        
        weights = weights / np.sum(weights) if np.sum(weights) > 0 else np.ones(n) / n
        
        expected_return = np.dot(weights, returns)
        expected_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = (expected_return - self.risk_free_rate) / expected_vol if expected_vol > 0 else 0
        
        div_ratio = self._diversification_ratio(weights, cov_matrix)
        
        corr_matrix = self._cov_to_corr(cov_matrix)
        
        return PortfolioResult(
            weights={s.name: w for s, w in zip(strategies, weights)},
            expected_return=expected_return,
            expected_volatility=expected_vol,
            sharpe_ratio=sharpe,
            diversification_ratio=div_ratio,
            correlation_matrix=corr_matrix,
            strategy_names=[s.name for s in strategies],
        )
    
    def _max_sharpe(self, returns: np.ndarray, cov_matrix: np.ndarray) -> np.ndarray:
        """最大夏普比率"""
        n = len(returns)
        
        inv_cov = np.linalg.pinv(cov_matrix + np.eye(n) * 1e-8)
        weights = np.dot(inv_cov, returns - self.risk_free_rate)
        weights = np.maximum(weights, 0)
        
        return weights / np.sum(weights) if np.sum(weights) > 0 else np.ones(n) / n
    
    def _min_volatility(self, cov_matrix: np.ndarray) -> np.ndarray:
        """最小波动率"""
        n = cov_matrix.shape[0]
        
        inv_cov = np.linalg.pinv(cov_matrix + np.eye(n) * 1e-8)
        ones = np.ones(n)
        weights = np.dot(inv_cov, ones)
        weights = np.maximum(weights, 0)
        
        return weights / np.sum(weights) if np.sum(weights) > 0 else np.ones(n) / n
    
    def _risk_parity(self, cov_matrix: np.ndarray) -> np.ndarray:
        """风险平价"""
        n = cov_matrix.shape[0]
        
        vols = np.sqrt(np.diag(cov_matrix))
        weights = 1 / vols
        weights = np.maximum(weights, 0)
        
        return weights / np.sum(weights)
    
    def _min_correlation(self, returns_matrix: np.ndarray) -> np.ndarray:
        """最小相关性"""
        n = returns_matrix.shape[1]
        
        corr_matrix = np.corrcoef(returns_matrix.T)
        avg_corr = np.mean(np.abs(corr_matrix - np.eye(n)), axis=1)
        
        weights = 1 / (avg_corr + 0.1)
        weights = np.maximum(weights, 0)
        
        return weights / np.sum(weights)
    
    def _apply_constraints(self, weights: np.ndarray, constraints: Dict) -> np.ndarray:
        """应用约束"""
        min_weight = constraints.get("min_weight", 0)
        max_weight = constraints.get("max_weight", 1)
        
        weights = np.clip(weights, min_weight, max_weight)
        
        return weights
    
    def _diversification_ratio(self, weights: np.ndarray, cov_matrix: np.ndarray) -> float:
        """计算分散化比率"""
        vols = np.sqrt(np.diag(cov_matrix))
        weighted_vol = np.dot(weights, vols)
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        return weighted_vol / portfolio_vol if portfolio_vol > 0 else 1
    
    def _cov_to_corr(self, cov_matrix: np.ndarray) -> np.ndarray:
        """协方差转相关系数"""
        vols = np.sqrt(np.diag(cov_matrix))
        outer_vols = np.outer(vols, vols)
        return cov_matrix / outer_vols
    
    def _empty_result(self) -> PortfolioResult:
        return PortfolioResult(
            weights={},
            expected_return=0,
            expected_volatility=0,
            sharpe_ratio=0,
            diversification_ratio=1,
            correlation_matrix=np.array([]),
            strategy_names=[],
        )
    
    def efficient_frontier(
        self,
        strategies: List[StrategyMetrics],
        n_points: int = 20,
    ) -> List[PortfolioResult]:
        """计算有效前沿"""
        if not strategies:
            return []
        
        results = []
        min_vol_result = self.optimize(strategies, OptimizationObjective.MIN_VOLATILITY)
        max_sharpe_result = self.optimize(strategies, OptimizationObjective.MAX_SHARPE)
        
        min_ret = min_vol_result.expected_return
        max_ret = max_sharpe_result.expected_return
        
        target_returns = np.linspace(min_ret, max_ret, n_points)
        
        for target_ret in target_returns:
            result = self._optimize_for_target_return(strategies, target_ret)
            if result:
                results.append(result)
        
        return results
    
    def _optimize_for_target_return(
        self,
        strategies: List[StrategyMetrics],
        target_return: float,
    ) -> Optional[PortfolioResult]:
        """优化指定目标收益"""
        n = len(strategies)
        returns = np.array([s.mean_return for s in strategies])
        
        min_len = min(len(s.returns) for s in strategies)
        all_returns = [s.returns[:min_len] for s in strategies]
        returns_matrix = np.array(all_returns).T
        cov_matrix = np.cov(returns_matrix.T) if returns_matrix.shape[1] > 1 else np.eye(n)
        
        weights = self._min_volatility(cov_matrix)
        
        current_return = np.dot(weights, returns)
        if abs(current_return - target_return) < 0.01:
            expected_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe = (current_return - self.risk_free_rate) / expected_vol if expected_vol > 0 else 0
            
            return PortfolioResult(
                weights={s.name: w for s, w in zip(strategies, weights)},
                expected_return=current_return,
                expected_volatility=expected_vol,
                sharpe_ratio=sharpe,
                diversification_ratio=self._diversification_ratio(weights, cov_matrix),
                correlation_matrix=self._cov_to_corr(cov_matrix),
                strategy_names=[s.name for s in strategies],
            )
        
        return None


class BlackLittermanModel:
    """
    Black-Litterman模型
    
    结合市场均衡收益和投资者观点
    """
    
    def __init__(self, tau: float = 0.05):
        self.tau = tau
    
    def optimize(
        self,
        strategies: List[StrategyMetrics],
        market_caps: Dict[str, float],
        views: Dict[str, float],
        view_confidences: Dict[str, float] = None,
    ) -> PortfolioResult:
        """
        Black-Litterman优化
        
        Args:
            strategies: 策略指标
            market_caps: 市值权重
            views: 投资者观点（预期收益）
            view_confidences: 观点置信度
        """
        if not strategies:
            return MeanVarianceOptimizer()._empty_result()
        
        n = len(strategies)
        names = [s.name for s in strategies]
        
        market_weights = np.array([market_caps.get(name, 1/n) for name in names])
        market_weights = market_weights / np.sum(market_weights)
        
        min_len = min(len(s.returns) for s in strategies)
        all_returns = [s.returns[:min_len] for s in strategies]
        returns_matrix = np.array(all_returns).T
        cov_matrix = np.cov(returns_matrix.T) if returns_matrix.shape[1] > 1 else np.eye(n)
        
        risk_aversion = 3.0
        equilibrium_returns = risk_aversion * np.dot(cov_matrix, market_weights)
        
        P = np.zeros((len(views), n))
        Q = np.zeros(len(views))
        omega = np.zeros((len(views), len(views)))
        
        for i, (name, view) in enumerate(views.items()):
            if name in names:
                idx = names.index(name)
                P[i, idx] = 1
                Q[i] = view
                confidence = view_confidences.get(name, 0.5) if view_confidences else 0.5
                omega[i, i] = 1 / confidence
        
        tau_cov = self.tau * cov_matrix
        
        try:
            M1 = np.linalg.pinv(tau_cov)
            M2 = np.dot(P.T, np.linalg.pinv(omega))
            combined_returns = np.linalg.pinv(M1 + np.dot(M2, P))
            combined_returns = np.dot(combined_returns, np.dot(M1, equilibrium_returns) + np.dot(M2, Q))
        except:
            combined_returns = equilibrium_returns
        
        inv_cov = np.linalg.pinv(cov_matrix + np.eye(n) * 1e-8)
        weights = np.dot(inv_cov, combined_returns)
        weights = np.maximum(weights, 0)
        weights = weights / np.sum(weights)
        
        expected_return = np.dot(weights, combined_returns)
        expected_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        return PortfolioResult(
            weights={name: w for name, w in zip(names, weights)},
            expected_return=expected_return,
            expected_volatility=expected_vol,
            sharpe_ratio=(expected_return - 0.03) / expected_vol if expected_vol > 0 else 0,
            diversification_ratio=1,
            correlation_matrix=np.corrcoef(returns_matrix.T) if returns_matrix.shape[1] > 1 else np.eye(n),
            strategy_names=names,
        )


class PortfolioRebalancer:
    """
    组合再平衡器
    """
    
    def __init__(self, threshold: float = 0.05):
        self.threshold = threshold
    
    def need_rebalance(
        self,
        target_weights: Dict[str, float],
        current_weights: Dict[str, float],
    ) -> bool:
        """判断是否需要再平衡"""
        for name in target_weights:
            target = target_weights.get(name, 0)
            current = current_weights.get(name, 0)
            if abs(target - current) > self.threshold:
                return True
        return False
    
    def calculate_trades(
        self,
        target_weights: Dict[str, float],
        current_weights: Dict[str, float],
        total_value: float,
    ) -> Dict[str, float]:
        """计算交易金额"""
        trades = {}
        for name in set(target_weights.keys()) | set(current_weights.keys()):
            target = target_weights.get(name, 0)
            current = current_weights.get(name, 0)
            trade = (target - current) * total_value
            trades[name] = trade
        return trades


_portfolio_optimizer = None


def get_portfolio_optimizer() -> MeanVarianceOptimizer:
    """获取组合优化器单例"""
    global _portfolio_optimizer
    if _portfolio_optimizer is None:
        _portfolio_optimizer = MeanVarianceOptimizer()
    return _portfolio_optimizer
