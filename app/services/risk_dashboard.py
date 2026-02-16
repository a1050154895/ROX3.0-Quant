#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
风控仪表盘系统
实时监控账户风险状态

功能：
1. 实时风险指标监控
2. 风险预警
3. 风险历史追踪
4. 风险报告生成
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """风险等级"""
    SAFE = "安全"
    LOW = "低风险"
    MEDIUM = "中风险"
    HIGH = "高风险"
    CRITICAL = "危险"


@dataclass
class RiskMetric:
    """风险指标"""
    name: str
    value: float
    threshold: float
    status: str
    description: str
    trend: str


@dataclass
class RiskDashboard:
    """风控仪表盘"""
    total_value: float
    cash: float
    position_value: float
    leverage: float
    
    daily_pnl: float
    daily_pnl_pct: float
    
    max_drawdown: float
    current_drawdown: float
    
    var_95: float
    cvar_95: float
    
    sharpe_ratio: float
    sortino_ratio: float
    
    position_count: int
    concentration: float
    
    risk_level: RiskLevel
    risk_score: float
    
    metrics: List[RiskMetric]
    warnings: List[str]
    
    timestamp: datetime


class RiskMonitor:
    """
    风险监控器
    
    实时监控账户风险状态
    """
    
    def __init__(self):
        self._history: List[RiskDashboard] = []
        self._alerts: List[Dict] = []
    
    def calculate_dashboard(
        self,
        positions: List[Dict],
        account: Dict,
        price_history: List[float] = None,
    ) -> RiskDashboard:
        """计算风控仪表盘"""
        total_value = account.get('total_value', 0)
        cash = account.get('cash', 0)
        position_value = total_value - cash
        
        leverage = position_value / cash if cash > 0 else 0
        
        daily_pnl = account.get('daily_pnl', 0)
        daily_pnl_pct = daily_pnl / total_value * 100 if total_value > 0 else 0
        
        max_drawdown = self._calculate_max_drawdown(price_history) if price_history else 0
        current_drawdown = self._calculate_current_drawdown(price_history) if price_history else 0
        
        returns = self._calculate_returns(price_history) if price_history else []
        var_95 = self._calculate_var(returns, 0.95) if returns else 0
        cvar_95 = self._calculate_cvar(returns, 0.95) if returns else 0
        
        sharpe = self._calculate_sharpe(returns) if returns else 0
        sortino = self._calculate_sortino(returns) if returns else 0
        
        position_count = len(positions)
        concentration = self._calculate_concentration(positions)
        
        risk_score = self._calculate_risk_score(
            leverage, current_drawdown, concentration, var_95
        )
        risk_level = self._determine_risk_level(risk_score)
        
        metrics = self._build_metrics(
            leverage, current_drawdown, concentration, var_95, sharpe
        )
        
        warnings = self._generate_warnings(
            leverage, current_drawdown, concentration, risk_level
        )
        
        dashboard = RiskDashboard(
            total_value=total_value,
            cash=cash,
            position_value=position_value,
            leverage=leverage,
            daily_pnl=daily_pnl,
            daily_pnl_pct=daily_pnl_pct,
            max_drawdown=max_drawdown,
            current_drawdown=current_drawdown,
            var_95=var_95,
            cvar_95=cvar_95,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            position_count=position_count,
            concentration=concentration,
            risk_level=risk_level,
            risk_score=risk_score,
            metrics=metrics,
            warnings=warnings,
            timestamp=datetime.now(),
        )
        
        self._history.append(dashboard)
        if len(self._history) > 100:
            self._history = self._history[-100:]
        
        return dashboard
    
    def _calculate_max_drawdown(self, prices: List[float]) -> float:
        """计算最大回撤"""
        if not prices:
            return 0
        
        peak = prices[0]
        max_dd = 0
        
        for price in prices:
            if price > peak:
                peak = price
            dd = (peak - price) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
        
        return max_dd * 100
    
    def _calculate_current_drawdown(self, prices: List[float]) -> float:
        """计算当前回撤"""
        if not prices:
            return 0
        
        peak = max(prices)
        current = prices[-1]
        
        return (peak - current) / peak * 100 if peak > 0 else 0
    
    def _calculate_returns(self, prices: List[float]) -> List[float]:
        """计算收益率"""
        if len(prices) < 2:
            return []
        
        return [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
    
    def _calculate_var(self, returns: List[float], confidence: float) -> float:
        """计算VaR"""
        if not returns:
            return 0
        
        return np.percentile(returns, (1 - confidence) * 100) * 100
    
    def _calculate_cvar(self, returns: List[float], confidence: float) -> float:
        """计算CVaR"""
        if not returns:
            return 0
        
        var = self._calculate_var(returns, confidence)
        tail_returns = [r for r in returns if r * 100 <= var]
        
        return np.mean(tail_returns) * 100 if tail_returns else var
    
    def _calculate_sharpe(self, returns: List[float], rf: float = 0.03) -> float:
        """计算夏普比率"""
        if not returns:
            return 0
        
        mean_ret = np.mean(returns) * 252
        std_ret = np.std(returns) * np.sqrt(252)
        
        return (mean_ret - rf) / std_ret if std_ret > 0 else 0
    
    def _calculate_sortino(self, returns: List[float], rf: float = 0.03) -> float:
        """计算索提诺比率"""
        if not returns:
            return 0
        
        mean_ret = np.mean(returns) * 252
        downside = [r for r in returns if r < 0]
        downside_std = np.std(downside) * np.sqrt(252) if downside else 0
        
        return (mean_ret - rf) / downside_std if downside_std > 0 else 0
    
    def _calculate_concentration(self, positions: List[Dict]) -> float:
        """计算持仓集中度"""
        if not positions:
            return 0
        
        values = [p.get('value', 0) for p in positions]
        total = sum(values)
        
        if total == 0:
            return 0
        
        weights = [v / total for v in values]
        hhi = sum(w ** 2 for w in weights)
        
        return hhi * 100
    
    def _calculate_risk_score(
        self,
        leverage: float,
        drawdown: float,
        concentration: float,
        var: float,
    ) -> float:
        """计算风险分数"""
        score = 0
        
        if leverage > 3:
            score += 30
        elif leverage > 2:
            score += 20
        elif leverage > 1:
            score += 10
        
        if drawdown > 20:
            score += 30
        elif drawdown > 10:
            score += 20
        elif drawdown > 5:
            score += 10
        
        if concentration > 50:
            score += 20
        elif concentration > 30:
            score += 10
        
        if abs(var) > 5:
            score += 20
        elif abs(var) > 3:
            score += 10
        
        return min(100, score)
    
    def _determine_risk_level(self, score: float) -> RiskLevel:
        """确定风险等级"""
        if score < 20:
            return RiskLevel.SAFE
        elif score < 40:
            return RiskLevel.LOW
        elif score < 60:
            return RiskLevel.MEDIUM
        elif score < 80:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL
    
    def _build_metrics(
        self,
        leverage: float,
        drawdown: float,
        concentration: float,
        var: float,
        sharpe: float,
    ) -> List[RiskMetric]:
        """构建指标列表"""
        metrics = []
        
        metrics.append(RiskMetric(
            name="杠杆率",
            value=leverage,
            threshold=2.0,
            status="正常" if leverage < 2 else "警告",
            description=f"当前杠杆{leverage:.2f}倍",
            trend="up" if leverage > 1.5 else "stable",
        ))
        
        metrics.append(RiskMetric(
            name="当前回撤",
            value=drawdown,
            threshold=10.0,
            status="正常" if drawdown < 10 else "警告",
            description=f"当前回撤{drawdown:.1f}%",
            trend="up" if drawdown > 5 else "stable",
        ))
        
        metrics.append(RiskMetric(
            name="持仓集中度",
            value=concentration,
            threshold=40.0,
            status="正常" if concentration < 40 else "警告",
            description=f"HHI指数{concentration:.1f}",
            trend="up" if concentration > 30 else "stable",
        ))
        
        metrics.append(RiskMetric(
            name="VaR(95%)",
            value=var,
            threshold=-3.0,
            status="正常" if var > -3 else "警告",
            description=f"95%置信下最大损失{var:.1f}%",
            trend="down" if var < -2 else "stable",
        ))
        
        metrics.append(RiskMetric(
            name="夏普比率",
            value=sharpe,
            threshold=1.0,
            status="正常" if sharpe > 1 else "警告",
            description=f"风险调整收益{sharpe:.2f}",
            trend="up" if sharpe > 1.5 else "stable",
        ))
        
        return metrics
    
    def _generate_warnings(
        self,
        leverage: float,
        drawdown: float,
        concentration: float,
        risk_level: RiskLevel,
    ) -> List[str]:
        """生成警告信息"""
        warnings = []
        
        if leverage > 2:
            warnings.append(f"杠杆率过高({leverage:.2f}倍)，建议降低仓位")
        
        if drawdown > 10:
            warnings.append(f"回撤较大({drawdown:.1f}%)，注意风险控制")
        
        if concentration > 40:
            warnings.append(f"持仓过于集中(HHI={concentration:.1f})，建议分散投资")
        
        if risk_level == RiskLevel.HIGH:
            warnings.append("风险等级较高，建议减少敞口")
        elif risk_level == RiskLevel.CRITICAL:
            warnings.append("风险等级危险，建议立即降低仓位！")
        
        return warnings
    
    def get_history(self, days: int = 7) -> List[RiskDashboard]:
        """获取历史记录"""
        cutoff = datetime.now() - timedelta(days=days)
        return [d for d in self._history if d.timestamp > cutoff]
    
    def get_alerts(self) -> List[Dict]:
        """获取告警记录"""
        return self._alerts.copy()


_risk_monitor = None


def get_risk_monitor() -> RiskMonitor:
    """获取风险监控器单例"""
    global _risk_monitor
    if _risk_monitor is None:
        _risk_monitor = RiskMonitor()
    return _risk_monitor
