#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROX 3.0 信号验证与回测系统
用于验证和评估7大核心交易信号的准确性和实效性

功能：
1. 信号历史追踪
2. 准确率统计
3. 回测验证
4. 信号衰减分析
5. 参数优化
"""

import logging
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from collections import deque
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SignalRecord:
    """信号记录"""
    signal_id: str
    signal_name: str
    code: str
    timestamp: str
    signal_type: str
    strength: float
    confidence: float
    score: float
    entry_price: float
    suggested_stop: Optional[float]
    suggested_target: Optional[float]
    triggers: List[str]
    
    actual_exit_price: Optional[float] = None
    actual_exit_time: Optional[str] = None
    holding_days: int = 0
    max_profit: float = 0.0
    max_drawdown: float = 0.0
    final_return: float = 0.0
    is_correct: Optional[bool] = None
    verified: bool = False


@dataclass
class SignalPerformance:
    """信号表现统计"""
    signal_name: str
    total_signals: int = 0
    verified_signals: int = 0
    correct_signals: int = 0
    total_return: float = 0.0
    avg_return: float = 0.0
    win_rate: float = 0.0
    avg_holding_days: float = 0.0
    avg_max_profit: float = 0.0
    avg_max_drawdown: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    
    recent_accuracy: float = 0.5
    recent_returns: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return asdict(self)


class SignalValidator:
    """
    信号验证器
    
    功能：
    1. 记录信号历史
    2. 验证信号结果
    3. 计算准确率
    4. 分析信号质量
    """
    
    def __init__(self, storage_dir: str = "data/signal_validation"):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        
        self.signal_records: Dict[str, SignalRecord] = {}
        self.performances: Dict[str, SignalPerformance] = {}
        self.recent_returns: Dict[str, deque] = {}
        
        self._load()
    
    def record_signal(self, 
                      signal_name: str,
                      code: str,
                      signal_type: str,
                      strength: float,
                      confidence: float,
                      score: float,
                      entry_price: float,
                      suggested_stop: float = None,
                      suggested_target: float = None,
                      triggers: List[str] = None) -> str:
        """记录信号"""
        signal_id = f"{signal_name}_{code}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        record = SignalRecord(
            signal_id=signal_id,
            signal_name=signal_name,
            code=code,
            timestamp=datetime.now().isoformat(),
            signal_type=signal_type,
            strength=strength,
            confidence=confidence,
            score=score,
            entry_price=entry_price,
            suggested_stop=suggested_stop,
            suggested_target=suggested_target,
            triggers=triggers or []
        )
        
        self.signal_records[signal_id] = record
        
        if signal_name not in self.performances:
            self.performances[signal_name] = SignalPerformance(signal_name=signal_name)
            self.recent_returns[signal_name] = deque(maxlen=100)
        
        self.performances[signal_name].total_signals += 1
        
        self._save()
        logger.info(f"记录信号: {signal_id}")
        
        return signal_id
    
    def verify_signal(self, 
                      signal_id: str,
                      exit_price: float,
                      exit_time: str = None,
                      price_history: List[float] = None) -> Dict:
        """验证信号结果"""
        if signal_id not in self.signal_records:
            return {"error": "信号不存在"}
        
        record = self.signal_records[signal_id]
        record.actual_exit_price = exit_price
        record.actual_exit_time = exit_time or datetime.now().isoformat()
        record.verified = True
        
        entry_time = datetime.fromisoformat(record.timestamp)
        exit_time_dt = datetime.fromisoformat(record.actual_exit_time)
        record.holding_days = (exit_time_dt - entry_time).days
        
        record.final_return = (exit_price - record.entry_price) / record.entry_price
        
        if price_history:
            max_price = max(price_history)
            min_price = min(price_history)
            record.max_profit = (max_price - record.entry_price) / record.entry_price
            record.max_drawdown = (record.entry_price - min_price) / record.entry_price
        
        if record.signal_type in ["强烈买入", "买入"]:
            record.is_correct = record.final_return > 0
        elif record.signal_type in ["卖出", "强烈卖出"]:
            record.is_correct = record.final_return < 0
        else:
            record.is_correct = abs(record.final_return) < 0.03
        
        self._update_performance(record)
        self._save()
        
        return {
            "signal_id": signal_id,
            "is_correct": record.is_correct,
            "return": record.final_return,
            "holding_days": record.holding_days,
            "max_profit": record.max_profit,
            "max_drawdown": record.max_drawdown,
        }
    
    def _update_performance(self, record: SignalRecord):
        """更新信号表现统计"""
        perf = self.performances.get(record.signal_name)
        if not perf:
            return
        
        perf.verified_signals += 1
        
        if record.is_correct:
            perf.correct_signals += 1
        
        perf.total_return += record.final_return
        perf.avg_return = perf.total_return / perf.verified_signals
        perf.win_rate = perf.correct_signals / perf.verified_signals if perf.verified_signals > 0 else 0
        
        if perf.verified_signals > 1:
            old_avg = perf.avg_holding_days * (perf.verified_signals - 1)
            perf.avg_holding_days = (old_avg + record.holding_days) / perf.verified_signals
        else:
            perf.avg_holding_days = record.holding_days
        
        perf.avg_max_profit = (perf.avg_max_profit * (perf.verified_signals - 1) + record.max_profit) / perf.verified_signals
        perf.avg_max_drawdown = (perf.avg_max_drawdown * (perf.verified_signals - 1) + record.max_drawdown) / perf.verified_signals
        
        self.recent_returns[record.signal_name].append(record.final_return)
        recent = list(self.recent_returns[record.signal_name])
        if recent:
            positive = sum(1 for r in recent if r > 0)
            perf.recent_accuracy = positive / len(recent)
            perf.recent_returns = recent[-20:]
        
        self._calculate_profit_factor(record.signal_name)
    
    def _calculate_profit_factor(self, signal_name: str):
        """计算盈亏比"""
        returns = list(self.recent_returns.get(signal_name, []))
        if not returns:
            return
        
        profits = [r for r in returns if r > 0]
        losses = [abs(r) for r in returns if r < 0]
        
        total_profit = sum(profits)
        total_loss = sum(losses)
        
        perf = self.performances.get(signal_name)
        if perf:
            perf.profit_factor = total_profit / total_loss if total_loss > 0 else float('inf') if total_profit > 0 else 0
    
    def get_performance_report(self) -> pd.DataFrame:
        """获取信号表现报告"""
        data = []
        for name, perf in self.performances.items():
            data.append({
                '信号名称': name,
                '总信号数': perf.total_signals,
                '已验证': perf.verified_signals,
                '正确数': perf.correct_signals,
                '胜率': f"{perf.win_rate:.1%}",
                '近期准确率': f"{perf.recent_accuracy:.1%}",
                '平均收益': f"{perf.avg_return:.2%}",
                '总收益': f"{perf.total_return:.2%}",
                '平均持仓天数': f"{perf.avg_holding_days:.1f}",
                '平均最大盈利': f"{perf.avg_max_profit:.2%}",
                '平均最大回撤': f"{perf.avg_max_drawdown:.2%}",
                '盈亏比': f"{perf.profit_factor:.2f}",
            })
        
        df = pd.DataFrame(data)
        if not df.empty:
            df = df.sort_values('胜率', ascending=False)
        return df
    
    def get_signal_recommendations(self) -> Dict[str, List[str]]:
        """获取信号优化建议"""
        recommendations = {}
        
        for name, perf in self.performances.items():
            recs = []
            
            if perf.win_rate < 0.4:
                recs.append("胜率较低，建议检查信号触发条件")
            
            if perf.avg_max_drawdown > 0.15:
                recs.append("平均回撤较大，建议优化止损策略")
            
            if perf.avg_holding_days > 20:
                recs.append("持仓时间较长，建议优化止盈策略")
            
            if perf.profit_factor < 1.0:
                recs.append("盈亏比小于1，建议调整仓位管理")
            
            if perf.recent_accuracy < perf.win_rate * 0.8:
                recs.append("近期表现下滑，建议检查市场环境变化")
            
            recommendations[name] = recs if recs else ["表现良好，继续观察"]
        
        return recommendations
    
    def _save(self):
        """保存数据"""
        records_data = {k: asdict(v) for k, v in self.signal_records.items()}
        with open(os.path.join(self.storage_dir, 'signal_records.json'), 'w', encoding='utf-8') as f:
            json.dump(records_data, f, ensure_ascii=False, indent=2)
        
        perf_data = {k: v.to_dict() for k, v in self.performances.items()}
        with open(os.path.join(self.storage_dir, 'signal_performance.json'), 'w', encoding='utf-8') as f:
            json.dump(perf_data, f, ensure_ascii=False, indent=2)
    
    def _load(self):
        """加载数据"""
        records_path = os.path.join(self.storage_dir, 'signal_records.json')
        if os.path.exists(records_path):
            try:
                with open(records_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for k, v in data.items():
                    self.signal_records[k] = SignalRecord(**v)
                logger.info(f"加载 {len(self.signal_records)} 条信号记录")
            except Exception as e:
                logger.error(f"加载信号记录失败: {e}")
        
        perf_path = os.path.join(self.storage_dir, 'signal_performance.json')
        if os.path.exists(perf_path):
            try:
                with open(perf_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for k, v in data.items():
                    self.performances[k] = SignalPerformance(**v)
                    self.recent_returns[k] = deque(v.get('recent_returns', []), maxlen=100)
                logger.info(f"加载 {len(self.performances)} 个信号表现记录")
            except Exception as e:
                logger.error(f"加载信号表现失败: {e}")


class SignalBacktester:
    """
    信号回测器
    
    功能：
    1. 历史数据回测
    2. 信号效果验证
    3. 参数优化
    """
    
    def __init__(self):
        self.backtest_results = []
    
    def backtest_signal(self,
                        df: pd.DataFrame,
                        signal_func,
                        signal_name: str,
                        lookback: int = 252,
                        holding_period: int = 5) -> Dict:
        """
        回测单个信号
        
        Args:
            df: OHLCV数据
            signal_func: 信号生成函数
            signal_name: 信号名称
            lookback: 回测天数
            holding_period: 持仓周期
        """
        if df.empty or len(df) < lookback:
            return {"error": "数据不足"}
        
        results = []
        c = df['close']
        
        for i in range(lookback, len(df) - holding_period):
            test_df = df.iloc[:i+1]
            
            try:
                signal_result = signal_func(test_df)
                
                if signal_result.signal.value in ["买入", "强烈买入"]:
                    entry_price = c.iloc[i]
                    exit_price = c.iloc[i + holding_period]
                    ret = (exit_price - entry_price) / entry_price
                    
                    results.append({
                        'date': df.index[i] if hasattr(df, 'index') else i,
                        'signal': signal_result.signal.value,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'return': ret,
                        'strength': signal_result.strength,
                        'confidence': signal_result.confidence,
                    })
            except Exception as e:
                continue
        
        if not results:
            return {"error": "无有效信号"}
        
        returns = [r['return'] for r in results]
        win_count = sum(1 for r in returns if r > 0)
        
        stats = {
            'signal_name': signal_name,
            'total_signals': len(results),
            'win_count': win_count,
            'win_rate': win_count / len(results),
            'avg_return': np.mean(returns),
            'total_return': sum(returns),
            'max_return': max(returns),
            'max_loss': min(returns),
            'sharpe_ratio': np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0,
            'avg_strength': np.mean([r['strength'] for r in results]),
            'avg_confidence': np.mean([r['confidence'] for r in results]),
        }
        
        self.backtest_results.append(stats)
        
        return stats
    
    def compare_signals(self) -> pd.DataFrame:
        """比较不同信号的表现"""
        if not self.backtest_results:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.backtest_results)
        df = df.sort_values('sharpe_ratio', ascending=False)
        return df


class SignalOptimizer:
    """
    信号优化器
    
    功能：
    1. 参数优化
    2. 阈值调整
    3. 权重优化
    """
    
    def __init__(self):
        self.optimization_results = []
    
    def optimize_thresholds(self,
                            df: pd.DataFrame,
                            signal_func,
                            param_name: str,
                            param_range: List[float],
                            metric: str = 'sharpe_ratio') -> Dict:
        """
        优化信号参数
        
        Args:
            df: OHLCV数据
            signal_func: 信号生成函数
            param_name: 参数名
            param_range: 参数范围
            metric: 优化目标指标
        """
        results = []
        
        for param_value in param_range:
            backtester = SignalBacktester()
            
            def wrapped_func(df_copy):
                return signal_func(df_copy, **{param_name: param_value})
            
            stats = backtester.backtest_signal(df, wrapped_func, f"param_{param_value}")
            
            if 'error' not in stats:
                results.append({
                    'param_value': param_value,
                    'metric_value': stats.get(metric, 0),
                    'stats': stats,
                })
        
        if not results:
            return {"error": "优化失败"}
        
        best = max(results, key=lambda x: x['metric_value'])
        
        self.optimization_results.append({
            'param_name': param_name,
            'best_value': best['param_value'],
            'best_metric': best['metric_value'],
            'all_results': results,
        })
        
        return {
            'param_name': param_name,
            'best_value': best['param_value'],
            'best_metric': best['metric_value'],
            'improvement': best['metric_value'] - results[0]['metric_value'] if results else 0,
        }


_global_validator = None
_global_backtester = None


def get_signal_validator() -> SignalValidator:
    global _global_validator
    if _global_validator is None:
        _global_validator = SignalValidator()
    return _global_validator


def get_signal_backtester() -> SignalBacktester:
    global _global_backtester
    if _global_backtester is None:
        _global_backtester = SignalBacktester()
    return _global_backtester
