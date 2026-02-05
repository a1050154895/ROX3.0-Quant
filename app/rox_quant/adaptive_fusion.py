"""
自适应信号融合模块
根据历史表现动态调整各信号权重
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import logging
import json
import os

logger = logging.getLogger(__name__)


@dataclass
class SignalPerformance:
    """信号历史表现"""
    signal_name: str
    total_predictions: int = 0
    correct_predictions: int = 0
    recent_accuracy: float = 0.5
    weight: float = 1.0
    last_updated: str = ""
    
    @property
    def accuracy(self) -> float:
        if self.total_predictions == 0:
            return 0.5
        return self.correct_predictions / self.total_predictions


class AdaptiveSignalFusion:
    """
    自适应信号融合器
    
    功能:
    1. 跟踪每个信号的历史表现
    2. 动态调整信号权重 (表现好的信号权重提升)
    3. 贝叶斯/EMA 平滑更新避免过拟合
    4. 信号相关性分析，降低冗余信号权重
    """
    
    def __init__(self, 
                 learning_rate: float = 0.1,
                 min_weight: float = 0.05,
                 max_weight: float = 0.5,
                 history_window: int = 100,
                 save_dir: str = "data/signal_performance"):
        """
        Args:
            learning_rate: 权重更新速率 (0-1)
            min_weight: 最小权重
            max_weight: 最大权重
            history_window: 历史窗口大小 (用于计算近期准确率)
            save_dir: 保存目录
        """
        self.learning_rate = learning_rate
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.history_window = history_window
        self.save_dir = save_dir
        
        # 信号表现跟踪
        self.performances: Dict[str, SignalPerformance] = {}
        
        # 近期预测历史 (用于计算近期准确率)
        self.recent_history: Dict[str, deque] = {}
        
        # 信号相关性矩阵
        self.correlation_matrix: Dict[Tuple[str, str], float] = {}
        
        os.makedirs(save_dir, exist_ok=True)
        self._load()
    
    def register_signal(self, signal_name: str, initial_weight: float = 1.0):
        """注册新信号"""
        if signal_name not in self.performances:
            self.performances[signal_name] = SignalPerformance(
                signal_name=signal_name,
                weight=initial_weight
            )
            self.recent_history[signal_name] = deque(maxlen=self.history_window)
            logger.info(f"注册信号: {signal_name}")
    
    def update(self, signal_name: str, prediction: int, actual: int):
        """
        更新信号表现
        
        Args:
            signal_name: 信号名
            prediction: 预测值 (1=涨, 0=跌)
            actual: 实际值 (1=涨, 0=跌)
        """
        if signal_name not in self.performances:
            self.register_signal(signal_name)
        
        perf = self.performances[signal_name]
        is_correct = 1 if prediction == actual else 0
        
        # 更新总体统计
        perf.total_predictions += 1
        perf.correct_predictions += is_correct
        
        # 更新近期历史
        self.recent_history[signal_name].append(is_correct)
        
        # 计算近期准确率 (EMA 平滑)
        recent = list(self.recent_history[signal_name])
        if recent:
            new_recent_acc = sum(recent) / len(recent)
            perf.recent_accuracy = (
                (1 - self.learning_rate) * perf.recent_accuracy +
                self.learning_rate * new_recent_acc
            )
        
        # 更新权重 (基于近期表现)
        self._update_weight(signal_name)
        
        import datetime
        perf.last_updated = datetime.datetime.now().isoformat()
    
    def _update_weight(self, signal_name: str):
        """根据表现更新权重"""
        perf = self.performances[signal_name]
        
        # 基础权重 = 近期准确率
        # 高于 50% 的信号权重提升，低于 50% 的降低
        base_weight = perf.recent_accuracy
        
        # 归一化到 [min_weight, max_weight]
        normalized = self.min_weight + (self.max_weight - self.min_weight) * (
            (base_weight - 0.3) / 0.4  # 假设准确率在 30%-70% 之间
        )
        
        perf.weight = max(self.min_weight, min(self.max_weight, normalized))
    
    def get_weights(self, signal_names: Optional[List[str]] = None) -> Dict[str, float]:
        """
        获取归一化权重
        
        Args:
            signal_names: 要获取的信号名列表，None 表示全部
        
        Returns:
            归一化权重字典 (总和为 1)
        """
        if signal_names is None:
            signal_names = list(self.performances.keys())
        
        weights = {}
        for name in signal_names:
            if name in self.performances:
                weights[name] = self.performances[name].weight
            else:
                weights[name] = 1.0  # 新信号默认权重
        
        # 归一化
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        
        return weights
    
    def fuse_predictions(self, predictions: Dict[str, Tuple[int, float]]) -> Tuple[int, float]:
        """
        融合多个信号的预测
        
        Args:
            predictions: {信号名: (预测值, 概率)}
        
        Returns:
            (融合预测, 融合概率)
        """
        if not predictions:
            return 0, 0.5
        
        weights = self.get_weights(list(predictions.keys()))
        
        # 加权投票
        weighted_prob = 0.0
        for name, (pred, prob) in predictions.items():
            w = weights.get(name, 1.0 / len(predictions))
            weighted_prob += w * prob
        
        # 预测方向
        fused_pred = 1 if weighted_prob > 0.5 else 0
        
        # 置信度调整
        confidence = abs(weighted_prob - 0.5) * 2
        
        return fused_pred, weighted_prob
    
    def get_performance_report(self) -> pd.DataFrame:
        """获取信号表现报告"""
        data = []
        for name, perf in self.performances.items():
            data.append({
                '信号名': name,
                '总预测数': perf.total_predictions,
                '正确数': perf.correct_predictions,
                '总体准确率': f"{perf.accuracy:.2%}",
                '近期准确率': f"{perf.recent_accuracy:.2%}",
                '当前权重': f"{perf.weight:.3f}",
                '最后更新': perf.last_updated
            })
        
        df = pd.DataFrame(data)
        if not df.empty:
            df = df.sort_values('近期准确率', ascending=False)
        return df
    
    def update_correlation(self, signal1: str, signal2: str, 
                          pred1: List[int], pred2: List[int]):
        """更新信号相关性"""
        if len(pred1) != len(pred2) or len(pred1) < 10:
            return
        
        corr = np.corrcoef(pred1, pred2)[0, 1]
        key = tuple(sorted([signal1, signal2]))
        self.correlation_matrix[key] = corr
        
        # 高相关信号降低其中一个的权重 (避免冗余)
        if abs(corr) > 0.8:
            perf1 = self.performances.get(signal1)
            perf2 = self.performances.get(signal2)
            if perf1 and perf2:
                # 保留表现更好的
                if perf1.recent_accuracy < perf2.recent_accuracy:
                    perf1.weight *= 0.7
                else:
                    perf2.weight *= 0.7
    
    def _save(self):
        """保存到文件"""
        data = {
            'performances': {
                k: {
                    'signal_name': v.signal_name,
                    'total_predictions': v.total_predictions,
                    'correct_predictions': v.correct_predictions,
                    'recent_accuracy': v.recent_accuracy,
                    'weight': v.weight,
                    'last_updated': v.last_updated
                }
                for k, v in self.performances.items()
            },
            'correlation_matrix': {
                f"{k[0]}|{k[1]}": v for k, v in self.correlation_matrix.items()
            }
        }
        
        path = os.path.join(self.save_dir, 'adaptive_fusion.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def _load(self):
        """从文件加载"""
        path = os.path.join(self.save_dir, 'adaptive_fusion.json')
        if not os.path.exists(path):
            return
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for k, v in data.get('performances', {}).items():
                self.performances[k] = SignalPerformance(**v)
                self.recent_history[k] = deque(maxlen=self.history_window)
            
            for k, v in data.get('correlation_matrix', {}).items():
                parts = k.split('|')
                if len(parts) == 2:
                    self.correlation_matrix[(parts[0], parts[1])] = v
            
            logger.info(f"加载 {len(self.performances)} 个信号表现记录")
        except Exception as e:
            logger.error(f"加载失败: {e}")
    
    def save(self):
        """公开保存方法"""
        self._save()
        logger.info("信号表现已保存")


# 便捷函数
_global_fusion = None

def get_adaptive_fusion() -> AdaptiveSignalFusion:
    """获取全局自适应融合器实例"""
    global _global_fusion
    if _global_fusion is None:
        _global_fusion = AdaptiveSignalFusion()
    return _global_fusion
