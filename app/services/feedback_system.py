import logging
from typing import Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)

class FeedbackSystem:
    """
    反馈系统类
    
    功能:
    1. 记录交易结果
    2. 分析交易绩效
    3. 调整模型参数
    4. 生成反馈报告
    """
    
    def __init__(self):
        self.trade_records = []
        self.trader_performance = {}
        self.model_parameters = {
            "risk_adjustment": 1.0,
            "trend_sensitivity": 0.5,
            "mean_reversion": 0.3,
            "volatility_threshold": 0.02
        }
    
    def record_trade(self, trader_id: str, symbol: str, side: str, 
                    quantity: int, price: float, result: Dict[str, Any]):
        """
        记录交易
        
        Args:
            trader_id: 交易员ID
            symbol: 股票代码
            side: 交易方向
            quantity: 交易数量
            price: 交易价格
            result: 交易结果
        """
        trade_record = {
            "trader_id": trader_id,
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        
        self.trade_records.append(trade_record)
        
        # 更新交易员绩效
        self._update_trader_performance(trader_id, result)
        
        # 调整模型参数
        self._adjust_model_parameters()
    
    def _update_trader_performance(self, trader_id: str, result: Dict[str, Any]):
        """
        更新交易员绩效
        
        Args:
            trader_id: 交易员ID
            result: 交易结果
        """
        if trader_id not in self.trader_performance:
            self.trader_performance[trader_id] = {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "total_pnl": 0,
                "win_rate": 0.0,
                "average_pnl": 0.0
            }
        
        performance = self.trader_performance[trader_id]
        performance["total_trades"] += 1
        
        pnl = result.get("pnl", 0)
        performance["total_pnl"] += pnl
        
        if pnl > 0:
            performance["winning_trades"] += 1
        elif pnl < 0:
            performance["losing_trades"] += 1
        
        # 更新统计指标
        performance["win_rate"] = performance["winning_trades"] / performance["total_trades"]
        performance["average_pnl"] = performance["total_pnl"] / performance["total_trades"]
    
    def _adjust_model_parameters(self):
        """
        调整模型参数
        """
        # 基于整体交易绩效调整参数
        if not self.trade_records:
            return
        
        # 计算整体绩效
        total_trades = len(self.trade_records)
        winning_trades = sum(1 for record in self.trade_records if record["result"].get("pnl", 0) > 0)
        total_pnl = sum(record["result"].get("pnl", 0) for record in self.trade_records)
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        average_pnl = total_pnl / total_trades if total_trades > 0 else 0
        
        # 根据绩效调整参数
        if win_rate < 0.4:
            # 胜率低，降低风险
            self.model_parameters["risk_adjustment"] = max(0.5, self.model_parameters["risk_adjustment"] - 0.1)
            self.model_parameters["trend_sensitivity"] = min(1.0, self.model_parameters["trend_sensitivity"] + 0.1)
        elif win_rate > 0.6:
            # 胜率高，增加风险
            self.model_parameters["risk_adjustment"] = min(1.5, self.model_parameters["risk_adjustment"] + 0.1)
            self.model_parameters["trend_sensitivity"] = max(0.1, self.model_parameters["trend_sensitivity"] - 0.1)
        
        if average_pnl < 0:
            # 平均盈利为负，增加均值回归
            self.model_parameters["mean_reversion"] = min(1.0, self.model_parameters["mean_reversion"] + 0.1)
        else:
            # 平均盈利为正，降低均值回归
            self.model_parameters["mean_reversion"] = max(0.1, self.model_parameters["mean_reversion"] - 0.1)
    
    def get_trader_performance(self, trader_id: str) -> Dict[str, Any]:
        """
        获取交易员绩效
        
        Args:
            trader_id: 交易员ID
        
        Returns:
            交易员绩效
        """
        return self.trader_performance.get(trader_id, {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_pnl": 0,
            "win_rate": 0.0,
            "average_pnl": 0.0
        })
    
    def get_all_performance(self) -> Dict[str, Any]:
        """
        获取所有交易员的整体绩效
        
        Returns:
            整体绩效
        """
        if not self.trade_records:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "total_pnl": 0,
                "win_rate": 0.0,
                "average_pnl": 0.0,
                "trader_count": len(self.trader_performance)
            }
        
        total_trades = len(self.trade_records)
        winning_trades = sum(1 for record in self.trade_records if record["result"].get("pnl", 0) > 0)
        total_pnl = sum(record["result"].get("pnl", 0) for record in self.trade_records)
        
        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": total_trades - winning_trades,
            "total_pnl": total_pnl,
            "win_rate": winning_trades / total_trades,
            "average_pnl": total_pnl / total_trades,
            "trader_count": len(self.trader_performance)
        }
    
    def get_model_parameters(self) -> Dict[str, float]:
        """
        获取模型参数
        
        Returns:
            模型参数
        """
        return self.model_parameters
    
    def generate_feedback_report(self) -> Dict[str, Any]:
        """
        生成反馈报告
        
        Returns:
            反馈报告
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "overall_performance": self.get_all_performance(),
            "trader_performance": self.trader_performance,
            "model_parameters": self.model_parameters,
            "recent_trades": self.trade_records[-10:]  # 最近10笔交易
        }
        
        logger.info(f"生成反馈报告: {report}")
        return report
    
    def reset(self):
        """
        重置反馈系统
        """
        self.trade_records = []
        self.trader_performance = {}
        self.model_parameters = {
            "risk_adjustment": 1.0,
            "trend_sensitivity": 0.5,
            "mean_reversion": 0.3,
            "volatility_threshold": 0.02
        }
        logger.info("反馈系统已重置")