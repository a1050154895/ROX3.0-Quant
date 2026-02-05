
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class MarketRegime:
    """
    市场体制过滤器 (Market Regime Filter)
    
    逻辑: "覆巢之下，安有完卵"。在系统性风险高发期（大盘主跌浪、流动性枯竭），
    任何个股策略的胜率都会大幅下降。
    
    功能:
    1. 识别大盘趋势 (Bull/Bear/Sideways)
    2. 识别市场情绪 (Panic/Greed)
    3. 提供全局交易开关 (is_safe_to_trade)
    """
    
    def __init__(self):
        self.status = "UNKNOWN"
        self.risk_score = 0.5  # 0.0 (Safe) -> 1.0 (Dangerous)
        self.regime_type = "NEUTRAL"
    
    def analyze(self, index_df: pd.DataFrame, volatility_window: int = 20) -> Dict[str, Any]:
        """
        分析市场体制
        
        Args:
            index_df: 大盘指数日线数据 (需包含 close, high, low, volume)
            
        Returns:
            Dict: 分析结果
        """
        if index_df is None or index_df.empty or len(index_df) < 60:
            logger.warning("大盘数据不足，无法分析体制，默认为中性")
            return self._default_status()
            
        try:
            close = index_df['close']
            
            # 1. 均线系统 (Trend)
            ma20 = close.rolling(window=20).mean().iloc[-1]
            ma60 = close.rolling(window=60).mean().iloc[-1]
            
            # 2. 波动率 (Volatility - ATR Style)
            # 简化版：Standard Deviation of Returns
            returns = close.pct_change()
            volatility = returns.rolling(window=volatility_window).std().iloc[-1]
            # 假设年化波动率 > 30% 为高危 (0.02 * sqrt(252) ≈ 0.31)
            is_high_volatility = volatility > 0.02 
            
            # 3. 市场状态判定
            if ma20 > ma60:
                if is_high_volatility:
                    self.regime_type = "VOLATILE_BULL" # 疯牛/震荡上行
                    self.risk_score = 0.4
                else:
                    self.regime_type = "QUIET_BULL"    # 慢牛 (最安全)
                    self.risk_score = 0.1
            else:
                if is_high_volatility:
                    self.regime_type = "CRASH"         # 暴跌/崩盘 (最危险)
                    self.risk_score = 0.9
                else:
                    self.regime_type = "BEAR"          # 阴跌/熊市
                    self.risk_score = 0.7
            
            # 4. 成交量确认 (Volume Confirmation)
            # 如果缩量下跌，风险稍低；放量下跌，风险极高
            
            self.status = "SAFE" if self.risk_score < 0.6 else "RISKY"
            
            result = {
                "regime": self.regime_type,
                "risk_score": self.risk_score,
                "status": self.status,
                "metrics": {
                    "ma20": round(ma20, 2),
                    "ma60": round(ma60, 2),
                    "volatility": round(volatility * 100, 2)
                }
            }
            logger.info(f"市场体制分析完成: {result['regime']} (Risk: {self.risk_score})")
            return result
            
        except Exception as e:
            logger.error(f"分析市场体制失败: {e}")
            return self._default_status()
    
    def is_safe_to_trade(self, strategy_type: str = "trend") -> bool:
        """
        全系统熔断开关
        
        Args:
            strategy_type: 策略类型 ('trend', 'reversion')
        """
        # 崩盘模式下，任何多头策略都应停止
        if self.regime_type == "CRASH":
            return False
            
        # 熊市模式下，趋势策略停止，反转/抄底策略可谨慎尝试
        if self.regime_type == "BEAR":
            return strategy_type == "reversion"
            
        # 震荡市，趋势策略可能会被反复打脸，建议降低仓位
        if self.regime_type == "VOLATILE_BULL" and strategy_type == "trend":
            # 这里返回 True 但外部应控制仓位
            return True
            
        return True

    def _default_status(self):
        return {
            "regime": "NEUTRAL",
            "risk_score": 0.5,
            "status": "UNKNOWN",
            "metrics": {}
        }
