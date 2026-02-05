"""
多信号融合模块
融合多个技术指标、策略信号和Kronos预测，生成综合交易信号
基于《量化交易从入门到精通》的7个核心信号系统
"""

import logging
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from app.rox_quant.kronos_adapter import KronosPrediction, KronosAdapter
    from app.rox_quant.trading_signals_advanced import AdvancedTradingSignals
    from app.rox_quant.risk_management_advanced import RiskManager
    from app.rox_quant.trading_parameters import ParameterSet

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """信号类型"""
    STRONG_BUY = 2  # 强烈买入
    BUY = 1  # 买入
    NEUTRAL = 0  # 中立
    SELL = -1  # 卖出
    STRONG_SELL = -2  # 强烈卖出


@dataclass
class Signal:
    """交易信号"""
    symbol: str  # 品种
    signal_type: SignalType  # 信号类型
    confidence: float  # 置信度 (0-1)
    reason: str  # 信号理由
    timestamp: pd.Timestamp  # 时间戳
    indicators: Dict[str, float] = None  # 支持的指标
    
    def __post_init__(self):
        if self.indicators is None:
            self.indicators = {}
    
    def __repr__(self) -> str:
        return f"Signal({self.symbol}, {self.signal_type.name}, conf={self.confidence:.2%}, {self.reason})"


class SignalFusion:
    """
    信号融合器
    
    功能：
    1. 融合7个核心交易信号系统（基于《量化交易从入门到精通》）
    2. 融合多个策略信号
    3. 加权计算综合信号
    4. 集成Kronos AI预测
    5. 应用高级风险管理
    
    7个核心信号：
    1. 趋势突破（海龟/唐奇安）
    2. MA系统 & 专业线
    3. 自适应均线 (AMA)
    4. ATR通道 & 金肯特纳
    5. RSI + 成本线
    6. MACD背离
    7. ADX / Aroon 趋势识别
    """
    
    def __init__(self, params: Optional["ParameterSet"] = None):
        self.params = params
        self.advanced_signals = None
        self.risk_manager = None
        self.ml_predictor = None
        self.adaptive_fusion = None
        self.feature_engineer = None
        
        try:
            from app.rox_quant.trading_signals_advanced import AdvancedTradingSignals
            from app.rox_quant.risk_management_advanced import RiskManager
            from app.rox_quant.trading_parameters import ParameterSet
            
            self.advanced_signals = AdvancedTradingSignals()
            if params is None:
                self.params = ParameterSet()
            self.risk_manager = RiskManager(self.params.risk)
            
            logger.info("✓ 高级信号系统已初始化")
            logger.info("✓ 已加载7个核心信号系统")
        except ImportError as e:
            logger.warning(f"高级模块加载失败: {e}，使用基础功能")
        
        # 初始化 ML 预测模块
        try:
            from app.rox_quant.ml_predictor import MLPredictor
            from app.rox_quant.adaptive_fusion import AdaptiveSignalFusion
            from app.rox_quant.feature_engineer import FeatureEngineer
            from app.rox_quant.market_regime import MarketRegime
            
            self.ml_predictor = MLPredictor()
            self.adaptive_fusion = AdaptiveSignalFusion()
            self.feature_engineer = FeatureEngineer()
            self.regime = MarketRegime()
            
            # 尝试加载已训练的模型
            if self.ml_predictor.load():
                logger.info("✓ ML 预测器已加载")
            else:
                logger.info("✓ ML 预测器已初始化 (未训练)")
            
            logger.info("✓ 自适应融合器已初始化")
            logger.info("✓ 市场体制过滤器已初始化")
        except ImportError as e:
            logger.warning(f"ML 模块加载失败: {e}")
            self.regime = None
    
    # ============ 技术指标计算 ============
    
    def calculate_macd(self, close: pd.Series, 
                      fast_period: int = 12, 
                      slow_period: int = 26,
                      signal_period: int = 9) -> Dict[str, pd.Series]:
        """
        计算MACD
        Returns:
            {
                'macd': MACD线
                'signal': 信号线
                'histogram': MACD直方图
            }
        """
        ema_fast = close.ewm(span=fast_period).mean()
        ema_slow = close.ewm(span=slow_period).mean()
        
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=signal_period).mean()
        histogram = macd - signal
        
        return {
            'macd': macd,
            'signal': signal,
            'histogram': histogram,
        }
    
    def calculate_moving_averages(self, close: pd.Series, 
                                 short_period: int = 5,
                                 long_period: int = 20) -> Dict[str, pd.Series]:
        """
        计算双均线
        """
        ma_short = close.rolling(window=short_period).mean()
        ma_long = close.rolling(window=long_period).mean()
        
        return {
            'ma_short': ma_short,
            'ma_long': ma_long,
        }
    
    def calculate_rsi(self, close: pd.Series, period: int = 14) -> pd.Series:
        """
        计算相对强弱指数 (RSI)
        """
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_bollinger_bands(self, close: pd.Series, 
                                 period: int = 20,
                                 std_dev: float = 2.0) -> Dict[str, pd.Series]:
        """
        计算布林带
        """
        ma = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()
        
        upper = ma + (std * std_dev)
        lower = ma - (std * std_dev)
        
        return {
            'upper': upper,
            'middle': ma,
            'lower': lower,
        }
    
    # ============ 讲座信号 ============
    
    def detect_low_volatility_bottom(self, low: pd.Series, 
                                    period: int = 20,
                                    threshold: float = 0.02) -> pd.Series:
        """
        检测"底背驰" - 价格创新低但波动率下降
        讲座中提到的技术面信号
        """
        # 计算波动率
        returns = low.pct_change().abs()
        volatility = returns.rolling(window=period).std()
        
        # 判断：是否创新低且波动率低
        is_new_low = low == low.rolling(window=period).min()
        is_low_vol = volatility < volatility.rolling(window=period).mean() * (1 - threshold)
        
        bottom_divergence = is_new_low & is_low_vol
        
        return bottom_divergence
    
    def detect_trend(self, close: pd.Series, 
                    short_period: int = 5,
                    long_period: int = 20) -> pd.Series:
        """
        检测趋势：双均线交叉
        讲座中提到的趋势策略
        """
        mas = self.calculate_moving_averages(close, short_period, long_period)
        ma_short = mas['ma_short']
        ma_long = mas['ma_long']
        
        # 上升趋势：短期均线 > 长期均线
        uptrend = ma_short > ma_long
        
        # 检测金叉 (短期向上穿过长期)
        cross_signal = (ma_short > ma_long) & (ma_short.shift(1) <= ma_long.shift(1))
        
        return uptrend.astype(int)
    
    # ============ 信号融合 ============
    
    def fuse_signals(self, signals: List[Signal], 
                    weights: Optional[Dict[str, float]] = None) -> Signal:
        """
        融合多个信号
        
        Args:
            signals: 信号列表
            weights: 各信号的权重
        
        Returns:
            融合后的综合信号
        """
        if not signals:
            return Signal(
                symbol="UNKNOWN",
                signal_type=SignalType.NEUTRAL,
                confidence=0,
                reason="无可融合的信号",
                timestamp=pd.Timestamp.now()
            )
        
        if weights is None:
            weights = {str(i): 1.0 / len(signals) for i in range(len(signals))}
        
        # 【零成本优化】Market Regime Filter
        # 如果市场处于崩溃(CRASH)或熊市(BEAR)，强制降低买入信号权重或直接 veto
        regime_veto = False
        regime_msg = ""
        if hasattr(self, 'regime') and self.regime:
             # 注意：这里假设 regime 状态已更新。实际使用中需要在外部调用 regime.analyze()
             # 或者在这里根据 timestamp 判断是否需要更新 (暂略)
             if not self.regime.is_safe_to_trade(strategy_type="trend"):
                 regime_veto = True
                 regime_msg = f"[市场风险高: {self.regime.regime_type}] "
        
        # 计算加权信号值
        weighted_signal = 0
        total_confidence = 0
        reasons = []
        indicators = {}
        
        for i, signal in enumerate(signals):
            weight = weights.get(str(i), 1.0 / len(signals))
            
            # 如果市场环境恶劣，拦截所有买入信号
            if regime_veto and signal.signal_type.value > 0:
                logger.debug(f"Market Regime Filtered Buy Signal: {signal.symbol}")
                continue 
                
            weighted_signal += signal.signal_type.value * weight * signal.confidence
            total_confidence += signal.confidence * weight
            reasons.append(f"{signal.reason} (conf={signal.confidence:.2%})")
            
            if signal.indicators:
                indicators.update(signal.indicators)
        
        # 如果被 Filter 导致无信号，补全理由
        if regime_veto and weighted_signal == 0:
             reasons.append(regime_msg + "交易熔断")
        
        # 判断综合信号
        if weighted_signal > 0.5:
            fused_type = SignalType.STRONG_BUY if weighted_signal > 1.5 else SignalType.BUY
        elif weighted_signal < -0.5:
            fused_type = SignalType.STRONG_SELL if weighted_signal < -1.5 else SignalType.SELL
        else:
            fused_type = SignalType.NEUTRAL
        
        return Signal(
            symbol=signals[0].symbol,
            signal_type=fused_type,
            confidence=min(total_confidence, 1.0),
            reason=" | ".join(reasons),
            timestamp=pd.Timestamp.now(),
            indicators=indicators
        )
    
    def generate_signal_from_ohlc(self, symbol: str, 
                                 ohlc: pd.DataFrame) -> Signal:
        """
        从OHLC数据生成综合信号
        
        整合多个技术指标的信号
        """
        if ohlc.empty:
            return Signal(
                symbol=symbol,
                signal_type=SignalType.NEUTRAL,
                confidence=0,
                reason="数据不足",
                timestamp=pd.Timestamp.now()
            )
        
        signals = []
        
        # 1. MACD信号
        macd_data = self.calculate_macd(ohlc['close'])
        last_histogram = macd_data['histogram'].iloc[-1]
        last_macd = macd_data['macd'].iloc[-1]
        last_signal = macd_data['signal'].iloc[-1]
        
        if last_macd > last_signal and last_histogram > 0:
            signals.append(Signal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                confidence=0.7,
                reason="MACD:金叉向上",
                timestamp=pd.Timestamp.now(),
                indicators={'macd': float(last_macd), 'signal': float(last_signal)}
            ))
        elif last_macd < last_signal and last_histogram < 0:
            signals.append(Signal(
                symbol=symbol,
                signal_type=SignalType.SELL,
                confidence=0.7,
                reason="MACD:死叉向下",
                timestamp=pd.Timestamp.now(),
                indicators={'macd': float(last_macd), 'signal': float(last_signal)}
            ))
        
        # 2. 均线信号
        ma_data = self.calculate_moving_averages(ohlc['close'])
        last_short = ma_data['ma_short'].iloc[-1]
        last_long = ma_data['ma_long'].iloc[-1]
        
        if last_short > last_long:
            signals.append(Signal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                confidence=0.6,
                reason="均线:短期>长期",
                timestamp=pd.Timestamp.now(),
                indicators={'ma_short': float(last_short), 'ma_long': float(last_long)}
            ))
        elif last_short < last_long:
            signals.append(Signal(
                symbol=symbol,
                signal_type=SignalType.SELL,
                confidence=0.6,
                reason="均线:短期<长期",
                timestamp=pd.Timestamp.now(),
                indicators={'ma_short': float(last_short), 'ma_long': float(last_long)}
            ))
        
        # 3. RSI信号
        rsi = self.calculate_rsi(ohlc['close'])
        last_rsi = rsi.iloc[-1]
        
        if last_rsi < 30:
            signals.append(Signal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                confidence=0.5,
                reason=f"RSI:超卖 ({last_rsi:.1f})",
                timestamp=pd.Timestamp.now(),
                indicators={'rsi': float(last_rsi)}
            ))
        elif last_rsi > 70:
            signals.append(Signal(
                symbol=symbol,
                signal_type=SignalType.SELL,
                confidence=0.5,
                reason=f"RSI:超买 ({last_rsi:.1f})",
                timestamp=pd.Timestamp.now(),
                indicators={'rsi': float(last_rsi)}
            ))
        
        # 融合所有信号
        if signals:
            fused = self.fuse_signals(signals)
            return fused
        else:
            return Signal(
                symbol=symbol,
                signal_type=SignalType.NEUTRAL,
                confidence=0.5,
                reason="指标无明确信号",
                timestamp=pd.Timestamp.now()
            )
    
    def generate_report(self, signal: Signal) -> str:
        """
        生成信号报告
        """
        report = f"""
╔════════════════════════════════════════╗
║        交易信号报告 (Rox Quant)         ║
╚════════════════════════════════════════╝

【品种】{signal.symbol}

【信号】
- 类型: {signal.signal_type.name} ({signal.signal_type.value:+d})
- 置信度: {signal.confidence:.2%}
- 理由: {signal.reason}

【支撑指标】
{chr(10).join(f"- {k}: {v:.4f}" for k, v in signal.indicators.items()) if signal.indicators else "- 无"}

【建议】
{self._get_action_advice(signal.signal_type)}

【时间】{signal.timestamp}
        """
        
        return report
    
    @staticmethod
    def _get_action_advice(signal_type: SignalType) -> str:
        """获取行动建议"""
        advice_map = {
            SignalType.STRONG_BUY: "🟢 强烈买入 - 建议积极建仓，风险可控条件下可考虑增加仓位",
            SignalType.BUY: "🟢 买入 - 建议适度建仓，评估风险后可逐步建立头寸",
            SignalType.NEUTRAL: "🟡 中立 - 观望，等待更明确的信号",
            SignalType.SELL: "🔴 卖出 - 建议逐步减仓，控制风险敞口",
            SignalType.STRONG_SELL: "🔴 强烈卖出 - 建议快速清仓，优先控制风险",
        }
        
        return advice_map.get(signal_type, "未知信号")
    
    # ============ 高级信号系统（基于量化交易从入门到精通） ============
    
    def generate_signal_from_advanced_system(self, 
                                            symbol: str,
                                            ohlc: pd.DataFrame,
                                            use_all_7_signals: bool = True) -> Signal:
        """
        使用7个核心信号系统生成综合信号
        
        Args:
            symbol: 品种代码
            ohlc: OHLC数据 (包含 high, low, close, volume)
            use_all_7_signals: 是否使用全部7个信号
        
        Returns:
            融合后的交易信号
        """
        if not self.advanced_signals or ohlc.empty:
            return self.generate_signal_from_ohlc(symbol, ohlc)
        
        try:
            close = ohlc['close']
            high = ohlc['high']
            low = ohlc['low']
            
            signals_dict = {}
            confidence_dict = {}
            
            # ========== 信号1：趋势突破 (海龟/唐奇安) ==========
            if use_all_7_signals or True:
                donchian = self.advanced_signals.donchian_breakout(
                    high, low,
                    period=self.params.signals.donchian_period if self.params else 20
                )
                
                if donchian['buy_signal'].iloc[-1]:
                    signals_dict['signal_1_donchian'] = SignalType.BUY
                    confidence_dict['signal_1_donchian'] = 0.75
                elif donchian['sell_signal'].iloc[-1]:
                    signals_dict['signal_1_donchian'] = SignalType.SELL
                    confidence_dict['signal_1_donchian'] = 0.75
            
            # ========== 信号2：MA系统 & 专业线 ==========
            ma_sys = self.advanced_signals.professional_ma_system(close)
            
            if ma_sys['bullish_alignment'].iloc[-1]:
                signals_dict['signal_2_ma_system'] = SignalType.BUY
                confidence_dict['signal_2_ma_system'] = 0.70
            elif ma_sys['bearish_alignment'].iloc[-1]:
                signals_dict['signal_2_ma_system'] = SignalType.SELL
                confidence_dict['signal_2_ma_system'] = 0.70
            
            # ========== 信号3：自适应均线 (AMA) ==========
            ama = self.advanced_signals.kaufman_adaptive_ma(close)
            
            ama_value = ama['ama'].iloc[-1]
            close_value = close.iloc[-1]
            
            if close_value > ama_value and close.iloc[-2] <= ama['ama'].iloc[-2]:
                signals_dict['signal_3_ama'] = SignalType.BUY
                confidence_dict['signal_3_ama'] = 0.65
            elif close_value < ama_value and close.iloc[-2] >= ama['ama'].iloc[-2]:
                signals_dict['signal_3_ama'] = SignalType.SELL
                confidence_dict['signal_3_ama'] = 0.65
            
            # ========== 信号4：ATR通道 & 金肯特纳 ==========
            keltner = self.advanced_signals.atr_keltner_channel(high, low, close)
            
            if keltner['breakout_up'].iloc[-1]:
                signals_dict['signal_4_keltner'] = SignalType.BUY
                confidence_dict['signal_4_keltner'] = 0.70
            elif keltner['breakout_down'].iloc[-1]:
                signals_dict['signal_4_keltner'] = SignalType.SELL
                confidence_dict['signal_4_keltner'] = 0.70
            
            # ========== 信号5：RSI + 成本线 ==========
            rsi_cost = self.advanced_signals.rsi_cost_line(close)
            
            if rsi_cost['buy_signal'].iloc[-1]:
                signals_dict['signal_5_rsi_cost'] = SignalType.BUY
                confidence_dict['signal_5_rsi_cost'] = 0.60
            elif rsi_cost['sell_signal'].iloc[-1]:
                signals_dict['signal_5_rsi_cost'] = SignalType.SELL
                confidence_dict['signal_5_rsi_cost'] = 0.60
            
            # ========== 信号6：MACD背离 ==========
            macd_div = self.advanced_signals.macd_divergence(close)
            
            if macd_div['bottom_divergence'].iloc[-1]:
                signals_dict['signal_6_macd'] = SignalType.BUY
                confidence_dict['signal_6_macd'] = 0.70
            elif macd_div['top_divergence'].iloc[-1]:
                signals_dict['signal_6_macd'] = SignalType.SELL
                confidence_dict['signal_6_macd'] = 0.70
            
            # ========== 信号7：ADX & Aroon ==========
            adx = self.advanced_signals.adx_trend_identifier(high, low)
            
            if adx['weak_to_strong'].iloc[-1]:
                signals_dict['signal_7_adx'] = SignalType.BUY
                confidence_dict['signal_7_adx'] = 0.65
            
            # ========== 融合所有信号 ==========
            if signals_dict:
                # 转换为Signal对象
                signal_objects = []
                weights = {}
                
                for idx, (name, sig_type) in enumerate(signals_dict.items()):
                    confidence = confidence_dict.get(name, 0.5)
                    signal_objects.append(
                        Signal(
                            symbol=symbol,
                            signal_type=sig_type,
                            confidence=confidence,
                            reason=f"信号系统 {name.split('_')[1]}",
                            timestamp=pd.Timestamp.now(),
                            indicators={name: sig_type.value}
                        )
                    )
                    weights[str(idx)] = confidence / sum(confidence_dict.values())
                
                # 融合
                fused = self.fuse_signals(signal_objects, weights)
                
                # 添加诊断信息
                fused.indicators['signal_7_count'] = len(signals_dict)
                fused.indicators['bullish_signals'] = sum(
                    1 for t in signals_dict.values() if t.value > 0
                )
                fused.indicators['bearish_signals'] = sum(
                    1 for t in signals_dict.values() if t.value < 0
                )
                
                logger.info(
                    f"✓ 7信号融合: {symbol} "
                    f"买={fused.indicators['bullish_signals']}, "
                    f"卖={fused.indicators['bearish_signals']}, "
                    f"综合={fused.signal_type.name}, "
                    f"置信度={fused.confidence:.2%}"
                )
                
                return fused
            else:
                return Signal(
                    symbol=symbol,
                    signal_type=SignalType.NEUTRAL,
                    confidence=0.5,
                    reason="7信号系统暂无明确指示",
                    timestamp=pd.Timestamp.now()
                )
        
        except Exception as e:
            logger.error(f"7信号系统计算异常: {e}")
            return self.generate_signal_from_ohlc(symbol, ohlc)
    
    def calculate_signal_score(self, ohlc: pd.DataFrame) -> float:
        """
        计算信号综合评分 (0-100)
        基于7个信号的加权组合
        """
        if not self.advanced_signals or ohlc.empty:
            return 50.0
        
        try:
            from app.rox_quant.trading_signals_advanced import SignalStrengthCalculator
            
            score = SignalStrengthCalculator.calculate_signal_score(
                ohlc, self.advanced_signals
            )
            
            return score.iloc[-1] if len(score) > 0 else 50.0
        
        except Exception as e:
            logger.error(f"信号评分计算失败: {e}")
            return 50.0
    
    def get_stop_losses_and_targets(self,
                                   entry_price: float,
                                   atr: float,
                                   direction: str = 'long') -> Dict[str, float]:
        """
        基于风险管理模块获取止损和止盈价位
        
        Returns:
            {'stop_loss': 价格, 'take_profit': 价格, ...}
        """
        if self.risk_manager:
            return self.risk_manager.calculate_stops(entry_price, atr, direction)
        else:
            # 降级：基本计算
            if direction == 'long':
                return {
                    'stop_loss': entry_price - atr * 2.0,
                    'take_profit': entry_price + atr * 3.0
                }
            else:
                return {
                    'stop_loss': entry_price + atr * 2.0,
                    'take_profit': entry_price - atr * 3.0
                }
    
    def fuse_with_advanced_system(self,
                                 ohlc: pd.DataFrame,
                                 symbol: str,
                                 kronos_adapter: Optional["KronosAdapter"] = None,
                                 kronos_weight: float = 0.2) -> Signal:
        """
        融合高级信号系统 + Kronos 预测
        
        Args:
            ohlc: OHLC数据
            symbol: 品种代码
            kronos_adapter: Kronos适配器
            kronos_weight: Kronos在融合中的权重
        
        Returns:
            融合后的终极信号
        """
        signals = []
        weights = {}
        
        # 1. 7信号系统
        advanced_signal = self.generate_signal_from_advanced_system(symbol, ohlc)
        signals.append(advanced_signal)
        weights['0'] = 1.0 - kronos_weight
        
        # 2. Kronos预测（如果可用）
        if kronos_adapter is not None:
            try:
                prediction = kronos_adapter.predict(
                    ohlc,
                    symbol,
                    lookback=min(400, len(ohlc)),
                    pred_len=20
                )
                
                if prediction is not None:
                    kronos_signal = self.generate_signal_from_kronos(prediction)
                    signals.append(kronos_signal)
                    weights['1'] = kronos_weight
            except Exception as e:
                logger.warning(f"Kronos预测失败: {e}")
        
        # 融合
        if len(signals) > 1:
            return self.fuse_signals(signals, weights)
        else:
            return signals[0]
    
    def generate_signal_from_kronos(self, 
                                   prediction: "KronosPrediction",
                                   kronos_weight: float = 0.3) -> Signal:
        """
        从 Kronos 预测生成交易信号
        
        Args:
            prediction: Kronos 预测结果
            kronos_weight: Kronos 信号在融合中的权重 (0-1)
        
        Returns:
            交易信号
        """
        if prediction is None:
            return Signal(
                symbol="UNKNOWN",
                signal_type=SignalType.NEUTRAL,
                confidence=0,
                reason="Kronos预测不可用",
                timestamp=pd.Timestamp.now()
            )
        
        # 根据预测方向和置信度生成信号
        if prediction.direction == "UP" and prediction.confidence > 0.6:
            signal_type = SignalType.BUY
            reason = f"Kronos:预测上涨 {prediction.expected_return:+.2%} (Conf={prediction.confidence:.2%})"
        elif prediction.direction == "DOWN" and prediction.confidence > 0.6:
            signal_type = SignalType.SELL
            reason = f"Kronos:预测下跌 {prediction.expected_return:+.2%} (Conf={prediction.confidence:.2%})"
        else:
            signal_type = SignalType.NEUTRAL
            reason = f"Kronos:信号不确定 (Dir={prediction.direction}, Conf={prediction.confidence:.2%})"
        
        return Signal(
            symbol=prediction.symbol,
            signal_type=signal_type,
            confidence=prediction.confidence,
            reason=reason,
            timestamp=pd.Timestamp.now(),
            indicators={
                'kronos_pred_close': prediction.predicted_close,
                'kronos_confidence': prediction.confidence,
                'kronos_return': prediction.expected_return,
            }
        )
    
    def fuse_with_kronos(self, 
                        ohlc: pd.DataFrame,
                        symbol: str,
                        kronos_adapter: Optional["KronosAdapter"] = None,
                        kronos_weight: float = 0.25) -> Signal:
        """
        融合技术指标和 Kronos 预测
        
        Args:
            ohlc: OHLC 数据
            symbol: 品种代码
            kronos_adapter: Kronos 适配器实例
            kronos_weight: Kronos 权重 (0-1)
        
        Returns:
            融合后的综合信号
        """
        signals = []
        weights = {}
        
        # 1. 技术指标信号
        ta_signal = self.generate_signal_from_ohlc(symbol, ohlc)
        signals.append(ta_signal)
        
        ta_weight = 1.0 - kronos_weight
        weights['0'] = ta_weight
        
        # 2. Kronos 预测信号（如果可用）
        if kronos_adapter is not None:
            try:
                prediction = kronos_adapter.predict(
                    ohlc,
                    symbol,
                    lookback=min(400, len(ohlc)),
                    pred_len=20
                )
                
                if prediction is not None:
                    kronos_signal = self.generate_signal_from_kronos(prediction)
                    signals.append(kronos_signal)
                    weights['1'] = kronos_weight
                    logger.info(f"✓ Kronos 预测已融合: {prediction}")
                else:
                    logger.warning(f"✗ Kronos 预测失败: {symbol}")
            except Exception as e:
                logger.error(f"Kronos 预测异常: {e}")
        
        # 融合所有信号
        if signals:
            fused = self.fuse_signals(signals, weights)
            return fused
        else:
            return ta_signal
    
    def set_kronos_weight(self, kronos_weight: float) -> bool:
        """
        设置 Kronos 在信号融合中的权重
        
        Args:
            kronos_weight: 权重 (0-1，0表示禁用Kronos)
        
        Returns:
            是否设置成功
        """
        if not (0 <= kronos_weight <= 1):
            logger.error(f"权重必须在 0-1 之间，当前值={kronos_weight}")
            return False
        
        self.kronos_weight = kronos_weight
        logger.info(f"✓ Kronos 权重已设置为 {kronos_weight:.2%}")
        return True


