"""
特征工程模块
从 OHLCV 数据提取高质量预测特征
基于《量化交易从入门到精通》和学术研究的特征集
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    特征工程器
    
    从原始 OHLCV 数据生成 50+ 个预测特征，分为以下类别:
    1. 价格动量特征 (Momentum)
    2. 波动率特征 (Volatility) 
    3. 成交量特征 (Volume)
    4. 趋势特征 (Trend)
    5. 技术指标特征 (Technical Indicators)
    6. 统计特征 (Statistical)
    """
    
    def __init__(self):
        self.feature_names = []
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        从 OHLCV 数据生成全部特征
        
        Args:
            df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
                Index should be datetime
        
        Returns:
            DataFrame with all features (NaN rows will be dropped)
        """
        if df.empty or len(df) < 60:
            logger.warning("数据不足60天，无法生成完整特征")
            return pd.DataFrame()
        
        # 确保列名标准化
        df = df.copy()
        df.columns = df.columns.str.lower()
        
        features = pd.DataFrame(index=df.index)
        
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume'] if 'volume' in df.columns else pd.Series(1, index=df.index)
        
        # ========== 1. 价格动量特征 ==========
        self._add_synthetic_l2_features(df, features) # 新增: 合成L2特征
        
        for period in [1, 3, 5, 10, 20, 60]:
            features[f'ret_{period}d'] = close.pct_change(period)
        
        # 动量 (Momentum)
        features['momentum_10'] = close / close.shift(10) - 1
        features['momentum_20'] = close / close.shift(20) - 1
        features['momentum_60'] = close / close.shift(60) - 1
        
        # 加速度 (二阶动量)
        features['momentum_accel'] = features['ret_5d'] - features['ret_5d'].shift(5)
        
        # ========== 2. 波动率特征 ==========
        # 历史波动率
        for period in [5, 10, 20, 60]:
            features[f'volatility_{period}d'] = close.pct_change().rolling(period).std() * np.sqrt(252)
        
        # ATR (Average True Range)
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        
        features['atr_14'] = tr.rolling(14).mean()
        features['atr_pct'] = features['atr_14'] / close  # 相对ATR
        
        # 布林带宽度
        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        features['bb_width'] = (4 * bb_std) / bb_mid
        features['bb_position'] = (close - (bb_mid - 2*bb_std)) / (4 * bb_std)
        
        # ========== 3. 成交量特征 ==========
        # 成交量变化
        features['volume_change_1d'] = volume.pct_change()
        features['volume_change_5d'] = volume.pct_change(5)
        
        # 成交量均值比
        features['volume_ratio_5'] = volume / volume.rolling(5).mean()
        features['volume_ratio_20'] = volume / volume.rolling(20).mean()
        
        # 量价关系
        features['volume_price_corr'] = close.rolling(20).corr(volume)
        
        # OBV 变化率
        obv = (np.sign(close.diff()) * volume).cumsum()
        features['obv_change_10'] = obv.pct_change(10)
        
        # ========== 4. 趋势特征 ==========
        # 均线
        for period in [5, 10, 20, 60]:
            ma = close.rolling(period).mean()
            features[f'ma_{period}_dist'] = (close - ma) / ma
        
        # 均线斜率
        features['ma_5_slope'] = close.rolling(5).mean().diff(5) / close.rolling(5).mean().shift(5)
        features['ma_20_slope'] = close.rolling(20).mean().diff(5) / close.rolling(20).mean().shift(5)
        
        # 均线交叉特征
        ma_5 = close.rolling(5).mean()
        ma_20 = close.rolling(20).mean()
        features['ma_cross'] = (ma_5 - ma_20) / ma_20
        features['ma_cross_signal'] = np.where(ma_5 > ma_20, 1, -1)
        
        # ========== 5. 技术指标特征 ==========
        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, 1e-10)
        features['rsi_14'] = 100 - (100 / (1 + rs))
        features['rsi_oversold'] = (features['rsi_14'] < 30).astype(int)
        features['rsi_overbought'] = (features['rsi_14'] > 70).astype(int)
        
        # MACD
        ema_12 = close.ewm(span=12).mean()
        ema_26 = close.ewm(span=26).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9).mean()
        features['macd'] = macd
        features['macd_signal'] = signal
        features['macd_hist'] = macd - signal
        features['macd_cross'] = np.where(macd > signal, 1, -1)
        
        # Stochastic %K %D
        low_14 = low.rolling(14).min()
        high_14 = high.rolling(14).max()
        features['stoch_k'] = 100 * (close - low_14) / (high_14 - low_14 + 1e-10)
        features['stoch_d'] = features['stoch_k'].rolling(3).mean()
        
        # ADX (趋势强度)
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        atr_14 = tr.rolling(14).mean()
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr_14)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr_14)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        features['adx'] = dx.rolling(14).mean()
        features['adx_strong_trend'] = (features['adx'] > 25).astype(int)
        
        # ========== 6. 统计特征 ==========
        # 偏度和峰度
        features['skewness_20'] = close.pct_change().rolling(20).skew()
        features['kurtosis_20'] = close.pct_change().rolling(20).kurt()
        
        # 最高/最低价位置
        features['high_20_dist'] = (high.rolling(20).max() - close) / close
        features['low_20_dist'] = (close - low.rolling(20).min()) / close
        
        # 价格位置 (0-1 范围)
        features['price_position_20'] = (close - low.rolling(20).min()) / (high.rolling(20).max() - low.rolling(20).min() + 1e-10)
        
        # ========== 7. 目标变量 (用于训练) ==========
        # 未来收益率 (这些在预测时需要排除)
        features['target_1d'] = close.shift(-1) / close - 1
        features['target_5d'] = close.shift(-5) / close - 1
        features['target_direction'] = np.where(features['target_1d'] > 0, 1, 0)
        
        # 保存特征名
        self.feature_names = [c for c in features.columns if not c.startswith('target_')]
        
        # 删除NaN行
        features = features.dropna()
        
        logger.info(f"生成特征: {len(self.feature_names)} 个，样本数: {len(features)}")
        
        return features
    
    
    def _add_synthetic_l2_features(self, df: pd.DataFrame, features: pd.DataFrame):
        """
        【零成本优化】合成 Level-2 特征
        """
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # 1. Money Flow Multiplier (MFM)
        range_len = high - low
        # 避免除以零
        mf_multiplier = ((close - low) - (high - close)) / range_len.replace(0, 1)
        
        # 2. Money Flow Volume (MFV)
        features['mf_vol'] = mf_multiplier * volume
        
        # 3. Chaikin Money Flow (CMF) - 20日累积 MF
        features['cmf_20'] = features['mf_vol'].rolling(window=20).sum() / volume.rolling(window=20).sum()
        
        # 4. Intraday Intensity
        features['intraday_intensity'] = (2 * close - high - low) / range_len.replace(0, 1) * volume
        
        # 5. Price-Volume Trend (PVT) Change
        pct_change = close.pct_change()
        features['pvt_change'] = pct_change * volume

    def get_feature_names(self, exclude_targets: bool = True) -> List[str]:
        """获取特征名列表"""
        if exclude_targets:
            return [f for f in self.feature_names if not f.startswith('target_')]
        return self.feature_names
    
    def get_feature_importance_names(self) -> Dict[str, str]:
        """返回特征名和中文描述的映射"""
        return {
            'ret_1d': '1日收益率',
            'ret_5d': '5日收益率',
            'ret_10d': '10日收益率',
            'ret_20d': '20日收益率',
            'momentum_10': '10日动量',
            'momentum_20': '20日动量',
            'volatility_10d': '10日波动率',
            'volatility_20d': '20日波动率',
            'atr_pct': '相对ATR',
            'bb_width': '布林带宽度',
            'bb_position': '布林带位置',
            'volume_ratio_5': '5日量比',
            'volume_ratio_20': '20日量比',
            'ma_5_dist': '5日均线偏离',
            'ma_20_dist': '20日均线偏离',
            'rsi_14': 'RSI(14)',
            'macd_hist': 'MACD柱',
            'adx': 'ADX趋势强度',
            'stoch_k': '随机指标K',
            'price_position_20': '20日价格位置',
            'mf_vol': '资金流强度(L2拟合)',
            'cmf_20': '20日CMF资金流',
            'intraday_intensity': '日内强度',
            'pvt_change': '量价趋势变动'
        }


# 便捷函数
def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """快速生成特征的便捷函数"""
    return FeatureEngineer().generate_features(df)
