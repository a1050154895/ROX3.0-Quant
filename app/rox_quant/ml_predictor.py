"""
机器学习预测器模块
集成多种 ML 模型进行股价方向预测
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging
import pickle
import os

logger = logging.getLogger(__name__)

# 检查可用的 ML 库
HAS_SKLEARN = False
HAS_TORCH = False

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score, classification_report
    HAS_SKLEARN = True
except ImportError:
    logger.warning("sklearn 未安装，部分功能不可用")

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    logger.warning("PyTorch 未安装，LSTM 模型不可用")


class PredictionType(Enum):
    """预测类型"""
    DIRECTION = "direction"  # 涨跌方向
    RETURN = "return"  # 收益率


@dataclass
class PredictionResult:
    """预测结果"""
    symbol: str
    prediction: int  # 1=涨, 0=跌
    probability: float  # 预测概率
    confidence: float  # 置信度
    model_name: str
    features_used: int
    timestamp: str


class LSTMModel(nn.Module):
    """LSTM 时序预测模型"""
    
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步
        last_out = lstm_out[:, -1, :]
        return self.fc(last_out)


class MLPredictor:
    """
    机器学习预测器
    
    支持的模型:
    1. RandomForest (集成学习)
    2. GradientBoosting (梯度提升)
    3. LogisticRegression (逻辑回归)
    4. LSTM (如果 PyTorch 可用)
    5. Ensemble (集成投票)
    """
    
    def __init__(self, model_dir: str = "data/models"):
        self.model_dir = model_dir
        self.models: Dict[str, Any] = {}
        self.scaler = StandardScaler() if HAS_SKLEARN else None
        self.is_fitted = False
        self.feature_names: List[str] = []
        
        # 确保模型目录存在
        os.makedirs(model_dir, exist_ok=True)
        
        # 初始化模型
        if HAS_SKLEARN:
            self.models['random_forest'] = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                random_state=42,
                n_jobs=-1
            )
            self.models['gradient_boosting'] = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
            self.models['logistic'] = LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        
        if HAS_TORCH:
            self.lstm_model = None  # 延迟初始化
            self.lstm_seq_len = 20
    
    def fit(self, features_df: pd.DataFrame, 
            target_col: str = 'target_direction',
            test_size: float = 0.2) -> Dict[str, float]:
        """
        训练所有模型
        
        Args:
            features_df: 特征 DataFrame (包含目标列)
            target_col: 目标列名
            test_size: 测试集比例
        
        Returns:
            各模型的准确率
        """
        if not HAS_SKLEARN:
            logger.error("sklearn 未安装，无法训练")
            return {}
        
        if target_col not in features_df.columns:
            logger.error(f"目标列 {target_col} 不存在")
            return {}
        
        # 准备数据
        self.feature_names = [c for c in features_df.columns if not c.startswith('target_')]
        X = features_df[self.feature_names].values
        y = features_df[target_col].values
        
        # 标准化
        X = self.scaler.fit_transform(X)
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False  # 时序数据不打乱
        )
        
        results = {}
        
        # 训练 sklearn 模型
        for name, model in self.models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                results[name] = acc
                logger.info(f"模型 {name} 准确率: {acc:.2%}")
            except Exception as e:
                logger.error(f"训练 {name} 失败: {e}")
                results[name] = 0.0
        
        # 训练 LSTM
        if HAS_TORCH and len(X_train) > self.lstm_seq_len:
            try:
                lstm_acc = self._train_lstm(X_train, y_train, X_test, y_test)
                results['lstm'] = lstm_acc
                logger.info(f"模型 LSTM 准确率: {lstm_acc:.2%}")
            except Exception as e:
                logger.error(f"训练 LSTM 失败: {e}")
        
        # 集成模型准确率 (简单投票)
        if len(results) > 1:
            ensemble_preds = self._ensemble_predict(X_test)
            ensemble_acc = accuracy_score(y_test, ensemble_preds)
            results['ensemble'] = ensemble_acc
            logger.info(f"集成模型准确率: {ensemble_acc:.2%}")
        
        self.is_fitted = True
        
        return results
    
    def _train_lstm(self, X_train, y_train, X_test, y_test, epochs: int = 50) -> float:
        """训练 LSTM 模型"""
        if not HAS_TORCH:
            return 0.0
        
        # 准备序列数据
        def create_sequences(X, y, seq_len):
            Xs, ys = [], []
            for i in range(len(X) - seq_len):
                Xs.append(X[i:i+seq_len])
                ys.append(y[i+seq_len])
            return np.array(Xs), np.array(ys)
        
        X_seq, y_seq = create_sequences(X_train, y_train, self.lstm_seq_len)
        X_test_seq, y_test_seq = create_sequences(X_test, y_test, self.lstm_seq_len)
        
        if len(X_seq) < 10:
            return 0.0
        
        # 转换为 Tensor
        X_tensor = torch.FloatTensor(X_seq)
        y_tensor = torch.FloatTensor(y_seq).unsqueeze(1)
        
        # 初始化模型
        input_size = X_seq.shape[2]
        self.lstm_model = LSTMModel(input_size)
        
        optimizer = torch.optim.Adam(self.lstm_model.parameters(), lr=0.001)
        criterion = nn.BCELoss()
        
        # 训练
        self.lstm_model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.lstm_model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
        
        # 评估
        self.lstm_model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test_seq)
            preds = self.lstm_model(X_test_tensor).numpy().flatten()
            preds_binary = (preds > 0.5).astype(int)
            acc = accuracy_score(y_test_seq, preds_binary)
        
        return acc
    
    def _ensemble_predict(self, X: np.ndarray) -> np.ndarray:
        """集成预测 (多数投票)"""
        predictions = []
        
        for name, model in self.models.items():
            try:
                pred = model.predict(X)
                predictions.append(pred)
            except Exception:
                pass
        
        if not predictions:
            return np.zeros(len(X))
        
        # 多数投票
        stacked = np.vstack(predictions)
        ensemble = (stacked.mean(axis=0) > 0.5).astype(int)
        return ensemble
    
    def predict(self, features_df: pd.DataFrame, 
                model_name: str = 'ensemble') -> List[PredictionResult]:
        """
        进行预测
        
        Args:
            features_df: 特征 DataFrame
            model_name: 使用的模型名 ('random_forest', 'gradient_boosting', 'ensemble', 'lstm')
        
        Returns:
            预测结果列表
        """
        if not self.is_fitted:
            logger.warning("模型未训练，返回空结果")
            return []
        
        if not HAS_SKLEARN:
            return []
        
        # 准备特征
        X = features_df[self.feature_names].values
        X_scaled = self.scaler.transform(X)
        
        results = []
        
        if model_name == 'ensemble':
            preds = self._ensemble_predict(X_scaled)
            # 获取概率 (使用 RandomForest)
            try:
                probs = self.models['random_forest'].predict_proba(X_scaled)[:, 1]
            except Exception:
                probs = np.where(preds == 1, 0.6, 0.4)
        elif model_name == 'lstm' and HAS_TORCH and self.lstm_model:
            # LSTM 预测需要序列
            if len(X_scaled) >= self.lstm_seq_len:
                X_seq = X_scaled[-self.lstm_seq_len:].reshape(1, self.lstm_seq_len, -1)
                self.lstm_model.eval()
                with torch.no_grad():
                    prob = self.lstm_model(torch.FloatTensor(X_seq)).item()
                preds = np.array([1 if prob > 0.5 else 0])
                probs = np.array([prob])
            else:
                return []
        elif model_name in self.models:
            model = self.models[model_name]
            preds = model.predict(X_scaled)
            try:
                probs = model.predict_proba(X_scaled)[:, 1]
            except Exception:
                probs = np.where(preds == 1, 0.6, 0.4)
        else:
            logger.error(f"未知模型: {model_name}")
            return []
        
        # 构建结果
        import datetime
        now = datetime.datetime.now().isoformat()
        
        for i in range(len(preds)):
            confidence = abs(probs[i] - 0.5) * 2  # 0-1 范围
            results.append(PredictionResult(
                symbol="",
                prediction=int(preds[i]),
                probability=float(probs[i]),
                confidence=float(confidence),
                model_name=model_name,
                features_used=len(self.feature_names),
                timestamp=now
            ))
        
        return results
    
    def predict_single(self, features_df: pd.DataFrame, 
                       symbol: str = "",
                       model_name: str = 'ensemble') -> Optional[PredictionResult]:
        """预测单个样本 (最新一行)"""
        if features_df.empty:
            return None
        
        last_row = features_df.tail(1)
        results = self.predict(last_row, model_name)
        
        if results:
            result = results[0]
            result.symbol = symbol
            return result
        return None
    
    def get_feature_importance(self) -> Dict[str, float]:
        """获取特征重要性 (来自 RandomForest)"""
        if 'random_forest' not in self.models or not self.is_fitted:
            return {}
        
        model = self.models['random_forest']
        importance = model.feature_importances_
        
        return dict(sorted(
            zip(self.feature_names, importance),
            key=lambda x: x[1],
            reverse=True
        ))
    
    def save(self, filename: str = "ml_predictor.pkl"):
        """保存模型"""
        path = os.path.join(self.model_dir, filename)
        data = {
            'models': self.models,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }
        
        if HAS_TORCH and self.lstm_model:
            data['lstm_state'] = self.lstm_model.state_dict()
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"模型已保存到 {path}")
    
    def load(self, filename: str = "ml_predictor.pkl") -> bool:
        """加载模型"""
        path = os.path.join(self.model_dir, filename)
        
        if not os.path.exists(path):
            logger.warning(f"模型文件不存在: {path}")
            return False
        
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            
            self.models = data['models']
            self.scaler = data['scaler']
            self.feature_names = data['feature_names']
            self.is_fitted = data['is_fitted']
            
            if HAS_TORCH and 'lstm_state' in data:
                input_size = len(self.feature_names)
                self.lstm_model = LSTMModel(input_size)
                self.lstm_model.load_state_dict(data['lstm_state'])
            
            logger.info(f"模型已从 {path} 加载")
            return True
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            return False


# 便捷函数
def train_predictor(features_df: pd.DataFrame) -> Tuple[MLPredictor, Dict[str, float]]:
    """快速训练预测器"""
    predictor = MLPredictor()
    results = predictor.fit(features_df)
    return predictor, results


def quick_predict(ohlcv_df: pd.DataFrame, symbol: str = "") -> Optional[PredictionResult]:
    """
    快速预测 (特征生成 + 预测)
    
    注意: 需要先调用 train_predictor 训练模型
    """
    from app.rox_quant.feature_engineer import FeatureEngineer
    
    fe = FeatureEngineer()
    features = fe.generate_features(ohlcv_df)
    
    predictor = MLPredictor()
    if predictor.load():
        return predictor.predict_single(features, symbol)
    return None
