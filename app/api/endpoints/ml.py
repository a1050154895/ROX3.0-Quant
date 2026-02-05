"""
ML 预测 API 端点
提供模型训练、预测、状态查询接口
"""

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging
import asyncio

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ml", tags=["ml"])


class TrainRequest(BaseModel):
    """训练请求"""
    symbols: List[str] = ["600519", "000001", "300750", "601318", "000858"]
    days: int = 500
    test_size: float = 0.2


class PredictRequest(BaseModel):
    """预测请求"""
    symbol: str
    model: str = "ensemble"  # ensemble, random_forest, gradient_boosting, lstm


class TrainResponse(BaseModel):
    """训练响应"""
    status: str
    models_accuracy: Dict[str, float]
    feature_count: int
    sample_count: int


class PredictResponse(BaseModel):
    """预测响应"""
    symbol: str
    prediction: str  # "涨" or "跌"
    probability: float
    confidence: float
    model: str
    features_used: int


# 全局状态
_training_status = {"is_training": False, "progress": "", "last_result": None}


@router.get("/status")
async def get_ml_status():
    """获取 ML 模块状态"""
    from app.rox_quant.ml_predictor import MLPredictor, HAS_SKLEARN, HAS_TORCH
    
    predictor = MLPredictor()
    is_loaded = predictor.load()
    
    return {
        "sklearn_available": HAS_SKLEARN,
        "torch_available": HAS_TORCH,
        "model_loaded": is_loaded,
        "is_training": _training_status["is_training"],
        "training_progress": _training_status["progress"],
        "last_train_result": _training_status["last_result"]
    }


@router.post("/train", response_model=TrainResponse)
async def train_models(req: TrainRequest, background_tasks: BackgroundTasks):
    """
    训练 ML 模型
    
    这是一个后台任务，会立即返回，训练在后台进行。
    使用 /ml/status 查询训练进度。
    """
    if _training_status["is_training"]:
        raise HTTPException(status_code=400, detail="已有训练任务在进行中")
    
    # 后台训练
    background_tasks.add_task(_train_models_task, req)
    
    return TrainResponse(
        status="started",
        models_accuracy={},
        feature_count=0,
        sample_count=0
    )


async def _train_models_task(req: TrainRequest):
    """后台训练任务"""
    global _training_status
    
    _training_status["is_training"] = True
    _training_status["progress"] = "准备数据..."
    
    try:
        import pandas as pd
        from app.rox_quant.feature_engineer import FeatureEngineer
        from app.rox_quant.ml_predictor import MLPredictor
        from app.rox_quant.data_provider import DataProvider
        
        provider = DataProvider()
        fe = FeatureEngineer()
        
        all_features = []
        
        for i, symbol in enumerate(req.symbols):
            _training_status["progress"] = f"获取数据 {i+1}/{len(req.symbols)}: {symbol}"
            
            try:
                # 获取历史数据
                history = provider.get_history(symbol, days=req.days)
                if not history or len(history) < 60:
                    continue
                
                # 转换为 DataFrame
                df = pd.DataFrame([{
                    'date': p.date,
                    'open': p.open or p.close,
                    'high': p.high or p.close,
                    'low': p.low or p.close,
                    'close': p.close,
                    'volume': p.volume or 0
                } for p in history])
                
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
                
                # 生成特征
                features = fe.generate_features(df)
                if not features.empty:
                    all_features.append(features)
                    
            except Exception as e:
                logger.error(f"处理 {symbol} 失败: {e}")
        
        if not all_features:
            _training_status["progress"] = "无有效数据"
            _training_status["is_training"] = False
            return
        
        # 合并所有数据
        _training_status["progress"] = "合并数据..."
        combined = pd.concat(all_features, ignore_index=True)
        
        # 训练模型
        _training_status["progress"] = "训练模型..."
        predictor = MLPredictor()
        results = predictor.fit(combined, test_size=req.test_size)
        
        # 保存模型
        predictor.save()
        
        _training_status["last_result"] = {
            "models_accuracy": results,
            "feature_count": len(fe.get_feature_names()),
            "sample_count": len(combined)
        }
        _training_status["progress"] = "训练完成"
        
    except Exception as e:
        logger.error(f"训练失败: {e}")
        _training_status["progress"] = f"训练失败: {e}"
    finally:
        _training_status["is_training"] = False


@router.post("/predict", response_model=PredictResponse)
async def predict_stock(req: PredictRequest):
    """
    预测股票方向
    
    需要先训练模型 (调用 /ml/train)
    """
    try:
        import pandas as pd
        from app.rox_quant.feature_engineer import FeatureEngineer
        from app.rox_quant.ml_predictor import MLPredictor
        from app.rox_quant.data_provider import DataProvider
        
        # 加载模型
        predictor = MLPredictor()
        if not predictor.load():
            raise HTTPException(status_code=400, detail="模型未训练，请先调用 /ml/train")
        
        # 获取数据
        provider = DataProvider()
        history = provider.get_history(req.symbol, days=120)
        
        if not history or len(history) < 60:
            raise HTTPException(status_code=400, detail="数据不足")
        
        # 转换为 DataFrame
        df = pd.DataFrame([{
            'date': p.date,
            'open': p.open or p.close,
            'high': p.high or p.close,
            'low': p.low or p.close,
            'close': p.close,
            'volume': p.volume or 0
        } for p in history])
        
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        
        # 生成特征
        fe = FeatureEngineer()
        features = fe.generate_features(df)
        
        if features.empty:
            raise HTTPException(status_code=400, detail="特征生成失败")
        
        # 预测
        result = predictor.predict_single(features, req.symbol, req.model)
        
        if result is None:
            raise HTTPException(status_code=500, detail="预测失败")
        
        return PredictResponse(
            symbol=req.symbol,
            prediction="涨" if result.prediction == 1 else "跌",
            probability=result.probability,
            confidence=result.confidence,
            model=result.model_name,
            features_used=result.features_used
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"预测失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/feature-importance")
async def get_feature_importance():
    """获取特征重要性排名"""
    from app.rox_quant.ml_predictor import MLPredictor
    
    predictor = MLPredictor()
    if not predictor.load():
        raise HTTPException(status_code=400, detail="模型未训练")
    
    importance = predictor.get_feature_importance()
    
    # 取前 20 个
    top_20 = dict(list(importance.items())[:20])
    
    return {
        "feature_importance": top_20,
        "total_features": len(importance)
    }


@router.get("/signal-performance")
async def get_signal_performance():
    """获取信号历史表现"""
    from app.rox_quant.adaptive_fusion import get_adaptive_fusion
    
    fusion = get_adaptive_fusion()
    report = fusion.get_performance_report()
    
    return {
        "performances": report.to_dict(orient="records") if not report.empty else [],
        "signal_count": len(report)
    }
