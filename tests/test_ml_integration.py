
import sys
import os
import pandas as pd
import numpy as np
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加项目根目录到 sys.path
sys.path.insert(0, os.path.abspath('.'))

def test_feature_engineering():
    logger.info("Running feature engineering test...")
    try:
        from app.rox_quant.feature_engineer import FeatureEngineer
        
        # 创建模拟 OHLCV 数据
        dates = pd.date_range(start='2023-01-01', periods=100)
        df = pd.DataFrame({
            'open': np.random.uniform(100, 110, 100),
            'high': np.random.uniform(105, 115, 100),
            'low': np.random.uniform(95, 105, 100),
            'close': np.random.uniform(100, 110, 100),
            'volume': np.random.randint(1000, 5000, 100)
        }, index=dates)
        
        fe = FeatureEngineer()
        features = fe.generate_features(df)
        
        if not features.empty and len(features.columns) > 10:
            logger.info(f"✓ Feature engineering successful. Generated {len(features.columns)} features.")
            return features
        else:
            logger.error("Feature engineering failed or generated empty features.")
            return None
    except ImportError as e:
        logger.error(f"Import error: {e}")
        return None
    except Exception as e:
        logger.error(f"Feature engineering error: {e}")
        return None

def test_ml_predictor(features):
    logger.info("Running ML predictor test...")
    if features is None:
        logger.warning("Skipping ML predictor test due to missing features.")
        return

    try:
        from app.rox_quant.ml_predictor import MLPredictor
        
        predictor = MLPredictor(model_dir="./models_test")
        
        # 训练
        logger.info("Training dummy model...")
        results = predictor.fit(features, target_col='target_direction', test_size=0.2)
        logger.info(f"Training results: {results}")
        
        # 保存
        predictor.save("test_model.pkl")
        
        # 加载
        predictor2 = MLPredictor(model_dir="./models_test")
        if predictor2.load("test_model.pkl"):
            logger.info("✓ Model save/load successful.")
            
            # 预测
            preds = predictor2.predict(features.tail(5))
            if preds:
                logger.info(f"✓ Prediction successful. Sample: {preds[0]}")
            else:
                logger.error("Prediction returned empty result.")
        else:
            logger.error("Model load failed.")
            
    except ImportError as e:
         logger.error(f"Import error: {e}")
    except Exception as e:
        logger.error(f"ML predictor error: {e}")

def test_adaptive_fusion():
    logger.info("Running adaptive fusion test...")
    try:
        from app.rox_quant.adaptive_fusion import AdaptiveSignalFusion
        
        fusion = AdaptiveSignalFusion(save_dir="./models_test/fusion")
        
        # 模拟信号更新
        signals = ["MACD", "RSI", "MA"]
        for i in range(50):
            # MACD 表现好 (80% 准确)
            fusion.update("MACD", 1, 1 if i % 5 != 0 else 0)
            # RSI 表现差 (20% 准确)
            fusion.update("RSI", 1, 0 if i % 5 != 0 else 1)
            # MA 随机 (50%)
            fusion.update("MA", 1, 1 if i % 2 == 0 else 0)
            
        weights = fusion.get_weights()
        logger.info(f"Learned weights: {weights}")
        
        if weights["MACD"] > weights["RSI"]:
             logger.info("✓ Adaptive fusion logic verified (MACD weight > RSI weight).")
        else:
             logger.error("Adaptive fusion logic failed.")
        
        fusion.save()
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
    except Exception as e:
        logger.error(f"Adaptive fusion error: {e}")

if __name__ == "__main__":
    if not os.path.exists("./models_test"):
        os.makedirs("./models_test")
        
    features = test_feature_engineering()
    test_ml_predictor(features)
    test_adaptive_fusion()
    
    # Clean up
    import shutil
    if os.path.exists("./models_test"):
        shutil.rmtree("./models_test")
    logger.info("Test cleanup complete.")
