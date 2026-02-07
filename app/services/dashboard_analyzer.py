# -*- coding: utf-8 -*-
"""
ROX 3.0 Deep Analysis Dashboard Service
Ported from daily_stock_analysis
"""
import logging
import pandas as pd
import numpy as np
import json
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
from enum import Enum
from app.rox_quant.llm import AIClient
from app.analysis.china_analyst import china_analyst

logger = logging.getLogger(__name__)

# ==========================================
# Enums and Data Types
# ==========================================

class TrendStatus(Enum):
    STRONG_BULL = "强势多头"
    BULL = "多头排列"
    WEAK_BULL = "弱势多头"
    CONSOLIDATION = "盘整"
    WEAK_BEAR = "弱势空头"
    BEAR = "空头排列"
    STRONG_BEAR = "强势空头"

class BuySignal(Enum):
    STRONG_BUY = "强烈买入"
    BUY = "买入"
    HOLD = "持有"
    WAIT = "观望"
    SELL = "卖出"
    STRONG_SELL = "强烈卖出"

@dataclass
class TrendAnalysisResult:
    trend_status: str = TrendStatus.CONSOLIDATION.value
    ma_alignment: str = ""
    trend_strength: float = 0.0
    ma5: float = 0.0
    ma10: float = 0.0
    ma20: float = 0.0
    bias_ma5: float = 0.0
    buy_signal: str = BuySignal.WAIT.value
    signal_score: int = 0
    signal_reasons: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)

# ==========================================
# Technical Analyzer (Ported Logic)
# ==========================================

class StockTrendAnalyzer:
    """
    Based on: MA5>MA10>MA20
    """
    def analyze(self, df: pd.DataFrame) -> TrendAnalysisResult:
        result = TrendAnalysisResult()
        
        if df is None or df.empty or len(df) < 20:
            return result
        
        # Ensure sorted
        df = df.sort_values('date').reset_index(drop=True)
        
        # Calc MA
        df['MA5'] = df['close'].rolling(window=5).mean()
        df['MA10'] = df['close'].rolling(window=10).mean()
        df['MA20'] = df['close'].rolling(window=20).mean()
        
        latest = df.iloc[-1]
        result.ma5 = float(latest['MA5'])
        result.ma10 = float(latest['MA10'])
        result.ma20 = float(latest['MA20'])
        price = float(latest['close'])
        
        # Trend Status
        if result.ma5 > result.ma10 > result.ma20:
            result.trend_status = TrendStatus.BULL.value
            result.ma_alignment = "多头排列 MA5>MA10>MA20"
            result.trend_strength = 75
            # Check for strong bull (divergence)
            if len(df) >= 5:
                prev = df.iloc[-5]
                if prev['MA20'] > 0:
                    prev_spread = (prev['MA5'] - prev['MA20']) / prev['MA20']
                    curr_spread = (result.ma5 - result.ma20) / result.ma20
                    if curr_spread > prev_spread and curr_spread > 0.05:
                        result.trend_status = TrendStatus.STRONG_BULL.value
                        result.trend_strength = 90
        elif result.ma5 < result.ma10 < result.ma20:
            result.trend_status = TrendStatus.BEAR.value
            result.ma_alignment = "空头排列"
            result.trend_strength = 25
        else:
            result.trend_status = TrendStatus.CONSOLIDATION.value
            result.ma_alignment = "均线纠缠"
            result.trend_strength = 50
            
        # Bias
        if result.ma5 > 0:
            result.bias_ma5 = (price - result.ma5) / result.ma5 * 100
            
        # Scoring
        score = 0
        reasons = []
        risks = []
        
        # Trend Score
        if result.trend_status == TrendStatus.STRONG_BULL.value: score += 40
        elif result.trend_status == TrendStatus.BULL.value: score += 30
        elif result.trend_status == TrendStatus.CONSOLIDATION.value: score += 10
        
        # Bias Score
        bias = result.bias_ma5
        if abs(bias) < 2: 
            score += 20
            reasons.append("股价贴近MA5，乖离率低")
        elif bias > 5:
            score -= 10
            risks.append("乖离率>5%，有回调风险")
            
        # Signal
        result.signal_score = max(0, min(100, score + 40)) # Base 40
        result.signal_reasons = reasons
        result.risk_factors = risks
        
        if result.signal_score > 80: result.buy_signal = BuySignal.STRONG_BUY.value
        elif result.signal_score > 60: result.buy_signal = BuySignal.BUY.value
        elif result.signal_score < 40: result.buy_signal = BuySignal.SELL.value
        
        return result

# ==========================================
# Dashboard Service
# ==========================================

class DashboardAnalyzer:
    
    SYSTEM_PROMPT = """你是一位专注于趋势交易的 A 股投资分析师，请根据提供的数据生成【决策仪表盘】JSON。

## 核心交易理念
1. **严进策略**：不追高 (乖离率>5%不买)
2. **趋势交易**：主做 MA5>MA10>MA20 多头排列
3. **筹码结构**：关注获利比例和筹码集中度

## 输出格式 (必须是合法的 JSON)
```json
{
    "sentiment_score": 0-100,
    "trend_prediction": "看多/震荡/看空",
    "operation_advice": "买入/持有/卖出/观望",
    "confidence_level": "高/中/低",
    "dashboard": {
        "core_conclusion": {
            "one_sentence": "一句话核心结论",
            "signal_type": "🟢买入/🟡持有/🔴卖出"
        },
        "battle_plan": {
            "sniper_points": {
                "ideal_buy": "价格",
                "stop_loss": "价格",
                "take_profit": "价格"
            },
            "action_checklist": [
                "✅ 检查项1",
                "⚠️ 检查项2"
            ]
        },
        "intelligence": {
            "risk_alerts": ["风险1"],
            "positive_catalysts": ["利好1"]
        }
    },
    "technical_analysis": "技术面分析文本",
    "fundamental_analysis": "基本面分析文本",
    "chip_analysis": "筹码分析文本"
}
```
"""

    def __init__(self):
        self.ai = AIClient()
        self.tech_analyzer = StockTrendAnalyzer()

    async def analyze(self, symbol: str, stock_name: str, df: pd.DataFrame, chip_data=None, realtime=None) -> Dict:
        # 1. Technical Analysis
        tech_result = self.tech_analyzer.analyze(df)
        
        # 2. Build Context
        context = {
            "code": symbol,
            "name": stock_name,
            "current_price": realtime.get('price') if realtime else "N/A",
            "technicals": asdict(tech_result),
            "chip_distribution": chip_data, # Expects dict
            "realtime_indicators": realtime # Volume ratio, turnover
        }
        
        # 3. Call AI
        # Check if A-Share (6 digits) -> Use Specialized ChinaAnalyst
        if len(symbol) == 6 and symbol.isdigit():
            try:
                price_val = 0.0
                if realtime and 'price' in realtime:
                    try:
                        price_val = float(realtime['price'])
                    except:
                        pass
                
                # Delegate to ChinaAnalyst
                logger.info(f"Using ChinaAnalyst for {stock_name} ({symbol})")
                return await china_analyst.analyze_stock(symbol, stock_name, price_val, context)
            except Exception as e:
                logger.error(f"ChinaAnalyst failed, falling back to default: {e}")
                # Fallback to default logic below

        prompt = f"""
分析对象: {stock_name} ({symbol})
当期数据: {json.dumps(context, ensure_ascii=False, indent=2)}

请根据上述数据，生成决策仪表盘。
"""
        
        try:
            client = self.ai.get_client()
            if not client:
                return {"error": "AI client not configured"}
                
            response = await client.chat.completions.create(
                model="deepseek-chat", # Or default from config
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            # Try to parse JSON
            try:
                # Remove markdown fences if present
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0]
                    
                result_json = json.loads(content)
                return result_json
            except json.JSONDecodeError:
                logger.error("Failed to parse AI response JSON")
                return {"error": "AI Response Parse Error", "raw_content": content}
                
        except Exception as e:
            logger.error(f"Dashboard analysis failed: {e}")
            return {"error": str(e)}

dashboard_analyzer = DashboardAnalyzer()
