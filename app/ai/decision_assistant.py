import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from typing import Dict, List, Optional, Tuple
import asyncio
from app.analysis.enhanced_signals import get_enhanced_signal_engine
from app.rox_quant.market_analysis import MarketAnalyzer
from app.utils.error_handling import safe_execute, retry
from app.rox_quant.signal_cache import SignalCache

class AIDecisionAssistant:
    def __init__(self):
        self.market_analyzer = MarketAnalyzer()
        self.recommendation_history = []
        self.confidence_threshold = 0.7
        self.signal_cache = SignalCache()
        self.model_parameters = {
            "trend_strength_threshold": 0.3,
            "volatility_threshold": 0.02,
            "signal_strength_weight": 0.6,
            "market_context_weight": 0.4,
            "confidence_boost_factor": 1.1,
            "volatility_penalty_factor": 0.9
        }
    
    @safe_execute
    async def analyze_market_context(self, symbol: str) -> Dict:
        """分析市场环境"""
        # 尝试从缓存获取市场环境分析
        cache_key = f"market_context_{symbol}_{datetime.now().strftime('%Y%m%d%H')}"
        cached_result = self.signal_cache.get(cache_key)
        if cached_result:
            return cached_result
        
        market_data = await self.market_analyzer.get_market_data(symbol)
        if not market_data:
            return {"market_phase": "unknown", "volatility": "low", "trend_strength": 0, "volume_trend": "stable", "momentum": 0}
        
        price_data = market_data.get("price_data", [])
        if len(price_data) < 30:
            return {"market_phase": "unknown", "volatility": "low", "trend_strength": 0, "volume_trend": "stable", "momentum": 0}
        
        prices = [p['close'] for p in price_data]
        volumes = [p.get('volume', 0) for p in price_data]
        returns = np.diff(prices) / prices[:-1]
        
        # 计算更多市场指标
        volatility = np.std(returns)
        trend_strength = np.abs(np.mean(returns) / volatility) if volatility > 0 else 0
        momentum = np.sum(returns[-10:]) / (10 * volatility) if volatility > 0 else 0
        
        # 分析成交量趋势
        volume_trend = "stable"
        if len(volumes) > 20:
            recent_volumes = volumes[-10:]
            past_volumes = volumes[-20:-10]
            volume_change = np.mean(recent_volumes) / np.mean(past_volumes) - 1
            if volume_change > 0.3:
                volume_trend = "increasing"
            elif volume_change < -0.3:
                volume_trend = "decreasing"
        
        # 更精细的市场阶段判断
        market_phase = "neutral"
        if trend_strength > self.model_parameters["trend_strength_threshold"]:
            if np.mean(returns) > 0:
                if momentum > 0.5:
                    market_phase = "strong_uptrend"
                else:
                    market_phase = "uptrend"
            else:
                if momentum < -0.5:
                    market_phase = "strong_downtrend"
                else:
                    market_phase = "downtrend"
        elif volatility > self.model_parameters["volatility_threshold"]:
            market_phase = "volatile"
        
        volatility_level = "low"
        if volatility > 0.03:
            volatility_level = "high"
        elif volatility > 0.015:
            volatility_level = "medium"
        
        result = {
            "market_phase": market_phase,
            "volatility": volatility_level,
            "trend_strength": trend_strength,
            "volatility_value": volatility,
            "volume_trend": volume_trend,
            "momentum": momentum
        }
        
        # 缓存结果，有效期1小时
        self.signal_cache.set(cache_key, result, expire_seconds=3600)
        return result
    
    @safe_execute
    async def generate_trade_recommendation(self, symbol: str, signals: Dict) -> Dict:
        """生成交易建议"""
        market_context = await self.analyze_market_context(symbol)
        
        recommendation = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "market_context": market_context,
            "signals": signals,
            "recommendation": "hold",
            "confidence": 0.5,
            "reasoning": [],
            "position_size": 0,
            "stop_loss": None,
            "take_profit": None,
            "risk_reward_ratio": None,
            "time_horizon": "short_term",
            "signal_strength": 0
        }
        
        # 计算信号强度分数
        buy_score = 0
        sell_score = 0
        buy_signals = []
        sell_signals = []
        
        for signal_name, signal_data in signals.items():
            if signal_name == "basic":
                continue
            
            signal_value = signal_data.get("value", 0)
            if signal_value > 0.7:
                buy_score += signal_value
                buy_signals.append((signal_name, signal_value))
            elif signal_value < -0.7:
                sell_score += abs(signal_value)
                sell_signals.append((signal_name, abs(signal_value)))
        
        # 排序信号强度
        buy_signals.sort(key=lambda x: x[1], reverse=True)
        sell_signals.sort(key=lambda x: x[1], reverse=True)
        
        # 计算总体信号强度
        total_score = buy_score + sell_score
        if total_score > 0:
            recommendation["signal_strength"] = abs(buy_score - sell_score) / total_score
            
            # 基于信号强度和市场环境的综合决策
            if buy_score > sell_score:
                recommendation["recommendation"] = "buy"
                base_confidence = buy_score / total_score
                
                # 详细的推理过程
                for signal_name, strength in buy_signals[:3]:  # 只显示前3个最强信号
                    recommendation["reasoning"].append(f"{signal_name}信号强度: {strength:.2f}")
                
                # 市场环境调整
                if market_context["market_phase"] in ["strong_uptrend", "uptrend"]:
                    base_confidence *= self.model_parameters["confidence_boost_factor"]
                    recommendation["reasoning"].append(f"市场处于{market_context['market_phase']}，增强买入信心")
                elif market_context["market_phase"] in ["strong_downtrend", "downtrend"]:
                    base_confidence *= 0.8
                    recommendation["reasoning"].append(f"市场处于{market_context['market_phase']}，减弱买入信心")
            else:
                recommendation["recommendation"] = "sell"
                base_confidence = sell_score / total_score
                
                # 详细的推理过程
                for signal_name, strength in sell_signals[:3]:  # 只显示前3个最强信号
                    recommendation["reasoning"].append(f"{signal_name}信号强度: {strength:.2f}")
                
                # 市场环境调整
                if market_context["market_phase"] in ["strong_downtrend", "downtrend"]:
                    base_confidence *= self.model_parameters["confidence_boost_factor"]
                    recommendation["reasoning"].append(f"市场处于{market_context['market_phase']}，增强卖出信心")
                elif market_context["market_phase"] in ["strong_uptrend", "uptrend"]:
                    base_confidence *= 0.8
                    recommendation["reasoning"].append(f"市场处于{market_context['market_phase']}，减弱卖出信心")
            
            # 波动调整
            if market_context["volatility"] == "high":
                recommendation["reasoning"].append("市场波动较大，建议谨慎操作")
                base_confidence *= self.model_parameters["volatility_penalty_factor"]
            
            # 成交量趋势调整
            if market_context["volume_trend"] == "increasing":
                if recommendation["recommendation"] == "buy" and market_context["market_phase"] in ["strong_uptrend", "uptrend"]:
                    base_confidence *= 1.05
                    recommendation["reasoning"].append("成交量增加，确认上涨趋势")
                elif recommendation["recommendation"] == "sell" and market_context["market_phase"] in ["strong_downtrend", "downtrend"]:
                    base_confidence *= 1.05
                    recommendation["reasoning"].append("成交量增加，确认下跌趋势")
            
            recommendation["confidence"] = min(base_confidence, 1.0)
        
        # 计算仓位大小和风险参数
        if recommendation["recommendation"] in ["buy", "sell"]:
            if recommendation["confidence"] > self.confidence_threshold:
                # 基于信心和市场波动的仓位大小计算
                volatility_adjustment = 1.0
                if market_context["volatility"] == "high":
                    volatility_adjustment = 0.7
                elif market_context["volatility"] == "medium":
                    volatility_adjustment = 0.9
                
                position_size = min(0.3, recommendation["confidence"] * 0.4 * volatility_adjustment)
                recommendation["position_size"] = position_size
            
            # 计算止损和止盈位
            latest_price = signals.get("basic", {}).get("close", 0)
            if latest_price > 0:
                if recommendation["recommendation"] == "buy":
                    # 基于波动率的止损位
                    stop_loss = latest_price * (1 - 0.03 * market_context.get("volatility_value", 0.02) / 0.02)
                    # 基于风险回报比的止盈位
                    take_profit = latest_price * 1.15
                else:
                    # 做空的止损和止盈
                    stop_loss = latest_price * (1 + 0.03 * market_context.get("volatility_value", 0.02) / 0.02)
                    take_profit = latest_price * 0.85
                
                recommendation["stop_loss"] = stop_loss
                recommendation["take_profit"] = take_profit
                
                # 计算风险回报比
                if stop_loss and take_profit:
                    if recommendation["recommendation"] == "buy":
                        risk = latest_price - stop_loss
                        reward = take_profit - latest_price
                    else:
                        risk = stop_loss - latest_price
                        reward = latest_price - take_profit
                    
                    if risk > 0:
                        recommendation["risk_reward_ratio"] = reward / risk
            
            # 基于市场环境的时间周期建议
            if market_context["momentum"] > 0.8:
                recommendation["time_horizon"] = "medium_term"
            elif market_context["momentum"] < -0.8:
                recommendation["time_horizon"] = "medium_term"
        
        self.recommendation_history.append(recommendation)
        if len(self.recommendation_history) > 200:  # 增加历史记录容量
            self.recommendation_history = self.recommendation_history[-200:]
        
        return recommendation
    
    @safe_execute
    async def get_portfolio_advice(self, portfolio: Dict) -> Dict:
        """获取投资组合建议"""
        advice = {
            "timestamp": datetime.now().isoformat(),
            "portfolio_health": "healthy",
            "diversification_score": 0,
            "risk_level": "medium",
            "suggestions": [],
            "asset_allocation": {},
            "sector_exposure": {},
            "industry_breakdown": {},
            "risk_metrics": {
                "portfolio_beta": 1.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0
            },
            "optimization_opportunities": []
        }
        
        total_value = sum(position.get("value", 0) for position in portfolio.get("positions", []))
        if total_value == 0:
            advice["suggestions"].append("投资组合价值为零，建议开始投资")
            return advice
        
        # 详细的行业和板块分析
        sector_exposure = {}
        industry_breakdown = {}
        for position in portfolio.get("positions", []):
            sector = position.get("sector", "unknown")
            industry = position.get("industry", "unknown")
            value = position.get("value", 0)
            
            sector_exposure[sector] = sector_exposure.get(sector, 0) + value
            industry_breakdown[industry] = industry_breakdown.get(industry, 0) + value
        
        for sector, value in sector_exposure.items():
            advice["sector_exposure"][sector] = value / total_value
        
        for industry, value in industry_breakdown.items():
            advice["industry_breakdown"][industry] = value / total_value
        
        # 更精确的多样化评分
        sector_count = len(sector_exposure)
        industry_count = len(industry_breakdown)
        
        # 基于行业和板块数量的多样化评分
        diversification_score = (min(sector_count / 12, 1.0) * 0.6) + (min(industry_count / 30, 1.0) * 0.4)
        advice["diversification_score"] = diversification_score
        
        # 健康状态评估
        if diversification_score < 0.3:
            advice["portfolio_health"] = "unhealthy"
            advice["suggestions"].append("投资组合行业集中度高，建议增加行业多样性")
        elif diversification_score < 0.6:
            advice["portfolio_health"] = "fair"
            advice["suggestions"].append("投资组合多样性一般，建议适当增加行业覆盖")
        else:
            advice["portfolio_health"] = "healthy"
        
        # 风险评估
        max_sector_exposure = max(advice["sector_exposure"].values(), default=0)
        max_industry_exposure = max(advice["industry_breakdown"].values(), default=0)
        
        if max_sector_exposure > 0.5:
            advice["risk_level"] = "high"
            advice["suggestions"].append(f"单个行业占比过高 ({max_sector_exposure:.2%})，建议降低集中度")
        elif max_sector_exposure > 0.3:
            advice["risk_level"] = "medium-high"
            advice["suggestions"].append(f"单个行业占比较高 ({max_sector_exposure:.2%})，建议适当分散")
        
        if max_industry_exposure > 0.4:
            advice["suggestions"].append(f"单个板块占比过高 ({max_industry_exposure:.2%})，建议降低集中度")
        
        # 现金比例分析
        cash_ratio = portfolio.get("cash", 0) / total_value
        if cash_ratio < 0.1:
            advice["suggestions"].append("现金储备不足，建议保留至少10%的现金以应对市场波动")
        elif cash_ratio > 0.5:
            advice["suggestions"].append("现金储备过多，建议适当增加投资以提高收益潜力")
        
        # 优化机会
        if diversification_score < 0.4:
            advice["optimization_opportunities"].append("增加行业多样性以降低系统性风险")
        
        if cash_ratio < 0.05:
            advice["optimization_opportunities"].append("增加现金储备以提高流动性和应对市场波动的能力")
        
        return advice
    
    @safe_execute
    def analyze_recommendation_history(self) -> Dict:
        """分析建议历史"""
        if len(self.recommendation_history) < 10:
            return {"message": "历史数据不足，无法分析"}
        
        recent_recommendations = self.recommendation_history[-50:]
        
        recommendation_counts = {"buy": 0, "sell": 0, "hold": 0}
        for rec in recent_recommendations:
            rec_type = rec.get("recommendation", "hold")
            if rec_type in recommendation_counts:
                recommendation_counts[rec_type] += 1
        
        # 计算更详细的统计数据
        avg_confidence = np.mean([rec.get("confidence", 0) for rec in recent_recommendations])
        avg_signal_strength = np.mean([rec.get("signal_strength", 0) for rec in recent_recommendations])
        
        # 按市场环境分析建议分布
        market_phase_analysis = {}
        for rec in recent_recommendations:
            market_phase = rec.get("market_context", {}).get("market_phase", "unknown")
            if market_phase not in market_phase_analysis:
                market_phase_analysis[market_phase] = {"buy": 0, "sell": 0, "hold": 0, "total": 0}
            
            rec_type = rec.get("recommendation", "hold")
            market_phase_analysis[market_phase][rec_type] += 1
            market_phase_analysis[market_phase]["total"] += 1
        
        return {
            "total_recommendations": len(recent_recommendations),
            "recommendation_distribution": recommendation_counts,
            "average_confidence": avg_confidence,
            "average_signal_strength": avg_signal_strength,
            "market_phase_analysis": market_phase_analysis,
            "period_covered": f"最近{len(recent_recommendations)}个建议"
        }
    
    @safe_execute
    async def get_market_insights(self) -> Dict:
        """获取市场洞察"""
        insights = {
            "timestamp": datetime.now().isoformat(),
            "market_sentiment": "neutral",
            "sentiment_score": 0.0,
            "key_observations": [],
            "risk_factors": [],
            "opportunities": [],
            "sector_analysis": {},
            "market_ breadth": {
                "advancing": 0,
                "declining": 0,
                "adv_dec_ratio": 1.0
            }
        }
        
        market_data = await self.market_analyzer.get_broad_market_data()
        if market_data:
            indices = market_data.get("indices", {})
            advancing = market_data.get("advancing", 0)
            declining = market_data.get("declining", 0)
            
            insights["market_breadth"]["advancing"] = advancing
            insights["market_breadth"]["declining"] = declining
            if declining > 0:
                insights["market_breadth"]["adv_dec_ratio"] = advancing / declining
            
            # 更精确的市场情绪分析
            if advancing > declining * 1.5:
                insights["market_sentiment"] = "bullish"
                insights["sentiment_score"] = min(advancing / (advancing + declining), 1.0)
                insights["key_observations"].append("市场上涨股票数量显著多于下跌股票")
            elif declining > advancing * 1.5:
                insights["market_sentiment"] = "bearish"
                insights["sentiment_score"] = -min(declining / (advancing + declining), 1.0)
                insights["key_observations"].append("市场下跌股票数量显著多于上涨股票")
            else:
                insights["market_sentiment"] = "neutral"
                insights["sentiment_score"] = 0.0
            
            # 指数分析
            for index_name, index_data in indices.items():
                if "change" in index_data:
                    change = index_data["change"]
                    if abs(change) > 2:
                        direction = "上涨" if change > 0 else "下跌"
                        insights["key_observations"].append(f"{index_name}大幅{direction}{abs(change):.2f}%")
            
            # 行业分析
            sector_performance = market_data.get("sector_performance", {})
            if sector_performance:
                top_sectors = sorted(sector_performance.items(), key=lambda x: x[1], reverse=True)[:3]
                bottom_sectors = sorted(sector_performance.items(), key=lambda x: x[1])[:3]
                
                for sector, performance in top_sectors:
                    insights["sector_analysis"][sector] = {
                        "performance": performance,
                        "rank": "top"
                    }
                
                for sector, performance in bottom_sectors:
                    insights["sector_analysis"][sector] = {
                        "performance": performance,
                        "rank": "bottom"
                    }
        
        # 基于市场情绪的风险和机会分析
        if insights["market_sentiment"] == "bullish":
            insights["risk_factors"].append("市场情绪过于乐观，可能存在回调风险")
            insights["opportunities"].append("关注强势行业的龙头股，顺势而为")
        elif insights["market_sentiment"] == "bearish":
            insights["risk_factors"].append("市场情绪低迷，可能继续下行")
            insights["opportunities"].append("关注防御性行业和超跌优质股")
        else:
            insights["risk_factors"].append("市场方向不明，建议保持谨慎")
            insights["opportunities"].append("关注业绩超预期的个股和行业轮动机会")
        
        # 通用风险因素
        insights["risk_factors"].append("市场波动可能增加，建议设置合理的止损位")
        insights["opportunities"].append("关注基本面良好且技术面走强的个股")
        
        return insights
    
    @safe_execute
    async def get_advanced_market_analysis(self) -> Dict:
        """获取高级市场分析"""
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "market_cycle_analysis": {
                "current_phase": "accumulation",
                "cycle_strength": 0.0
            },
            "liquidity_analysis": {
                "market_liquidity": "adequate",
                "fund_flow": "neutral"
            },
            "volatility_forecast": {
                "short_term": "stable",
                "medium_term": "stable"
            },
            "intermarket_analysis": {
                "equity_bond_ratio": 0.0,
                "commodity_equity_correlation": 0.0
            }
        }
        
        # 这里可以添加更复杂的市场分析逻辑
        # 例如：市场周期分析、流动性分析、波动率预测等
        
        return analysis
    
    @safe_execute
    def get_model_performance(self) -> Dict:
        """获取模型性能评估"""
        if len(self.recommendation_history) < 20:
            return {"message": "历史数据不足，无法评估模型性能"}
        
        recent_recommendations = self.recommendation_history[-100:]
        
        # 计算模型性能指标
        avg_confidence = np.mean([rec.get("confidence", 0) for rec in recent_recommendations])
        avg_signal_strength = np.mean([rec.get("signal_strength", 0) for rec in recent_recommendations])
        recommendation_variety = len(set([rec.get("recommendation", "hold") for rec in recent_recommendations]))
        
        # 计算决策一致性
        if len(recent_recommendations) > 1:
            recommendations = [rec.get("recommendation", "hold") for rec in recent_recommendations]
            consistency = sum(1 for i in range(1, len(recommendations)) if recommendations[i] == recommendations[i-1]) / (len(recommendations) - 1)
        else:
            consistency = 0.0
        
        return {
            "total_predictions": len(recent_recommendations),
            "average_confidence": avg_confidence,
            "average_signal_strength": avg_signal_strength,
            "recommendation_variety": recommendation_variety,
            "decision_consistency": consistency,
            "model_health": "healthy" if avg_confidence > 0.6 else "needs_review"
        }