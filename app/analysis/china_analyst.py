# -*- coding: utf-8 -*-
"""
ROX 3.0 China Market Analyst (A-Share Specialized)
Ported and adapted from TradingAgents-CN
"""
import logging
import pandas as pd
import akshare as ak
import json
import asyncio
from typing import Dict, Any, Optional

from app.rox_quant.llm import AIClient

logger = logging.getLogger(__name__)

class ChinaAnalyst:
    """
    Specialized Analyst for China A-Shares.
    Integrates fundamental data, industry analysis, and strict professional prompting.
    """

    SYSTEM_PROMPT = """你是一位专业的中国股市分析师，专门分析A股市场。您具备深厚的中国股市知识和丰富的本土投资经验。

您的专业领域包括：
1. **A股市场分析**: 深度理解A股的独特性，包括涨跌停制度、T+1交易、融资融券等
2. **中国经济政策**: 熟悉货币政策、财政政策对股市的影响机制
3. **行业板块轮动**: 掌握中国特色的板块轮动规律和热点切换
4. **监管环境**: 了解证监会政策、退市制度、注册制等监管变化
5. **资金面分析**: 分析北向资金、主力资金流向

分析重点：
- **数据驱动**: 所有观点必须基于提供的真实数据，严禁编造数据。
- **基本面**: 关注营收、净利润、ROE、毛利率等核心财务指标及其同比环比变化。
- **估值**: 结合PE(TTM)、PB、PS等估值指标，判断当前估值水位。
- **风险提示**: 明确指出潜在的财务风险、行业风险或政策风险。

请基于提供的数据，生成【决策仪表盘】JSON。

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
                "ideal_buy": "价格/区间",
                "stop_loss": "价格/区间",
                "take_profit": "价格/区间"
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
    "technical_analysis": "技术面分析文本 (结合均线、成交量等)",
    "fundamental_analysis": "基本面分析文本 (结合财务指标、行业地位)",
    "chip_analysis": "筹码与资金分析文本"
}
```
"""

    def __init__(self):
        self.ai = AIClient()

    async def analyze_stock(self, symbol: str, stock_name: str, price: float = 0.0, technique_data: Dict = None) -> Dict[str, Any]:
        """
        Main entry point for analyzing an A-share stock.
        """
        try:
            # 1. Fetch Fundamental Data
            fundamentals = await self._get_fundamentals(symbol, price)
            
            # 2. Construct Prompt
            prompt = self._construct_prompt(symbol, stock_name, price, fundamentals, technique_data)
            
            # 3. Call LLM
            report_json = await self._call_llm(prompt)
            
            return report_json

        except Exception as e:
            logger.error(f"ChinaAnalyst analysis failed for {symbol}: {e}")
            return {"error": str(e)}

    async def _get_fundamentals(self, symbol: str, current_price: float) -> Dict[str, Any]:
        """
        Fetch and calculate fundamental metrics using AkShare.
        """
        loop = asyncio.get_event_loop()
        code = symbol[-6:] # Ensure 6 digits
        
        metrics = {
            "pe_ttm": "N/A",
            "pb": "N/A", 
            "total_mv": "N/A",
            "roe": "N/A",
            "gross_margin": "N/A",
            "net_margin": "N/A",
            "revenue_growth": "N/A",
            "profit_growth": "N/A",
            "industry": "N/A"
        }

        try:
            # 1. Fetch Basic Info for Market Cap & Industry
            # ak.stock_individual_info_em(symbol="000001")
            stock_info = await loop.run_in_executor(None, lambda: ak.stock_individual_info_em(symbol=code))
            
            if stock_info is not None and not stock_info.empty:
                info_dict = dict(zip(stock_info['item'], stock_info['value']))
                metrics['industry'] = str(info_dict.get('行业', 'N/A'))
                metrics['total_mv'] = str(info_dict.get('总市值', 'N/A'))
            
            # 2. Fetch Financial Indicators
            fin_df = await loop.run_in_executor(None, lambda: ak.stock_financial_analysis_indicator(symbol=code))
            
            if fin_df is not None and not fin_df.empty:
                # Latest report is usually first row
                latest = fin_df.iloc[0]
                
                # Helper to extract value safely with suffix
                def get_val_suffix(val, suffix=""):
                    try:
                        return f"{float(val):.2f}{suffix}"
                    except:
                        return str(val) if pd.notna(val) else "N/A"

                # Check columns existence
                col_map = {
                    'gross_margin': ['销售毛利率(%)'],
                    'net_margin': ['销售净利率(%)'],
                    'roe': ['净资产收益率(%)', '加权净资产收益率(%)'],
                    'revenue_growth': ['主营业务收入增长率(%)', '营业收入同比增长率(%)'],
                    'profit_growth': ['净利润增长率(%)', '归属净利润同比增长率(%)']
                }
                
                for key, patterns in col_map.items():
                    for p in patterns:
                        if p in latest:
                            metrics[key] = get_val_suffix(latest[p], "%")
                            break
            
            # 3. Fetch Valuation Metrics (PE/PB) using Spot Data (reliable fallback)
            # ak.stock_zh_a_spot_em() returns all stocks, we filter.
            # This might be heavy (5000 rows) but is reliable for PE-TTM/PB.
            try:
                spot_df = await loop.run_in_executor(None, lambda: ak.stock_zh_a_spot_em())
                if spot_df is not None and not spot_df.empty:
                    # Filter by code
                    # Columns: 序号, 代码, 名称, 最新价, 涨跌幅, 涨跌额, 成交量, 成交额, 振幅, 最高, 最低, 今开, 昨收, 量比, 换手率, 市盈率-动态, 市净率
                    # Code in spot_df usually doesn't have prefix, e.g. "000001"
                    target = spot_df[spot_df['代码'] == code]
                    if not target.empty:
                        row = target.iloc[0]
                        metrics['pe_ttm'] = get_val_suffix(row.get('市盈率-动态'), "倍")
                        metrics['pb'] = get_val_suffix(row.get('市净率'), "倍")
                        
                        # Use spot price if passed price is 0
                        if current_price <= 0 and '最新价' in row:
                             try:
                                 current_price = float(row['最新价'])
                             except:
                                 pass
            except Exception as e:
                logger.warning(f"Spot data fetch failed: {e}")

        except Exception as e:
            logger.warning(f"Error fetching fundamentals for {symbol}: {e}")

        return metrics

    def _construct_prompt(self, symbol: str, stock_name: str, price: float, fundamentals: Dict[str, Any], technique_data: Dict = None) -> str:
        # Convert dict to nicely formatted string if needed, or just dump JSON
        tech_str = json.dumps(technique_data, ensure_ascii=False, indent=2) if technique_data else "暂无"
        
        return f"""
分析对象: {stock_name} ({symbol})
当前价格: {price}

【数据面板】
1. 核心财务指标:
   - 行业: {fundamentals.get('industry')}
   - 总市值: {fundamentals.get('total_mv')}
   - 市盈率(TTM): {fundamentals.get('pe_ttm')}
   - 市净率(PB): {fundamentals.get('pb')}
   - ROE: {fundamentals.get('roe')}
   - 毛利率: {fundamentals.get('gross_margin')}
   - 净利率: {fundamentals.get('net_margin')}
   - 营收增长率: {fundamentals.get('revenue_growth')}
   - 净利增长率: {fundamentals.get('profit_growth')}

2. 技术面概要:
{tech_str}

请综合上述基本面和技术面数据，严格按照JSON格式生成决策仪表盘。
"""

    async def _call_llm(self, prompt: str) -> Dict[str, Any]:
        client = self.ai.get_client()
        if not client:
            return {"error": "AI Client not configured."}
        
        try:
            response = await client.chat.completions.create(
                model="deepseek-chat", 
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            
            # Clean markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            return json.loads(content)
        except Exception as e:
            logger.error(f"LLM Call failed: {e}")
            return {"error": "Analysis generation failed", "raw": str(e)}

china_analyst = ChinaAnalyst()
