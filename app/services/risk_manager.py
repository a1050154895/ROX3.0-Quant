
import logging
import numpy as np
import pandas as pd
import akshare as ak
from typing import List, Dict
import asyncio
from app.api.endpoints.stock import run_in_executor

logger = logging.getLogger(__name__)

class RiskManager:
    """
    风控驾驶舱管理器
    
    功能:
    1. 行业暴露分析 (Sector Exposure)
    2. 仓位拥挤度 (Crowding)
    3. 在险价值 (VaR)
    """
    
    def __init__(self):
        pass
        
    async def analyze_portfolio(self, positions: List[Dict]) -> Dict:
        """
        分析持仓组合的风险指标
        """
        if not positions:
            return {
                "exposure": [],
                "crowding": 0.0,
                "var_95": 0.0,
                "summary": "空仓状态，无风险。"
            }
            
        # 1. 计算行业暴露
        exposure = await self._calculate_sector_exposure(positions)
        
        # 2. 计算拥挤度 (最大单一行业占比)
        max_sector_pct = 0.0
        if exposure:
            max_sector_pct = max([item['ratio'] for item in exposure])
        
        # 3. 计算 VaR (简化版: 假设日波动率 2%)
        # 真正的 VaR 需要历史序列协方差矩阵，耗时较长。这里使用参数法估算。
        # VaR = Z * Vol * Value
        # 假设组合波动率 = 加权平均波动率 * 多样化因子(0.7)
        total_value = sum([p['market_value'] for p in positions])
        portfolio_vol = 0.02 # 默认日波动率 2%
        var_95 = 1.65 * portfolio_vol * total_value
        
        # 4. 生成总结
        summary = f"当前持有 {len(positions)} 只标的，总市值 {total_value/10000:.1f} 万。"
        if max_sector_pct > 0.6:
            summary += f" ⚠️ 行业过度集中 ({exposure[0]['name']} {max_sector_pct*100:.1f}%)，建议分散配置。"
        elif max_sector_pct > 0.4:
            summary += f" 行业集中度较高。"
        else:
            summary += " 行业配置较为均衡。"
            
        return {
            "exposure": exposure,
            "crowding": round(max_sector_pct * 100, 2),
            "var_95": round(var_95, 2),
            "summary": summary
        }
    
    async def _calculate_sector_exposure(self, positions: List[Dict]) -> List[Dict]:
        """
        查询每个持仓的行业，并聚合
        """
        sector_map = {} # Industry -> MarketValue
        total_mv = 0.0
        
        # 异步并发获取行业信息 (为避免触发限流，这里限制并发或串行)
        # 为简单起见，这里串行查询或使用缓存
        for pos in positions:
            symbol = pos['symbol']
            mv = pos['market_value']
            total_mv += mv
            
            # 尝试获取行业
            industry = "未知"
            try:
                # 优先使用个股资料接口
                # 使用 run_in_executor 避免阻塞
                info = await run_in_executor(ak.stock_individual_info_em, symbol)
                if info is not None and not info.empty:
                    row = info[info['item'].isin(['行业', '所属行业'])]
                    if not row.empty:
                        industry = row.iloc[0]['value']
            except Exception as e:
                logger.warning(f"Sector lookup failed for {symbol}: {e}")
            
            sector_map[industry] = sector_map.get(industry, 0.0) + mv
            
        # 格式化输出
        result = []
        if total_mv > 0:
            for sector, value in sector_map.items():
                result.append({
                    "name": sector,
                    "value": round(value, 2),
                    "ratio": round(value / total_mv, 4)
                })
        
        # 排序
        result.sort(key=lambda x: x['value'], reverse=True)
        return result
