
import logging
import pandas as pd
import asyncio
import akshare as ak
from datetime import datetime, timedelta
from typing import List, Dict, Any

from app.services.portfolio_manager import PortfolioManager
from app.api.endpoints.stock import run_in_executor, _normalize_code
from app.rox_quant.signal_fusion import SignalFusion, SignalType

logger = logging.getLogger(__name__)

class AutoTrader:
    """
    自动交易机器人
    
    1. 获取K线数据
    2. 调用 SignalFusion 生成信号
    3. 调用 PortfolioManager 执行交易
    """
    
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.pm = PortfolioManager(user_id)
        # 初始化信号融合器 (可能比较耗时，考虑单例或缓存)
        self.fusion = SignalFusion() 
        logger.info(f"AutoTrader initialized for user {user_id}")

    async def run_batch(self, symbols: List[str]) -> Dict[str, Any]:
        """
        批量运行自动交易逻辑
        """
        results = []
        logs = []
        
        for symbol in symbols:
            symbol = _normalize_code(symbol)
            log_entry = f"[{symbol}] Analysis: "
            
            try:
                # 1. 获取 OHLC 数据 (最近 200 天，用于指标计算)
                end_date = datetime.now().strftime("%Y%m%d")
                start_date = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")
                
                # 减少网络请求频率
                await asyncio.sleep(0.5) 
                
                ohlc = await run_in_executor(
                    ak.stock_zh_a_hist, 
                    symbol, "daily", start_date, end_date, "qfq"
                )
                
                if ohlc is None or ohlc.empty or len(ohlc) < 60:
                    log_entry += "Insufficient data."
                    logs.append(log_entry)
                    continue
                    
                # 统一列名
                rename_map = {
                    "日期": "date", "开盘": "open", "收盘": "close", 
                    "最高": "high", "最低": "low", "成交量": "volume"
                }
                ohlc = ohlc.rename(columns=rename_map)
                
                # 2. 生成信号
                # 使用 fuse_with_advanced_system (7信号系统)
                signal = self.fusion.fuse_with_advanced_system(ohlc, symbol)
                
                log_entry += f"{signal.signal_type.name} (Conf={signal.confidence:.2f}, Reason={signal.reason})"
                
                # 3. 执行交易
                # 简单策略: Strong Buy -> 买入 200股; Buy -> 买入 100股
                # Sell -> 卖出 50%; Strong Sell -> 清仓
                
                price = ohlc['close'].iloc[-1]
                name = symbol # 暂无名称无需再查
                executed = False
                
                if signal.signal_type == SignalType.STRONG_BUY:
                    qty = 200
                    if self.pm.execute_order(symbol, name, "buy", price, qty, reason=signal.reason):
                        log_entry += " -> BUY 200 EXECUTED"
                        executed = True
                elif signal.signal_type == SignalType.BUY:
                    qty = 100
                    if self.pm.execute_order(symbol, name, "buy", price, qty, reason=signal.reason):
                        log_entry += " -> BUY 100 EXECUTED"
                        executed = True
                        
                elif signal.signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
                    # 查询持仓
                    positions = self.pm.get_positions()
                    target_pos = next((p for p in positions if p['symbol'] == symbol), None)
                    
                    if target_pos:
                        curr_qty = target_pos['quantity']
                        sell_qty = curr_qty if signal.signal_type == SignalType.STRONG_SELL else int(curr_qty / 2)
                        
                        if sell_qty > 0:
                            if self.pm.execute_order(symbol, name, "sell", price, sell_qty, reason=signal.reason):
                                log_entry += f" -> SELL {sell_qty} EXECUTED"
                                executed = True
                    else:
                        log_entry += " -> No position to sell"

                results.append({
                    "symbol": symbol,
                    "signal": signal.signal_type.name,
                    "confidence": signal.confidence,
                    "executed": executed
                })
                
            except Exception as e:
                logger.error(f"Auto trade failed for {symbol}: {e}")
                log_entry += f" ERROR: {str(e)}"
            
            logs.append(log_entry)
            
        return {
            "summary": f"Processed {len(symbols)} symbols",
            "results": results,
            "logs": logs
        }
