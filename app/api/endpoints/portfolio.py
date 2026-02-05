
from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any
import logging
from datetime import datetime, timedelta
import akshare as ak
import pandas as pd
from app.services.portfolio_manager import PortfolioManager

# 模拟获取当前用户 ID，实际应从 Token 解析
async def get_current_user_id():
    return 1  # 默认用户 ID

router = APIRouter()
logger = logging.getLogger(__name__)

def _normalize_code(code: str) -> str:
    """统一为 6 位股票代码"""
    code = str(code).strip()
    if len(code) > 6: code = code[-6:]
    return code.zfill(6)

@router.get("/replay/chart/{symbol}")
async def get_replay_chart(symbol: str, user_id: int = Depends(get_current_user_id)):
    """
    获取复盘图表数据：
    包含最近半年的 K 线数据和该股票的买卖点标注。
    """
    try:
        symbol = _normalize_code(symbol)
        
        # 1. 获取交易记录
        pm = PortfolioManager(user_id=user_id)
        trades = pm.get_trades_history(symbol=symbol)
        
        # 2. 获取K线数据 (最近180天)
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=180)).strftime("%Y%m%d")
        
        # 运行在 executor 中避免阻塞
        loop = asyncio.get_event_loop()
        try:
            hist = await loop.run_in_executor(
                None, 
                lambda: ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
            )
        except Exception as e:
            logger.warning(f"Fetch kline failed: {e}")
            hist = pd.DataFrame()

        kline_data = []
        if hist is not None and not hist.empty:
            # 统一列名: 日期, 开盘, 收盘, 最高, 最低, 成交量
            # akshare 返回列名通常是中文
            rename_map = {
                "日期": "date", "开盘": "open", "收盘": "close", 
                "最高": "high", "最低": "low", "成交量": "volume"
            }
            hist = hist.rename(columns=rename_map)
            # 确保日期格式统一 YYYY-MM-DD
            hist['date'] = pd.to_datetime(hist['date']).dt.strftime('%Y-%m-%d')
            kline_data = hist[['date', 'open', 'close', 'high', 'low', 'volume']].to_dict(orient='records')
            
        return {
            "symbol": symbol,
            "kline": kline_data,
            "trades": trades
        }
    except Exception as e:
        logger.error(f"Replay chart error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/auto-trade")
async def trigger_auto_trade(
    payload: Dict[str, List[str]], 
    user_id: int = Depends(get_current_user_id)
):
    """
    触发自动交易
    Payload: { "symbols": ["600519", "000001"] }
    """
    try:
        symbols = payload.get("symbols", [])
        if not symbols:
             raise HTTPException(status_code=400, detail="Symbol list is empty")
             
        from app.services.auto_trader import AutoTrader
        trader = AutoTrader(user_id)
        result = await trader.run_batch(symbols)
        return result
        
    except Exception as e:
        logger.error(f"Auto trade trigger failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/risk/dashboard")
async def get_risk_dashboard(user_id: int = Depends(get_current_user_id)):
    """
    风控驾驶舱数据：
    包含行业暴露、拥挤度、VaR等指标。
    """
    try:
        pm = PortfolioManager(user_id=user_id)
        positions = pm.get_positions()
        
        from app.services.risk_manager import RiskManager
        rm = RiskManager()
        risk_data = await rm.analyze_portfolio(positions)
        
        return risk_data
    except Exception as e:
        logger.error(f"Risk dashboard error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/summary", response_model=Dict[str, Any])
async def get_portfolio_summary(user_id: int = Depends(get_current_user_id)):
    """
    获取模拟账户概览
    """
    try:
        pm = PortfolioManager(user_id=user_id)
        summary = pm.get_account_summary()
        if not summary:
            # 自动初始化
            return {"message": "Account initialized", "balance": 1000000.0}
        return summary
    except Exception as e:
        logger.error(f"Failed to get portfolio summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/positions", response_model=List[Dict[str, Any]])
async def get_portfolio_positions(user_id: int = Depends(get_current_user_id)):
    """
    获取当前持仓
    """
    try:
        pm = PortfolioManager(user_id=user_id)
        return pm.get_positions()
    except Exception as e:
        logger.error(f"Failed to get positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/order")
async def place_order(
    symbol: str, 
    side: str, 
    quantity: int, 
    price: float, 
    name: str = "",
    user_id: int = Depends(get_current_user_id)
):
    """
    手动下单 (测试用)
    """
    try:
        pm = PortfolioManager(user_id=user_id)
        success = pm.execute_order(symbol, name, side, price, quantity, reason="Manual Order")
        if success:
            return {"status": "success", "message": f"Order {side} {symbol} filled"}
        else:
            raise HTTPException(status_code=400, detail="Order failed (Insufficient funds or holdings)")
    except Exception as e:
        logger.error(f"Order placement failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
