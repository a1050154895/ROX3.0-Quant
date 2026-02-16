from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Optional
from app.auth import get_current_user, User
from app.db import (
    get_db, get_accounts, create_trade, close_trade, get_trades, get_history,
    get_psychology_stats, get_watchlist, add_watchlist, remove_watchlist, clear_history,
    get_open_trades_with_risk, create_condition_order, cancel_condition_order,
    get_pending_condition_orders, fill_condition_order,
    DB_PATH
)

router = APIRouter()
logger = logging.getLogger(__name__)


def _get_current_price(symbol: str) -> Optional[float]:
    """获取股票现价（新浪），供风控/条件单检查用"""
    import re
    try:
        c = str(symbol).strip().zfill(6)
        prefix = "sh" if c.startswith(("6", "5", "9")) or c.startswith("688") else "sz"
        import requests
        url = f"http://hq.sinajs.cn/list={prefix}{c}"
        r = requests.get(url, headers={"Referer": "http://finance.sina.com.cn/"}, timeout=3)
        m = re.search(r'"([^"]+)"', r.text)
        if m and "," in m.group(1):
            parts = m.group(1).split(",")
            if len(parts) > 3:
                return float(parts[3])
    except Exception:
        pass
    return None


class TradeRequest(BaseModel):
    symbol: str
    account_type: str = "sim"
    name: str = "Unknown"
    side: str
    open_price: float
    open_quantity: int
    strategy_note: str = ""
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_pct: Optional[float] = None

class CloseTradeRequest(BaseModel):
    trade_id: int
    close_price: float

class WatchlistRequest(BaseModel):
    stock_name: str
    stock_code: str
    sector: str = ""


class ConditionOrderRequest(BaseModel):
    symbol: str
    name: str = ""
    side: str = "buy"
    trigger_type: str  # price_above | price_below | time
    trigger_value: Optional[float] = None
    trigger_time: Optional[str] = None
    price: float = 0.0
    quantity: int = 100
    account_type: str = "sim"
    note: str = ""

@router.get("/accounts")
def api_get_accounts(current_user: User = Depends(get_current_user), conn: sqlite3.Connection = Depends(get_db)):
    return JSONResponse(get_accounts(conn, current_user.id))

def _check_risk_params(price: float, stop_loss: Optional[float], take_profit: Optional[float]) -> Optional[str]:
    """风控参数校验：止损/止盈与现价逻辑"""
    if stop_loss is not None and price > 0 and stop_loss >= price:
        return "止损价应低于当前买入价"
    if take_profit is not None and price > 0 and take_profit <= price:
        return "止盈价应高于当前买入价"
    return None

@router.post("/open")
def api_open_trade(req: TradeRequest, current_user: User = Depends(get_current_user), conn: sqlite3.Connection = Depends(get_db)):
    try:
        if req.account_type not in ['sim', 'real']:
            req.account_type = 'sim'
            
        # 模拟盘增强：尝试获取实时行情作为成交价（市价单逻辑）
        # 如果是模拟交易，且为了模拟真实感，我们强制拉取一次最新价
        if req.account_type == 'sim':
            real_price = _get_current_price(req.symbol)
            if real_price and real_price > 0:
                # 记录原始请求价格和真实成交价格的差异（可选），这里直接覆盖
                req.open_price = real_price
                
        err = _check_risk_params(req.open_price, getattr(req, 'stop_loss', None), getattr(req, 'take_profit', None))
        if err:
            return JSONResponse({"error": err}, status_code=400)
        
        # 验证交易参数
        if not req.symbol or not isinstance(req.symbol, str):
            return JSONResponse({"error": "无效的股票代码"}, status_code=400)
        if req.open_price <= 0:
            return JSONResponse({"error": "无效的价格"}, status_code=400)
        if req.open_quantity <= 0 or req.open_quantity % 100 != 0:
            return JSONResponse({"error": "无效的数量"}, status_code=400)
        
        trade_data = req.dict()
        trade_data['order_type'] = 'market'
        trade_data['stop_loss'] = getattr(req, 'stop_loss', None)
        trade_data['take_profit'] = getattr(req, 'take_profit', None)
        trade_data['order_status'] = 'pending'
        
        if req.account_type == 'real':
            try:
                from app.rox_quant.trade_executor import get_trader
                trader = get_trader('real')
                
                # 先检查连接状态
                if not trader._ensure_connected():
                    return JSONResponse({"error": "无法连接到交易账户"}, status_code=503)
                
                # 执行交易
                res = trader.buy(req.symbol, req.open_price, req.open_quantity)
                
                # 处理交易结果
                if res.get('status') == 'error':
                    error_code = res.get('code', 'UNKNOWN_ERROR')
                    error_msg = res.get('msg', '实盘下单失败')
                    logger.error(f"Real trade failed: {error_code} - {error_msg}")
                    
                    # 根据错误类型返回不同的状态码
                    if error_code in ('CONNECTION_ERROR', 'INSUFFICIENT_FUNDS'):
                        return JSONResponse({"error": error_msg, "code": error_code}, status_code=503)
                    elif error_code in ('INVALID_PARAMETER', 'INVALID_PRICE', 'INVALID_AMOUNT'):
                        return JSONResponse({"error": error_msg, "code": error_code}, status_code=400)
                    elif error_code == 'MARKET_CLOSED':
                        return JSONResponse({"error": error_msg, "code": error_code}, status_code=423)
                    else:
                        return JSONResponse({"error": error_msg, "code": error_code}, status_code=400)
                elif res.get('status') == 'blocked':
                    return JSONResponse({"error": res.get('msg', '交易被阻止'), "code": res.get('code')}, status_code=403)
                
                # 更新交易数据
                trade_data['order_status'] = 'sent'
                trade_data['order_id'] = res.get('order_id')
                trade_data['execution_info'] = res.get('raw')
                
            except Exception as e:
                logger.error(f"Real trader execution failed: {e}")
                return JSONResponse({"error": f"实盘执行失败: {str(e)}", "code": "EXECUTION_ERROR"}, status_code=502)
        
        # 创建交易记录
        trade_id = create_trade(conn, current_user.id, trade_data)
        
        # 返回详细的交易信息
        return JSONResponse({
            "status": "ok", 
            "msg": "下单成功",
            "trade_id": trade_id,
            "order_id": trade_data.get('order_id'),
            "account_type": req.account_type,
            "symbol": req.symbol,
            "price": req.open_price,
            "quantity": req.open_quantity,
            "stop_loss": trade_data.get('stop_loss'),
            "take_profit": trade_data.get('take_profit')
        })
    except Exception as e:
        logger.error(f"API open trade failed: {e}")
        return JSONResponse({"error": f"系统错误: {str(e)}"}, status_code=500)

@router.post("/close")
def api_close_trade(req: CloseTradeRequest, current_user: User = Depends(get_current_user), conn: sqlite3.Connection = Depends(get_db)):
    try:
        success = close_trade(conn, current_user.id, req.trade_id, req.close_price)
        if success:
            return JSONResponse({"status": "ok"})
        else:
            return JSONResponse({"error": "Trade not found"}, status_code=404)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@router.get("/trades")
def api_list_trades(account_type: str = "sim", status: str = None, current_user: User = Depends(get_current_user), conn: sqlite3.Connection = Depends(get_db)):
    status = status if status != "all" else None
    return JSONResponse({"trades": get_trades(conn, current_user.id, account_type, status)})

@router.get("/history")
def api_history_list(limit: int = 20, current_user: User = Depends(get_current_user), conn: sqlite3.Connection = Depends(get_db)):
    rows = get_history(conn, current_user.id, limit)
    return JSONResponse({"history": rows})

@router.delete("/history")
async def api_clear_history(current_user: User = Depends(get_current_user), conn: sqlite3.Connection = Depends(get_db)):
    clear_history(conn, current_user.id)
    return JSONResponse({"status": "ok"})

@router.get("/dashboard")
def api_trading_dashboard(current_user: User = Depends(get_current_user), conn: sqlite3.Connection = Depends(get_db)):
    accounts = get_accounts(conn, current_user.id)
    sim_trades = get_trades(conn, current_user.id, "sim", "closed")
    real_trades = get_trades(conn, current_user.id, "real", "closed")
    moods = get_psychology_stats(conn, current_user.id)
    return JSONResponse({
        "accounts": accounts,
        "sim_trades": sim_trades,
        "real_trades": real_trades,
        "moods": moods
    })

def _parse_ts(ts):
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None

async def _generate_review_summary(label: str, in_range: list) -> str:
    """调用 LLM 生成复盘摘要"""
    try:
        from app.rox_quant.llm import AIClient
        lines = [f"- {t.get('side','')} {t.get('symbol','')} {t.get('name','')} 价格{t.get('open_price')} 数量{t.get('open_quantity')} 时间{t.get('open_time')}" for t in in_range]
        text = "\n".join(lines[:30])
        prompt = f"以下为{label}交易记录（仅作参考）：\n{text}\n请用 2～4 句话总结得失与可改进点，语气专业简洁，不要列举单笔明细。"
        client = AIClient()
        if client.client:
            out = await client.chat_with_search(prompt, context="你是量化交易复盘助手，根据用户提供的交易记录给出简短复盘。")
            return (out or "").strip() or f"{label}共 {len(in_range)} 笔成交，建议结合仓位与止损执行情况复盘。"
    except Exception as e:
        logger.warning(f"LLM review failed: {e}")
    return f"{label}共 {len(in_range)} 笔成交，建议结合仓位与止损执行情况复盘。"

@router.get("/review")
async def api_trade_review(
    period: str = Query("week", description="week | month"),
    current_user: User = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_db),
):
    """按周/月返回交易复盘摘要；有成交时尝试调用 LLM 生成得失与策略建议。"""
    now = datetime.utcnow()
    if period == "month":
        since = now - timedelta(days=30)
        label = "本月"
    else:
        since = now - timedelta(days=7)
        label = "本周"
    all_trades = []
    for acc in ("sim", "real"):
        for status in ("open", "closed"):
            all_trades.extend(get_trades(conn, current_user.id, acc, status))
    in_range = []
    for t in all_trades:
        ot = _parse_ts(t.get("open_time"))
        if ot and ot.replace(tzinfo=None) >= since:
            in_range.append(t)
    buy_n = sum(1 for t in in_range if (t.get("side") or "").lower() == "buy")
    sell_n = len(in_range) - buy_n
    if not in_range:
        return JSONResponse({
            "summary": f"{label}暂无成交记录。记录交易后即可生成复盘。",
            "trades_count": 0,
        })
    summary = await _generate_review_summary(label, in_range)
    return JSONResponse({"summary": summary, "trades_count": len(in_range)})


@router.get("/review/advanced")
async def api_advanced_trade_review(
    period: str = Query("week", description="week | month | quarter | year"),
    current_user: User = Depends(get_current_user),
):
    """高级AI交易复盘，生成详细的交易分析和改进建议"""
    try:
        from app.services.ai_review_service import get_ai_review_service
        ai_review_service = get_ai_review_service()
        
        # 生成AI复盘报告
        review = await ai_review_service.generate_review(current_user.id, period)
        
        return JSONResponse(review)
    except Exception as e:
        logger.error(f"Error in advanced trade review: {e}")
        return JSONResponse({"error": "生成复盘报告失败"}, status_code=500)

@router.get("/watchlist")
async def api_get_watchlist(current_user: User = Depends(get_current_user), conn: sqlite3.Connection = Depends(get_db)):
    items = get_watchlist(conn, current_user.id)
    return JSONResponse({"items": items})

@router.post("/watchlist")
async def api_add_watchlist(req: WatchlistRequest, current_user: User = Depends(get_current_user), conn: sqlite3.Connection = Depends(get_db)):
    success = add_watchlist(conn, current_user.id, req.stock_name, req.stock_code, req.sector)
    if success:
        return JSONResponse({"status": "ok"})
    else:
        return JSONResponse({"error": "Already exists"}, status_code=400)

@router.delete("/watchlist/{stock_code}")
async def api_remove_watchlist(stock_code: str, current_user: User = Depends(get_current_user), conn: sqlite3.Connection = Depends(get_db)):
    remove_watchlist(conn, current_user.id, stock_code)
    return JSONResponse({"status": "ok"})

@router.get("/condition-orders")
async def api_list_condition_orders(current_user: User = Depends(get_current_user), conn: sqlite3.Connection = Depends(get_db)):
    try:
        cur = conn.execute(
            "SELECT id, symbol, name, side, trigger_type, trigger_value, trigger_time, price, quantity, status, created_at, filled_at FROM condition_orders WHERE user_id = ? ORDER BY created_at DESC",
            (current_user.id,)
        )
        items = [dict(row) for row in cur.fetchall()]
    except sqlite3.OperationalError:
        items = []
    return JSONResponse({"items": items})


@router.post("/condition-orders")
def api_create_condition_order(req: ConditionOrderRequest, current_user: User = Depends(get_current_user), conn: sqlite3.Connection = Depends(get_db)):
    if req.trigger_type not in ("price_above", "price_below", "time"):
        return JSONResponse({"error": "trigger_type 需为 price_above | price_below | time"}, status_code=400)
    if req.trigger_type.startswith("price") and req.trigger_value is None:
        return JSONResponse({"error": "价格条件需填写 trigger_value"}, status_code=400)
    data = req.dict()
    oid = create_condition_order(conn, current_user.id, data)
    if oid is None:
        return JSONResponse({"error": "创建失败"}, status_code=500)
    return JSONResponse({"status": "ok", "id": oid})


@router.delete("/condition-orders/{order_id}")
def api_cancel_condition_order(order_id: int, current_user: User = Depends(get_current_user), conn: sqlite3.Connection = Depends(get_db)):
    if cancel_condition_order(conn, current_user.id, order_id):
        return JSONResponse({"status": "ok"})
    return JSONResponse({"error": "未找到或已成交/已撤销"}, status_code=404)


@router.post("/check-risk")
def api_check_risk(current_user: User = Depends(get_current_user), conn: sqlite3.Connection = Depends(get_db)):
    """检查持仓止损/止盈并执行平仓（模拟盘写库，实盘需执行器）"""
    closed = []
    for t in get_open_trades_with_risk(conn, current_user.id):
        price = _get_current_price(t["symbol"])
        if price is None:
            continue
        op = float(t["open_price"])
        sl, tp = t.get("stop_loss"), t.get("take_profit")
        trigger = False
        if sl is not None and price <= float(sl):
            trigger = True
        if tp is not None and price >= float(tp):
            trigger = True
        if trigger and close_trade(conn, current_user.id, t["id"], price):
            closed.append({"trade_id": t["id"], "symbol": t["symbol"], "close_price": price})
    return JSONResponse({"status": "ok", "closed": closed})


@router.post("/check-conditions")
def api_check_conditions(current_user: User = Depends(get_current_user), conn: sqlite3.Connection = Depends(get_db)):
    """检查待触发的条件单（价格条件），触发则下单并标记已成交"""
    filled = []
    for co in get_pending_condition_orders(conn, None):
        if co["trigger_type"] not in ("price_above", "price_below"):
            continue
        price = _get_current_price(co["symbol"])
        if price is None:
            continue
        tv = float(co["trigger_value"] or 0)
        hit = (co["trigger_type"] == "price_above" and price >= tv) or (co["trigger_type"] == "price_below" and price <= tv)
        if not hit:
            continue
        trade_data = {
            "account_type": co["account_type"], "symbol": co["symbol"], "name": co.get("name") or co["symbol"],
            "side": co["side"], "open_price": float(co.get("price") or price), "open_quantity": int(co.get("quantity") or 100),
            "strategy_note": f"条件单#{co['id']}"
        }
        create_trade(conn, co["user_id"], trade_data)
        fill_condition_order(conn, co["id"])
        filled.append({"condition_order_id": co["id"], "symbol": co["symbol"], "price": price})
    return JSONResponse({"status": "ok", "filled": filled})


# --- Risk Management API ---
@router.get("/risk/summary")
def api_get_risk_summary(current_user: User = Depends(get_current_user)):
    """获取用户风险摘要"""
    try:
        from app.services.risk_monitor import get_risk_monitor
        risk_monitor = get_risk_monitor()
        summary = risk_monitor.get_risk_summary(current_user.id)
        return JSONResponse(summary)
    except Exception as e:
        logger.error(f"Error getting risk summary: {e}")
        return JSONResponse({"error": "获取风险摘要失败"}, status_code=500)

@router.get("/risk/alerts")
def api_get_risk_alerts(current_user: User = Depends(get_current_user), limit: int = 50):
    """获取用户风险警报"""
    try:
        from app.services.risk_monitor import get_risk_monitor
        risk_monitor = get_risk_monitor()
        alerts = risk_monitor.get_alerts(current_user.id, limit=limit)
        return JSONResponse({"alerts": alerts})
    except Exception as e:
        logger.error(f"Error getting risk alerts: {e}")
        return JSONResponse({"error": "获取风险警报失败"}, status_code=500)

@router.post("/risk/start-monitor")
def api_start_risk_monitor(current_user: User = Depends(get_current_user)):
    """启动风险监控服务"""
    try:
        from app.services.risk_monitor import start_risk_monitor
        start_risk_monitor()
        return JSONResponse({"status": "ok", "message": "风险监控服务已启动"})
    except Exception as e:
        logger.error(f"Error starting risk monitor: {e}")
        return JSONResponse({"error": "启动风险监控服务失败"}, status_code=500)

@router.post("/risk/stop-monitor")
def api_stop_risk_monitor(current_user: User = Depends(get_current_user)):
    """停止风险监控服务"""
    try:
        from app.services.risk_monitor import stop_risk_monitor
        stop_risk_monitor()
        return JSONResponse({"status": "ok", "message": "风险监控服务已停止"})
    except Exception as e:
        logger.error(f"Error stopping risk monitor: {e}")
        return JSONResponse({"error": "停止风险监控服务失败"}, status_code=500)

# --- Event Bus Integration ---
from app.core.event_bus import EventBus

async def handle_strategy_signal(data: dict):
    """
    Handle signal from strategy engine.
    """
    logger.info(f"Received strategy signal: {data}")
    signals = data.get("signals", {})
    if not signals:
        return

    try:
        # Create fresh connection
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        
        # Assume user_id=1 for automation
        user_id = 1 
        
        for symbol, weight in signals.items():
            if weight > 0:
                # Basic logic: weight > 0 -> Buy
                trade_data = {
                    "symbol": symbol,
                    "account_type": "sim",
                    "name": symbol, 
                    "side": "buy",
                    "open_price": 0.0, 
                    "open_quantity": 100, 
                    "strategy_note": f"Auto Signal: {data.get('node_id')}",
                    "order_type": "market"
                }
                create_trade(conn, user_id, trade_data)
                
        conn.close()
        logger.info(f"Auto-created {len(signals)} orders")
        
    except Exception as e:
        logger.error(f"Failed to process signals: {e}")

def setup_trade_listeners():
    EventBus().subscribe("strategy_signals", handle_strategy_signal)
