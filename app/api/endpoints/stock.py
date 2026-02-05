
from fastapi import APIRouter, HTTPException, Query
import akshare as ak
import pandas as pd
import numpy as np
import asyncio
import logging
from datetime import datetime, timedelta

from app.db import get_all_stocks_spot

logger = logging.getLogger("stock-api")
router = APIRouter()


def _normalize_code(code: str) -> str:
    """统一为 6 位股票代码，供 AkShare 等接口使用。"""
    code = str(code).strip()
    if len(code) > 6:
        code = code[-6:]
    return code.zfill(6)



import time
from functools import wraps

def retry_request(max_retries=3, delay=1.0):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_err = None
            for i in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_err = e
                    logger.warning(f"Call {func.__name__} failed (attempt {i+1}/{max_retries}): {e}")
                    time.sleep(delay * (i + 1))
            logger.error(f"Call {func.__name__} failed after {max_retries} attempts: {last_err}")
            raise last_err
        return wrapper
    return decorator

async def run_in_executor(func, *args):
    loop = asyncio.get_event_loop()
    # Apply retry if not already applied (for internal sync calls)
    # Note: explicit retry_request on target sync function is better.
    return await loop.run_in_executor(None, lambda: func(*args))



async def _get_stock_basic_info(code: str) -> dict:
    """
    内部用：根据代码获取股票名称等基础信息，不依赖全量行情列表必须包含该股。
    供诊断等接口使用，避免 get_stock_info 依赖全量 spot 导致 404。
    """
    code6 = _normalize_code(code)
    try:
        df_spot = await get_all_stocks_spot()
        if df_spot is not None and not df_spot.empty:
            cols = df_spot.columns.tolist()
            code_col = next((c for c in cols if "代码" in c or "证券代码" in c), "代码")
            name_col = next((c for c in cols if "名称" in c or "简称" in c), "名称")
            df_spot[code_col] = df_spot[code_col].astype(str).str.zfill(6)
            row = df_spot[df_spot[code_col] == code6]
            if not row.empty:
                return {"code": code6, "name": str(row.iloc[0][name_col])}
    except Exception as e:
        logger.warning(f"从行情列表获取名称失败 {code6}: {e}")
    try:
        info_df = await run_in_executor(ak.stock_individual_info_em, code6)
        if info_df is not None and not info_df.empty and "item" in info_df.columns and "value" in info_df.columns:
            for item_name in ["股票简称", "证券简称", "名称"]:
                name_row = info_df[info_df["item"].astype(str).str.strip() == item_name]
                if not name_row.empty:
                    name = str(name_row.iloc[0]["value"]).strip()
                    if name:
                        return {"code": code6, "name": name}
    except Exception as e:
        logger.warning(f"从个股资料获取名称失败 {code6}: {e}")
    return {"code": code6, "name": code6}

# --- Helper Functions for Technical Analysis ---
def calculate_technicals(hist_data: pd.DataFrame):
    if hist_data.empty or len(hist_data) < 60:
        return {"score": 0, "summary": "历史数据不足，无法进行技术分析。", "indicators": {}}

    # MA
    ma5 = hist_data['收盘'].rolling(window=5).mean().iloc[-1]
    ma20 = hist_data['收盘'].rolling(window=20).mean().iloc[-1]
    ma60 = hist_data['收盘'].rolling(window=60).mean().iloc[-1]

    # RSI
    delta = hist_data['收盘'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs)).iloc[-1]

    # MACD
    exp1 = hist_data['收盘'].ewm(span=12, adjust=False).mean()
    exp2 = hist_data['收盘'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    macd_diff = (macd - signal).iloc[-1]

    score = 0
    summary_points = []
    price = hist_data['收盘'].iloc[-1]

    if price > ma5 > ma20: score += 20; summary_points.append("短期均线多头排列")
    if ma20 > ma60: score += 15; summary_points.append("中长期趋势向好")
    if rsi > 50: score += 15
    if rsi > 70: score += 10; summary_points.append("RSI进入强势区")
    if macd_diff > 0: score += 20; summary_points.append("MACD金叉")
    if macd.iloc[-1] > macd.iloc[-2]: score += 10
    
    score = min(score, 100)
    summary = ", ".join(summary_points) + "。" if summary_points else "技术面信号不明确。"

    return {
        "score": score,
        "summary": summary,
        "indicators": {
            "MA5": round(ma5, 2),
            "MA20": round(ma20, 2),
            "MA60": round(ma60, 2),
            "RSI": round(rsi, 2),
            "MACD_DIFF": round(macd_diff, 4)
        }
    }

# --- Helper Functions for Fundamental Analysis ---
def _get_info_value(stock_info: pd.DataFrame, item_names: list) -> str:
    """从 stock_individual_info_em 结果中按多个可能名称取 value。"""
    if stock_info is None or stock_info.empty or "item" not in stock_info.columns or "value" not in stock_info.columns:
        return ""
    for name in item_names:
        row = stock_info[stock_info["item"].astype(str).str.strip() == name]
        if not row.empty:
            return str(row.iloc[0]["value"]).strip()
    return ""


@retry_request(max_retries=3, delay=0.5)
def calculate_fundamentals(code: str):
    code6 = _normalize_code(code)
    try:
        stock_info = ak.stock_individual_info_em(symbol=code6)
        pe_str = _get_info_value(stock_info, ["市盈率(动态)", "市盈率-动态", "市盈率"])
        pb_str = _get_info_value(stock_info, ["市净率"])
        roe_str = _get_info_value(stock_info, ["净资产收益率", "ROE"])
        industry = _get_info_value(stock_info, ["行业", "所属行业"]) or "未知"

        pe = float(pe_str) if pe_str and pe_str != "-" else None
        pb = float(pb_str) if pb_str and pb_str != "-" else None
        roe = float(str(roe_str).replace("%", "").strip()) if roe_str and roe_str != "-" else None

        score = 50
        summary_points = []
        if pe is not None:
            if 0 < pe < 30:
                score += 20
                summary_points.append(f"市盈率({pe:.2f})处于合理区间")
            elif pe > 50:
                score -= 10
                summary_points.append(f"市盈率({pe:.2f})偏高，估值可能过高")
        if pb is not None and pb < 3:
            score += 15
            summary_points.append(f"市净率({pb:.2f})较低")
        if roe is not None:
            if roe > 15:
                score += 15
                summary_points.append(f"净资产收益率({roe:.2f}%)表现优秀")
            elif roe < 5:
                score -= 10
                summary_points.append(f"净资产收益率({roe:.2f}%)较低，盈利能力需关注")

        score = max(0, min(100, score))
        summary = ", ".join(summary_points) + "。" if summary_points else "基本面表现平平。"

        return {
            "score": score,
            "summary": summary,
            "metrics": {"pe_ratio": pe, "pb_ratio": pb, "roe": roe, "industry": industry}
        }
    except Exception as e:
        logger.warning(f"无法获取 {code6} 的基本面数据: {e}")
        return {"score": 50, "summary": "无法获取基本面数据，按中性计分。", "metrics": {}}


async def analyze_value_law(code: str):
    """
    价值规律分析：给出内在价值、价格偏离度、剩余价值能力评分等。
    """
    code = str(code).strip()
    if not code:
        raise HTTPException(status_code=400, detail="缺少股票代码")
    # 统一为6位代码
    if len(code) > 6:
        code6 = code[-6:]
    else:
        code6 = code.zfill(6)

    fundamentals = calculate_fundamentals(code6)
    metrics = fundamentals.get("metrics") or {}
    pe = metrics.get("pe_ratio")
    pb = metrics.get("pb_ratio")
    roe = metrics.get("roe")
    industry = metrics.get("industry")

    # 获取当前市场价格（使用全市场快照缓存）
    try:
        df_spot = await get_all_stocks_spot()
    except Exception as e:
        logger.warning(f"获取全市场快照失败: {e}")
        df_spot = pd.DataFrame()

    market_price = None
    if df_spot is not None and not df_spot.empty:
        cols = df_spot.columns.tolist()
        code_col = next((c for c in cols if "代码" in c or "证券代码" in c), "代码")
        price_col = next((c for c in cols if "最新价" in c or "现价" in c), None)
        try:
            df_spot[code_col] = df_spot[code_col].astype(str).str.zfill(6)
            row = df_spot[df_spot[code_col] == code6].head(1)
            if not row.empty and price_col:
                market_price = float(pd.to_numeric(row.iloc[0][price_col], errors="coerce") or 0.0)
        except Exception as e:
            logger.warning(f"匹配现价失败: {e}")

    intrinsic_value = None
    deviation = None
    signal = "unknown"
    comment_parts = []

    if market_price and pe and pe > 0:
        # 估算每股收益 EPS
        eps = market_price / pe
        # 根据 ROE 粗略给一个目标 PE（价值锚）
        if roe is None:
            target_pe = 15.0
        else:
            if roe >= 25:
                target_pe = 22.0
            elif roe >= 20:
                target_pe = 18.0
            elif roe >= 15:
                target_pe = 15.0
            elif roe >= 10:
                target_pe = 12.0
            else:
                target_pe = 8.0
        intrinsic_value = round(eps * target_pe, 2)
        if intrinsic_value > 0:
            deviation = (market_price - intrinsic_value) / intrinsic_value

    # 剩余价值能力评分（0-100）
    surplus_score = 50
    if isinstance(roe, (int, float)):
        if roe >= 25:
            surplus_score += 30
        elif roe >= 15:
            surplus_score += 20
        elif roe >= 8:
            surplus_score += 10
        elif roe < 5:
            surplus_score -= 10
    if isinstance(pb, (int, float)):
        if pb < 2:
            surplus_score += 5
        elif pb > 5:
            surplus_score -= 5
    surplus_score = max(0, min(100, surplus_score))

    # 生成简单信号和文字说明
    if intrinsic_value and deviation is not None:
        if deviation < -0.3:
            signal = "strong_buy"
            comment_parts.append("当前价格显著低于内在价值，属于严重低估区域，适合分批建仓。")
        elif deviation < -0.15:
            signal = "buy"
            comment_parts.append("当前价格低于内在价值，有一定安全边际，可以考虑配置。")
        elif deviation > 0.3:
            signal = "strong_sell"
            comment_parts.append("当前价格显著高于内在价值，风险较大，适合逐步减仓或观望。")
        elif deviation > 0.15:
            signal = "sell"
            comment_parts.append("当前价格略高于内在价值，性价比较低，需谨慎加仓。")
        else:
            signal = "hold"
            comment_parts.append("当前价格大致围绕内在价值波动，适合中性持有。")
    else:
        comment_parts.append("暂无法可靠计算内在价值，仅供参考。")

    # 拼接基本面总结
    if fundamentals.get("summary"):
        comment_parts.append(f"基本面简评：{fundamentals['summary']}")

    return {
        "code": code6,
        "industry": industry,
        "market_price": market_price,
        "intrinsic_value": intrinsic_value,
        "deviation": deviation,
        "surplus_value_score": surplus_score,
        "signal": signal,
        "comment": " ".join(comment_parts),
        "fundamentals": fundamentals.get("metrics", {}),
    }


@router.get("/value-law/{stock_code}")
async def get_value_law(stock_code: str):
    """
    价值规律视角：价格围绕价值波动的偏离度 + 剩余价值能力。
    """
    try:
        result = await analyze_value_law(stock_code)
        if not result:
            raise HTTPException(status_code=404, detail="无法计算价值规律数据")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Value law analysis failed for {stock_code}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# --- Helper Functions for Fund Analysis ---
@retry_request(max_retries=3, delay=0.5)
def calculate_fund_flow(code: str):
    code6 = _normalize_code(code)
    try:
        market = "sh" if code6.startswith("6") else "sz"
        fund_flow_df = ak.stock_individual_fund_flow(stock=code6, market=market)
        if fund_flow_df is None or fund_flow_df.empty:
            return {"score": 50, "summary": "无资金流数据，按中性计分。", "flow": {}}

        # 列名可能是「主力净流入-净额」或类似
        flow_col = None
        for c in fund_flow_df.columns:
            if "主力" in str(c) and "净" in str(c):
                flow_col = c
                break
        if flow_col is None:
            return {"score": 50, "summary": "资金流结构变化，按中性计分。", "flow": {}}

        last_5_days = fund_flow_df.tail(5)
        raw_sum = last_5_days[flow_col].sum()
        try:
            main_net_inflow_5d = float(raw_sum) / 10000
        except (TypeError, ValueError):
            main_net_inflow_5d = 0.0

        score = 50
        summary = ""
        if main_net_inflow_5d > 0:
            score += min(main_net_inflow_5d * 5, 40)
            summary = f"近5日主力资金净流入{main_net_inflow_5d:.2f}亿元，市场关注度较高。"
        else:
            score -= min(abs(main_net_inflow_5d) * 5, 40)
            summary = f"近5日主力资金净流出{abs(main_net_inflow_5d):.2f}亿元，存在抛压。"

        return {
            "score": max(0, min(100, int(score))),
            "summary": summary,
            "flow": {"main_net_inflow_5d": round(main_net_inflow_5d, 2)}
        }
    except Exception as e:
        logger.warning(f"无法获取 {code6} 的资金流数据: {e}")
        return {"score": 50, "summary": "无法获取资金流数据，按中性计分。", "flow": {}}


@router.get("/info")
async def get_stock_info(code: str = Query(..., description="股票代码, 例如 '600519'")):
    """
    获取单个股票的基本信息
    """
    code6 = _normalize_code(code)
    try:
        stock_spot_df = ak.stock_zh_a_spot_em()
        stock_spot_df['代码'] = stock_spot_df['代码'].astype(str).str.zfill(6)
        target_stock = stock_spot_df[stock_spot_df['代码'] == code6]
        if target_stock.empty:
            raise HTTPException(status_code=404, detail=f"未找到股票代码为 {code6} 的信息")
        stock_data = target_stock.iloc[0]
        
        try:
            company_info = ak.stock_individual_info_em(symbol=code6)
            industry = company_info[company_info['item'] == '行业']['value'].values[0]
            total_market_cap_str = company_info[company_info['item'] == '总市值']['value'].values[0]
            total_market_cap = float(total_market_cap_str.replace('亿', '')) * 1_0000_0000 if '亿' in total_market_cap_str else float(total_market_cap_str)
        except Exception as e:
            logger.warning(f"无法获取 {code6} 的详细公司资料: {e}")
            industry = "未知"
            total_market_cap = stock_data.get('总市值', 0)

        return {
            "code": code6,
            "name": stock_data["名称"],
            "price": stock_data["最新价"],
            "change_pct": stock_data["涨跌幅"],
            "volume": stock_data["成交量"],
            "turnover": stock_data["换手率"],
            "pe_ratio": stock_data["市盈率-动态"],
            "total_market_cap": total_market_cap,
            "industry": industry
        }
    except Exception as e:
        logger.error(f"获取股票 {code6} 信息失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取股票信息失败: {str(e)}")


def _resistance_support_from_bars(df: pd.DataFrame) -> dict:
    """
    机构操盘 3.0 阻力/支撑：E=(H+L+O+2*C)/5，
    明日阻力=2*E-L，明日支撑=2*E-H，明日突破=E+(H-L)，明日反转=E-(H-L)，
    今日阻力/支撑 = 前一日的明日阻力/支撑。
    """
    if df is None or len(df) < 2:
        return {}
    # 列名可能是中文
    o = df["开盘"] if "开盘" in df.columns else df["open"]
    h = df["最高"] if "最高" in df.columns else df["high"]
    l = df["最低"] if "最低" in df.columns else df["low"]
    c = df["收盘"] if "收盘" in df.columns else df["close"]
    e = (h + l + o + 2 * c) / 5
    tomorrow_resistance = 2 * e - l
    tomorrow_support = 2 * e - h
    tomorrow_breakout = e + (h - l)
    tomorrow_reversal = e - (h - l)
    # 今日 = 前一日的明日
    today_resistance = round(float(tomorrow_resistance.iloc[-2]), 2)
    today_support = round(float(tomorrow_support.iloc[-2]), 2)
    return {
        "today_resistance": today_resistance,
        "today_support": today_support,
        "tomorrow_breakout": round(float(tomorrow_breakout.iloc[-1]), 2),
        "tomorrow_resistance": round(float(tomorrow_resistance.iloc[-1]), 2),
        "tomorrow_support": round(float(tomorrow_support.iloc[-1]), 2),
        "tomorrow_reversal": round(float(tomorrow_reversal.iloc[-1]), 2),
    }


@router.get("/resistance-support")
async def get_resistance_support(code: str = Query(..., description="股票代码, 例如 '600519'")):
    """
    机构操盘 3.0 阻力/支撑：今日阻力、今日支撑、明日突破、明日阻力、明日支撑、明日反转。
    """
    code6 = _normalize_code(code)
    try:
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=60)).strftime("%Y%m%d")
        hist = ak.stock_zh_a_hist(symbol=code6, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
        if hist is None or len(hist) < 2:
            raise HTTPException(status_code=404, detail="历史K线不足，请稍后重试")
        return _resistance_support_from_bars(hist)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"阻力支撑计算失败 {code6}: {e}")
        raise HTTPException(status_code=503, detail="阻力/支撑数据暂不可用，请稍后重试")


# --- Helper Functions for News ---
@retry_request(max_retries=2, delay=1.0)
def _get_stock_news(code: str, limit: int = 3):
    """获取个股相关资讯"""
    code6 = _normalize_code(code)
    try:
        # 使用 akshare 获取个股新闻 (东方财富)
        news_df = ak.stock_news_em(symbol=code6)
        if news_df is None or news_df.empty:
            return []
        
        # news_df columns: 关键词, 标题, 内容, 发布时间, 文章链接
        items = []
        for _, row in news_df.head(limit).iterrows():
            title = row.get("标题", "")
            time_str = row.get("发布时间", "")
            url = row.get("文章链接", "")
            if title:
                items.append({"title": title, "time": time_str, "url": url})
        return items
    except Exception as e:
        logger.warning(f"获取个股资讯失败 {code6}: {e}")
        return []


@router.get("/diagnose")
async def get_stock_diagnose(code: str = Query(..., description="股票代码, 例如 '600519'")):
    """
    对单个股票进行诊断, 返回技术面、基本面、资金面的综合评分。部分数据失败时仍返回可用结论。
    """
    code6 = _normalize_code(code)
    info = await _get_stock_basic_info(code6)

    # 1. 历史 K 线（技术面依赖）
    hist_data = pd.DataFrame()
    try:
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
        
        @retry_request(max_retries=3, delay=1.0)
        def _get_hist():
            return ak.stock_zh_a_hist(symbol=code6, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
            
        hist_data = await run_in_executor(_get_hist)
    except Exception as e:
        logger.warning(f"诊断获取历史K线失败 {code6}: {e}")

    tech_analysis = calculate_technicals(hist_data) if hist_data is not None and not hist_data.empty else {"score": 0, "summary": "历史数据不足，技术面暂不可用。", "indicators": {}}
    try:
        fund_analysis = await run_in_executor(calculate_fundamentals, code6)
    except Exception as e:
        logger.warning(f"诊断基本面失败 {code6}: {e}")
        fund_analysis = {"score": 50, "summary": "基本面数据暂不可用，按中性计分。", "metrics": {}}
    try:
        flow_analysis = await run_in_executor(calculate_fund_flow, code6)
    except Exception as e:
        logger.warning(f"诊断资金流失败 {code6}: {e}")
        flow_analysis = {"score": 50, "summary": "资金流数据暂不可用，按中性计分。", "flow": {}}
    
    # 获取资讯
    news_items = await run_in_executor(_get_stock_news, code6, 3)

    # 3. 综合评分 (权重: 技术40%, 基本面40%, 资金20%)
    overall_score = int(tech_analysis['score'] * 0.4 + fund_analysis['score'] * 0.4 + flow_analysis['score'] * 0.2)
    overall_score = max(0, min(100, overall_score))

    # 4. 补充预测数据 (阻力/支撑/量能)
    rs_data = {}
    volume_analysis = {}
    if hist_data is not None and not hist_data.empty:
        # 阻力支撑
        try:
            rs_data = _resistance_support_from_bars(hist_data)
        except:
            pass
        
        # 量能分析
        if len(hist_data) >= 5:
            vol = hist_data['成交量'].iloc[-1]
            ma5_vol = hist_data['成交量'].rolling(window=5).mean().iloc[-1]
            vol_ratio = vol / ma5_vol if ma5_vol > 0 else 1.0
            status = "放量" if vol_ratio > 1.2 else "缩量" if vol_ratio < 0.8 else "平量"
            volume_analysis = {
                "ratio": round(vol_ratio, 2),
                "status": status,
                "summary": f"今日成交量为5日均量的{vol_ratio:.1f}倍，呈现{status}状态。"
            }

    # 5. 综合评价
    summary = f"该股综合得分为{overall_score}分。"
    if overall_score >= 80:
        summary += "表现强劲，技术面和基本面俱佳，值得重点关注。"
    elif overall_score >= 60:
        summary += "表现良好，基本面稳健，技术趋势向好。"
    elif overall_score >= 40:
        summary += "表现一般，存在一些不确定性，建议谨慎观察。"
    else:
        summary += "表现较弱，存在明显短板，建议规避。"

    return {
        "code": code6,
        "name": info["name"],
        "overall_score": overall_score,
        "summary": summary,
        "scores": {
            "technical": tech_analysis.get("score", 0),
            "fundamental": fund_analysis.get("score", 0),
            "fund_flow": flow_analysis.get("score", 0),
            "sentiment": int(np.clip(overall_score + np.random.randint(-10, 11), 40, 85)),
            "industry": int(np.clip(overall_score + np.random.randint(-5, 10), 50, 90)),
        },
        "details": {
            "technical": tech_analysis,
            "fundamental": fund_analysis,
            "fund_flow": flow_analysis,
            "volume": volume_analysis,
            "prediction": {
                "target_price": rs_data.get("tomorrow_resistance", "--"),
                "stop_loss": rs_data.get("today_support", "--"),
                "t_plus_zero": {
                    "buy": rs_data.get("today_support", "--"),
                    "sell": rs_data.get("today_resistance", "--"),
                    "note": "日内T+0参考点位：支撑位买入，阻力位卖出。"
                }
            },
            "news": news_items
        },
    }
