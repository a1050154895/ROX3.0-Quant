"""
卢式作战室 API 端点
路由前缀: /api/lu/
接口列表：
  GET /api/lu/three-flows          - 三流雷达（战略层）
  GET /api/lu/four-matrix          - 四矩阵切换面板（战略层）
  GET /api/lu/discipline           - 334纪律面板（仓位纪律层）
  GET /api/lu/candidates           - 候选池（执行层）
  GET /api/lu/analyze-symbol       - 卢式个股分析（六层结构化输出）

注意：所有接口返回辅助判断数据，不构成自动交易信号，仓位由用户手工决定。
"""
import asyncio
import logging
from datetime import datetime, timedelta

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from app.services.lu_service import LuService

logger = logging.getLogger("lu-api")
router = APIRouter(prefix="/lu", tags=["卢式作战室"])


# ─────────────────────── 作战室快照接口 ───────────────────────

@router.get("/three-flows", summary="三流雷达（战略层）")
async def get_three_flows():
    """获取三流代理观察快照：流量 / 流速 / 流向，仅作辅助判断，不构成自动结论。"""
    return LuService.get_three_flows_snapshot()


@router.get("/four-matrix", summary="四矩阵切换面板（战略层）")
async def get_four_matrix():
    """获取四类资产相对强弱快照：黄金 / 能源 / 股票风险资产 / 现金防御。"""
    return LuService.get_four_matrix_snapshot()


@router.get("/discipline", summary="334纪律面板（仓位纪律层）")
async def get_discipline():
    """获取 334 仓位纪律面板：账户层三分法 + 单笔层三段法。仓位由用户手工决定。"""
    return LuService.get_334_discipline_snapshot()


@router.get("/candidates", summary="候选池（执行层）")
async def get_candidates():
    """获取候选标的方向池：包含方向、阶段、MACD状态。仅作观察参考，不构成买卖推荐。"""
    return LuService.get_candidate_pool()


# ─────────────────────── 卢式个股分析（六层结构化） ───────────────────────

# 行业 → 主题/方向 映射表（基础规则版，可持续扩展）
_INDUSTRY_THEME_MAP = {
    "黄金": "黄金",
    "贵金属": "黄金",
    "银行": "防御",
    "保险": "防御",
    "公用事业": "防御",
    "电力": "防御",
    "石油": "能源/原油",
    "石化": "能源/原油",
    "煤炭": "能源/原油",
    "有色金属": "资源",
    "钢铁": "资源",
    "化工": "资源",
    "农业": "资源",
    "半导体": "科技",
    "电子": "科技",
    "通信": "科技",
    "软件": "科技",
    "医药": "防御",
    "消费": "普通风险资产",
    "零售": "普通风险资产",
    "房地产": "普通风险资产",
    "建筑": "普通风险资产",
}

def _infer_theme(industry: str) -> str:
    """按行业名称关键词推断卢式主题/方向"""
    if not industry or industry in ("未知", ""):
        return "普通风险资产"
    for kw, theme in _INDUSTRY_THEME_MAP.items():
        if kw in industry:
            return theme
    return "普通风险资产"


def _calc_lu_technicals(hist: pd.DataFrame) -> dict:
    """
    复用 stock.py 的K线数据，计算卢式所需的技术状态。
    返回：structure_stage（主图阶段）、macd_status（MACD节奏，列表）、
          macd_detail（原始 MACD 值，供调试）。
    """
    result = {
        "structure_stage": "结构未明",
        "macd_status": ["数据不足"],
        "macd_detail": {},
        "ma_detail": {},
    }
    if hist is None or hist.empty or len(hist) < 30:
        return result

    # 统一列名（ashare_lite 可能返回英文列名，akshare 返回中文）
    col_map = {"open": "开盘", "high": "最高", "low": "最低", "close": "收盘", "volume": "成交量"}
    hist = hist.rename(columns=col_map)

    if "收盘" not in hist.columns:
        return result

    close = hist["收盘"].astype(float)

    # ─── MACD (12, 26, 9) ───
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    hist_bar = (dif - dea) * 2  # MACD 柱体

    dif_now = float(dif.iloc[-1])
    dea_now = float(dea.iloc[-1])
    bar_now = float(hist_bar.iloc[-1])
    bar_prev = float(hist_bar.iloc[-2]) if len(hist_bar) >= 2 else bar_now

    macd_signals = []
    if dif_now > dea_now:
        macd_signals.append("金叉")
    else:
        macd_signals.append("死叉")

    if dif_now > 0 and dea_now > 0:
        macd_signals.append("零上强势")
    elif dif_now < 0 and dea_now < 0:
        macd_signals.append("零下修复")

    if abs(bar_now) > abs(bar_prev):
        macd_signals.append("柱体放大")
    else:
        macd_signals.append("柱体缩短")

    result["macd_status"] = macd_signals
    result["macd_detail"] = {
        "dif": round(dif_now, 4),
        "dea": round(dea_now, 4),
        "bar": round(bar_now, 4),
    }

    # ─── 均线（5/20/60）用于识别主图阶段 ───
    n = len(close)
    ma5 = float(close.rolling(5).mean().iloc[-1]) if n >= 5 else None
    ma20 = float(close.rolling(20).mean().iloc[-1]) if n >= 20 else None
    ma60 = float(close.rolling(60).mean().iloc[-1]) if n >= 60 else None
    price_now = float(close.iloc[-1])
    price_prev_peak = float(close.rolling(20).max().iloc[-2]) if n >= 21 else price_now

    result["ma_detail"] = {
        "ma5": round(ma5, 2) if ma5 else None,
        "ma20": round(ma20, 2) if ma20 else None,
        "ma60": round(ma60, 2) if ma60 else None,
        "price": round(price_now, 2),
    }

    # ─── 主图阶段识别（基础规则版） ───
    # 卢式阶段：左脚 / 确认 / 主升 / 右肩风险 / 破位 / 修复中 / 结构未明
    try:
        if ma5 and ma20 and ma60:
            if price_now < ma20 and price_now < ma60:
                # 价格远在均线下方
                if "零下修复" in macd_signals and "金叉" in macd_signals:
                    stage = "左脚"
                elif "零下修复" in macd_signals:
                    stage = "修复中"
                elif "死叉" in macd_signals:
                    stage = "破位"
                else:
                    stage = "结构未明"
            elif price_now > ma5 > ma20 and ma20 > ma60:
                # 短中长期均线多头
                if "柱体放大" in macd_signals and "金叉" in macd_signals and "零上强势" in macd_signals:
                    stage = "主升"
                elif "金叉" in macd_signals:
                    stage = "确认"
                else:
                    stage = "修复中"
            elif price_now > ma20 and price_now > price_prev_peak * 0.95:
                # 高位，接近前高，注意右肩风险
                if "柱体缩短" in macd_signals or "死叉" in macd_signals:
                    stage = "右肩风险"
                else:
                    stage = "主升"
            elif price_now < ma20 and price_now > ma60:
                stage = "修复中"
            else:
                stage = "结构未明"
        else:
            stage = "修复中"
    except Exception:
        stage = "结构未明"

    result["structure_stage"] = stage
    return result


def _infer_direction_and_matrix(theme: str) -> tuple[str, str]:
    """
    根据主题与当前作战室三流/四矩阵快照，推断方向支持度与矩阵位置。
    兼容新版 (relative_strength + trend_arrow) 与旧版 (strength + trend 中文值) 字段格式。
    """
    try:
        tf = LuService.get_three_flows_snapshot()
        fm = LuService.get_four_matrix_snapshot()
        bias = tf.get("summary_bias", "偏黄金")   # 当前三流偏向
        assets = {a["name"]: a for a in fm.get("assets", [])}
    except Exception:
        bias = "未知"
        assets = {}

    # 主题 → 三流偏向匹配
    theme_flow_map = {
        "黄金": "偏黄金",
        "能源/原油": "偏能源",
        "资源": "偏能源",
        "科技": "偏风险资产",
        "普通风险资产": "偏风险资产",
        "防御": "偏防御",
    }
    expected_bias = theme_flow_map.get(theme, "偏风险资产")

    if bias == expected_bias:
        direction_bias = "较强支持"
    elif "偏防御" in bias and theme in ("防御",):
        direction_bias = "较强支持"
    elif "偏黄金" in bias and theme in ("普通风险资产", "科技"):
        direction_bias = "方向不占优"
    else:
        direction_bias = "中度支持"

    # 四矩阵位置推断
    asset_score_map = {
        "黄金": "黄金",
        "能源/原油": "能源/原油",
        "科技": "股票风险资产",
        "普通风险资产": "股票风险资产",
        "资源": "能源/原油",
        "防御": "现金/防御",
    }
    asset_key = asset_score_map.get(theme, "股票风险资产")
    asset_info = assets.get(asset_key, {})
    # 优先读 relative_strength，fallback 到 strength（兼容旧格式）
    score = asset_info.get("relative_strength") or asset_info.get("strength", 50)
    # 优先读 trend_arrow 符号字段，兼容中文 trend 值("增强"/"转强"/"分化"/"稳定"/"走弱")
    trend = asset_info.get("trend_arrow", "")
    if not trend:
        trend_text = asset_info.get("trend", "→")
        _trend_map = {
            "增强": "↑", "转强": "↑", "分化": "→",
            "稳定": "→", "走弱": "↓", "下行": "↓",
            "up": "↑", "down": "↓", "neutral": "→",
        }
        trend = _trend_map.get(trend_text, "→")

    if score >= 75 and trend == "↑":
        matrix_position = "增强区"
    elif score >= 65 or (score >= 55 and trend == "↑"):
        matrix_position = "转强区"
    elif score >= 45:
        matrix_position = "分化区"
    elif trend == "↓" and score < 55:
        matrix_position = "防御区"
    else:
        matrix_position = "观望区"

    return direction_bias, matrix_position


def _infer_discipline_advice(structure_stage: str, macd_status: list, direction_bias: str) -> str:
    """根据结构、MACD、方向三要素，推断334纪律建议标签。"""
    has_golden = "金叉" in macd_status
    has_dead = "死叉" in macd_status
    has_up = "零上强势" in macd_status
    has_expand = "柱体放大" in macd_status
    not_support = direction_bias in ("方向不占优",)

    if structure_stage == "破位" or (has_dead and not has_up):
        return "风险控制"
    if structure_stage == "右肩风险":
        return "风险控制"
    if not_support:
        return "观察为主"
    if structure_stage == "主升" and has_golden and has_up and has_expand:
        return "可考虑主升布局"
    if structure_stage == "确认" and has_golden:
        return "可考虑二仓确认"
    if structure_stage == "左脚" and has_golden:
        return "可考虑首仓"
    if structure_stage in ("修复中", "结构未明"):
        return "观察为主"
    return "观察为主"


def _build_lu_summary(
    name: str, theme: str, direction_bias: str, matrix_position: str,
    structure_stage: str, macd_status: list, discipline_advice: str
) -> str:
    """生成克制风格的卢式总结文字。"""
    direction_ok = direction_bias in ("较强支持", "中度支持")
    structure_ok = structure_stage in ("确认", "主升", "左脚")
    macd_ok = "金叉" in macd_status and "零上强势" in macd_status

    # 总结按三要素组合生成
    if direction_ok and structure_ok and macd_ok:
        return (
            f"所属方向（{theme}）当前具备一定支持，结构进入{structure_stage}阶段，"
            f"MACD节奏较好，可作为观察候选，注意结合334纪律分批介入。"
        )
    elif not direction_ok:
        return (
            f"所属方向（{theme}）当前{direction_bias}，"
            f"不建议轻易动用334进行布局，以观察为主。"
        )
    elif not structure_ok:
        return (
            f"方向（{theme}）尚可，但结构处于{structure_stage}，"
            f"节奏尚未明确，适合继续观察，待结构清晰后再行评估。"
        )
    else:
        return (
            f"方向（{theme}）{direction_bias}，结构{structure_stage}，"
            f"MACD存在{'金叉' if '金叉' in macd_status else '死叉'}信号，综合需谨慎对待。"
        )


@router.get("/analyze-symbol", summary="卢式个股分析（六层结构化输出）")
async def analyze_symbol(
    symbol: str = Query(..., description="股票代码，如 600519"),
):
    """
    按卢式框架对单个股票输出结构化分析结果（六层）：
    1. 基本信息层 2. 方向判断层（战略层）3. 结构判断层（主图）
    4. 节奏判断层（MACD） 5. 334纪律建议层 6. 总结层

    本接口为辅助判断工具，所有结论均为参考，不构成自动交易信号。
    仓位由用户手工决定，系统仅作纪律提示。
    """
    from app.api.endpoints.stock import _normalize_code, _get_stock_basic_info
    from app.api.endpoints.stock import calculate_fundamentals
    from app.rox_quant.datasources import ashare_lite
    import akshare as ak

    code = _normalize_code(symbol)

    # ─── 1. 并行拉取基本信息 + K 线历史 ───
    async def _fetch_basic():
        try:
            return await asyncio.wait_for(_get_stock_basic_info(code), timeout=5.0)
        except Exception as e:
            logger.warning(f"Lu analyze basic info failed {code}: {e}")
            return {"code": code, "name": code}

    async def _fetch_history():
        loop = asyncio.get_running_loop()
        try:
            df = await loop.run_in_executor(
                None, lambda: ashare_lite.get_price(code, count=120, frequency="1d")
            )
            if df is not None and not df.empty:
                return df
        except Exception as e:
            logger.warning(f"Lu ashare_lite failed {code}: {e}")
        try:
            end = datetime.now().strftime("%Y%m%d")
            start = (datetime.now() - timedelta(days=200)).strftime("%Y%m%d")
            return await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start, end_date=end, adjust="qfq"),
                ),
                timeout=8.0,
            )
        except Exception as e:
            logger.warning(f"Lu akshare hist failed {code}: {e}")
            return None

    async def _fetch_fund():
        loop = asyncio.get_running_loop()
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(None, calculate_fundamentals, code), timeout=5.0
            )
        except Exception as e:
            logger.warning(f"Lu fundamentals failed {code}: {e}")
            return {"score": 50, "summary": "", "metrics": {}}

    basic, hist, fund = await asyncio.gather(
        _fetch_basic(), _fetch_history(), _fetch_fund(), return_exceptions=True
    )

    if isinstance(basic, Exception):
        basic = {"code": code, "name": code}
    if isinstance(fund, Exception):
        fund = {"score": 50, "summary": "", "metrics": {}}

    # ─── 2. 行业 & 主题 ───
    industry = "未知"
    try:
        metrics = fund.get("metrics", {}) if isinstance(fund, dict) else {}
        industry = metrics.get("industry") or "未知"
    except Exception:
        pass
    theme = _infer_theme(industry)

    # ─── 3. 技术分析（结构 + MACD） ───
    hist_df = hist if isinstance(hist, pd.DataFrame) else None
    tech = _calc_lu_technicals(hist_df)
    structure_stage = tech["structure_stage"]
    macd_status = tech["macd_status"]

    # ─── 4. 方向 & 矩阵位置 ───
    direction_bias, matrix_position = _infer_direction_and_matrix(theme)

    # ─── 5. 334 纪律建议 ───
    discipline_advice = _infer_discipline_advice(structure_stage, macd_status, direction_bias)

    # ─── 6. 总结 ───
    name = basic.get("name", code) if isinstance(basic, dict) else code
    summary = _build_lu_summary(
        name, theme, direction_bias, matrix_position,
        structure_stage, macd_status, discipline_advice
    )

    return {
        # 基本信息层
        "symbol": code,
        "name": name,
        "industry": industry,
        "theme": theme,
        # 方向判断层（战略层）
        "direction_bias": direction_bias,
        "matrix_position": matrix_position,
        "direction_note": "本判断基于当前作战室三流/四矩阵辅助观察，仅供参考，不构成自动结论。",
        # 结构判断层
        "structure_stage": structure_stage,
        "ma_detail": tech.get("ma_detail", {}),
        # 节奏判断层
        "macd_status": macd_status,
        "macd_detail": tech.get("macd_detail", {}),
        # 334纪律建议层
        "discipline_advice": discipline_advice,
        "risk_note": "仓位由用户手工决定，系统仅做纪律提示。",
        # 总结层
        "summary": summary,
        # 元数据
        "as_of": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "data_note": "当前结构/MACD判断基于日线K线规则映射，三流/四矩阵目前为作战室快照参考值，后续可接入实时指标。",
    }

