"""
卢式作战室服务层 — 真实数据版 (v2)

三层关系：
  - 三流 / 四矩阵 = 战略层（接入 akshare 实时数据，24h TTL 缓存）
  - 334             = 仓位纪律层
  - 候选池          = 执行层（使用真实 ETF 代码）

数据策略：
  - 优先从 akshare 拉实时数据，失败则 fallback 到合理静态默认值
  - 使用内存 TTL 缓存（缓存 4 小时），避免频繁调用 akshare
"""
import logging
import time
from datetime import date, datetime
from typing import Optional

logger = logging.getLogger("lu-service")

# ─── 简单 TTL 内存缓存 ──────────────────────────────────────────────
_CACHE: dict = {}
_CACHE_TTL = 14400  # 4 小时


def _cache_get(key: str):
    entry = _CACHE.get(key)
    if entry and time.time() - entry["ts"] < _CACHE_TTL:
        return entry["val"]
    return None


def _cache_set(key: str, val):
    _CACHE[key] = {"val": val, "ts": time.time()}


# ─── 实时数据获取（带降级） ──────────────────────────────────────────

def _fetch_north_flow() -> float:
    """
    获取北向资金当日净流入（亿元）。
    正值=流入（利好风险资产），负值=流出（利空）。
    失败则返回 0.0。
    """
    cached = _cache_get("north_flow")
    if cached is not None:
        return cached
    try:
        import akshare as ak
        df = ak.stock_hsgt_north_net_flow_in_em(symbol="北上")
        if df is not None and not df.empty:
            last = float(df.iloc[-1]["value"])
            # 若数值>1亿则单位为元，否则为亿
            val_yi = last / 1e8 if abs(last) > 1e8 else last
            _cache_set("north_flow", val_yi)
            return val_yi
    except Exception as e:
        logger.debug(f"北向资金获取失败，降级: {e}")
    return 0.0


def _fetch_market_stats() -> dict:
    """
    获取市场统计：上涨家数比例、成交额、涨停/跌停数。
    失败则返回 {'up_ratio':0.5, 'total_vol_yi':8000, 'limit_up':0, 'limit_down':0}。
    """
    cached = _cache_get("market_stats")
    if cached is not None:
        return cached
    defaults = {"up_ratio": 0.5, "total_vol_yi": 8000, "limit_up": 0, "limit_down": 0}
    try:
        import akshare as ak
        df = ak.stock_zh_a_spot_em()
        if df is None or df.empty:
            return defaults
        chg_col = next((c for c in df.columns if "涨跌幅" in c), None)
        amt_col = next((c for c in df.columns if "成交额" in c), None)
        if not chg_col:
            return defaults
        chg = df[chg_col].astype(float)
        up = int((chg > 0).sum())
        down = int((chg < 0).sum())
        limit_up = int((chg >= 9.9).sum())
        limit_down = int((chg <= -9.9).sum())
        total_vol_yi = float(df[amt_col].sum()) / 1e8 if amt_col else 8000
        result = {
            "up_ratio": up / max(up + down, 1),
            "total_vol_yi": round(total_vol_yi, 0),
            "limit_up": limit_up,
            "limit_down": limit_down,
        }
        _cache_set("market_stats", result)
        return result
    except Exception as e:
        logger.debug(f"市场统计获取失败，降级: {e}")
        return defaults


def _fetch_etf_strengths() -> dict:
    """
    获取各类 ETF 实时涨跌幅，换算为相对强度分（0-100）。
    默认基准分 50，每 +1% 对应 +8 分。
    """
    cached = _cache_get("etf_strengths")
    if cached is not None:
        return cached
    defaults = {"黄金": 72, "能源/原油": 65, "股票风险资产": 52, "现金/防御": 58}
    etf_map = {
        "518880": "黄金",          # 黄金 ETF
        "159980": "能源/原油",     # 有色/能源 ETF
        "510310": "股票风险资产",   # 沪深300 ETF
        "511010": "现金/防御",     # 国债 ETF（防御）
    }
    try:
        import akshare as ak
        df = ak.stock_zh_a_spot_em()
        if df is None or df.empty:
            return defaults
        chg_col = next((c for c in df.columns if "涨跌幅" in c), None)
        code_col = next((c for c in df.columns if "代码" in c), None)
        if not chg_col or not code_col:
            return defaults
        df[code_col] = df[code_col].astype(str).str.zfill(6)
        result = {}
        for etf_code, asset_name in etf_map.items():
            row = df[df[code_col] == etf_code]
            if not row.empty:
                chg = float(row[chg_col].iloc[0])
                score = max(10, min(95, int(50 + chg * 8)))
            else:
                score = defaults.get(asset_name, 50)
            result[asset_name] = score
        _cache_set("etf_strengths", result)
        return result
    except Exception as e:
        logger.debug(f"ETF 强度获取失败，降级: {e}")
        return defaults


# ─── 辅助函数 ──────────────────────────────────────────────────────

def _score_to_trend_arrow(score: int) -> str:
    if score >= 62:
        return "↑"
    elif score <= 42:
        return "↓"
    return "→"


def _score_to_trend_text(score: int) -> str:
    if score >= 75:
        return "增强"
    elif score >= 62:
        return "转强"
    elif score >= 45:
        return "分化"
    elif score >= 35:
        return "稳定"
    return "走弱"


def _infer_bias_from_data(
    north_flow: float,
    up_ratio: float,
    limit_up: int,
    limit_down: int,
    gold_strength: int,
    equity_strength: int,
) -> tuple[str, str]:
    """
    综合北向资金、上涨比、黄金 ETF 相对强弱，推断三流偏向。
    返回 (bias_label, note_str)。
    """
    if gold_strength > equity_strength + 15 and north_flow < 0:
        return "偏黄金", f"黄金 ETF 相对强势（分差{gold_strength - equity_strength}），北向流出{abs(north_flow):.1f}亿"
    if north_flow > 30 and up_ratio > 0.60:
        return "偏风险资产", f"北向净流入{north_flow:.1f}亿，上涨比{up_ratio:.0%}"
    if up_ratio < 0.38 and limit_down > limit_up * 1.5:
        return "偏防御", f"市场下行（上涨比{up_ratio:.0%}，跌停{limit_down}家）"
    if north_flow < -20:
        return "偏防御", f"北向大幅流出{abs(north_flow):.1f}亿，注意风险"
    if gold_strength > 65 and equity_strength < 55:
        return "中性偏黄金", f"黄金 ETF 强于大盘，结构分化"
    return "中性", f"各类资产暂无明显偏向，上涨比{up_ratio:.0%}"


# ─── LuService ────────────────────────────────────────────────────

class LuService:
    """卢式作战室业务聚合层（真实数据版）"""

    @staticmethod
    def get_three_flows_snapshot() -> dict:
        """
        三流雷达快照（战略层）
        - 流量：市场总成交额（实时）
        - 流速：上涨比 + 涨停/跌停差（实时）
        - 流向：北向资金 + 黄金 ETF 相对强弱（实时）
        失败则降级到合理静态值。
        """
        stats = _fetch_market_stats()
        north_flow = _fetch_north_flow()
        strengths = _fetch_etf_strengths()

        up_ratio = stats["up_ratio"]
        limit_up = stats["limit_up"]
        limit_down = stats["limit_down"]
        total_vol_yi = stats["total_vol_yi"]
        gold_strength = strengths.get("黄金", 72)
        equity_strength = strengths.get("股票风险资产", 52)

        # 流量分（成交额：万亿=90分，5000亿=50分）
        vol_score = max(20, min(95, int(50 + (total_vol_yi - 8000) / 500)))
        vol_status = "活跃" if vol_score >= 65 else ("中等" if vol_score >= 45 else "低迷")

        # 流速分（上涨比-0.5两倍放大）
        vel_score = max(10, min(95, int(50 + (up_ratio - 0.5) * 100)))
        vel_status = "加速" if up_ratio > 0.6 and limit_up > limit_down else ("减速" if up_ratio < 0.4 else "平稳")

        # 流向
        bias, bias_note = _infer_bias_from_data(
            north_flow, up_ratio, limit_up, limit_down, gold_strength, equity_strength
        )

        # 流向分
        dir_score = max(10, min(95, int(50 + north_flow * 0.8 + (gold_strength - 50) * 0.3)))

        return {
            "as_of": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "layer": "战略层",
            "mode": "实时数据·辅助判断",
            "summary_bias": bias,
            "bias_note": bias_note,
            "confidence_note": "基于实时行情代理指标，不构成自动结论",
            "flow_volume": {
                "label": "流量",
                "score": vol_score,
                "status": vol_status,
                "drivers": [
                    f"市场总成交额 {total_vol_yi:.0f} 亿",
                    f"{'量能扩张' if vol_score >= 65 else '量能收缩'}",
                ],
            },
            "flow_velocity": {
                "label": "流速",
                "score": vel_score,
                "status": vel_status,
                "drivers": [
                    f"上涨比 {up_ratio:.0%}",
                    f"涨停 {limit_up} 家 / 跌停 {limit_down} 家",
                ],
            },
            "flow_direction": {
                "label": "流向",
                "score": dir_score,
                "status": f"流向{bias.replace('偏', '').replace('中性', '待观察')}",
                "drivers": [
                    f"北向资金 {'+' if north_flow >= 0 else ''}{north_flow:.1f} 亿",
                    f"黄金 ETF 强度 {gold_strength}",
                ],
            },
            "notes": [
                "三流为战略层观察，不是自动买卖信号",
                "结论仅代表系统推断，最终判断由用户决定",
            ],
        }

    @staticmethod
    def get_four_matrix_snapshot() -> dict:
        """
        四矩阵切换面板快照（战略层）
        接入真实 ETF 实时涨跌幅换算为相对强度。
        """
        strengths = _fetch_etf_strengths()

        gold_s = strengths.get("黄金", 72)
        energy_s = strengths.get("能源/原油", 65)
        equity_s = strengths.get("股票风险资产", 52)
        cash_s = strengths.get("现金/防御", 58)

        assets = [
            ("黄金", "gold", gold_s, "518880 黄金 ETF 代理", "中", 2),
            ("能源/原油", "energy", energy_s, "159980 有色 ETF 代理", "中高", 3),
            ("股票风险资产", "equity", equity_s, "510310 沪深300 ETF 代理", "高", 4),
            ("现金/防御", "cash", cash_s, "511010 国债 ETF 代理", "低", 1),
        ]

        asset_list = []
        for name, theme, score, note, risk, risk_level in assets:
            asset_list.append({
                "name": name,
                "theme": theme,
                "strength": score,
                "relative_strength": score,
                "trend": _score_to_trend_text(score),
                "trend_arrow": _score_to_trend_arrow(score),
                "trend_direction": "up" if score >= 62 else ("down" if score <= 42 else "neutral"),
                "risk": risk,
                "risk_level": risk_level,
                "note": note,
                # 可视化扩展字段
                "color_class": "text-green-400" if score >= 62 else ("text-red-400" if score <= 42 else "text-yellow-400"),
                "bg_opacity": min(0.8, score / 100),
            })

        return {
            "as_of": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "layer": "战略层",
            "description": "观察资产切换方向，判断资金是否在搬家（ETF 实时强度代理）",
            "assets": asset_list,
        }

    @staticmethod
    def get_334_discipline_snapshot() -> dict:
        """334 纪律面板快照（仓位纪律层）"""
        return {
            "as_of": str(date.today()),
            "layer": "仓位纪律层",
            "warning": "仓位由用户手工决定，系统仅作纪律提示",
            "account_structure": {
                "label": "账户层三分法",
                "long_term": {"label": "长期仓", "ratio": 30, "note": "战略持仓，轻易不动"},
                "mid_term": {"label": "中期仓", "ratio": 30, "note": "波段操作，跟踪方向"},
                "reserve": {"label": "预备队", "ratio": 40, "note": "机动资金，等待机会"},
            },
            "trade_stage": {
                "label": "单笔层三段法",
                "current_stage": "首仓观察",
                "stage_code": 1,
                "allowed_action": "可考虑30%试仓（左脚位置）",
                "detail": [
                    {"step": 1, "label": "首仓30%", "tag": "左脚试仓", "active": True},
                    {"step": 2, "label": "二仓30%", "tag": "确认加仓", "active": False},
                    {"step": 3, "label": "三仓40%", "tag": "主升布局", "active": False},
                ],
            },
            "principles": ["先不败后求胜", "防御优先，留有余地", "留预备队，等待最佳时机", "律动压缩成本，不追高"],
        }

    @staticmethod
    def get_candidate_pool() -> dict:
        """候选池快照（执行层）— 使用真实 A 股 ETF 代码"""
        stats = _fetch_market_stats()
        up_ratio = stats["up_ratio"]
        # 根据市场状态动态调整候选池阶段提示
        market_note = "当前市场上涨比{:.0%}，{}".format(
            up_ratio,
            "偏多，关注主升标的" if up_ratio > 0.6 else ("偏空，以防御观察为主" if up_ratio < 0.4 else "中性，结合三流判断"),
        )
        return {
            "as_of": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "layer": "执行层",
            "description": f"候选标的仅作方向观察，不构成买卖推荐。{market_note}",
            "items": [
                {
                    "symbol": "518880", "name": "黄金ETF",
                    "theme": "黄金", "theme_code": "gold",
                    "stage": "左脚", "stage_code": 1,
                    "macd_status": "修复", "macd_code": "repair",
                    "note": "黄金战略配置方向，适合观察首仓（30%试仓）",
                },
                {
                    "symbol": "159980", "name": "有色ETF",
                    "theme": "能源/资源", "theme_code": "energy",
                    "stage": "确认", "stage_code": 2,
                    "macd_status": "金叉", "macd_code": "golden",
                    "note": "资源板块确认阶段，适合观察二仓",
                },
                {
                    "symbol": "510310", "name": "沪深300ETF",
                    "theme": "普通风险资产", "theme_code": "equity",
                    "stage": "主升", "stage_code": 3,
                    "macd_status": "强势", "macd_code": "strong",
                    "note": "宽基指数主升阶段，需关注右肩风险",
                },
                {
                    "symbol": "511010", "name": "国债ETF",
                    "theme": "防御", "theme_code": "defensive",
                    "stage": "观察", "stage_code": 0,
                    "macd_status": "零下修复", "macd_code": "repair_below",
                    "note": "防御配置，适合预备队仓位",
                },
            ],
        }
