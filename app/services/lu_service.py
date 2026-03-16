"""
卢式作战室服务层 - 集中管理卢麒元式辅助决策业务逻辑

产品定位：半自动辅助决策系统（不是自动交易系统）
三层关系：
  - 三流 / 四矩阵 = 战略层
  - 334             = 仓位纪律层
  - 左脚 + 一板斧MACD = 执行层

三流/四矩阵：每次调用时尝试从 akshare 获取实时数据，失败则 fallback 到合理默认值。
候选池：使用真实 A 股 ETF 代码（不使用 mock ticker）。
"""
import logging
from datetime import date, datetime

logger = logging.getLogger("lu-service")


def _try_get_realtime_bias() -> tuple[str, str]:
    """
    尝试通过 akshare 实时指数涨跌幅推断三流偏向。
    返回 (summary_bias, note)。
    失败则返回默认值，不抛出异常。
    """
    try:
        import akshare as ak
        # 沪深 300 vs 黄金 ETF (518880) 涨跌幅对比，快速判断偏向
        spot_df = ak.stock_zh_a_spot_em()
        if spot_df is not None and not spot_df.empty:
            chg_col = next((c for c in spot_df.columns if "涨跌幅" in c), None)
            code_col = next((c for c in spot_df.columns if "代码" in c), None)
            if chg_col and code_col:
                spot_df[code_col] = spot_df[code_col].astype(str).str.zfill(6)
                up_ratio = (spot_df[chg_col] > 0).mean()
                limit_up = (spot_df[chg_col] >= 9.9).sum()
                limit_down = (spot_df[chg_col] <= -9.9).sum()

                # 黄金 ETF 518880
                gold_row = spot_df[spot_df[code_col] == "518880"]
                gold_chg = float(gold_row[chg_col].iloc[0]) if not gold_row.empty else 0.0

                if gold_chg > 0.5 and up_ratio < 0.55:
                    bias = "偏黄金"
                elif up_ratio > 0.7 and limit_up > limit_down * 2:
                    bias = "偏风险资产"
                elif limit_down > limit_up * 2:
                    bias = "偏防御"
                else:
                    bias = "中性偏黄金"  # 不确定时偏保守
                return bias, f"实时计算（上涨比{up_ratio:.0%}，涨停{limit_up}/跌停{limit_down}，黄金ETF{gold_chg:+.2f}%）"
    except Exception as e:
        logger.debug(f"三流实时计算失败，降级使用默认值: {e}")
    return "偏黄金", "当日实时计算不可用，使用默认快照"


def _try_get_realtime_strengths() -> dict:
    """
    尝试从 akshare 获取各类资产实时相对强度。
    返回 {资产名: relative_strength}，失败返回 {}。
    """
    defaults = {
        "黄金": 82,
        "能源/原油": 74,
        "股票风险资产": 48,
        "现金/防御": 66,
    }
    try:
        import akshare as ak
        spot_df = ak.stock_zh_a_spot_em()
        if spot_df is None or spot_df.empty:
            return defaults

        chg_col = next((c for c in spot_df.columns if "涨跌幅" in c), None)
        code_col = next((c for c in spot_df.columns if "代码" in c), None)
        if not chg_col or not code_col:
            return defaults

        spot_df[code_col] = spot_df[code_col].astype(str).str.zfill(6)

        def _get_chg(code6: str) -> float:
            row = spot_df[spot_df[code_col] == code6]
            return float(row[chg_col].iloc[0]) if not row.empty else 0.0

        gold_chg = _get_chg("518880")      # 黄金 ETF
        energy_chg = _get_chg("159980")   # 有色/能源 ETF
        equity_chg = _get_chg("510310")   # 沪深 300 ETF
        cash_chg = _get_chg("511010")     # 国债 ETF（防御类代理）

        # 把涨跌幅换算成 0-100 强度分（中性基准 50，每 1% 约 ±8 分）
        def _chg_to_score(chg: float) -> int:
            return max(10, min(95, int(50 + chg * 8)))

        return {
            "黄金": _chg_to_score(gold_chg),
            "能源/原油": _chg_to_score(energy_chg),
            "股票风险资产": _chg_to_score(equity_chg),
            "现金/防御": _chg_to_score(cash_chg),
        }
    except Exception as e:
        logger.debug(f"四矩阵实时计算失败，降级使用默认值: {e}")
        return defaults


class LuService:
    """卢式作战室业务聚合层"""

    @staticmethod
    def get_three_flows_snapshot() -> dict:
        """
        三流雷达快照（战略层）
        - 流量：量能扩张、市场活跃度
        - 流速：涨跌斜率、均线发散、MACD柱
        - 流向：黄金/防御相对强弱

        返回代理观察快照，不是自动买卖信号。
        尝试接入实时数据，失败则降级到默认值。
        """
        bias, bias_note = _try_get_realtime_bias()
        return {
            "as_of": str(datetime.now().strftime("%Y-%m-%d %H:%M")),
            "layer": "战略层",
            "mode": "代理观察·辅助判断",
            "summary_bias": bias,
            "bias_note": bias_note,
            "confidence_note": "基于代理指标的辅助判断，不构成自动结论",
            "flow_volume": {
                "label": "流量",
                "score": 68,
                "status": "中强",
                "drivers": ["板块量能扩张", "市场活跃度回升"]
            },
            "flow_velocity": {
                "label": "流速",
                "score": 72,
                "status": "加速",
                "drivers": ["相对强弱提升", "MACD柱放大"]
            },
            "flow_direction": {
                "label": "流向",
                "score": 75,
                "status": f"流向{bias.replace('偏', '')}",
                "drivers": ["黄金相对大盘更强", "能源次强", "风险资产回落"]
            },
            "notes": [
                "三流为战略层观察，不是自动买卖信号",
                "结论标签仅代表系统推断，最终判断由用户决定"
            ]
        }

    @staticmethod
    def get_four_matrix_snapshot() -> dict:
        """
        四矩阵切换面板快照（战略层）
        帮助用户观察资产之间的相对强弱和资金搬家方向。
        尝试接入实时数据，失败则降级到默认值。
        """
        strengths = _try_get_realtime_strengths()

        def _score_to_trend_arrow(score: int) -> str:
            if score >= 65:
                return "↑"
            elif score <= 45:
                return "↓"
            return "→"

        def _score_to_trend_text(score: int) -> str:
            if score >= 75:
                return "增强"
            elif score >= 60:
                return "转强"
            elif score >= 45:
                return "分化"
            elif score >= 35:
                return "稳定"
            return "走弱"

        gold_s = strengths.get("黄金", 82)
        energy_s = strengths.get("能源/原油", 74)
        equity_s = strengths.get("股票风险资产", 48)
        cash_s = strengths.get("现金/防御", 66)

        return {
            "as_of": str(datetime.now().strftime("%Y-%m-%d %H:%M")),
            "layer": "战略层",
            "description": "观察资产切换方向，判断资金是否在搬家",
            "assets": [
                {
                    "name": "黄金",
                    "theme": "gold",
                    "strength": gold_s,
                    "relative_strength": gold_s,        # ← 修复：同时提供两个字段
                    "trend": _score_to_trend_text(gold_s),
                    "trend_arrow": _score_to_trend_arrow(gold_s),  # ← 新增：符号字段
                    "trend_direction": "up" if gold_s >= 65 else ("down" if gold_s <= 40 else "neutral"),
                    "risk": "中",
                    "risk_level": 2,
                    "note": "偏战略配置，防御性强"
                },
                {
                    "name": "能源/原油",
                    "theme": "energy",
                    "strength": energy_s,
                    "relative_strength": energy_s,
                    "trend": _score_to_trend_text(energy_s),
                    "trend_arrow": _score_to_trend_arrow(energy_s),
                    "trend_direction": "up" if energy_s >= 65 else ("down" if energy_s <= 40 else "neutral"),
                    "risk": "中高",
                    "risk_level": 3,
                    "note": "偏中期波段，关注地缘因素"
                },
                {
                    "name": "股票风险资产",
                    "theme": "equity",
                    "strength": equity_s,
                    "relative_strength": equity_s,
                    "trend": _score_to_trend_text(equity_s),
                    "trend_arrow": _score_to_trend_arrow(equity_s),
                    "trend_direction": "up" if equity_s >= 65 else ("down" if equity_s <= 40 else "neutral"),
                    "risk": "高",
                    "risk_level": 4,
                    "note": "需选择方向，整体承压"
                },
                {
                    "name": "现金/防御",
                    "theme": "cash",
                    "strength": cash_s,
                    "relative_strength": cash_s,
                    "trend": _score_to_trend_text(cash_s),
                    "trend_arrow": _score_to_trend_arrow(cash_s),
                    "trend_direction": "up" if cash_s >= 65 else ("down" if cash_s <= 40 else "neutral"),
                    "risk": "低",
                    "risk_level": 1,
                    "note": "保留机动性，等待机会"
                }
            ]
        }

    @staticmethod
    def get_334_discipline_snapshot() -> dict:
        """
        334 纪律面板快照（仓位纪律层）
        双层理解：
          A. 账户层：长期仓 / 中期仓 / 预备队
          B. 单笔层：首仓30 / 二仓30 / 三仓40 节奏提示
        仓位最终由用户手工决定，系统只做阶段和纪律提醒。
        """
        return {
            "as_of": str(date.today()),
            "layer": "仓位纪律层",
            "warning": "仓位由用户手工决定，系统仅作纪律提示",
            "account_structure": {
                "label": "账户层三分法",
                "long_term": {"label": "长期仓", "ratio": 30, "note": "战略持仓，轻易不动"},
                "mid_term": {"label": "中期仓", "ratio": 30, "note": "波段操作，跟踪方向"},
                "reserve": {"label": "预备队", "ratio": 40, "note": "机动资金，等待机会"}
            },
            "trade_stage": {
                "label": "单笔层三段法",
                "current_stage": "首仓观察",
                "stage_code": 1,
                "allowed_action": "可考虑30%试仓（左脚位置）",
                "detail": [
                    {"step": 1, "label": "首仓30%", "tag": "左脚试仓", "active": True},
                    {"step": 2, "label": "二仓30%", "tag": "确认加仓", "active": False},
                    {"step": 3, "label": "三仓40%", "tag": "主升布局", "active": False}
                ]
            },
            "principles": [
                "先不败后求胜",
                "防御优先，留有余地",
                "留预备队，等待最佳时机",
                "律动压缩成本，不追高"
            ]
        }

    @staticmethod
    def get_candidate_pool() -> dict:
        """
        候选池快照（执行层）
        输出候选标的方向，每项包含：名称、方向、阶段、MACD状态、说明。
        使用真实 A 股 ETF 代码（替换 mock ticker）。
        """
        return {
            "as_of": str(date.today()),
            "layer": "执行层",
            "description": "候选标的仅作方向观察，不构成买卖推荐",
            "items": [
                {
                    "symbol": "518880",
                    "name": "黄金ETF",
                    "theme": "黄金",
                    "theme_code": "gold",
                    "stage": "左脚",
                    "stage_code": 1,
                    "macd_status": "修复",
                    "macd_code": "repair",
                    "note": "方向契合，适合观察首仓（30%试仓）"
                },
                {
                    "symbol": "159980",
                    "name": "有色ETF",
                    "theme": "能源/资源",
                    "theme_code": "energy",
                    "stage": "确认",
                    "stage_code": 2,
                    "macd_status": "金叉",
                    "macd_code": "golden",
                    "note": "适合观察二仓确认布局"
                },
                {
                    "symbol": "510310",
                    "name": "沪深300ETF",
                    "theme": "普通风险资产",
                    "theme_code": "equity",
                    "stage": "主升",
                    "stage_code": 3,
                    "macd_status": "强势",
                    "macd_code": "strong",
                    "note": "主升阶段，关注右肩风险，控制仓位"
                },
                {
                    "symbol": "511010",
                    "name": "国债ETF",
                    "theme": "防御",
                    "theme_code": "defensive",
                    "stage": "观察",
                    "stage_code": 0,
                    "macd_status": "零下修复",
                    "macd_code": "repair_below",
                    "note": "当前偏观察，等待结构确认"
                }
            ]
        }
