from datetime import date
from typing import Any, Dict, List


class LuService:
    """卢式作战室服务层（MVP，使用代理指标与示例数据）。"""

    @staticmethod
    def _today() -> str:
        return str(date.today())

    @staticmethod
    def _base_meta() -> Dict[str, Any]:
        return {
            "as_of": LuService._today(),
            "is_mock": True,
            "mode": "辅助决策",
        }

    @staticmethod
    def get_three_flows_snapshot() -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "layer": "战略层",
            "summary_bias": "偏黄金",
            "confidence_note": "基于代理指标的辅助判断，不构成自动结论",
            "flow_volume": {
                "score": 68,
                "status": "中强",
                "drivers": ["板块量能扩张", "市场活跃度回升"],
                "proxy": "成交额变化、板块活跃度",
            },
            "flow_velocity": {
                "score": 72,
                "status": "加速",
                "drivers": ["相对强弱提升", "MACD柱放大"],
                "proxy": "涨跌斜率、MACD柱体扩张",
            },
            "flow_direction": {
                "score": 75,
                "status": "流向黄金/防御",
                "drivers": ["黄金相对大盘更强", "能源次强", "风险资产回落"],
                "proxy": "黄金vs大盘、能源vs黄金、防御vs风险",
            },
            "notes": [
                "三流为战略层观察，不是自动买卖信号",
                "结论用于方向过滤，不替代用户判断",
            ],
        }
        return {**LuService._base_meta(), **payload}

    @staticmethod
    def get_four_matrix_snapshot() -> Dict[str, Any]:
        assets: List[Dict[str, Any]] = [
            {
                "name": "黄金",
                "strength": 82,
                "trend": "增强",
                "risk": "中",
                "note": "偏战略配置",
            },
            {
                "name": "能源/原油",
                "strength": 74,
                "trend": "转强",
                "risk": "中高",
                "note": "偏中期波段",
            },
            {
                "name": "股票/风险资产",
                "strength": 48,
                "trend": "分化",
                "risk": "高",
                "note": "需精选方向",
            },
            {
                "name": "现金/防御",
                "strength": 66,
                "trend": "稳定",
                "risk": "低",
                "note": "保留机动性",
            },
        ]
        payload = {
            "layer": "战略层",
            "assets": assets,
            "notes": ["四矩阵用于观察资产切换，不用于单点买卖指令"],
        }
        return {**LuService._base_meta(), **payload}

    @staticmethod
    def get_334_discipline_snapshot() -> Dict[str, Any]:
        payload = {
            "layer": "仓位纪律层",
            "account_structure": {
                "long_term": 30,
                "mid_term": 30,
                "reserve": 40,
            },
            "trade_stage": {
                "current_stage": "首仓观察",
                "allowed_action": "可考虑30%试仓",
                "warning": "仅提示，不自动执行",
            },
            "single_trade_rhythm": [
                "首仓观察 / 左脚试仓（30）",
                "二仓确认（30）",
                "三仓主升布局（40）",
            ],
            "stage_tags": ["观察", "可首仓", "可二仓确认", "可主升布局", "风险控制"],
            "principles": [
                "先看方向，再看结构，再看节奏",
                "先不败后求胜",
                "防御优先",
                "留预备队",
                "仓位由用户手工决定",
            ],
        }
        return {**LuService._base_meta(), **payload}

    @staticmethod
    def get_candidate_pool() -> Dict[str, Any]:
        payload = {
            "layer": "执行层",
            "items": [
                {
                    "symbol": "GOLD_PROXY_1",
                    "name": "黄金示例标的",
                    "theme": "黄金",
                    "stage": "左脚",
                    "macd_status": "零下修复",
                    "note": "方向契合，适合观察首仓",
                },
                {
                    "symbol": "OIL_PROXY_1",
                    "name": "能源示例标的",
                    "theme": "能源",
                    "stage": "确认",
                    "macd_status": "金叉",
                    "note": "适合观察二仓确认",
                },
                {
                    "symbol": "DEF_PROXY_1",
                    "name": "防御示例标的",
                    "theme": "防御",
                    "stage": "主升",
                    "macd_status": "零上强势",
                    "note": "趋势明确，但仍需纪律跟踪",
                },
                {
                    "symbol": "RISK_PROXY_1",
                    "name": "风险资产示例",
                    "theme": "风险",
                    "stage": "右肩风险",
                    "macd_status": "死叉",
                    "note": "警惕节奏失真，重视风险控制",
                },
            ],
            "notes": ["候选池是方向与执行的衔接层，非自动下单器"],
        }
        return {**LuService._base_meta(), **payload}
