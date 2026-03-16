"""
卢式作战室服务层 - 集中管理卢麒元式辅助决策业务逻辑

产品定位：半自动辅助决策系统（不是自动交易系统）
三层关系：
  - 三流 / 四矩阵 = 战略层
  - 334             = 仓位纪律层
  - 左脚 + 一板斧MACD = 执行层

所有 mock 数据集中在此文件，便于后续替换为真实指标。
"""
from datetime import date


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
        """
        return {
            "as_of": str(date.today()),
            "layer": "战略层",
            "mode": "代理观察·辅助判断",
            "summary_bias": "偏黄金",
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
                "status": "流向黄金/防御",
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
        """
        return {
            "as_of": str(date.today()),
            "layer": "战略层",
            "description": "观察资产切换方向，判断资金是否在搬家",
            "assets": [
                {
                    "name": "黄金",
                    "theme": "gold",
                    "strength": 82,
                    "trend": "增强",
                    "trend_direction": "up",
                    "risk": "中",
                    "risk_level": 2,
                    "note": "偏战略配置，防御性强"
                },
                {
                    "name": "能源/原油",
                    "theme": "energy",
                    "strength": 74,
                    "trend": "转强",
                    "trend_direction": "up",
                    "risk": "中高",
                    "risk_level": 3,
                    "note": "偏中期波段，关注地缘因素"
                },
                {
                    "name": "股票风险资产",
                    "theme": "equity",
                    "strength": 48,
                    "trend": "分化",
                    "trend_direction": "neutral",
                    "risk": "高",
                    "risk_level": 4,
                    "note": "需选择方向，整体承压"
                },
                {
                    "name": "现金/防御",
                    "theme": "cash",
                    "strength": 66,
                    "trend": "稳定",
                    "trend_direction": "neutral",
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
        初期为可扩展 mock 结构，便于后续接入真实算法。
        """
        return {
            "as_of": str(date.today()),
            "layer": "执行层",
            "description": "候选标的仅作方向观察，不构成买卖推荐",
            "items": [
                {
                    "symbol": "GOLD_PROXY_1",
                    "name": "黄金方向示例",
                    "theme": "黄金",
                    "theme_code": "gold",
                    "stage": "左脚",
                    "stage_code": 1,
                    "macd_status": "修复",
                    "macd_code": "repair",
                    "note": "方向契合，适合观察首仓（30%试仓）"
                },
                {
                    "symbol": "OIL_PROXY_1",
                    "name": "能源方向示例",
                    "theme": "能源",
                    "theme_code": "energy",
                    "stage": "确认",
                    "stage_code": 2,
                    "macd_status": "金叉",
                    "macd_code": "golden",
                    "note": "适合观察二仓确认布局"
                },
                {
                    "symbol": "RESOURCE_PROXY_1",
                    "name": "资源方向示例",
                    "theme": "资源",
                    "theme_code": "resource",
                    "stage": "主升",
                    "stage_code": 3,
                    "macd_status": "强势",
                    "macd_code": "strong",
                    "note": "主升阶段，关注右肩风险，控制仓位"
                },
                {
                    "symbol": "DEFENSIVE_PROXY_1",
                    "name": "防御资产示例",
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
