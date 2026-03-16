"""
卢式作战室 API 端点
路由前缀: /api/lu/
四个接口：
  GET /api/lu/three-flows   - 三流雷达（战略层）
  GET /api/lu/four-matrix   - 四矩阵切换面板（战略层）
  GET /api/lu/discipline    - 334纪律面板（仓位纪律层）
  GET /api/lu/candidates    - 候选池（执行层）

注意：接口返回辅助判断数据，不构成自动交易信号。
"""
from fastapi import APIRouter
from app.services.lu_service import LuService

router = APIRouter(prefix="/lu", tags=["卢式作战室"])


@router.get("/three-flows", summary="三流雷达（战略层）")
async def get_three_flows():
    """
    获取三流代理观察快照：流量 / 流速 / 流向。
    返回综合方向标签，仅作辅助判断，不构成自动结论。
    """
    return LuService.get_three_flows_snapshot()


@router.get("/four-matrix", summary="四矩阵切换面板（战略层）")
async def get_four_matrix():
    """
    获取四类资产相对强弱快照：黄金 / 能源 / 股票风险资产 / 现金防御。
    帮助用户判断资金搬家方向。
    """
    return LuService.get_four_matrix_snapshot()


@router.get("/discipline", summary="334纪律面板（仓位纪律层）")
async def get_discipline():
    """
    获取 334 仓位纪律面板：账户层三分法 + 单笔层三段法。
    仓位由用户手工决定，系统仅作阶段提示。
    """
    return LuService.get_334_discipline_snapshot()


@router.get("/candidates", summary="候选池（执行层）")
async def get_candidates():
    """
    获取候选标的方向池：包含方向、阶段、MACD状态。
    候选项仅作观察参考，不构成买卖推荐。
    """
    return LuService.get_candidate_pool()
