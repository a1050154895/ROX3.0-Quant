from fastapi import APIRouter

from app.services.lu_service import LuService

router = APIRouter(prefix="/lu", tags=["Lu War Room"])


@router.get("/three-flows")
async def get_three_flows_snapshot():
    return LuService.get_three_flows_snapshot()


@router.get("/four-matrix")
async def get_four_matrix_snapshot():
    return LuService.get_four_matrix_snapshot()


@router.get("/discipline")
async def get_334_discipline_snapshot():
    return LuService.get_334_discipline_snapshot()


@router.get("/candidates")
async def get_candidate_pool():
    return LuService.get_candidate_pool()
