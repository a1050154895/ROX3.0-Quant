from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any, List

from app.services.trading_simulation import trading_simulation

router = APIRouter(prefix="/trading-simulation", tags=["trading-simulation"])

@router.post("/start")
async def start_simulation(duration_seconds: int = 3600) -> Dict[str, Any]:
    """
    开始交易模拟
    
    Args:
        duration_seconds: 模拟持续时间（秒）
    
    Returns:
        模拟状态
    """
    try:
        # 在后台运行模拟
        import asyncio
        asyncio.create_task(trading_simulation.run_simulation(duration_seconds))
        
        return {
            "status": "success",
            "message": f"交易模拟已开始，持续时间: {duration_seconds}秒",
            "simulation_id": trading_simulation.simulation_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"启动模拟失败: {str(e)}")

@router.post("/stop")
async def stop_simulation() -> Dict[str, Any]:
    """
    停止交易模拟
    
    Returns:
        模拟状态
    """
    try:
        trading_simulation.stop_simulation()
        return {
            "status": "success",
            "message": "交易模拟已停止"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"停止模拟失败: {str(e)}")

@router.post("/reset")
async def reset_simulation() -> Dict[str, Any]:
    """
    重置交易模拟
    
    Returns:
        模拟状态
    """
    try:
        trading_simulation.reset_simulation()
        return {
            "status": "success",
            "message": "交易模拟已重置"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"重置模拟失败: {str(e)}")

@router.get("/status")
async def get_simulation_status() -> Dict[str, Any]:
    """
    获取模拟状态
    
    Returns:
        模拟状态
    """
    try:
        return trading_simulation.get_simulation_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取状态失败: {str(e)}")

@router.get("/traders")
async def get_traders_status() -> List[Dict[str, Any]]:
    """
    获取所有交易员状态
    
    Returns:
        交易员状态列表
    """
    try:
        return trading_simulation.get_traders_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取交易员状态失败: {str(e)}")

@router.get("/market")
async def get_market_snapshot() -> Dict[str, Any]:
    """
    获取市场快照
    
    Returns:
        市场快照
    """
    try:
        return trading_simulation.get_market_snapshot()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取市场快照失败: {str(e)}")

@router.get("/performance")
async def get_performance() -> Dict[str, Any]:
    """
    获取交易绩效
    
    Returns:
        交易绩效
    """
    try:
        feedback_system = trading_simulation.feedback_system
        return {
            "overall": feedback_system.get_all_performance(),
            "traders": feedback_system.trader_performance,
            "model_parameters": feedback_system.get_model_parameters()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取绩效失败: {str(e)}")

@router.get("/report")
async def get_feedback_report() -> Dict[str, Any]:
    """
    获取反馈报告
    
    Returns:
        反馈报告
    """
    try:
        feedback_system = trading_simulation.feedback_system
        return feedback_system.generate_feedback_report()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成报告失败: {str(e)}")