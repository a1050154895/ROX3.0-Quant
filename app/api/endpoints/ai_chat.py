from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List

from app.services.ai_chat import ai_chat_service

router = APIRouter(prefix="/ai-chat", tags=["ai-chat"])

@router.get("/rooms")
async def get_rooms() -> List[Dict[str, Any]]:
    """
    获取所有聊天室
    
    Returns:
        聊天室列表
    """
    try:
        return ai_chat_service.get_all_rooms()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取聊天室失败: {str(e)}")

@router.post("/rooms")
async def create_room(name: str, topic: str, description: str = "") -> Dict[str, Any]:
    """
    创建聊天室
    
    Args:
        name: 聊天室名称
        topic: 聊天主题
        description: 聊天室描述
    
    Returns:
        聊天室信息
    """
    try:
        return ai_chat_service.create_room(name, topic, description)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创建聊天室失败: {str(e)}")

@router.get("/rooms/{room_id}")
async def get_room(room_id: str) -> Dict[str, Any]:
    """
    获取聊天室详情
    
    Args:
        room_id: 聊天室ID
    
    Returns:
        聊天室信息
    """
    try:
        room = ai_chat_service.get_room(room_id)
        if not room:
            raise HTTPException(status_code=404, detail="聊天室不存在")
        return room
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取聊天室失败: {str(e)}")

@router.delete("/rooms/{room_id}")
async def delete_room(room_id: str) -> Dict[str, Any]:
    """
    删除聊天室
    
    Args:
        room_id: 聊天室ID
    
    Returns:
        删除结果
    """
    try:
        success = ai_chat_service.delete_room(room_id)
        if not success:
            raise HTTPException(status_code=404, detail="聊天室不存在")
        return {"status": "success", "message": "聊天室已删除"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除聊天室失败: {str(e)}")

@router.get("/rooms/{room_id}/messages")
async def get_room_messages(room_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    """
    获取聊天室消息
    
    Args:
        room_id: 聊天室ID
        limit: 消息数量限制
    
    Returns:
        消息列表
    """
    try:
        return ai_chat_service.get_room_messages(room_id, limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取消息失败: {str(e)}")

@router.post("/rooms/{room_id}/messages")
async def send_message(room_id: str, sender: str, content: str) -> Dict[str, Any]:
    """
    发送消息
    
    Args:
        room_id: 聊天室ID
        sender: 发送者
        content: 消息内容
    
    Returns:
        消息信息
    """
    try:
        return ai_chat_service.send_message(room_id, sender, content)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"发送消息失败: {str(e)}")

@router.post("/rooms/{room_id}/participants")
async def add_participant(room_id: str, participant: str) -> Dict[str, Any]:
    """
    添加参与者
    
    Args:
        room_id: 聊天室ID
        participant: 参与者名称
    
    Returns:
        添加结果
    """
    try:
        ai_chat_service.add_participant(room_id, participant)
        return {"status": "success", "message": f"参与者 {participant} 已添加"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"添加参与者失败: {str(e)}")

@router.delete("/rooms/{room_id}/participants/{participant}")
async def remove_participant(room_id: str, participant: str) -> Dict[str, Any]:
    """
    移除参与者
    
    Args:
        room_id: 聊天室ID
        participant: 参与者名称
    
    Returns:
        移除结果
    """
    try:
        ai_chat_service.remove_participant(room_id, participant)
        return {"status": "success", "message": f"参与者 {participant} 已移除"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"移除参与者失败: {str(e)}")

@router.post("/init-default-rooms")
async def init_default_rooms() -> Dict[str, Any]:
    """
    初始化默认聊天室
    
    Returns:
        初始化结果
    """
    try:
        ai_chat_service.init_default_rooms()
        return {"status": "success", "message": "默认聊天室已初始化"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"初始化默认聊天室失败: {str(e)}")