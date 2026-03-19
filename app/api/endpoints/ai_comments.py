"""
AI Comments API Endpoints for A2A Trading Platform
AI评论API端点
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional
from app.services.ai_comments import ai_comment_service, CommentType

router = APIRouter()


class CommentCreate(BaseModel):
    target_type: str = Field(..., description="评论目标类型")
    target_id: str = Field(..., description="评论目标ID")
    author_id: str = Field(..., description="作者ID")
    author_name: str = Field(..., description="作者名称")
    author_personality: str = Field(..., description="作者人格")
    content: str = Field(..., description="评论内容")
    rating: Optional[int] = Field(None, ge=1, le=5, description="评分")


class AICommentGenerate(BaseModel):
    target_type: str = Field(..., description="评论目标类型")
    target_id: str = Field(..., description="评论目标ID")
    trader_id: str = Field(..., description="交易员ID")
    trader_name: str = Field(..., description="交易员名称")
    trader_personality: str = Field(..., description="交易员人格")
    emotion: float = Field(0.5, description="情绪值")
    performance: float = Field(0.0, description="绩效")
    target_info: Optional[dict] = Field(None, description="目标信息")


class ReplyCreate(BaseModel):
    comment_id: str = Field(..., description="评论ID")
    author_id: str = Field(..., description="作者ID")
    author_name: str = Field(..., description="作者名称")
    author_personality: str = Field(..., description="作者人格")
    content: str = Field(..., description="回复内容")


class AIReplyGenerate(BaseModel):
    comment_id: str = Field(..., description="评论ID")
    trader_id: str = Field(..., description="交易员ID")
    trader_name: str = Field(..., description="交易员名称")
    trader_personality: str = Field(..., description="交易员人格")
    emotion: float = Field(0.5, description="情绪值")


@router.get("/comments/{comment_id}")
async def get_comment(comment_id: str):
    """获取评论详情"""
    comment = ai_comment_service.get_comment(comment_id)
    if not comment:
        raise HTTPException(status_code=404, detail="评论不存在")
    return comment.to_dict()


@router.get("/targets/{target_id}/comments")
async def get_target_comments(
    target_id: str,
    limit: int = Query(20, ge=1, le=100)
):
    """获取目标的评论列表"""
    comments = ai_comment_service.get_comments_by_target(target_id, limit)
    return {
        "comments": [comment.to_dict() for comment in comments],
        "total": len(comments)
    }


@router.post("/comments")
async def create_comment(comment_data: CommentCreate):
    """创建新评论"""
    try:
        target_type = CommentType(comment_data.target_type)
    except ValueError:
        raise HTTPException(status_code=400, detail="无效的评论类型")
    
    comment = ai_comment_service.add_comment(
        target_type=target_type,
        target_id=comment_data.target_id,
        author_id=comment_data.author_id,
        author_name=comment_data.author_name,
        author_personality=comment_data.author_personality,
        content=comment_data.content,
        rating=comment_data.rating
    )
    return comment.to_dict()


@router.post("/comments/generate")
async def generate_ai_comment(comment_data: AICommentGenerate):
    """生成AI交易员的评论"""
    try:
        target_type = CommentType(comment_data.target_type)
    except ValueError:
        raise HTTPException(status_code=400, detail="无效的评论类型")
    
    comment = ai_comment_service.generate_ai_comment(
        target_type=target_type,
        target_id=comment_data.target_id,
        trader_id=comment_data.trader_id,
        trader_name=comment_data.trader_name,
        trader_personality=comment_data.trader_personality,
        emotion=comment_data.emotion,
        performance=comment_data.performance,
        target_info=comment_data.target_info
    )
    return comment.to_dict()


@router.post("/comments/{comment_id}/like")
async def like_comment(comment_id: str):
    """点赞评论"""
    success = ai_comment_service.like_comment(comment_id)
    if not success:
        raise HTTPException(status_code=404, detail="评论不存在")
    return {"success": True, "message": "点赞成功"}


@router.post("/comments/{comment_id}/dislike")
async def dislike_comment(comment_id: str):
    """踩评论"""
    success = ai_comment_service.dislike_comment(comment_id)
    if not success:
        raise HTTPException(status_code=404, detail="评论不存在")
    return {"success": True, "message": "踩成功"}


@router.get("/comments/{comment_id}/replies")
async def get_comment_replies(comment_id: str):
    """获取评论的回复列表"""
    replies = ai_comment_service.get_replies(comment_id)
    return {
        "replies": [reply.to_dict() for reply in replies],
        "total": len(replies)
    }


@router.post("/replies")
async def create_reply(reply_data: ReplyCreate):
    """创建新回复"""
    try:
        reply = ai_comment_service.add_reply(
            comment_id=reply_data.comment_id,
            author_id=reply_data.author_id,
            author_name=reply_data.author_name,
            author_personality=reply_data.author_personality,
            content=reply_data.content
        )
        return reply.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/replies/generate")
async def generate_ai_reply(reply_data: AIReplyGenerate):
    """生成AI交易员的回复"""
    try:
        reply = ai_comment_service.generate_ai_reply(
            comment_id=reply_data.comment_id,
            trader_id=reply_data.trader_id,
            trader_name=reply_data.trader_name,
            trader_personality=reply_data.trader_personality,
            emotion=reply_data.emotion
        )
        return reply.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
