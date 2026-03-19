"""
AI Comment Service for A2A (AI-to-AI) Trading Platform
AI评论服务模块，用于AI交易员对股票、策略等进行评论
"""

import random
import time
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum


class CommentType(Enum):
    STOCK_COMMENT = "stock_comment"
    STRATEGY_COMMENT = "strategy_comment"
    TRADE_COMMENT = "trade_comment"
    MARKET_COMMENT = "market_comment"
    ANALYSIS_COMMENT = "analysis_comment"


@dataclass
class Comment:
    id: str
    target_type: CommentType
    target_id: str
    author_id: str
    author_name: str
    author_personality: str
    content: str
    timestamp: datetime
    likes: int = 0
    dislikes: int = 0
    replies: List[str] = field(default_factory=list)
    sentiment: float = 0.5
    rating: Optional[int] = None
    
    def to_dict(self):
        return {
            "id": self.id,
            "target_type": self.target_type.value,
            "target_id": self.target_id,
            "author_id": self.author_id,
            "author_name": self.author_name,
            "author_personality": self.author_personality,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "likes": self.likes,
            "dislikes": self.dislikes,
            "replies": self.replies,
            "sentiment": self.sentiment,
            "rating": self.rating
        }


@dataclass
class Reply:
    id: str
    comment_id: str
    author_id: str
    author_name: str
    author_personality: str
    content: str
    timestamp: datetime
    likes: int = 0
    
    def to_dict(self):
        return {
            "id": self.id,
            "comment_id": self.comment_id,
            "author_id": self.author_id,
            "author_name": self.author_name,
            "author_personality": self.author_personality,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "likes": self.likes
        }


class AICommentGenerator:
    """AI评论内容生成器"""
    
    STOCK_COMMENTS = {
        "positive": [
            "这只股票的基本面很扎实，值得长期持有",
            "技术面显示有上涨趋势，可以关注",
            "公司业绩不错，估值合理",
            "行业前景看好，这只股票有潜力",
            "从财务数据看，这家公司很健康"
        ],
        "negative": [
            "技术面有点疲软，短期可能调整",
            "估值偏高，风险较大",
            "行业竞争激烈，要谨慎",
            "财务数据不太理想，建议观望",
            "市场情绪不好，可能受影响"
        ],
        "neutral": [
            "目前走势不明确，建议等待信号",
            "需要更多信息才能判断",
            "风险和机会并存，要谨慎",
            "可以小仓位试探，不要重仓",
            "建议设置好止损再操作"
        ]
    }
    
    STRATEGY_COMMENTS = {
        "positive": [
            "这个策略思路不错，值得尝试",
            "回测效果很好，实盘应该也不错",
            "逻辑清晰，风险可控",
            "我也在用类似的策略，效果不错",
            "这个策略在当前市场环境下很适用"
        ],
        "negative": [
            "策略过于复杂，可能过拟合",
            "风险控制不够，要小心",
            "回测数据可能有问题",
            "实盘效果可能不如回测",
            "这个策略在某些市场环境下会失效"
        ],
        "neutral": [
            "策略还可以，但需要优化",
            "建议增加一些过滤条件",
            "可以尝试调整参数",
            "需要更多数据验证",
            "可以考虑和其他策略组合"
        ]
    }
    
    TRADE_COMMENTS = {
        "positive": [
            "这笔交易时机把握得很好",
            "仓位控制合理，值得学习",
            "止损设置得很到位",
            "盈利目标设置合理",
            "风险收益比不错"
        ],
        "negative": [
            "入场时机不太理想",
            "仓位有点重，风险较大",
            "止损设置得太宽了",
            "这个位置追高不太合适",
            "建议等待更好的入场点"
        ],
        "neutral": [
            "这笔交易中规中矩",
            "可以优化一下入场时机",
            "建议调整一下仓位",
            "止损可以设置得更合理",
            "需要更多耐心等待机会"
        ]
    }
    
    MARKET_COMMENTS = {
        "positive": [
            "市场整体向好，可以积极参与",
            "政策面利好，市场情绪不错",
            "资金面宽松，有利于股市",
            "经济数据向好，支撑股市上涨",
            "技术面显示市场处于上升趋势"
        ],
        "negative": [
            "市场风险增加，要保持谨慎",
            "政策不确定性较大",
            "资金面紧张，不利于股市",
            "经济数据不及预期",
            "技术面显示市场可能调整"
        ],
        "neutral": [
            "市场处于震荡期，要耐心等待",
            "多空因素交织，方向不明",
            "建议保持中性仓位",
            "关注市场变化，灵活应对",
            "不要盲目乐观或悲观"
        ]
    }
    
    ANALYSIS_COMMENTS = {
        "positive": [
            "分析得很透彻，逻辑清晰",
            "数据详实，结论可靠",
            "观点独到，很有启发",
            "分析方法科学，值得学习",
            "这篇分析很有价值"
        ],
        "negative": [
            "分析不够全面，缺少关键因素",
            "数据可能存在偏差",
            "结论过于武断",
            "分析方法有待改进",
            "建议增加更多数据支撑"
        ],
        "neutral": [
            "分析还可以，但可以更深入",
            "建议补充一些细节",
            "可以从其他角度分析",
            "需要更多数据验证",
            "观点中肯，但不够深入"
        ]
    }
    
    @classmethod
    def generate_comment(
        cls,
        comment_type: CommentType,
        author_name: str,
        author_personality: str,
        emotion: float,
        performance: float,
        target_info: Dict = None
    ) -> str:
        """生成AI评论"""
        
        if performance > 5:
            sentiment = "positive"
        elif performance < -5:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        if comment_type == CommentType.STOCK_COMMENT:
            templates = cls.STOCK_COMMENTS[sentiment]
        elif comment_type == CommentType.STRATEGY_COMMENT:
            templates = cls.STRATEGY_COMMENTS[sentiment]
        elif comment_type == CommentType.TRADE_COMMENT:
            templates = cls.TRADE_COMMENTS[sentiment]
        elif comment_type == CommentType.MARKET_COMMENT:
            templates = cls.MARKET_COMMENTS[sentiment]
        elif comment_type == CommentType.ANALYSIS_COMMENT:
            templates = cls.ANALYSIS_COMMENTS[sentiment]
        else:
            templates = ["这个不错"]
        
        comment = random.choice(templates)
        
        personality_additions = {
            "理性投资者": ["，从数据来看确实如此。", "，分析得很到位。"],
            "激进交易者": ["，我觉得很有机会！", "，可以大胆尝试！"],
            "保守投资者": ["，不过还是要谨慎。", "，风险控制很重要。"],
            "技术派": ["，技术指标也支持这个判断。", "，从技术面看也是这样。"],
            "价值投资者": ["，长期价值确实不错。", "，基本面支撑这个观点。"],
            "成长投资者": ["，成长空间很大。", "，未来发展前景好。"],
            "量化交易者": ["，数据模型也显示这个结论。", "，量化分析结果一致。"],
            "趋势跟踪者": ["，趋势确实在往这个方向发展。", "，顺势而为很重要。"],
            "逆向投资者": ["，不过市场可能过度反应了。", "，逆向思考一下。"],
            "短线交易者": ["，短线可以操作一下。", "，抓住机会快进快出。"]
        }
        
        if author_personality in personality_additions:
            addition = random.choice(personality_additions[author_personality])
            comment += addition
        
        return comment
    
    @classmethod
    def generate_reply(
        cls,
        parent_comment: str,
        author_name: str,
        author_personality: str,
        emotion: float
    ) -> str:
        """生成回复"""
        
        reply_templates = [
            "同意你的观点",
            "我觉得你说得有道理",
            "不过我有不同看法",
            "补充一点",
            "关于这个我也想说",
            "你的分析很到位",
            "我也有同感"
        ]
        
        reply = random.choice(reply_templates)
        
        if "同意" in reply or "有道理" in reply or "同感" in reply:
            additions = [
                "，确实是这样。",
                "，我也这么认为。",
                "，支持你的观点。"
            ]
        elif "不同看法" in reply:
            additions = [
                "，我觉得可能不太一样。",
                "，从另一个角度看。",
                "，不过也要考虑其他因素。"
            ]
        else:
            additions = [
                "，希望对大家有帮助。",
                "，一起讨论。",
                "，欢迎交流。"
            ]
        
        reply += random.choice(additions)
        
        return reply


class AICommentService:
    """AI评论服务"""
    
    def __init__(self):
        self.comments: Dict[str, Comment] = {}
        self.replies: Dict[str, List[Reply]] = {}
        self.target_comments: Dict[str, List[str]] = {}
    
    def add_comment(
        self,
        target_type: CommentType,
        target_id: str,
        author_id: str,
        author_name: str,
        author_personality: str,
        content: str,
        rating: Optional[int] = None
    ) -> Comment:
        """添加评论"""
        comment_id = f"comment_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"
        
        comment = Comment(
            id=comment_id,
            target_type=target_type,
            target_id=target_id,
            author_id=author_id,
            author_name=author_name,
            author_personality=author_personality,
            content=content,
            timestamp=datetime.now(),
            sentiment=self._analyze_sentiment(content),
            rating=rating
        )
        
        self.comments[comment_id] = comment
        self.replies[comment_id] = []
        
        if target_id not in self.target_comments:
            self.target_comments[target_id] = []
        self.target_comments[target_id].append(comment_id)
        
        return comment
    
    def add_reply(
        self,
        comment_id: str,
        author_id: str,
        author_name: str,
        author_personality: str,
        content: str
    ) -> Reply:
        """添加回复"""
        if comment_id not in self.comments:
            raise ValueError(f"Comment {comment_id} not found")
        
        reply_id = f"reply_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"
        
        reply = Reply(
            id=reply_id,
            comment_id=comment_id,
            author_id=author_id,
            author_name=author_name,
            author_personality=author_personality,
            content=content,
            timestamp=datetime.now()
        )
        
        self.replies[comment_id].append(reply)
        self.comments[comment_id].replies.append(reply_id)
        
        return reply
    
    def get_comment(self, comment_id: str) -> Optional[Comment]:
        """获取评论"""
        return self.comments.get(comment_id)
    
    def get_comments_by_target(self, target_id: str, limit: int = 20) -> List[Comment]:
        """获取目标的评论"""
        if target_id not in self.target_comments:
            return []
        
        comment_ids = self.target_comments[target_id][-limit:]
        return [self.comments[cid] for cid in comment_ids if cid in self.comments]
    
    def get_replies(self, comment_id: str) -> List[Reply]:
        """获取评论的回复"""
        return self.replies.get(comment_id, [])
    
    def like_comment(self, comment_id: str) -> bool:
        """点赞评论"""
        if comment_id not in self.comments:
            return False
        
        self.comments[comment_id].likes += 1
        return True
    
    def dislike_comment(self, comment_id: str) -> bool:
        """踩评论"""
        if comment_id not in self.comments:
            return False
        
        self.comments[comment_id].dislikes += 1
        return True
    
    def generate_ai_comment(
        self,
        target_type: CommentType,
        target_id: str,
        trader_id: str,
        trader_name: str,
        trader_personality: str,
        emotion: float,
        performance: float,
        target_info: Dict = None
    ) -> Comment:
        """生成AI交易员的评论"""
        content = AICommentGenerator.generate_comment(
            comment_type=target_type,
            author_name=trader_name,
            author_personality=trader_personality,
            emotion=emotion,
            performance=performance,
            target_info=target_info
        )
        
        rating = None
        if target_type == CommentType.STOCK_COMMENT:
            rating = random.randint(3, 5) if performance > 0 else random.randint(1, 3)
        elif target_type == CommentType.STRATEGY_COMMENT:
            rating = random.randint(3, 5) if performance > 0 else random.randint(2, 4)
        
        return self.add_comment(
            target_type=target_type,
            target_id=target_id,
            author_id=trader_id,
            author_name=trader_name,
            author_personality=trader_personality,
            content=content,
            rating=rating
        )
    
    def generate_ai_reply(
        self,
        comment_id: str,
        trader_id: str,
        trader_name: str,
        trader_personality: str,
        emotion: float
    ) -> Reply:
        """生成AI交易员的回复"""
        comment = self.get_comment(comment_id)
        if not comment:
            raise ValueError(f"Comment {comment_id} not found")
        
        content = AICommentGenerator.generate_reply(
            parent_comment=comment.content,
            author_name=trader_name,
            author_personality=trader_personality,
            emotion=emotion
        )
        
        return self.add_reply(
            comment_id=comment_id,
            author_id=trader_id,
            author_name=trader_name,
            author_personality=trader_personality,
            content=content
        )
    
    def _analyze_sentiment(self, content: str) -> float:
        """简单的情感分析"""
        positive_words = ["好", "不错", "优秀", "看好", "上涨", "盈利", "机会", "潜力"]
        negative_words = ["差", "不好", "风险", "下跌", "亏损", "谨慎", "担心", "问题"]
        
        positive_count = sum(1 for word in positive_words if word in content)
        negative_count = sum(1 for word in negative_words if word in content)
        
        total = positive_count + negative_count
        if total == 0:
            return 0.5
        
        return positive_count / total


ai_comment_service = AICommentService()
