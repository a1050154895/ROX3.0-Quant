import uuid
import logging
import random
from datetime import datetime
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class AIChatService:
    """
    AI聊天服务
    
    功能:
    1. 管理多个聊天室
    2. 处理消息发送和接收
    3. 生成AI回复
    4. 支持不同的聊天主题
    """
    
    def __init__(self):
        self.rooms = {}
        self.trader_personalities = {
            "conservative": {"name": "保守型投资者", "personality": "谨慎、稳重，偏好低风险投资"},
            "aggressive": {"name": "激进型投资者", "personality": "大胆、激进，偏好高风险高回报"},
            "analytical": {"name": "技术分析专家", "personality": "理性、分析能力强，基于数据做决策"},
            "methodical": {"name": "基本面分析师", "personality": "严谨、系统，注重基本面研究"},
            "impulsive": {"name": "短线交易者", "personality": "冲动、反应快，喜欢频繁交易"},
            "patient": {"name": "长线投资者", "personality": "耐心、持久，持有时间长"},
            "balanced": {"name": "波段交易者", "personality": "平衡、灵活，兼顾短期和长期"},
            "contrarian": {"name": "逆向投资者", "personality": "逆向思维，喜欢反向操作"},
            "naive": {"name": "初学者", "personality": "新手、好奇，不断学习"}
        }
    
    def create_room(self, name: str, topic: str, description: str = "") -> Dict[str, Any]:
        """
        创建聊天室
        
        Args:
            name: 聊天室名称
            topic: 聊天主题
            description: 聊天室描述
        
        Returns:
            聊天室信息
        """
        room_id = str(uuid.uuid4())
        room = {
            "id": room_id,
            "name": name,
            "topic": topic,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "messages": [],
            "participants": []
        }
        self.rooms[room_id] = room
        logger.info(f"创建聊天室: {name} (ID: {room_id})")
        return room
    
    def get_room(self, room_id: str) -> Optional[Dict[str, Any]]:
        """
        获取聊天室
        
        Args:
            room_id: 聊天室ID
        
        Returns:
            聊天室信息
        """
        return self.rooms.get(room_id)
    
    def get_all_rooms(self) -> List[Dict[str, Any]]:
        """
        获取所有聊天室
        
        Returns:
            聊天室列表
        """
        return list(self.rooms.values())
    
    def send_message(self, room_id: str, sender: str, content: str) -> Dict[str, Any]:
        """
        发送消息
        
        Args:
            room_id: 聊天室ID
            sender: 发送者
            content: 消息内容
        
        Returns:
            消息信息
        """
        room = self.get_room(room_id)
        if not room:
            raise ValueError(f"聊天室不存在: {room_id}")
        
        message = {
            "id": str(uuid.uuid4()),
            "sender": sender,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "type": "user"
        }
        
        room["messages"].append(message)
        
        # 生成AI回复
        ai_response = self.generate_ai_response(room, message)
        if ai_response:
            room["messages"].append(ai_response)
        
        logger.info(f"消息发送到聊天室 {room_id}: {sender}: {content}")
        return message
    
    def generate_ai_response(self, room: Dict[str, Any], message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        生成AI回复
        
        Args:
            room: 聊天室信息
            message: 用户消息
        
        Returns:
            AI回复消息
        """
        # 随机选择一个AI交易员作为回复者
        personalities = list(self.trader_personalities.keys())
        selected_personality = random.choice(personalities)
        trader_info = self.trader_personalities[selected_personality]
        
        # 根据聊天主题和消息内容生成回复
        topic = room.get("topic", "general")
        content = message.get("content", "")
        
        # 生成回复内容
        response_content = self._generate_response_content(topic, content, selected_personality)
        
        # 创建AI回复消息
        ai_message = {
            "id": str(uuid.uuid4()),
            "sender": trader_info["name"],
            "content": response_content,
            "timestamp": datetime.now().isoformat(),
            "type": "ai",
            "personality": selected_personality
        }
        
        return ai_message
    
    def _generate_response_content(self, topic: str, content: str, personality: str) -> str:
        """
        生成回复内容
        
        Args:
            topic: 聊天主题
            content: 用户消息内容
            personality: AI交易员个性
        
        Returns:
            回复内容
        """
        # 根据不同主题和个性生成回复
        if topic == "market":
            responses = {
                "conservative": "近期市场波动较大，建议保持谨慎，关注低风险蓝筹股。",
                "aggressive": "市场机会很多，建议关注热点板块，积极布局高成长股。",
                "analytical": "根据技术分析，当前市场处于震荡整理阶段，需要等待明确信号。",
                "methodical": "从基本面来看，部分优质企业估值已经合理，可以考虑逐步建仓。",
                "impulsive": "市场热点切换很快，建议跟随趋势，快进快出。",
                "patient": "投资是长期的事情，不要被短期波动影响，坚持价值投资。",
                "balanced": "建议采取均衡策略，兼顾价值和成长，控制仓位。",
                "contrarian": "市场情绪已经过度悲观，可能是逆向布局的好时机。",
                "naive": "我是新手，正在学习市场分析，希望大家多指教。"
            }
        elif topic == "strategy":
            responses = {
                "conservative": "我偏好价值投资策略，关注低PE、高分红的股票。",
                "aggressive": "我喜欢动量策略，追逐市场热点，获取短期收益。",
                "analytical": "我使用技术分析结合基本面，寻找最佳入场点。",
                "methodical": "我注重基本面研究，长期持有优质企业。",
                "impulsive": "我喜欢短线交易，快进快出，追求短期利润。",
                "patient": "我是长线投资者，持有时间通常在3年以上。",
                "balanced": "我采用波段交易策略，结合技术和基本面。",
                "contrarian": "我喜欢逆向投资，在市场恐慌时买入。",
                "naive": "我正在学习各种投资策略，希望找到适合自己的方法。"
            }
        elif topic == "tech":
            responses = {
                "conservative": "科技创新很重要，但要注意估值风险。",
                "aggressive": "科技股是未来的方向，应该大胆布局。",
                "analytical": "从技术发展趋势来看，AI和半导体是长期看好的方向。",
                "methodical": "科技企业的基本面分析需要关注研发投入和商业模式。",
                "impulsive": "科技股波动大，适合短线交易。",
                "patient": "科技投资需要长期视角，关注行业领导者。",
                "balanced": "建议在科技股和传统行业之间保持平衡。",
                "contrarian": "当科技股被过度抛售时，往往是买入的好时机。",
                "naive": "我对科技行业了解不多，希望大家多分享见解。"
            }
        else:
            responses = {
                "conservative": "投资需要谨慎，风险控制是第一位的。",
                "aggressive": "机会总是留给有准备的人，要敢于抓住机会。",
                "analytical": "基于数据和分析做决策，避免情绪化交易。",
                "methodical": "投资是一项系统工程，需要严谨的研究和规划。",
                "impulsive": "市场变化很快，要及时调整策略。",
                "patient": "耐心是投资者最好的品质，不要急于求成。",
                "balanced": "保持平衡的投资组合，分散风险。",
                "contrarian": "众人恐惧我贪婪，众人贪婪我恐惧。",
                "naive": "我是投资新手，希望向大家学习。"
            }
        
        return responses.get(personality, "感谢分享，我会认真考虑你的观点。")
    
    def add_participant(self, room_id: str, participant: str):
        """
        添加参与者
        
        Args:
            room_id: 聊天室ID
            participant: 参与者名称
        """
        room = self.get_room(room_id)
        if room and participant not in room["participants"]:
            room["participants"].append(participant)
            logger.info(f"参与者 {participant} 加入聊天室 {room_id}")
    
    def remove_participant(self, room_id: str, participant: str):
        """
        移除参与者
        
        Args:
            room_id: 聊天室ID
            participant: 参与者名称
        """
        room = self.get_room(room_id)
        if room and participant in room["participants"]:
            room["participants"].remove(participant)
            logger.info(f"参与者 {participant} 离开聊天室 {room_id}")
    
    def delete_room(self, room_id: str) -> bool:
        """
        删除聊天室
        
        Args:
            room_id: 聊天室ID
        
        Returns:
            是否删除成功
        """
        if room_id in self.rooms:
            del self.rooms[room_id]
            logger.info(f"删除聊天室: {room_id}")
            return True
        return False
    
    def get_room_messages(self, room_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        获取聊天室消息
        
        Args:
            room_id: 聊天室ID
            limit: 消息数量限制
        
        Returns:
            消息列表
        """
        room = self.get_room(room_id)
        if not room:
            return []
        return room["messages"][-limit:]
    
    def init_default_rooms(self):
        """
        初始化默认聊天室
        """
        default_rooms = [
            {
                "name": "市场讨论",
                "topic": "market",
                "description": "讨论市场走势、热点板块和投资机会"
            },
            {
                "name": "策略交流",
                "topic": "strategy",
                "description": "分享投资策略、交易方法和风险管理"
            },
            {
                "name": "科技前沿",
                "topic": "tech",
                "description": "讨论科技行业发展、创新趋势和投资机会"
            },
            {
                "name": "综合讨论",
                "topic": "general",
                "description": "自由讨论各种投资相关话题"
            }
        ]
        
        for room_config in default_rooms:
            self.create_room(**room_config)
        
        logger.info("初始化默认聊天室完成")

# 创建全局AI聊天服务实例
ai_chat_service = AIChatService()
ai_chat_service.init_default_rooms()