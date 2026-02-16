import logging
import json
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import sqlite3

from app.rox_quant.llm import AIClient
from app.rox_quant.data_provider import DataProvider

logger = logging.getLogger(__name__)

class AIReviewService:
    """
    AI交易复盘服务
    负责分析交易数据，生成详细的交易统计，并通过LLM生成智能复盘报告
    """
    
    def __init__(self, db_path: str = "rox.db"):
        self.db_path = db_path
        self.ai_client = AIClient()
        self.data_provider = DataProvider()
    
    def get_trade_history(self, user_id: int, since: datetime, until: Optional[datetime] = None) -> List[Dict]:
        """获取指定时间范围内的交易历史"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute(
                "SELECT id, symbol, name, account_type, side, open_price, open_quantity, close_price, open_time, close_time, stop_loss, take_profit, strategy_note "
                "FROM trades WHERE user_id = ?",
                (user_id,)
            )
            
            trades = []
            for row in cursor.fetchall():
                trade = {
                    "id": row[0],
                    "symbol": row[1],
                    "name": row[2],
                    "account_type": row[3],
                    "side": row[4],
                    "open_price": row[5],
                    "open_quantity": row[6],
                    "close_price": row[7],
                    "open_time": row[8],
                    "close_time": row[9],
                    "stop_loss": row[10],
                    "take_profit": row[11],
                    "strategy_note": row[12]
                }
                
                # 过滤时间范围
                open_time = self._parse_timestamp(trade.get("open_time"))
                if open_time and open_time >= since:
                    if until is None or open_time <= until:
                        trades.append(trade)
            
            return trades
        except Exception as e:
            logger.error(f"Error getting trade history: {e}")
            return []
        finally:
            if conn:
                conn.close()
    
    def analyze_trades(self, trades: List[Dict]) -> Dict[str, Any]:
        """分析交易数据，生成统计指标"""
        if not trades:
            return {
                "total_trades": 0,
                "buy_trades": 0,
                "sell_trades": 0,
                "closed_trades": 0,
                "open_trades": 0,
                "win_rate": 0,
                "average_win": 0,
                "average_loss": 0,
                "profit_factor": 0,
                "total_pnl": 0,
                "average_holding_time": 0,
                "stop_loss_triggers": 0,
                "take_profit_triggers": 0,
                "sector_distribution": {},
                "account_distribution": {},
                "strategy_performance": {}
            }
        
        # 基础统计
        total_trades = len(trades)
        buy_trades = sum(1 for t in trades if t.get("side", "").lower() == "buy")
        sell_trades = total_trades - buy_trades
        closed_trades = sum(1 for t in trades if t.get("close_price") is not None)
        open_trades = total_trades - closed_trades
        
        # 盈亏分析
        wins = []
        losses = []
        total_pnl = 0
        stop_loss_triggers = 0
        take_profit_triggers = 0
        holding_times = []
        
        for trade in trades:
            if trade.get("close_price") is not None:
                # 已平仓交易
                open_price = float(trade.get("open_price", 0))
                close_price = float(trade.get("close_price", 0))
                quantity = float(trade.get("open_quantity", 0))
                
                if trade.get("side", "").lower() == "buy":
                    pnl = (close_price - open_price) * quantity
                else:
                    pnl = (open_price - close_price) * quantity
                
                total_pnl += pnl
                
                if pnl > 0:
                    wins.append(pnl)
                elif pnl < 0:
                    losses.append(abs(pnl))
                
                # 计算持仓时间
                open_time = self._parse_timestamp(trade.get("open_time"))
                close_time = self._parse_timestamp(trade.get("close_time"))
                if open_time and close_time:
                    holding_minutes = (close_time - open_time).total_seconds() / 60
                    holding_times.append(holding_minutes)
                
                # 检查是否触发止盈止损
                stop_loss = trade.get("stop_loss")
                take_profit = trade.get("take_profit")
                if stop_loss and close_price <= float(stop_loss):
                    stop_loss_triggers += 1
                if take_profit and close_price >= float(take_profit):
                    take_profit_triggers += 1
        
        # 计算胜率和盈亏比
        win_rate = len(wins) / closed_trades if closed_trades > 0 else 0
        average_win = sum(wins) / len(wins) if wins else 0
        average_loss = sum(losses) / len(losses) if losses else 0
        profit_factor = sum(wins) / sum(losses) if losses else 0
        average_holding_time = sum(holding_times) / len(holding_times) if holding_times else 0
        
        # 行业分布分析
        sector_distribution = self._analyze_sector_distribution(trades)
        
        # 账户分布分析
        account_distribution = {}
        for trade in trades:
            account_type = trade.get("account_type", "unknown")
            account_distribution[account_type] = account_distribution.get(account_type, 0) + 1
        
        # 策略表现分析
        strategy_performance = self._analyze_strategy_performance(trades)
        
        return {
            "total_trades": total_trades,
            "buy_trades": buy_trades,
            "sell_trades": sell_trades,
            "closed_trades": closed_trades,
            "open_trades": open_trades,
            "win_rate": win_rate,
            "average_win": average_win,
            "average_loss": average_loss,
            "profit_factor": profit_factor,
            "total_pnl": total_pnl,
            "average_holding_time": average_holding_time,
            "stop_loss_triggers": stop_loss_triggers,
            "take_profit_triggers": take_profit_triggers,
            "sector_distribution": sector_distribution,
            "account_distribution": account_distribution,
            "strategy_performance": strategy_performance
        }
    
    async def generate_review(self, user_id: int, period: str = "week") -> Dict[str, Any]:
        """生成交易复盘报告"""
        # 计算时间范围
        now = datetime.utcnow()
        if period == "month":
            since = now - timedelta(days=30)
            label = "本月"
        elif period == "quarter":
            since = now - timedelta(days=90)
            label = "本季度"
        elif period == "year":
            since = now - timedelta(days=365)
            label = "本年"
        else:
            since = now - timedelta(days=7)
            label = "本周"
        
        # 获取交易历史
        trades = self.get_trade_history(user_id, since)
        
        # 分析交易数据
        analysis = self.analyze_trades(trades)
        
        # 生成AI复盘
        ai_summary = await self._generate_ai_summary(label, trades, analysis)
        
        # 生成改进建议
        improvement_suggestions = await self._generate_improvement_suggestions(analysis)
        
        # 生成下一步策略建议
        strategy_suggestions = await self._generate_strategy_suggestions(analysis)
        
        return {
            "period": period,
            "label": label,
            "time_range": {
                "since": since.isoformat(),
                "until": now.isoformat()
            },
            "analysis": analysis,
            "ai_summary": ai_summary,
            "improvement_suggestions": improvement_suggestions,
            "strategy_suggestions": strategy_suggestions,
            "trade_count": len(trades)
        }
    
    async def _generate_ai_summary(self, label: str, trades: List[Dict], analysis: Dict) -> str:
        """生成AI复盘摘要"""
        if not trades:
            return f"{label}暂无交易记录，建议开始模拟交易以积累数据。"
        
        try:
            # 构造详细的prompt
            stats_text = self._format_analysis_stats(analysis)
            trade_samples = self._format_trade_samples(trades[:10])  # 只取前10笔交易作为样本
            
            prompt = f"""
            你是一位专业的量化交易复盘专家，需要根据以下交易数据和统计信息，生成一份详细的交易复盘报告。
            
            【时间范围】
            {label}
            
            【交易统计】
            {stats_text}
            
            【交易样本】
            {trade_samples}
            
            【要求】
            1. 用中文撰写，语气专业、客观、简洁
            2. 首先总结本周期的整体交易表现
            3. 分析交易中的优势和不足
            4. 基于统计数据，提供具体的改进建议
            5. 结合市场环境，给出下一步的策略建议
            6. 报告长度控制在300-500字
            """
            
            if self.ai_client.client:
                out = await self.ai_client.chat_with_search(
                    prompt,
                    context="你是专业的量化交易复盘专家，擅长分析交易数据并给出有深度的复盘报告。"
                )
                return (out or "").strip()
            else:
                return self._generate_fallback_summary(label, analysis)
                
        except Exception as e:
            logger.error(f"Error generating AI summary: {e}")
            return self._generate_fallback_summary(label, analysis)
    
    async def _generate_improvement_suggestions(self, analysis: Dict) -> List[str]:
        """生成改进建议"""
        try:
            prompt = f"""
            基于以下交易统计数据，生成5条具体的改进建议：
            
            {json.dumps(analysis, ensure_ascii=False, indent=2)}
            
            要求：
            1. 每条建议要具体可行
            2. 结合统计数据中的具体问题
            3. 建议要具有可操作性
            4. 用中文回复，每条建议单独一行
            """
            
            if self.ai_client.client:
                out = await self.ai_client.chat_with_search(
                    prompt,
                    context="你是专业的交易改进顾问，擅长基于数据分析给出具体的交易改进建议。"
                )
                if out:
                    suggestions = [s.strip() for s in out.split('\n') if s.strip()]
                    return suggestions[:5]  # 最多返回5条建议
            
            return self._generate_fallback_improvements(analysis)
            
        except Exception as e:
            logger.error(f"Error generating improvement suggestions: {e}")
            return self._generate_fallback_improvements(analysis)
    
    async def _generate_strategy_suggestions(self, analysis: Dict) -> Dict[str, Any]:
        """生成策略建议"""
        try:
            prompt = f"""
            基于以下交易统计数据，生成下一步的策略建议：
            
            {json.dumps(analysis, ensure_ascii=False, indent=2)}
            
            要求：
            1. 分析当前交易表现的优势和劣势
            2. 给出具体的仓位管理建议
            3. 给出止盈止损策略建议
            4. 给出行业配置建议
            5. 给出交易频率和时机建议
            6. 用中文回复，结构清晰
            """
            
            if self.ai_client.client:
                out = await self.ai_client.chat_with_search(
                    prompt,
                    context="你是专业的交易策略顾问，擅长基于数据分析给出全面的策略建议。"
                )
                if out:
                    return {"suggestions": out.strip()}
            
            return {"suggestions": "建议保持当前策略，适当调整仓位管理，控制交易频率，严格执行止盈止损策略。"}
            
        except Exception as e:
            logger.error(f"Error generating strategy suggestions: {e}")
            return {"suggestions": "建议保持当前策略，适当调整仓位管理，控制交易频率，严格执行止盈止损策略。"}
    
    def _parse_timestamp(self, timestamp: Optional[str]) -> Optional[datetime]:
        """解析时间戳"""
        if not timestamp:
            return None
        try:
            return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except Exception:
            return None
    
    def _analyze_sector_distribution(self, trades: List[Dict]) -> Dict[str, int]:
        """分析行业分布"""
        # 简化的行业分布分析，实际应用中可以通过股票代码映射到行业
        sector_distribution = {}
        for trade in trades:
            # 这里简化处理，实际应该通过股票代码查询行业
            sector = "未知行业"
            sector_distribution[sector] = sector_distribution.get(sector, 0) + 1
        return sector_distribution
    
    def _analyze_strategy_performance(self, trades: List[Dict]) -> Dict[str, Dict]:
        """分析策略表现"""
        strategy_performance = {}
        for trade in trades:
            strategy = trade.get("strategy_note", "默认策略")
            if strategy not in strategy_performance:
                strategy_performance[strategy] = {
                    "trades": 0,
                    "wins": 0,
                    "losses": 0,
                    "total_pnl": 0
                }
            
            perf = strategy_performance[strategy]
            perf["trades"] += 1
            
            if trade.get("close_price") is not None:
                open_price = float(trade.get("open_price", 0))
                close_price = float(trade.get("close_price", 0))
                quantity = float(trade.get("open_quantity", 0))
                
                if trade.get("side", "").lower() == "buy":
                    pnl = (close_price - open_price) * quantity
                else:
                    pnl = (open_price - close_price) * quantity
                
                perf["total_pnl"] += pnl
                if pnl > 0:
                    perf["wins"] += 1
                elif pnl < 0:
                    perf["losses"] += 1
        
        return strategy_performance
    
    def _format_analysis_stats(self, analysis: Dict) -> str:
        """格式化分析统计数据"""
        stats = []
        stats.append(f"总交易笔数: {analysis['total_trades']}")
        stats.append(f"买入笔数: {analysis['buy_trades']}")
        stats.append(f"卖出笔数: {analysis['sell_trades']}")
        stats.append(f"已平仓笔数: {analysis['closed_trades']}")
        stats.append(f"未平仓笔数: {analysis['open_trades']}")
        stats.append(f"胜率: {analysis['win_rate']:.2%}")
        stats.append(f"平均盈利: {analysis['average_win']:.2f}")
        stats.append(f"平均亏损: {analysis['average_loss']:.2f}")
        stats.append(f"盈亏比: {analysis['profit_factor']:.2f}")
        stats.append(f"总盈亏: {analysis['total_pnl']:.2f}")
        stats.append(f"平均持仓时间: {analysis['average_holding_time']:.2f}分钟")
        stats.append(f"止损触发次数: {analysis['stop_loss_triggers']}")
        stats.append(f"止盈触发次数: {analysis['take_profit_triggers']}")
        return "\n".join(stats)
    
    def _format_trade_samples(self, trades: List[Dict]) -> str:
        """格式化交易样本"""
        samples = []
        for trade in trades:
            side = trade.get("side", "")
            symbol = trade.get("symbol", "")
            name = trade.get("name", "")
            open_price = trade.get("open_price", "")
            close_price = trade.get("close_price", "")
            quantity = trade.get("open_quantity", "")
            open_time = trade.get("open_time", "")
            
            sample = f"- {side} {symbol} {name} 开仓价{open_price} 平仓价{close_price} 数量{quantity} 时间{open_time}"
            samples.append(sample)
        
        return "\n".join(samples)
    
    def _generate_fallback_summary(self, label: str, analysis: Dict) -> str:
        """生成兜底摘要"""
        total_trades = analysis['total_trades']
        win_rate = analysis['win_rate']
        total_pnl = analysis['total_pnl']
        
        if total_trades == 0:
            return f"{label}暂无交易记录，建议开始模拟交易以积累数据。"
        
        if total_pnl > 0:
            pnl_desc = f"总盈利 {total_pnl:.2f}"
        else:
            pnl_desc = f"总亏损 {total_pnl:.2f}"
        
        return f"{label}共 {total_trades} 笔交易，胜率 {win_rate:.2%}，{pnl_desc}。建议结合仓位管理和止盈止损策略进行优化，控制交易频率，提高交易质量。"
    
    def _generate_fallback_improvements(self, analysis: Dict) -> List[str]:
        """生成兜底改进建议"""
        suggestions = [
            "严格执行止盈止损策略，控制单笔交易风险",
            "优化仓位管理，避免过度集中持仓",
            "控制交易频率，提高交易质量",
            "加强对市场趋势的判断，顺势而为",
            "定期复盘，总结交易经验教训"
        ]
        return suggestions

# 全局AI复盘服务实例
_ai_review_service_instance = None
_ai_review_service_lock = threading.Lock()

def get_ai_review_service() -> AIReviewService:
    """获取AI复盘服务实例"""
    global _ai_review_service_instance
    
    with _ai_review_service_lock:
        if _ai_review_service_instance is None:
            _ai_review_service_instance = AIReviewService()
        
    return _ai_review_service_instance
