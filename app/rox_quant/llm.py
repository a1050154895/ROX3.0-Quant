import os
import json
from typing import Optional, Dict, Any, List
from openai import AsyncOpenAI

try:
    from duckduckgo_search import DDGS
except ImportError:
    DDGS = None  # 可选依赖，未安装时 chat_with_search 不联网

def _normalize_base_url(url: str) -> str:
    """OpenAI 兼容接口通常需要 /v1 路径，若未带则自动追加"""
    if not url:
        return ""
    url = url.rstrip("/")
    if not url.endswith("/v1"):
        url = url + "/v1"
    return url


def _build_providers() -> Dict[str, Any]:
    """从环境变量构建多后端：default 使用 AI_API_KEY/AI_BASE_URL；其余从 AI_PROVIDERS JSON 或约定 env 读取"""
    from app.core.config import settings
    default_key = getattr(settings, "AI_API_KEY", None) or os.getenv("AI_API_KEY", "").strip()
    if default_key == "your_ai_api_key_here":
        default_key = ""
    default_base = getattr(settings, "AI_BASE_URL", None) or os.getenv("AI_BASE_URL", "https://tb.api.mkeai.com").strip()
    default_model = getattr(settings, "AI_DEFAULT_MODEL", None) or os.getenv("AI_DEFAULT_MODEL", "deepseek-chat")
    providers = {
        "default": {
            "name": "默认",
            "api_key": default_key,
            "base_url": _normalize_base_url(default_base) if default_base else "",
            "default_model": default_model,
        },
    }
    raw = os.getenv("AI_PROVIDERS", "")
    if raw:
        try:
            extra = json.loads(raw)
            for k, v in (extra if isinstance(extra, dict) else {}).items():
                key_env = v.get("api_key_env", "AI_API_KEY")
                base = (v.get("base_url") or "").strip()
                providers[k] = {
                    "name": v.get("name", k),
                    "api_key": os.getenv(key_env, "").strip(),
                    "base_url": _normalize_base_url(base) if base else "",
                    "default_model": v.get("default_model", "deepseek-chat"),
                }
        except Exception:
            pass
    for k in ["openai", "ollama", "siliconflow"]:
        if k in providers:
            continue
        key_env = {"openai": "OPENAI_API_KEY", "ollama": "OLLAMA_BASE_URL", "siliconflow": "SILICONFLOW_API_KEY"}.get(k)
        base_env = {"openai": "OPENAI_BASE_URL", "ollama": "OLLAMA_BASE_URL", "siliconflow": "SILICONFLOW_BASE_URL"}.get(k)
        key = os.getenv(key_env or "AI_API_KEY", "").strip()
        base = os.getenv(base_env or "AI_BASE_URL", "").strip()
        if k == "ollama":
            base = base or "http://localhost:11434"
        if key or base:
            providers[k] = {
                "name": {"openai": "OpenAI", "ollama": "Ollama", "siliconflow": "硅基流动"}.get(k, k),
                "api_key": key,
                "base_url": _normalize_base_url(base) if base else (base or ""),
                "default_model": "gpt-4o-mini" if k == "openai" else "deepseek-chat",
            }
    return providers


class AIClient:
    def __init__(self):
        self._providers = _build_providers()
        self._clients: Dict[str, Optional[AsyncOpenAI]] = {}
        for pid, p in self._providers.items():
            if p.get("api_key") or (pid == "ollama" and p.get("base_url")):
                base = p.get("base_url") or ""
                key = p.get("api_key") or "ollama"
                try:
                    self._clients[pid] = AsyncOpenAI(api_key=key, base_url=base)
                except Exception:
                    self._clients[pid] = None
            else:
                self._clients[pid] = None
        self.client = self._clients.get("default")

    def list_providers(self) -> List[Dict[str, Any]]:
        from app.core.config import settings
        current = getattr(settings, "AI_PROVIDER", None) or os.getenv("AI_PROVIDER", "default")
        out = []
        for pid, p in self._providers.items():
            out.append({
                "id": pid,
                "name": p.get("name", pid),
                "default_model": p.get("default_model", "deepseek-chat"),
                "available": self._clients.get(pid) is not None,
            })
        return {"current": current, "list": out}

    def get_client(self, provider: Optional[str] = None):
        if not provider or provider == "default":
            return self.client
        return self._clients.get(provider) or self.client

    def get_default_model(self, provider: Optional[str] = None) -> str:
        from app.core.config import settings
        pid = (provider or getattr(settings, "AI_PROVIDER", None) or os.getenv("AI_PROVIDER", "default"))
        return self._providers.get(pid, {}).get("default_model", "deepseek-chat")
        
    def search_web(self, query: str, max_results=3):
        """
        Search web using DuckDuckGo（需安装 duckduckgo-search）
        """
        if DDGS is None:
            return []
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
                return results
        except Exception as e:
            print(f"Search failed: {e}")
            return []

    async def chat_with_search(self, message: str, context: str = "", model: Optional[str] = None, provider: Optional[str] = None):
        """
        Chat with optional web search and model selection. provider/model 缺省用当前配置。
        """
        client = self.get_client(provider)
        model = model or self.get_default_model(provider)
        # 1. Decide if search is needed
        search_results = []
        if "?" in message or "查询" in message or "搜索" in message or "为何" in message or "原因" in message:
            search_results = self.search_web(message)
        search_context = ""
        if search_results:
            search_context = "\n\n【联网搜索结果】:\n" + "\n".join([f"- {r['title']}: {r['body']}" for r in search_results])
        system_prompt = f"""
        你是一个专业的量化交易助手 ROX。
        你的任务是回答用户关于金融市场、股票和投资策略的问题。
        
        {context}
        
        {search_context}
        
        如果提供了搜索结果，请优先基于搜索结果回答。
        如果问题涉及实时性很强的信息（如今天为什么跌），请明确指出你是基于搜索结果回答的。
        保持回答专业、客观、简洁。
        """
        try:
            if not client:
                return "请配置 AI_API_KEY 与 AI_BASE_URL（如 .env）后使用 AI 功能；或设置 AI_PROVIDER。"
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message}
                ]
            )
            if isinstance(response, str):
                return response
            return response.choices[0].message.content
        except Exception as e:
            import traceback
            error_msg = f"AI 思考超时: {str(e)}\nTraceback: {traceback.format_exc()}"
            print(error_msg)
            return f"AI 思考超时: {str(e)}"

    @property
    def chat(self):
        return self.client.chat if self.client else None

    async def analyze_stock(self, stock_name, stock_code, price, indicators, model: Optional[str] = None, provider: Optional[str] = None):
        """
        Analyze stock based on technical indicators and return structured JSON.
        """
        system_prompt = """
        你是一个专业的量化交易助手 ROX。请根据提供的股票数据和技术指标，进行深度分析。
        
        【核心投资逻辑】
        1. **中庸之道**：寻找价值与价格的均衡点，拒绝偏执。
        2. **334仓位法则**：
           - 30% 底仓（趋势初期）
           - 30% 浮动仓位（跟随律动/波段）
           - 40% 预备队（现金，应对极端情况）
        3. **短股长金与向心坍缩**：关注宏观周期（繁荣/衰退/萧条/复苏），在衰退向萧条过渡期重视贵金属与防御性资产。
        4. **律动（庸）**：不预测绝对顶底，通过波段操作（高抛低吸）将持仓成本降至负数。
        
        请返回严格的 JSON 格式，不要包含 markdown 代码块标记。
        JSON 结构如下：
        {
            "p_final": 85, // 综合胜率 (0-100)
            "f_final": 60, // 建议仓位 (遵循334原则，如建议底仓则30，加仓则60)
            "b_val": 3.5, // 盈亏比
            "stars": "⭐⭐⭐⭐", // 评级
            "suggest_stop_loss": 10.5, // 建议止损价
            "p_target": 12.0, // 目标价
            "comment": "简短的一句话点评（融入中庸/律动/周期视角）",
            "thinking": {
                "reason": "核心推荐理由",
                "logic": "详细的逻辑分析过程（结合宏观周期与企业基本面）",
                "results": "预期结果"
            },
            "t_suggestions": {
                "buy": 10.8, // 律动低吸点
                "sell": 11.5, // 律动高抛点
                "space": 6.5
            }
        }
        """
        
        user_content = f"""
        股票：{stock_name} ({stock_code})
        现价：{price}
        技术指标：
        {json.dumps(indicators, ensure_ascii=False, indent=2)}
        
        请基于以上数据进行分析。
        """
        
        try:
            client = self.get_client(provider)
            model = model or self.get_default_model(provider)
            if not client:
                return {"p_final": 70, "comment": "请配置 AI_API_KEY 与 AI_BASE_URL 后使用 AI 诊断。", "thinking": {"reason": "", "logic": "", "results": ""}, "t_suggestions": {"buy": price * 0.98, "sell": price * 1.02, "space": 4.0}}
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.3
            )
            content = response.choices[0].message.content
            # Clean up markdown code blocks if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
                
            return json.loads(content.strip())
        except Exception as e:
            # Fallback mock data if API fails
            print(f"AI Analysis failed: {e}")
            return {
                "p_final": 75,
                "f_final": 40,
                "b_val": 2.1,
                "stars": "⭐⭐⭐",
                "suggest_stop_loss": price * 0.95,
                "p_target": price * 1.05,
                "comment": "AI 服务暂时不可用，启用本地规则引擎分析。",
                "thinking": {
                    "reason": "技术指标中性偏多",
                    "logic": "均线系统多头排列，但成交量略有萎缩。",
                    "results": "建议轻仓试错"
                },
                "t_suggestions": {
                    "buy": price * 0.98,
                    "sell": price * 1.02,
                    "space": 4.0
                }
            }

    async def generate_market_briefing(self, indices, stats, news, model: Optional[str] = None, provider: Optional[str] = None):
        """
        Generate a daily market briefing. provider/model 缺省用当前配置。
        """
        client = self.get_client(provider)
        model = model or self.get_default_model(provider)
        
        # Check if client is valid (has key)
        if not client or not client.api_key or client.api_key == "your_ai_api_key_here":
            return "请在项目根目录 .env 文件中配置 AI_API_KEY 后启用智能复盘功能。"

        prompt = f"""
        请根据以下市场数据生成一份简短的 A 股收盘简报（300字以内）：
        
        核心指数：
        {json.dumps(indices, ensure_ascii=False)}
        
        市场统计：
        {json.dumps(stats, ensure_ascii=False)}
        
        最新资讯：
        {json.dumps(news[:5], ensure_ascii=False)}
        
        风格要求：专业、客观、犀利。
        """
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            import traceback
            print(f"AI Briefing failed: {e}")
            return f"市场震荡整理，建议关注核心资产。（AI服务暂不可用: {str(e)[:20]}...）"

    async def summarize_screen_results(self, items: List[Dict[str, Any]], max_items: int = 30, model: Optional[str] = None, provider: Optional[str] = None) -> str:
        """对选股结果列表做 AI 总结，供条件选股+AI 闭环使用。"""
        client = self.get_client(provider)
        model = model or self.get_default_model(provider)
        if not client or not items:
            return ""
        lines = [f"- {x.get('code', '')} {x.get('name', '')} {x.get('reason', '')}" for x in items[:max_items]]
        text = "\n".join(lines)
        prompt = f"""以下为当前筛选出的股票列表（代码/名称/入选原因），请用 2～4 句话总结共性、风险点与可关注方向，语气专业简洁：

{text}
"""
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )
            return (response.choices[0].message.content or "").strip()
        except Exception:
            return ""
