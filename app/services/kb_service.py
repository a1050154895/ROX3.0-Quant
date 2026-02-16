import os
import logging
from typing import List, Dict, Any, Optional
from app.core.config import settings

try:
    from ddgs import DDGS
except ImportError:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        DDGS = None

logger = logging.getLogger("kb-service")

class KBService:
    _instance = None
    _documents_dir_cache: Optional[List[Dict[str, Any]]] = None
    _embedded_docs_cache: Optional[List[Dict[str, Any]]] = None
    _kb_instance: Optional[Any] = None
    
    # Built-in fallback data (the "basic" knowledge for demo purposes)
    KNOWLEDGE_BASE_DATA: List[Dict] = [
        {
            "id": "k_001", "title": "什么是K线？",
            "summary": "K线图（Candlestick Charts）又称蜡烛图...是以每个分析周期的开盘价、最高价、最低价和收盘价绘制而成。",
            "tags": ["基础", "图表"], "source": "Rox 内置"
        },
        {
            "id": "k_002", "title": "MACD指标详解",
            "summary": "MACD称为异同移动平均线...由快的EMA12减去慢的EMA26得到DIF...",
            "tags": ["技术分析", "指标"], "source": "Rox 内置"
        }
    ]

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(KBService, cls).__new__(cls)
        return cls._instance
    
    def _get_kb_instance(self):
        """获取或创建KnowledgeBase实例"""
        if self._kb_instance is None:
            try:
                from app.rox_quant.knowledge_base import KnowledgeBase
                self._kb_instance = KnowledgeBase()
                self._kb_instance.load_embedded()
            except Exception as e:
                logger.error(f"Failed to create KnowledgeBase instance: {e}")
                self._kb_instance = None
        return self._kb_instance

    def _extract_text_snippet(self, fp: str, ext: str, max_chars: int = 1000) -> str:
        """Lightweight text extraction for search previews."""
        content = ""
        try:
            if ext in [".txt", ".md", ".py"]:
                with open(fp, "r", encoding="utf-8", errors="ignore") as fh:
                    content = (fh.read() or "")[:max_chars]
            elif ext == ".docx":
                try:
                    from docx import Document
                    d = Document(fp)
                    content = ("\n".join([p.text for p in d.paragraphs]) or "")[:max_chars]
                except Exception: content = ""
            elif ext == ".pdf":
                try:
                    from pdfminer.high_level import extract_text as _pdf_extract
                    content = (_pdf_extract(fp) or "")[:max_chars]
                except Exception: content = ""
        except Exception: content = ""
        return content.strip() + ("..." if len(content) >= max_chars else "")

    def _get_documents_from_data_dir(self) -> List[Dict[str, Any]]:
        """Scan app/data/documents for user-imported files."""
        if self._documents_dir_cache is not None:
            return self._documents_dir_cache
            
        try:
            doc_dir = os.path.join(settings.BASE_DIR, "app", "data", "documents")
            if not os.path.isdir(doc_dir):
                self._documents_dir_cache = []
                return []
                
            out = []
            # 限制处理的文件数量，避免请求卡住
            max_files = 50
            processed_files = 0
            
            for root, _, files in os.walk(doc_dir):
                for f in files:
                    if processed_files >= max_files:
                        break
                    
                    if f.startswith("~$") or f.startswith("."): continue
                    fp = os.path.join(root, f)
                    ext = os.path.splitext(fp)[1].lower()
                    if ext not in [".txt", ".md", ".docx", ".pdf", ".py"]: continue
                    
                    title = os.path.splitext(f)[0]
                    # Tag based on folder name (e.g., "strategies", "books")
                    subdir = os.path.basename(root)
                    tags = [subdir] if subdir in ["strategies", "books"] else []
                    
                    # 优化：对于 PDF 文件，只获取文件名而不提取文本，避免耗时操作
                    if ext == ".pdf":
                        snippet = "[PDF文档]"
                    else:
                        snippet = self._extract_text_snippet(fp, ext)
                        
                    out.append({
                        "title": title,
                        "summary": snippet,
                        "tags": tags,
                        "source": f"本地文档/{subdir}",
                        "path": fp
                    })
                    processed_files += 1
                
                if processed_files >= max_files:
                    break
            
            self._documents_dir_cache = out
            logger.info(f"Loaded {len(out)} documents from {doc_dir}")
            return out
        except Exception as e:
            logger.error(f"Failed to scan documents: {e}")
            self._documents_dir_cache = []
            return []

    def get_embedded_docs(self) -> List[Dict[str, Any]]:
        """Load embedded docs from codebase (if any)."""
        if self._embedded_docs_cache is not None:
            return self._embedded_docs_cache
        
        try:
            kb = self._get_kb_instance()
            if not kb or len(kb.documents) == 0:
                self._embedded_docs_cache = []
                return []
            
            out = []
            for doc in kb.documents:
                snippet = (doc.content or "")[:500]
                out.append({
                    "title": doc.title,
                    "summary": snippet,
                    "tags": ["内置"],
                    "source": "Rox 核心库"
                })
            
            self._embedded_docs_cache = out
            return out
        except Exception as e:
            logger.error(f"Failed to get embedded docs: {e}")
            self._embedded_docs_cache = []
            return []

    def search_local(self, query: str, limit: int = 5) -> List[Dict]:
        """Search all local sources (Built-in + User Docs)."""
        q_lower = query.strip().lower()
        if not q_lower: return []
        
        # 1. Collect all sources
        all_docs = []
        all_docs.extend(self.KNOWLEDGE_BASE_DATA)
        all_docs.extend(self.get_embedded_docs())
        all_docs.extend(self._get_documents_from_data_dir())
        
        # 2. Filter
        results = []
        for d in all_docs:
            score = 0
            title = (d.get("title") or "").lower()
            summary = (d.get("summary") or "").lower()
            tags = " ".join(d.get("tags") or []).lower()
            
            # Simple keyword matching scoring
            if q_lower in title: score += 10
            if q_lower in tags: score += 5
            if q_lower in summary: score += 1
            
            # Special logic for multi-term queries
            terms = q_lower.split()
            if len(terms) > 1:
                matches = sum(1 for t in terms if t in title or t in summary)
                if matches > 0: score += matches * 2

            if score > 0:
                d_copy = d.copy()
                d_copy["_score"] = score
                results.append(d_copy)
                
        # 3. Sort by score
        results.sort(key=lambda x: x["_score"], reverse=True)
        return results[:limit]

    def search_web(self, query: str, limit: int = 3) -> List[Dict]:
        """Perform a web search using DuckDuckGo."""
        if not DDGS: return []
        try:
            results = []
            # Add context keywords to improve relevance for financial queries
            search_q = f"{query} 股票 财经 投资"
            
            # 尝试使用新的DDGS API
            try:
                ddgs = DDGS()
                # 新API可能使用不同的参数
                for r in ddgs.text(search_q, max_results=limit):
                    results.append({
                        "title": r.get("title"),
                        "summary": r.get("body"),
                        "source": r.get("href"),
                        "type": "web"
                    })
            except Exception as new_api_error:
                logger.warning(f"New DDGS API failed: {new_api_error}")
                # 尝试使用旧的API格式
                try:
                    with DDGS() as ddgs:
                        for r in ddgs.text(search_q, region='cn-zh', safesearch='Moderate', max_results=limit):
                            results.append({
                                "title": r.get("title"),
                                "summary": r.get("body"),
                                "source": r.get("href"),
                                "type": "web"
                            })
                except Exception as old_api_error:
                    logger.warning(f"Old DDGS API failed: {old_api_error}")
                    return []
            
            return results
        except Exception as e:
            logger.warning(f"Web search failed: {e}")
            return []

    def refresh_cache(self):
        """Force re-scan of the data directory."""
        self._documents_dir_cache = None
        self._embedded_docs_cache = None
        self._kb_instance = None
        self._get_documents_from_data_dir()
