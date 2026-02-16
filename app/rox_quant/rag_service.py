#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG增强服务
集成知识库到AI系统，提供智能知识检索和融合能力
"""

import logging
from typing import List, Dict, Any, Optional
from app.rox_quant.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)

class RAGService:
    """RAG增强服务"""
    
    def __init__(self):
        """初始化RAG服务"""
        self.kb = KnowledgeBase()
        # 加载嵌入式知识库
        self.kb.load_embedded()
        logger.info(f"✅ RAG服务初始化完成，已加载 {self.kb.size()} 个文档")
    
    def search_knowledge(self, query: str, limit: int = 3, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """搜索知识库"""
        try:
            results = self.kb.search(query, limit=limit, category=category)
            
            # 格式化结果
            formatted_results = []
            for doc in results:
                formatted_results.append({
                    "title": doc.title,
                    "content": doc.content,
                    "category": doc.category,
                    "keywords": doc.keywords,
                    "summary": doc.content[:300] + "..." if len(doc.content) > 300 else doc.content
                })
            
            return formatted_results
        except Exception as e:
            logger.error(f"搜索知识库失败: {e}")
            return []
    
    def get_context_for_query(self, query: str, max_context_length: int = 2000) -> str:
        """获取查询的上下文知识"""
        try:
            # 搜索相关文档
            results = self.kb.search(query, limit=5)
            
            if not results:
                return ""
            
            context_parts = []
            total_length = 0
            
            for doc in results:
                # 计算文档相关度
                relevance = self._calculate_relevance(query, doc)
                
                # 提取关键部分
                key_part = self._extract_key_part(query, doc.content)
                
                if key_part:
                    part = f"【{doc.title}】\n{key_part}\n"
                    part_length = len(part)
                    
                    if total_length + part_length <= max_context_length:
                        context_parts.append(part)
                        total_length += part_length
                    else:
                        break
            
            context = "\n".join(context_parts)
            return context
        except Exception as e:
            logger.error(f"获取上下文失败: {e}")
            return ""
    
    def _calculate_relevance(self, query: str, doc) -> float:
        """计算查询与文档的相关度"""
        try:
            # 简单的相关度计算
            query_lower = query.lower()
            title_score = doc.title.lower().count(query_lower) * 2
            content_score = doc.content.lower().count(query_lower)
            keyword_score = sum(1 for kw in doc.keywords if query_lower in kw.lower()) * 1.5
            
            total_score = title_score + content_score + keyword_score
            return total_score
        except Exception:
            return 0.0
    
    def _extract_key_part(self, query: str, content: str) -> str:
        """提取与查询相关的关键部分"""
        try:
            # 简单实现：查找包含查询关键词的段落
            paragraphs = content.split('\n')
            relevant_paragraphs = []
            
            for para in paragraphs:
                if query.lower() in para.lower() and len(para) > 50:
                    relevant_paragraphs.append(para.strip())
                    if len(relevant_paragraphs) >= 2:
                        break
            
            if relevant_paragraphs:
                return "\n".join(relevant_paragraphs[:2])
            
            # 如果没有找到相关段落，返回开头部分
            return content[:200]
        except Exception:
            return content[:200]
    
    def get_multi_context(self, query: str, categories: List[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """获取多类别上下文"""
        try:
            if not categories:
                categories = list(self.kb.get_categories())
            
            context_by_category = {}
            
            for category in categories:
                results = self.kb.search(query, limit=2, category=category)
                if results:
                    context_by_category[category] = [
                        {
                            "title": doc.title,
                            "summary": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content
                        }
                        for doc in results
                    ]
            
            return context_by_category
        except Exception as e:
            logger.error(f"获取多类别上下文失败: {e}")
            return {}
    
    def get_document_context(self, doc_index: int) -> Dict[str, Any]:
        """获取特定文档的上下文"""
        try:
            if doc_index >= self.kb.size():
                return {}
            
            doc = self.kb.documents[doc_index]
            
            # 获取相关文档
            related = self.kb.get_related_documents(doc_index, limit=3)
            
            return {
                "document": {
                    "title": doc.title,
                    "content": doc.content,
                    "category": doc.category,
                    "keywords": doc.keywords
                },
                "related": [
                    {
                        "title": related_doc.title,
                        "similarity": similarity,
                        "summary": related_doc.content[:150] + "..."
                    }
                    for similarity, related_doc in related
                ]
            }
        except Exception as e:
            logger.error(f"获取文档上下文失败: {e}")
            return {}
    
    def build_knowledge_graph(self) -> Dict[str, Any]:
        """构建知识图谱"""
        try:
            return self.kb.build_knowledge_graph()
        except Exception as e:
            logger.error(f"构建知识图谱失败: {e}")
            return {"nodes": [], "edges": []}
    
    def get_category_distribution(self) -> Dict[str, int]:
        """获取类别分布"""
        try:
            categories = self.kb.get_categories()
            distribution = {}
            
            for category in categories:
                docs = self.kb.get_documents_by_category(category)
                distribution[category] = len(docs)
            
            return distribution
        except Exception as e:
            logger.error(f"获取类别分布失败: {e}")
            return {}
    
    def update_knowledge(self):
        """更新知识库"""
        try:
            # 重新加载知识库
            self.kb.load_embedded()
            logger.info(f"✅ 知识库更新完成，当前包含 {self.kb.size()} 个文档")
            return True
        except Exception as e:
            logger.error(f"更新知识库失败: {e}")
            return False

# 全局RAG服务实例
rag_service = None

def get_rag_service() -> RAGService:
    """获取RAG服务实例"""
    global rag_service
    if not rag_service:
        rag_service = RAGService()
    return rag_service