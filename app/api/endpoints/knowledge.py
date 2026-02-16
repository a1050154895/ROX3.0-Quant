from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Any, Optional
from app.rox_quant.knowledge_base import KnowledgeBase
from app.rox_quant.vector_db_manager import VectorDatabaseManager
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/knowledge", tags=["knowledge"])

# 初始化知识库和向量数据库管理器
kb = KnowledgeBase()
vector_db = VectorDatabaseManager()

# 加载知识库
kb.load_embedded()

@router.get("/stats", response_model=Dict[str, Any])
async def get_knowledge_stats():
    """获取知识库统计信息"""
    try:
        stats = {
            "total_documents": kb.size(),
            "categories": list(kb.get_categories()),
            "top_keywords": kb.get_top_keywords(limit=20),
            "vector_db_stats": vector_db.get_statistics()
        }
        return stats
    except Exception as e:
        logger.error(f"获取知识库统计信息失败: {e}")
        raise HTTPException(status_code=500, detail="获取知识库统计信息失败")

@router.get("/categories", response_model=Dict[str, List[str]])
async def get_categories():
    """获取所有类别及其包含的文档"""
    try:
        categories = kb.get_categories()
        result = {}
        
        for category in categories:
            docs = kb.get_documents_by_category(category)
            result[category] = [doc.title for doc in docs]
        
        return result
    except Exception as e:
        logger.error(f"获取类别失败: {e}")
        raise HTTPException(status_code=500, detail="获取类别失败")

@router.get("/search", response_model=List[Dict[str, Any]])
async def search_knowledge(
    query: str = Query(..., description="搜索查询"),
    limit: int = Query(5, description="返回结果数量"),
    category: Optional[str] = Query(None, description="按类别过滤")
):
    """搜索知识库"""
    try:
        results = kb.search(query, limit=limit, category=category)
        
        # 格式化结果
        formatted_results = []
        for doc in results:
            formatted_results.append({
                "title": doc.title,
                "category": doc.category,
                "keywords": doc.keywords,
                "content_snippet": doc.content[:300] + "..." if len(doc.content) > 300 else doc.content,
                "path": doc.path
            })
        
        return formatted_results
    except Exception as e:
        logger.error(f"搜索知识库失败: {e}")
        raise HTTPException(status_code=500, detail="搜索知识库失败")

@router.get("/document/{doc_index}", response_model=Dict[str, Any])
async def get_document(
    doc_index: int,
    include_related: bool = Query(True, description="是否包含相关文档")
):
    """获取单个文档详情"""
    try:
        if doc_index >= kb.size():
            raise HTTPException(status_code=404, detail="文档不存在")
        
        doc = kb.documents[doc_index]
        
        result = {
            "title": doc.title,
            "category": doc.category,
            "keywords": doc.keywords,
            "content": doc.content,
            "path": doc.path
        }
        
        # 获取相关文档
        if include_related:
            related = kb.get_related_documents(doc_index, limit=5)
            result["related_documents"] = [
                {
                    "title": related_doc.title,
                    "similarity": similarity,
                    "category": related_doc.category,
                    "index": kb.documents.index(related_doc)
                }
                for similarity, related_doc in related
            ]
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取文档详情失败: {e}")
        raise HTTPException(status_code=500, detail="获取文档详情失败")

@router.get("/graph", response_model=Dict[str, Any])
async def get_knowledge_graph():
    """获取知识图谱"""
    try:
        graph = kb.build_knowledge_graph()
        return graph
    except Exception as e:
        logger.error(f"获取知识图谱失败: {e}")
        raise HTTPException(status_code=500, detail="获取知识图谱失败")

@router.get("/recommend", response_model=List[Dict[str, Any]])
async def get_recommendations(
    doc_index: int = Query(..., description="文档索引"),
    limit: int = Query(5, description="推荐数量")
):
    """获取相关文档推荐"""
    try:
        related = kb.get_related_documents(doc_index, limit=limit)
        
        recommendations = []
        for similarity, related_doc in related:
            recommendations.append({
                "title": related_doc.title,
                "similarity": similarity,
                "category": related_doc.category,
                "index": kb.documents.index(related_doc),
                "keywords": related_doc.keywords
            })
        
        return recommendations
    except Exception as e:
        logger.error(f"获取推荐失败: {e}")
        raise HTTPException(status_code=500, detail="获取推荐失败")

@router.get("/categories/{category}", response_model=List[Dict[str, Any]])
async def get_documents_by_category(
    category: str,
    limit: int = Query(100, description="返回数量")
):
    """按类别获取文档"""
    try:
        docs = kb.get_documents_by_category(category)
        
        results = []
        for doc in docs[:limit]:
            results.append({
                "title": doc.title,
                "keywords": doc.keywords,
                "content_snippet": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                "index": kb.documents.index(doc)
            })
        
        return results
    except Exception as e:
        logger.error(f"按类别获取文档失败: {e}")
        raise HTTPException(status_code=500, detail="按类别获取文档失败")

@router.post("/optimize")
async def optimize_vector_db():
    """优化向量数据库"""
    try:
        success = vector_db.optimize()
        if success:
            # 重新加载知识库
            kb.load_embedded()
            return {"status": "success", "message": "向量数据库优化成功"}
        else:
            raise HTTPException(status_code=500, detail="向量数据库优化失败")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"优化向量数据库失败: {e}")
        raise HTTPException(status_code=500, detail="优化向量数据库失败")

@router.post("/rebuild")
async def rebuild_vector_db(
    documents_dir: str = Query(..., description="文档目录路径")
):
    """重建向量数据库"""
    try:
        from pathlib import Path
        docs_path = Path(documents_dir)
        
        if not docs_path.exists():
            raise HTTPException(status_code=400, detail="文档目录不存在")
        
        success = vector_db.rebuild(docs_path)
        if success:
            # 重新加载知识库
            kb.load_embedded()
            return {"status": "success", "message": "向量数据库重建成功"}
        else:
            raise HTTPException(status_code=500, detail="向量数据库重建失败")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"重建向量数据库失败: {e}")
        raise HTTPException(status_code=500, detail="重建向量数据库失败")

@router.get("/context")
async def get_context(
    topic: str = Query(..., description="主题")
):
    """获取特定主题的上下文知识"""
    try:
        context = kb.get_context_for_algo(topic)
        return {"context": context}
    except Exception as e:
        logger.error(f"获取上下文知识失败: {e}")
        raise HTTPException(status_code=500, detail="获取上下文知识失败")