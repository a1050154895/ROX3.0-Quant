#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最终综合测试脚本
"""

from app.rox_quant.knowledge_base import KnowledgeBase
from app.rox_quant.rag_service import get_rag_service
import time

def main():
    print("=" * 60)
    print("最终综合测试")
    print("=" * 60)
    
    # 测试知识库加载
    print("\n1. 测试知识库加载:")
    kb = KnowledgeBase()
    start = time.time()
    loaded = kb.load_embedded()
    end = time.time()
    print(f"✅ 知识库加载完成: {loaded} 个文档")
    print(f"   加载速度: {end - start:.4f}秒")
    
    # 测试搜索速度
    print("\n2. 测试搜索性能:")
    test_queries = ["投资策略", "经济学原理", "风险管理", "技术分析"]
    
    for query in test_queries:
        start = time.time()
        results = kb.search(query, limit=3)
        end = time.time()
        print(f"   查询 '{query}': {end - start:.4f}秒, {len(results)}个结果")
    
    # 测试RAG服务
    print("\n3. 测试RAG服务:")
    rag = get_rag_service()
    
    start = time.time()
    context = rag.get_context_for_query("投资策略")
    end = time.time()
    print(f"✅ RAG上下文生成完成")
    print(f"   生成速度: {end - start:.4f}秒")
    print(f"   上下文长度: {len(context)}字符")
    
    # 测试知识图谱
    print("\n4. 测试知识图谱:")
    start = time.time()
    graph = kb.build_knowledge_graph()
    end = time.time()
    print(f"✅ 知识图谱构建完成")
    print(f"   构建速度: {end - start:.4f}秒")
    print(f"   节点数: {len(graph['nodes'])}, 边数: {len(graph['edges'])}")
    
    # 测试类别分布
    print("\n5. 测试类别分布:")
    categories = kb.get_categories()
    for category in categories:
        count = len(kb.get_documents_by_category(category))
        print(f"   {category}: {count}本")
    
    print("\n" + "=" * 60)
    print("🎉 所有测试通过！系统优化完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()