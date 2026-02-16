#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
书籍内容提取和处理系统
用于批量处理ROX 3.0中的书籍资源，构建知识库
"""

import os
import sys
import logging
import time
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.rox_quant.knowledge_base import KnowledgeBase

def get_books_directory():
    """获取书籍目录路径"""
    # 当前脚本所在目录
    script_dir = Path(__file__).parent
    # 书籍目录应该在 app/data/documents
    books_dir = script_dir / "app" / "data" / "documents"
    return books_dir

def process_books():
    """处理所有书籍"""
    logger.info("=" * 80)
    logger.info("📚 开始处理书籍资源")
    logger.info("=" * 80)
    
    start_time = time.time()
    
    # 获取书籍目录
    books_dir = get_books_directory()
    if not books_dir.exists():
        logger.error(f"❌ 书籍目录不存在: {books_dir}")
        return False
    
    logger.info(f"📁 书籍目录: {books_dir}")
    
    # 初始化知识库
    kb = KnowledgeBase()
    
    # 构建嵌入式知识库
    processed_count = kb.build_embedded_from_dir(str(books_dir))
    
    if processed_count == 0:
        logger.error("❌ 未处理任何书籍")
        return False
    
    # 加载构建好的知识库
    loaded_count = kb.load_embedded()
    
    # 统计信息
    logger.info("\n" + "=" * 80)
    logger.info("📊 处理统计")
    logger.info("=" * 80)
    logger.info(f"✅ 处理书籍数量: {processed_count}")
    logger.info(f"✅ 加载书籍数量: {loaded_count}")
    
    # 类别统计
    categories = kb.get_categories()
    logger.info(f"📋 识别到的类别: {categories}")
    
    for category in categories:
        category_docs = kb.get_documents_by_category(category)
        logger.info(f"  - {category}: {len(category_docs)}本")
    
    # 关键词统计
    top_keywords = kb.get_top_keywords(limit=15)
    logger.info("\n🔥 热门关键词:")
    for keyword, count in top_keywords.items():
        logger.info(f"  - {keyword}: {count}次")
    
    # 知识图谱大小
    graph = kb.build_knowledge_graph()
    logger.info(f"\n🔗 知识图谱统计:")
    logger.info(f"  - 节点数量: {len(graph['nodes'])}")
    logger.info(f"  - 边数量: {len(graph['edges'])}")
    
    # 测试搜索功能
    test_queries = ["投资", "经济学", "风险控制", "技术分析"]
    logger.info("\n" + "=" * 80)
    logger.info("🔍 搜索测试")
    logger.info("=" * 80)
    
    for query in test_queries:
        results = kb.search(query, limit=3)
        logger.info(f"\n📝 查询: '{query}'")
        if results:
            for i, doc in enumerate(results, 1):
                logger.info(f"  {i}. 《{doc.title}》 (类别: {doc.category})")
        else:
            logger.info("  ❌ 无结果")
    
    # 测试相关文档推荐
    if kb.size() > 0:
        logger.info("\n" + "=" * 80)
        logger.info("🤝 相关文档推荐测试")
        logger.info("=" * 80)
        
        # 测试第一本书的相关推荐
        related = kb.get_related_documents(0, limit=3)
        if related:
            logger.info(f"\n📚 与《{kb.documents[0].title}》相关的书籍:")
            for i, (score, doc) in enumerate(related, 1):
                logger.info(f"  {i}. 《{doc.title}》 (相似度: {score:.2f})")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    logger.info("\n" + "=" * 80)
    logger.info("🎉 书籍处理完成")
    logger.info("=" * 80)
    logger.info(f"⏱️  耗时: {elapsed_time:.2f}秒")
    logger.info(f"📦 生成的知识库文件: app/rox_quant/assets/embedded_kb.json")
    
    return True

def main():
    """主函数"""
    try:
        success = process_books()
        if success:
            logger.info("\n✅ 书籍处理系统运行成功！")
            return 0
        else:
            logger.error("\n❌ 书籍处理系统运行失败！")
            return 1
    except Exception as e:
        logger.error(f"\n💥 运行时错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())