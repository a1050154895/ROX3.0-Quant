#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
向量数据库管理系统
用于管理书籍和文档的向量嵌入数据
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VectorDatabaseManager:
    """向量数据库管理器"""
    
    def __init__(self, db_path: Optional[Path] = None):
        """初始化向量数据库管理器"""
        if db_path:
            self.db_path = db_path
        else:
            # 默认路径
            self.db_path = Path(__file__).parent / "assets" / "embedded_kb.json"
        
        self.db_dir = self.db_path.parent
        self.db_dir.mkdir(parents=True, exist_ok=True)
        
        self._cache = None
        self._last_modified = 0
    
    def get_db_path(self) -> Path:
        """获取数据库路径"""
        return self.db_path
    
    def exists(self) -> bool:
        """检查数据库是否存在"""
        return self.db_path.exists()
    
    def load(self) -> Dict[str, Any]:
        """加载向量数据库"""
        if not self.exists():
            logger.warning("向量数据库不存在，返回空数据")
            return {"documents": []}
        
        # 检查文件是否被修改
        current_modified = os.path.getmtime(self.db_path)
        if self._cache and self._last_modified == current_modified:
            return self._cache
        
        try:
            with open(self.db_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 标准化数据格式
            if isinstance(data, list):
                # 旧格式（直接是文档列表）
                data = {"documents": data}
            
            # 更新缓存
            self._cache = data
            self._last_modified = current_modified
            
            logger.info(f"✅ 加载向量数据库，包含 {len(data.get('documents', []))} 个文档")
            return data
        except Exception as e:
            logger.error(f"❌ 加载向量数据库失败: {e}")
            return {"documents": []}
    
    def save(self, data: Dict[str, Any]) -> bool:
        """保存向量数据库"""
        try:
            with open(self.db_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            # 更新缓存
            self._cache = data
            self._last_modified = os.path.getmtime(self.db_path)
            
            logger.info(f"✅ 保存向量数据库，包含 {len(data.get('documents', []))} 个文档")
            return True
        except Exception as e:
            logger.error(f"❌ 保存向量数据库失败: {e}")
            return False
    
    def add_document(self, document: Dict[str, Any]) -> bool:
        """添加单个文档"""
        data = self.load()
        documents = data.get('documents', [])
        
        # 检查是否已存在
        existing_paths = [doc.get('path') for doc in documents if doc.get('path')]
        if document.get('path') in existing_paths:
            logger.warning(f"文档已存在: {document.get('path')}")
            return False
        
        documents.append(document)
        data['documents'] = documents
        
        return self.save(data)
    
    def update_document(self, document_path: str, updated_document: Dict[str, Any]) -> bool:
        """更新文档"""
        data = self.load()
        documents = data.get('documents', [])
        
        updated = False
        for i, doc in enumerate(documents):
            if doc.get('path') == document_path:
                documents[i] = updated_document
                updated = True
                break
        
        if not updated:
            logger.warning(f"文档不存在: {document_path}")
            return False
        
        data['documents'] = documents
        return self.save(data)
    
    def delete_document(self, document_path: str) -> bool:
        """删除文档"""
        data = self.load()
        documents = data.get('documents', [])
        
        original_count = len(documents)
        documents = [doc for doc in documents if doc.get('path') != document_path]
        
        if len(documents) == original_count:
            logger.warning(f"文档不存在: {document_path}")
            return False
        
        data['documents'] = documents
        return self.save(data)
    
    def get_document(self, document_path: str) -> Optional[Dict[str, Any]]:
        """获取单个文档"""
        data = self.load()
        documents = data.get('documents', [])
        
        for doc in documents:
            if doc.get('path') == document_path:
                return doc
        
        return None
    
    def search_by_category(self, category: str) -> List[Dict[str, Any]]:
        """按类别搜索文档"""
        data = self.load()
        documents = data.get('documents', [])
        
        return [doc for doc in documents if doc.get('category') == category]
    
    def search_by_keyword(self, keyword: str) -> List[Dict[str, Any]]:
        """按关键词搜索文档"""
        data = self.load()
        documents = data.get('documents', [])
        
        results = []
        for doc in documents:
            if keyword.lower() in doc.get('title', '').lower() or \
               keyword.lower() in doc.get('content', '').lower() or \
               any(keyword.lower() in kw.lower() for kw in doc.get('keywords', [])):
                results.append(doc)
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取数据库统计信息"""
        data = self.load()
        documents = data.get('documents', [])
        
        stats = {
            "total_documents": len(documents),
            "categories": {},
            "has_vector": 0,
            "no_vector": 0,
            "file_types": {}
        }
        
        for doc in documents:
            # 类别统计
            category = doc.get('category', 'unknown')
            stats['categories'][category] = stats['categories'].get(category, 0) + 1
            
            # 向量统计
            if doc.get('vector'):
                stats['has_vector'] += 1
            else:
                stats['no_vector'] += 1
            
            # 文件类型统计
            path = doc.get('path', '')
            if path:
                ext = os.path.splitext(path)[1].lower()
                stats['file_types'][ext] = stats['file_types'].get(ext, 0) + 1
        
        return stats
    
    def optimize(self) -> bool:
        """优化向量数据库"""
        try:
            logger.info("开始优化向量数据库...")
            
            # 加载数据
            data = self.load()
            documents = data.get('documents', [])
            
            # 清理无效文档
            valid_documents = []
            invalid_count = 0
            
            for doc in documents:
                if doc.get('title') and doc.get('content'):
                    valid_documents.append(doc)
                else:
                    invalid_count += 1
            
            # 去重
            seen_paths = set()
            unique_documents = []
            duplicate_count = 0
            
            for doc in valid_documents:
                path = doc.get('path', '')
                if path and path in seen_paths:
                    duplicate_count += 1
                    continue
                unique_documents.append(doc)
                if path:
                    seen_paths.add(path)
            
            # 保存优化后的数据
            optimized_data = {"documents": unique_documents}
            success = self.save(optimized_data)
            
            if success:
                logger.info(f"✅ 向量数据库优化完成:")
                logger.info(f"   - 原始文档数: {len(documents)}")
                logger.info(f"   - 无效文档数: {invalid_count}")
                logger.info(f"   - 重复文档数: {duplicate_count}")
                logger.info(f"   - 优化后文档数: {len(unique_documents)}")
            
            return success
        except Exception as e:
            logger.error(f"❌ 优化向量数据库失败: {e}")
            return False
    
    def backup(self, backup_dir: Optional[Path] = None) -> Path:
        """备份向量数据库"""
        if not self.exists():
            logger.warning("向量数据库不存在，无法备份")
            return None
        
        if not backup_dir:
            backup_dir = self.db_dir / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成备份文件名
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        backup_path = backup_dir / f"embedded_kb_backup_{timestamp}.json"
        
        try:
            # 复制文件
            import shutil
            shutil.copy2(self.db_path, backup_path)
            
            logger.info(f"✅ 备份向量数据库到: {backup_path}")
            return backup_path
        except Exception as e:
            logger.error(f"❌ 备份向量数据库失败: {e}")
            return None
    
    def restore(self, backup_path: Path) -> bool:
        """从备份恢复向量数据库"""
        if not backup_path.exists():
            logger.warning(f"备份文件不存在: {backup_path}")
            return False
        
        try:
            # 复制文件
            import shutil
            shutil.copy2(backup_path, self.db_path)
            
            # 清除缓存
            self._cache = None
            self._last_modified = 0
            
            logger.info(f"✅ 从备份恢复向量数据库: {backup_path}")
            return True
        except Exception as e:
            logger.error(f"❌ 恢复向量数据库失败: {e}")
            return False
    
    def rebuild(self, documents_dir: Path) -> bool:
        """重建向量数据库"""
        from app.rox_quant.knowledge_base import KnowledgeBase
        
        try:
            logger.info(f"开始从目录重建向量数据库: {documents_dir}")
            
            kb = KnowledgeBase()
            processed_count = kb.build_embedded_from_dir(str(documents_dir))
            
            if processed_count == 0:
                logger.error("未处理任何文档")
                return False
            
            logger.info(f"✅ 重建向量数据库完成，处理了 {processed_count} 个文档")
            return True
        except Exception as e:
            logger.error(f"❌ 重建向量数据库失败: {e}")
            return False

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='向量数据库管理工具')
    parser.add_argument('--action', choices=['info', 'optimize', 'backup', 'restore', 'rebuild'], 
                      default='info', help='操作类型')
    parser.add_argument('--path', type=Path, help='指定数据库路径')
    parser.add_argument('--backup', type=Path, help='备份文件路径')
    parser.add_argument('--dir', type=Path, help='文档目录路径')
    
    args = parser.parse_args()
    
    # 初始化管理器
    manager = VectorDatabaseManager(args.path)
    
    if args.action == 'info':
        # 显示数据库信息
        stats = manager.get_statistics()
        logger.info("\n" + "=" * 60)
        logger.info("📊 向量数据库统计信息")
        logger.info("=" * 60)
        logger.info(f"总文档数: {stats['total_documents']}")
        logger.info(f"有向量的文档: {stats['has_vector']}")
        logger.info(f"无向量的文档: {stats['no_vector']}")
        logger.info("\n类别分布:")
        for category, count in stats['categories'].items():
            logger.info(f"  - {category}: {count}")
        logger.info("\n文件类型分布:")
        for ext, count in stats['file_types'].items():
            logger.info(f"  - {ext}: {count}")
        
    elif args.action == 'optimize':
        # 优化数据库
        success = manager.optimize()
        if success:
            logger.info("✅ 优化成功！")
        else:
            logger.error("❌ 优化失败！")
        
    elif args.action == 'backup':
        # 备份数据库
        backup_path = manager.backup()
        if backup_path:
            logger.info(f"✅ 备份成功: {backup_path}")
        else:
            logger.error("❌ 备份失败！")
        
    elif args.action == 'restore':
        # 恢复数据库
        if not args.backup:
            logger.error("请指定备份文件路径")
        else:
            success = manager.restore(args.backup)
            if success:
                logger.info("✅ 恢复成功！")
            else:
                logger.error("❌ 恢复失败！")
        
    elif args.action == 'rebuild':
        # 重建数据库
        if not args.dir:
            logger.error("请指定文档目录路径")
        else:
            success = manager.rebuild(args.dir)
            if success:
                logger.info("✅ 重建成功！")
            else:
                logger.error("❌ 重建失败！")

if __name__ == '__main__':
    main()