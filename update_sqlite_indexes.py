#!/usr/bin/env python3
"""
执行SQLite数据库索引更新
"""
import sqlite3
import os
import logging

from app.core.config import settings

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def execute_sql_file(db_path, sql_file):
    """
    执行SQL文件中的所有语句
    """
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # 读取SQL文件
            with open(sql_file, 'r', encoding='utf-8') as f:
                sql_script = f.read()
            
            # 分割SQL语句并执行
            statements = sql_script.split(';')
            
            for statement in statements:
                statement = statement.strip()
                if statement:
                    try:
                        if statement.startswith('SELECT'):
                            # 对于查询语句，执行并打印结果
                            cursor.execute(statement)
                            results = cursor.fetchall()
                            logger.info(f"执行查询: {statement}")
                            for row in results:
                                logger.info(f"  结果: {row}")
                        else:
                            # 对于其他语句，直接执行
                            cursor.execute(statement)
                            logger.info(f"执行语句: {statement}")
                    except sqlite3.Error as e:
                        logger.warning(f"执行语句失败: {statement}\n错误: {e}")
            
            conn.commit()
            logger.info("索引更新完成")
            
    except Exception as e:
        logger.error(f"执行SQL文件失败: {e}")
        raise

if __name__ == "__main__":
    db_path = settings.DB_PATH
    sql_file = "update_sqlite_indexes.sql"
    
    if not os.path.exists(sql_file):
        logger.error(f"SQL文件不存在: {sql_file}")
        exit(1)
    
    logger.info(f"开始更新SQLite数据库索引")
    logger.info(f"数据库路径: {db_path}")
    logger.info(f"SQL文件: {sql_file}")
    
    execute_sql_file(db_path, sql_file)
    logger.info("索引更新完成")