#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试知识库搜索功能
"""

import requests

def test_search():
    """测试知识库搜索"""
    base_url = "http://localhost:8002/api/knowledge"
    
    print("=" * 60)
    print("测试知识库搜索功能")
    print("=" * 60)
    
    queries = [
        "Python量化",
        "量化投资",
        "技术分析",
        "投资策略"
    ]
    
    for query in queries:
        print(f"\n🔍 搜索: '{query}'")
        try:
            response = requests.get(f"{base_url}/search", params={"query": query, "limit": 3})
            if response.status_code == 200:
                results = response.json()
                print(f"✅ 找到 {len(results)} 个结果")
                for i, result in enumerate(results):
                    title = result.get('title', '无标题')
                    category = result.get('category', '无分类')
                    print(f"   {i+1}. {title} ({category})")
            else:
                print(f"❌ 搜索失败: {response.status_code}")
        except Exception as e:
            print(f"❌ 错误: {e}")
    
    print("\n" + "=" * 60)
    print("搜索测试完成")
    print("=" * 60)

if __name__ == "__main__":
    test_search()