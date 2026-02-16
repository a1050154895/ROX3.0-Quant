#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试知识中心API
"""

import requests
import json

def test_api_endpoints():
    """测试所有API端点"""
    base_url = "http://localhost:8002/api/knowledge"
    
    print("=" * 60)
    print("测试知识中心API端点")
    print("=" * 60)
    
    # 测试1: 获取统计信息
    print("\n1. 测试统计信息API:")
    try:
        response = requests.get(f"{base_url}/stats")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 成功获取统计信息:")
            print(f"   - 总文档数: {data.get('total_documents', 0)}")
            print(f"   - 分类: {data.get('categories', [])}")
        else:
            print(f"❌ 失败: {response.status_code}")
    except Exception as e:
        print(f"❌ 错误: {e}")
    
    # 测试2: 搜索功能
    print("\n2. 测试搜索API:")
    try:
        response = requests.get(f"{base_url}/search?query=投资")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 成功搜索，返回 {len(data)} 条结果")
            for i, item in enumerate(data[:3]):
                print(f"   {i+1}. {item.get('title', '无标题')}")
        else:
            print(f"❌ 失败: {response.status_code}")
    except Exception as e:
        print(f"❌ 错误: {e}")
    
    # 测试3: 按类别获取文档
    print("\n3. 测试类别API:")
    try:
        response = requests.get(f"{base_url}/categories")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 成功获取类别:")
            for category, docs in data.items():
                print(f"   - {category}: {len(docs)}本")
        else:
            print(f"❌ 失败: {response.status_code}")
    except Exception as e:
        print(f"❌ 错误: {e}")
    
    # 测试4: 知识图谱
    print("\n4. 测试知识图谱API:")
    try:
        response = requests.get(f"{base_url}/graph")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 成功获取知识图谱:")
            print(f"   - 节点数: {len(data.get('nodes', []))}")
            print(f"   - 边数: {len(data.get('edges', []))}")
        else:
            print(f"❌ 失败: {response.status_code}")
    except Exception as e:
        print(f"❌ 错误: {e}")
    
    print("\n" + "=" * 60)
    print("API测试完成")
    print("=" * 60)

def test_frontend():
    """测试前端页面"""
    frontend_url = "http://localhost:8002/knowledge"
    
    print("\n" + "=" * 60)
    print("测试前端页面")
    print("=" * 60)
    
    try:
        response = requests.get(frontend_url)
        if response.status_code == 200:
            print(f"✅ 前端页面可访问")
            print(f"   - 页面大小: {len(response.text)} 字节")
            print(f"   - 访问地址: {frontend_url}")
        else:
            print(f"❌ 前端页面访问失败: {response.status_code}")
    except Exception as e:
        print(f"❌ 错误: {e}")

if __name__ == "__main__":
    test_api_endpoints()
    test_frontend()