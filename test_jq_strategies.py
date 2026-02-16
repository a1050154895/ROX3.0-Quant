#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试聚宽策略API
"""

import requests
import json

def test_jq_strategies_api():
    """测试聚宽策略API端点"""
    base_url = "http://localhost:8002/api/jq_strategies"
    
    print("=" * 60)
    print("测试聚宽策略API端点")
    print("=" * 60)
    
    # 测试1: 获取策略统计信息
    print("\n1. 测试策略统计API:")
    try:
        response = requests.get(f"{base_url}/stats")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 成功获取统计信息:")
            print(f"   - 总策略数: {data.get('total_strategies', 0)}")
            print(f"   - 总大小: {data.get('total_size', 0)} 字节")
            print(f"   - 平均大小: {data.get('average_size', 0):.2f} 字节")
            print(f"   - Python文件数: {data.get('python_files', 0)}")
        else:
            print(f"❌ 失败: {response.status_code}")
    except Exception as e:
        print(f"❌ 错误: {e}")
    
    # 测试2: 获取策略分类
    print("\n2. 测试策略分类API:")
    try:
        response = requests.get(f"{base_url}/categories")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 成功获取分类:")
            print(f"   - 总策略数: {data.get('total', 0)}")
            print(f"   - 分类:")
            for category, count in data.get('categories', {}).items():
                print(f"     * {category}: {count}个")
        else:
            print(f"❌ 失败: {response.status_code}")
    except Exception as e:
        print(f"❌ 错误: {e}")
    
    # 测试3: 获取策略列表
    print("\n3. 测试策略列表API:")
    try:
        response = requests.get(f"{base_url}/list")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 成功获取策略列表: {len(data)}个策略")
            for i, strategy in enumerate(data[:5]):
                print(f"   {i+1}. {strategy.get('name', '无名称')} ({strategy.get('size', 0)} 字节)")
        else:
            print(f"❌ 失败: {response.status_code}")
    except Exception as e:
        print(f"❌ 错误: {e}")
    
    # 测试4: 获取策略详情
    print("\n4. 测试策略详情API:")
    try:
        response = requests.get(f"{base_url}/list")
        if response.status_code == 200:
            strategies = response.json()
            if strategies:
                strategy_name = strategies[0]['name']
                response = requests.get(f"{base_url}/info/{strategy_name}")
                if response.status_code == 200:
                    data = response.json()
                    print(f"✅ 成功获取策略详情:")
                    print(f"   - 名称: {data.get('name', '无')}")
                    print(f"   - 大小: {data.get('size', 0)} 字节")
                    print(f"   - 行数: {data.get('lines', 0)}")
                    print(f"   - 描述: {data.get('description', '无')[:50]}...")
                else:
                    print(f"❌ 失败: {response.status_code}")
        else:
            print(f"❌ 无法获取策略列表")
    except Exception as e:
        print(f"❌ 错误: {e}")
    
    # 测试5: 执行策略
    print("\n5. 测试策略执行API:")
    try:
        response = requests.get(f"{base_url}/list")
        if response.status_code == 200:
            strategies = response.json()
            if strategies:
                strategy_name = strategies[0]['name']
                payload = {
                    "strategy_name": strategy_name,
                    "params": {}
                }
                response = requests.post(f"{base_url}/execute", json=payload)
                if response.status_code == 200:
                    data = response.json()
                    print(f"✅ 策略执行请求已发送:")
                    print(f"   - 成功: {data.get('success', False)}")
                    print(f"   - 消息: {data.get('message', '无')[:100]}...")
                else:
                    print(f"❌ 失败: {response.status_code}")
        else:
            print(f"❌ 无法获取策略列表")
    except Exception as e:
        print(f"❌ 错误: {e}")
    
    print("\n" + "=" * 60)
    print("API测试完成")
    print("=" * 60)

def test_frontend():
    """测试前端页面"""
    frontend_url = "http://localhost:8002/strategies"
    
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
    test_jq_strategies_api()
    test_frontend()