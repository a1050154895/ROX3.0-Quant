#!/usr/bin/env python3
"""
ROX 3.0 Quant 快速测试脚本
验证系统核心功能是否正常运行
"""

import sys
import json
import requests
from datetime import datetime

BASE_URL = "http://127.0.0.1:8099"

def print_header(text):
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)

def test_health():
    """测试健康检查"""
    print_header("测试 1: 系统健康检查")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 系统状态: {data.get('status', 'unknown')}")
            print(f"   时间: {data.get('timestamp', 'N/A')}")
            return True
        else:
            print(f"❌ 健康检查失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 无法连接服务器: {e}")
        return False

def test_market_data():
    """测试市场数据API"""
    print_header("测试 2: 市场数据获取")
    try:
        print("获取主要指数...")
        response = requests.get(f"{BASE_URL}/api/market/indices", timeout=10)
        if response.status_code == 200:
            data = response.json()
            indices = data.get('indices', [])
            print(f"✅ 成功获取 {len(indices)} 个指数")
            if indices:
                print(f"   示例: {indices[0].get('name', 'N/A')} - {indices[0].get('price', 'N/A')}")
            return True
        else:
            print(f"❌ 获取指数失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 获取市场数据失败: {e}")
        return False

def test_realtime_quotes():
    """测试实时行情"""
    print_header("测试 3: 实时行情")
    try:
        print("获取实时行情...")
        response = requests.get(f"{BASE_URL}/api/market/spot", timeout=10)
        if response.status_code == 200:
            data = response.json()
            items = data.get('items', [])
            print(f"✅ 成功获取 {len(items)} 条行情数据")
            if items:
                print(f"   示例: {items[0].get('name', 'N/A')} - {items[0].get('price', 'N/A')}")
            return True
        else:
            print(f"❌ 获取实时行情失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 获取实时行情失败: {e}")
        return False

def test_lu_analysis():
    """测试卢式分析"""
    print_header("测试 4: 卢式分析功能")
    try:
        print("获取三流数据...")
        response = requests.get(f"{BASE_URL}/api/lu/three-flows", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 卢式三流分析服务正常")
            print(f"   北向资金: {data.get('north_money', 'N/A')}")
            print(f"   成交额: {data.get('total_volume', 'N/A')}")
            print(f"   上涨比: {data.get('rise_ratio', 'N/A')}")
            return True
        else:
            print(f"❌ 卢式分析失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 卢式分析失败: {e}")
        return False

def test_stock_analysis():
    """测试股票诊断"""
    print_header("测试 5: 个股诊断（贵州茅台）")
    try:
        print("分析股票 600519...")
        response = requests.get(
            f"{BASE_URL}/api/lu/analyze-symbol",
            params={"symbol": "600519"},
            timeout=15
        )
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 个股诊断服务正常")
            print(f"   股票代码: {data.get('code', 'N/A')}")
            print(f"   股票名称: {data.get('name', 'N/A')}")
            return True
        else:
            print(f"❌ 个股诊断失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 个股诊断失败: {e}")
        return False

def test_trading_simulation():
    """测试交易模拟"""
    print_header("测试 6: 交易模拟引擎")
    try:
        print("获取交易模拟状态...")
        response = requests.get(f"{BASE_URL}/api/trading-simulation/trading-simulation/status", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 交易模拟引擎正常")
            print(f"   状态: {data.get('status', 'unknown')}")
            return True
        else:
            print(f"❌ 交易模拟查询失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 交易模拟服务失败: {e}")
        return False

def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("  ROX 3.0 Quant 系统功能测试")
    print(f"  测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    tests = [
        ("系统健康检查", test_health),
        ("市场数据获取", test_market_data),
        ("实时行情", test_realtime_quotes),
        ("卢式分析", test_lu_analysis),
        ("个股诊断", test_stock_analysis),
        ("交易模拟", test_trading_simulation),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ 测试 '{name}' 异常: {e}")
            results.append((name, False))

    # 打印测试摘要
    print_header("测试结果摘要")
    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status} - {name}")

    print(f"\n总计: {passed}/{total} 项测试通过")

    if passed == total:
        print("\n🎉 所有测试通过！系统运行正常！")
        print(f"\n请访问: {BASE_URL}")
        print(f"API文档: {BASE_URL}/docs")
        return 0
    else:
        print(f"\n⚠️  有 {total - passed} 项测试失败，请检查系统配置")
        return 1

if __name__ == "__main__":
    sys.exit(main())
