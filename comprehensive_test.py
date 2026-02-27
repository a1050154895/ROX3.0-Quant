#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROX 3.0 全面功能检测脚本
覆盖所有前端页面、API端点、WebSocket等
"""

import requests
import json
import time
import sys
from datetime import datetime

BASE_URL = "http://127.0.0.1:8008"
TIMEOUT = 15

# 统计
results = {
    "pass": [],
    "fail": [],
    "warn": [],
    "error": [],
}

def check_endpoint(category, name, url, method="GET", expected_status=200, check_json=True, payload=None, params=None):
    """检测单个端点"""
    full_url = f"{BASE_URL}{url}" if not url.startswith("http") else url
    print(f"  [{category}] {name}...", end=" ", flush=True)
    try:
        if method == "GET":
            response = requests.get(full_url, timeout=TIMEOUT, params=params)
        elif method == "POST":
            response = requests.post(full_url, json=payload, timeout=TIMEOUT, params=params)
        elif method == "PUT":
            response = requests.put(full_url, json=payload, timeout=TIMEOUT, params=params)
        elif method == "DELETE":
            response = requests.delete(full_url, timeout=TIMEOUT, params=params)

        status = response.status_code
        
        if status == expected_status:
            if check_json:
                try:
                    data = response.json()
                    if isinstance(data, (dict, list)):
                        size = len(json.dumps(data, ensure_ascii=False))
                        print(f"✅ PASS (HTTP {status}, {size} chars)")
                        results["pass"].append({
                            "category": category,
                            "name": name,
                            "url": url,
                            "status": status,
                            "data_size": size,
                            "data_preview": str(data)[:200]
                        })
                        return True, data
                    else:
                        print(f"⚠️ WARN (非标准JSON类型: {type(data).__name__})")
                        results["warn"].append({
                            "category": category,
                            "name": name,
                            "url": url,
                            "reason": f"非标准JSON类型: {type(data).__name__}"
                        })
                        return False, None
                except json.JSONDecodeError:
                    print(f"❌ FAIL (响应不是JSON)")
                    results["fail"].append({
                        "category": category,
                        "name": name,
                        "url": url,
                        "reason": "响应不是JSON",
                        "response_preview": response.text[:200]
                    })
                    return False, None
            else:
                size = len(response.text)
                print(f"✅ PASS (HTTP {status}, HTML/Text {size} chars)")
                results["pass"].append({
                    "category": category,
                    "name": name,
                    "url": url,
                    "status": status,
                    "data_size": size
                })
                return True, response.text
        elif status == 404:
            print(f"❌ FAIL (404 Not Found)")
            results["fail"].append({
                "category": category,
                "name": name,
                "url": url,
                "reason": f"HTTP 404 Not Found"
            })
            return False, None
        elif status == 422:
            try:
                err = response.json()
                print(f"⚠️ WARN (422 参数验证失败: {str(err)[:100]})")
                results["warn"].append({
                    "category": category,
                    "name": name,
                    "url": url,
                    "reason": f"422 参数验证: {str(err)[:200]}"
                })
            except:
                print(f"⚠️ WARN (HTTP 422)")
                results["warn"].append({
                    "category": category,
                    "name": name,
                    "url": url,
                    "reason": "HTTP 422"
                })
            return False, None
        elif status == 500:
            try:
                err = response.json()
                print(f"❌ FAIL (500 服务器错误: {str(err)[:100]})")
            except:
                print(f"❌ FAIL (HTTP 500: {response.text[:100]})")
            results["fail"].append({
                "category": category,
                "name": name,
                "url": url,
                "reason": f"HTTP 500: {response.text[:200]}"
            })
            return False, None
        else:
            print(f"⚠️ WARN (HTTP {status})")
            results["warn"].append({
                "category": category,
                "name": name,
                "url": url,
                "reason": f"HTTP {status}"
            })
            return False, None

    except requests.exceptions.ConnectionError:
        print(f"❌ ERROR (连接失败)")
        results["error"].append({
            "category": category,
            "name": name,
            "url": url,
            "reason": "连接失败-服务可能未启动"
        })
        return False, None
    except requests.exceptions.Timeout:
        print(f"❌ ERROR (请求超时>{TIMEOUT}s)")
        results["error"].append({
            "category": category,
            "name": name,
            "url": url,
            "reason": f"请求超时>{TIMEOUT}s"
        })
        return False, None
    except Exception as e:
        print(f"❌ ERROR ({str(e)[:80]})")
        results["error"].append({
            "category": category,
            "name": name,
            "url": url,
            "reason": str(e)[:200]
        })
        return False, None


def test_server_health():
    """测试服务器健康状态"""
    print("\n" + "="*60)
    print("🏥 1. 服务器健康检测")
    print("="*60)
    ok, data = check_endpoint("健康检查", "健康端点 /health", "/health")
    if not ok:
        print("\n⛔ 服务器未运行或健康检查失败，后续测试可能无法进行!")
        return False
    
    check_endpoint("健康检查", "API文档 /docs", "/docs", check_json=False)
    return True


def test_frontend_pages():
    """测试所有前端页面"""
    print("\n" + "="*60)
    print("🖥️ 2. 前端页面检测")
    print("="*60)
    
    pages = [
        ("Landing页面", "/"),
        ("ROX 2.0 UI", "/home"),
        ("经典版 1.0", "/classic"),
        ("专业版 3.0", "/pro"),
        ("策略工坊", "/builder"),
        ("市场热力图", "/map"),
        ("知识中心", "/knowledge"),
        ("策略中心", "/strategies"),
    ]
    
    for name, url in pages:
        check_endpoint("前端页面", name, url, check_json=False)


def test_market_data_apis():
    """测试市场数据API"""
    print("\n" + "="*60)
    print("📊 3. 市场数据 API 检测")
    print("="*60)
    
    # 市场指数
    check_endpoint("市场数据", "指数行情 /api/market/indices", "/api/market/indices")
    
    # K线数据
    check_endpoint("市场数据", "K线-上证指数", "/api/market/kline", params={"code": "sh000001"})
    check_endpoint("市场数据", "K线-贵州茅台", "/api/market/kline", params={"code": "sh600519"})
    
    # 分时数据
    check_endpoint("市场数据", "分时-上证指数", "/api/market/fenshi", params={"code": "sh000001"})
    
    # 龙虎榜
    check_endpoint("市场数据", "龙虎榜", "/api/market/dragon-tiger")
    
    # 板块/行业数据
    check_endpoint("市场数据", "板块轮动", "/api/market/rotation")
    check_endpoint("市场数据", "热门板块", "/api/market/hot-sectors")
    check_endpoint("市场数据", "资金流向", "/api/market/capital-flow")
    
    # 涨跌停
    check_endpoint("市场数据", "涨停板", "/api/market/limit-up")
    check_endpoint("市场数据", "跌停板", "/api/market/limit-down")
    
    # A股市场概览
    check_endpoint("市场数据", "A股实时行情", "/api/market/ashare/realtime")
    check_endpoint("市场数据", "A股搜索", "/api/market/ashare/search", params={"q": "茅台"})
    
    # 市场分析
    check_endpoint("市场数据", "市场分析概览", "/api/market/analysis/overview")
    check_endpoint("市场数据", "市场情绪", "/api/market/analysis/sentiment")


def test_stock_apis():
    """测试个股相关API"""
    print("\n" + "="*60)
    print("🔍 4. 个股诊断 API 检测")
    print("="*60)
    
    # 个股诊断
    check_endpoint("个股诊断", "诊断-贵州茅台600519", "/api/stock/diagnose", params={"code": "600519"})
    check_endpoint("个股诊断", "诊断-中国平安601318", "/api/stock/diagnose", params={"code": "601318"})
    
    # 智能选股
    check_endpoint("个股", "智能选股", "/api/stock/smart-pick")
    check_endpoint("个股", "每周金股", "/api/stock/weekly-gold")


def test_analysis_apis():
    """测试分析相关API"""
    print("\n" + "="*60)
    print("📈 5. 分析 API 检测")
    print("="*60)
    
    check_endpoint("分析", "AI分析面板-600519", "/api/analysis/dashboard/600519")
    check_endpoint("分析", "技术分析-600519", "/api/analysis/technical/600519")
    check_endpoint("分析", "基本面分析-600519", "/api/analysis/fundamental/600519")


def test_macro_apis():
    """测试宏观数据API"""
    print("\n" + "="*60)
    print("🌍 6. 宏观数据 API 检测")
    print("="*60)
    
    check_endpoint("宏观数据", "宏观指标(Phase6)", "/api/macro/indicators")
    check_endpoint("宏观数据", "宏观数据(Legacy)", "/api/market/macro")
    check_endpoint("宏观数据", "GDP", "/api/macro/gdp")
    check_endpoint("宏观数据", "CPI", "/api/macro/cpi")
    check_endpoint("宏观数据", "货币供应M2", "/api/macro/money-supply")


def test_ai_apis():
    """测试AI相关API"""
    print("\n" + "="*60)
    print("🤖 7. AI 功能 API 检测")
    print("="*60)
    
    # AI Chat
    check_endpoint("AI", "AI聊天", "/api/ai/chat", method="POST", payload={
        "message": "你好，请简单介绍一下贵州茅台",
        "model": "deepseek-chat"
    })
    
    # AI 市场简报
    check_endpoint("AI", "AI市场简报", "/api/ai/market-brief")
    
    # AI models
    check_endpoint("AI", "AI模型列表", "/api/ai/models")


def test_strategy_apis():
    """测试策略相关API"""
    print("\n" + "="*60)
    print("⚙️ 8. 策略 API 检测")
    print("="*60)
    
    check_endpoint("策略", "策略列表", "/api/strategy/list")
    check_endpoint("策略", "策略模板", "/api/strategy/templates")
    
    # 策略市场
    check_endpoint("策略市场", "策略市场列表", "/api/marketplace/strategies")
    check_endpoint("策略市场", "策略分类", "/api/marketplace/categories")
    
    # 聚宽策略
    check_endpoint("策略", "聚宽策略列表", "/api/strategies/jq/list")


def test_backtest_apis():
    """测试回测相关API"""
    print("\n" + "="*60)
    print("🔄 9. 回测 API 检测")
    print("="*60)
    
    check_endpoint("回测", "回测历史", "/api/backtest/history")


def test_portfolio_apis():
    """测试组合相关API"""
    print("\n" + "="*60)
    print("💼 10. 投资组合 API 检测")
    print("="*60)
    
    check_endpoint("投资组合", "组合总览(模拟)", "/api/portfolio/summary", params={"mode": "sim"})
    check_endpoint("投资组合", "组合总览(实盘)", "/api/portfolio/summary", params={"mode": "real"})
    check_endpoint("投资组合", "持仓列表(模拟)", "/api/portfolio/positions", params={"mode": "sim"})


def test_knowledge_apis():
    """测试知识库API"""
    print("\n" + "="*60)
    print("📚 11. 知识库 API 检测")
    print("="*60)
    
    check_endpoint("知识库", "知识库搜索", "/api/kb/search", params={"q": "均线"})
    check_endpoint("知识库", "知识库文档列表", "/api/kb/documents")
    check_endpoint("知识库", "知识中心分类", "/api/knowledge/categories")
    check_endpoint("知识库", "知识中心热门", "/api/knowledge/popular")


def test_system_apis():
    """测试系统API"""
    print("\n" + "="*60)
    print("🔧 12. 系统 API 检测")
    print("="*60)
    
    check_endpoint("系统", "系统状态", "/api/system/status")
    check_endpoint("系统", "系统信息", "/api/system/info")
    
    # 设置
    check_endpoint("设置", "AI设置", "/api/settings/ai")
    check_endpoint("设置", "数据源状态", "/api/settings/datasource")


def test_trade_apis():
    """测试交易API"""
    print("\n" + "="*60)
    print("💰 13. 交易 API 检测")
    print("="*60)
    
    check_endpoint("交易", "订单列表", "/api/trade/orders")
    check_endpoint("交易", "交易信号", "/api/trade/signals")


def test_philosophy_apis():
    """测试哲学方法论API"""
    print("\n" + "="*60)
    print("🧠 14. 哲学方法论 API 检测")
    print("="*60)
    
    check_endpoint("哲学", "矛盾分析-600519", "/api/philosophy/contradiction-analysis/600519")
    check_endpoint("哲学", "价值规律-600519", "/api/philosophy/value-law/600519")


def test_professional_apis():
    """测试专业系统API"""
    print("\n" + "="*60)
    print("🏢 15. 专业系统 API 检测")
    print("="*60)
    
    check_endpoint("专业系统", "Level2数据", "/api/professional/level2", params={"code": "600519"})
    check_endpoint("专业系统", "资金雷达", "/api/professional/fund-radar", params={"code": "600519"})


def test_eastern_wisdom_apis():
    """测试东方智慧量化API"""
    print("\n" + "="*60)
    print("☯️ 16. 东方智慧量化 API 检测")
    print("="*60)
    
    check_endpoint("东方智慧", "易经分析-600519", "/api/eastern-wisdom/yijing/600519")
    check_endpoint("东方智慧", "天干地支时序-600519", "/api/eastern-wisdom/stems-branches/600519")


def test_ml_apis():
    """测试机器学习API"""
    print("\n" + "="*60)
    print("🧪 17. 机器学习 API 检测")
    print("="*60)
    
    check_endpoint("机器学习", "ML预测-600519", "/api/ml/predict/600519")
    check_endpoint("机器学习", "ML模型列表", "/api/ml/models")


def test_alert_apis():
    """测试预警API"""
    print("\n" + "="*60)
    print("🔔 18. 价格预警 API 检测")
    print("="*60)
    
    check_endpoint("预警", "预警列表", "/api/alerts/list")


def test_export_apis():
    """测试导出API"""
    print("\n" + "="*60)
    print("📤 19. 数据导出 API 检测")
    print("="*60)
    
    check_endpoint("导出", "导出格式列表", "/api/export/formats")


def test_info_apis():
    """测试市场资讯API"""
    print("\n" + "="*60)
    print("📰 20. 市场资讯 API 检测")
    print("="*60)
    
    check_endpoint("资讯", "市场快讯", "/api/info/news")
    check_endpoint("资讯", "研报", "/api/info/reports")


def test_lu_prediction_apis():
    """测试卢麒元预测API"""
    print("\n" + "="*60)
    print("📐 21. 卢麒元预测系统 API 检测")
    print("="*60)
    
    check_endpoint("卢麒元预测", "货币分析", "/api/lu-prediction/monetary-analysis")
    check_endpoint("卢麒元预测", "财政预测", "/api/lu-prediction/fiscal-prediction")


def test_agents_apis():
    """测试多智能体API"""
    print("\n" + "="*60)
    print("🤝 22. 多智能体 API 检测")
    print("="*60)
    
    check_endpoint("多智能体", "智能体列表", "/api/agents/list")


def print_summary():
    """打印检测摘要"""
    print("\n" + "="*80)
    print("📋 ROX 3.0 全面功能检测报告")
    print("="*80)
    print(f"检测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"服务器地址: {BASE_URL}")
    
    total = len(results["pass"]) + len(results["fail"]) + len(results["warn"]) + len(results["error"])
    
    print(f"\n📊 总计检测: {total} 项")
    print(f"   ✅ 通过: {len(results['pass'])} 项")
    print(f"   ❌ 失败: {len(results['fail'])} 项")
    print(f"   ⚠️ 警告: {len(results['warn'])} 项")
    print(f"   🔴 错误: {len(results['error'])} 项")
    
    pass_rate = len(results["pass"]) / total * 100 if total > 0 else 0
    print(f"   🎯 通过率: {pass_rate:.1f}%")
    
    if results["fail"]:
        print(f"\n{'='*60}")
        print("❌ 失败项详情:")
        print(f"{'='*60}")
        for i, item in enumerate(results["fail"], 1):
            print(f"  {i}. [{item['category']}] {item['name']}")
            print(f"     URL: {item['url']}")
            print(f"     原因: {item['reason'][:150]}")
    
    if results["error"]:
        print(f"\n{'='*60}")
        print("🔴 错误项详情:")
        print(f"{'='*60}")
        for i, item in enumerate(results["error"], 1):
            print(f"  {i}. [{item['category']}] {item['name']}")
            print(f"     URL: {item['url']}")
            print(f"     原因: {item['reason'][:150]}")
    
    if results["warn"]:
        print(f"\n{'='*60}")
        print("⚠️ 警告项详情:")
        print(f"{'='*60}")
        for i, item in enumerate(results["warn"], 1):
            print(f"  {i}. [{item['category']}] {item['name']}")
            print(f"     URL: {item['url']}")
            print(f"     原因: {item['reason'][:150]}")
    
    # 按分类统计
    print(f"\n{'='*60}")
    print("📊 按分类统计:")
    print(f"{'='*60}")
    
    categories = {}
    for status_type in ["pass", "fail", "warn", "error"]:
        for item in results[status_type]:
            cat = item["category"]
            if cat not in categories:
                categories[cat] = {"pass": 0, "fail": 0, "warn": 0, "error": 0}
            categories[cat][status_type] += 1
    
    for cat, stats in sorted(categories.items()):
        total_cat = sum(stats.values())
        cat_rate = stats["pass"] / total_cat * 100 if total_cat > 0 else 0
        status_icon = "✅" if cat_rate == 100 else ("⚠️" if cat_rate >= 50 else "❌")
        print(f"  {status_icon} {cat}: {stats['pass']}/{total_cat} 通过 ({cat_rate:.0f}%)")
    
    return results


if __name__ == "__main__":
    print("🚀 ROX 3.0 全面功能检测开始")
    print(f"   目标: {BASE_URL}")
    print(f"   时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. 健康检查
    server_ok = test_server_health()
    
    if not server_ok:
        print("\n⛔ 服务器未启动，请先启动服务器再运行测试!")
        print("   启动命令: python3 -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8008")
        sys.exit(1)
    
    # 2. 前端页面
    test_frontend_pages()
    
    # 3. 市场数据
    test_market_data_apis()
    
    # 4. 个股诊断
    test_stock_apis()
    
    # 5. 分析API
    test_analysis_apis()
    
    # 6. 宏观数据
    test_macro_apis()
    
    # 7. AI功能
    test_ai_apis()
    
    # 8. 策略
    test_strategy_apis()
    
    # 9. 回测
    test_backtest_apis()
    
    # 10. 投资组合
    test_portfolio_apis()
    
    # 11. 知识库
    test_knowledge_apis()
    
    # 12. 系统
    test_system_apis()
    
    # 13. 交易
    test_trade_apis()
    
    # 14. 哲学方法论
    test_philosophy_apis()
    
    # 15. 专业系统
    test_professional_apis()
    
    # 16. 东方智慧
    test_eastern_wisdom_apis()
    
    # 17. 机器学习
    test_ml_apis()
    
    # 18. 预警
    test_alert_apis()
    
    # 19. 导出
    test_export_apis()
    
    # 20. 资讯
    test_info_apis()
    
    # 21. 卢麒元预测
    test_lu_prediction_apis()
    
    # 22. 多智能体
    test_agents_apis()
    
    # 打印摘要
    final_results = print_summary()
    
    # 保存结果到JSON
    with open("test_results.json", "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    print(f"\n📁 详细结果已保存到: test_results.json")
