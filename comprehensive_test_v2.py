#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROX 3.0 全面功能检测脚本 v2
基于 OpenAPI 规范中的实际路由进行测试
"""

import requests
import json
import time
import sys
from datetime import datetime

BASE_URL = "http://127.0.0.1:8008"
TIMEOUT = 15

results = {"pass": [], "fail": [], "warn": [], "error": []}

def check(category, name, url, method="GET", expected_status=200, check_json=True, payload=None, params=None):
    full_url = f"{BASE_URL}{url}"
    print(f"  [{category}] {name}...", end=" ", flush=True)
    try:
        kw = {"timeout": TIMEOUT, "params": params}
        if method == "GET":
            resp = requests.get(full_url, **kw)
        elif method == "POST":
            resp = requests.post(full_url, json=payload, **kw)
        elif method == "DELETE":
            resp = requests.delete(full_url, **kw)
        else:
            resp = requests.request(method, full_url, json=payload, **kw)

        sc = resp.status_code
        entry = {"category": category, "name": name, "url": url, "status": sc}

        if sc == expected_status:
            if check_json:
                try:
                    data = resp.json()
                    sz = len(json.dumps(data, ensure_ascii=False))
                    print(f"✅ PASS ({sc}, {sz} chars)")
                    entry["data_size"] = sz
                    entry["preview"] = str(data)[:150]
                    results["pass"].append(entry)
                    return True, data
                except:
                    print(f"❌ FAIL (非JSON)")
                    entry["reason"] = "非JSON响应"
                    results["fail"].append(entry)
                    return False, None
            else:
                sz = len(resp.text)
                print(f"✅ PASS ({sc}, {sz} chars)")
                entry["data_size"] = sz
                results["pass"].append(entry)
                return True, resp.text
        else:
            reason = f"HTTP {sc}"
            try:
                reason += f": {str(resp.json())[:100]}"
            except:
                reason += f": {resp.text[:100]}"
            print(f"{'❌ FAIL' if sc in (404,500) else '⚠️ WARN'} ({reason[:80]})")
            entry["reason"] = reason
            if sc in (404, 500):
                results["fail"].append(entry)
            else:
                results["warn"].append(entry)
            return False, None
    except requests.exceptions.Timeout:
        print(f"🔴 TIMEOUT (>{TIMEOUT}s)")
        results["error"].append({"category": category, "name": name, "url": url, "reason": f"Timeout>{TIMEOUT}s"})
        return False, None
    except Exception as e:
        print(f"🔴 ERROR ({str(e)[:60]})")
        results["error"].append({"category": category, "name": name, "url": url, "reason": str(e)[:200]})
        return False, None


print("🚀 ROX 3.0 全面功能检测 v2")
print(f"   目标: {BASE_URL}")
print(f"   时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# =====================================
# 1. 健康检查
# =====================================
print("\n" + "="*60)
print("🏥 1. 服务器健康检查")
print("="*60)
ok, _ = check("健康检查", "GET /health", "/health")
if not ok:
    print("⛔ 服务器未响应!")
    sys.exit(1)
check("健康检查", "GET /docs", "/docs", check_json=False)
check("健康检查", "GET /api/system/status", "/api/system/status")
check("健康检查", "GET /api/system/health", "/api/system/health")
check("健康检查", "GET /api/system/ready", "/api/system/ready")

# =====================================
# 2. 前端页面
# =====================================
print("\n" + "="*60)
print("🖥️ 2. 前端页面检测 (8个)")
print("="*60)
for name, url in [
    ("Landing / 落地页", "/"),
    ("ROX 2.0 UI /home", "/home"),
    ("经典版 1.0 /classic", "/classic"),
    ("专业版 3.0 /pro", "/pro"),
    ("策略工坊 /builder", "/builder"),
    ("市场热力图 /map", "/map"),
    ("知识中心 /knowledge", "/knowledge"),
    ("策略中心 /strategies", "/strategies"),
]:
    check("前端页面", name, url, check_json=False)

# =====================================
# 3. 市场数据 API
# =====================================
print("\n" + "="*60)
print("📊 3. 市场数据 API (18个)")
print("="*60)
check("市场数据", "指数行情", "/api/market/indices")
check("市场数据", "K线(上证)", "/api/market/kline", params={"code": "sh000001"})
check("市场数据", "K线(茅台)", "/api/market/kline", params={"code": "sh600519"})
check("市场数据", "分时(上证)", "/api/market/fenshi", params={"code": "sh000001"})
check("市场数据", "龙虎榜", "/api/market/dragon-tiger")
check("市场数据", "板块轮动", "/api/market/rotation")
check("市场数据", "市场概览", "/api/market/overview")
check("市场数据", "行情排名", "/api/market/rankings")
check("市场数据", "板块资金流", "/api/market/sector-fund-flow")
check("市场数据", "板块资金流v2", "/api/market/sector-flow")
check("市场数据", "市场情绪", "/api/market/sentiment")
check("市场数据", "市场统计", "/api/market/stats")
check("市场数据", "概念板块", "/api/market/concepts")
check("市场数据", "市场热力图数据", "/api/market/heatmap/data")
check("市场数据", "现货行情", "/api/market/spot")
check("市场数据", "技术指标", "/api/market/indicators")
check("市场数据", "实时报价", "/api/market/quotes")
check("市场数据", "股票搜索建议", "/api/market/stock-suggest", params={"keyword": "茅台"})

# =====================================
# 4. 个股诊断 API
# =====================================
print("\n" + "="*60)
print("🔍 4. 个股诊断 API (5个)")
print("="*60)
check("个股", "诊断(茅台600519)", "/api/stock/diagnose", params={"code": "600519"})
check("个股", "诊断(平安601318)", "/api/stock/diagnose", params={"code": "601318"})
check("个股", "个股信息", "/api/stock/info", params={"code": "600519"})
check("个股", "压力支撑", "/api/stock/resistance-support", params={"code": "600519"})
check("个股", "股票联想", "/api/stock/suggest", params={"keyword": "茅台"})

# =====================================
# 5. 分析 API
# =====================================
print("\n" + "="*60)
print("📈 5. 分析 API (6个)")
print("="*60)
check("分析", "AI分析面板(600519)", "/api/analysis/dashboard/600519")
check("分析", "亢龙有悔(600519)", "/api/analysis/kang-long-you-hui/600519")
check("分析", "三色共振(600519)", "/api/analysis/three-color-resonance/600519")
check("分析", "精准交易(600519)", "/api/analysis/precise-trading/600519")
check("分析", "暗池资金(600519)", "/api/analysis/dark-pool-fund/600519")
check("分析", "游资分析(600519)", "/api/analysis/hot-money/600519")

# =====================================
# 6. 宏观数据 API
# =====================================
print("\n" + "="*60)
print("🌍 6. 宏观数据 API (1个)")
print("="*60)
check("宏观", "宏观指标", "/api/macro/indicators")

# =====================================
# 7. AI 功能 API
# =====================================
print("\n" + "="*60)
print("🤖 7. AI 功能 API (6个)")
print("="*60)
check("AI", "AI聊天(POST)", "/api/ai/chat", method="POST", payload={"message": "你好"})
check("AI", "AI分析(POST)", "/api/ai/analyze", method="POST", payload={"code": "600519", "question": "简要分析"})
check("AI", "AI提供商列表", "/api/ai/providers")
check("AI", "AI模板列表", "/api/ai/templates")
check("AI", "AI决策-板块表现", "/api/ai/decision/sector-performance")
check("AI", "AI决策-市场洞察", "/api/ai/decision/market-insights")

# =====================================
# 8. 策略 API
# =====================================
print("\n" + "="*60)
print("⚙️ 8. 策略 API (8个)")
print("="*60)
check("策略", "策略列表", "/api/strategies/list")
check("策略", "策略分类", "/api/strategies/categories")
check("策略", "策略统计", "/api/strategies/stats")
check("策略", "策略健康检查", "/api/strategies/health")
check("策略", "策略历史", "/api/strategies/history")
check("策略", "策略选股", "/api/strategy/screen")
check("策略", "回测策略列表", "/api/strategy/backtest/strategies")
check("策略", "经典CTA回测", "/api/strategy/backtest/classic_cta")

# =====================================
# 9. 策略市场
# =====================================
print("\n" + "="*60)
print("🏪 9. 策略市场 API (2个)")
print("="*60)
check("策略市场", "策略列表", "/api/marketplace/list")
check("策略市场", "策略详情(item_1)", "/api/marketplace/item/1")

# =====================================
# 10. 回测 API
# =====================================
print("\n" + "="*60)
print("🔄 10. 回测 API (2个)")
print("="*60)
check("回测", "回测健康", "/api/backtest/health")
check("回测", "回测执行(POST)", "/api/backtest/run", method="POST", payload={
    "strategy": "ma_cross", "code": "600519", "start_date": "2024-01-01", "end_date": "2024-12-31"
})

# =====================================
# 11. 投资组合 API
# =====================================
print("\n" + "="*60)
print("💼 11. 投资组合 API (3个)")
print("="*60)
check("投资组合", "组合总览(sim)", "/api/portfolio/summary", params={"mode": "sim"})
check("投资组合", "组合总览(real)", "/api/portfolio/summary", params={"mode": "real"})
check("投资组合", "持仓列表", "/api/portfolio/positions", params={"mode": "sim"})

# =====================================
# 12. 知识库 API
# =====================================
print("\n" + "="*60)
print("📚 12. 知识库 API (6个)")
print("="*60)
check("知识库", "KB搜索", "/api/kb/search", params={"query": "均线"})
check("知识库", "知识分类", "/api/knowledge/categories")
check("知识库", "知识搜索", "/api/knowledge/search", params={"q": "量化"})
check("知识库", "知识统计", "/api/knowledge/stats")
check("知识库", "知识推荐", "/api/knowledge/recommend")
check("知识库", "知识上下文", "/api/knowledge/context")

# =====================================
# 13. 系统 & 设置 API
# =====================================
print("\n" + "="*60)
print("🔧 13. 系统 & 设置 API (4个)")
print("="*60)
check("系统", "系统监控状态", "/api/system/system/monitor/status")
check("系统", "系统监控摘要", "/api/system/system/monitor/summary")
check("系统", "系统监控告警", "/api/system/system/monitor/alerts")
check("设置", "AI设置(GET)", "/api/settings/ai")

# =====================================
# 14. 交易 API
# =====================================
print("\n" + "="*60)
print("💰 14. 交易 API (7个)")
print("="*60)
check("交易", "交易账户", "/api/trade/accounts")
check("交易", "交易面板", "/api/trade/dashboard")
check("交易", "成交记录", "/api/trade/trades")
check("交易", "历史记录", "/api/trade/history")
check("交易", "交易复盘", "/api/trade/review")
check("交易", "高级复盘", "/api/trade/review/advanced")
check("交易", "条件单列表", "/api/trade/condition-orders")

# =====================================
# 15. 哲学方法论 API
# =====================================
print("\n" + "="*60)
print("🧠 15. 哲学方法论 API (4个)")
print("="*60)
check("哲学", "矛盾分析", "/api/philosophy/contradictions")
check("哲学", "价值散点", "/api/philosophy/value-scatter")
check("哲学预测", "理论指南", "/api/philosophy-prediction/theory-guide")
check("哲学预测", "马克思分析(600519)", "/api/philosophy-prediction/marxist/600519")

# =====================================
# 16. 专业系统 API
# =====================================
print("\n" + "="*60)
print("🏢 16. 专业系统 API (4个)")
print("="*60)
check("专业", "策略模板列表", "/api/professional/strategy-templates")
check("专业", "专业系统健康", "/api/professional/system-health")
check("专业Plus", "信号(600519)", "/api/professional-plus/signals/600519")
check("专业Plus", "信号v2(600519)", "/api/professional-plus/signals-v2/600519")

# =====================================
# 17. 东方智慧API
# =====================================
print("\n" + "="*60)
print("☯️ 17. 东方智慧量化 API (6个)")
print("="*60)
check("东方智慧", "易经(600519)", "/api/eastern-wisdom/iching/600519")
check("东方智慧", "道家(600519)", "/api/eastern-wisdom/daoist/600519")
check("东方智慧", "儒家(600519)", "/api/eastern-wisdom/confucian/600519")
check("东方智慧", "阳明心学(600519)", "/api/eastern-wisdom/yangming/600519")
check("东方智慧", "孙子兵法(600519)", "/api/eastern-wisdom/sunzi/600519")
check("东方智慧", "理论指南", "/api/eastern-wisdom/theory-guide")

# =====================================
# 18. 卢麒元预测 API
# =====================================
print("\n" + "="*60)
print("📐 18. 卢麒元预测 API (4个)")
print("="*60)
check("卢麒元", "方法论", "/api/lu-prediction/methodology")
check("卢麒元", "预测(600519)", "/api/lu-prediction/predict/600519")
check("卢麒元", "市场阶段", "/api/lu-prediction/market-phase")
check("卢麒元", "准确率统计", "/api/lu-prediction/accuracy-stats")

# =====================================
# 19. 机器学习 API
# =====================================
print("\n" + "="*60)
print("🧪 19. 机器学习 API (3个)")
print("="*60)
check("机器学习", "ML状态", "/api/ml/status")
check("机器学习", "特征重要性", "/api/ml/feature-importance")
check("机器学习", "信号表现", "/api/ml/signal-performance")

# =====================================
# 20. 预警 API
# =====================================
print("\n" + "="*60)
print("🔔 20. 预警 API (1个)")
print("="*60)
check("预警", "预警列表", "/api/alerts/list")

# =====================================
# 21. 导出 API
# =====================================
print("\n" + "="*60)
print("📤 21. 导出 API (3个)")
print("="*60)
check("导出", "导出市场数据", "/api/export/market-data/600519")
check("导出", "导出龙虎榜", "/api/export/dragon-tiger")
check("导出", "导出自选股", "/api/export/watchlist")

# =====================================
# 22. 资讯 API
# =====================================
print("\n" + "="*60)
print("📰 22. 资讯 API (2个)")
print("="*60)
check("资讯", "市场快讯", "/api/info/news")
check("资讯", "公告(600519)", "/api/info/notices/600519")

# =====================================
# 23. 多智能体 API
# =====================================
print("\n" + "="*60)
print("🤝 23. 多智能体 API (2个)")
print("="*60)
check("智能体", "智能体信息", "/api/agents/info")
check("智能体", "多智能体分析(POST)", "/api/agents/analyze", method="POST", payload={"code": "600519"})

# =====================================
# 24. 账户 API
# =====================================
print("\n" + "="*60)
print("👤 24. 账户管理 API (2个)")
print("="*60)
check("账户", "账户列表", "/api/accounts/accounts/")
check("账户", "综合绩效", "/api/accounts/accounts/performance/combined")

# =====================================
# 25. 自选股 & 预警
# =====================================
print("\n" + "="*60)
print("⭐ 25. 自选股 API (2个)")
print("="*60)
check("自选股", "自选股列表", "/api/market/watchlist")
check("自选股", "交易观察列表", "/api/trade/watchlist")


# ====================================
# 汇总
# ====================================
print("\n" + "="*80)
print("📋 ROX 3.0 全面功能检测报告 v2")
print("="*80)
print(f"检测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"服务器: {BASE_URL}")

total = len(results["pass"]) + len(results["fail"]) + len(results["warn"]) + len(results["error"])
print(f"\n📊 总计: {total} 项")
print(f"   ✅ 通过: {len(results['pass'])}")
print(f"   ❌ 失败: {len(results['fail'])}")
print(f"   ⚠️ 警告: {len(results['warn'])}")
print(f"   🔴 错误: {len(results['error'])}")
prate = len(results["pass"]) / total * 100 if total else 0
print(f"   🎯 通过率: {prate:.1f}%")

if results["fail"]:
    print(f"\n{'='*60}")
    print("❌ 失败项:")
    print(f"{'='*60}")
    for i, it in enumerate(results["fail"], 1):
        print(f"  {i}. [{it['category']}] {it['name']}  URL: {it['url']}")
        print(f"     原因: {it.get('reason','')[:120]}")

if results["error"]:
    print(f"\n{'='*60}")
    print("🔴 错误项:")
    print(f"{'='*60}")
    for i, it in enumerate(results["error"], 1):
        print(f"  {i}. [{it['category']}] {it['name']}  URL: {it['url']}")
        print(f"     原因: {it.get('reason','')[:120]}")

if results["warn"]:
    print(f"\n{'='*60}")
    print("⚠️ 警告项:")
    print(f"{'='*60}")
    for i, it in enumerate(results["warn"], 1):
        print(f"  {i}. [{it['category']}] {it['name']}  URL: {it['url']}")
        print(f"     原因: {it.get('reason','')[:120]}")

# 按分类
print(f"\n{'='*60}")
print("📊 分类统计:")
print(f"{'='*60}")
cats = {}
for st in ["pass","fail","warn","error"]:
    for it in results[st]:
        c = it["category"]
        if c not in cats: cats[c] = {"pass":0,"fail":0,"warn":0,"error":0}
        cats[c][st] += 1
for c, s in sorted(cats.items()):
    t = sum(s.values())
    r = s["pass"]/t*100 if t else 0
    ic = "✅" if r==100 else ("⚠️" if r>=50 else "❌")
    print(f"  {ic} {c}: {s['pass']}/{t} ({r:.0f}%)")

with open("test_results_v2.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\n📁 结果保存到: test_results_v2.json")
