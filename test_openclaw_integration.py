"""
OpenClaw集成测试脚本
测试ROX平台与OpenClaw Gateway的通信
"""

import asyncio
import requests
import json
from datetime import datetime

class OpenClawIntegrationTest:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.test_results = []
    
    def log_test(self, test_name: str, success: bool, message: str = ""):
        """记录测试结果"""
        result = {
            "test": test_name,
            "success": success,
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        self.test_results.append(result)
        
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {test_name}")
        if message:
            print(f"   {message}")
    
    async def test_health_check(self):
        """测试健康检查"""
        try:
            response = requests.get(f"{self.base_url}/api/openclaw/health", timeout=5)
            data = response.json()
            
            success = response.status_code == 200
            self.log_test(
                "健康检查",
                success,
                f"状态: {data.get('status', 'unknown')}, Gateway: {data.get('gateway_url', 'N/A')}"
            )
            return success
        except Exception as e:
            self.log_test("健康检查", False, str(e))
            return False
    
    async def test_trade_signal(self):
        """测试交易信号发送"""
        try:
            payload = {
                "symbol": "600519",
                "action": "buy",
                "price": 1800.50,
                "confidence": 0.85,
                "reason": "技术面突破关键阻力位 - 测试信号"
            }
            
            response = requests.post(
                f"{self.base_url}/api/openclaw/signal",
                json=payload,
                timeout=10
            )
            data = response.json()
            
            success = response.status_code == 200
            self.log_test(
                "交易信号发送",
                success,
                f"股票: {payload['symbol']}, 操作: {payload['action']}, 结果: {data.get('status', 'unknown')}"
            )
            return success
        except Exception as e:
            self.log_test("交易信号发送", False, str(e))
            return False
    
    async def test_telegram_message(self):
        """测试Telegram消息发送"""
        try:
            payload = {
                "chat_id": "test_chat_id",
                "message": "*🚀 测试消息*\n\n这是一条来自ROX的测试消息。\n\n⏰ 时间: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "parse_mode": "Markdown"
            }
            
            response = requests.post(
                f"{self.base_url}/api/openclaw/telegram",
                json=payload,
                timeout=10
            )
            data = response.json()
            
            success = response.status_code == 200
            self.log_test(
                "Telegram消息发送",
                success,
                f"状态: {data.get('status', 'unknown')}"
            )
            return success
        except Exception as e:
            self.log_test("Telegram消息发送", False, str(e))
            return False
    
    async def test_broadcast(self):
        """测试消息广播"""
        try:
            payload = {
                "message": "📢 广播测试消息 - ROX平台OpenClaw集成测试",
                "channels": ["telegram"],
                "recipients": {}
            }
            
            response = requests.post(
                f"{self.base_url}/api/openclaw/broadcast",
                json=payload,
                timeout=10
            )
            data = response.json()
            
            success = response.status_code == 200
            self.log_test(
                "消息广播",
                success,
                f"状态: {data.get('status', 'unknown')}"
            )
            return success
        except Exception as e:
            self.log_test("消息广播", False, str(e))
            return False
    
    async def test_trader_update(self):
        """测试交易员动态"""
        try:
            payload = {
                "trader_id": "test_trader_001",
                "trader_name": "测试交易员",
                "action": "trade_executed",
                "symbol": "000001",
                "profit": 5.2,
                "emotion": 0.8
            }
            
            response = requests.post(
                f"{self.base_url}/api/openclaw/trader-update",
                json=payload,
                timeout=10
            )
            data = response.json()
            
            success = response.status_code == 200
            self.log_test(
                "交易员动态",
                success,
                f"交易员: {payload['trader_name']}, 动作: {payload['action']}"
            )
            return success
        except Exception as e:
            self.log_test("交易员动态", False, str(e))
            return False
    
    async def test_risk_alert(self):
        """测试风险预警"""
        try:
            payload = {
                "alert_level": "warning",
                "title": "测试风险预警",
                "message": "这是一条测试风险预警消息",
                "details": {
                    "risk_type": "market_volatility",
                    "severity": "medium"
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/openclaw/risk-alert",
                json=payload,
                timeout=10
            )
            data = response.json()
            
            success = response.status_code == 200
            self.log_test(
                "风险预警",
                success,
                f"级别: {payload['alert_level']}, 标题: {payload['title']}"
            )
            return success
        except Exception as e:
            self.log_test("风险预警", False, str(e))
            return False
    
    async def test_enable_disable(self):
        """测试启用/禁用功能"""
        try:
            # 测试禁用
            response = requests.post(f"{self.base_url}/api/openclaw/disable", timeout=5)
            data = response.json()
            
            if response.status_code != 200:
                self.log_test("启用/禁用功能", False, "禁用失败")
                return False
            
            # 测试启用
            response = requests.post(f"{self.base_url}/api/openclaw/enable", timeout=5)
            data = response.json()
            
            success = response.status_code == 200
            self.log_test(
                "启用/禁用功能",
                success,
                f"状态切换: {data.get('status', 'unknown')}"
            )
            return success
        except Exception as e:
            self.log_test("启用/禁用功能", False, str(e))
            return False
    
    async def run_all_tests(self):
        """运行所有测试"""
        print("=" * 60)
        print("OpenClaw集成测试")
        print("=" * 60)
        print()
        
        tests = [
            self.test_health_check,
            self.test_trade_signal,
            self.test_telegram_message,
            self.test_broadcast,
            self.test_trader_update,
            self.test_risk_alert,
            self.test_enable_disable
        ]
        
        for test in tests:
            await test()
            print()
        
        # 打印总结
        print("=" * 60)
        print("测试总结")
        print("=" * 60)
        
        total = len(self.test_results)
        passed = sum(1 for r in self.test_results if r["success"])
        failed = total - passed
        
        print(f"总计: {total} 个测试")
        print(f"通过: {passed} 个 ✅")
        print(f"失败: {failed} 个 ❌")
        print()
        
        if failed > 0:
            print("失败的测试:")
            for result in self.test_results:
                if not result["success"]:
                    print(f"  - {result['test']}: {result['message']}")
        
        return failed == 0

async def main():
    """主函数"""
    tester = OpenClawIntegrationTest()
    success = await tester.run_all_tests()
    
    if success:
        print("\n🎉 所有测试通过！OpenClaw集成成功！")
    else:
        print("\n⚠️ 部分测试失败，请检查配置和网络连接")

if __name__ == "__main__":
    asyncio.run(main())
