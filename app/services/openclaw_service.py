"""
OpenClaw股票数据API服务集成
"""
import os
import sys
import logging
import time
from typing import Optional, List, Dict, Any, Union
from functools import lru_cache

logger = logging.getLogger(__name__)

# 添加openclaw skill路径
openclaw_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                            "openclaw-stock-data-skill")
if os.path.exists(openclaw_path):
    sys.path.insert(0, openclaw_path)

try:
    from stock_api import (
        get_stock_list, get_daily_data, get_history_data,
        get_finance_data, get_financial_indicator, get_main_fund_flow,
        get_main_fund_flow_overview, get_cyq_chips,
        get_bond_daily, get_bond_history, get_bond_indicator_daily,
        get_etf_list, get_etf_daily, get_etf_history,
        get_index_history, get_index_realtime_history,
        get_dragon_tiger
    )
    OPENCLAW_AVAILABLE = True
except ImportError as e:
    logger.warning(f"无法导入OpenClaw库: {e}")
    OPENCLAW_AVAILABLE = False

# 默认配置
DEFAULT_API_DOMAIN = "data.diemeng.chat"
DEFAULT_API_KEY = ""

class OpenClawService:
    def __init__(self, api_key: Optional[str] = None, api_domain: Optional[str] = None):
        self.api_key = api_key or os.environ.get("STOCK_API_KEY", DEFAULT_API_KEY)
        self.api_domain = api_domain or os.environ.get("STOCK_API_DOMAIN", DEFAULT_API_DOMAIN)
        self._available = OPENCLAW_AVAILABLE and self.api_key
        
        if self._available:
            logger.info(f"OpenClaw服务已初始化 (domain: {self.api_domain})")
    
    @property
    def available(self) -> bool:
        return self._available
    
    def set_config(self, api_key: str, api_domain: Optional[str] = None):
        """设置API配置"""
        self.api_key = api_key
        if api_domain:
            self.api_domain = api_domain
        os.environ["STOCK_API_KEY"] = api_key
        os.environ["STOCK_API_DOMAIN"] = self.api_domain
        self._available = OPENCLAW_AVAILABLE and self.api_key
        
    # ===========股票实时与历史数据===========
    
    def get_stock_snapshot(self, stock_codes: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """获取股票实时快照"""
        if not self.available:
            raise RuntimeError("OpenClaw服务不可用")
        try:
            result = get_stock_list()
            return result.get("list", [])
        except Exception as e:
            logger.error(f"获取股票快照失败: {e}")
            raise
    
    def get_stock_daily(self, stock_codes: Union[str, List[str], None],
                       start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取股票日K线数据"""
        if not self.available:
            raise RuntimeError("OpenClaw服务不可用")
        try:
            result = get_daily_data(stock_codes, start_date, end_date)
            return result.get("list", [])
        except Exception as e:
            logger.error(f"获取日K线数据失败: {e}")
            raise
    
    def get_stock_history(self, stock_codes: Union[str, List[str], None],
                          level: str = "1min",
                          start_time: Optional[str] = None,
                          end_time: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取股票分钟级历史数据"""
        if not self.available:
            raise RuntimeError("OpenClaw服务不可用")
        try:
            result = get_history_data(stock_codes, level, start_time, end_time)
            return result.get("list", [])
        except Exception as e:
            logger.error(f"获取历史分钟数据失败: {e}")
            raise
    
    # ===========财务数据===========
    
    def get_finance_data(self, stock_codes: Union[str, List[str], None],
                        start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取财务数据（PE, PB, PS, 市值, PE百分位等）"""
        if not self.available:
            raise RuntimeError("OpenClaw服务不可用")
        try:
            result = get_finance_data(stock_codes, start_date, end_date)
            return result.get("list", [])
        except Exception as e:
            logger.error(f"获取财务数据失败: {e}")
            raise
    
    def get_financial_indicators(self, stock_codes: Union[str, List[str], None]) -> List[Dict[str, Any]]:
        """获取详细财务指标"""
        if not self.available:
            raise RuntimeError("OpenClaw服务不可用")
        try:
            result = get_financial_indicator(stock_codes)
            return result.get("list", [])
        except Exception as e:
            logger.error(f"获取财务指标失败: {e}")
            raise
    
    # ===========主力资金流向===========
    
    def get_main_fund_flow(self, stock_codes: Optional[List[str]] = None,
                          start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取主力资金流向详情（大单、中单、小单）"""
        if not self.available:
            raise RuntimeError("OpenClaw服务不可用")
        try:
            result = get_main_fund_flow(start_date, end_date, stock_codes)
            return result.get("list", [])
        except Exception as e:
            logger.error(f"获取主力资金流向失败: {e}")
            raise
    
    def get_main_fund_overview(self, start_date: Optional[str] = None,
                                end_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取主力资金流向概览"""
        if not self.available:
            raise RuntimeError("OpenClaw服务不可用")
        try:
            result = get_main_fund_flow_overview(start_date, end_date)
            return result.get("list", [])
        except Exception as e:
            logger.error(f"获取主力资金概览失败: {e}")
            raise
    
    def get_cyq_chips(self, stock_codes: Optional[List[str]] = None,
                     start_date: Optional[str] = None,
                     end_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取筹码分布"""
        if not self.available:
            raise RuntimeError("OpenClaw服务不可用")
        try:
            result = get_cyq_chips(start_date, end_date, stock_codes)
            return result.get("list", [])
        except Exception as e:
            logger.error(f"获取筹码分布失败: {e}")
            raise
    
    # ===========可转债数据===========
    
    def get_bond_list(self) -> List[Dict[str, Any]]:
        """获取可转债列表"""
        if not self.available:
            raise RuntimeError("OpenClaw服务不可用")
        try:
            result = get_bond_list()
            return result.get("list", [])
        except Exception as e:
            logger.error(f"获取可转债列表失败: {e}")
            raise
    
    def get_bond_daily_data(self, bond_codes: Optional[List[str]] = None,
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取可转债日线数据"""
        if not self.available:
            raise RuntimeError("OpenClaw服务不可用")
        try:
            result = get_bond_daily(bond_codes, start_date, end_date)
            return result.get("list", [])
        except Exception as e:
            logger.error(f"获取可转债日线数据失败: {e}")
            raise
    
    def get_bond_history_data(self, bond_codes: Optional[List[str]] = None,
                             level: str = "5min",
                             start_time: Optional[str] = None,
                             end_time: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取可转债分钟级历史数据"""
        if not self.available:
            raise RuntimeError("OpenClaw服务不可用")
        try:
            result = get_bond_history(bond_codes, level, start_time, end_time)
            return result.get("list", [])
        except Exception as e:
            logger.error(f"获取可转债历史数据失败: {e}")
            raise
    
    def get_bond_indicators(self, bond_codes: Optional[List[str]] = None,
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取可转债指标数据（纯债价值、转股溢价等）"""
        if not self.available:
            raise RuntimeError("OpenClaw服务不可用")
        try:
            result = get_bond_indicator_daily(bond_codes, start_date, end_date)
            return result.get("list", [])
        except Exception as e:
            logger.error(f"获取可转债指标失败: {e}")
            raise
    
    # ===========ETF数据===========
    
    def get_etf_list(self) -> List[Dict[str, Any]]:
        """获取ETF列表"""
        if not self.available:
            raise RuntimeError("OpenClaw服务不可用")
        try:
            return get_etf_list()
        except Exception as e:
            logger.error(f"获取ETF列表失败: {e}")
            raise
    
    def get_etf_daily_data(self, etf_codes: Optional[List[str]] = None,
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取ETF日线数据"""
        if not self.available:
            raise RuntimeError("OpenClaw服务不可用")
        try:
            result = get_etf_daily(etf_codes, start_date, end_date)
            return result.get("list", [])
        except Exception as e:
            logger.error(f"获取ETF日线数据失败: {e}")
            raise
    
    def get_etf_history_data(self, etf_codes: Optional[List[str]] = None,
                            level: str = "5min",
                            start_time: Optional[str] = None,
                            end_time: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取ETF分钟级历史数据"""
        if not self.available:
            raise RuntimeError("OpenClaw服务不可用")
        try:
            result = get_etf_history(etf_codes, level, start_time, end_time)
            return result.get("list", [])
        except Exception as e:
            logger.error(f"获取ETF历史数据失败: {e}")
            raise
    
    # ===========指数与板块===========
    
    def get_index_history_data(self, index_codes: Optional[List[str]] = None,
                              level: str = "1min",
                              start_time: Optional[str] = None,
                              end_time: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取指数分钟级历史数据"""
        if not self.available:
            raise RuntimeError("OpenClaw服务不可用")
        try:
            result = get_index_history(index_codes, level, start_time, end_time)
            return result.get("list", [])
        except Exception as e:
            logger.error(f"获取指数历史数据失败: {e}")
            raise
    
    # ===========龙虎榜===========
    
    def get_dragon_tiger(self) -> List[Dict[str, Any]]:
        """获取龙虎榜数据"""
        if not self.available:
            raise RuntimeError("OpenClaw服务不可用")
        try:
            result = get_dragon_tiger()
            return result.get("list", [])
        except Exception as e:
            logger.error(f"获取龙虎榜数据失败: {e}")
            raise

# 全局实例
_global_service: Optional[OpenClawService] = None

def get_openclaw_service(api_key: Optional[str] = None, api_domain: Optional[str] = None) -> OpenClawService:
    """获取OpenClaw服务单例"""
    global _global_service
    if _global_service is None:
        _global_service = OpenClawService(api_key, api_domain)
    elif api_key or api_domain:
        _global_service.set_config(api_key or _global_service.api_key, 
                                    api_domain or _global_service.api_domain)
    return _global_service

def format_stock_code(code: str, market: str = "auto") -> str:
    """格式化股票代码为OpenClaw格式 (SH/SZ后缀)"""
    code = str(code).strip()
    if code.endswith(('.SH', '.SZ', '.BJ')):
        return code
    if code.startswith(('60', '688')):
        return f"{code}.SH"
    elif code.startswith(('00', '30')):
        return f"{code}.SZ"
    elif code.startswith(('43', '83', '87')):
        return f"{code}.BJ"
    return code

