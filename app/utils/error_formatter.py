# -*- coding: utf-8 -*-
"""
错误格式化器

将技术性错误信息转换为用户友好的提示，明确指出问题所在（数据源、大模型、配置等），并提供可操作的解决建议。
"""

from enum import Enum
from typing import Dict, Optional, Any
import re


class ErrorCategory(Enum):
    """错误类别枚举"""
    LLM_CONFIG_ERROR = "大模型配置错误"
    DATA_SOURCE_ERROR = "数据源错误"
    API_KEY_ERROR = "API Key 错误"
    NETWORK_ERROR = "网络连接错误"
    CONFIG_ERROR = "配置错误"
    VALIDATION_ERROR = "参数验证错误"
    UNKNOWN_ERROR = "未知错误"


class ErrorFormatter:
    """错误格式化器"""
    
    # LLM 提供商映射
    LLM_PROVIDERS = {
        "deepseek": "DeepSeek",
        "google": "Google Gemini",
        "baidu": "百度文心一言",
        "alibaba": "通义千问",
        "azure": "Azure OpenAI",
        "openai": "OpenAI",
    }
    
    # 数据源映射
    DATA_SOURCES = {
        "tushare": "Tushare",
        "akshare": "AKShare",
        "baostock": "BaoStock",
        "efinance": "efinance",
        "alltick": "AllTick",
        "eastmoney": "东方财富网",
    }
    
    @classmethod
    def format_error(cls, error_message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """
        格式化错误信息
        
        Args:
            error_message: 原始错误信息
            context: 上下文信息，包含错误来源等
            
        Returns:
            格式化后的错误信息字典
        """
        if context is None:
            context = {}
        
        # 1. 分类错误
        category = cls._categorize_error(error_message, context)
        
        # 2. 生成友好的标题和消息
        title, message = cls._generate_friendly_message(error_message, context, category)
        
        # 3. 生成解决建议
        suggestion = cls._generate_suggestion(error_message, context, category)
        
        return {
            "category": category.value,
            "title": title,
            "message": message,
            "suggestion": suggestion,
            "technical_detail": error_message
        }
    
    @classmethod
    def _categorize_error(cls, error_message: str, context: Dict[str, Any]) -> ErrorCategory:
        """
        分类错误
        
        Args:
            error_message: 原始错误信息
            context: 上下文信息
            
        Returns:
            错误类别
        """
        error_lower = error_message.lower()
        
        # 检查上下文
        if "llm_provider" in context:
            return ErrorCategory.LLM_CONFIG_ERROR
        
        if "data_source" in context:
            return ErrorCategory.DATA_SOURCE_ERROR
        
        # 检查错误信息关键词
        if any(keyword in error_lower for keyword in ["api key", "api_key", "token", "认证"]):
            return ErrorCategory.API_KEY_ERROR
        
        if any(keyword in error_lower for keyword in ["网络", "connection", "timeout", "connect"]):
            return ErrorCategory.NETWORK_ERROR
        
        if any(keyword in error_lower for keyword in ["配置", "config", "setting"]):
            return ErrorCategory.CONFIG_ERROR
        
        if any(keyword in error_lower for keyword in ["参数", "validation", "invalid", "required"]):
            return ErrorCategory.VALIDATION_ERROR
        
        return ErrorCategory.UNKNOWN_ERROR
    
    @classmethod
    def _generate_friendly_message(cls, error_message: str, context: Dict[str, Any], category: ErrorCategory) -> tuple:
        """
        生成友好的错误标题和消息
        
        Args:
            error_message: 原始错误信息
            context: 上下文信息
            category: 错误类别
            
        Returns:
            (标题, 消息) 元组
        """
        error_lower = error_message.lower()
        
        # LLM 配置错误
        if category == ErrorCategory.LLM_CONFIG_ERROR:
            provider = context.get("llm_provider", "").lower()
            provider_name = cls.LLM_PROVIDERS.get(provider, "大模型")
            
            if "api key" in error_lower or "invalid" in error_lower:
                return (
                    f"❌ {provider_name} API Key 无效",
                    f"{provider_name} 的 API Key 无效或未配置。"
                )
            elif "quota" in error_lower or "limit" in error_lower:
                return (
                    f"❌ {provider_name} 配额不足",
                    f"{provider_name} 的 API 配额已耗尽，请稍后再试或切换到其他模型。"
                )
            else:
                return (
                    f"❌ {provider_name} 配置错误",
                    f"{provider_name} 配置出现问题，请检查相关设置。"
                )
        
        # 数据源错误
        elif category == ErrorCategory.DATA_SOURCE_ERROR:
            data_source = context.get("data_source", "").lower()
            source_name = cls.DATA_SOURCES.get(data_source, "数据源")
            
            if "api key" in error_lower or "token" in error_lower:
                return (
                    f"❌ {source_name} API Key 无效",
                    f"{source_name} 的 API Key 无效或未配置。"
                )
            elif "timeout" in error_lower or "connection" in error_lower:
                return (
                    f"❌ {source_name} 连接失败",
                    f"无法连接到 {source_name} 服务，请检查网络连接。"
                )
            else:
                return (
                    f"❌ {source_name} 数据获取失败",
                    f"无法从 {source_name} 获取数据，请检查配置或稍后再试。"
                )
        
        # API Key 错误
        elif category == ErrorCategory.API_KEY_ERROR:
            return (
                "❌ API Key 错误",
                "API Key 无效或未配置，请检查相关设置。"
            )
        
        # 网络错误
        elif category == ErrorCategory.NETWORK_ERROR:
            return (
                "❌ 网络连接错误",
                "网络连接失败，请检查您的网络设置或稍后再试。"
            )
        
        # 配置错误
        elif category == ErrorCategory.CONFIG_ERROR:
            return (
                "❌ 配置错误",
                "系统配置出现问题，请检查相关设置。"
            )
        
        # 参数验证错误
        elif category == ErrorCategory.VALIDATION_ERROR:
            return (
                "❌ 参数错误",
                "请求参数无效，请检查输入信息。"
            )
        
        # 未知错误
        else:
            return (
                "❌ 未知错误",
                "系统出现未知错误，请稍后再试。"
            )
    
    @classmethod
    def _generate_suggestion(cls, error_message: str, context: Dict[str, Any], category: ErrorCategory) -> str:
        """
        生成解决建议
        
        Args:
            error_message: 原始错误信息
            context: 上下文信息
            category: 错误类别
            
        Returns:
            解决建议
        """
        error_lower = error_message.lower()
        
        # LLM 配置错误
        if category == ErrorCategory.LLM_CONFIG_ERROR:
            provider = context.get("llm_provider", "").lower()
            provider_name = cls.LLM_PROVIDERS.get(provider, "大模型")
            
            if "api key" in error_lower or "invalid" in error_lower:
                return (
                    f"💡 请检查以下几点：\n"  
                    f"1. 在「系统设置 → 大模型配置」中检查 {provider_name} 的 API Key 是否正确\n"  
                    f"2. 确认 API Key 是否已激活且有效\n"  
                    f"3. 尝试重新生成 API Key 并更新配置\n"  
                    f"4. 或者切换到其他可用的大模型"
                )
            elif "quota" in error_lower or "limit" in error_lower:
                return (
                    f"💡 请检查以下几点：\n"  
                    f"1. 等待 {provider_name} 配额重置（通常为每天）\n"  
                    f"2. 切换到其他可用的大模型\n"  
                    f"3. 考虑升级 {provider_name} 账户以获得更高配额"
                )
            else:
                return (
                    f"💡 请检查以下几点：\n"  
                    f"1. 在「系统设置 → 大模型配置」中检查 {provider_name} 的配置\n"  
                    f"2. 确认网络连接正常\n"  
                    f"3. 尝试重启应用\n"  
                    f"4. 或者切换到其他可用的大模型"
                )
        
        # 数据源错误
        elif category == ErrorCategory.DATA_SOURCE_ERROR:
            data_source = context.get("data_source", "").lower()
            source_name = cls.DATA_SOURCES.get(data_source, "数据源")
            
            if "api key" in error_lower or "token" in error_lower:
                return (
                    f"💡 请检查以下几点：\n"  
                    f"1. 在「系统设置 → 数据源配置」中检查 {source_name} 的 API Key/Token 是否正确\n"  
                    f"2. 确认 API Key 是否已激活且有效\n"  
                    f"3. 尝试重新生成 API Key 并更新配置\n"  
                    f"4. 或者切换到其他可用的数据源"
                )
            elif "timeout" in error_lower or "connection" in error_lower:
                return (
                    f"💡 请检查以下几点：\n"  
                    f"1. 检查网络连接是否正常\n"  
                    f"2. 确认 {source_name} 服务是否正常运行\n"  
                    f"3. 尝试稍后再试\n"  
                    f"4. 或者切换到其他可用的数据源"
                )
            else:
                return (
                    f"💡 请检查以下几点：\n"  
                    f"1. 检查 {source_name} 配置是否正确\n"  
                    f"2. 确认网络连接正常\n"  
                    f"3. 尝试稍后再试\n"  
                    f"4. 或者切换到其他可用的数据源"
                )
        
        # API Key 错误
        elif category == ErrorCategory.API_KEY_ERROR:
            return (
                "💡 请检查以下几点：\n"  
                "1. 检查相关服务的 API Key 是否正确配置\n"  
                "2. 确认 API Key 是否已激活且有效\n"  
                "3. 尝试重新生成 API Key 并更新配置"
            )
        
        # 网络错误
        elif category == ErrorCategory.NETWORK_ERROR:
            return (
                "💡 请检查以下几点：\n"  
                "1. 检查网络连接是否正常\n"  
                "2. 确认相关服务是否可访问\n"  
                "3. 尝试重启网络设备\n"  
                "4. 稍后再试"
            )
        
        # 配置错误
        elif category == ErrorCategory.CONFIG_ERROR:
            return (
                "💡 请检查以下几点：\n"  
                "1. 在「系统设置」中检查相关配置\n"  
                "2. 确认配置参数是否正确\n"  
                "3. 尝试重置为默认配置并重新设置"
            )
        
        # 参数验证错误
        elif category == ErrorCategory.VALIDATION_ERROR:
            return (
                "💡 请检查以下几点：\n"  
                "1. 检查输入参数是否正确\n"  
                "2. 确认所有必填字段都已填写\n"  
                "3. 按照提示信息修正输入"
            )
        
        # 未知错误
        else:
            return (
                "💡 请尝试以下操作：\n"  
                "1. 刷新页面或重启应用\n"  
                "2. 检查网络连接\n"  
                "3. 稍后再试\n"  
                "4. 如果问题持续，请联系技术支持"
            )
