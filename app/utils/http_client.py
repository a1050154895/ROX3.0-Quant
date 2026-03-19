import aiohttp
import json
import logging
from typing import Optional, Dict, Any, List, Union
from urllib.parse import urlencode

logger = logging.getLogger(__name__)

class AsyncHTTPClient:
    """异步HTTP客户端，基于aiohttp"""
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def initialize(self):
        """初始化客户端"""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(
                total=30.0,
                connect=10.0,
                sock_read=20.0,
                sock_connect=10.0
            )
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers={
                    'Content-Type': 'application/json',
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.1 Safari/605.1.15'
                }
            )
    
    async def close(self):
        """关闭客户端"""
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def get(self, url: str, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """发送GET请求"""
        await self.initialize()
        
        try:
            if params:
                url = f"{url}?{urlencode(params)}"
            
            logger.debug(f"GET {url}")
            
            async with self.session.get(url, headers=headers) as response:
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientError as e:
            logger.error(f"GET request failed: {e}")
            raise
    
    async def post(self, url: str, data: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """发送POST请求"""
        await self.initialize()
        
        try:
            logger.debug(f"POST {url} {data}")
            
            async with self.session.post(url, json=data, headers=headers) as response:
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientError as e:
            logger.error(f"POST request failed: {e}")
            raise
    
    async def post_form(self, url: str, data: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """发送表单POST请求"""
        await self.initialize()
        
        try:
            logger.debug(f"POST FORM {url} {data}")
            
            if headers is None:
                headers = {}
            headers['Content-Type'] = 'application/x-www-form-urlencoded'
            
            async with self.session.post(url, data=data, headers=headers) as response:
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientError as e:
            logger.error(f"POST FORM request failed: {e}")
            raise
    
    async def fetch_text(self, url: str, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None) -> str:
        """获取文本内容"""
        await self.initialize()
        
        try:
            if params:
                url = f"{url}?{urlencode(params)}"
            
            logger.debug(f"GET TEXT {url}")
            
            async with self.session.get(url, headers=headers) as response:
                response.raise_for_status()
                return await response.text()
        except aiohttp.ClientError as e:
            logger.error(f"Fetch text failed: {e}")
            raise

# 创建全局异步HTTP客户端实例
async_http_client = AsyncHTTPClient()

async def async_get(url: str, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """便捷的异步GET请求函数"""
    async with AsyncHTTPClient() as client:
        return await client.get(url, params, headers)

async def async_post(url: str, data: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """便捷的异步POST请求函数"""
    async with AsyncHTTPClient() as client:
        return await client.post(url, data, headers)

async def async_fetch_text(url: str, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None) -> str:
    """便捷的异步文本获取函数"""
    async with AsyncHTTPClient() as client:
        return await client.fetch_text(url, params, headers)