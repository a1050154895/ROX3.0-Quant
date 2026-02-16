#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
信号计算缓存机制

优化信号计算性能，减少重复计算
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple, List
import pandas as pd

logger = logging.getLogger(__name__)


class SignalCache:
    """
    信号计算缓存
    
    缓存信号计算结果，避免重复计算
    支持基于时间和数据的过期机制
    """
    
    def __init__(self, max_size: int = 2000, expire_seconds: int = 300, cleanup_interval: int = 60):
        """
        初始化缓存
        
        Args:
            max_size: 缓存最大容量
            expire_seconds: 缓存过期时间（秒）
            cleanup_interval: 缓存清理间隔（秒）
        """
        self.max_size = max_size
        self.expire_seconds = expire_seconds
        self.cleanup_interval = cleanup_interval
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, datetime] = {}
        self.last_cleanup = datetime.now()
        self.hit_count = 0
        self.miss_count = 0
        self.eviction_count = 0
    
    def _generate_key(self, signal_name: str, code: str, df_hash: str) -> str:
        """
        生成缓存键
        
        Args:
            signal_name: 信号名称
            code: 股票代码
            df_hash: 数据哈希值
        
        Returns:
            缓存键
        """
        return f"{signal_name}:{code}:{df_hash}"
    
    def _get_df_hash(self, df: pd.DataFrame) -> str:
        """
        获取DataFrame的哈希值
        
        Args:
            df: 数据框
        
        Returns:
            哈希值
        """
        if df.empty:
            return "empty"
        
        # 更高效的哈希计算方式
        # 基于最近10条数据的关键列
        recent_df = df.tail(min(10, len(df)))
        
        # 只使用关键列进行哈希计算
        key_columns = ['close', 'high', 'low', 'volume']
        available_columns = [col for col in key_columns if col in recent_df.columns]
        
        if not available_columns:
            # 如果没有关键列，使用所有列
            hash_df = recent_df
        else:
            hash_df = recent_df[available_columns]
        
        # 使用更高效的哈希计算方法
        import hashlib
        md5_hash = hashlib.md5()
        md5_hash.update(hash_df.to_csv(index=False).encode('utf-8'))
        return md5_hash.hexdigest()
    
    def get(self, signal_name: str, code: str, df: pd.DataFrame) -> Optional[Any]:
        """
        获取缓存的信号结果
        
        Args:
            signal_name: 信号名称
            code: 股票代码
            df: 数据框
        
        Returns:
            缓存的结果，如果不存在或过期则返回None
        """
        try:
            # 定期清理过期缓存
            self._periodic_cleanup()
            
            df_hash = self._get_df_hash(df)
            key = self._generate_key(signal_name, code, df_hash)
            
            if key not in self.cache:
                self.miss_count += 1
                return None
            
            # 检查是否过期
            cached_data = self.cache[key]
            timestamp = cached_data['timestamp']
            
            if (datetime.now() - timestamp).total_seconds() > self.expire_seconds:
                del self.cache[key]
                if key in self.access_times:
                    del self.access_times[key]
                self.miss_count += 1
                return None
            
            # 更新访问时间
            self.access_times[key] = datetime.now()
            self.hit_count += 1
            
            return cached_data['result']
            
        except Exception as e:
            logger.error(f"获取缓存失败: {e}")
            self.miss_count += 1
            return None
    
    def set(self, signal_name: str, code: str, df: pd.DataFrame, result: Any) -> None:
        """
        设置缓存的信号结果
        
        Args:
            signal_name: 信号名称
            code: 股票代码
            df: 数据框
            result: 信号结果
        """
        try:
            df_hash = self._get_df_hash(df)
            key = self._generate_key(signal_name, code, df_hash)
            
            # 检查缓存大小
            if len(self.cache) >= self.max_size:
                # 使用更智能的缓存淘汰策略
                self._evict_cache()
            
            # 设置缓存
            self.cache[key] = {
                'result': result,
                'timestamp': datetime.now(),
                'signal_name': signal_name,
                'code': code
            }
            self.access_times[key] = datetime.now()
            
        except Exception as e:
            logger.error(f"设置缓存失败: {e}")
    
    def _periodic_cleanup(self):
        """
        定期清理过期缓存
        """
        if (datetime.now() - self.last_cleanup).total_seconds() > self.cleanup_interval:
            self._cleanup_expired()
            self.last_cleanup = datetime.now()
    
    def _cleanup_expired(self):
        """
        清理过期缓存
        """
        expired_keys = []
        current_time = datetime.now()
        
        for key, cached_data in self.cache.items():
            timestamp = cached_data['timestamp']
            if (current_time - timestamp).total_seconds() > self.expire_seconds:
                expired_keys.append(key)
        
        for key in expired_keys:
            if key in self.cache:
                del self.cache[key]
            if key in self.access_times:
                del self.access_times[key]
        
        if expired_keys:
            logger.debug(f"清理了 {len(expired_keys)} 个过期缓存")
    
    def _evict_cache(self):
        """
        智能缓存淘汰策略
        """
        # 首先清理过期缓存
        self._cleanup_expired()
        
        # 如果仍然超出容量，使用LRU策略
        if len(self.cache) >= self.max_size:
            # 按访问时间排序，删除最久未使用的
            sorted_keys = sorted(self.access_times.items(), key=lambda x: x[1])
            evict_count = len(self.cache) - self.max_size + 10  # 多删除一些，避免频繁清理
            
            for key, _ in sorted_keys[:evict_count]:
                if key in self.cache:
                    del self.cache[key]
                if key in self.access_times:
                    del self.access_times[key]
                self.eviction_count += 1
            
            logger.debug(f"淘汰了 {evict_count} 个缓存项")
    
    def clear(self) -> None:
        """
        清空缓存
        """
        self.cache.clear()
        self.access_times.clear()
        self.hit_count = 0
        self.miss_count = 0
        self.eviction_count = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            缓存统计信息
        """
        hit_rate = self.hit_count / (self.hit_count + self.miss_count) * 100 if (self.hit_count + self.miss_count) > 0 else 0
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'expire_seconds': self.expire_seconds,
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': round(hit_rate, 2),
            'eviction_count': self.eviction_count,
            'last_cleanup': self.last_cleanup.isoformat() if self.last_cleanup else None
        }
    
    def batch_get(self, items: List[Tuple[str, str, pd.DataFrame]]) -> Dict[Tuple[str, str, str], Any]:
        """
        批量获取缓存
        
        Args:
            items: 包含(signal_name, code, df)的列表
        
        Returns:
            缓存结果字典，键为(signal_name, code, df_hash)
        """
        results = {}
        
        for signal_name, code, df in items:
            df_hash = self._get_df_hash(df)
            key = self._generate_key(signal_name, code, df_hash)
            
            if key in self.cache:
                cached_data = self.cache[key]
                timestamp = cached_data['timestamp']
                
                if (datetime.now() - timestamp).total_seconds() <= self.expire_seconds:
                    results[(signal_name, code, df_hash)] = cached_data['result']
                    self.access_times[key] = datetime.now()
                    self.hit_count += 1
                else:
                    # 删除过期缓存
                    del self.cache[key]
                    if key in self.access_times:
                        del self.access_times[key]
                    self.miss_count += 1
            else:
                self.miss_count += 1
        
        return results
    
    def batch_set(self, items: List[Tuple[str, str, pd.DataFrame, Any]]) -> None:
        """
        批量设置缓存
        
        Args:
            items: 包含(signal_name, code, df, result)的列表
        """
        for signal_name, code, df, result in items:
            self.set(signal_name, code, df, result)


# 全局缓存实例
_signal_cache = None


def get_signal_cache() -> SignalCache:
    """
    获取信号缓存实例
    
    Returns:
        信号缓存实例
    """
    global _signal_cache
    if _signal_cache is None:
        _signal_cache = SignalCache()
    return _signal_cache
