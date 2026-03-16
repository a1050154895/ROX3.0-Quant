#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""轻量级 psutil 兼容层（无外部依赖）。"""

from __future__ import annotations

from dataclasses import dataclass
import os
import time


@dataclass
class _MemInfo:
    percent: float


@dataclass
class _DiskInfo:
    percent: float


_last_total = None
_last_idle = None


def _read_proc_stat():
    with open('/proc/stat', 'r', encoding='utf-8') as f:
        line = f.readline()
    parts = line.split()
    values = [float(x) for x in parts[1:8]]
    idle = values[3] + values[4]
    total = sum(values)
    return total, idle


def cpu_percent(interval: float = 0.0) -> float:
    global _last_total, _last_idle

    if interval and interval > 0:
        total1, idle1 = _read_proc_stat()
        time.sleep(interval)
        total2, idle2 = _read_proc_stat()
        total_delta = max(total2 - total1, 1e-9)
        idle_delta = max(idle2 - idle1, 0.0)
        usage = max(0.0, min(100.0, (1.0 - idle_delta / total_delta) * 100.0))
        return usage

    total, idle = _read_proc_stat()
    if _last_total is None or _last_idle is None:
        _last_total, _last_idle = total, idle
        return 0.0

    total_delta = max(total - _last_total, 1e-9)
    idle_delta = max(idle - _last_idle, 0.0)
    _last_total, _last_idle = total, idle
    usage = max(0.0, min(100.0, (1.0 - idle_delta / total_delta) * 100.0))
    return usage


def virtual_memory() -> _MemInfo:
    mem_total = 0
    mem_available = 0
    with open('/proc/meminfo', 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('MemTotal:'):
                mem_total = float(line.split()[1])
            elif line.startswith('MemAvailable:'):
                mem_available = float(line.split()[1])
    if mem_total <= 0:
        return _MemInfo(percent=0.0)
    used = max(mem_total - mem_available, 0.0)
    return _MemInfo(percent=max(0.0, min(100.0, used / mem_total * 100.0)))


def disk_usage(path: str) -> _DiskInfo:
    st = os.statvfs(path)
    total = st.f_blocks * st.f_frsize
    free = st.f_bavail * st.f_frsize
    if total <= 0:
        return _DiskInfo(percent=0.0)
    used = max(total - free, 0)
    return _DiskInfo(percent=max(0.0, min(100.0, used / total * 100.0)))
