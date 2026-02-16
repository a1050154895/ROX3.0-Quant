#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据库模块
整合所有数据库操作

模块结构：
- connection: 连接池管理
- users: 用户管理
- trades: 交易记录
- watchlist: 自选股
- alerts: 预警系统
- strategies: 策略存储
"""

from app.db import (
    # 连接管理
    get_conn,
    release_conn,
    get_db,
    get_db_context,
    get_pool_stats,
    ensure_schema,
    init_db,
    # 用户管理
    create_user,
    get_user_by_username,
    get_user_by_id,
    # 交易管理
    create_trade,
    get_open_trades_with_risk,
    close_trade,
    get_trades,
    get_history,
    add_history,
    clear_history,
    # 自选股
    get_watchlist,
    add_watchlist,
    remove_watchlist,
    # 预警系统
    create_alert,
    list_alerts,
    delete_alert,
    get_pending_alerts,
    mark_alert_triggered,
    # 条件单
    create_condition_order,
    cancel_condition_order,
    get_pending_condition_orders,
    fill_condition_order,
    # 策略存储
    create_visual_strategy,
    get_visual_strategy,
    get_all_visual_strategies,
    update_visual_strategy,
    delete_visual_strategy,
    # Prompt模板
    list_prompt_templates,
    get_prompt_template,
    save_prompt_template,
    # 行情缓存
    get_realtime_quotes_sina,
    clear_spot_cache,
)

__all__ = [
    # 连接管理
    "get_conn",
    "release_conn",
    "get_db",
    "get_db_context",
    "get_pool_stats",
    "ensure_schema",
    "init_db",
    # 用户管理
    "create_user",
    "get_user_by_username",
    "get_user_by_id",
    # 交易管理
    "create_trade",
    "get_open_trades_with_risk",
    "close_trade",
    "get_trades",
    "get_history",
    "add_history",
    "clear_history",
    # 自选股
    "get_watchlist",
    "add_watchlist",
    "remove_watchlist",
    # 预警系统
    "create_alert",
    "list_alerts",
    "delete_alert",
    "get_pending_alerts",
    "mark_alert_triggered",
    # 条件单
    "create_condition_order",
    "cancel_condition_order",
    "get_pending_condition_orders",
    "fill_condition_order",
    # 策略存储
    "create_visual_strategy",
    "get_visual_strategy",
    "get_all_visual_strategies",
    "update_visual_strategy",
    "delete_visual_strategy",
    # Prompt模板
    "list_prompt_templates",
    "get_prompt_template",
    "save_prompt_template",
    # 行情缓存
    "get_realtime_quotes_sina",
    "clear_spot_cache",
]
