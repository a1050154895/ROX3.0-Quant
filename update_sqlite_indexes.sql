-- SQLite 数据库索引优化脚本

-- 为 marketplace_items 表添加索引
CREATE INDEX IF NOT EXISTS idx_marketplace_items_category ON marketplace_items(category);
CREATE INDEX IF NOT EXISTS idx_marketplace_items_rating ON marketplace_items(rating);
CREATE INDEX IF NOT EXISTS idx_marketplace_items_created_at ON marketplace_items(created_at);

-- 为 marketplace_comments 表添加索引
CREATE INDEX IF NOT EXISTS idx_marketplace_comments_item_id ON marketplace_comments(item_id);
CREATE INDEX IF NOT EXISTS idx_marketplace_comments_user_id ON marketplace_comments(user_id);
CREATE INDEX IF NOT EXISTS idx_marketplace_comments_created_at ON marketplace_comments(created_at);

-- 为 psychology 表添加索引
CREATE INDEX IF NOT EXISTS idx_psychology_user_id ON psychology(user_id);
CREATE INDEX IF NOT EXISTS idx_psychology_trade_id ON psychology(trade_id);
CREATE INDEX IF NOT EXISTS idx_psychology_account_type ON psychology(account_type);
CREATE INDEX IF NOT EXISTS idx_psychology_log_time ON psychology(log_time);

-- 为 crypto_spot 表添加索引
CREATE INDEX IF NOT EXISTS idx_crypto_spot_price ON crypto_spot(price);
CREATE INDEX IF NOT EXISTS idx_crypto_spot_change_24h ON crypto_spot(change_24h);
CREATE INDEX IF NOT EXISTS idx_crypto_spot_updated_at ON crypto_spot(updated_at);

-- 为 global_spot 表添加索引
CREATE INDEX IF NOT EXISTS idx_global_spot_price ON global_spot(price);
CREATE INDEX IF NOT EXISTS idx_global_spot_change_pct ON global_spot(change_pct);
CREATE INDEX IF NOT EXISTS idx_global_spot_updated_at ON global_spot(updated_at);

-- 为 users 表添加索引
CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);
CREATE INDEX IF NOT EXISTS idx_users_created_at ON users(created_at);

-- 为 accounts 表添加索引
CREATE INDEX IF NOT EXISTS idx_accounts_balance ON accounts(balance);
CREATE INDEX IF NOT EXISTS idx_accounts_total_assets ON accounts(total_assets);

-- 为 positions 表添加索引
CREATE INDEX IF NOT EXISTS idx_positions_market_value ON positions(market_value);
CREATE INDEX IF NOT EXISTS idx_positions_unrealized_pnl ON positions(unrealized_pnl);
CREATE INDEX IF NOT EXISTS idx_positions_updated_at ON positions(updated_at);

-- 为 trades 表添加索引
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_side ON trades(side);
CREATE INDEX IF NOT EXISTS idx_trades_open_time ON trades(open_time);
CREATE INDEX IF NOT EXISTS idx_trades_close_time ON trades(close_time);

-- 为 history 表添加索引
CREATE INDEX IF NOT EXISTS idx_history_stock_code ON history(stock_code);
CREATE INDEX IF NOT EXISTS idx_history_sector ON history(sector);
CREATE INDEX IF NOT EXISTS idx_history_rating ON history(rating);

-- 为 watchlist 表添加索引
CREATE INDEX IF NOT EXISTS idx_watchlist_sector ON watchlist(sector);
CREATE INDEX IF NOT EXISTS idx_watchlist_added_at ON watchlist(added_at);

-- 为 condition_orders 表添加索引
CREATE INDEX IF NOT EXISTS idx_condition_orders_symbol ON condition_orders(symbol);
CREATE INDEX IF NOT EXISTS idx_condition_orders_side ON condition_orders(side);
CREATE INDEX IF NOT EXISTS idx_condition_orders_trigger_type ON condition_orders(trigger_type);
CREATE INDEX IF NOT EXISTS idx_condition_orders_created_at ON condition_orders(created_at);

-- 为 alerts 表添加索引
CREATE INDEX IF NOT EXISTS idx_alerts_symbol ON alerts(symbol);
CREATE INDEX IF NOT EXISTS idx_alerts_alert_type ON alerts(alert_type);
CREATE INDEX IF NOT EXISTS idx_alerts_created_at ON alerts(created_at);

-- 查看所有索引
SELECT name FROM sqlite_master WHERE type='index' ORDER BY name;