-- 更新数据库索引以优化查询性能

-- 为users表添加索引
CREATE INDEX IF NOT EXISTS idx_users_is_active ON users(is_active);
CREATE INDEX IF NOT EXISTS idx_users_is_superuser ON users(is_superuser);

-- 为portfolios表添加索引
CREATE INDEX IF NOT EXISTS idx_portfolios_total_value ON portfolios(total_value);
CREATE INDEX IF NOT EXISTS idx_portfolios_total_return ON portfolios(total_return);

-- 为stocks表添加索引
CREATE INDEX IF NOT EXISTS idx_stocks_market ON stocks(market);
CREATE INDEX IF NOT EXISTS idx_stocks_current_price ON stocks(current_price);

-- 为portfolio_holdings表添加索引
CREATE INDEX IF NOT EXISTS idx_portfolio_holdings_return_rate ON portfolio_holdings(return_rate);
CREATE INDEX IF NOT EXISTS idx_portfolio_holdings_current_value ON portfolio_holdings(current_value);

-- 为transactions表添加索引
CREATE INDEX IF NOT EXISTS idx_transactions_type ON transactions(transaction_type);
CREATE INDEX IF NOT EXISTS idx_transactions_amount ON transactions(total_amount);

-- 为market_data表添加复合索引
CREATE INDEX IF NOT EXISTS idx_market_data_stock_date ON market_data(stock_id, date);
CREATE INDEX IF NOT EXISTS idx_market_data_close_price ON market_data(close_price);

-- 为strategies表添加索引
CREATE INDEX IF NOT EXISTS idx_strategies_is_active ON strategies(is_active);
CREATE INDEX IF NOT EXISTS idx_strategies_type ON strategies(strategy_type);

-- 为ai_conversations表添加索引
CREATE INDEX IF NOT EXISTS idx_ai_conversations_type ON ai_conversations(conversation_type);
CREATE INDEX IF NOT EXISTS idx_ai_conversations_created_at ON ai_conversations(created_at);

-- 为knowledge_base表添加全文索引
CREATE INDEX IF NOT EXISTS idx_knowledge_base_title ON knowledge_base USING gin(to_tsvector('chinese', title));
CREATE INDEX IF NOT EXISTS idx_knowledge_base_content ON knowledge_base USING gin(to_tsvector('chinese', content));

-- 优化现有索引
-- 重新创建一些索引以提高性能
DROP INDEX IF EXISTS idx_market_data_stock_id;
CREATE INDEX IF NOT EXISTS idx_market_data_stock_id ON market_data(stock_id);

DROP INDEX IF EXISTS idx_transactions_portfolio_id;
CREATE INDEX IF NOT EXISTS idx_transactions_portfolio_id ON transactions(portfolio_id);

-- 查看所有索引
SELECT tablename, indexname, indexdef 
FROM pg_indexes 
WHERE schemaname = 'public' 
ORDER BY tablename, indexname;