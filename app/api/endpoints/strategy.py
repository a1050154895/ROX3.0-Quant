from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import pandas as pd
from datetime import datetime, timedelta
import asyncio
import logging
from app.rox_quant.backtest_engine import BacktestEngine
from app.rox_quant.performance_metrics import PerformanceMetrics
from app.rox_quant.strategies.classic_cta import CTAStrategies
from app.services.data_fetcher import DataFetcher
from app.analysis.xunlongjue import xunlongjue_signal

# Import new engine components
from app.quant.engine import QuantEngine
from app.quant.data_provider import get_data_provider
from app.strategies import ai_demo
from app.rox_quant.strategies.ranking_strategy import RankingStrategy
from app.rox_quant.strategies.jq_10y_52x import JQTenYearFiftyTwoTimes
from app.rox_quant.strategies.jq_small_cap import JQSmallCap
from app.rox_quant.strategies.jq_etf_rotation import JQETFRotation
from app.rox_quant.strategies.jq_dragon import JQDragonTrend
from app.rox_quant.strategies.book_small_cap_timing import BookSmallCapTiming
from app.rox_quant.strategies.book_dual_thrust import BookDualThrust
from app.rox_quant.strategies.book_double_ma import BookDoubleMA
from app.rox_quant.strategies.book_turtle import BookTurtle

router = APIRouter()
data_fetcher = DataFetcher()
logger = logging.getLogger(__name__)

STRATEGY_MAP = {
    "jq_10y_52x": JQTenYearFiftyTwoTimes,
    "jq_small_cap": JQSmallCap,
    "jq_etf_rotation": JQETFRotation,
    "jq_dragon": JQDragonTrend,
    "book_small_cap_timing": BookSmallCapTiming,
    "book_dual_thrust": BookDualThrust,
    "book_double_ma": BookDoubleMA,
    "book_turtle": BookTurtle
}

@router.get("/backtest/strategies")
async def get_strategies():
    """
    List available JoinQuant strategies
    """
    return {
        "strategies": [
            {"id": "jq_10y_52x", "name": "10年52倍 (ARBR)", "description": "Weekly rebalance based on ARBR sentiment factor"},
            {"id": "jq_small_cap", "name": "Small Cap Rotation", "description": "Low price/Small cap rotation with trend filter"},
            {"id": "jq_etf_rotation", "name": "ETF Momentum", "description": "Multi-asset ETF rotation based on R-Squared Momentum"},
            {"id": "jq_dragon", "name": "Dragon Trend", "description": "Follow stocks with consecutive Limit Ups"},
            {"id": "book_small_cap_timing", "name": "Small Cap + 2-8 Timing", "description": "A-Share Classic: Small Cap Selection with Index Timing Filter"},
            {"id": "book_dual_thrust", "name": "Dual Thrust", "description": "Classic CTA: Range Breakout Strategy"},
            {"id": "book_double_ma", "name": "Double Moving Average", "description": "Classic Trend: Golden Cross / Death Cross"},
            {"id": "book_turtle", "name": "Turtle Trading", "description": "Classic Trend: Donchian Channel Breakout"}
        ]
    }

class PythonExecRequest(BaseModel):
    code: str
    stock_code: str = "600519"

class BacktestRequest(BaseModel):
    start_date: str
    end_date: str
    capital: float = 100000.0
    stock_pool: Optional[List[str]] = None

@router.post("/backtest/jq_strategy/{strategy_id}")
async def backtest_jq_strategy(strategy_id: str, req: BacktestRequest):
    """
    Generic Backtest for JoinQuant Strategies
    """
    if strategy_id not in STRATEGY_MAP:
        raise HTTPException(status_code=404, detail=f"Strategy {strategy_id} not found")
        
    try:
        StrategyClass = STRATEGY_MAP[strategy_id]
        # Initialize with custom pool if provided
        if req.stock_pool:
            strategy = StrategyClass(stock_pool=req.stock_pool)
        else:
            strategy = StrategyClass()
            
        results = strategy.run(
            graph_json="", 
            start_date=req.start_date, 
            end_date=req.end_date, 
            initial_capital=req.capital
        )
        return {
            "status": "success",
            "strategy": strategy_id,
            "results": results,
            "metrics": results.get("metrics", {})
        }
    except Exception as e:
        logger.error(f"JQ Strategy Backtest failed: {e}")
        return {"status": "error", "error": str(e)}

@router.get("/backtest/jq_strategy/{strategy_id}")
async def backtest_jq_strategy_get(
    strategy_id: str,
    start_date: str = "2024-01-01", 
    end_date: str = "2024-04-01", 
    initial_capital: float = 1000000.0,
    stock_pool: Optional[str] = None # Comma separated
):
    """
    Generic Backtest for JoinQuant Strategies (GET version)
    """
    if strategy_id not in STRATEGY_MAP:
        raise HTTPException(status_code=404, detail=f"Strategy {strategy_id} not found")
        
    try:
        StrategyClass = STRATEGY_MAP[strategy_id]
        pool_list = stock_pool.split(",") if stock_pool else None
        
        if pool_list:
            strategy = StrategyClass(stock_pool=pool_list)
        else:
            strategy = StrategyClass()
            
        results = strategy.run(
            graph_json="", 
            start_date=start_date, 
            end_date=end_date, 
            initial_capital=initial_capital
        )
        
        # Format for frontend compatibility if needed
        equity_data = []
        if "equity_curve" in results and "dates" in results:
            dates = results["dates"]
            values = results["equity_curve"]
            min_len = min(len(dates), len(values))
            for i in range(min_len):
                equity_data.append({"date": dates[i], "equity": values[i]})
        
        return {
            "summary": f"Strategy {strategy_id} Backtest Complete",
            "metrics": results.get("metrics", {}),
            "equity": equity_data,
            "raw_results": results
        }
    except Exception as e:
        logger.error(f"JQ Strategy Backtest failed: {e}")
        return {"error": str(e)}


@router.post("/backtest/ai_qbot")
async def backtest_ai_qbot(req: BacktestRequest):
    """
    AI Qbot Backtest Endpoint
    """
    try:
        # Mock backtest logic for now, or connect to actual backtest engine
        # In a real scenario, this would call BacktestEngine
        
        # Generate some mock history data
        start = pd.to_datetime(req.start_date)
        end = pd.to_datetime(req.end_date)
        dates = pd.date_range(start=start, end=end, freq='B') # Business days
        
        history = []
        val = req.capital
        import random
        
        for d in dates:
            change = random.uniform(-0.02, 0.025) # -2% to +2.5%
            val = val * (1 + change)
            history.append({
                "date": d.strftime("%Y-%m-%d"),
                "portfolio_value": round(val, 2)
            })
            
        return {
            "status": "success",
            "history": history,
            "final_value": round(val, 2),
            "return_pct": round((val - req.capital) / req.capital * 100, 2)
        }
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        return {"status": "error", "error": str(e)}

@router.get("/screen")
async def api_screen_xunlongjue(
    codes: str = Query(None, description="股票代码，逗号分隔，如 600519,000001。不传则使用默认池"),
    max_codes: int = Query(50, ge=1, le=200, description="未传 codes 时从默认池取的数量"),
):
    """
    寻龙诀选股：倍量 + 突破前高 + 涨幅>5% + 涨停 + 昨日未涨停 + 量价配合。
    返回满足条件的代码列表及简要原因。
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=250)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    
    candidates = []

    if codes:
        candidates = [{"code": c.strip(), "name": ""} for c in codes.split(",") if c.strip()]
    else:
        # 使用缓存的全市场实时数据进行预筛选
        from app.db import get_all_stocks_spot
        try:
            df_spot = await get_all_stocks_spot()
            if df_spot is not None and not df_spot.empty:
                # 预筛选：涨幅 > 2.5% (寻龙诀要求 >5% 或涨停，放宽到 2.5% 以防漏网)
                # 且量比 > 0.8 (如有)
                mask = (pd.to_numeric(df_spot['涨跌幅'], errors='coerce') > 2.5)
                
                # 如果有量比列，也筛选一下
                if '量比' in df_spot.columns:
                     mask = mask & (pd.to_numeric(df_spot['量比'], errors='coerce') > 0.8)
                
                filtered_df = df_spot[mask].copy()
                
                # 按涨幅排序，取前 max_codes
                if not filtered_df.empty:
                    # 优先看涨幅大的
                    filtered_df = filtered_df.sort_values(by='涨跌幅', ascending=False).head(max_codes)
                    
                    for _, row in filtered_df.iterrows():
                        candidates.append({
                            "code": str(row['代码']),
                            "name": str(row['名称'])
                        })
            
            # 如果预筛选没结果（大盘全跌），或者 API 失败，使用备选池
            if not candidates:
                 logger.warning("预筛选未找到符合条件的股票，使用默认热门股")
                 static_codes = [
                    "600519", "000001", "600036", "601318", "000858", "300750", "601899", "002594", 
                    "300059", "601012", "600900", "000002", "601166", "600030", "600887", "002475"
                 ]
                 candidates = [{"code": c, "name": ""} for c in static_codes]
                 
        except Exception as e:
            logger.warning("获取默认股票池失败: %s", e)
            candidates = [{"code": "600519", "name": "贵州茅台"}]

    results = []
    
    # 并发获取 K 线并计算
    async def _process(item):
        code = item['code']
        try:
            df = await data_fetcher.get_daily_kline(code, start_str, end_str)
            if df is None or df.empty or len(df) < 60:
                return None
            r = xunlongjue_signal(df, code=code)
            if r["pass"]:
                return {
                    "code": code,
                    "name": item['name'],
                    "reason": r["reason"],
                    "detail": r["detail"],
                }
        except Exception as e:
            logger.debug("寻龙诀 %s 计算异常: %s", code, e)
        return None

    tasks = [_process(c) for c in candidates]
    if tasks:
        batch_results = await asyncio.gather(*tasks)
        results = [r for r in batch_results if r is not None]

    # Return results directly. If empty, frontend should handle it.
    
    return {"items": results, "total": len(results)}


class ScreenWithAIRequest(BaseModel):
    screen_type: str = "xunlongjue"
    params: Optional[dict] = None
    max_results: int = 20
    provider: Optional[str] = None
    model: Optional[str] = None


@router.post("/screen-with-ai")
async def api_screen_with_ai(req: ScreenWithAIRequest):
    """
    条件选股 + AI 总结（参考 go-stock AI 智能选股）。
    先执行选股（如寻龙诀），再调用 AI 对结果做 2～4 句话总结。
    """
    params = req.params or {}
    codes = params.get("codes", "")
    max_codes = params.get("max_codes", 30)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=250)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    if codes:
        candidates = [{"code": c.strip(), "name": ""} for c in str(codes).split(",") if c.strip()]
    else:
        # 使用缓存的全市场实时数据进行预筛选
        from app.db import get_all_stocks_spot
        candidates = []
        try:
            df_spot = await get_all_stocks_spot()
            if df_spot is not None and not df_spot.empty:
                mask = (pd.to_numeric(df_spot['涨跌幅'], errors='coerce') > 2.5)
                if '量比' in df_spot.columns:
                     mask = mask & (pd.to_numeric(df_spot['量比'], errors='coerce') > 0.8)
                
                filtered_df = df_spot[mask].copy()
                if not filtered_df.empty:
                    filtered_df = filtered_df.sort_values(by='涨跌幅', ascending=False).head(max_codes)
                    for _, row in filtered_df.iterrows():
                        candidates.append({
                            "code": str(row['代码']),
                            "name": str(row['名称'])
                        })
            
            if not candidates:
                 static_codes = ["600519", "000001", "600036", "601318", "000858"]
                 candidates = [{"code": c, "name": ""} for c in static_codes]
        except Exception as e:
            logger.warning("获取默认股票池失败: %s", e)
            candidates = [{"code": "600519", "name": "贵州茅台"}]

    results = []
    
    # 并发获取 K 线数据
    async def _process_code(item):
        code = item['code']
        try:
            df = await data_fetcher.get_daily_kline(code, start_str, end_str)
            if df is None or df.empty or len(df) < 60:
                return None
            r = xunlongjue_signal(df, code=code)
            if r["pass"]:
                return {
                    "code": code,
                    "name": item['name'],
                    "reason": r["reason"],
                    "detail": r["detail"],
                }
        except Exception as e:
            logger.debug("寻龙诀 %s 计算异常: %s", code, e)
        return None

    tasks = [_process_code(c) for c in candidates]
    batch_results = await asyncio.gather(*tasks)
    results = [r for r in batch_results if r is not None]

    # Fallback: 如果未筛出结果，注入演示数据（避免用户以为功能坏了）
    # 但不要强制给茅台，除非真的需要演示
    # if not results:
    #     results.append({
    #         "code": "600519",
    #         "name": "贵州茅台",
    #         "reason": "【示例】倍量突破",
    #         "detail": "当日涨幅 10.0% (涨停)，量比 2.5，突破 30 日新高 (演示数据)"
    #     })

    items = results[: req.max_results]
    ai_summary = ""
    if req.screen_type == "xunlongjue" and items:
        try:
            from app.rox_quant.llm import AIClient
            client = AIClient()
            ai_summary = await client.summarize_screen_results(
                items, max_items=req.max_results, model=req.model, provider=req.provider
            )
        except Exception as e:
            logger.warning("AI 选股总结失败: %s", e)
    return {"items": items, "total": len(results), "ai_summary": ai_summary or ""}


@router.post("/python-exec")
async def python_exec(req: PythonExecRequest):
    """Python 沙箱执行（受限环境，仅允许数据分析和策略计算）

    注意：这不是“完美沙箱”，但会做足够多的防御来避免常见危险用法与报错。
    """
    import io
    import ast
    import traceback
    import builtins as py_builtins
    from contextlib import redirect_stdout, redirect_stderr

    code = (req.code or "").strip()
    if not code:
        return {"error": "代码为空"}
    if len(code) > 10000:
        return {"error": "代码长度不能超过 10000 字符"}

    # 1) AST 级别拦截：禁止 import / 读写文件 / 动态执行等
    banned_call_names = {"eval", "exec", "compile", "open", "__import__", "input"}
    try:
        tree = ast.parse(code, mode="exec")
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                return {"error": "不允许 import / from 导入（沙箱已内置 pd/np/df）"}
            if isinstance(node, ast.Call):
                fn = node.func
                if isinstance(fn, ast.Name) and fn.id in banned_call_names:
                    return {"error": f"不允许调用 {fn.id}()"}
            # 禁止访问 __dunder__ 属性（常见逃逸路径）
            if isinstance(node, ast.Attribute) and isinstance(node.attr, str) and node.attr.startswith("__"):
                return {"error": "不允许访问双下划线属性"}
            if isinstance(node, ast.Name) and isinstance(node.id, str) and node.id.startswith("__"):
                return {"error": "不允许使用双下划线变量"}
    except SyntaxError as e:
        return {"error": f"语法错误: {e}"}

    # 2) 准备数据：尽量提供统一字段
    try:
        df = await data_fetcher.fetch_kline_data(req.stock_code, "2020-01-01", "2024-12-31")
        if df is None or df.empty:
            return {"error": "无法获取数据"}

        # 3) 执行环境
        # 限制 global 环境，只暴露 pandas/numpy/df 等
        import pandas as pd
        import numpy as np
        
        safe_globals = {
            "__builtins__": {
                "range": range,
                "len": len,
                "print": print, # 捕获 stdout
                "list": list,
                "dict": dict,
                "set": set,
                "tuple": tuple,
                "str": str,
                "int": int,
                "float": float,
                "bool": bool,
                "sum": sum,
                "min": min,
                "max": max,
                "abs": abs,
                "round": round,
                "sorted": sorted,
                "enumerate": enumerate,
                "zip": zip,
            },
            "pd": pd,
            "np": np,
            "df": df,
            "results": {},  # 输出结果容器
        }
        
        # 捕获 stdout
        f_stdout = io.StringIO()
        f_stderr = io.StringIO()
        
        # 4) 线程池执行（避免阻塞主循环）
        # 虽然 Python GIL 仍在，但至少不会阻塞 asyncio loop
        loop = asyncio.get_running_loop()
        
        def _safe_exec():
            with redirect_stdout(f_stdout), redirect_stderr(f_stderr):
                exec(code, safe_globals)
                
        await loop.run_in_executor(None, _safe_exec)
        
        # 获取结果
        output = f_stdout.getvalue()
        error = f_stderr.getvalue()
        results = safe_globals.get("results", {})
        
        # 尝试序列化 results (处理 numpy 类型)
        def _serialize(obj):
            if isinstance(obj, (pd.DataFrame, pd.Series)):
                return json.loads(obj.to_json(orient="records" if isinstance(obj, pd.DataFrame) else "index"))
            if isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            if isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            return str(obj)
            
        final_results = {}
        for k, v in results.items():
            try:
                final_results[k] = _serialize(v)
            except Exception as e:
                final_results[k] = f"<Unserializable: {e}>"

        return {
            "status": "ok",
            "output": output,
            "error": error,
            "results": final_results
        }

    except Exception as e:
        logger.error(f"Python exec error: {traceback.format_exc()}")
        return {"error": f"执行异常: {str(e)}"}


class BacktestRequest(BaseModel):
    symbol: str
    start_date: str
    end_date: str
    initial_capital: float

class AIBacktestRequest(BaseModel):
    start_date: str = "2023-01-01"
    end_date: str = "2023-12-31"
    capital: float = 100000.0

class RankingRotationRequest(BaseModel):
    pool: str = "600519,000858,601318,300750,000001,600036,002594,601899,601012,603288" # Comma separated
    start_date: str = "2023-01-01"
    end_date: str = "2023-12-31"
    capital: float = 1000000.0
    top_n: int = 5
    change_num: int = 1

@router.post("/backtest/ranking_rotation")
async def backtest_ranking_rotation(req: RankingRotationRequest):
    """
    BigQuant Style Top-N Ranking & Rotation Strategy (Migrated).
    Logic:
    1. Select stocks from pool.
    2. Rank by simple factor (Momentum 20d) - replacing complex AI model.
    3. Hold Top N.
    4. Rebalance daily (Sell losers, Buy winners).
    """
    try:
        pool_list = [x.strip() for x in req.pool.split(",") if x.strip()]
        if not pool_list:
            return {"status": "error", "message": "Stock pool is empty"}
            
        strategy = RankingStrategy(
            initial_capital=req.capital,
            top_n=req.top_n,
            change_num=req.change_num
        )
        
        result = await asyncio.to_thread(
            strategy.run_backtest, 
            stock_pool=pool_list,
            start_date=req.start_date,
            end_date=req.end_date
        )
        
        if "error" in result:
            return {"status": "error", "message": result["error"]}
            
        return {
            "status": "success",
            "message": "Ranking Rotation Backtest Complete",
            "result": result
        }
    except Exception as e:
        logger.error(f"Ranking Rotation Backtest Failed: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}

@router.post("/backtest/ai_qbot")
async def backtest_ai_qbot(req: AIBacktestRequest):
    """
    Execute AI Strategy (Qbot Style) using EMQuant Data
    """
    try:
        # 1. Setup Provider (Try EMQuant, fallback to Mock)
        provider = get_data_provider(use_mock=False)
        
        # 2. Setup Engine
        engine = QuantEngine(provider)
        # Update capital if needed (Engine defaults to 100000)
        engine.context.portfolio["cash"] = req.capital
        engine.context.portfolio["total_value"] = req.capital
        
        # 3. Load Strategy
        engine.load_strategy(ai_demo)
        
        # 4. Run Backtest
        results = engine.run_backtest(req.start_date, req.end_date)
        
        # 5. Format Results for UI
        # Main.js expects: status, final_portfolio_value, history: [{date, portfolio_value}]
        
        history = []
        for r in results:
            history.append({
                "date": r['date'],
                "portfolio_value": r['value']
            })
        
        final_val = history[-1]['portfolio_value'] if history else req.capital
        
        return {
            "status": "success",
            "message": "AI Strategy Backtest Complete",
            "final_portfolio_value": final_val,
            "history": history
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "status": "error", 
            "message": f"AI Backtest Failed: {str(e)}"
        }

@router.get("/backtest/lu_qiyuan")
async def backtest_lu_qiyuan(
    symbol: str, 
    start_date: str = "2020-01-01", 
    end_date: str = "2023-12-31", 
    initial_capital: float = 100000.0,
    short_window: int = 5, 
    long_window: int = 60, 
    stop_loss_pct: float = 0.07, 
    take_profit_pct: float = 0.20
):
    """
    执行卢麒元的趋势跟踪策略回测
    """
    try:
        # 1. 数据获取
        df = await data_fetcher.get_daily_kline(symbol, start_date, end_date)
        if df.empty:
            raise HTTPException(status_code=404, detail="无法获取该股票代码的K线数据")

        # 2. 策略逻辑
        df['short_ma'] = df['close'].rolling(window=short_window).mean()
        df['long_ma'] = df['close'].rolling(window=long_window).mean()
        
        df['signal'] = 0
        df.loc[df['short_ma'] > df['long_ma'], 'signal'] = 1
        df.loc[df['short_ma'] < df['long_ma'], 'signal'] = -1
        
        df['position'] = df['signal'].shift(1).fillna(0)

        # 3. 回测引擎
        engine = BacktestEngine(
            data=df,
            initial_capital=initial_capital,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct
        )
        equity_curve, trade_logs = engine.run()

        if equity_curve.empty:
            return {"summary": "回测期间无交易信号", "metrics": {}, "equity": [], "logs": []}

        # 4. 性能指标计算
        metrics = PerformanceMetrics(equity_curve, trade_logs)
        
        summary = (
            f"陆奇沅趋势跟踪策略回测完成 ({symbol})。\n"
            f"回测周期: {start_date} 到 {end_date}\n"
            f"最终权益: {equity_curve['equity'].iloc[-1]:.2f}\n"
            f"年化收益率: {metrics.annualized_return():.2%}\n"
            f"最大回撤: {metrics.max_drawdown():.2%}\n"
            f"夏普比率: {metrics.sharpe_ratio():.2f}\n"
            f"胜率: {metrics.win_rate():.2%}"
        )
        
        equity_data = [
            {"date": str(index.date()), "equity": row["equity"]}
            for index, row in equity_curve.iterrows()
        ]

        return {
            "summary": summary,
            "metrics": {
                "final_equity": f"{equity_curve['equity'].iloc[-1]:.2f}",
                "annualized_return": f"{metrics.annualized_return():.2%}",
                "max_drawdown": f"{metrics.max_drawdown():.2%}",
                "sharpe_ratio": f"{metrics.sharpe_ratio():.2f}",
                "win_rate": f"{metrics.win_rate():.2%}"
            },
            "equity": equity_data,
            "logs": sorted(trade_logs, key=lambda x: x['date'])
        }
    except Exception as e:
        return {"error": f"回测失败: {str(e)}"}


@router.get("/backtest/classic_cta")
async def backtest_classic_cta(
    symbol: str,
    start_date: str = "2020-01-01",
    end_date: str = "2023-12-31",
    initial_capital: float = 100000.0,
    strategy: str = "dual_thrust",
    n: int = 4,
    k1: float = 0.5,
    k2: float = 0.5,
    window: int = 20,
    num_std: float = 2.0,
):
    """
    Classic CTA 策略回测（自 Rox 1.0 整合）:
    - dual_thrust: 双均线突破趋势跟踪
    - r_breaker: R-Breaker 枢轴反转
    - bollinger_reversion: 布林带均值回归
    """
    try:
        df = await data_fetcher.get_daily_kline(symbol, start_date, end_date)
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail="无历史数据")

        # DataFetcher 已返回 date, open, close, high, low, volume
        cta = CTAStrategies(initial_capital=initial_capital)

        if strategy == "dual_thrust":
            result = cta.run_dual_thrust(df, n=n, k1=k1, k2=k2)
        elif strategy == "r_breaker":
            result = cta.run_r_breaker(df)
        elif strategy == "bollinger_reversion":
            result = cta.run_bollinger_reversion(df, window=window, num_std=num_std)
        else:
            raise HTTPException(status_code=400, detail="未知策略类型，支持: dual_thrust, r_breaker, bollinger_reversion")

        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"回测失败: {str(e)}")
