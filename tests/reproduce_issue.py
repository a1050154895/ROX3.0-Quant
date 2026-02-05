
import asyncio
import logging
import sys
import os
import pandas as pd
import time

# Setup path and logging
sys.path.insert(0, os.path.abspath('.'))
# Force disable proxy
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""
os.environ["all_proxy"] = ""
os.environ["no_proxy"] = "*"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_diagnose(code="600519"):
    try:
        from app.api.endpoints.stock import get_stock_diagnose, _get_stock_basic_info
        import akshare as ak
        
        logger.info(f"Testing diagnose for {code}...")
        
        # 1. Test basic info
        t0 = time.time()
        logger.info("Step 1: _get_stock_basic_info")
        try:
            info = await asyncio.wait_for(_get_stock_basic_info(code), timeout=5.0)
            logger.info(f"Step 1 done in {time.time()-t0:.2f}s: {info}")
        except Exception as e:
            logger.error(f"Step 1 failed: {e}")

        # 2. Test hist data (Direct AkShare)
        t0 = time.time()
        logger.info("Step 2: ak.stock_zh_a_hist")
        try:
            loop = asyncio.get_event_loop()
            res = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: ak.stock_zh_a_hist(symbol=code, period="daily", start_date="20230101", end_date="20231231", adjust="qfq")),
                timeout=10.0
            )
            logger.info(f"Step 2 done in {time.time()-t0:.2f}s: {len(res) if res is not None else 'None'} rows")
        except Exception as e:
             logger.error(f"Step 2 failed: {e}")

        # 3. Full diagnose
        t0 = time.time()
        logger.info("Step 3: get_stock_diagnose (full)")
        result = await asyncio.wait_for(get_stock_diagnose(code), timeout=20.0)
        logger.info(f"Diagnose Result: Score={result.get('overall_score')}, Summary={result.get('summary')}")
        
    except Exception as e:
        logger.error(f"Diagnose Failed: {e}", exc_info=True)

async def test_screen():
    try:
        from app.api.endpoints.strategy import api_screen_xunlongjue
        logger.info("Testing xunlongjue screen...")
        # Force using default pool
        result = await asyncio.wait_for(api_screen_xunlongjue(codes=None, max_codes=5), timeout=20.0)
        items = result.get("items", [])
        logger.info(f"Screen Result: Found {len(items)} items.")
        if not items:
            logger.warning("No items found. Logic might be too strict.")
        else:
            for item in items[:3]:
                logger.info(f"Item: {item}")
    except Exception as e:
        logger.error(f"Screen Failed: {e}", exc_info=True)

async def main():
    await test_diagnose()
    await test_screen()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
