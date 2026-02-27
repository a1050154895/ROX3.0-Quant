#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试akshare库是否能正常获取市场数据
"""

import akshare as ak

print('Testing akshare...')
try:
    # 测试获取A股实时行情
    df = ak.stock_zh_a_spot_em()
    print(f'Success! Got {len(df)} stocks')
    print('Sample data:')
    print(df.head())
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
