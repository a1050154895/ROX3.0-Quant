"""
DataFrame 通用工具函数

功能:
1. 模糊匹配列名
2. 安全类型转换
3. 数据清洗

解决问题:
- 消除代码中 30+ 处重复的列名查找逻辑
- 统一数值转换和 NaN 处理
"""
import pandas as pd
from typing import List, Optional, Any, Union
import re


def find_column(
    df: pd.DataFrame,
    patterns: Union[str, List[str]],
    default: Optional[str] = None,
    exact: bool = False
) -> Optional[str]:
    """
    模糊匹配 DataFrame 列名
    
    Args:
        df: 目标 DataFrame
        patterns: 匹配模式（字符串或字符串列表）
        default: 未找到时的默认值
        exact: 是否精确匹配
    
    Returns:
        匹配到的列名，或 default
        
    Example:
        # 查找涨跌幅列
        pct_col = find_column(df, ['涨跌幅', '涨幅', 'pct_chg'], default='涨跌幅')
        
        # 查找名称列
        name_col = find_column(df, '名称')
    """
    if df is None or df.empty:
        return default
    
    if isinstance(patterns, str):
        patterns = [patterns]
    
    cols = df.columns.tolist()
    
    for pattern in patterns:
        for col in cols:
            if exact:
                if col == pattern:
                    return col
            else:
                if pattern in col:
                    return col
    
    return default


def find_columns(
    df: pd.DataFrame,
    mappings: dict
) -> dict:
    """
    批量模糊匹配列名
    
    Args:
        df: 目标 DataFrame
        mappings: {目标名: [匹配模式列表]} 或 {目标名: 匹配模式}
    
    Returns:
        {目标名: 实际列名}
        
    Example:
        cols = find_columns(df, {
            'code': ['代码', '证券代码', 'code'],
            'name': ['名称', '简称', 'name'],
            'price': ['最新价', '现价', 'close'],
        })
    """
    result = {}
    for key, patterns in mappings.items():
        result[key] = find_column(df, patterns)
    return result


def safe_float(value: Any, default: float = 0.0) -> float:
    """
    安全转换为浮点数
    
    Args:
        value: 要转换的值
        default: 转换失败时的默认值
    
    Returns:
        浮点数
        
    Example:
        price = safe_float(row['最新价'])
        price = safe_float('12.34%'.replace('%', ''))
    """
    if value is None:
        return default
    
    if isinstance(value, (int, float)):
        # 处理 NaN
        if value != value:  # NaN check
            return default
        return float(value)
    
    try:
        # 移除常见后缀
        s = str(value).strip()
        s = s.replace('%', '').replace('亿', '').replace('万', '')
        s = s.replace(',', '')
        
        if not s or s == '-' or s == '--':
            return default
        
        return float(s)
    except (ValueError, TypeError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """安全转换为整数"""
    return int(safe_float(value, float(default)))


def normalize_stock_code(code: Any) -> str:
    """
    标准化股票代码为6位数字
    
    Args:
        code: 原始代码（可能带前缀如 sh600519）
    
    Returns:
        6位纯数字代码
        
    Example:
        normalize_stock_code('sh600519')  # '600519'
        normalize_stock_code('600519')    # '600519'
        normalize_stock_code(600519)      # '600519'
    """
    if code is None:
        return ''
    
    s = str(code).strip()
    
    # 移除常见前缀
    for prefix in ['sh', 'sz', 'SH', 'SZ']:
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    
    # 移除后缀
    for suffix in ['.SH', '.SZ', '.sh', '.sz']:
        if s.endswith(suffix):
            s = s[:-len(suffix)]
            break
    
    # 提取数字部分
    match = re.search(r'(\d{6})', s)
    if match:
        return match.group(1)
    
    # 尝试补零
    if s.isdigit():
        return s.zfill(6)[:6]
    
    return s


def validate_stock_code(code: str) -> bool:
    """
    验证股票代码是否有效
    
    Args:
        code: 股票代码
    
    Returns:
        是否有效
    """
    if not code:
        return False
    
    code = normalize_stock_code(code)
    
    if not re.match(r'^\d{6}$', code):
        return False
    
    # 基本规则检查
    # 沪市: 60xxxx, 68xxxx (科创板)
    # 深市: 00xxxx, 30xxxx (创业板)
    valid_prefixes = ('60', '68', '00', '30')
    return code[:2] in valid_prefixes


def clean_numeric_column(
    df: pd.DataFrame,
    column: str,
    default: float = 0.0
) -> pd.DataFrame:
    """
    清洗 DataFrame 的数值列
    
    Args:
        df: 目标 DataFrame
        column: 列名
        default: 无效值的默认值
    
    Returns:
        处理后的 DataFrame (原地修改)
    """
    if column not in df.columns:
        return df
    
    df[column] = df[column].apply(lambda x: safe_float(x, default))
    return df


def stock_code_prefix(code: str) -> str:
    """
    根据股票代码返回交易所前缀
    
    Args:
        code: 6位股票代码
    
    Returns:
        'sh' 或 'sz'
    """
    code = normalize_stock_code(code)
    
    # 沪市
    if code.startswith(('6', '5', '9')) or code.startswith('688'):
        return 'sh'
    
    # 深市
    return 'sz'


def format_market_cap(value: float) -> str:
    """
    格式化市值显示
    
    Args:
        value: 市值（元）
    
    Returns:
        格式化字符串
        
    Example:
        format_market_cap(12345678900)  # '123.46亿'
    """
    if value is None or value <= 0:
        return '--'
    
    if value >= 1e12:
        return f'{value / 1e12:.2f}万亿'
    elif value >= 1e8:
        return f'{value / 1e8:.2f}亿'
    elif value >= 1e4:
        return f'{value / 1e4:.2f}万'
    else:
        return f'{value:.2f}'
