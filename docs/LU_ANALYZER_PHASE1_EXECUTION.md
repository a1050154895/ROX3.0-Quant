# 卢式分析器升级 - 第三阶段（P2）实施说明

## 本阶段目标
1. 建立市场状态识别（牛/熊/震荡）能力。
2. 引入自适应权重，按市场状态与风险偏好动态调整。
3. 提供候选池扫描与排序能力，支持组合级选股入口。

## 已实施内容
- 协议版本升级到 `lu-analyzer-v3`。
- `predict-v2` 输出新增：
  - `analysis.market_regime`
  - `analysis.adaptive_weights`
- 新增 `POST /api/lu-prediction/scan-v2`：
  - 输入代码列表
  - 输出全量排序和 TopN leaders
- 保留兼容层：`GET /predict/{code}` 不破坏旧调用。

## Phase-3 能力点
- 市场状态识别：`bull / bear / range`
- 自适应权重：结合 `regime + risk_profile`
- 排行扫描：以 `composite_score` 排序输出

## 下一步建议（可继续做）
1. 引入真实宏观危机概率输入并接入时序存储。
2. 增加组合回测和择时切换收益统计。
3. 增加前端雷达图与驱动因子可视化。
