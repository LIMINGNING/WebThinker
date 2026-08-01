# V2-C 风险分级证据优化全量实验报告

## 1. 实验目的

V2-C 聚焦开放域问答智能体的证据处理问题。在不增加原始搜索预算的前提下，辅助模型维护结构化证据状态，程序区分阻断性证据缺口与非阻断性验证风险，主模型仍根据自然语言证据摘要完成推理和作答。

本次实验在 GAIA `dev` 全量 103 题上运行，用于检验 V2-C 相对网页抓取修复基线的回答质量变化及其检索成本。

## 2. V2-C 方法

1. 辅助模型在内部维护原子证据，包括主体、关系、值、时间范围和来源。
2. 主模型只接收辅助模型生成的自然语言证据摘要和验证提示，不直接接收证据 JSON。
3. 将证据问题分为两档：阻断性缺口触发补搜；软验证风险最多触发一次核验。
4. `complete` 只是辅助模型的停止建议。主模型若仍提出合法且未执行的查询，可推翻该建议并继续搜索。
5. 对比较、计数、集合差和共享时间截止等问题增加通用完整性校验，不针对具体题目或实体硬编码。
6. 所有主模型搜索和证据补搜共同计入 `max_search_limit=15`，不获得额外预算；受原框架计数逻辑影响，单题实际最多执行 14 次搜索。
7. 到达预算后不放弃作答，而是要求主模型依据当前最佳证据输出答案。

## 3. 实验配置

| 配置项 | 基线 | V2-C |
| --- | --- | --- |
| 数据集 | GAIA dev，103 题 | GAIA dev，103 题 |
| 主模型 | deepseek-v4-pro，non-thinking | deepseek-v4-pro，non-thinking |
| 辅助模型 | deepseek-v4-pro，non-thinking | deepseek-v4-pro，non-thinking |
| 搜索引擎 | Serper | Serper |
| Top-K | 5 | 5 |
| 最大搜索限制 | 15 | 15 |
| 最大生成 token | 4096 | 4096 |
| 整题总超时 | 关闭 | 关闭 |
| Benchmark leakage filter | 开启 | 开启 |
| 证据完整性优化 | 关闭 | V2-C |

两组核心运行参数一致，但使用不同日期、不同空缓存运行实时网页搜索。因此该对比适合说明当前实现的阶段效果，不能等同于冻结检索结果后的严格可重复对照实验。

## 4. 全量结果

| 指标 | 基线 | V2-C | 变化 |
| --- | ---: | ---: | ---: |
| EM | 45.63% | 54.37% | +8.74 个百分点 |
| ACC | 50.49% | 57.28% | +6.80 个百分点 |
| F1 | 50.86% | 57.11% | +6.25 个百分点 |
| Math Equal | 46.60% | 55.34% | +8.74 个百分点 |
| 有效答案 | 94/103 | 101/103 | +7 |
| 总搜索次数 | 462 | 587 | +125（+27.06%） |
| 平均每题搜索次数 | 4.49 | 5.70 | +1.21 |
| 达到实际 14 次上限的题目 | 15 | 13 | -2 |
| 总运行时间 | 3:28:28 | 7:06:43 | +3:38:15 |

V2-C 的三个难度等级 ACC 均有提升：Level 1 从 64.10% 提升到 71.79%，Level 2 从 44.23% 提升到 50.00%，Level 3 从 33.33% 提升到 41.67%。

## 5. 逐题变化

按 ACC 判断，V2-C 有 15 题由错变对，8 题由对变错，净提升 7 题。

- 错变对：`9, 13, 27, 38, 50, 67, 68, 81, 102, 111, 130, 134, 139, 156, 162`
- 对变错：`2, 6, 12, 23, 43, 85, 118, 145`

按 EM 判断，V2-C 有 16 题由错变对，7 题由对变错，净提升 9 题。

- 错变对：`9, 13, 16, 27, 38, 50, 67, 68, 81, 100, 102, 130, 134, 139, 156, 162`
- 对变错：`2, 6, 12, 23, 43, 85, 118`

重点回归题在本次全量运行中均答对：ID 42 输出 `Harbinger, Tidal`，ID 136 输出 `519`，ID 151 输出 `21`。

## 6. 控制器行为

V2-C 最终证据状态为：`complete` 31 题、`partial` 44 题、`unknown` 28 题。控制器执行了 82 次阻断性证据补搜和 3 次软风险核验；11 题在预算或流程条件限制下进入“依据当前最佳证据作答”。

泄露过滤器在 V2-C 日志中记录了 27 题、79 条被过滤搜索结果。基线 manifest 同样显示过滤器开启，但旧日志未记录逐条过滤结果，因此不能根据日志字段为零推断基线未执行过滤。

## 7. 结论与限制

本次全量实验表明，V2-C 在有效答案率和回答质量上均优于参考基线，说明风险分级证据完整性控制具有继续研究的价值。提升并非免费获得：总搜索次数增加 27.06%，总运行时间约翻倍。当前结果支持“提高证据完整性可以改善回答质量”，暂不支持“同时提高系统效率”。

后续论文实验应至少补充：固定缓存或同批搜索结果下的成对实验、多个随机运行的均值和方差、V2-C 各组成部分的消融实验，以及 8 个由对变错案例的归因分析。优先消融阻断性补搜、软风险核验、共享时间范围校验和预算耗尽后的强制作答。

## 8. 产物位置

- V2-C 完整日志：`outputs/gaia.deepseek-v4-pro.evidence-completeness-v2c-risk-tiers.aux-summary.leak-filtered.non-thinking.webthinker/dev.8.1,2_40.41.json`
- V2-C 逐题评分：`outputs/gaia.deepseek-v4-pro.evidence-completeness-v2c-risk-tiers.aux-summary.leak-filtered.non-thinking.webthinker/dev.8.1,2_40.41.metrics.json`
- V2-C 汇总评分：`outputs/gaia.deepseek-v4-pro.evidence-completeness-v2c-risk-tiers.aux-summary.leak-filtered.non-thinking.webthinker/dev.8.1,2_40.41.metrics.overall.json`
- V2-C 运行清单：`outputs/gaia.deepseek-v4-pro.evidence-completeness-v2c-risk-tiers.aux-summary.leak-filtered.non-thinking.webthinker/dev.8.1,2_40.41.manifest.json`
- 参考基线汇总评分：`outputs/gaia.deepseek-v4-pro.leak-filtered.non-thinking.webthinker/dev.7.30,5_37.41.metrics.overall.json`
- V2-C 实现分支：`experiment/evidence-completeness-v2c-risk-tiers`
- 本次全量代码版本：`21bff9d88c823945c38f8a8334c01dbd78f8f6ad`
