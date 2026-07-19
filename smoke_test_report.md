# OPD Process Reward 续写模型对比测试报告

## 1. 实验目的

验证在 OPD 训练流程中，使用**更强的续写模型**（Skywork-OR1-7B，7B）与**当前学生模型**（DeepSeek-R1-Distill-Qwen-1.5B，1.5B）作为 process reward 续写模型，得到的 process reward 信号是否存在显著差异。

## 2. 实验设置

- **数据集**：`datasets/sky_candidates.parquet`，随机抽取 100 条。
- **主轨迹生成模型**：`DeepSeek-R1-Distill-Qwen-1.5B`（即当前学生模型）。
- **续写模型**：
  - Skywork-OR1-7B（7B，更强的模型）
  - DeepSeek-R1-Distill-Qwen-1.5B（1.5B，当前模型）
- **Process reward 计算方式**：完全复用 `OPD/on_policy_distillation.sh` 中 `verl/verl/trainer/ppo/opd_consistency.py` 的逻辑。
  - 对主轨迹按 OPD 触发词拆分 segment；
  - 对每个 segment endpoint 构造 prefix prompt；
  - 每个 prefix 用续写模型生成 K=8 个 suffix；
  - 用数学 reward 函数对完整续写打分，得到 endpoint accuracy；
  - 差分得到 process reward，并 merge 同号 segment。
- **关键超参数**：temperature=0.7，top_k=50，max_tokens=300，max_total_length=8192。

## 3. 样本说明

- 总样本数：100
- 主轨迹中**没有任何 segment** 的样本：31 / 100（原因：答案未出现在主轨迹、无法拆分出 segment、或 ground_truth 为空）
- 本次分析**已剔除**这 31 条无意义样本，仅对剩余 **69 条**进行续写模型对比。

## 4. 主要结果

### 4.1 整体差异

| 指标 | Skywork-OR1-7B | DeepSeek-R1-Distill-Qwen-1.5B | 差异 |
|------|----------------|-------------------------------|------|
| 逐行平均 PR 均值 | 0.672 | 0.574 | **+0.098** |
| 逐行平均 PR 中位数 | 1.000 | 0.333 | — |
| 逐行平均 PR 标准差 | 0.379 | 0.391 | — |
| 逐行平均 PR 相关系数 | — | — | **0.415** |
| 完全相同 PR 的行数 | 25 / 69 | 25 / 69 | **36%** |
| 合并后 segment 数量相同的行数 | 31 / 69 | 31 / 69 | **45%** |
| 逐行平均 PR 差异 > 0.1 的行数 | 32 / 69 | 32 / 69 | **46%** |
| 逐行平均 PR 差异 > 0.3 的行数 | 24 / 69 | 24 / 69 | **35%** |
| 最大逐行差异 | 0.8 | 0.8 | — |

> 注：PR 均值、中位数、标准差、相关系数仅在“有至少一个 segment”的 69 行上计算。

### 4.2 多 segment 情况

- 合并后 segment 数 > 1 的样本：约 26 行（Skywork 统计）。
- 其中两模型都 > 1 segment 的样本：22 行。
- 在这 22 条多 segment 样本中，**只有 2 条**完全相同；其余 20 条均存在差异。
- 两模型合并后 segment 数量不同的行数：38 / 69（55%），说明对同一主轨迹的“过程质量”分段判断也存在明显分歧。

### 4.3 分布图

![Process Reward 对比图](/data/chenyang/OPD/smoke_test_report_figure.png)

图中已在分析前剔除无 segment 样本，包含：
- 左上图：逐行平均 PR 散点图，红线为 y=x；
- 右上图：逐行平均 PR 分布直方图；
- 左下图：两模型差异分布；
- 右下图：合并后 segment 数量分布。

## 5. 结论与讨论

1. **剔除无 segment 样本后，两模型并不完全一致**：仅 36% 的样本给出完全相同 PR，45% 样本 segment 数量相同，最大单条差异达 0.8。这说明“效果差不多”只在粗粒度上成立，细到具体轨迹时存在不少分歧。

2. **Skywork 整体给出的 PR 更高**：逐行平均 PR 均值 Skywork 0.672 vs DeepSeek 0.574，高约 0.10。中位数上 Skywork 为 1.0，DeepSeek 为 0.333。这表明更强的续写模型对复杂推理路径更“乐观”，更容易把中间步骤的续写引向正确答案。

3. **差异集中在多 segment 复杂轨迹**：单 segment 样本（通常是答案直接出现在主轨迹中）两者几乎一致；多 segment 样本中差异显著放大。这说明如果训练数据以简单/短推理为主，换用 7B 续写模型收益很小；但如果数据以长推理为主，奖励信号会明显偏移。

4. **实际训练意义**：如果 OPD 目标是让 process reward 更精细地反映长推理过程的质量，那么 7B 续写模型确实会改变信号；如果只是为了保证最终答案正确率，1.5B 和 7B 的差异较小。是否值得承担 7B 模型的推理开销，需要根据数据里长轨迹的比例来权衡。

## 6. 局限性与后续工作

- 样本量仅 100，且来自 `sky_candidates` 单一数据集；
- 主轨迹由当前 1.5B 模型生成，若主轨迹本身由 7B 模型生成，续写对比可能不同；
- 部分样本 ground_truth 为空，已在分析前剔除；
- 建议后续：
  - 增加样本量；
  - 统计主轨迹的 segment 数量分布，看看长推理样本占比；
  - 人工抽查差异最大的样本，判断 Skywork 的更高 PR 是否更合理。

---

**报告生成时间**：2026-07-08
**脚本位置**：`/data/chenyang/OPD/smoke_test_process_reward.py`
**数据位置**：`/data/chenyang/OPD/smoke_test_pr_results.parquet`
**图表位置**：`/data/chenyang/OPD/smoke_test_report_figure.png`
