# 外部物理数据强基线排行榜（Batch 5）

项目地址：[multimodal-flood-forecasting-BXY](https://github.com/mrrabbitss/multimodal-flood-forecasting-BXY)

## 1. 本轮目标与保护原则

本轮不再单纯增加模型数量，而是建立统一、严格、可解释的外部物理数据排行榜。所有新增工作都位于 `src/external_*.py`、独立实验目录和新版结果目录中：

- 原始 Conv-LSTM 源码、历史 checkpoint 和历史实验结果未被修改或覆盖；
- 六个模型使用相同数据窗口、事件划分、输入信息、预测提前量和评估代码；
- 超参数只在事件隔离的验证集上选择，调参阶段不计算测试指标；
- U-RNN Lite、FNO2D-History、SimVP Lite 均明确标注为 adapted baseline，不冒充论文原版复现；
- 结果同时报告 5 seeds 均值、标准差、逐事件中位数/IQR 和配对事件置信区间。

提交前重新计算的历史 Conv-LSTM checkpoint SHA-256 仍为 `388a5ebd7517a54b2d12dad0a73ede0f6587d9bc8a0c96e91b180507958b598f`，与 P0 审计记录一致。

## 2. 新增模型与工程实现

### 2.1 U-RNN Lite（adapted）

参考[官方 U-RNN 仓库](https://github.com/holmescao/U-RNN)的多尺度递归思想，实现了三尺度 ConvGRU 编码器和 U-Net 式解码器：全分辨率、1/2 分辨率和 1/4 分辨率分别维护时序状态，解码阶段融合多尺度跳连。该实现没有声称逐层复刻论文全部结构。

### 2.2 FNO2D-History（adapted）

将完整 60 分钟历史沿通道维提升为二维场，使用多层 Fourier Neural Operator 同时建模局部卷积和全局频谱关系。实现中解决了两个真实 GPU/AMP 问题：

1. FFT 和复数 `einsum` 在 AMP 下不支持 `ComplexHalf`，因此频谱分支局部固定为 FP32；
2. `GradScaler` 不能直接缩放复数参数梯度，因此频谱权重以两个实数通道存储，前向时再视为复数。

这样既保留频谱运算，又能在 RTX GPU 上稳定使用混合精度训练。

### 2.3 SimVP Lite（adapted）

实现帧级编码器、时空 translator 和解码器。translator 使用多尺度深度卷积与门控残差块，在不引入递归状态的情况下联合建模完整历史。

### 2.4 统一输出约束

六个模型都使用物理水深输出和相同的 state-aware residual 预测：输出头零初始化，因此 epoch 0 精确等价于 persistence。模型只有在验证集上学到有效增量后才会替代初始 checkpoint，避免随机初始化直接破坏合理基线。

## 3. 数据与统一协议

正式 Batch 5 排行榜使用 LarNO UKEA 8 m / 5 min 数据：

| 项目 | 设置 |
|---|---|
| 空间/时间分辨率 | 8 m / 5 min |
| 历史输入 | 12 帧，即过去 60 分钟 |
| 联合预测 | +5、+15、+30、+60 分钟 |
| 雨量协议 | past-only，不向模型提供未来降雨 |
| 输入通道 | 历史水深、当前雨量、3/6 步累计雨量、DEM、不透水面、排水口、有效掩膜 |
| 事件划分 | 6 train / 2 validation / 12 test，事件完全隔离 |
| 样本数 | 234 train / 52 validation / 312 test |
| 评估抽样 | validation/test 固定使用 split seed 44；不随模型 seed 改变 |
| 随机种子 | 42、44、52、77、2026 |
| 最大训练轮数 | 12 epochs |
| 早停 | patience 4，回载最佳验证 checkpoint |
| 洪水阈值 | 主阈值 0.10 m；敏感性 0.05/0.10/0.20/0.30 m |
| 硬件 | NVIDIA GeForce RTX 5060 Laptop GPU |

UKEA 不提供不透水面和排水口图层时，对应通道按适配器协议填零；该限制在所有模型中一致。

## 4. 无测试泄漏的学习率选择

每个模型只在同一验证事件上比较 `1e-4`、`3e-4`、`1e-3` 三个候选。`selection_metrics.json` 明确保存 `test_evaluated: false`。

| 模型 | 选择学习率 | 最佳验证 loss | 验证 MAE (cm) | 验证 CSI | 最佳 epoch |
|---|---:|---:|---:|---:|---:|
| Conv-LSTM | 1e-3 | 0.04179 | 0.699 | 0.8016 | 3 |
| Conv-LSTM + Attention | 3e-4 | 0.04715 | 1.012 | 0.7423 | 7 |
| CNN-Temporal Transformer | 1e-4 | 0.04062 | 0.638 | 0.8080 | 5 |
| U-RNN Lite (adapted) | 1e-4 | 0.03811 | 0.716 | 0.8130 | 5 |
| FNO2D-History (adapted) | 1e-3 | 0.03704 | 0.778 | 0.8241 | 7 |
| SimVP Lite (adapted) | 1e-4 | 0.03833 | 0.630 | 0.8212 | 2 |

完整候选表和曲线见 [学习率选择记录](results/external_leaderboard_v2/lr_selection/LEARNING_RATE_SELECTION.md)。

## 5. 扩展评估体系

除 MAE、RMSE、CSI、POD、FAR 外，本轮新增：

- 湿区 MAE/RMSE：只统计真实水深不低于 0.10 m 的像素；
- 干区预测深度与干区 false-positive rate；
- 1 像素容差的淹没边界 precision、recall 和 F1；
- `0–0.10`、`0.10–0.30`、`0.30–0.50`、`>0.50 m` 水深分箱误差；
- 峰值时刻 MAE；
- 每个测试事件的独立指标、中位数和 IQR；
- 模型参数量、训练时间、最佳 epoch、推理延迟和推理峰值 CUDA 分配。

## 6. 五种子正式结果

| 模型 | MAE cm | RMSE cm | CSI | 相对 persistence MAE | CSI 增益 | 延迟 ms/sample | 推理 CUDA MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Conv-LSTM** | **1.883 +/- 0.062** | 7.373 | 0.7504 | **-16.4%** | +6.2 pp | 1.63 | **23.4** |
| Conv-LSTM + Attention | 1.965 +/- 0.090 | 7.602 | 0.7413 | -12.8% | +5.3 pp | 1.66 | 35.9 |
| CNN-Temporal Transformer | 1.971 +/- 0.111 | **7.343** | 0.7566 | -12.5% | +6.8 pp | 3.84 | 199.4 |
| U-RNN Lite (adapted) | 2.033 +/- 0.079 | 7.945 | 0.7595 | -9.7% | +7.1 pp | 5.93 | 30.2 |
| **FNO2D-History (adapted)** | 1.889 +/- 0.073 | 7.363 | **0.7883** | -16.1% | **+10.0 pp** | **0.61** | 36.3 |
| SimVP Lite (adapted) | 1.986 +/- 0.085 | 7.897 | 0.7647 | -11.8% | +7.6 pp | 2.81 | 26.5 |

这里的 `-16.4%` 表示相对 persistence 的 MAE 降低 16.4%。推理 CUDA 数值是在统一 inference benchmark 中测量的分配峰值，不代表整个训练进程的系统显存占用。

![六模型总览](results/external_leaderboard_v2/figures/larno_ukea_ukea_model_overview.png)

## 7. 物理风险表现

| 模型 | 湿区 RMSE cm | 干区预测深度 cm | 边界 F1 | 峰值时刻误差 min | 事件 MAE 增益中位数 [IQR] |
|---|---:|---:|---:|---:|---:|
| Conv-LSTM | 22.457 | 0.857 | 0.9017 | 0.64 | 8.3% [0.5, 13.3] |
| Conv-LSTM + Attention | 23.102 | 0.931 | 0.8968 | 2.08 | 1.0% [-8.6, 11.6] |
| CNN-Temporal Transformer | **22.232** | 0.933 | 0.9035 | 3.14 | -6.1% [-18.9, 1.1] |
| U-RNN Lite (adapted) | 24.111 | 1.021 | 0.9094 | **0.10** | 10.1% [-11.2, 14.3] |
| **FNO2D-History (adapted)** | 22.441 | **0.777** | **0.9265** | 0.80 | **15.2% [5.1, 18.6]** |
| SimVP Lite (adapted) | 24.055 | 0.973 | 0.9111 | 2.42 | 14.4% [1.4, 22.1] |

![物理诊断](results/external_leaderboard_v2/figures/larno_ukea_ukea_physical_diagnostics.png)

![逐事件鲁棒性](results/external_leaderboard_v2/figures/larno_ukea_ukea_event_robustness.png)

## 8. 配对事件统计与长提前量

FNO 与最低 MAE 的 Conv-LSTM 在 12 个相同测试事件上进行配对：

- FNO 减 Conv-LSTM 的事件平均 MAE 差为 `+0.006 cm`，bootstrap 95% CI 为 `[-0.046, +0.067]`，区间跨零；两者不能被解释为具有可靠 MAE 差异；
- FNO 减 Conv-LSTM 的事件平均 CSI 差为 `+0.0577`，95% CI 为 `[+0.0409, +0.0714]`；事件置换检验 `p=0.00098`；
- 因此合理结论是：两者 MAE 基本持平，而 FNO 的淹没范围识别优势稳定存在。

在 +60 分钟：

| 方法 | MAE cm | RMSE cm | CSI |
|---|---:|---:|---:|
| Persistence | 4.911 | 18.177 | 0.4863 |
| Conv-LSTM | 3.965 | 14.144 | 0.6118 |
| **FNO2D-History** | **3.878** | **14.138** | **0.6841** |

FNO 在 +60 分钟相对 persistence 降低 MAE `21.0%`，CSI 提高 `19.8` 个百分点，说明频谱全局建模的价值主要体现在较长提前量和淹没边界。

![提前量曲线](results/external_leaderboard_v2/figures/larno_ukea_ukea_horizon_curves.png)

![水深分箱误差](results/external_leaderboard_v2/figures/larno_ukea_ukea_depth_stratified_errors.png)

## 9. 定性空间案例

代表性窗口按目标水深变化幅度自动选择，不使用任何模型误差作为筛选条件。下图展示 seed 42 的同一 `+60 min` 目标、persistence 和六个模型；定量结论仍来自全部 12 个事件和 5 seeds。

![空间预测比较](results/external_leaderboard_v2/figures/larno_ukea_ukea_spatial_forecast.png)

![空间误差比较](results/external_leaderboard_v2/figures/larno_ukea_ukea_spatial_error.png)

完整四提前量误差矩阵见 [horizon error matrix](results/external_leaderboard_v2/figures/larno_ukea_ukea_horizon_error_matrix.png)。

## 10. 可以支持的结论

1. **不存在单一维度上的绝对赢家。** Conv-LSTM 具有最低平均 MAE和最低推理 CUDA 分配；FNO 具有最高 CSI、最佳边界 F1、最低干区虚警深度、最佳 60 分钟结果和最低延迟。
2. **FNO 是当前最强的综合风险预测候选。** 它在 MAE 与 Conv-LSTM 统计持平的同时，显著提高淹没范围识别，并在 12 个事件上的 MAE 增益 IQR 全部为正。
3. **保留的 Conv-LSTM 仍然有明确工程价值。** 仅 22,084 个参数，低显存、低 MAE、峰值时刻误差小，是准确且紧凑的部署基线。
4. **Attention 没有带来收益。** 当前注意力版本在 MAE、RMSE、CSI 和边界 F1 上都未超过基础 Conv-LSTM，说明简单增加注意力并不等于性能提升。
5. **U-RNN Lite 更擅长峰值时刻，但深度误差较高。** 其峰值时刻 MAE 最低，但平均 MAE/RMSE和延迟不占优。
6. **SimVP Lite 的逐事件 MAE 中位数较强。** 其总体像素加权 MAE并非最佳，但事件级中位数和 IQR表现稳定，值得后续改进 translator。
7. **Transformer 的 RMSE较好但稳定性不足。** 它具有最低平均 RMSE，但逐事件 MAE 增益中位数为负、显存分配最高，当前实现不适合作为首选部署模型。

## 11. 结论边界与下一步

- 本轮严格六模型排行榜只完成 LarNO UKEA；UrbanFlood24 仍是上一版三模型、三种子、稀疏采样结果，不能与本表直接合并；
- 三个新增强基线是适配实现，正式论文对比仍应补充官方 U-RNN、官方 SimVPv2 或严格复现；
- 当前协议是 past-only rainfall nowcast，未来降雨 forcing 应作为独立排行榜；
- 本轮未报告统一 FLOPs，因为 FFT 与 Transformer 内核用普通 Conv/Linear hook 会系统性漏计；在引入可覆盖频谱算子的分析工具前不输出误导性数字；
- 下一阶段 P0：UrbanFlood24 六模型 5 seeds、留一地点泛化、UKEA 到 Urban 的迁移评估；
- 下一阶段 P1：动态水深/降雨与静态地形双分支主模型、边界损失、60 分钟专门校准和概率不确定性。

## 12. 可复现产物

- [完整自动报告](results/external_leaderboard_v2/EXTERNAL_PHYSICAL_BENCHMARK.md)
- [模型总表 CSV](results/external_leaderboard_v2/external_model_summary.csv)
- [逐提前量 CSV](results/external_leaderboard_v2/external_horizon_summary.csv)
- [阈值敏感性 CSV](results/external_leaderboard_v2/external_threshold_summary.csv)
- [逐事件 CSV](results/external_leaderboard_v2/external_per_event.csv)
- [配对事件比较 CSV](results/external_leaderboard_v2/external_pairwise_event_comparison.csv)
- [水深分箱 CSV](results/external_leaderboard_v2/external_depth_bin_summary.csv)
- [学习率选择记录](results/external_leaderboard_v2/lr_selection/LEARNING_RATE_SELECTION.md)

正式运行命令示例：

```bash
python -m src.run_external_benchmark \
  --dataset larno_ukea \
  --models convlstm,convlstm_attention,cnn_temporal_transformer,urnn_lite,fno2d_history,simvp_lite \
  --seeds 42,44,52,77,2026 \
  --epochs 12 --early_stop_patience 4 \
  --model_lrs "convlstm=0.001,convlstm_attention=0.0003,cnn_temporal_transformer=0.0001,urnn_lite=0.0001,fno2d_history=0.001,simvp_lite=0.0001"
```
