# LLM 推理故障模式编目（结构化视图）

> 本文件由 `python3 tools/catalog_tool.py build` 从 `catalog/entries/*.yaml` 生成，**不要手改**。
> 字段说明见 `catalog/SCHEMA.md`，受控词表见 `catalog/vocab.yaml`。

共 1342 条；状态：active 1342

| 原因子系统 | 条数 |
|---|---|
| 硬件 | 22 |
| 算子/内核 | 304 |
| 编译器 | 43 |
| 运行时/执行 | 68 |
| 集合通信 | 13 |
| 并行切分 | 9 |
| KV缓存 | 277 |
| 调度/服务 | 64 |
| 输出解析 | 52 |
| 解码/采样 | 115 |
| 量化 | 143 |
| 位置编码 | 33 |
| 分词/模板 | 28 |
| 模型定义 | 42 |
| 权重/检查点 | 32 |
| 数据管线 | 1 |
| 优化器 | 2 |
| 损失计算 | 1 |
| MoE路由 | 21 |
| 重计算/卸载 | 5 |
| RL训练回路 | 1 |
| 观测/评测 | 14 |
| 配置/参数 | 5 |
| 构建/发布 | 5 |
| 模型自身 | 16 |
| 外部攻击 | 18 |
| 待定 | 8 |

## 硬件

### INF-0001 `sdc_multibit_warp_spray` GPU 多 bit SDC 呈 warp 对齐成簇污染隐状态通道

`故障` · `active` · `infra_off` · `yes` · `verified` · `var` · 发现 待定 · 旧号 1.1

- **原因对象**：硬件 / 硬件 / GPU/加速器计算硅、显存
- **原因行为**：GPU/加速器 SDC 同时翻转多个 bit（单 bit 不足 40%，多 bit 超 60%），翻转常在同一 SIMT warp（32 lanes）内对齐、呈空间周期性；rowhammer 成簇形态与之部分重叠。
- **影响对象**：隐状态；输出分布（静默错误）
- **影响详情**：落到 Linear/matmul 时，同一 block、同一 token 的隐状态中若干通道被成簇共同污染，而非单通道尖峰；logits 看似合理但错误，NaN/Inf 仅约 1%，几乎不崩溃，表现为输出细微漂移，偶尔冒出明显错误 token。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://arxiv.org/abs/2605.04213
- **关联**：var:INF-0211

### INF-0002 `sdc_data_dependent_gated` mercurial core 在特定数据模式下确定性算错（输入门控 SDC）

`故障` · `active` · `ambiguous` · `yes` · `verified` · `NEW` · 发现 待定 · 旧号 1.2

- **原因对象**：硬件 / 硬件 / mercurial core 执行单元
- **原因行为**：执行单元只在特定数据模式加指令序列下算错，计算错误事件强依赖工作负载与频率/电压/温度；中间激活落入特定区间时确定性地给出错误结果。
- **影响对象**：数值结果；输出内容（静默错误）
- **影响详情**：多数 forward pass 干净，偶尔中间激活落入触发区的 token 被污染，看似罕见、不可复现的幻觉，但对同一触发输入可复现；被污染的 pass 留下 off-manifold 离群点，单次 pass 的 geometric 检查会漏检。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://sigops.org/s/conferences/hotos/2021/papers/hotos21-s01-hochschild.pdf；https://arxiv.org/abs/2102.11245

### INF-0003 `sdc_gradient_magnitude` 磨损型硅缺陷在 attention/GEMM 中产生有界错误值的 SDC

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 待定 · 旧号 1.3

- **原因对象**：硬件 / 硬件 / 计算硅（磨损型缺陷）
- **原因行为**：硅缺陷或 wear-out 在 attention/GEMM 重计算中、仅在特定操作数模式下产生错误但有界的数值，错误值始终在 fp32 可表示范围内、不是 NaN。
- **影响对象**：数值结果；输出内容（静默错误）
- **影响详情**：输出出现细微错误或漂移的 token，无 NaN、无崩溃、无任何错误信号；带缺陷设备可连续数月通过各类检查，约每千台设备一例。幅度低且有界，cheap 类度量可能无法与量化噪声或幻觉区分。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://engineering.fb.com/2025/07/22/data-infrastructure/how-meta-keeps-its-ai-hardware-reliable/；https://arxiv.org/abs/2102.11245

### INF-0004 `voltage_droop_didt` GEMM 功耗振荡引发 di/dt 电压跌落、绕过 ECC 的 SDC

`故障` · `active` · `infra_off` · `no` · `verified` · `NEW` · 发现 待定 · 旧号 1.4

- **原因对象**：硬件 / 硬件 / GPU 供电网络（PDN）
- **原因行为**：GEMM/attention kernel 的 MHz 级功耗振荡与 GPU PDN 谐振频率（A100 约 35 MHz）对齐时，di/dt 引发瞬态供电跌落，造成时序违例并绕过 ECC。
- **影响对象**：数值结果；隐状态（静默错误）
- **影响详情**：某次 GEMM/attention 结果出现细微随机错误 bit，某 block 隐状态被瞬态、低幅度扰动，有时波及多通道，几乎不产生 NaN；超过 100 mV 的 droop 每次推理运行约出现两次，形态像量化噪声。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://2026.hpca-conf.org/details/hpca-2026-main-conference/39/

### INF-0005 `seu_register_file` 单粒子翻转污染 GPU 寄存器堆中的激活或累加器

`故障` · `active` · `infra_off` · `yes` · `verified` · `var` · 发现 待定 · 旧号 1.5

- **原因对象**：硬件 / 硬件 / GPU SM 寄存器堆/shared memory
- **原因行为**：单粒子翻转翻掉 SM 寄存器堆或 shared-mem scratch 中保存中间激活片段或 GEMM 累加器的某个 bit；故障只存活一条指令，可能只污染一个 thread-lane。
- **影响对象**：数值结果；输出内容（静默错误）
- **影响详情**：瞬态、通常只影响单个 token：一个 token 略微出错，或某激活矩阵的部分行被弄乱，极少出现 NaN。寄存器堆注入故障约 36% 产生 SDC（shared memory 23%、global memory 11%）。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://computerresearch.org/index.php/computer/article/download/102474/23373/41996
- **关联**：var:INF-0211

### INF-0006 `training_sdc_checkpoint_inherit` 预训练期 SDC 扰动权重更新并被继承进发布的 checkpoint

`故障` · `active` · `ambiguous` · `yes` · `verified` · `NEW` · 发现 待定 · 旧号 1.6

- **原因对象**：硬件 / 硬件 / 预训练集群计算硬件
- **原因行为**：LLM 预训练期间发生静默数据损坏，静默扰动了一次权重更新。
- **影响对象**：模型权重；训练轨迹（质量退化）
- **影响详情**：扰动存活进发布的 checkpoint，部署时 forward 本身干净却带着偏离的权重运行：perplexity 略差、偶尔给出更糟的答案。没有干净参考 checkpoint 时，单模型自参照的隐藏态检测全部漏检。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://aclanthology.org/2025.acl-long.996/；https://arxiv.org/abs/2502.12340；https://arxiv.org/abs/2604.00726；https://arxiv.org/abs/2608.11034；https://arxiv.org/abs/2604.10390；https://arxiv.org/abs/2608.30769；https://arxiv.org/abs/2609.19743

### INF-0007 `finite_value_sdc_dominance` SDC 约 99% 为有限合法值，值域守卫无法检出

`结论` · `active` · `infra_off` · `not_target` · `verified` · `rationale` · 发现 待定 · 旧号 1.7

- **原因对象**：硬件 / 硬件 / SDC（原文未细分硬件部件）
- **原因行为**：硬件 SDC 产生的错误值绝大多数落在合法有限数值范围内（wrong-but-bounded），只有约 1.01% 表现为 NaN/±Inf。
- **影响对象**：数值结果；隐状态（静默错误）
- **影响详情**：约 99% 的真实损坏不产生 NaN/Inf，基于 NaN/Inf 检查或 min/max 阈值的值域守卫全部漏检，系统层面看不到这些错误；只有刻画隐状态几何/信息结构的度量才有机会暴露。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://arxiv.org/abs/2605.04213

### INF-0025 `ttmetal_quasar_upper_tile_counter_mirror_corruption` Quasar 高位 tile counter 位宽错配致镜像计数器损坏

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-03 · 旧号 1.25

- **原因对象**：硬件 / 硬件 / Tensix 高位 tile counter
- **原因行为**：Quasar 每个 Tensix 的高 16 个 tile counter（TC≥16）默认不镜像到 Overlay，且 remapper 用 5 位表示 TC≥16，而 overlay tensix 只用 4 位。
- **影响对象**：数值结果（静默错误）
- **影响详情**：编程或使用 TC A（≥16）时 A%16 对应的计数器被静默损坏，credit/同步计数出错，数据流静默错乱；既有 intra-tensix DFB 测试只验证 host 配置与单 credit push/pop，容量不冲突时不会触发。无 crash、无 NaN。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/pull/51873

### INF-0071 `vllm_sm120_sustained_load_recurring_xid13_chipwide_warp_errors` SM120 持续负载下反复出现 chip 级 Xid 13 warp 错误

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-13 · 旧号 1.72

- **原因对象**：硬件 / 待定 / SM120 GPU 或 marlin kernel
- **原因行为**：原文未定位根因：chip 范围 warp 错误在持续负载下反复出现并累积，非单次 transient；原文推断为硬件级 SDC 或 kernel 在持续热负载/特定模式下不稳定。
- **影响对象**：可用性/进度（报错）
- **影响详情**：RTX PRO 6000 Blackwell 上 vLLM 运行 Nemotron-3.5-Lightning-30B-A3B-NVFP4（marlin MoE、hybrid Mamba）持续数小时后反复出现 Xid 13 warp 错误，导致 GPU fault，可能伴随静默错误结果。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/52225

### INF-0073 `uep_bit_position_sensitivity_map_fp16_bf16_fp32` FP16/BF16/FP32 位翻转按 bit 位置的故障敏感度图谱

`结论` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-13 · 旧号 1.74

- **原因对象**：硬件 / 硬件 / FP16/BF16/FP32 数据的位翻转
- **原因行为**：硬件 SDC 或位翻转使 FP16、BF16、FP32 格式的数据发生单 bit 或连续 2/3-bit 翻转（论文以故障注入刻画）。
- **影响对象**：数值结果（静默错误）
- **影响详情**：16 个 workload 实测：低于 Xsafe（FP16=6、BF16=4、FP32=15）的低位 fraction bit 翻转使任务指标降幅<1%，upper fraction bit 敏感度上升，exponent-mantissa 边界单 bit 翻转致灾难性崩溃。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://arxiv.org/abs/2607.19623

### INF-0093 `ttmetal_llk_workaround_removal_rmsnorm_pcc_regression` RMSNorm 移除 LLK workaround 致 PCC 回归

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-19 · 旧号 1.94

- **原因对象**：硬件 / tt-metal / RMSNorm LLK dest-reuse
- **原因行为**：PR #53419 把 RMSNorm 的 LLK workaround 换成 mul_reuse_dest_tiles，而 dest-reuse 数值通路本身有边界缺陷（与 #53693 同根），原 workaround 实为绕开该通路。
- **影响对象**：数值结果（质量退化）
- **影响详情**：Blackhole 上 DeepSeek v3_d_p RMSNorm e2e 测试 PCC 降至约 0.984（3.2K: 0.9843、4K: 0.9848），低于阈值；e2e 断言可捕获，若阈值放宽或无 PCC 门则为静默精度退化。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/53649

### INF-0097 `llm_gpu_soft_error_instruction_level_fi_study` LLM 推理 GPU 软错误指令级故障注入研究

`结论` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-08-20 · 旧号 1.98

- **原因对象**：硬件 / 硬件 / GPU 指令执行软错误
- **原因行为**：GPU 软错误作用于 LLM 推理执行的指令结果；研究以指令级故障注入从模型架构、参数规模、任务复杂度三轴刻画 LLM 的软错误韧性。
- **影响对象**：数值结果（静默错误）
- **影响详情**：研究结论：decode 循环、KV 复用、注意力读写模式等 LLM 特有执行形态，使其软错误敏感度与通用应用、CNN 等既有 GPU 负载显著不同，可为容错机制设计提供依据。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://arxiv.org/abs/2601.19912

### INF-0211 `sdc_spike` 单粒子翻转在 decoder 层输出隐状态上的加性通道尖峰

`故障` · `active` · `infra_off` · `yes` · `verified` · `NEW` · 发现 待定 · 旧号 1.212

- **原因对象**：硬件 / 硬件 / decoder block L 输出激活
- **原因行为**：single-event upset 在 decoder block L 输出的最后一个 token 上，对 k 个非 massive 通道各加 c 倍健康 per-channel 标准差的加性尖峰。
- **影响对象**：隐状态；输出内容（静默错误）
- **影响详情**：幅度足够大（c=30）时 greedy 答案几乎必然翻转，无 NaN、无崩溃、无运行时错误信号，表现为一次莫名的错误回答；深度剖面在注入层后呈 STEP 跳变、末层部分回落。原型阶段实测，量值不跨栈继承。
- **改进措施**：检测 待定；修复 待定
- **来源**：引用：本项目早期实测条目（原型阶段实测，量值不跨栈继承；

### INF-0212 `sdc_spike_massive` SDC 尖峰落在 massive 激活通道，逼近检测器盲区

`故障` · `active` · `infra_off` · `yes` · `verified` · `NEW` · 发现 待定 · 旧号 1.213

- **原因对象**：硬件 / 硬件 / massive 激活通道（SDC 尖峰）
- **原因行为**：SDC 在单个 token 的激活上造成 c·sigma 量级尖峰，且尖峰落在本身数值巨大、承担 attention-sink/隐式偏置角色的 massive-activation 通道上。
- **影响对象**：隐状态（静默错误）
- **影响详情**：无崩溃，单 token 输出被推偏，同等 c 下对下游 attention 分布的破坏更结构性；因 massive 通道健康方差极大，几何类检测器分离度系统性低于 1.212，激活分位移统计可能反而更敏感。
- **改进措施**：检测 待定；修复 待定
- **来源**：引用：本项目早期实测条目（原型阶段实测，量值不跨栈继承；

### INF-0295 `hbm_xid_spare_row_exhaustion` HBM 备用行池老化耗尽，产生与 rowhammer 相同的权重行损坏

`故障` · `active` · `infra_off` · `no` · `documented` · `var` · 发现 待定 · 旧号 2.2

- **原因对象**：硬件 / 硬件 / HBM 备用行/row-remap 池
- **原因行为**：老化的 HBM 备用行/row-remap 池被耗尽（对应 Xid 63/94 告警趋势），成因是磨损老化而非主动攻击。
- **影响对象**：模型权重；隐状态（静默错误）
- **影响详情**：持久、空间成簇的连续权重行损坏（相邻输出通道），签名与 rowhammer（2.1）完全相同；隐藏态层面无法区分两者，只能靠 Xid 事件、设备年龄等运维证据。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://arxiv.org/abs/2503.11901
- **关联**：var:INF-0315

### INF-0296 `rangeguard_bounded_approximate_error_correction` RangeGuard：面向 DNN 权重/激活的有界近似纠错

`方法` · `active` · `infra_off` · `not_target` · `documented` · `NEW` · 发现 2026-08-17 · 旧号 2.3

- **原因对象**：硬件 / 硬件 / DNN 关键权重/激活张量
- **原因行为**：所针对故障：DNN 权重/激活发生位翻转或 SDC 类损坏（含 2.1/2.2 显存故障与第 3 类 BFA 的权重损坏）。
- **影响对象**：模型权重；隐状态（静默错误）
- **影响详情**：RangeGuard 为关键张量维护值域锚点，运行时检测超出有界误差的损坏并近似恢复，开销远低于全量 ECC，对推理近似无损，可作为服务内闭环恢复参照。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://arxiv.org/abs/2605.04563

### INF-0304 `bit_position_sensitivity_unequal_error_protection` DNN 推理内存逐 bit 位故障敏感度刻画与不等差错保护

`结论` · `active` · `infra_off` · `partial` · `documented` · `var` · 发现 2026-07-31 · 旧号 3.8

- **原因对象**：硬件 / 硬件 / 推理内存中的权重 bit
- **原因行为**：所针对故障：DNN 推理内存中的权重 bit 翻转，不同 bit 位的敏感度差异极大。
- **影响对象**：模型权重；输出内容（静默错误）
- **影响详情**：影响呈重尾分布：符号位、指数高位等少量关键 bit 翻转致输出灾难性偏差，多数 mantissa 低位近乎无影响，无 crash；覆盖 16 个工作负载与三种浮点格式，据此提出只保护高敏感 bit 的 UEP。
- **改进措施**：检测 待定；修复 待定
- **来源**：http://arxiv.org/abs/2607.19623v1
- **关联**：var:INF-0302

### INF-0307 `spine_gdb_binary_level_accumulated_fault_profiler` SPINE：GDB 注入累积权重位翻转的二进制级故障谱 profiler

`方法` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-16 · 旧号 3.11

- **原因对象**：硬件 / 硬件 / 内存中的量化网络权重
- **原因行为**：推理期间辐射诱发的位翻转持续、累积地翻转内存中量化网络的权重位（SPINE 经 GDB 向 edge CPU 上的目标推理二进制注入以刻画）
- **影响对象**：模型权重；输出内容（静默错误）
- **影响详情**：无 crash；累积扰动在真实内存布局下映射为误预测，per-layer 敏感度随拓扑、量化力度与内存布局而异，可据此推导选择性加固与分层保护优先级
- **改进措施**：检测 待定；修复 待定
- **来源**：https://arxiv.org/abs/2606.19526

### INF-0310 `bitflipscope_fault_localization_recovery` BitFlipScope：LLM 位翻转损坏的可扩展定位与恢复

`方法` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-08-17 · 旧号 3.14

- **原因对象**：硬件 / 硬件 / LLM 权重位（位翻转损坏）
- **原因行为**：LLM 权重矩阵中的位发生翻转，损坏落在特定层乃至具体权重位上
- **影响对象**：模型权重；输出内容（质量退化）
- **影响详情**：损坏表现为 LLM 输出异常；BitFlipScope 在线检测异常后经轻量探针/层间信号定位到具体层与权重位，并用冗余副本或重构恢复
- **改进措施**：检测 待定；修复 待定
- **来源**：https://arxiv.org/abs/2512.22174

### INF-0315 `weight_bitflip` weight_bitflip：Linear 权重 bit 翻转造成持久静默损坏

`故障` · `active` · `infra_off` · `yes` · `verified` · `NEW` · 发现 待定 · 旧号 3.19

- **原因对象**：硬件 / 硬件 / 中层 MLP down_proj 权重位
- **原因行为**：Linear 权重中一个或 n 个 bit 被 XOR 翻转并持久保留，形成静默权重损坏；按位类别分符号位、指数位（exp_high=30、exp_low=23）与尾数位
- **影响对象**：模型权重；隐状态；输出内容（静默错误）
- **影响详情**：exp_high 翻转使输出显著劣化，损坏经残差流只在注入层下游的隐状态显形（评估层在上游必漏检）；mantissa 单翻转在 fp32 上几乎无可观测变化，模型可带损坏权重长期服务无人察觉
- **改进措施**：检测 待定；修复 待定
- **来源**：引用：本项目早期实测条目（原型阶段实测，量值不跨栈继承；

### INF-0316 `weight_bitflip_naninf` weight_bitflip_naninf：指数位翻转致权重 nan/inf

`故障` · `active` · `infra_off` · `yes` · `documented` · `NEW` · 发现 待定 · 旧号 3.20

- **原因对象**：硬件 / 硬件 / Linear 权重指数高位（exp_high）
- **原因行为**：权重指数高位（典型为 bit 30）被翻转，使该权重元素变为 nan/inf；属 3.19 weight_bitflip 的结果分类，无独立注入机制
- **影响对象**：模型权重；输出分布；输出内容（数值异常）
- **影响详情**：nan/inf 进入残差流后整条前向被污染，logits 全为 nan，解码输出为空或垃圾；cheap 族 nan_inf_flag 一票命中，但含 nan 隐状态会使 geometric/spectral/statistical 度量全部退化为 nan
- **改进措施**：检测 待定；修复 待定
- **来源**：引用：本项目早期实测条目（原型阶段实测，量值不跨栈继承；

### INF-1256 `prefix_cache_shared_block_bitflip_silent_divergence` 共享前缀 KV 块位翻转导致跨请求静默发散

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-07-30 · 旧号 K4.9

- **原因对象**：硬件 / 硬件 / GPU DRAM 位翻转（共享前缀 KV 块）
- **原因行为**：GPU DRAM 上 rowhammer 类位翻转或 row-remap 耗尽的物理故障翻转共享前缀 KV 块中的 bit；vLLM 前缀缓存的共享块是单一物理副本且无完整性校验，损坏被所有共享请求复用。
- **影响对象**：KV状态；输出内容；跨请求/跨样本数据（静默错误）
- **影响详情**：共享该前缀的请求得到连贯但错误的输出：16 个 BF16 bit 位中 13 个翻转产生 coherent-but-altered 响应，无 clean baseline 不可区分；仅共享前缀的请求受影响，损害无时间衰减、随请求数线性累积。
- **改进措施**：检测 待定；修复 待定
- **来源**：http://arxiv.org/abs/2604.17249v2

## 算子/内核

### INF-0008 `sfpu_isclose_inf_nan_bitpattern_miscompare` SFPU isclose 把整型位模式当浮点比较致 Inf/NaN 误判

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-07-29 · 旧号 1.8

- **原因对象**：算子/内核 / tt-metal / ckernel_sfpu_isclose.h
- **原因行为**：Inf/NaN 修正分支用整型位模式 inf_bits 与 vFloat 比较，因 sfpi 无 (vFloat, int32_t) 重载也无 vFloat→vInt 转换，实际走 float >= 重载，把位模式当浮点量级比较。
- **影响对象**：数值结果（静默错误）
- **影响详情**：在 +Inf/NaN 输入上 isclose 静默返回错误值，无 assert、无 NaN、无 crash；依赖 isclose 做 Inf/NaN 处理的上层算子（梯度裁剪、loss 监控、稳定性检查）在含异常值的张量上得到错误判定。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/51220；https://github.com/tenstorrent/tt-metal/issues/51540

### INF-0010 `bf16_nan_canonicalization_silent_inf` tt-metal bf16 NaN 经设备 op 被静默改写为 Inf

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-07-30 · 旧号 1.10

- **原因对象**：算子/内核 / tt-metal / ttnn 设备 op 的 bf16 NaN 传播
- **原因行为**：bfloat16 NaN 经过任何设备 op（包括不做算术也不转格式的 to_layout、neg、abs、typecast）都被改成同符号 Infinity，返回途中 NaN 传播语义被静默改变。
- **影响对象**：数值结果（静默错误）
- **影响详情**：依赖 NaN 作控制信号的上层算子（梯度裁剪、稳定性检查、稀疏激活门控的 NaN 屏蔽）在 Tenstorrent 上失效，控制流分支判错；host round-trip 与 ttnn.clone 能保留 NaN。无 crash、无 assert。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/51557

### INF-0011 `sfpu_exp_overflow_guard_missing` Quasar SFPU exp 丢失上下溢守卫返回静默垃圾值

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-07-30 · 旧号 1.11

- **原因对象**：算子/内核 / tt-metal / _sfpu_exp_fp32_accurate_
- **原因行为**：把 2^i 直接写入 IEEE-754 指数字段，仅在 i+127∈[1,254] 时合法；Quasar 移植时丢掉了 Blackhole 原有的上溢→inf、下溢→0 两端守卫，越界时 i+127 溢出到符号位并被位移截断。
- **影响对象**：数值结果；隐状态（静默错误）
- **影响详情**：指数越界时 exp 返回静默垃圾且常为负值，如 exp(-88.5)→-inf、exp(-100)→-4.3e33；exp 位于 softmax、GELU、SiLU 门控路径，负值会产生反号激活，可能翻转专家路由或注意力分布。无 crash。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/51543

### INF-0013 `sglang_cutedsl_2cta_cluster_barrier_missing` SGLang 2-CTA GEMM 退出前缺 cluster 同步致崩溃

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `var` · 发现 2026-07-30 · 旧号 1.13

- **原因对象**：算子/内核 / sglang / cutedsl_bf16_gemm 2-CTA
- **原因行为**：2-CTA cluster GEMM kernel 只在入口做 cluster 同步，退出前没有 cluster barrier，而跨 CTA 的 peer 读取持续到 kernel 末尾。
- **影响对象**：可用性/进度（报错）
- **影响详情**：CTA 0 退出后 CTA 1 仍向其 SMEM 发 barrier/TMA 事件，报 CUDBG_EXCEPTION_CLUSTER_BLOCK_NOT_PRESENT，decode 进程崩溃，崩溃前无静默错误输出；见于 Kimi-K3 服务（TP8、PD 分离、B300）。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/32907
- **关联**：var:INF-0009

### INF-0014 `ttmetal_sfpswap_nan_propagation_drop` ttnn max/min/clamp 由 SFPSWAP 实现致静默吞掉 NaN

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-07-31 · 旧号 1.14

- **原因对象**：算子/内核 / tt-metal / max/min/clamp/hardtanh
- **原因行为**：这些 op 各由 1–2 条裸 SFPSWAP 比较交换指令实现，两个输出始终是输入的置换，无法让 max 与 min 在含 NaN 输入上同时返回 NaN。
- **影响对象**：数值结果（静默错误）
- **影响详情**：NaN 输入下 maximum/minimum/clamp/hardtanh 静默返回有限值而非 NaN，违反 IEEE 754/PyTorch NaN 传播约定；NaN 被吞掉后下游继续用错误值计算，单帧检测器难以与正确值区分。无 crash。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/51470

### INF-0018 `ttmetal_fmod_out_of_range_wrong_sign` ttnn.fmod/remainder 对大输入返回越界错号余数

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-01 · 旧号 1.18

- **原因对象**：算子/内核 / tt-metal / ttnn.fmod/remainder 内核
- **原因行为**：ttnn.fmod(x, scalar) 及同内核结构的 ttnn.remainder 对较大 |x| 的余数计算有误；原文未给出具体实现缺陷。
- **影响对象**：数值结果（静默错误）
- **影响详情**：|x| 约 >1e7 时返回数学上不可能的值：幅值可超过模数、符号可与 x 相反，如 fmod(1e8,3) 得 0（应为 1）、fmod(1e10,3) 得 -1021。无 crash，静默污染依赖模运算的下游计算。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/51661

### INF-0023 `ttmetal_sort_bf16_integer_collapse_nondeterministic_index` ttnn.sort 用 bf16 比较致大整数排序索引非确定且可错

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-02 · 旧号 1.23

- **原因对象**：算子/内核 / tt-metal / ttnn.sort 的 TopK SFPU LLK
- **原因行为**：topk_local_sort 默认用 bf16 比较 tile 值，>256 的整数舍入到同一 bf16 值被视为相等，等值元素保留 merge-sort 网络碰巧放置的顺序。
- **影响对象**：数值结果（静默错误）
- **影响详情**：对 dtype 为 UINT32/INT32 且值 >256 的输入，ttnn.sort 返回的 indices 非确定且可能错误，影响依赖确定排序的 top-k 选择、argsort 索引等下游。无 crash、无 NaN。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/pull/51562

### INF-0024 `ttmetal_tilize_pack_reconfig_missing_wrong_dtype_output` ttnn.tilize 缺 pack 格式重配置致输出字节流格式错误

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-03 · 旧号 1.24

- **原因对象**：算子/内核 / tt-metal / retile kernel、pad reader
- **原因行为**：retile compute kernel 进入 tilize 前未调用 pack_reconfig_data_format 切换 packer 格式；padding reader 把任意偏移 L1 地址强转 uint32_t* 整字写入。
- **影响对象**：数值结果（静默错误）
- **影响详情**：tilize 同时改 tile 形状和 dtype（如 BFLOAT16→BFLOAT8_B）时，每个 tile 以输入格式打包，输出张量损坏；未填充行宽非 4 字节倍数（如 bf16 width=31）时静默丢尾部字节并覆盖最后的真实输入字节。无 crash、无 NaN。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/pull/51903

### INF-0026 `ttmetal_fmod_remainder_unary_scalar_inverted_fixup` ttnn.fmod/remainder 标量路径 fix-up 反转致错号余数

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `var` · 发现 2026-08-03 · 旧号 1.26

- **原因对象**：算子/内核 / tt-metal / fmod/remainder unary 标量路径
- **原因行为**：unary 标量路径的 fix-up 循环被反转，且 guard return 不可达；binary kernel 已带的正确修复没有同步到 unary 标量变体。
- **影响对象**：数值结果（静默错误）
- **影响详情**：unary 标量 fmod/remainder 返回符号错误、超出 [0, |s|) 范围的余数，静默产出错误数值。无 crash、无 NaN。与 1.18 同族，只是触发路径不同（unary scalar vs binary）。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/51441
- **关联**：var:INF-0018

### INF-0027 `ttmetal_sfpu_softsign_reciprocal_underflow_zero` tt-metal softsign 倒数中间值下溢致大输入静默返回 0

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `var` · 发现 2026-08-04 · 旧号 1.27

- **原因对象**：算子/内核 / tt-metal / calculate_softsign()
- **原因行为**：calculate_softsign() 把 x/(1+|x|) 分解为 v*reciprocal(1+|v|)，大输入时倒数中间值低于最小正规数被 flush-to-zero，乘积坍缩为 0。
- **影响对象**：数值结果（静默错误）
- **影响详情**：大输入时 softsign 静默返回 0，而非饱和端的 ±1；Wormhole 约从 1.06e38 起出错，Blackhole 早一个 binade（约 8.51e37）。现有 run_activation_unary_test 按标准正态采样输入，测不到该量级。无 crash、无 NaN。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/pull/51979
- **关联**：var:INF-0011

### INF-0028 `ttmetal_logaddexp_naive_exp_decomposition_overflow_inf` ttnn.logaddexp 朴素分解中间 exp 溢出静默返回 inf

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-08-04 · 旧号 1.28

- **原因对象**：算子/内核 / tt-metal / ttnn.logaddexp(2) 数学分解
- **原因行为**：ttnn.logaddexp/logaddexp2 采用朴素分解 log(exp(a)+exp(b))，|a| 或 |b| 超过约 88.7（FP32 exp 溢出阈值）时中间 exp 结果溢出。
- **影响对象**：数值结果（数值异常）
- **影响详情**：大幅值输入时无 crash，logaddexp/logaddexp2 返回 inf 而非正确的有限值；修复只覆盖 SFPU 与 binary_ng 路径，legacy FPU 路径仍保留旧分解。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/pull/52107；https://github.com/tenstorrent/tt-metal/pull/53720

### INF-0029 `ttmetal_sfpu_rng_lane_seed_correlation_nonuniform` ttnn.rand SFPU PRNG lane 与 seed 相关致非均匀采样

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-08-04 · 旧号 1.29

- **原因对象**：算子/内核 / tt-metal / ttnn.rand/uniform PRNG 播种
- **原因行为**：硬件 XNOR LFSR 的全 1 锁死态未受保护，lane salt 依赖先前 SFPU 操作残留的可变 LREG 状态，导致 PRNG lane 与相邻 per-core seed 之间存在相关性。
- **影响对象**：数值结果（静默错误）
- **影响详情**：ttnn.rand/ttnn.uniform 输出偏离均匀分布，出现 seed 依赖的系统性样本均值偏移，远超 IID 采样误差；无 crash、无 NaN，可静默影响 dropout、采样、初始化等下游计算。涉及 Wormhole 与 Blackhole LLK。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/pull/52024

### INF-0030 `ttmetal_fpu_reduce_10bit_mantissa_truncation_topk_tiebreak` FPU reduce 截断尾数致 tie-break argmax 选错

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-08-05 · 旧号 1.30

- **原因对象**：算子/内核 / tt-metal / greedy tie-break 索引归约
- **原因行为**：tie-break 用 float32 ttnn.min 求并列最大值的最低全局 index；FLOAT32 min 走 FPU reduce 路径，源寄存器截到 10-bit 尾数，100k 量级词表 index 被舍入成不存在的值。
- **影响对象**：数值结果；输出内容（静默错误）
- **影响详情**：存在并列最大值时，后续 eq() 匹配不到任何 index、winner mask 全零，argmax 静默返回 position 0 而非最低全局 index 的候选，影响采样与 top-k 选择；无 crash、无 NaN。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/pull/52177

### INF-0031 `ttmetal_blackhole_dest_reuse_sync_race_nondeterministic_welford` Blackhole dest 复用竞态致 layernorm 输出非确定

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-08-05 · 旧号 1.31

- **原因对象**：算子/内核 / tt-metal / LLK 目标复用 dummy unpack 同步
- **原因行为**：Blackhole 目标复用的 dummy unpack（UNPACR_NOP）清除保留的 source bank 时不等待 DEST→source 搬移完成，可能在 math 把 DEST 移入 SrcA/SrcB 时清掉 bank。
- **影响对象**：数值结果（不可复现）
- **影响详情**：背靠背执行大张量 FP32 Welford layernorm 时，同一输入跨次运行输出不同；仅在 WATCHER 禁用时出现，启用 WATCHER 时其同步屏障掩盖了竞态。无 crash、无 NaN。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/52252；https://github.com/tenstorrent/tt-metal/pull/52256

### INF-0032 `ttmetal_sfpu_polygamma_euler_maclaurin_out_of_domain` ttnn.polygamma 负输入超出渐近展开有效域返回错误有限值

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-06 · 旧号 1.32

- **原因对象**：算子/内核 / tt-metal / ttnn.polygamma 尾部渐近校正
- **原因行为**：Euler-Maclaurin 尾部校正固定在 z = x + NUM_TERMS（NUM_TERMS=6）处展开，该渐近展开只对大 z 有效；负输入时 z 坍缩，且 op 无输入校验、无文档说明有效范围。
- **影响对象**：数值结果（静默错误）
- **影响详情**：负输入时 polygamma 返回错误的有限值（含符号翻转），而数学上它对所有非整数 x 都良定义；无 crash、无 NaN、无 assert，Wormhole 与 Blackhole 结果字节一致。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/52278

### INF-0033 `ttmetal_sfpu_i1_erfc_pinned_to_fit_edge_constant` ttnn.i1/erfc 输入钳位无痕迹致域外返回单一常量

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-06 · 旧号 1.33

- **原因对象**：算子/内核 / tt-metal / ttnn.i1/erfc SFPU 输入钳位
- **原因行为**：i1 与 erfc 把输入钉在近似有效范围边缘再求值，钳位不留 flag 或 NaN；ttnn.i1 钉在 ±88.5，取的是 exp 的 FP32 溢出点而非 i1 自身的溢出点（91.90626）。
- **影响对象**：数值结果（静默错误）
- **影响详情**：域外输入一律返回同一个有限常量，下游无法检测；穷举 bf16 时 65024 个正常 pattern 中 31132 个（47.9%）返回 1.15477e+37，其中 31120 个正确答案为 ±Inf。无 crash、无 NaN。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/pull/52202

### INF-0034 `ttmetal_sort_packer_format_not_reset_stride2_corruption` ttnn.sort packer 32-bit 态未重置致输出交织损坏

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-06 · 旧号 1.34

- **原因对象**：算子/内核 / tt-metal / ttnn.sort packer 格式重配
- **原因行为**：多核 sort 以 UInt32 打包 index tile 后 packer 格式寄存器留在 32-bit 态，后续 pack_untilize_init 重配时未完全重置 32-bit gasket/accumulator 状态。
- **影响对象**：数值结果（数值异常）
- **影响详情**：row-major DRAM 多核路径上张量宽度大到需 uint32 index 时，packer 把 4-byte datum 写进 2-byte bf16 value 输出缓冲，出现 stride-2 交织损坏，每隔一个 datum 为 inf/nan；下游消费垃圾值之前不 crash。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/pull/52271

### INF-0035 `ttmetal_matmul_ktile_pad_blockfloat_uninit_l1` block-float matmul K-tile padding 未初始化

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-06 · 旧号 1.35

- **原因对象**：算子/内核 / tt-metal / pad_last_ktile 块浮点填充
- **原因行为**：pad_last_ktile()/pad_last_transposed_ktile() 只处理 Float32 与 Float16_b，对 bfloat8_b/bfloat4_b 共享指数块浮点格式不初始化 K 维 padding。
- **影响对象**：数值结果（静默错误）
- **影响详情**：bfloat8_b/bfloat4_b matmul 在 K 非 tile 对齐时，陈旧 L1 数据参与归约，结果静默错误；无 crash、无 NaN。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/pull/52302

### INF-0036 `ttmetal_pad_float32_packed_as_bf16_halves` ttnn pad 缺 FLOAT32 分支致填充值被按 bf16 打包

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-06 · 旧号 1.36

- **原因对象**：算子/内核 / tt-metal / pad 程序工厂 dtype 分发
- **原因行为**：pad 的 5 个程序工厂只有 INT32/UINT32 与 UINT16 分支、无 FLOAT32 分支，float32 张量落入 bf16 分支：填充值被舍入为 16 位，再复制成两个 bf16 半字打包进 uint32 slot。
- **影响对象**：数值结果（静默错误）
- **影响详情**：float32 张量做 pad 时，padding 区域每个 4-byte slot 写入错误值（bf16 舍入版本而非原始 float32），无 crash、无 NaN；5 个工厂损坏程度不同，有的低位正确、有的全错。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/pull/52326

### INF-0038 `ttmetal_moe_selective_reduce_wrong_token_map_stride` MoE selective_reduce_combine 步长差一

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-06 · 旧号 1.38

- **原因对象**：算子/内核 / tt-metal / selective_reduce_combine
- **原因行为**：host 用 total_tokens 除 expert 行得 dense_token_maps_stride_elm，但每行实含 total_tokens + 1 个条目（含 -1 终止符），传给 kernel 的步长差一。
- **影响对象**：数值结果（静默错误）
- **影响详情**：Watcher 启用时单设备 FullLocal B=1 确定性触发 reader.cpp:58 assert；Watcher 禁用时步长差一可能产生静默错误的 MoE reduce 结果。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/52371

### INF-0040 `triton_softmax_floor_exp_precision_mismatch` Triton softmax 后 floor 边界舍入与 PyTorch 不一致

`故障` · `active` · `ambiguous` · `partial` · `documented` · `var` · 发现 2026-08-06 · 旧号 1.40

- **原因对象**：算子/内核 / triton / tl.softmax 后 floor 舍入
- **原因行为**：tl.softmax 归一化输出恰落在整数边界时，floor 的舍入方向与 PyTorch 参考实现不一致。
- **影响对象**：数值结果（静默错误）
- **影响详情**：32×32 float32 kernel 做 softmax→sum→floor→exp，对确定性 ramp 输入返回与 PyTorch 参考不同的结果，后续 exp 放大差异；无 crash、无 NaN。最初在 triton-ascend 暴露，后确认 GPU Triton 亦有。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/triton-lang/triton/issues/11165
- **关联**：var:INF-0317

### INF-0041 `ttmetal_sort_rowmajor_dram_multicore_garbage` ttnn.sort ROW_MAJOR 多核 factory 返回非排列垃圾值

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-07 · 旧号 1.41

- **原因对象**：算子/内核 / tt-metal / ttnn.sort 单行多核 factory
- **原因行为**：ttnn.sort 的 SortProgramFactorySingleRowMultiCore 处理 ROW_MAJOR 输入的实现有误；具体缺陷机制原文未给出，待定。
- **影响对象**：数值结果（数值异常）
- **影响详情**：ROW_MAJOR 输入宽度足以触发该 factory 时，sort 返回既非排序也非输入排列的结果，含输入中不存在的值（Wormhole B0 上出现 ±inf，输入范围仅 ±3.33），不报错；TILE 布局同形状及另两条 ROW_MAJOR factory 均正确。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/52532

### INF-0042 `ttmetal_welford_fp32_fpu_path_precision_loss` Float32 Welford 静默走 FPU 路径致精度损失

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-07 · 旧号 1.42

- **原因对象**：算子/内核 / tt-metal / layernorm Welford 路径选择
- **原因行为**：Welford 程序对 Float32 输入在 fp32_dest_acc_en=false 时走 FPU 路径、unpacker 截断到 tf32/bf16 风格精度，未强制唯一有效的 SFPU/UnpackToDestFp32 路径。
- **影响对象**：数值结果（静默错误）
- **影响详情**：Float32 layernorm 且 fp32_dest_acc_en=false 时 op 照常返回结果，精度静默损失，表现为普通数值差异而非配置错误；无 crash、无 NaN。与 #52081（非 Welford 版）同根因。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/52307

### INF-0043 `ttmetal_sharded_softmax_subblock_w_unchecked` sharded softmax subblock_w 超 Dest 容量无校验

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-07 · 旧号 1.43

- **原因对象**：算子/内核 / tt-metal / sharded softmax 参数校验
- **原因行为**：sharded softmax 对 subblock_w 的唯一约束是 block_w % subblock_w == 0，未检查 subblock_w 是否超过 Dest 容量。
- **影响对象**：数值结果（静默错误）
- **影响详情**：subblock_w 超过 Dest 容量（如 12）时返回垃圾张量，行和高达 47.42（应为 1），形状/dtype 正确且无错误、警告或日志；Blackhole P300 上 sentence_bert 形状 (1,1,36864,384) bf16 复现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/52040

### INF-0044 `ttmetal_topk_singlecore_ignores_indices_tensor` ttnn.topk 单核路径忽略传入的 indices_tensor

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-07 · 旧号 1.44；1.70

- **原因对象**：算子/内核 / tt-metal / ttnn.topk 单核 factory
- **原因行为**：单核 topk_single_core_program_factory 硬编码 GENERATE_INDICES="1"，忽略传入的 indices_tensor，返回生成的 iota；多核 factory 才读取该 tensor。
- **影响对象**：数值结果（静默错误）
- **影响详情**：单核形状（W<8192、非 2 的幂、W≥65535、k>64 等）调用 ttnn.topk(..., indices_tensor=...) 时，返回 [0,1,2,...] 而非传入的索引（如全 777）；无 crash、无 NaN。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/51921

### INF-0046 `ttmetal_bfloat8b_subtile_unary_allzeros_sdc` bfloat8_b 子 tile unary 运算静默返回全零

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-07 · 旧号 1.46

- **原因对象**：算子/内核 / tt-metal / bfloat8_b unary 计算路径
- **原因行为**：block-float（bfloat8_b）unary/reciprocal 计算路径在 TILE_LAYOUT、4-D 且末两维小于 32×32 的子 tile 形状下算错；同形状 bfloat16/float32 正确，具体机制待定。
- **影响对象**：数值结果；隐状态（静默错误）
- **影响详情**：reciprocal/rsqrt/log/log2/log10 返回全零（含严格正输入的 reciprocal），div 在 tile 对齐 2-D 下也错；rsqrt 用于 RMSNorm 时小 trailing 维的归一化输出被静默清零。Blackhole 复现，无 crash、无 NaN。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/51831

### INF-0048 `ttmetal_xielu_large_neg_inf_nan` ttnn.xielu 大负输入因指数下溢钳位返回 Inf/NaN

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-08-08 · 旧号 1.48

- **原因对象**：算子/内核 / tt-metal / _sfpu_neg_exp_f32_ 下溢钳位
- **原因行为**：xielu 大负分支调用的私有 helper _sfpu_neg_exp_f32_ 把缩放后的指数钳在 UNDERFLOW_THRESHOLD = -126.5，指数重构的下溢钳位区间有误。
- **影响对象**：数值结果（数值异常）
- **影响详情**：x∈[-95,-80]（正确值约 26）时返回 Inf（bf16）/NaN（fp32）及约 1e38 的符号翻转值；Wormhole B0 实测 60 点中 bf16 有 1 个非有限、9 个 >1e30，fp32 有 3 个非有限。xielu(-90) 得 2.34e37，正确为 26.2。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/52607；https://github.com/tenstorrent/tt-metal/issues/54046

### INF-0049 `ttmetal_pow_swallows_nan_operand` ttnn.pow log2 快速路径把 NaN 操作数改写为有限值

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-08-08 · 旧号 1.49

- **原因对象**：算子/内核 / tt-metal / ttnn.pow/rpow log2 快速路径
- **原因行为**：log2 快速路径用 sfpi::setexp(abs_base, 127) 归一化底数，setexp(NaN,127) 保留非零尾数、强制指数 127，把 NaN 改写为有限值；两个 power kernel 均无 is_nan 守卫。
- **影响对象**：数值结果（静默错误）
- **影响详情**：pow(NaN, y) 与 pow(x, NaN) 返回有限值而非 NaN，违反 IEEE-754 NaN 传播要求；上游已产生的 NaN 被吞掉，损坏沿计算图无痕传播。无 crash、无 assert。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/52593

### INF-0050 `ttmetal_sfpu_saturation_silent_constant` SFPU unary kernel 域外饱和钳位静默返回常数

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-08-08 · 旧号 1.50

- **原因对象**：算子/内核 / tt-metal / SFPU unary kernel 饱和钳位
- **原因行为**：erfc、digamma、softplus、i1（及已修复的 tanh）在拟合域外用饱和操作（sfpi::min、范围缩减索引 clamp）把参数钳入有效域，饱和不留任何 trace。
- **影响对象**：数值结果（静默错误）
- **影响详情**：域外输入时返回常数：erfc 在 |x|≥5、i1 在 |x|>88.5、digamma 在 LUT 拟合底以下、softplus 尾部恰为 0；无 crash、无 NaN、无 assert，没有任何信号。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/52178

### INF-0051 `ttmetal_bf16_inf_inf_returns_zero` bf16 add/sub 对 Inf-Inf 返回 0 而非 NaN

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-08-08 · 旧号 1.51

- **原因对象**：算子/内核 / tt-metal / bf16 eltwise binary 路径
- **原因行为**：bf16 操作数的 eltwise 运算与 fp32 走不同 compute engine，bf16 所走较旧 binary 路径的 NaN 生成逻辑缺失或不同。
- **影响对象**：数值结果（静默错误）
- **影响详情**：bf16 下 Inf-Inf、Inf*0、Inf/Inf 返回 0.0（div 返回 +Inf），fp32 同表达式同形状返回 NaN；本应为 NaN 的结果变成 0 后静默传播（如残差连接出现 Inf-Inf 处）。Blackhole P300 实测。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/52141

### INF-0052 `ttmetal_div_inf_markstein_residual_nan` SFPU div 的 Markstein 残差精炼对 Inf 分母产出 NaN

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-08-08 · 旧号 1.52

- **原因对象**：算子/内核 / tt-metal / SFPU div Markstein 残差精炼
- **原因行为**：fp32 DEST 累加下 calculate_sfpu_binary_div 先由 reciprocal_iter<2>(Inf) 得到正确的 0，随后 Markstein 残差精炼未守卫 Inf，0*Inf 得 NaN 覆盖正确结果。
- **影响对象**：数值结果（数值异常）
- **影响详情**：分母为 Inf 时 a/Inf 返回 NaN 而非 IEEE-754 要求的 0（torch.div 验证），无 crash、无 assert；下游若做归一化（如 softmax 前的 1/‖x‖）则 NaN 继续传播。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/52112

### INF-0053 `ttmetal_xlogy_dead_nan_guard` xlogy 的 NaN 守卫用 == nan 恒假致 NaN 被吞

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-08-08 · 旧号 1.53

- **原因对象**：算子/内核 / tt-metal / xlogy NaN 守卫
- **原因行为**：xlogy 的 NaN 守卫写成 in1 == nan，而 IEEE-754 中 NaN≠NaN，该比较恒假，守卫永不触发。
- **影响对象**：数值结果（静默错误）
- **影响详情**：xlogy(x, NaN) 落入 x*log(x) 路径，返回有限值（x=8 时约 89）而非 NaN；NaN 被静默吞掉，损坏无痕传播。无 crash、无 assert。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/52036

### INF-0054 `ttmetal_pow_bf16_relative_error_grows_with_exponent` ttnn.pow bf16 路径相对误差随指数无界增长且溢出返回 NaN

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-08-09 · 旧号 1.54

- **原因对象**：算子/内核 / tt-metal / ttnn.pow bf16 SFPU 路径
- **原因行为**：bf16 路径按 2^(y·log2x) 算 pow，log2 只用单一 3 阶多项式，其绝对误差（最大约 2.8e-3）被指数 y 放大，相对误差约 2e-3·|y| 无上界；结果 ≥2^128 时指数回绕。
- **影响对象**：数值结果（静默错误）
- **影响详情**：bf16/BFLOAT8_B/BFLOAT4_B 下浮点或张量指数的 pow/rpow 在整个文档化输入域静默返回有限错误值，如 pow(0.996094,100.0) 得 0.8008（真值 0.6761）；溢出时返回 NaN 而非 inf。int32 指数重载准确。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/52675

### INF-0057 `ttmetal_silu_ignores_approximation_mode_flag` tt-metal silu 静默忽略 math_approx_mode 近似模式

`故障` · `active` · `ambiguous` · `config` · `documented` · `NEW` · 发现 2026-08-10 · 旧号 1.57

- **原因对象**：算子/内核 / tt-metal / calculate_silu 近似模式参数
- **原因行为**：silu_tile 把 APPROX 传给 silu_tile_init 却不传给 calculate_silu，后者是 llk_sfpu 唯一不声明 APPROXIMATION_MODE 的激活函数，silu_init 也丢弃该参数。
- **影响对象**：性能/资源（性能退化）
- **影响详情**：调用方设置 math_approx_mode=true 时 silu 仍走精确 sigmoid（精确 exp + reciprocal 迭代）而非近似路径，flag 静默不生效，无 crash、无 NaN；期望更快的近似模式实际得到更慢的精确模式。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/pull/52735

### INF-0059 `ttmetal_group_norm_nonzero_tile_padding_silent_wrong` ttnn.group_norm 假设 tile padding 为零致静默错误

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-11 · 旧号 1.59

- **原因对象**：算子/内核 / tt-metal / group_norm 的 padding 反校正
- **原因行为**：group_norm 的 #50682 修正假设 tile padding 为零并据此反校正，但 ttnn 不保证零填充，exp、softmax、slice、reshape 等 producer 会留下非零 padding。
- **影响对象**：数值结果（静默错误）
- **影响详情**：H*W 非 tile 对齐且输入来自非 host-upload producer 时 group_norm 静默输出错误值，interleaved 与 block-sharded 均受影响，后者灾难性；现有测试都经 from_torch 等零填充路径构建输入，测试套件全部通过。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/52685

### INF-0060 `ttmetal_polygamma_negative_domain_euler_maclaurin_divergence` ttnn.polygamma 负输入域渐近展开发散返回错误值

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-11 · 旧号 1.60

- **原因对象**：算子/内核 / tt-metal / ttnn.polygamma 尾校正
- **原因行为**：polygamma kernel 在固定偏移 z=x+6 处应用 Euler-Maclaurin 尾校正，该展开仅对大 z 有效；x 为负时 z 变小、为零或为负，kernel 无输入域验证，也无文档化范围。
- **影响对象**：数值结果（静默错误）
- **影响详情**：x<0 时无 crash，返回静默错误有限值：x=-4.5 起退化，x=-5.5 渐近级数发散，x=-6.0 除零，x<-6 符号反转；Wormhole 与 Blackhole 实现字节相同，均受影响。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/52278

### INF-0061 `ttmetal_sfpu_clamp_silent_constant_outside_fit_domain` tt-metal SFPU unary kernel 拟合域外静默返回常量

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-11 · 旧号 1.61

- **原因对象**：算子/内核 / tt-metal / erfc、digamma、softplus、i1
- **原因行为**：多个 SFPU unary kernel 在拟合域外返回常量：erfc（|x|≥5）、digamma（低于 LUT 拟合底）、softplus（尾部 clamp 到 0）、i1（|x|>88.5）；SFPU 饱和静默，不抛异常、不设标志。
- **影响对象**：数值结果（静默错误）
- **影响详情**：输入落在拟合域外时输出为常量，无 crash、无 NaN，调用者无法区分正确饱和到边界值与域外返回的无意义常量；相关 6 个独立报告互不提及。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/52178

### INF-0062 `ttmetal_moe_selective_reduce_combine_wrong_dense_token_map_stride` selective_reduce_combine 的 stride 推导错误

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-11 · 旧号 1.62

- **原因对象**：算子/内核 / tt-metal / selective_reduce_combine
- **原因行为**：host 以 total_tokens 推导 token-map 行 stride，但每行实有 total_tokens+1 个条目（含 -1 terminator，kernel 也按 +1 索引），stride 因此错误。
- **影响对象**：数值结果（静默错误）
- **影响详情**：dense token-map 路径下 Watcher 关闭时 combine 输出数据静默损坏；启用 Watcher 时 FullLocal B=1、output_height_shard_dim=4 确定性触发 reader.cpp:58 断言。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/52371

### INF-0063 `ttmetal_sdpa_decode_fp32_dest_acc_silent_wrong_96head_hang` sdpa_decode 开启 fp32 dest 累加时静默错误且 96 头挂起

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-12 · 旧号 1.63

- **原因对象**：算子/内核 / tt-metal / sdpa_decode fp32 dest 累加
- **原因行为**：sdpa_decode 在 fp32_dest_acc_en=True 时，fp32 dest 累加路径的 reduction/累加逻辑在特定 head count 和 shape 组合下产生数值错误或死锁。
- **影响对象**：数值结果；可用性/进度（静默错误）
- **影响详情**：启用 fp32 目标累加的 SDPA decode 对大多数 shape 静默产生错误结果，无 crash；query heads 为 96 时挂起。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/50564

### INF-0064 `ttmetal_scatter_single_index_silent_drop` ttnn.scatter 单索引写入 rank-1 operand 被静默丢弃

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-12 · 旧号 1.64

- **原因对象**：算子/内核 / tt-metal / ttnn.scatter 预变换早退
- **原因行为**：pre_scatter_transform_tensor 对 shape {1} 早退，index/source 仍为 rank-1 而 input 已变 4D，维度计算越界使 in_bounds() 恒假，跳过 scatter。
- **影响对象**：数值结果（静默错误）
- **影响详情**：rank-1 operand 且仅 1 个索引时编译运行成功，但写入被静默丢弃、输出即 input 原值，如 jnp.zeros(3).at[jnp.array([0])].set(7) 返回 [0,0,0]；两个及以上索引正常。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/52946

### INF-0065 `ttmetal_pow_bf16_highside_overflow_wraps_to_nan` ttnn.pow bf16 路径高端溢出回绕为 NaN 而非 +inf

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `var` · 发现 2026-08-12 · 旧号 1.65

- **原因对象**：算子/内核 / tt-metal / ttnn.pow bf16 路径高端 guard
- **原因行为**：bf16/非 fp32 dest 的 pow 路径只对 z=y*log2(x) 做低端 clamp（z<-127），缺少 fp32 路径已有的高端 guard，z≥128 时 biased exponent 重建回绕而非饱和。
- **影响对象**：数值结果（数值异常）
- **影响详情**：应为 +inf 的大结果返回 NaN，如 pow(2.0,128.0)、10.0**39.0、1000.0**15.0，无 crash；NaN 可在下游传播。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/pull/52707
- **关联**：var:ttmetal_pow_relative_error_linear_growth

### INF-0066 `ttmetal_tril_triu_mask_multiply_inf_to_nan` ttnn.tril/triu 用掩码乘法致屏蔽区 inf/NaN 变为 NaN

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-08-12 · 旧号 1.66

- **原因对象**：算子/内核 / tt-metal / ttnn.tril/triu 掩码乘法实现
- **原因行为**：ttnn.tril/triu 以 multiply(input, mask) 实现，mask 为 0/1 张量；按 IEEE-754，0 乘 ±inf 或 NaN 得 NaN，乘法无法实现 select 语义。
- **影响对象**：数值结果（数值异常）
- **影响详情**：fp32 输入在屏蔽区含 inf/NaN 时该区域输出 NaN 而非 0，如 32×32 fp32 全 -inf 张量 tril 后 496/496 个屏蔽元素为 NaN；bf16 路径因指数域窄侥幸不受影响。无 crash。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/52038

### INF-0067 `ttmetal_div_finite_by_inf_fp32_returns_nan` ttnn.div 在 fp32 DEST 累加下 a/inf 返回 NaN

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-08-12 · 旧号 1.67

- **原因对象**：算子/内核 / tt-metal / ttnn.div residual 精化路径
- **原因行为**：启用 fp32 DEST 累加时，ttnn.div 先以 reciprocal 乘被除数得到正确商，随后的 residual 精化路径把该结果污染为 NaN。
- **影响对象**：数值结果（数值异常）
- **影响详情**：fp32 DEST 累加下对任意有限 a，div(a, inf) 返回 NaN 而非 IEEE-754 要求的 0，无 crash；依赖除以 inf 做数值稳定化的计算（如 attention 归一化、LayerNorm）可能把 NaN 传播到整个激活。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/52114

### INF-0068 `ttmetal_bf16_add_sub_inf_minus_inf_returns_zero` ttnn bf16 加减 Inf−Inf 返回 0 掩盖数值异常

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-08-12 · 旧号 1.68

- **原因对象**：算子/内核 / tt-metal / binary_ng bf16 add/sub 路径
- **原因行为**：bf16 add/subtract 走 binary_ng kernel 路径，该路径对 Inf−Inf 计算出 0.0，违反 IEEE-754（应为 NaN）；fp32 路径正确。
- **影响对象**：数值结果（静默错误）
- **影响详情**：无 crash、无 NaN：本应以 NaN 传播暴露的数值异常被掩盖，错误结果看起来合法，依赖 NaN 做异常检测的下游无法察觉。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/52141

### INF-0069 `ttmetal_sharded_softmax_subblock_w_exceeds_dest_silent_wrong` sharded softmax 不校验 subblock_w 超 Dest 容量

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-12 · 旧号 1.69

- **原因对象**：算子/内核 / tt-metal / softmax subblock_w 边界检查
- **原因行为**：sharded softmax 对 subblock_w 无 bounds 检查，subblock_w 超出 Dest 容量时 partial reduction 阶段丢弃超出 Dest 的数据。
- **影响对象**：数值结果（静默错误）
- **影响详情**：用户传入合法但超出硬件 Dest 限制的 subblock_w 时，softmax 归一化结果静默错误，无 crash、无报错。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/52040

### INF-0070 `ttmetal_sort_row_major_dram_multicore_wrong_values` ttnn.sort ROW_MAJOR DRAM 多核路径返回错误值

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-12 · 旧号 1.71

- **原因对象**：算子/内核 / tt-metal / ttnn.sort ROW_MAJOR 多核路径
- **原因行为**：ttnn.sort 多核路径的内存布局/分片逻辑在 ROW_MAJOR 输入下与 single-core 或 TILE_LAYOUT 路径不一致。
- **影响对象**：数值结果（静默错误）
- **影响详情**：ROW_MAJOR 输入走 DRAM 多核路径时排序结果静默错误，无 crash。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/52532

### INF-0072 `ttmetal_minimax_h3_audio_tsharded_conv_pre_returns_uninitialized_memory` MiniMax-H3 T-sharded conv_pre 返回未初始化内存

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-08-13 · 旧号 1.73

- **原因对象**：算子/内核 / tt-metal / conv_pre 的 T-sharded 实现
- **原因行为**：MiniMax-H3 audio decode 最宽的 conv_pre（2048→1024，k=7）在 T-sharded 路径返回未初始化内存；其他 conv shape 逐位一致，缺陷限于该 shape 的 sharded 实现。
- **影响对象**：数值结果；输出内容（静默错误）
- **影响详情**：无 crash；每个 T-sharded decode 都返回完全错误的音频，conv_pre 输出 absmax 2.6e+11 而参考值为 1.391（约 −204 dB）。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/pull/53073

### INF-0074 `ttm_legacy_compat_reciprocal_newton_seed_precision_loss` reciprocal 兼容内核沿用 Grayskull 牛顿初值致精度损失

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-14 · 旧号 1.75

- **原因对象**：算子/内核 / tt-metal / _reciprocal_compat_ 牛顿初值
- **原因行为**：Blackhole/Wormhole 的 _reciprocal_compat_ 为与 Grayskull 硬连线常量一致，沿用 1.442695 作 Newton-Raphson 初值，而源码注释写明理想值约为 4/3；新硬件并无此限制。
- **影响对象**：数值结果（静默错误）
- **影响详情**：在 [0.5,1) 全部 2²³ 个 fp32 值上实测（Blackhole P300）最大相对误差 1.4752e-03（约 0.38 bf16 ULP），比理想种子差 9.6 倍；经该兼容路径的 op 输出带系统性精度底噪，可淹没依赖 rsqrt 的归一化/attention 中的低幅差异。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/53106

### INF-0075 `ttm_global_avg_pool_non_tile_divisible_shape_structural_pcc_error` global_avg_pool2d 非 tile 整除形状按错误行聚合

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-15 · 旧号 1.76

- **原因对象**：算子/内核 / tt-metal / global_avg_pool2d 填充聚合
- **原因行为**：Blackhole 上 global_avg_pool2d 对非 tile 整除形状（dim2 由 7 pad 到 32）按错误的行做聚合，而不只是把填充区一起平均。
- **影响对象**：数值结果（静默错误）
- **影响详情**：resnet50 unpadded 变体形状 [1,7,7,2048] 输出量级合理但值错，PCC 仅 0.548，跨运行逐位一致、非 flaky；tile 整除形状 PCC 0.99999。以相关性阈值验收的流水线会漏检。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/53267

### INF-0077 `ttmetal_tt_assert_release_nofail_input_validation_gone` ttnn 入口用 TT_ASSERT 做输入校验在 Release 构建失效

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-16 · 旧号 1.78

- **原因对象**：算子/内核 / tt-metal / 复合 op 入口 TT_ASSERT 校验
- **原因行为**：polyval、complex_tensor、mac 三个 ttnn 用户入口用 TT_ASSERT 做输入校验，而 TT_ASSERT 由 #ifdef DEBUG 包裹，Release 构建中被整体编译掉。
- **影响对象**：数值结果（静默错误）
- **影响详情**：Release 构建下无断言、无 crash：polyval 在空系数 vector 上越界读 coeffs[0]，垃圾系数参与计算；complex_tensor 的 shape 校验消失，非法形状静默通过。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/53285

### INF-0078 `ttmetal_lerp_unbranched_endpoint_violation` ttnn.lerp 无分支公式违反 lerp(a,b,1)==b 端点语义

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-16 · 旧号 1.79

- **原因对象**：算子/内核 / tt-metal / ttnn.lerp 插值公式
- **原因行为**：_lerp 与 _lerp_overload 对标量和张量权重都只用 input+weight*(end-input) 单一公式，该式在 weight=1.0 时因浮点舍入不精确等于 end，未像 torch.lerp 那样分支保端点。
- **影响对象**：数值结果（静默错误）
- **影响详情**：33.8% 的采样对不满足 lerp(a,b,1.0)==b，无 crash、无 NaN；依赖 lerp(x,y,1)==y 的插值/推理路径产生偏差。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/53279

### INF-0079 `ttmetal_lcm_int32_api_15bit_kernel_assumption` ttnn.lcm 内核假设约15-bit操作数，int32 大值静默溢出

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-16 · 旧号 1.80

- **原因对象**：算子/内核 / tt-metal / ttnn.lcm SFPU 实现
- **原因行为**：ttnn.lcm API 接受完整 int32 范围，但 SFPU 实现 calculate_sfpu_lcm 依赖仅对约 15-bit 操作数成立的假设（gcd_body<15> 模板宽度、u16 终乘），两处均无断言或钳制。
- **影响对象**：数值结果（静默错误）
- **影响详情**：操作数超过约 15-bit 时经 u16 终乘静默回绕/溢出，LCM 结果错误；无 crash、无报错。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/53278

### INF-0083 `ttmetal_embedding_tilized_index_scratch_overrun` ttnn embedding TILE 索引 scratch 缓冲过小致越界读写

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-17 · 旧号 1.84

- **原因对象**：算子/内核 / tt-metal / embedding 索引 scratch 缓冲
- **原因行为**：EmbeddingsTilizedIndicesProgramFactory 按每次消费 16 个索引分配 index scratch buffer，kernel 却按 aligned page size 一次读入整 tile，越过缓冲。
- **影响对象**：数值结果（静默错误）
- **影响详情**：index tensor 为 TILE 布局的 embedding op 中，越界读写破坏 scratch 之后的内存（后续 CB/缓冲），下游输出静默错误；无 crash、无 NaN。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/53435

### INF-0084 `ttmetal_embedding_pad_token_face_col_slot_mismatch` PADDED embedding pad token 槽位错位致预缓存失效

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-17 · 旧号 1.85

- **原因对象**：算子/内核 / tt-metal / embedding reader 运行时参数
- **原因行为**：PADDED 下 TilizedIndices factory 把 col_offset % FACE_HEIGHT（face 列索引）当 pad token 填入 runtime-arg 槽 6，真实 pad_token 在槽 7 未读。
- **影响对象**：性能/资源（性能退化）
- **影响详情**：PADDED embedding 走 TilizedIndices factory 时，weights[pad_token] 预缓存缓存了错误的行，声明的优化路径静默不生效；输出因逐 token 比较兜底仍正确，无 crash。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/53432

### INF-0085 `ttmetal_glu_padded_logical_width_split_mismatch` ttnn GLU 族 padded/logical 坐标混用致切分错位

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-17 · 旧号 1.86

- **原因对象**：算子/内核 / tt-metal / split_tensor_for_glu
- **原因行为**：split_tensor_for_glu 从 padded 最后一维计算折半点，却把坐标交给按 logical 坐标解释的 ttnn::slice，logical 宽度非 tile 宽度倍数时切分点错位。
- **影响对象**：数值结果；隐状态（静默错误）
- **影响详情**：ttnn.glu/reglu/geglu/swiglu 在最后一维非 tile 对齐时，门控两半包含错误元素，MLP 输出静默错误；无 crash、无诊断。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/53283

### INF-0086 `ttmetal_remainder_int32_scalar_garbage_nan_pattern` ttnn.remainder int32 张量配标量除数结果全错

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `var` · 发现 2026-08-17 · 旧号 1.87

- **原因对象**：算子/内核 / tt-metal / remainder 标量 SFPU 快路径
- **原因行为**：标量重载在无输出 dtype 时走 unary SFPU 快路径，仅特判 UINT32；INT32 输入落入 float kernel，被按 float32 重解释（小正整数成 denormal≈0、负整数成 NaN 位模式）。
- **影响对象**：数值结果（静默错误）
- **影响详情**：ttnn.remainder(int32 张量, Python 标量) 对每个测试除数均错：正被除数返回 0，负被除数返回 2139095041（0x7F800001）；无 crash，下游若用作索引/掩码会二次出错。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/53389；https://github.com/tenstorrent/tt-metal/pull/53476
- **关联**：var:INF-0018

### INF-0092 `ttmetal_elwmul_dest_reuse_last_tile_face3_corruption` Blackhole ELWMUL dest-reuse 末 tile 数值损坏

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-19 · 旧号 1.93

- **原因对象**：算子/内核 / tt-metal / ELWMUL dest-reuse LLK 通路
- **原因行为**：Blackhole 上 ELWMUL 配 dest reuse（DEST_TO_SRCA/SRCB）时，累加到 DEST bank 最高 dst_index 的最后一个 tile 在 face 3 数据通路上产生错误值，与精度模式无关。
- **影响对象**：数值结果（静默错误）
- **影响详情**：填满 DEST bank 的大 shape（32×32 tile、≥8 组）下每 8-tile 组第 8 个 tile 的 face 3 偏离 golden（224 个中 28 个），小 shape 完全正确，形成小测试通过、大负载损坏的盲区；无 crash。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/53693

### INF-0095 `ttmetal_argmax_multicore_range_partition_gaps` tt-metal 多核 argmax 工作划分缺陷：越界读、挂死与错值

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-20 · 旧号 1.96

- **原因对象**：算子/内核 / tt-metal / argmax 多核工作划分与 multicast
- **原因行为**：cores 多于 blocks 时末个 core 的 read size 下溢；distribute_work_to_cores 只特判 2 个 range，≥3 个时全落入 cores0，multicast 只覆盖前两个；单核路径亦挂死。
- **影响对象**：数值结果；可用性/进度（静默错误）
- **影响详情**：multicore argmax 在 ≥3 个 core-range 或 cores>blocks 拓扑下，第 3+ range 的 core 等不到 start_sem 而挂死，或越界 NOC read 污染结果、argmax 错值，下游解码静默选错 token。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/53769

### INF-0098 `ttmetal_nop_injector_llk_unpack_sync_race_family` LLK unpack→math 同步点时序竞态族（NOP 注入暴露）

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-08-21 · 旧号 1.99

- **原因对象**：算子/内核 / tt-metal / LLK unpack→math 同步握手
- **原因行为**：WH ELWMUL dest-reuse 的 unpack→math handoff 有时序竞态：llk_unpack_AB.h 的 STALLWAIT/SEMGET 同步点延迟时，unpack 与 math 失同步，前次结果被读前被覆盖。
- **影响对象**：数值结果（静默错误）
- **影响详情**：注入器关闭时输出正常；TT-NOP 在同步点注入 NOP/STALL 时任意 shape 下数值静默 mismatch，两轮 1–100 sweep 确定性命中同样 8 个 variant；同族见 BH mul_reduce_scalar、WH generalized_moe_gate。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/53823

### INF-0101 `ttmetal_logsigmoid_sfpu_missing_branch_range_splits` ttnn.log_sigmoid SFPU 区间分裂缺分支，fp32 静默大误差

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-08-22 · 旧号 1.102

- **原因对象**：算子/内核 / tt-metal / ttnn.log_sigmoid SFPU 内核
- **原因行为**：SFPU 内核采用三段区间分裂实现：x≤−4 分支缺失使输入原样返回，x>4 正尾部只用单项级数，中段多项式在 x=0 处偏离 ln2 达 7.12e-4。
- **影响对象**：数值结果（静默错误）
- **影响详情**：fp32 log_sigmoid 全域静默误差，峰值 9.13e-3（131,755 ULP，对比 PyTorch golden）；无 crash、无 NaN。bf16 几乎不受影响，bf16-only sweep 长期掩盖该缺陷。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/53787

### INF-0102 `ttmetal_remainder_sfpu_overshoot_correction_bypass` ttnn.remainder fp32 商估计校正不足，结果错一个除数倍

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-08-23 · 旧号 1.103

- **原因对象**：算子/内核 / tt-metal / SFPU remainder 商校正逻辑
- **原因行为**：ckernel_sfpu_remainder.h 用倒数乘法估计商，截断后只做一次 −1 过冲校正；商估计偏差超过 1 时结果可为负，而迭代清理循环只在 v >= s 时运行，负误差被完全旁路。
- **影响对象**：数值结果（静默错误）
- **影响详情**：Wormhole/Blackhole 上 fp32 标量 remainder 在 |被除数/除数| ≳ 2^23 时结果错一个除数的整数倍，如 remainder(1.42606e7, 3.0) 得 1.0（应为 2.0）；无 crash、无 NaN。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/54048

### INF-0105 `ttmetal_sdpa_sum_df_bf16_to_fp32_precision_pcc_regression` SDPA sum_df 升 fp32 致 SDXL VAE PCC 回归

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-25 · 旧号 1.106

- **原因对象**：算子/内核 / tt-metal / SDPA sum_df 中间格式
- **原因行为**：PR #49948 让 SDPA online-softmax 累加和 sum_df 随 fp32_dest_acc_en 从 bf16 升为 fp32；更高精度累加改变 rounding 轨迹，与参考实现的误差结构失配，并非一致改进。
- **影响对象**：数值结果（静默错误）
- **影响详情**：Blackhole 上 SDXL VAE decoder 开启 fp32_dest_acc_en 时 PCC 明显恶化（#50999），后由 PR #50930 放宽阈值绕过，门禁不再报出但缺陷仍在（#54165 调查中）。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/54165

### INF-0106 `ttmetal_gelu_sfpu_lut_declmal_lut16_constant_mismatch` GELU SFPU LUT 十进制与十六进制常量不一致（待裁定）

`故障` · `active` · `infra_off` · `hw_specific` · `speculative` · `NEW` · 发现 2026-08-25 · 旧号 1.107

- **原因对象**：算子/内核 / tt-metal / gelu_init 分段线性 LUT 常量
- **原因行为**：gelu_init 以 imm32 打包 6 段分段线性 LUT，declmal（十进制）与 lut16（十六进制）两套常量不一致，lreg4 lo 差异极大，两者必有一错且尚未裁定；双源常量无交叉校验。
- **影响对象**：数值结果（静默错误）
- **影响详情**：待裁定。若 hex 常量错，Wormhole B0 上 GELU 在 [0.5,1.0) 段截距错约 3 个数量级，该段输出整体静默偏移，常规容差可能吞掉；若注释错则只误导后续维护。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/54196

### INF-0107 `ttmetal_pow_bw_negative_base_composite_log_nan_inf` ttnn.pow_bw 负底数梯度经复合 log 路径变为 +inf

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-08-26 · 旧号 1.108

- **原因对象**：算子/内核 / tt-metal / ttnn.pow_bw 复合公式 backward
- **原因行为**：pow_bw 的 backward 以 exp(exponent·log(a)) 类复合公式实现，未特判负底数：负底数 log(a) 产生 NaN，并在 bf16 打包时坍缩为 +inf（0x7F80）。
- **影响对象**：梯度（数值异常）
- **影响详情**：exponent=2.0 时全部 32,639 个负有限 bf16 底数的梯度返回 +inf（正确值 2a 有限），与幅值无关；正底数精确，N300/P300 逐位一致；下游依赖该梯度的训练被污染。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/54439

### INF-0108 `ttmetal_scalar_comp_kernels_lane_complement_nan_semantics_divergence` ttnn.ge/le 标量重载比较 kernel 不支持 NaN 语义

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-08-26 · 旧号 1.109

- **原因对象**：算子/内核 / tt-metal / ttnn.ge/le 标量重载比较 kernel
- **原因行为**：tensor-scalar 比较 kernel（2023）中 ge 与 lt 评估同一谓词、返回相反常数，ge=1−lt 按构造成立，无法实现 IEEE NaN 语义；tensor-tensor 重载（2026）则已显式拒绝 NaN。
- **影响对象**：数值结果（静默错误）
- **影响详情**：含 NaN 输入时 ttnn.ge/le 的 scalar 重载静默给出非 IEEE 结果，与 tensor 重载不一致，NaN 掩码/过滤逻辑随重载而变；heaviside 无法发出 NaN。Wormhole/Blackhole 表现一致。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/54447

### INF-0113 `ttmetal_exp2_fp32_missing_overflow_guard_exponent_wraparound` ttnn.exp2 fp32 精确路径缺上溢 guard 致指数回绕

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-08-27 · 旧号 1.114

- **原因对象**：算子/内核 / tt-metal / ttnn.exp2 fp32 精确路径
- **原因行为**：以 2^round(x)·2^f 重构时只有下溢 guard（e<1→0）而缺上溢 guard；x≥128.5 时 e≥256，setexp 仅保留低 8 位致指数回绕，前一行已算出的 +inf 被覆盖丢弃。
- **影响对象**：数值结果（静默错误）
- **影响详情**：fp32_dest_acc_en 下 x≥128.5 返回看似普通的有限值并静默传播（exp2(256)=1.0、exp2(255)=0.5），x∈[128,128.5) 返回 NaN，正确应为 +inf。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/54054

### INF-0114 `ttmetal_sqrt_rsqrt_negative_zero_domain_signbit_misclassification` SFPU sqrt/rsqrt 将 −0.0 误判为负域

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-08-27 · 旧号 1.115

- **原因对象**：算子/内核 / tt-metal / ckernel_sfpu_sqrt.h 负域判定
- **原因行为**：以 v_if(x < 0.0f) 判负域置 NaN，比较把 −0.0 归入负数（IEEE 中 −0.0 属定义域）；fast_and_approximate_mode 去掉守卫后直接对 −0.0 位型跑 Newton 迭代。
- **影响对象**：数值结果（数值异常）
- **影响详情**：输入 −0.0 时 fp32 sqrt→NaN、bf16 sqrt→inf、fp32 rsqrt→NaN（应 −inf）、bf16 rsqrt→+inf，快速近似模式返回 7e-20 级垃圾；P300/N300 一致，+0.0 正确；下游 RMSNorm 类计算被毒化。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/53975

### INF-0115 `ttmetal_pow_binary_zero_base_nan_exponent_silent_zero` binary pow 零底数×NaN 指数静默返回 +0

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-08-27 · 旧号 1.116

- **原因对象**：算子/内核 / tt-metal / ttnn.pow binary SFPU 路径
- **原因行为**：binary SFPU 路径的 pow 对零底数 × NaN 指数未正确传播 NaN；#53367 修复 pow(0,+inf) 时顺带把该情形的错误输出从 inf 改成 +0。
- **影响对象**：数值结果（静默错误）
- **影响详情**：ttnn.pow(zeros, NaN 指数) 在 Blackhole binary 路径返回 +0，torch 与 unary 路径均返回 NaN；NaN 哨兵被洗成合法零，下游 NaN 检测失效；padding/masked 零与 NaN 哨兵相遇即触发。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/53922

### INF-0116 `ttmetal_xlogy_bw_eq_nan_dead_where_clauses` xlogy_bw 用 eq 比较 NaN，NaN 守卫成死代码

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `var` · 发现 2026-08-27 · 旧号 1.117

- **原因对象**：算子/内核 / tt-metal / ttnn.xlogy_bw NaN 守卫
- **原因行为**：xlogy_bw 以 where(eq(x, nan), nan, grad) 守卫两条梯度，而 IEEE 下 NaN≠NaN 使 eq 恒为 0，NaN 分支永不选中，4/19 个 dispatch 成为死代码。
- **影响对象**：梯度；性能/资源（静默错误）
- **影响详情**：含 NaN 输入时 xlogy backward 梯度失去 NaN 传播语义，结果由其余 15 个 dispatch 决定；死 dispatch 白耗周期（P300 fp32 每次调用 27M core-cycles 花在入队）。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/53872
- **关联**：var:INF-0053

### INF-0117 `ttmetal_kda_invert_doubling_cancellation_collapse` KDA 倍增展开求逆在 bf16 中间量下灾难性相消

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-08-29 · 旧号 1.118

- **原因对象**：算子/内核 / tt-metal / KDA invert_doubling 求逆
- **原因行为**：UT 变换求逆由 Horner（30 次 matmul）改为倍增展开 (I+N)(I+N²)…(I+N¹⁶)；显式幂中间量在真实 chunk 上达 O(1e2) 而结果 ≤1，bf16 舍位下灾难性相消吃掉全部有效位。
- **影响对象**：数值结果；隐状态（静默错误）
- **影响详情**：Kimi-K3 真实权重下 68/69 层 KDA 输出与 torch 参考不相关（layer 1 PCC 0.696），数值合法、幅度正常，有限性检查全部漏检；仅 N 下溢≈0 的 layer 0 通过；CI 合成数据 ‖N‖ 小时测不出。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/54813

### INF-0118 `ttmetal_matmul_multicore_reuse_oversubscribed_block_silent_drop` MatmulMultiCoreReuse 块数超核数时过剩块静默不算

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-08-29 · 旧号 1.119

- **原因对象**：算子/内核 / tt-metal / MatmulMultiCoreReuse 块数校验
- **原因行为**：程序隐式假定每核恰写一个输出块，但块数超过核数时无校验：校验函数对 interleaved 输入（sharded==nullptr）提前 return，过剩输出块永不被计算。
- **影响对象**：数值结果（静默错误）
- **影响详情**：Blackhole 110 核、per_core_M=5 得 128 块时输出尾部保留 buffer 旧内容，PCC≈−0.000009，无错误无警告；若先跑的 matmul 在同 buffer 留下正确值则误判通过、像非确定。MLA wkv_b2 生产 shape 可达。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/54798

### INF-0119 `ttmetal_celu_exponent_wraparound_wrong_sign_finite_garbage` celu 的 expm1 单向 clamp 致指数回绕返回反号有限值

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `var` · 发现 2026-08-29 · 旧号 1.120

- **原因对象**：算子/内核 / tt-metal / celu 的 expm1_cw_clamped
- **原因行为**：expm1_cw_clamped 只 clamp 指数低侧（max(x, −87.0f)）无上侧限制，setexp 只写指数域低 8 位、溢出回绕而非饱和；负 alpha 经 host 预计算 1.0f/alpha 使内核有效参数无界。
- **影响对象**：数值结果（静默错误）
- **影响详情**：ttnn.celu 取负 alpha 时，golden 为 −inf 处返回符号相反的有限垃圾（+1.0 / −6.24e9），无 crash；assert_with_ulp 报告无界误差。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/54770
- **关联**：var:INF-0113

### INF-0120 `ttmetal_sdpa_decode_mask_seed_scalar_store_noc_drain_race` sdpa_decode 非零 mask 种子 store 与 NoC 回读竞态

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `var` · 发现 2026-08-29 · 旧号 1.121

- **原因对象**：算子/内核 / tt-metal / sdpa_decode fill_tile 屏障
- **原因行为**：fill_tile 对非零值以每 tile 512 次标量 volatile store 构建 mask 种子后，copy_tile 立即经 NoC 回读同地址，其间无任何屏障；零值路径有 barrier，非零路径没有。
- **影响对象**：数值结果；性能/资源（静默错误）
- **影响详情**：causal/sliding-window/block-padding mask 的非零种子路径 decode 每次可达：512 次/tile 标量 store 拖慢构建；竞态窗口内 mask 可能回读到 stale 值，致 attention mask 错误。上游 #50373 曾命中同构竞态。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/54708
- **关联**：var:INF-0009

### INF-0121 `ttmetal_quasar_softplus_threshold_fix_not_propagated` Quasar softplus 阈值边界修复未随移植传播

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-29 · 旧号 1.122

- **原因对象**：算子/内核 / tt-metal / Quasar softplus 阈值分支
- **原因行为**：softplus 在 beta*x == threshold 时仍走线性分支；WH/BH 已修复该边界，但修复未同步到 Quasar 后端各自维护的内核副本。
- **影响对象**：数值结果（静默错误）
- **影响详情**：Quasar 上 softplus 在 beta*x 恰等于 threshold 时输出偏离参考，无 crash；源 arch 的回归测试对 Quasar 无保护。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/54857

### INF-0122 `ttmetal_floor_div_host_reciprocal_exact_divisible_off_by_one` ttnn.floor_div 标量倒数舍入致整除输入差一

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-08-29 · 旧号 1.123

- **原因对象**：算子/内核 / tt-metal / ttnn.floor_div 标量倒数预计算
- **原因行为**：host 侧把除以标量改写为乘以预计算倒数 1.0f/alpha，倒数舍入误差使恰好整除的商在浮点乘法后略小于真实整数（如 6/3→1.9999999…），floor 取到 n−1。
- **影响对象**：数值结果（静默错误）
- **影响详情**：ttnn.floor_div(tensor, float) 对恰好整除的输入静默差一，非整除输入正确；随机均匀测试难以命中，而生产中的索引/分桶常落在整除边界。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/54847

### INF-0124 `ttmetal_tanh_bw_fp32_inherits_bf16_asymptotic_tail` tanh_bw fp32 路径沿用 BF16 渐近尾式致导数偏大

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-08-30 · 旧号 1.125

- **原因对象**：算子/内核 / tt-metal / tanh_bw 渐近尾式（fp32 路径）
- **原因行为**：tanh_derivative 的 |x|≥3 尾分支返回 4·exp(−2|x|)，丢掉分母 (1+exp(−2|x|))⁻²；该近似按 BF16 精度目标设计，却被 fp32 路径原样复用，BF16 穷举校验全绿使缺陷存活。
- **影响对象**：梯度（静默错误）
- **影响详情**：FLOAT32 输入 |x|≥3 时 sech² 偏大最多 65,346 fp32 ULP（相对误差 4.96e-3），|x|=3.0 处阶跃不连续；BF16 路径仅 1 ULP；训练梯度/微分校验在 |x|≈3 邻域失真。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/54882

### INF-0125 `ttmetal_digamma_fp32_shared_log_body_precision_floor` digamma fp32 渐近路径复用低精度 log 体致千分位误差

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-08-30 · 旧号 1.126

- **原因对象**：算子/内核 / tt-metal / digamma fp32 渐近路径 log 体
- **原因行为**：FLOAT32 路径在 x>102 走 Bernoulli 渐近式，调用按低精度目标实现的共享 _calculate_log_body_no_init_，log 体的精度承诺低于调用方 dtype。
- **影响对象**：数值结果（静默错误）
- **影响详情**：x>102 全域 fp32 digamma 静默千分位误差：中位 4,543、p99 8,403、最大 9,080 ULP（x=1820.104 处 7.50204 vs 7.50637），[0.01,102] 中位 1 ULP；harness 未设 -DINP_FLOAT32 曾掩蔽测量。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/54855

### INF-0128 `ttmetal_mul_reduce_scalar_zero_flag_state_leak` mul_reduce_scalar 继承泄漏的 zero-flag 状态

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-31 · 旧号 1.129

- **原因对象**：算子/内核 / tt-metal / LLK mul_reduce_scalar 初始化
- **原因行为**：mul_reduce_scalar 初始化未调用 _configure_default_zero_flag_state_ 复位，前序 copy_init() 设的 Zero_Flag PRESERVE 状态泄漏进 GAPOOL/MVMUL。
- **影响对象**：数值结果（静默错误）
- **影响详情**：copy 型 sum_reduce_scalar 后接 mul_reduce_scalar 时，denormal 输入的标量归约在硅上静默偏离语义（应 flush 为零的乘积累积出非零残差）；真硅无任何信号，仅 TTSim 以 not modeled 拒绝。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/pull/54970

### INF-0129 `ttmetal_llk_dest_src_bank_wait_drain_fix_33_residual_sites` LLK dest-reuse 裸 STALLWAIT 修复未传播，残留 33 处

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `var` · 发现 2026-09-01 · 旧号 1.130

- **原因对象**：算子/内核 / tt-metal / LLK dest-reuse 握手等待
- **原因行为**：dest-reuse 握手在 MOVD2A/MOVD2B 前只等 SRCA_VLD、不排空 math pipeline，dest 可能仍被未完成的数学读占用；#52256 只修两处，16 个文件 33 处站点仍为裸等待。
- **影响对象**：数值结果；可用性/进度（不可复现）
- **影响详情**：间歇性、依赖时序的 dest buffer 读旧值或冲突，表现为概率性数值损坏或挂起，无确定性错误信号；WH 与 BH 上使用残留站点的生产内核路径（如 transpose_dest、dest-reuse 变体）均可触发。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/55106
- **关联**：var:INF-0097

### INF-0130 `ttmetal_grid_sample_fp32_input_unrelated_values_silent` ttnn.grid_sample 对 FLOAT32 输入静默返回无关值

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-09-01 · 旧号 1.131

- **原因对象**：算子/内核 / tt-metal / grid_sample FLOAT32 内核路径
- **原因行为**：dtype 支持矩阵声明 FLOAT32 可用，但内核路径（或其数据布局/打包假设）只对 BFLOAT16 正确，fp32 路径产出与参考零相关的确定值；两代 arch 结果不同，损坏位于各自 arch 的内核实现层。
- **影响对象**：数值结果（静默错误）
- **影响详情**：FLOAT32 输入通过支持校验、不报错执行，输出与参考毫无关系；n150 与 p150b 同输入返回彼此不同的值，均有限且跨运行 bit-stable，可避开 NaN 与非确定性探针；BFLOAT16 输入正确。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/55017

### INF-0131 `ttmetal_concat_s2s_absolute_dim_rank_confusion_silent_wrong_data` ConcatS2S 按绝对 dim 判宽高，rank-3 拼接错位

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-09-01 · 旧号 1.132

- **原因对象**：算子/内核 / tt-metal / ConcatS2SMulti 工厂 dim 判定
- **原因行为**：工厂用 2==dim/3==dim 绝对字面量判定 height/width，应为 rank-relative；rank-3 width concat 被误判为 height，rank-5 sharded width 无可用路径。
- **影响对象**：数值结果（静默错误）
- **影响详情**：rank-3 张量 width concat 时分片堆叠在下方而非并排、尾列置零，shape 不变故下游不报错；大 shard 高度下错误写偏移可能 L1 越界。词元级 concat（如 embeddings 拼接）会静默换轴。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/55032

### INF-0132 `ttmetal_rms_norm_pre_all_gather_2d_grid_l1_overalloc_plus_hang` rms_norm_pre_all_gather 2D 路径过分配并挂起

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `var` · 发现 2026-09-01 · 旧号 1.133

- **原因对象**：算子/内核 / tt-metal / rms_norm 2D core grid 工厂
- **原因行为**：use_2d_core_grid 工厂复制了默认工厂（#54697 同缺陷）的 per-row x² 缓冲分配模式，按最宽行分配造成过分配；2D 划分下另有独立的同步/依赖缺陷。
- **影响对象**：可用性/进度（挂起）
- **影响详情**：宽行 fp32_dest_acc 配置下静态 L1 分配超限，构建期报错；H=W=1024 的普通 shape 上无任何输出地挂起，hang 阻塞了该 2D 路径的全部验证。仅 use_2d_core_grid=True 时触发（Wormhole 复现）。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/55075
- **关联**：var:INF-0104

### INF-0136 `ttmetal_bias_gelu_approximate_default_no_exact_escape` ttnn.bias_gelu 默认近似 GELU 且无精确模式开关

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-09-02 · 旧号 1.137

- **原因对象**：算子/内核 / tt-metal / bias_gelu dispatch 与绑定
- **原因行为**：BIAS_GELU dispatch 以 get_defines(GELU,{0}) 合入 defines，实测与近似变体逐位相同；绑定不暴露 fast_and_approximate_mode 参数，文档也未提近似。
- **影响对象**：数值结果（静默错误）
- **影响详情**：对 float64 golden 最大误差 2.34e-02（分开调用精确 gelu 为 6.8e-07，约 34000 倍），x=−3 处恰返回 0（真值 −0.00398），过零区相对误差达 100%；无 crash、无 NaN，调用方无法请求精确形式。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/55130

### INF-0137 `ttmetal_matmul_string_activation_table_forces_approximate_kernel` matmul 字符串 activation 表把 exp/log 硬编码为近似

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-09-02 · 旧号 1.138

- **原因对象**：算子/内核 / tt-metal / unary 字符串表 approximate 标志
- **原因行为**：unary 算子表中 EXP/LOG/LOG1P/LOG2/LOG10 的 approximate 标志硬编码为 true，与表内“裸名=精确”约定及 LLK/nanobind 默认相反，无任何字符串拼写可达精确内核。
- **影响对象**：数值结果（静默错误）
- **影响详情**：matmul/linear 字符串 activation（exp/log 等）与 dit_rms_norm_unary_fused 静默走近似内核：百分位相对误差、按 x/ln2 周期、不随平均消掉；log 近似对 −0.0/subnormal 出错。nightly 测试改写了调用，该表从未上设备。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/55265

### INF-0138 `ttmetal_matmul_bias_softplus_epilogue_drops_beta_reciprocal` matmul softplus epilogue 丢 beta 参数致阈值下归零

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-09-02 · 旧号 1.139

- **原因对象**：算子/内核 / tt-metal / bias+SOFTPLUS epilogue 参数
- **原因行为**：matmul/linear 的 bias+SOFTPLUS epilogue 不转发 activation_param2（beta_reciprocal）；host 已正确计算该参数，其余工厂均推送，唯 bias epilogue 丢失。
- **影响对象**：数值结果；测试判定（静默错误）
- **影响详情**：阈值以下的 softplus 输出恰为 0，非线性段整段归零，gating/Mamba 类融合用户静默损坏；nightly 用大 K 随机输入，PCC 0.994931 对阈值 0.994 通过，atol/rtol 过松，测试分布放行了该 bug。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/55266

### INF-0139 `ttmetal_quasar_cmp_zero_predicate_nan_returns_true` Quasar 比零谓词 gtz/gez/ltz/lez 对 NaN 返回真

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `var` · 发现 2026-09-02 · 旧号 1.140

- **原因对象**：算子/内核 / tt-metal / Quasar sfpu_comp 比零谓词
- **原因行为**：Quasar 的比零谓词只按符号位与“幅值≠0”分类 lane，缺少 WH/BH 同名内核已有的 |v|<=inf 拒绝项，NaN 被当作正/负数；WH/BH 的修复从未同步到 Quasar 拷贝。
- **影响对象**：数值结果（静默错误）
- **影响详情**：NaN 输入下 gtz/gez（+NaN）或 ltz/lez（−NaN）返回 1.0，流入比零谓词的掩码/过滤/select 静默极性反转；无 crash，仅 Quasar 后端出现，WH/BH 正确拒绝 NaN。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/55273
- **关联**：var:INF-0121

### INF-0140 `ttmetal_bh_unpacr_nop_wh_packed_constants_misencode` Blackhole p_unpacr_nop 沿用 WH 常量致编码相反

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-09-02 · 旧号 1.141

- **原因对象**：算子/内核 / tt-metal / BH p_unpacr_nop 常量
- **原因行为**：BH 的 p_unpacr_nop 携带三个 Wormhole 打包值常量；BH UNPACR_NOP 为 9 操作数编码且 Unpack_Pop 仅 2 位，WH 值溢出邻位，各常量编码出与名字相反的控制，且无合法性校验。
- **影响对象**：数值结果；可用性/进度（静默错误）
- **影响详情**：当前主线零引用，属潜伏缺陷；移植代码以 BH 9 操作数形式使用同名常量即触发：bank 交接静默失败产生错数据，或 SET_DVALID 致 DVALID 永不发布而挂起，无编译期/运行期诊断。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/55269

### INF-0142 `ttmetal_avg_pool2d_global_fastpath_interleaved_pad_wrong_mean` avg_pool2d 全局快路径把交错 pad 当尾部后缀致均值错

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `var` · 发现 2026-09-02 · 旧号 1.143

- **原因对象**：算子/内核 / tt-metal / avg_pool2d 全局快路径 fold
- **原因行为**：pad_to_tile 使 NHWC 每行带交错 W_pad，快路径把 (N,H,W) fold 成单一归约轴，#54410 守卫只校验 pad 数量并假设 pad 在尾部；RM reader 按逻辑行数读页、跨过交错 pad。
- **影响对象**：数值结果（静默错误）
- **影响详情**：padded ROW_MAJOR 输入满足 H*ceil32(W)==ceil32(H*W) 时输出被静默缩放：[1,2,17,64] 全 1 输入期望均值 1.0 实得 0.559，最差偏差 44%；H≤64、W≤128 内 328 个组合命中，无报错。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/55227
- **关联**：var:INF-0075

### INF-0143 `ttmetal_conv_depthwise_activation_guard_term_missing_silent_drop` depthwise conv 守卫漏掉 activation，激活被静默丢弃

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-09-02 · 旧号 1.144

- **原因对象**：算子/内核 / tt-metal / is_1d_depthwise_conv 守卫
- **原因行为**：is_1d_depthwise_conv 守卫排除了 has_bias，却漏掉同样不能融合的 activation，致带 activation 无 bias 的 conv 进入不处理 activation 的专用内核。
- **影响对象**：数值结果；隐状态（静默错误）
- **影响详情**：带 activation 且无 bias 的 1D depthwise conv 拿到前激活输出（ReLU 类恒等直通），无报错；以 depthwise conv 为主干的 MobileNet/ConvNeXt 类架构特征图系统性漂移。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/55272

### INF-0144 `ttmetal_bh_nop_injector_tiny_tiles_unpack_stallwait_hang` BH tiny_tiles unpack STALLWAIT 时序扰动下挂起

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `var` · 发现 2026-09-02 · 旧号 1.145

- **原因对象**：算子/内核 / tt-metal / BH _llk_unpack_A_ 握手
- **原因行为**：unpack 线程等待数据的 STALLWAIT 处存在 unpack→math 握手竞态，handoff 被延迟时 math 侧永久等待。
- **影响对象**：可用性/进度（挂起）
- **影响详情**：用 TT-NOP 注入器在 BH tiny_tiles_multi_op 的 unpack:2 STALLWAIT 以 sync 模式延迟 6 拍即挂起，手工插 6 个 UNPACR_NOP 同样挂起；无错数据、无 crash。同族竞态已在 WH/BH 多个融合内核出现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/55180
- **关联**：var:INF-0098

### INF-0146 `ttmetal_cbrt_magic_seed_subnormal_flush_zero_or_6306x_low` ttnn.cbrt 对 subnormal 输入返回 0 或数量级错值

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-09-03 · 旧号 1.147

- **原因对象**：算子/内核 / tt-metal / ckernel_sfpu_cbrt.h 种子
- **原因行为**：以 Moroz magic-constant 为 Newton 迭代播种，该位型重构只在正常指数域成立，subnormal 无隐含前导 1 致种子错位；SFPMAD 还可能把 subnormal 操作数 flush 成 +0。
- **影响对象**：数值结果（静默错误）
- **影响详情**：FLOAT32 每个 subnormal 输入都错：被 flush 时返回 ±0，否则偏低可达 6306 倍（29% 偏差 >1%）；真值是普通 normal 数（约 1.1e-15 至 2.3e-13），正常输入 ≤3 ULP。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/55391

### INF-0147 `ttmetal_lgamma_no_argument_shift_middle_domain_474k_ulp` ttnn.lgamma 无参数移位，0.2<x<2 误差达数十万 ULP

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-09-03 · 旧号 1.148

- **原因对象**：算子/内核 / tt-metal / sfpu_lgamma Stirling 求值
- **原因行为**：直接在输入处（x<0.5 时先反射 z=1−x）求 Stirling 渐近级数，未像库标准做法那样先递推移位到大 z；渐近级数在小 z 处不收敛。
- **影响对象**：数值结果（静默错误）
- **影响详情**：FLOAT32 在 0.2<x<2 误差达数十万 ULP：x=0.6 处 474,555 ULP，峰值相对误差 9.04e-2（x=0.5129），误差为 (x−0.5) 的偶函数；x≥2 与 x<0.15 准确，内核注释却自称 0–3 ULP。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/55356

### INF-0148 `ttmetal_binary_ng_rank7_broadcast_dim_m6_collapsed_counter_oob` binary_ng 在 rank≥7 且 dim -6 广播时越界读出错值

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-09-03 · 旧号 1.149

- **原因对象**：算子/内核 / tt-metal / binary_ng extract_nD_dims
- **原因行为**：rank>5 时 extract_nD_dims 把 dim -6 及更外维度乘成单一标量计数器，广播维落在折叠范围内时读指针步进结构被压平、越界读相邻存储；而 op validation 显式允许 dim -6 广播。
- **影响对象**：数值结果（静默错误）
- **影响详情**：ttnn.add/subtract/multiply 在 rank≥7 且沿 dim -6 广播时输出 shape 正确但值错，最小复现 6 个 tile 中 5 个错；降一秩或无广播均正确，dim -7 广播被 TT_FATAL 拒绝。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/55351

### INF-0149 `ttmetal_rms_norm_rank0_epsilon_dropped_sign_normalize` rms_norm rank-0 分支丢弃 epsilon，退化为 x/|x|

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-09-03 · 旧号 1.150

- **原因对象**：算子/内核 / tt-metal / rmsnorm.cpp rank-0 分支
- **原因行为**：rank-0 分支以 divide(input, abs(input)) 实现，epsilon 从未被读取（只转发给 rank≥1 路径），计算 x/|x| 而非 x/sqrt(x²+eps)；测试只覆盖 rank≥1。
- **影响对象**：数值结果（静默错误）
- **影响详情**：rank-0 输入 x=0 时返回 NaN，|x|<sqrt(eps) 的值返回恰为 ±1（golden 为 0 与 1 之间的小数），与注册 golden 及 torch rms_norm 分歧；无 crash。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/55347

### INF-0150 `ttmetal_argmax_comparator_nan_zero_sign_semantics_dtype_split` ttnn.argmax 比较器 NaN/±0 语义错且两 dtype 不一致

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-09-03 · 旧号 1.151

- **原因对象**：算子/内核 / tt-metal / argmax 位型比较器 helper
- **原因行为**：float32_greater 令 NaN 永不大于任何值，bfloat16_greater 无 NaN/零处理、按符号位使 +0 大于 −0；argmax 的严格大于更新与裸 == tie 消费二者，四个 reader 均继承。
- **影响对象**：数值结果（静默错误）
- **影响详情**：非 SFPU 路径上含 NaN 或 ±0 的行 argmax 索引静默出错：fp32 永不选中 NaN（torch 视 NaN 为最大），bf16 让后来的 +0 胜过早先的 −0；同输入两种 dtype 返回不同索引且都异于 torch。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/55346

### INF-0151 `ttmetal_xlogy_zero_first_operand_nan_guard_second_only` ttnn.xlogy(0,y) 在 y≤0 时返回 NaN 而非 0

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-09-03 · 旧号 1.152

- **原因对象**：算子/内核 / tt-metal / sfpu_binary XLOGY 守卫
- **原因行为**：XLOGY 分支的 NaN 守卫只检查第二操作数（in1<0 或 NaN），缺少 x==0 短路：xlogy(0,0) 走 log(0)=−inf 再乘 0 得 NaN，xlogy(0,−1) 直接命中 NaN 分支。
- **影响对象**：数值结果；训练信号（数值异常）
- **影响详情**：torch/scipy 约定 x==0 时恒返 0，ttnn 返回 NaN；交叉熵/KL 中零目标对零或负预测很常见，单个此类元素即可把整个 loss 变为 NaN；无 crash。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/55320

### INF-0152 `ttmetal_xlogy_bw_addcdiv_bw_constant_sign_infinity_zero_divisor` xlogy_bw/addcdiv_bw 零除数处梯度写常量符号无穷

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-09-03 · 旧号 1.153

- **原因对象**：算子/内核 / tt-metal / xlogy_bw、addcdiv_bw 零除数臂
- **原因行为**：零除数处 where(eqz(other), sign(grad)*inf, …) 写常量符号无穷，忽略 sign(input_a)，缺 input_a==0 的 NaN 臂；同文件 div_bw 已正确，两 op 从未对齐。
- **影响对象**：梯度（静默错误）
- **影响详情**：other==0 时梯度符号静默错误（torch 符号为 sign(grad*self)），self==0 时应返回 NaN 却返回 ±inf，错误以不同形态向下游传播；无 crash。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/55316

### INF-0153 `ttmetal_div_subnormal_denominator_reciprocal_nan` ttnn.div/div_no_nan 对 subnormal 分母返回 NaN

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-09-03 · 旧号 1.154

- **原因对象**：算子/内核 / tt-metal / div 的 SFPU reciprocal 路径
- **原因行为**：div 以 a*(1/b) 实现，SFPU reciprocal 在 subnormal 分母上产生非有限中间值；与 #52094 大指数端同 kernel 同策略，PR #52340 只修了大端且只修 Wormhole。
- **影响对象**：数值结果（数值异常）
- **影响详情**：|b|<2^-126 时 div(x,x) 返回 NaN 而非 1，div_no_nan(0,b) 同样返回 NaN，而商本为普通数；边界精确在 2^-126 正常、2^-127 NaN，bf16 下显形为 inf；无 crash。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/55294

### INF-0154 `ttmetal_hardswish_bw_inverted_boundaries_rounded_third_constant` ttnn.hardswish_bw 边界比较反置且斜率常数舍入不足致梯度错

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-09-03 · 旧号 1.155

- **原因对象**：算子/内核 / tt-metal / ttnn.hardswish_bw 分支边界
- **原因行为**：hardswish_bw 的嵌套 where 在 −3 用 lt、在 +3 用 le，与 torch 参考的 <=/< 逐边相反；另斜率常数写成 0.3333f，而前向 kernel 用正确舍入的 1/3。
- **影响对象**：梯度（静默错误）
- **影响详情**：x=±3 精确值处走错分支，x=−3 处梯度返回 −0.4999（应为 0，符号也错）；fp32 下 (−3,3) 全域偏差 336–2097 ULP，如 x=2.5 返回 1.33325。无 crash，nightly uniform(−100,100)+PCC 测试未拦截。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/55291

### INF-0155 `ttmetal_atanh_bw_zero_upstream_gradient_pervasive_nan` ttnn.atanh_bw 对上游梯度为零的元素一律返回 NaN

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-09-03 · 旧号 1.156

- **原因对象**：算子/内核 / tt-metal / ttnn.atanh_bw NaN 守卫
- **原因行为**：atanh_bw 的 where(eqz(grad), nan, …) 守卫本意处理 |x|=1 极点的 0×inf，却只判断 grad 是否为零、不检查极点条件；第二守卫只救回 x==0。
- **影响对象**：梯度（数值异常）
- **影响详情**：0<|x|<1 内上游梯度恰为零的元素一律返回 NaN；attention mask、dropout、padding、where 常产生零梯度，一个零即可使 NaN 在 backward 中扩散，进入权重更新即得 NaN 权重。无 crash。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/55290

### INF-0156 `ttmetal_rpow_bw_differentiates_x_base_instead_of_base_x` ttnn.rpow_bw 误按 pow 求导且注册 golden 同错

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-09-03 · 旧号 1.157

- **原因对象**：算子/内核 / tt-metal / unary_backward 的 rpow_bw；观测/评测 / tt-metal / rpow_bw 的 backward golden
- **原因行为**：rpow 前向是 base**x，rpow_bw 却按 pow 求导返回 grad*base*x**(base-1)，正确应为 base**x*ln(base)；注册的 backward golden 复刻了同一错误公式。
- **影响对象**：梯度；测试判定（静默错误）
- **影响详情**：两式仅在 x==1 相等，实测梯度 286068 vs 正确 628.7（约 455 倍）；对每个 x<0（非整数 base）强制返回 NaN，而真值有限。op 与 golden 同错，常规对拍测试无法发现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/55289

### INF-0157 `ttmetal_unary_chain_specialized_head_kernel_drops_tail_ops` ttnn.unary_chain 头 op 走特化 kernel 时丢弃链尾

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-09-03 · 旧号 1.158

- **原因对象**：算子/内核 / tt-metal / unary_chain 的 kernel 选择
- **原因行为**：get_compute_kernel_path 只按链头 op 选 kernel，而 hardswish/logit/lgamma 等七个特化 kernel 不含 SFPU_OP_CHAIN_0 展开点，链被截断成单个头 op。
- **影响对象**：数值结果（静默错误）
- **影响详情**：头 op 为 hardswish/logit/lgamma/logsigmoid/identity/where_tss/mac_tss 且链长>1 时后续 op 不执行，无错误无警告；如 hardswish→square 返回未平方结果（出现负值），fusion 路径输出偏移但形态正常。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/55298

### INF-0158 `ttmetal_dtype_kwarg_silently_ignored_nine_signatures_nullopt_hardcoded` 九个 ttnn 签名接受 dtype 参数却静默丢弃

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-09-03 · 旧号 1.159

- **原因对象**：算子/内核 / tt-metal / 九个 op 的 dtype 转发与池快路径
- **原因行为**：div 各 dispatch 在输出 dtype 槽硬编码 nullopt；pow/max/min/gcd/lcm/bias_gelu 标量版注释掉参数再传 nullopt；global_avg_pool2d 快路径不用 dtype。
- **影响对象**：数值结果（静默错误）
- **影响详情**：传入的 dtype= 被静默丢弃，如 div(a,b,dtype=float32) 仍返回 bf16，fp32/int32/bfloat8_b 均不生效；global_avg_pool2d 同一参数在快路径静默丢弃、普通路径 TT_FATAL 拒绝。下游按声明 dtype 消费会错误解释位型。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/55304；https://github.com/tenstorrent/tt-metal/issues/55305；https://github.com/tenstorrent/tt-metal/issues/55311

### INF-0160 `ttmetal_nop_injector_hadamard_h128_unpack_math_sync_mismatch_race` BH hadamard h128 unpack/math 同步竞态致数据错

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `var` · 发现 2026-09-03 · 旧号 1.161

- **原因对象**：算子/内核 / tt-metal / LLK hadamard h128 同步握手
- **原因行为**：Blackhole hadamard h128 融合内核的 unpack→math 握手同步存在竞态（_llk_unpack_hadamard_h128_ 的 unpack:2 STALLWAIT 等位点），math 可读到错误数据。
- **影响对象**：数值结果（静默错误）
- **影响详情**：TT-NOP 注入器在 unpack:2 STALLWAIT 施加时序扰动后输出错数据（非 hang），fidelity/normalize/dest_sync/dest_acc 全参数面 15 个组合失败；math STALLWAIT 与 unpack SEMGET 处亦有 mismatch。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/55184
- **关联**：var:INF-0098

### INF-0161 `ttmetal_logsigmoid_large_positive_bf16_exp_double_negation_inf` ttnn.log_sigmoid 大正 bf16 输入双重取负发散至 −inf

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `var` · 发现 2026-09-04 · 旧号 1.162

- **原因对象**：算子/内核 / tt-metal / SFPU logsigmoid 大正分支
- **原因行为**：ckernel_sfpu_logsigmoid.h 大正输入分支对已取负的 x 再次取负后求 exp，实际算出 exp(+x) 而非 exp(−x)，符号处理双重否定。
- **影响对象**：数值结果（数值异常）
- **影响详情**：bf16 输入 x>~172 时结果平滑发散：x=176 得 −0.977、x=256 得 −5.7e34、x≥~266 得 −inf，正确值应约为 0；x≤170 与 golden 吻合。无 crash 无警告，出现在 1.102 修复之后。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/55457
- **关联**：var:INF-0101

### INF-0162 `ttmetal_batch_norm_running_var_biased_missing_bessel` batch_norm 更新 running_var 未做 Bessel 校正

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-09-04 · 旧号 1.163

- **原因对象**：算子/内核 / tt-metal / batch_norm running 统计更新
- **原因行为**：training 模式把有偏 batch 方差（分母 m）mean_NHW(square(centered)) 直接写入 running_var，从不施加 Bessel 校正 m/(m−1)，与 PyTorch/TF 参考实现不一致。
- **影响对象**：数值结果；模型权重（静默错误）
- **影响详情**：running_var 随 batch 大小系统性偏低 (m−1)/m（m=2 时达 50%），并经 momentum 累积；推理期用偏小方差归一化，输出分布系统性偏移，小 batch 训练再部署受害最深。无 crash、无 NaN。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/55438

### INF-0163 `ttmetal_polygamma_postacc_nfactor_scale_band_zero` ttnn.polygamma n≥7 时在宽 x 带上静默返回 0

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-09-04 · 旧号 1.164

- **原因对象**：算子/内核 / tt-metal / ttnn.polygamma 级数累加
- **原因行为**：级数在未缩放域累加，n! 尺度因子在累加之后才施加，累加中间值先一步溢出/下溢。
- **影响对象**：数值结果（静默错误）
- **影响详情**：n∈[7,11] 时在一大段普通正 x 上返回精确 0.0（如 n=11、x=3142.29 返回 0，真值 1.23e-32，误差 1.678e7 ULP）；硬零之前误差已渐增，n=11 在 x=1448 即达 8.8 ULP。n≤11 属公开 API 范围，无 crash。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/55105

### INF-0164 `ttmetal_atan2_reciprocal_underflow_third_unguarded_site` ttnn.atan2 大操作数下 reciprocal 下溢致结果坍缩到轴

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `var` · 发现 2026-09-04 · 旧号 1.165

- **原因对象**：算子/内核 / tt-metal / ttnn.atan2 reciprocal 调用点
- **原因行为**：atan2 内部 1/|·| reciprocal 中间值在操作数大于 2^126 时下溢，该调用点未加守卫；#52340 的修复未横向覆盖到此处。
- **影响对象**：数值结果（静默错误）
- **影响详情**：max(|x|,|y|)>2^126 的普通操作数对返回精确 0 或精确 π/2，角度误差最大 0.785 rad（45°）；max=2^125 时仍精确，边界锐利，fp32/bf16 均可达。无 crash。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/55124
- **关联**：var:INF-0153

### INF-0165 `ttmetal_conv1d_dual_weight_prep_paths_silent_divergence` ttnn.conv1d 两条权重准备路径结果静默分歧

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-09-04 · 旧号 1.166

- **原因对象**：算子/内核 / tt-metal / ttnn.conv1d 权重准备路径
- **原因行为**：经 prepare_conv_weights 的 prepared 路径与 internal 路径文档声明可互换，但在特定 input_length 区间对 padding/tilize 状态的假设不一致。
- **影响对象**：数值结果（静默错误）
- **影响详情**：Wormhole 上 input_length∈[8193,8704] 时，同一次卷积两条路径结果静默分歧，最高达 1e37 量级，约半数受影响 case 输出仍看似有限；trace capture 只能走 prepared 路径。无 throw、无 warning。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/55545

### INF-0166 `ttmetal_int_min_residual_correction_integer_div_family` int32 除法族在 INT32_MIN 上残差校正静默出错

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-09-04 · 旧号 1.167

- **原因对象**：算子/内核 / tt-metal / int32 除法族残差校正
- **原因行为**：截断/向下取整整除先近似再以残差符号/幅值校正收尾，而 INT32_MIN 的绝对值在 int32 中不可表示，残差幅值与符号判定在该值上可能出错。
- **影响对象**：数值结果（静默错误）
- **影响详情**：WH/BH 上 div trunc/floor、remainder、fmod 及 scalar 提升路径在 INT32_MIN 输入时返回有限、符号正确但错误的值，可通过多数烟雾测试；文档化的未定义情形仅除零与 INT32_MIN/−1。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/55502

### INF-0169 `ttmetal_matmul_fused_activation_param_assembly_errors` matmul/bmm 融合 selu/softplus 激活参数装配错误

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `var` · 发现 2026-09-05 · 旧号 1.170

- **原因对象**：算子/内核 / tt-metal / matmul/bmm 融合激活参数装配
- **原因行为**：融合激活向 ActivationApplyHelper 推参数时 selu 的 alpha/scale 顺序颠倒（三处副本），无参调用的硬编码默认值与注释意图不符；bmm 融合 softplus 从不传第三参（beta 倒数），默认为 0。
- **影响对象**：数值结果（静默错误）
- **影响详情**：matmul/bmm 带 activation=selu 时全域输出错值，融合 softplus 在每个 beta 下都错，无 crash；CI 容差 rtol=15.0×K（K=512 即 768,000%）随内维放大，几乎接受任何错误，未能拦截。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/pull/55550
- **关联**：var:INF-0138

### INF-0170 `ttmetal_prod_bw_zero_input_nan_gradient` ttnn.prod_bw 输入含零元素时梯度为 NaN

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `var` · 发现 2026-09-05 · 旧号 1.171

- **原因对象**：算子/内核 / tt-metal / ttnn.prod_bw 梯度公式
- **原因行为**：prod_bw 走朴素公式 reciprocal(input)×(prod×grad)，未对输入中的零元素特判，reciprocal(0)=+inf 与 prod=0 相乘得 NaN。
- **影响对象**：梯度（数值异常）
- **影响详情**：输入含精确零元素时梯度为 NaN，而数学正确值有限（单零位置为其余元素积×grad）；NaN 梯度静默传播，微调/distillation 路径受损。无 crash。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/pull/55564
- **关联**：var:INF-0155

### INF-0171 `ttmetal_noc_mcast_incl_src_sender_sent_vs_landed_check` MoE act sender 回环多播以 SENT 代 LANDED 致竞态

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-09-05 · 旧号 1.172

- **原因对象**：算子/内核 / tt-metal / MoE act sender 回环多播同步
- **原因行为**：act sender 以 NIU_MST_NONPOSTED_WR_REQ_SENT（请求离开本 NIU）为依据把自身 INCL_SRC 回环块发布到 CB，而不等数据落回本核 L1，发送方一侧跳过 wait。
- **影响对象**：隐状态（不可复现）
- **影响详情**：Blackhole 上 compute 可能在发送方自身拷贝落地前读取该 CB slot，读到旧值或未初始化 L1，FFN 激活偶发损坏；时序依赖、非确定复现，deepseek_prefill MoE 全专家路径共享该模式。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/55551

### INF-0175 `ttmetal_blackhole_bitwise_right_shift_oob_guard_register_selector_as_modifier` Blackhole bitwise_right_shift 越界移位返回错值

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `var` · 发现 2026-09-06 · 旧号 1.176

- **原因对象**：算子/内核 / tt-metal / ckernel_sfpu_shift.h 守卫
- **原因行为**：右移范围守卫把寄存器选择符 p_sfpu::LCONST_0 传到指令修饰符位置，且缺左移兄弟实现中的 TTI_SFPCOMPC，符号扩展块在本应清零的 lane 上执行。
- **影响对象**：数值结果（静默错误）
- **影响详情**：Blackhole 上移位量超出 [0,31] 时返回错值而非 0，负移位变成左移（8>>−1=16、1024>>−8=262144）；80 个值/移位组合中 25 个错。无 crash。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/pull/55572
- **关联**：var:INF-0102

### INF-0177 `ttmetal_norm_backward_zero_slice_singular_reciprocal_nan` moreh_norm_backward 全零切片梯度为 NaN

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `var` · 发现 2026-09-06 · 旧号 1.178

- **原因对象**：算子/内核 / tt-metal / moreh_norm_backward 倒数链
- **原因行为**：按 dx=sign(x)·|x|^(p-1)·y·dy·(1/y^p) 计算且全程无 y==0 守卫，丢掉了 PyTorch 每分支都有的 masked_fill_(norm==0, 0)；p<1 时 |0|^(p-1) 同样未处理。
- **影响对象**：梯度（数值异常）
- **影响详情**：被归约切片全零（零初始化 embedding 行、padding/全 mask 位、死通道）时 recip_tile 得非有限值，与零因子相乘成 NaN（PyTorch 返回 0）；NaN 梯度静默传播，经优化器毁掉整个参数张量的下一步。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/55585
- **关联**：var:INF-0170

### INF-0178 `ttmetal_transpose_wh_height_to_width_shard_silent_all_zero` ttnn.transpose 高度分片转宽度分片时静默输出全零

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `var` · 发现 2026-09-06 · 旧号 1.179

- **原因对象**：算子/内核 / tt-metal / transpose WH sharded 路由
- **原因行为**：输入 ROW_MAJOR+HEIGHT_SHARDED、输出 WIDTH_SHARDED 时，shard 形状转换路径把每核块数 num_hw_blocks_per_core 算成 0，生成空的数据搬运计划而非报错。
- **影响对象**：数值结果（静默错误）
- **影响详情**：transpose(x,-2,-1) 及经其路由的 permute(0,1,3,2) 静默输出全零，PCC 为 0.01/0.0097/−0.0003（DRAM 输出为 1.0）；无 error 无 fatal，显式 memory_config=DRAM 即正确。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/55582
- **关联**：var:ttmetal_all_broadcast_accessor_cross_address_space

### INF-0181 `ttmetal_softplus_fp32_magic_round_constant_domain_overflow` ttnn.softplus fp32 大负输入返回 inf/NaN/巨值

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-09-08 · 旧号 1.182

- **原因对象**：算子/内核 / tt-metal / softplus_exp_negative 尾部
- **原因行为**：把 z=-a*INV_LN2 未钳位传给 _sfpu_round_to_nearest_int32_，magic 常量 c231 只在 |z|≤2^22 有效；z<-c231 时整数减法回绕产生任意位模式，指数重构上溢。
- **影响对象**：数值结果；隐状态（数值异常）
- **影响详情**：float32 下 beta*x 低于约 −8.72e6 时返回 +inf、NaN 或高达 3.28e38 的有限值（精确值应为 0）；beta=1 时约 840 万个 bit pattern 受影响，范围随 1/beta 缩放；作为激活毒化下游前向，Wormhole 同样受影响。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/55798

### INF-0182 `ttmetal_reciprocal_bh_additive_newton_term_subnormal_flush` BH reciprocal fp32 大操作数丢弃 Newton 精炼

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-09-08 · 旧号 1.183

- **原因对象**：算子/内核 / tt-metal / BH reciprocal fast_24b 精炼
- **原因行为**：Newton 精炼按加法 MAD y=t2*y+y 计算，|x|≥2^119 时精炼项 t2*y 低于 FLT_MIN 被 SFPU flush 为零，未精炼的约 7-bit SFPARECIP 种子被原样存储。
- **影响对象**：数值结果（静默错误）
- **影响详情**：fp32 dest 累加开启时，|x|≥2^119 的七个 binade 全部值带 0.56% 相对误差（90,174 ULP），2^111 起已有部分值受损；无 crash、无 NaN，下游归一化/attention 分母静默失准。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/55797

### INF-0183 `ttmetal_unary_preallocated_output_shape_guard_self_compare` ttnn unary op 预分配输出形状守卫自比较、永不触发

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-09-08 · 旧号 1.184

- **原因对象**：算子/内核 / tt-metal / unary 设备 op 形状守卫
- **原因行为**：TT_FATAL 形状守卫里 computed_output_shape 在预分配分支内再次取 output_tensor 的 logical_shape，守卫退化为自身比较 x==x，恒真。
- **影响对象**：数值结果（静默错误）
- **影响详情**：调用方传入形状不符的预分配 output_tensor 时不报错：形状过大时内核越过输入 buffer 末尾读 tile（越界读），过小时静默返回截断结果；影响全部 unary op。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/55800

### INF-0184 `ttmetal_from_torch_host_float_to_int_cast_ub_arch_divergence` 宿主浮点→整数 static_cast UB 致结果随 CPU 架构而异

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-09-08 · 旧号 1.185

- **原因对象**：算子/内核 / tt-metal / 宿主 transform_buffers 转换
- **原因行为**：from_torch/to_dtype/宿主 typecast/from_span 的浮点→整数转换对每个元素做无范围检查的 static_cast，越界、inf、NaN 值触发 C++ 未定义行为。
- **影响对象**：数值结果（静默错误）
- **影响详情**：FLOAT32/BFLOAT16 源转整数 dtype 时越界/inf/NaN 元素静默得到错值：x86_64 给 INT_MIN，aarch64 饱和且 NaN→0，两者互异且都与设备侧转换不一致；x86 上验证通过不代表 arm 上结果相同。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/55825

### INF-0185 `ttmetal_sigmoid_bw_s_cancel_zero_gradient` ttnn.sigmoid_bw 大正输入时梯度恒为 0

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `var` · 发现 2026-09-08 · 旧号 1.186

- **原因对象**：算子/内核 / tt-metal / sigmoid_bw 导数合成
- **原因行为**：导数按 s*(1-s) 由前向结果合成，s=sigmoid(x) 舍入到精确 1 后 rsub(s,1) 灾难性相消为 0，信息已不在 s 中。
- **影响对象**：梯度（静默错误）
- **影响详情**：x≥6.25（bf16）或 x≥16.75（fp32）时返回恒 0 梯度，而真导数直到 |x|=87.34 仍是正常浮点数；无 crash、无 NaN，训练/微调静默梯度消失。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/55724
- **关联**：var:INF-0170

### INF-0186 `ttmetal_expm1_bw_plus_one_cancellation_zero_gradient` ttnn.expm1_bw 大负输入时梯度恒为 0

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `var` · 发现 2026-09-08 · 旧号 1.187

- **原因对象**：算子/内核 / tt-metal / expm1_bw 导数合成
- **原因行为**：导数按 grad*(expm1(x)+1) 由前向结果合成，expm1(x) 趋近 −1 舍入后 +1 相消为零，未直接使用 exp(x)。
- **影响对象**：梯度（静默错误）
- **影响详情**：x≤−6.25（bf16）或 x≤−17（fp32）时返回恒 0 梯度，而真导数 exp(x) 直到 −87.34 仍是正常数；无 crash，与 sigmoid_bw 构成正/负饱和的镜像对。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/55723
- **关联**：var:INF-0170

### INF-0187 `ttmetal_sign_nan_returns_plus_one_poisons_backward_idioms` sign(NaN) 返回 +1，经反向习语把 NaN 梯度变成 +inf

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `var` · 发现 2026-09-08 · 旧号 1.188

- **原因对象**：算子/内核 / tt-metal / SFPU sign 内核 NaN 路径
- **原因行为**：SFPSETCC 的 NaN 路径落到 +1，sign(NaN) 返回 +1 而其自身 golden 返回 0；golden 测试的 NaN 探针从未覆盖 sign，分歧对 CI 不可见。
- **影响对象**：数值结果；梯度（静默错误）
- **影响详情**：NaN 输入被静默换成 +1；经 sign(grad)*inf 习语的 8 个 backward（acos/asin/log/logit/logiteps/reciprocal/erfinv/div_bw）在 NaN 梯度来袭时输出 +inf 而非 NaN，依赖 NaN 传播的检查失灵。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/55827
- **关联**：var:INF-0150

### INF-0188 `ttmetal_repeat_bw_threefold_failure_classes` ttnn.repeat_bw 三类输入失败：abort、空梯度列表、错形状梯度

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `var` · 发现 2026-09-08 · 旧号 1.189

- **原因对象**：算子/内核 / tt-metal / repeat_bw（moreh_sum 路径）
- **原因行为**：backward 另写一套与 forward 不共享语义的实现：假设末两维 tile 对齐、dim-2/3 分支漏返回、把分块求和写成整轴求和。
- **影响对象**：梯度（静默错误）
- **影响详情**：forward repeat 位精确时：末两维非 32 倍数则 TT_FATAL abort；dim-2/3 重复静默返回空梯度列表 []；重复维 extent>1 时返回错误形状梯度（(1,3,32,32) 得 (1,1,32,32)），均无警告。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/55799
- **关联**：var:INF-0131

### INF-0189 `ttmetal_typecast_float_int32_positive_overflow_saturates_int32_min` typecast float→int32 正溢出饱和成 INT32_MIN

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-09-09 · 旧号 1.190

- **原因对象**：算子/内核 / tt-metal / SFPU typecast 正溢出饱和
- **原因行为**：SFPU typecast 内核只加载单一饱和常量 0x80000000（INT_MIN），二补码取反仅在 in<0 时触发，正溢出 lane 没有通往 INT32_MAX 的路径。
- **影响对象**：数值结果（静默错误）
- **影响详情**：≥2^31 的 float 输入均返回 −2147483648（符号反转，如 3e9），NaN 也返回 INT32_MIN（torch 给 0），覆盖 19.34% 的 float32 域；负输入正确，自家 golden 正确 clamp，下游索引/位移拿到错误符号。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/55933

### INF-0190 `ttmetal_matmul_program_config_silent_hifi2_to_lofi_fidelity_drop` matmul 显式调度参数令 fidelity 静默降为 LoFi

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-09-09 · 旧号 1.191

- **原因对象**：算子/内核 / tt-metal / matmul fidelity 默认补偿规则
- **原因行为**：create_matmul_attributes 仅在用户未给 program_config、core_grid 且输入非低精度时把 bf16 math fidelity 提升到 HiFi2，任何显式调度参数都会关闭该补偿。
- **影响对象**：数值结果（静默错误）
- **影响详情**：matmul/linear/addmm 传 core_grid 或 program_config 且未给 compute_kernel_config 时 bf16 降为 LoFi：实测 95.0% 输出 word 改变、快 14%，文档与返回值均无提示；models/ 中 146 处调用受影响。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/55889

### INF-0192 `ttmetal_log_family_bw_signed_zero_and_polygamma_pole_sign_hardcode` ttnn log 族与 polygamma_bw 在极点返回错误符号的无穷梯度

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `var` · 发现 2026-09-09 · 旧号 1.193

- **原因对象**：算子/内核 / tt-metal / log 族 bw 与 polygamma_bw
- **原因行为**：log 族在极点用 where(ltz(grad),…) 硬编码单一无穷符号，不区分 ±0.0 的符号位；polygamma_bw 按硬编码 n 列表（2/4/6/8/10）而非 n 的奇偶性选无穷符号。
- **影响对象**：梯度（数值异常）
- **影响详情**：x=−0.0 时 log/log2/log10_bw 返回 +inf*sign(grad)，torch 为 −inf*sign(grad)；polygamma_bw 在 n=0 与偶数 n≥12 时无穷符号反转；与 SFPU 精度无关，用理想原语也复现，下游可致 NaN 或更新方向反转。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/55931
- **关联**：var:INF-0152

### INF-0194 `sglang_triton_causal_conv1d_update_state_layout_triple_mishandle` SGLang causal_conv1d_update 回退核错读卷积状态

`故障` · `active` · `infra_off` · `other_stack` · `verified` · `var` · 发现 2026-09-09 · 旧号 1.195

- **原因对象**：算子/内核 / sglang / causal_conv1d_update 回退核
- **原因行为**：Triton fallback 按有效长度 width−1 读 conv_state 头部而非尾部窗口；声明 cache_seqlens_ptr 却从不读，按线性而非循环缓冲读写状态；width=5 被接受但只处理 width 2–4。
- **影响对象**：KV状态；数值结果（静默错误）
- **影响详情**：conv_state 长于 width−1、使用循环缓冲游标或 width=5 时，Mamba/混合架构的 conv state 更新读到错误历史窗口，结果与 torch conv1d 参考不一致，无报错；JIT 内核测试不覆盖该 fallback。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/38622
- **关联**：var:INF-0131

### INF-0195 `ttmetal_slice_rm_sharded_core_mapping_bounding_box_vs_corerange_order` ttnn.slice RM sharded 按包围盒顺序分发参数致错 core

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-09-10 · 旧号 1.196

- **原因对象**：算子/内核 / tt-metal / slice RM sharded 参数分发
- **原因行为**：程序工厂按输出 shard 网格包围盒从 (0,0) 行扫描分发 runtime args、从不读 start_coord，而 shard 实际按 corerange_to_cores 的逐矩形顺序放置。
- **影响对象**：数值结果（静默错误）
- **影响详情**：shard 网格不从 (0,0) 起、包围盒含空洞或由多个矩形组成时，shard i 的参数发给不持有它的 core，slice 输出静默错误（错 shard 数据或未初始化内存）；唯一检查是 TT_ASSERT，release 构建中不运行。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/56162

### INF-0196 `ttmetal_softmax_sfpu_exp_sanitize_swallows_nan_inf_uniform_or_zero` softmax 对全 −inf 行给均匀分布、含 inf/NaN 行给全 0

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-09-10 · 旧号 1.197

- **原因对象**：算子/内核 / tt-metal / softmax 的 exp sanitize
- **原因行为**：x−max 产生的 NaN 被 SFPU exp 的 sanitize 钳位求值为 exp(−88.5) 或 flush，各 lane 得同一微小值；含 +inf 行经 sum 溢出路径得 0，NaN 未被传播。
- **影响对象**：数值结果；隐状态（静默错误）
- **影响详情**：全 −inf 行（如全 padding 的 attention 行）返回均匀 1/64≈0.0156、行和 0.97，含 +inf/NaN 行返回全 0，而 golden 与 torch 均为 NaN；无报错，fp32/bf16 与所有 program factory 均受影响。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/56056

### INF-0197 `ttmetal_llk_unpack_pipelined_dummy_publication_bh_silent_wrong_values` BH LLK unpack 流水化改动致 SDPA 输出全噪声

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-09-10 · 旧号 1.198

- **原因对象**：算子/内核 / tt-metal / BH LLK unpack 等待流水化
- **原因行为**：#55107 的 BH unpack 半边把 dummy publication 流水化（Stall_Clr_Cntrl=1），放宽一个等待，改变其与下游消费的耦合时序，而 ring-attention 类内核依赖旧耦合。
- **影响对象**：数值结果；输出内容（静默错误）
- **影响详情**：4x32 BH Galaxy 上 MiniMax-H3 ring-attention SDPA 的每个 t2va 生成都是噪声（视频 1.7–3.3 bits/px/frame、音频钉在 0 dBFS），服务返回 200，首个请求起确定性复现；revert 该 commit 即 43/43 干净。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/56120

### INF-0200 `ttmetal_quasar_copy_via_memmove_double_uncached_base_dfb_oob_zero_read` Quasar memmove 重复叠加 uncached 基址致越窗读零

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-09-11 · 旧号 1.201

- **原因对象**：算子/内核 / tt-metal / Quasar copy_via_memmove
- **原因行为**：DFB getter 自 #52769 起已返回 uncached 别名，copy_via_memmove 仍再叠加一次 MEM_L1_UNCACHED_BASE，实际读 src+2×基址，落在两个 L1 窗口之外。
- **影响对象**：数值结果（静默错误）
- **影响详情**：行宽非 8 倍数的 tiled reshape_view 走 memmove 路径：ZEBU RTL 上 753/1152 失配、736 个零，越窗读返回 0 不 fault（仅 craq-sim abort）；对齐输入 bit-exact。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/56283

### INF-0201 `ttmetal_moreh_norm_backward_pinf_ninf_ub_cast_zero_or_garbage_gradients` moreh_norm_backward 对 p=±inf 无特判致梯度归零或垃圾

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `var` · 发现 2026-09-11 · 旧号 1.202

- **原因对象**：算子/内核 / tt-metal / moreh_norm_backward p 特判
- **原因行为**：backward 工厂对 p=±inf 无特判（forward 有），把 p、p−1 直接送入取整/小数拆分函数：static_cast<uint32_t>(±inf) 为 UB，decimal=inf−inf 得 NaN。
- **影响对象**：梯度（静默错误）
- **影响详情**：ttsim bf16 确证：p=+inf 时 argmax 元素梯度全为 0.0，部分非 max 元素带高至 1.5e38 或 inf 的垃圾；p=−inf 时非 argmin 元素残留非零梯度，最大达全 dy 幅度；无 crash。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/56248
- **关联**：var:INF-0177

### INF-0202 `ttmetal_quasar_reshard_same_width_dual_risc_shared_scratch_dfb_clobber` Quasar 同宽 reshard 双 RISC 共享 scratch 互踩

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-09-11 · 旧号 1.203

- **原因对象**：算子/内核 / tt-metal / Quasar 同宽 reshard reader
- **原因行为**：同一 core 双 DM RISC 跑同一 reader，都写 scratch+src_offset，src_offset 在每个远端 core 归零且无 per-RISC 偏移；mainline #53028 修复未带到 Quasar。
- **影响对象**：数值结果（静默错误）
- **影响详情**：HEIGHT_SHARDED→HEIGHT_SHARDED、L1 输出、非对齐 RM 行宽（如宽 3 的 bf16）时，64 行按 32+32 分给两 RISC 且都从 src_offset=0 写，输出在 [0,0,31,2] 处失配，无 crash。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/56226

### INF-0203 `ttmetal_quasar_unpacr_nop_flavour_field_carries_blackhole_encoding` Quasar p_unpacr 的 NOP flavour 沿用 BH 编码

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `var` · 发现 2026-09-11 · 旧号 1.204

- **原因对象**：算子/内核 / tt-metal / Quasar LLK p_unpacr 参数表
- **原因行为**：p_unpacr 参数表携带 BH 的 Unpack_Pop 编码，而 Quasar 该位是 Nop_type，值合法但语义错位：UNP_CLRSRC（值 1）在 Quasar 上实为 cycle-delay NOP。
- **影响对象**：数值结果（静默错误）
- **影响详情**：llk_unpack_dummy 请求的 SrcA 清除从未发生，前置 STALLWAIT 等待不存在的清除，指令照常执行不 fault；当前 in-tree 调用点恰取值 0/2 碰巧正确（潜伏缺陷），新调用点按字面语义使用即中招。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/56204
- **关联**：var:INF-0140

### INF-0204 `ttmetal_quasar_batched_pack_tile_ignores_dfb_stride_entry_collapse` 批量 pack_tile 忽略 DFB stride 致 tile 塌缩丢失

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-09-11 · 旧号 1.205

- **原因对象**：算子/内核 / tt-metal / LLK get_output_tile_index
- **原因行为**：批量 pack_tile 计算批内 tile 地址时每 tile 只推进 1 个 entry，未按 DFB 契约以 stride_size_tiles 步进，与消费端地址计算分歧。
- **影响对象**：数值结果（静默错误）
- **影响详情**：Quasar 上 STRIDED DFB 批量 pack_tile 时批内连续 tile 塌缩到同一偏移被覆写，N tile 只覆盖 ceil(N/S) 个槽，输出出现整块陈旧/缺失数据，无 crash；单 tile 打包不受影响。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/56194

### INF-0205 `vllm_triton_attention_softcap_exp_quotient_overflow_nan` vLLM attention softcap 手工 tanh 展开溢出致 NaN

`故障` · `active` · `infra_off` · `yes` · `verified` · `var` · 发现 2026-09-13 · 旧号 1.206

- **原因对象**：算子/内核 / vllm / Triton 注意力 apply_softcap
- **原因行为**：apply_softcap 把 x·tanh(S/x) 手工展开为 exp 商式，未用数值稳定形式也未钳位中间 exp；|S/x| 超过约 88 时分子分母同时溢出为 inf，商为 inf/inf=NaN。
- **影响对象**：数值结果；输出内容（数值异常）
- **影响详情**：softcap 模型（如 Gemma-2，softcap=50）pre-softcap score 超过约 4400 时 attention 整行输出 NaN，生成静默崩坏，无 crash；unified/DiffKV/INT4 三个 Triton attention 内核均受影响。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/56578
- **关联**：var:INF-0161

### INF-0206 `ttmetal_bf16_trunc_floor_ceil_frac_nan_to_inf_sign_rewrite` ttnn bf16 舍入类算子把 NaN 改写为 ±inf

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `var` · 发现 2026-09-13 · 旧号 1.207

- **原因对象**：算子/内核 / tt-metal / bf16 舍入类 LLK（含 frac）
- **原因行为**：bf16 舍入 LLK 未按 IEEE 754 NaN 传播约定处理非有限输入，把 NaN 位模式当大数截断，按符号位返回 ±inf；frac 对 NaN 与 ±inf 一律返回 +inf。fp32 路径为独立实现。
- **影响对象**：数值结果（静默错误）
- **影响详情**：bf16 trunc/floor/ceil/frac 输入含 NaN/±inf 时无 crash，NaN 被静默改写为 ±inf 或 +inf，以 NaN 作掩蔽/哨兵的下游失去检测锚点；fp32 结果全对，同一算子跨 dtype 语义分裂；bf16 对 −0.0 还返回 +0.0。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/56414
- **关联**：var:INF-0196

### INF-0207 `ttmetal_bitwise_right_shift_uint32_arithmetic_shift_half_domain_wrong` ttnn 右移算子对 UINT32 做符号扩展且 golden 同错

`故障` · `active` · `infra_off` · `yes` · `verified` · `var` · 发现 2026-09-13 · 旧号 1.208

- **原因对象**：算子/内核 / tt-metal / LLK calculate_right_shift；观测/评测 / tt-metal / integer_golden.py 黄金值
- **原因行为**：LLK calculate_right_shift 只给 UInt16 走无符号分支，UInt32 落入 vInt 符号扩展分支、高位补 1；测试 golden 函数 integer_golden.py 复刻了同样的算术右移。
- **影响对象**：数值结果；测试判定（静默错误）
- **影响详情**：UINT32 输入 bit31 置位（值域一半）时右移结果静默错值且貌似合理，如 0x80000000>>31 返回 4294967295（应为 1），均匀采样 50% lane 错，无 crash；现有单测以同错 golden 为准，CI 无法发现，纯 Python 即可复现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/56385
- **关联**：var:INF-0175

### INF-0208 `ttmetal_moreh_softmin_dead_software_define_inf_rows_all_zero` ttnn moreh.softmin 编译开关失效致含 ±inf 行全零

`故障` · `active` · `infra_off` · `yes` · `verified` · `var` · 发现 2026-09-13 · 旧号 1.209

- **原因对象**：算子/内核 / tt-metal / moreh softmin 程序工厂与内核
- **原因行为**：程序工厂定义 SOFTMIN=1，但 kernels 中没有任何 #ifdef SOFTMIN 分支消费该宏，全部内核按 softmax 变体计算，e=−(x−m) 对 inf 输入产生 NaN 后被吞为零。
- **影响对象**：数值结果（静默错误）
- **影响详情**：归约维含 +inf 或 −inf 的行静默返回全零（torch 分别返回良定义分布/全 NaN 行），无 crash；有限输入与正确实现一致，各并行路径一致出错；同 setup 下 moreh_softmax/moreh_logsoftmax 处理正确。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/56371
- **关联**：var:INF-0201

### INF-0209 `ttmetal_pad_single_core_row_major_ignores_front_padding` ttnn.pad 单核 row-major 工厂忽略前向填充

`故障` · `active` · `infra_off` · `yes` · `verified` · `NEW` · 发现 2026-09-13 · 旧号 1.210

- **原因对象**：算子/内核 / tt-metal / ttnn.pad 单核 RM reader
- **原因行为**：单核 row-major reader kernel 从不读取 input_tensor_start（前向填充量），只实现尾部填充路径；工厂静默接受不支持的前缀语义而不 TT_FATAL。
- **影响对象**：数值结果（静默错误）
- **影响详情**：ROW_MAJOR、use_multicore=False 且前向填充>0 时，pad 值被追加到 stick 尾部而非前缀，数据整体错位，无任何错误；多核默认工厂正确，tt-mlir 恒用多核、现有单核测试前向填充全为零，该路径零覆盖。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/56323

### INF-0213 `ttmetal_gdn_chunk_decay_exp_factorization_ftz_state_nan_wipe` ttnn GDN 衰减因子指数拆分在 FTZ 下毁坏循环状态

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-09-14 · 旧号 1.214

- **原因对象**：算子/内核 / tt-metal / chunk_gdn_prep 衰减因子计算
- **原因行为**：把可正常表示的 dl=exp(g_sum) 拆成 exp(g_sum−decay)·exp(decay) 两个指数之积，中间量在 Tensix FTZ/DAZ 下各自下溢为 +0.0 或上溢为 +inf，0×inf 即 NaN。
- **影响对象**：KV状态；数值结果（数值异常）
- **影响详情**：强负门控（g0≤−87.34）时，若 g_sum−g0≥88.72 则 recurrent state 被 NaN 不可逆毒化、后续 chunk 全输出 NaN，否则 state 归零造成灾难性遗忘；无 crash，长序列/强衰减模型最易命中，WH B0 硅片与 BH ttsim 验证。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/56550

### INF-0214 `ttmetal_gelu_tanh_variant_negative_tail_cancellation_zero` ttnn gelu Tanh 变体负尾灾难性相消致输出归零

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `var` · 发现 2026-09-14 · 旧号 1.215

- **原因对象**：算子/内核 / tt-metal / ckernel_sfpu_gelu Tanh 变体
- **原因行为**：负尾处 tanh(u) 距 −1 仅 1–2 ULP，half_x*t + half_x 成为两个近等量相减，结果有效位全部来自 tanh 的末 1–2 ULP，发生灾难性相消。
- **影响对象**：数值结果（静默错误）
- **影响详情**：x<−6 时输出恒为 0，[−6,−4] 相对误差最高 196.6%，fp32 与 bf16 同病，无 crash 无 NaN；处于门控/掩蔽路径时形成系统性偏置；与 erf 变体同类问题（#56490）为独立内核函数，需分别修复。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/56517
- **关联**：var:INF-0161

### INF-0215 `ttmetal_pow_partial_fma_arg_reduction_cancellation_near_base_one` ttnn fp32 pow 部分融合 FMA 约化在底数近 1 时相消失精

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-09-14 · 旧号 1.216

- **原因对象**：算子/内核 / tt-metal / pow log2 约化（binary/unary）
- **原因行为**：log2 约化写成 z = m*recip − recip，按真 FMA 无限精度假设视为与 (m−1)*recip 等价，但 SFPMAD 仅部分融合（积只多 4 位），底数近 1 时相消只剩约 4 位有效位。
- **影响对象**：数值结果（静默错误）
- **影响详情**：底数近 1.0 且指数大时 pow 静默错值，误差随指数放大：指数 1000 时 250 ULP，3e6 时相对误差 4.37%，远超文档 3 ULP 界；无 crash、无 NaN，binary 与 unary 两内核均受影响。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/56518

### INF-0216 `ttmetal_where_logical_not_signed_zero_predicate_branch_flip` ttnn fp32 where/logical_not 把 −0.0 判为非零

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `var` · 发现 2026-09-14 · 旧号 1.217

- **原因对象**：算子/内核 / tt-metal / where/logical_not SFPU 判零
- **原因行为**：fp32 谓词按 INT32 位模式判零（SFPSETCC LREG_EQ0），未按 ISA 要求先把 −0.0 冲刷为 +0；#43831 对零比较算子的修复未覆盖 calculate_logical_not 所在头文件。
- **影响对象**：数值结果（静默错误）
- **影响详情**：谓词含 −0.0 的 lane 静默选择 true 分支（IEEE、torch 与 golden 均选 false），logical_not(−0.0) 返回 0，掩蔽/选择逻辑翻转且无任何信号；仅 fp32 路径暴露，bf16 因 unpacker 预先冲刷 −0.0 不受影响。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/56522
- **关联**：var:INF-0108

### INF-0219 `ttmetal_topk_nan_sign_magnitude_bit_order_minus_nan_drops_out_of_topk` ttnn.topk 按位模式排序致 −NaN 掉出 top-k

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-09-15 · 旧号 1.220

- **原因对象**：算子/内核 / tt-metal / ttnn.topk SFPU 排序网络比较器
- **原因行为**：SFPU local-sort/merge 网络把浮点按原始位模式的 (sign, magnitude) 当整数键排序，未实现 IEEE 全序、无 NaN 分支，−NaN 因符号位被排在 −inf 之下。
- **影响对象**：数值结果（静默错误）
- **影响详情**：largest=True 且行内含 −NaN 时，NaN 静默掉出 top-k，返回值与索引集合全错（torch.topk 把 NaN 排最前），无 crash；+NaN 场景因位序巧合与 torch 一致，偶发 +NaN 测试无法发现；bf16 与 fp32 索引排列相同。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/56626

### INF-0220 `ttmetal_binary_ng_fused_activation_mixed_dtype_operand_unpack_bitfield_rotation` ttnn binary 融合激活在双操作数 dtype 不一致时错误解包

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `var` · 发现 2026-09-16 · 旧号 1.221

- **原因对象**：算子/内核 / tt-metal / ttnn binary 融合激活 unpack
- **原因行为**：融合激活路径的第二操作数直送 unpack 时未按其真实 dtype 设置解包格式，dtype 元数据只沿主操作数传播，位流按错误宽度/格式重解释，呈位域旋转形态。
- **影响对象**：数值结果（静默错误）
- **影响详情**：ttnn.add/subtract/multiply 带融合激活且两操作数 dtype 不同（如 fp32×bf16、bf16×fp8_b）时，1024 个元素中 659–1012 个错但量级貌似合理，无 crash 无 NaN；同 dtype 或拆成先运算后单独激活两步均正确。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/56738
- **关联**：var:INF-0169

### INF-0222 `flashinfer_trtllm_gen_sm103a_context_fmha_tmem_consumer_dependency_timing_perturbation` flashinfer sm103a FMHA cubin 对指令时序扰动不鲁棒

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `var` · 发现 2026-09-16 · 旧号 1.223

- **原因对象**：算子/内核 / flashinfer / sm103a context FMHA cubin
- **原因行为**：persistent context FMHA 经 TMEM 回读 softmax 行 max 统计，消费者指令对该回读存在未声明的时序依赖，指令调度变化时读到陈旧统计。
- **影响对象**：数值结果（静默错误）
- **影响详情**：B300/sm_103a 上 trtllm_ragged_attention_deepseek 的热路径 LDTM 或其前 BRA 指令被插桩/延迟时，输出约 99% 元素错（max abs diff 1e5–3.3e6），无 CUDA error、无 inf；无扰动时位级确定，标准 CI 不可见。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/flashinfer-ai/flashinfer/issues/5252
- **关联**：var:vllm_sm120_fp8_scaled_mm_xid13

### INF-0223 `ttmetal_gather_rank5_independent_fold_operand_extents_divergence` ttnn.gather rank>4 两操作数独立折叠致取错值

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-09-17 · 旧号 1.224

- **原因对象**：算子/内核 / tt-metal / gather ND→4D 折叠预处理
- **原因行为**：rank>4 时 input 与 index 各自按自身 extents 把 leading dims 折叠为 dim-0 线性 id，kernel 却把该 id 当共享坐标联合迭代；内部 leading dim 不等时两侧发散。
- **影响对象**：数值结果（静默错误）
- **影响详情**：rank>4 且 index 内部 leading dim 小于 input（host 校验允许的合法形状）时约半数输出元素静默错值，如 (2,3,4,5,64)×(2,2,4,5,64) 错 2556/5120；访问均在界内，无 assert 无 crash，形状与 dtype 正确。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/56880

### INF-0224 `ttmetal_scatter_rank5_independent_fold_row_misplacement_and_skip` ttnn.scatter rank>4 独立折叠致行错位写入与漏写

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `var` · 发现 2026-09-17 · 旧号 1.225

- **原因对象**：算子/内核 / tt-metal / scatter ND→4D 折叠与 reader
- **原因行为**：rank>4 时 input/index/src 各自按自身 extents 折叠到 4D，reader kernel 用 input 折叠坐标按 index strides 解析 index stick，两侧折叠不一致时解析到错误行。
- **影响对象**：数值结果（静默错误）
- **影响详情**：index/src 内部 leading dim 小于 input 时，源行写进错误位置、原目标行被跳过，Blackhole 实测 199–2407 元素不符，无 crash 无 assert；bf16 reduction scatter 同样中招，现有 rank-5 测试未覆盖。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/56876
- **关联**：var:INF-0223

### INF-0225 `ttmetal_concat_block_sharded_width_cursor_advances_by_shard_capacity` ttnn block-sharded concat 宽度游标按容量推进致错位

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-09-17 · 旧号 1.226

- **原因对象**：算子/内核 / tt-metal / ConcatBlockSharded 宽度游标
- **原因行为**：宽度分支以 input_shard_w*shard_grid_w（分配容量）推进每个输入的列游标，而非张量真实宽度；末宽度 shard 含 padding 时 padding 列被当作数据列逐输入累积错位。
- **影响对象**：数值结果（静默错误）
- **影响详情**：block-sharded concat dim=-1 且 W 不能被 grid 列数整除（并满足 L1 行对齐）时，首输入之后每个输入晚一列落位，间隙为陈旧 L1 数据、尾部被丢弃，约 50% 元素错，形状正确无 crash；整除形状全部通过。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/56827

### INF-0226 `ttmetal_pow_int32_scalar_exponent_fast_path_float_bit_reinterpretation` ttnn pow(int32,0–3) 快速路径返回 float 位模式

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-09-17 · 旧号 1.227

- **原因对象**：算子/内核 / tt-metal / pow 标量指数 0–3 快速路径
- **原因行为**：pow 标量 int 指数 0–3 快速路径直接调用 power_iterative，二者均无 dtype 校验，绕过 float_only 的 dtype policy；kernel 按 float 计算后写回 INT32 张量。
- **影响对象**：数值结果（静默错误）
- **影响详情**：INT32 输入且标量指数 0–3 时返回 float 位模式重解释的整数（如 1.0f→1065353216、NaN→2143289344），pow(x,1)≠x，无报错；指数≥4 被正常拒绝，float32 输入与张量指数路径不受影响。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/56853

### INF-0227 `vllm_staged_write_tensor_apply_write_int32_row_offset_wraparound` vLLM StagedWriteTensor int32 行偏移回绕越界写

`故障` · `active` · `infra_off` · `partial` · `verified` · `NEW` · 发现 2026-09-17 · 旧号 1.228

- **原因对象**：算子/内核 / vllm / _apply_write_kernel 行偏移
- **原因行为**：_apply_write_kernel 的行偏移 row_idx*row_stride 中 row_idx 为 int32 张量、row_stride 被 Triton 特化为 int32，乘积超过 2^31 时回绕为负偏移。
- **影响对象**：可用性/进度；数值结果（报错）
- **影响详情**：max_model_len 约 ≥512k（如 1M、TP8 MI308X）时首个真实请求即令 8 个 TP worker 齐崩 Memory access fault，伪装成长上下文内核 bug；回绕后若仍落在同一 allocation 内则静默写坏其他行；<512k 上下文正常。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/57030

### INF-0229 `flashinfer_blackwell_gdn_chunked_tmem_stage_released_before_tcgen05_ld_completes` flashinfer GDN 内核 TMEM stage 读未完成即释放

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-09-18 · 旧号 1.230

- **原因对象**：算子/内核 / flashinfer / GDN chunked cg1 stage 释放
- **原因行为**：compute group 1 两处 TMEM stage 在 cute.copy 发出异步 tcgen05.ld 后立即 arrive 释放，漏插 load fence，生产者下一次 tcgen05.mma 可覆写在途读的 stage。
- **影响对象**：数值结果；KV状态（静默错误）
- **影响详情**：Blackwell（sm_103a）上 GDN chunked 状态/输出被下一 chunk 数据按竞态窗口混合，静默错值，无 crash 无 NaN；vLLM Qwen3-Next 类 hybrid 模型首 prefill 即经过该内核；缺陷只见于 SASS 控制位。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/flashinfer-ai/flashinfer/issues/5334

### INF-0230 `ttmetal_groupnorm_sharded_welford_batch_stride_uses_block_hw_reads_previous_batch` groupnorm Welford 内核批步进常量错用致后续批错值

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-09-18 · 旧号 1.231

- **原因对象**：算子/内核 / tt-metal / groupnorm Welford 批步进
- **原因行为**：Welford 内核不读 compile arg 18（每批 tile 数），改用单 group tile 跨度 block_hw 作批间步进，每核多 group 时第 b 批起点提前、落进前一批数据。
- **影响对象**：数值结果（静默错误）
- **影响详情**：use_welford=True、单核持多批（num_batches_per_core≥2）且每核多 group（block_wt<per_core_Nt）时首批正确、后续每批输出全错，use_welford=False 同输入正确；报告者为静态分析推导，未上设备验证。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/56955

### INF-0234 `ttmetal_streaming_sdpa_prefill_fp32_dest_rowsum_bf16_packing_rounding_compounds` streaming SDPA 行和仍为 bf16，长序列精度逐块劣化

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `var` · 发现 2026-09-20 · 旧号 1.235

- **原因对象**：算子/内核 / tt-metal / streaming SDPA 行和格式
- **原因行为**：开 fp32_dest_acc_en 时只有 QK 分数保持 fp32，online-softmax 行和在融合 rescale 中与 bf16 输出累加器共享同一 pack 格式，bf16 舍入随 K chunk 逐块复合。
- **影响对象**：数值结果（静默错误）
- **影响详情**：causal、8 heads、d128 下 S=131072 流式 rmse 0.0097 vs legacy 0.0053（劣化 1.83×），S=8192 时与 legacy 相当；S≥~65k 长序列触发，开 fp32_dest_acc_en 的用户以为获得更高精度。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/57095
- **关联**：var:INF-0105

### INF-0237 `ttmetal_nonzero_rowmajor_sharded_reader_bank_major_page_id_double_mapping` nonzero ROW_MAJOR sharded reader 页索引双重映射

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-09-21 · 旧号 1.238

- **原因对象**：算子/内核 / tt-metal / nonzero sharded reader
- **原因行为**：reader 构造 bank-major 设备页索引交给要求逻辑 row-major 页索引的 TensorAccessor，后者再次做页→shard→bank 映射，shard 定向被计入两次。
- **影响对象**：数值结果（静默错误）
- **影响详情**：WIDTH/BLOCK sharded 输入且每 shard 多于一页行时读错页，nonzero 报告来自错误行/列的坐标；访问均在界内，无 assert；grid_w==1 或每 bank 一页时不触发。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/57163

### INF-0239 `ttmetal_topk_column_split_residual_tensix_state_poisons_next_indexer_score_run` topk column split 遗留 Tensix 状态毒化下一内核

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `var` · 发现 2026-09-21 · 旧号 1.240

- **原因对象**：算子/内核 / tt-metal / topk column split 内核
- **原因行为**：column split 内核修改的 Tensix 配置/累加器状态在程序结束后未复位，后续在同一核列上启动的无关内核继承该状态。
- **影响对象**：数值结果（静默错误）
- **影响详情**：Blackhole 上 indexer_score 紧接 topk 测试运行时 PCC 恒 0.5409，175 个错元素集中在每 core column 一个 8×8 块；单独运行、板复位后正确，中间插入 eltwise add 可恢复，结果依赖测试顺序。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/57205
- **关联**：var:INF-0128

### INF-0241 `ttmetal_unary_sfpu_sfpstore_truncates_instead_of_rne_square_vs_mul_diverge` unary SFPU 存 bf16 截断而非 RNE

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-09-22 · 旧号 1.242

- **原因对象**：算子/内核 / tt-metal / LLK unary SFPU 内核 bf16 存储
- **原因行为**：square、rad2deg 等七个 unary SFPU 内核 fp32 计算后不做软件 RNE，直接依赖 SFPSTORE 窄化存 bf16，截断低 16 位（round-toward-zero）。
- **影响对象**：数值结果（静默错误）
- **影响详情**：ttnn.square(x) 与 ttnn.mul(x,x) 在 440/1024 元素不一致；七个内核的 inexact 结果单向偏零至多 1 bf16 ULP，rad2deg(-4.5) 得 -256.0（应 -258.0）；小样本或宽容差门禁恒过。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/57408

### INF-0242 `ttmetal_slice_noninner_dim_truncation_misaligned_begin_misses_row_skip` ttnn.slice 非对齐 begin 时漏跳一行，读到应丢弃的行

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-09-22 · 旧号 1.243

- **原因对象**：算子/内核 / tt-metal / ttnn.slice row-major 游标
- **原因行为**：非最内维被截断且最内维 begin 非对齐时，内核跨批次边界推进游标少算一次整行跳跃，读取起点落在应丢弃的行上。
- **影响对象**：数值结果（静默错误）
- **影响详情**：(5,65,1025) fp32 做 slice([0,0,1],[5,64,1025]) 有 6/320 输出行错，坏行内容恰为应丢弃的前一行，形状正确且无报错；bf16 坏行降为 1，ROW_MAJOR/TILE 同错。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/57382

### INF-0243 `ttmetal_approx_tanh_sfplut_reland_e2e_vision_quality_regression_below_gate` 近似 tanh SFPLUT 致 Llama Vision e2e 质量退化

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-09-22 · 旧号 1.244

- **原因对象**：算子/内核 / tt-metal / 近似 tanh SFPLUT 拟合
- **原因行为**：#57179 reland 把 fast_and_approximate_mode 的 tanh 改为 6 项 SFPLUT 拟合，拟合误差逐元素小、在单元测试容差内。
- **影响对象**：输出内容；数值结果（质量退化）
- **影响详情**：Llama 3.2-90B-Vision 以近似 tanh 门控 cross-attn/MLP，e2e BERTScore F1 均值 0.722→0.6815 跌破 0.69 门禁，bisect 精确复现；单元/op 级测试全绿，仅 e2e 质量门可见。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/57413

### INF-0244 `ttmetal_sdpa_implicit_kv_tile_padding_reaches_output_nonmultiple32_pcc_zero` SDPA 隐式 KV tile padding 泄漏到输出

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-09-23 · 旧号 1.245

- **原因对象**：算子/内核 / tt-metal / SDPA 内核 KV 尾 tile 长度掩码
- **原因行为**：非 causal、无 attn_mask 时 SDPA 内核未对 K/V 尾 tile 的 padding 位置施加（或未完整施加）逻辑长度掩码，QK 分数与 P·V 读取都放行 padding。
- **影响对象**：数值结果（静默错误）
- **影响详情**：序列长非 32 倍数且 padding 含非零值时输出错：K/V padding 填 1e38 时 PCC −0.003、max abs err 5.5e36，仅 K 填时 PCC 0.973；不 raise、输出有限，长度对齐时全对。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/57608

### INF-0245 `ttmetal_log_bf16_accurate_mode_signed_zero_exponent_before_normalisation_plus_inf` ttnn.log bf16 accurate 模式对 −0.0 返回 +inf

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `var` · 发现 2026-09-23 · 旧号 1.246

- **原因对象**：算子/内核 / tt-metal / calculate_log_body
- **原因行为**：calculate_log_body 在 −0.0 规范化（a*1.0f+0.0f）之前读取指数、之后读取尾数，同一值的两个位域取自不同时刻，组合出错误符号的中间量。
- **影响对象**：数值结果（静默错误）
- **影响详情**：ttnn.log/log2/log10 在 bf16、默认 accurate 模式下对 x=−0.0 返回 +inf（应为 −inf），fp32 路径正确；任何下溢的负乘积/负商都会产生 −0.0。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/57538
- **关联**：var:INF-0192

### INF-0247 `ttmetal_softmax_float32_default_approximate_exp_7_5pct_element_error` softmax 对 float32 默认近似 exp，误差达 7.5%

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `var` · 发现 2026-09-24 · 旧号 1.248

- **原因对象**：算子/内核 / tt-metal / softmax 近似模式默认配置
- **原因行为**：softmax_init_compute_kernel_config 对所有输入 dtype 把近似模式硬编码为 true，float32 输入也走 bf16 级近似 exp 内核。
- **影响对象**：数值结果（静默错误）
- **影响详情**：Blackhole P150 fp32 下逐元素误差最大 7.45e-2、行和 0.96–0.97 不归一，全 ones 行每元素恒错 2.91e-2；numeric_stable、HiFi4+fp32_dest_acc_en 均不改变结果，math_approx_mode=False 可规避。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/57633
- **关联**：var:INF-0136

### INF-0248 `ttmetal_prod_fp32_srca_tf32_mantissa_dest_to_srca_roundtrip_10bit` ttnn.prod fp32 经 SrcA 截为 TF32，长乘积坍塌

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `var` · 发现 2026-09-24 · 旧号 1.249

- **原因对象**：算子/内核 / tt-metal / ttnn.prod SrcA 数据通路
- **原因行为**：输入 tile 以 UnpackToSrc 进 SrcA 只保留 TF32 10 位尾数，运行积每步经 mul_reuse_dest_tiles<DEST_TO_SRCA> 再次穿过 SrcA，fp32 精度只覆盖 DEST 累加器。
- **影响对象**：数值结果（静默错误）
- **影响详情**：float32 输入下乘积按约 10 位尾数形成并向下偏置：4096 元素错 99%，72,192 值乘积返回 3.89e-31（真值 98.9），全张量 prod() 返 denormal；(1+2⁻¹⁰) 因子被乘成 1.0。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/57634
- **关联**：var:INF-0318

### INF-0249 `ttmetal_typecast_uint32_float32_double_rounding_bit31_one_ulp` typecast uint32→float32 双重舍入致 1 ULP 偏差

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `var` · 发现 2026-09-24 · 旧号 1.250

- **原因对象**：算子/内核 / tt-metal / SFPU typecast uint32→fp32
- **原因行为**：bit 31 置位的输入先把低 31 位 SFPCAST 成 fp32（第一次 RNE），再加 2^31（第二次 RNE），第二次舍入在第一次制造的伪 tie 上可落到远侧。
- **影响对象**：数值结果（静默错误）
- **影响详情**：[2^31, 2^32) 全域 16.47% 输入返回距正确舍入值 1 ULP 的邻值（如 3221225663 得 3221225472，torch 为 3221225728）；输入 <2^31 精确，Wormhole/Blackhole 同病。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/57643
- **关联**：var:INF-0189

### INF-0250 `ttmetal_fmod_remainder_exact_zero_loses_dividend_sign` fmod/remainder 精确零结果丢失被除数符号

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `var` · 发现 2026-09-24 · 旧号 1.251

- **原因对象**：算子/内核 / tt-metal / fmod/remainder 零结果符号
- **原因行为**：fmod 的 a−trunc_div*b 对负被除数精确零得 +0.0，两个 fix-up 循环只处理非零结果，a==b 臂直接赋 0.0f；remainder 的符号校正显式跳过零结果。
- **影响对象**：数值结果（静默错误）
- **影响详情**：tensor-tensor 负被除数整除时返回 +0.0（torch 为 −0.0），如 fmod(-4.0, 2.0)；+0.0==−0.0 使现有测试与 PCC/容差门看不见，以 signbit 分支的下游（1/x、atan2）产生符号分歧。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/57714
- **关联**：var:INF-0245

### INF-0251 `ttmetal_matmul_2d_mcast_dram_width_shard_first_read_unclamped_narrow_per_core_n_garbage` 2D mcast matmul DRAM 分片首读未钳制，宽网格输出垃圾

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-09-24 · 旧号 1.252

- **原因对象**：算子/内核 / tt-metal / mcast matmul in1 首读游标
- **原因行为**：in1 DRAM width-sharded 时工厂的 DRAM bank 游标首读未用 per_core_N 钳制 worker_core_stride（后续读取有钳制），首读过冲使游标越过下一列所需 tile，其后每列读错权重。
- **影响对象**：数值结果（数值异常）
- **影响详情**：per_core_N < ceil(N/DRAM bank 数) 时（P150 8 bank 下 grid 9、13 列）全形状输出 inf/垃圾，config 照常过验证、无 raise；tt_transformers prefill 因此被限在 ≤8 列网格。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/57732

### INF-0253 `ttmetal_layer_norm_post_all_gather_bf16_stats_e_x2_minus_mean_squared_collapse` 分布式 layernorm bf16 统计相消致方差塌缩或 NaN

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `var` · 发现 2026-09-25 · 旧号 1.254

- **原因对象**：算子/内核 / tt-metal / 分布式 layernorm 方差计算
- **原因行为**：分布式 layernorm 默认路径按 E[x²]−E[x]² 计算方差，统计默认 bf16、中间 tile 也存 bf16；fp32_dest_acc_en 时 stats 输入仍经 bf16 舍入，相减发生灾难性相消。
- **影响对象**：数值结果；隐状态（数值异常）
- **影响详情**：interleaved 路径下 mean≫std 激活（σ≲μ/23）方差塌缩为 0，输出 ±10⁶ 量级；fp32_dest_acc_en 时方差可为负，整行 NaN；σ≲μ/7 即超 10% 方差误差，与 eps 无关；来源为无设备内核算术模拟。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/57861
- **关联**：var:INF-0230

### INF-0255 `ttmetal_moreh_norm_no_max_rescale_subnormal_pow_flush_zero` ttnn.moreh_norm 无重缩放致小幅向量范数被清零

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `var` · 发现 2026-09-25 · 旧号 1.256

- **原因对象**：算子/内核 / tt-metal / ttnn.moreh_norm 复合归约链
- **原因行为**：按朴素 (Σ|x|^p)^(1/p) 直接形成 |x|^p，无 max|x| 重缩放；|x|<1 且 p 较大时中间量落入 fp32 subnormal 域，被 Tensix DAZ/FTZ 清零。
- **影响对象**：数值结果（静默错误）
- **影响详情**：小幅度向量的范数静默返回精确 0，同 dtype 的 torch.linalg.vector_norm 返回正常有限值（如 p=2、[1e-20,3e-20]；p=16、[0.002,0.003]）；以范数做归一化/裁剪/相似度的管线分母为 0 或触发除零分支。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/57869
- **关联**：var:INF-0182

### INF-0256 `ttmetal_expm1_log1p_minus_zero_signed_zero_lost_reconstruction_gate` ttnn.expm1/log1p 对 −0.0 输入返回 +0.0 丢失零符号

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `var` · 发现 2026-09-25 · 旧号 1.257

- **原因对象**：算子/内核 / tt-metal / SFPU expm1、log1p 内核
- **原因行为**：accurate 路径下 expm1 在 j=0 时跳过重建门，返回值 r·s+u 中 +0.0 与 −0.0 相加销毁零符号且无恢复步骤；log1p 首句 u=a+1.0 即丢失输入零符号；WH/BH 双副本相同。
- **影响对象**：数值结果（静默错误）
- **影响详情**：−0.0 输入经 expm1/log1p 返回 +0.0（IEEE 与 torch 为 −0.0），+0.0 输入一致，fp32/bf16 同病；依赖 signbit 的下游（1/x→±inf、atan2、区间边界）出现符号级分歧，数值容差门恒绿。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/57847；https://github.com/tenstorrent/tt-metal/issues/57848
- **关联**：var:INF-0250

### INF-0257 `ttmetal_addcmul_bw_grad_times_tensor_before_scalar_intermediate_inf` ttnn.addcmul_bw 先乘两大操作数致中间溢出返回 inf 梯度

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `var` · 发现 2026-09-25 · 旧号 1.258

- **原因对象**：算子/内核 / tt-metal / ttnn.addcmul_bw 反向实现
- **原因行为**：ternary_backward.cpp 中 grad_a/grad_b 两处按 (grad×tensorN)×value 顺序相乘，先乘两个大操作数；未像姊妹 addcdiv_bw 与前向 addcmul 那样先折叠 value。
- **影响对象**：梯度（数值异常）
- **影响详情**：grad 与 tensor 幅值大时 grad×tensorN 超出 FLT_MAX 变 ±inf，返回 inf 梯度，而数学梯度是普通有限 fp32（value 小）；inf 沿反传链污染，遇 0 再生成 NaN。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/57867
- **关联**：var:INF-0170

### INF-0258 `ttmetal_floor_div_int32_scalar_truncated_reciprocal_allzero_float_floor_on_int_bits` ttnn.floor_div INT32 张量÷标量路径全域静默错值

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `var` · 发现 2026-09-26 · 旧号 1.259

- **原因对象**：算子/内核 / tt-metal / floor_div 标量路径、floor
- **原因行为**：INT32÷标量复合路径直接 floor(multiply(input,1/value))：乘法把小于 1 的浮点标量截断为 0；ttnn.floor 无 dtype 校验，对 INT32 原始位型跑浮点 SFPU 内核。
- **影响对象**：数值结果（静默错误）
- **影响详情**：INT32 tensor÷Python int 标量的 floor_div 输出多为 0：s=2/3/7/−2 错误率 100%、s=1000 为 99.6%、s=1 约 50%；输出 dtype 正常且无任何信号，同输入 ttnn.div(rounding_mode="floor") 全对。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/58000
- **关联**：var:INF-0122

### INF-0259 `ttmetal_typecast_int_to_bf16_ties_away_from_zero_dual_rounding_rule_split` ttnn.typecast 整数→bf16 的 tie 远离零舍入

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `var` · 发现 2026-09-26 · 旧号 1.260

- **原因对象**：算子/内核 / tt-metal / ttnn.typecast 整数→bf16 臂
- **原因行为**：整数源→BFLOAT16 臂经 SFPCAST 转 fp32 后复用硬件 rounder（tie 远离零），而同 op 的 fp32→bf16 臂为软件 RNE，同一算子按源 dtype 施加两种舍入规则。
- **影响对象**：数值结果（静默错误）
- **影响详情**：整数转 bf16 时恰好 tie 的值有一半比 torch 结果大 1 ULP，偏差方向系统性向大、不抵消；[256,65535] 穷举 1024 个 tie 中 512 个错，[256,512) 内 1/4 整数错；非 tie 输入全对。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/58001
- **关联**：var:INF-0249

### INF-0261 `ttmetal_matmul_1d_in0_mcast_sharded_output_out_block_w_lt_per_core_n_wrong` matmul in0-mcast sharded 输出子块偏窄致错值

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `var` · 发现 2026-09-27 · 旧号 1.262

- **原因对象**：算子/内核 / tt-metal / matmul 工厂 sharded 输出与验证器
- **原因行为**：输出子块验证谓词把 out_block_h==1 当作普适豁免，放行 sharded 输出下 out_block_w<per_core_N 的配置；而 partials 跨 N 块原地共享输出 shard，写路径与之不符。
- **影响对象**：数值结果（静默错误）
- **影响详情**：width-sharded L1 输出且 out_block_w<per_core_N 时 PCC 降至 0.10–0.63，比值越小越差；out_block_w==per_core_N、interleaved 或 DRAM 输出全对，稀疏测试易漏网；2D 工厂同缺陷。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/58046
- **关联**：var:INF-0118

### INF-0262 `ttmetal_binary_ng_rowmajor_mixed_float_dtypes_both_operands_at_operand_a_stride` binary_ng 行主序混合 dtype 共用 A 的 stride

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `var` · 发现 2026-09-28 · 旧号 1.263

- **原因对象**：算子/内核 / tt-metal / binary_ng 行主序 reader 步长
- **原因行为**：行主序 binary_ng 的 host 端只打包一个按 A 元素大小计算的 reader stride，reader 对两个 circular buffer 用同一字节偏移、长度与 L1 行距，未按 B 的元素大小缩放。
- **影响对象**：数值结果（静默错误）
- **影响详情**：bf16 与 fp32 混合（任一顺序）的行主序二元运算逐行错值：fp32 操作数行的后半被换成下一行前半；行/列/标量/混合广播均失败，同输入 TILE 布局正确。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/58078
- **关联**：var:INF-0220

### INF-0263 `ttmetal_matmul_reuse_tiny_a_tiles_blockfloat_b_multi_kblock_stale_reads` matmul Reuse 在矮 A tile 多 K 块时读陈旧数据

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `var` · 发现 2026-09-28 · 旧号 1.264

- **原因对象**：算子/内核 / tt-metal / matmul Reuse 工厂多 K 块累加
- **原因行为**：Reuse 工厂在 A tile 高度<16、B 为 block-float 且 K 拆多块时，分块 K 累加路径读取未初始化/陈旧数据；验证器接受该配置（同条件下 mcast 配置被拒）。
- **影响对象**：数值结果（静默错误）
- **影响详情**：满足 in0_block_w<Kt 时 PCC 降至 0.75 直至 −0.0035，错值随运行变化，复现输出见 NaN；单 K 块、bfloat16 B 或 32 行 A tile 均正确，按验证器指引选配置恰好落入坏路径。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/58181
- **关联**：var:INF-0118

### INF-0264 `ttmetal_packer_l1_acc_ignored_two_kblocks_partials_spilled_in_output_format` matmul 恰 2 个 K 块时静默关闭 packer_l1_acc

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-09-28 · 旧号 1.265

- **原因对象**：算子/内核 / tt-metal / packer_l1_acc_en 谓词
- **原因行为**：三工厂的 packer_l1_acc_en 谓词无 bias 时要求 num_blocks>2，恰 2 块即关闭 L1 累加；partials CB 格式绑定该 flag，关闭后部分和以输出格式落盘。
- **影响对象**：数值结果（静默错误）
- **影响详情**：2D mcast/1D in1-mcast/Reuse 工厂下 block-float 输出且 K 恰拆 2 块时部分和被舍入到 block float，精度劣化（bf4_b PCC 0.962，4 块为 0.973），2 块反比 4 块更不准；无任何警告。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/58177

### INF-0265 `ttmetal_datamovement_fp32_to_bf16_cast_truncates_rtz_vs_typecast_rne_split` 数据搬运 op 的 fp32→bf16 转换截断而非 RNE

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-09-29 · 旧号 1.266

- **原因对象**：算子/内核 / tt-metal / 数据搬运 op 的 datacopy 路径
- **原因行为**：clone/copy/to_memory_config/重分片等数据搬运 op 带 dtype= 的 fp32→bf16 转换在 datacopy 路径直接截断尾数（向零舍入），不做 RNE。
- **影响对象**：数值结果（静默错误）
- **影响详情**：约 50% 元素比 RNE 参照偏小一档，最大误差 1 ulp、平均带号误差 −0.5 ulp，Wormhole 与 Blackhole 一致；同一张量经 ttnn.typecast 或 from_torch 得 RNE 结果，不同入口数值不一致。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/58371

### INF-0266 `ttmetal_datamovement_int_float_dtype_conversions_bit_reinterpretation_garbage` 数据搬运 op 整数↔浮点 dtype 转换按位重解释产出垃圾值

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `var` · 发现 2026-09-29 · 旧号 1.267

- **原因对象**：算子/内核 / tt-metal / 数据搬运 op 的 datacopy 路径
- **原因行为**：clone/copy/to_memory_config/重分片/tilize/to_layout 等数据搬运 op 的 dtype= 跨类转换经 datacopy kernel 直接搬运位型、不做数值转换，校验却全部接受。
- **影响对象**：数值结果（静默错误）
- **影响详情**：整数↔浮点及窄整型转换后输出与输入无关：[0,255] 精确可表示整数实测 99.5–99.8% 元素错，12 种组合全错，无任何报错；同张量用 ttnn.typecast 转换精确。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/58372
- **关联**：var:INF-0226

### INF-0267 `ttmetal_binary_ng_shard_aliasing_on_partial_layout_match_silent_wrong_answer` binary_ng shard 别名判定只比部分布局属性致静默错值

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-09-29 · 旧号 1.268

- **原因对象**：算子/内核 / tt-metal / is_native_L1_sharding 谓词
- **原因行为**：is_native_L1_sharding 判定 shard 能否原地别名为 CB 时，tensor-tensor 分支只比 shard grid，scalar 与 subtile 广播分支只比 ShardSpec，未比完整布局。
- **影响对象**：数值结果（静默错误）
- **影响详情**：L1-sharded 输入而输出分片方案不同（height→width 同 grid、height→block 同 ShardSpec）时内核按错误布局读数，n150 上 add PCC 0.22–0.52，无警告；全 operand 同 memory config 时正确。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/58467

### INF-0268 `ttmetal_fused_scale_mask_softmax_tile_padding_participates_in_denominator_non32_widths` scale_mask_softmax 非 32 倍宽度 padding 进分母

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `var` · 发现 2026-09-29 · 旧号 1.269

- **原因对象**：算子/内核 / tt-metal / scale_mask_softmax 融合小核
- **原因行为**：融合小核（softmax.cpp）在应用用户 mask 后未对最后 tile 消费 MASK_PADDED_DATA，padding 列没有隐式 -inf mask，参与 softmax 分母。
- **影响对象**：数值结果；隐状态（静默错误）
- **影响详情**：逻辑宽度非 32 倍数（如 17/50/197/1001）且提供用户/causal mask 时，输出行和不为 1，最低约 0.04，概率质量漏进 padding 列，注意力输出塌缩；32 对齐宽度、plain ttnn.softmax 与大张量 masked 内核不受影响。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/58495
- **关联**：var:INF-0244

### INF-0269 `ttmetal_div_fp32_signed_zero_divisor_special_value_arms_wrong` ttnn.divide fp32 有符号零除数臂对特殊值返回错误结果

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `var` · 发现 2026-09-29 · 旧号 1.270

- **原因对象**：算子/内核 / tt-metal / SFPU 二元 div 零除数臂
- **原因行为**：零除数臂中 in1==0 与 in0==0 两个比较都编译为 SFPSETCC int32 判零，只匹配 +0 位型：NaN/−0 被除数不被识别，除以 −0 时整个臂不进入，改走 in0*(1/−0)。
- **影响对象**：数值结果（静默错误）
- **影响详情**：除以 +0 时 NaN 被除数返回 ±inf、−0 被除数返回 −inf（均应 NaN）；除以 −0 时 subnormal 被除数返回 NaN（应 ∓inf）。全位型扫描 33,554,429 对错误，广播与 Python 标量除数同错，NaN 传播在除零点断裂。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/58272
- **关联**：var:INF-0153

### INF-0270 `ttmetal_slice_rank1_ndsharded_gt2_shards_wrong_data_or_hang` ttnn.slice rank-1 ND 分片超 2 shard 错值或挂死

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-09-29 · 旧号 1.271

- **原因对象**：算子/内核 / tt-metal / SliceRmProgramFactory 页游标
- **原因行为**：SliceRmProgramFactory 对 rank-1、row-major、ND-sharded（CONTIGUOUS_1D）输入按错误的页/shard 游标读取数据。
- **影响对象**：数值结果；可用性/进度（静默错误）
- **影响详情**：shard 数>2 时切片从 shard 边界起错到末尾（4 shards 自 index 64、8 shards 自 index 32），部分配置设备挂死；输出为 L1 interleaved 亦错，等价 rank-2 形状与≤2 shards 正确。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/58370

### INF-0271 `vllm_vendored_fla_chunk_o_gate_loaded_unmasked_before_exp_nan_propagation` vLLM 内置 FLA chunk_o 先 exp 后 mask 致 NaN

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-10-09 · 旧号 1.272

- **原因对象**：算子/内核 / vllm / vendored chunk_o.py 内核
- **原因行为**：vendored chunk_o.py 把 token mask 计算移到 USE_G 块之后，gate 以 boundary_check 无 other 加载并先于 mask 被 exp，未定义 lane 值进入输出累加器。
- **影响对象**：数值结果；隐状态（数值异常）
- **影响详情**：GDN/线性注意力模型在 chunked prefill 产生部分边界 chunk 时，该路径输出可被 NaN/inf 静默污染（而非有限错值）；上游 fla 同块先算 mask 并以 other=0.0 加载。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/59404

### INF-0272 `trtllm_group_rmsnorm_smem_init_race_and_onewarp_path_never_normalizes_input1` TRT-LLM group RMSNorm 初始化竞态与漏归一化第二输入

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-10-09 · 旧号 1.273

- **原因对象**：算子/内核 / tensorrt-llm / groupRmsNormKernels.cu
- **原因行为**：两缺陷并存：①多 warp 写共享统计的共享内存初始化竞态，warp 归约和丢失；②one-warp large-batch 路径循环漏掉第二个输入的 rsqrt 归一化。
- **影响对象**：数值结果；隐状态（静默错误）
- **影响详情**：A100 对 fp64 参照：group_rms_norm_base 几乎所有调用输出偏大（例 0.0262 vs 0.0187），heuristic 回退同错；group_rms_norm_large_batch 第二输入全部错（例 2448.0 vs 1.19）。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/NVIDIA/TensorRT-LLM/issues/19955

### INF-0273 `trtllm_convert_type_returns_tensor_viewing_local_array_dangling` TRT-LLM convert_type 返回局部 frag 视图致悬垂读取

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-10-09 · 旧号 1.274

- **原因对象**：算子/内核 / tensorrt-llm / convert_type 设备 helper
- **原因行为**：util_convert_type 与 flash::convert_type 把转换结果存入函数局部 frag 后以 make_rmem_ptr(&frag) 返回其视图，返回即悬垂，读取为未定义行为。
- **影响对象**：数值结果（静默错误）
- **影响详情**：fused MoE epilogue 与 flash_fwd_mla splitkv combine 中：nvcc 构建读陈旧寄存器值、碰巧可用；clang -O3 构建下 SwiGLU 与其喂入的 MMA 被 DCE，内核写出未写入的寄存器垃圾。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/NVIDIA/TensorRT-LLM/issues/19943；https://github.com/NVIDIA/TensorRT-LLM/issues/20035

### INF-0278 `ttmetal_logaddexp_bf16_sign_flip_222k_ulp_near_zero` ttnn.logaddexp bf16 近零符号翻转、偏差 222k ULP

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-10-09 · 旧号 1.279

- **原因对象**：算子/内核 / tt-metal / ttnn.logaddexp* bf16 路径
- **原因行为**：logaddexp/logaddexp2 的 bf16 实现在 log-sum-exp 近零相消路径上存在实现错误。
- **影响对象**：数值结果（静默错误）
- **影响详情**：bf16 输入且真值接近零时输出符号翻转，偏差最高 222k ULP，无 crash；消费 logaddexp 的损失/概率合并链静默错值。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/58669

### INF-0279 `ttmetal_digamma_lut_extrapolation_outside_fitted_domain_negative_axis_and_tiny_positive` ttnn.digamma LUT 在拟合域外裸外推致错值与符号翻转

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `var` · 发现 2026-10-09 · 旧号 1.280

- **原因对象**：算子/内核 / tt-metal / ttnn.digamma SFPU 有理 LUT
- **原因行为**：SFPU piecewise-rational LUT 仅拟合 [0.01,102]，无负轴分支、无反射公式、无域守卫（仅有 x>102 渐近分支），负输入与 0<x<1.2125e-4 由拟合式裸外推。
- **影响对象**：数值结果（静默错误）
- **影响详情**：bf16 下 −1.5 返 +4.15（应 +0.70）、−2.5 返 −0.26（符号反）；1e-7 返 +2.01e7（符号翻、3.0e7 ULP），digamma(0) 返 +6.68e6（应 −inf）；golden 测试 2-ULP 门槛在 x<1.2125e-4 失效。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/58682；https://github.com/tenstorrent/tt-metal/issues/59462
- **关联**：var:INF-0125

### INF-0280 `ttmetal_fp32_to_bf16_helpers_lose_nan_classification_exhaustive` fp32_to_bf16 转换把 NaN 变成 ±0/+Inf

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `var` · 发现 2026-10-09 · 旧号 1.281

- **原因对象**：算子/内核 / tt-metal / fp32_to_bf16 系列转换 helper
- **原因行为**：设备侧 bfloat16.h 中 RNE 路径的舍入加法可进位越过指数域；截断路径丢弃 NaN 唯一的非零尾数位，丢失 NaN/Inf 区分。
- **影响对象**：数值结果（静默错误）
- **影响详情**：设备侧 fp32→bf16 转换时 NaN 静默变为 ±0（quiet NaN）或 +Inf（signaling NaN），无报错；穷举全部 fp32 NaN 位型，RNE 路径有 131,072 个编码变非 NaN，NaN 传播诊断与 NaN-mask 语义被破坏。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/59126
- **关联**：var:INF-0265

### INF-0281 `ttmetal_quasar_generic_typecast_int32_sign_magnitude_vs_twos_complement` Quasar typecast 误按符号幅值处理 Int32 负值

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `var` · 发现 2026-10-09 · 旧号 1.282

- **原因对象**：算子/内核 / tt-metal / Quasar 通用 typecast 内核；观测/评测 / tt-metal / test_sfpu_compute 打包
- **原因行为**：通用 typecast 路径 _calculate_typecast_arith_sfp_rows_ 按 sign-magnitude 解释 Int32，而 L1/Dest 原生为补码；测试按同一错误假设打包 Int32。
- **影响对象**：数值结果；测试判定（静默错误）
- **影响详情**：含负值的 Int32→Float32 与 Float32→Int32 转换静默输出 −2³¹ 量级垃圾，无报错；因测试与内核同错未被捕获，emu-quasar 单测中补码输入 FAIL；Int32→Int16/UInt16/UInt8 未检查。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/59016
- **关联**：var:INF-0266

### INF-0282 `ttmetal_ttnn_sum_mean_fp32_rowmajor_w_gt_256_wrong` ttnn.sum/mean 对 fp32 ROW_MAJOR 宽行结果错误

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-10-09 · 旧号 1.283

- **原因对象**：算子/内核 / tt-metal / ttnn.sum/mean 宽行归约路径
- **原因行为**：fp32 ROW_MAJOR 宽行归约路径实现错误，原文未给出具体根因。
- **影响对象**：数值结果（静默错误）
- **影响详情**：输入为 fp32、ROW_MAJOR 布局且 W>256 时，ttnn.sum 与 ttnn.mean 静默返回错误结果，无报错。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/58695

### INF-0283 `ttmetal_ttnn_embedding_tile_layout_indices_batch_gt1_only_batch0_correct` ttnn.embedding TILE indices 仅 batch 0 正确

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-10-09 · 旧号 1.284

- **原因对象**：算子/内核 / tt-metal / ttnn.embedding TILE 索引工厂
- **原因行为**：EmbeddingsTilizedIndicesProgramFactory 假设 B*S 已折叠，对 TILE 布局 indices 跳过 reshape 与 shape 校验（校验只对 ROW_MAJOR 跑），按扁平索引寻址。
- **影响对象**：数值结果；隐状态（静默错误）
- **影响详情**：indices 为 TILE 布局、rank 3/4 且首维>1（如 [B,1,1,S]）时，batch 1..B−1 每个 token 取到权重第 0 行或 tile padding 残留值；输出形状正常、无报错，ROW_MAJOR indices 正确。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/59133

### INF-0284 `ttmetal_unary_special_functions_map_nan_to_finite_values_family` ttnn erf/erfc/i1/gelu 把 NaN 输入静默映射为有限值

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `var` · 发现 2026-10-09 · 旧号 1.285

- **原因对象**：算子/内核 / tt-metal / erf/erfc/i1/gelu 的 clamp
- **原因行为**：clamp 依赖 SFPSWAP min/max，而 SFPSWAP 按符号幅值位排序 NaN 而非视为 unordered，+NaN 被当作大于 +Inf 夹到边界后走正常指令流；两处 clamp 均受影响。
- **影响对象**：数值结果（静默错误）
- **影响详情**：NaN 输入时 erf 返回 ±1.0，erfc、i1 返回有限值，gelu 对负 NaN 返回 0，WH 与 BH 一致；上游 NaN 被洗白成普通数，NaN 传播诊断与 NaN-mask 语义失效。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/59184
- **关联**：var:INF-0196

### INF-0285 `ttmetal_sharded_softmax_tile_padding_in_row_sum_non32_last_dim` sharded softmax 把 tile padding 计入行和

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `var` · 发现 2026-10-09 · 旧号 1.286

- **原因对象**：算子/内核 / tt-metal / sharded softmax 族 reader
- **原因行为**：sharded reader 未对末维的 tile-padding 列施加隐式 −inf mask，padding 列参与行和归一化。
- **影响对象**：数值结果；隐状态（静默错误）
- **影响详情**：末维非 32 倍数时 [1,1,64,100] height-sharded 行和仅 0.73–0.79（应为 1），最低到 0，24/24 配置全部泄漏，causal 场景概率几乎全丢；同数据 interleaved 路径正确，无报错。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/59818
- **关联**：var:INF-0268

### INF-0287 `ttmetal_wormhole_matmul_unpack_mop_covers_first_8_streamed_tiles_9to16_wrong` Wormhole matmul unpack MOP 隔次调用错值

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-10-09 · 旧号 1.288

- **原因对象**：算子/内核 / tt-metal / LLK matmul unpack MOP 配置
- **原因行为**：unpack config context 1 的 MOP 只覆盖前 8 个 streamed tiles，超过 8 tile 的块依赖前次调用残留的状态。
- **影响对象**：数值结果（静默错误）
- **影响详情**：Wormhole 上单方向 9–16 streamed tiles 的 full sync block matmul 每隔一次调用结果错误（重复调用时第二次起错），无报错。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/59125

### INF-0288 `ttmetal_dst_full_sync_en_validation_admits_subblocks_kernels_compute_wrong` matmul 校验放行会算错的 subblock 尺寸

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `var` · 发现 2026-10-09 · 旧号 1.289

- **原因对象**：算子/内核 / tt-metal / matmul 工厂 subblock 校验
- **原因行为**：dst_full_sync_en 使能时校验谓词未覆盖合法 subblock 约束，放行内核实际会算错的 subblock 尺寸。
- **影响对象**：数值结果（静默错误）
- **影响详情**：dst_full_sync_en 下特定 subblock 配置的 matmul 静默错值且校验不拦截；2D/1D/Reuse 三工厂与 all_gather_minimal_matmul_async（含 quasar mcast_1d）均受影响。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/59689
- **关联**：var:ttmetal_matmul_multicore_reuse_oversubscription_silent_tile_drop

### INF-0290 `ttmetal_group_norm_fp32_dest_subblock_over_4_tiles_halfsync_wrong_stats` group_norm fp32 DEST subblock 超容量错值

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-10-12 · 旧号 1.291

- **原因对象**：算子/内核 / tt-metal / group_norm interleaved 工厂
- **原因行为**：mcast/no-mcast 工厂用 get_max_subblock(block_wt, 8) 取通道 subblock，未按 fp32 DEST half-sync 的 4 tile 封顶，kernel 单次请求 5–8 个 tile。
- **影响对象**：数值结果（静默错误）
- **影响详情**：fp32 DEST + interleaved 输入且每组跨 ≥5 channel tile（C/G≥160）时归一化统计确定性错值，误差随组宽增长（C=1024 max err 1.00 vs 0.011）；dst_full_sync_en 或 use_welford 时正确。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/59848

### INF-0291 `ttmetal_wormhole_reduce_f16_in_f32_out_max_and_scalar_paths_wrong` Wormhole reduce F16 入 F32 出时 MAX 错误

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-10-12 · 旧号 1.292

- **原因对象**：算子/内核 / tt-metal / LLK reduce F16/fp32 格式对
- **原因行为**：LLK reduce 未正确处理 Float16 SrcA 与 fp32 Dest 的格式配对，非近期回归。
- **影响对象**：数值结果（静默错误）
- **影响详情**：Wormhole B0 上 Float16 输入 + Float32 输出时 Row/Column/Scalar MAX 返回错误量级（1.10→6e4–3e14），Scalar Sum/Average 也错，Row/Column Sum/Average 正确；CI 未参数化 Float16 输入。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/60008

### INF-0292 `ttmetal_blackhole_where_bf16_predicate_fp32_values_bit_reinterpret_garbage` ttnn.where 混合 bf16/fp32 时几乎全 lane 错值

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-10-12 · 旧号 1.293

- **原因对象**：算子/内核 / tt-metal / ttnn.where predicate 读取路径
- **原因行为**：混合 dtype where 的 predicate 读取路径把 bf16 predicate 位型按 fp32 语义消费，导致选择错误分支。
- **影响对象**：数值结果（静默错误）
- **影响详情**：Blackhole 上 bf16 predicate + fp32 数值时 65,503–65,536/65,536 个 lane 输出垃圾（如 −1.1258→2.2e-22），四种 predicate 形状均复现；全 bf16 或全 fp32 正确，无报错。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/60223

### INF-0293 `ttmetal_xielu_fp32_negative_branch_zero_crossing_cancellation` ttnn.xielu FLOAT32 负分支在零交叉点附近灾难性相消

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-10-12 · 旧号 1.294

- **原因对象**：算子/内核 / tt-metal / ttnn.xielu 负分支公式
- **原因行为**：负分支 alpha_n*(exp(x)-1-x)+0.5*x 在自身零交叉点 x* 附近两操作数相消 1,000–21,000 倍，未切换到防相消形式，exp 近似残差主导结果。
- **影响对象**：数值结果（静默错误）
- **影响详情**：FLOAT32 输出时负半轴零交叉邻域相对误差最坏 53.6%，alpha_n=1.0 下 bf16 量化点最坏 2,816 ULP；bf16 输出掩盖该缺陷，binding 对 FLOAT32 无精度警示，无报错。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/58670

### INF-0317 `fp_reduction_order_nondeterminism` FP 归约顺序随 batch/kernel/拓扑变化致 temp=0 输出不确定

`故障` · `active` · `ambiguous` · `config` · `verified` · `NEW` · 发现 待定 · 旧号 4.1

- **原因对象**：算子/内核 / 通用 / split-K matmul 等归约 kernel
- **原因行为**：FP 加法不满足结合律，归约顺序随 batch shape（split-K/stream-K 切分）、kernel 选择（auto-tuner）、TP all-reduce 拓扑和 chunked-prefill 重排而变。
- **影响对象**：输出内容；数值结果（不可复现）
- **影响详情**：同一 prompt 在 temp=0 下跨 run、跨负载、跨 batch 得到不同 greedy token，benchmark 分数抖动，偶有完全发散的 completion；vLLM 开 VLLM_BATCH_INVARIANT=1 在约 44 并发下仍有 logprob 差异。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/；https://arxiv.org/abs/2506.09501

### INF-0318 `tf32_matmul_truncation` Ampere+ 上 GEMM 默认走 TF32 静默截断尾数

`故障` · `active` · `model_on` · `config` · `verified` · `NEW` · 发现 待定 · 旧号 4.2

- **原因对象**：算子/内核 / pytorch / cuBLAS/cuDNN GEMM 默认 TF32
- **原因行为**：Ampere+ GPU 上 cuBLAS/cuDNN 的 GEMM 默认走 TF32 tensor core，只读取 mantissa 高 10 bit，无代码改动即静默截断出约 1e-3 相对误差
- **影响对象**：数值结果；输出分布（静默错误）
- **影响详情**：每层引入小而系统性的舍入并跨层累积，偶尔顶翻低置信度 argmax，输出整体连贯；同一 matmul 在 V100 与 A100 上结果不同，单帧 geometric 检测基本无效，需对照 FP32 参考
- **改进措施**：检测 待定；修复 待定
- **来源**：https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html

### INF-0319 `flashattention_bf16_residual_drift` FlashAttention 重排浮点运算致跨 backend 数值漂移

`结论` · `active` · `model_on` · `config` · `verified` · `NEW` · 发现 待定 · 旧号 4.3

- **原因对象**：算子/内核 / flash-attention / FA tiled/online softmax
- **原因行为**：FlashAttention 相对 eager 重排浮点运算顺序（tiled/online softmax，P@V 用 BF16、running max/sum 用 fp32），FA2 前向跨 run 不确定，fp16 进一步放大偏差
- **影响对象**：隐状态；输出分布（不可复现）
- **影响详情**：eager/sdpa/flash 三个 backend 非 bitwise 一致（带 mask 时尤甚），BF16 单次前向偏差比 baseline 大约一个数量级；表现为极小平滑的 logit/隐状态差异，很少翻转 greedy token 但会移动 logprob
- **改进措施**：检测 待定；修复 待定
- **来源**：https://arxiv.org/abs/2405.02803

### INF-0320 `fp16_overflow_ceiling` fp16 激活越过 65504 上限溢出为 inf/NaN

`故障` · `active` · `infra_off` · `yes` · `documented` · `var` · 发现 待定 · 旧号 4.4

- **原因对象**：算子/内核 / 通用 / logits 与 RMSNorm 归约
- **原因行为**：fp16 最大可表示值约 65504，长上下文或大激活下 QK attention logits、vocab logits 或 RMSNorm 方差归约越过上限溢出为 inf，并在后续运算中变成 NaN
- **影响对象**：数值结果；输出内容（数值异常）
- **影响详情**：NaN 传播导致垃圾输出或空生成，多见于长上下文或大激活；默认 fp32 setup 下基本不存在，需显式把中间激活降到 fp16 才能复现
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/37732；https://github.com/vllm-project/vllm/issues/16579；https://github.com/vllm-project/vllm/issues/22881；https://github.com/pytorch/pytorch/issues/66707
- **关联**：var:INF-0212

### INF-0321 `cudnn_algo_nondeterminism` cudnn.benchmark 自动调优选不同算法致跨 run 结果不一

`故障` · `active` · `ambiguous` · `yes` · `documented` · `NEW` · 发现 待定 · 旧号 4.5

- **原因对象**：算子/内核 / pytorch / cuDNN benchmark 算法自动选择
- **原因行为**：cudnn.benchmark=True 时 auto-tuner 依 input shape/batch 选择不同的 convolution/GEMM 算法，同一机器后续运行仍可能选中不同算法
- **影响对象**：数值结果；输出内容（不可复现）
- **影响详情**：同一 model+input 跨 run/跨 batch 输出低位数字变化，tie 处偶尔翻转 token；单次 pass 看似正常，只在 run-to-run 方差中显形，单次采样评测会误标为 clean
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/pytorch/pytorch/blob/master/docs/source/notes/randomness.rst

### INF-0322 `vllm_persistent_topk_silent_candidate_drop_narrow_range` vLLM persistent_topk 窄值域下静默丢弃真实 top-k 候选

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-08-11 · 旧号 4.6

- **原因对象**：算子/内核 / vllm / persistent_topk 直方图 top-k
- **原因行为**：radix/直方图 top-k 原语在多元素落入同一粗直方图 bin 时选值出错，丢弃真实 top-k 候选并替换为严格更小的值（不是 tie-break 差异）
- **影响对象**：数值结果（静默错误）
- **影响详情**：无 crash、无 NaN、无 -1 slot 或重复索引，只能与参考对比发现；logits 值域窄（trained indexer head 常态）时触发，k=512/1024/2048 均受影响，k=2048 长行最易触发（B300 sm_103）
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/51782

### INF-0325 `ttmetal_addcmul_composite_llk_associativity_mismatch` ttnn.addcmul 两条实现乘法结合序不同致数值不一致

`故障` · `active` · `ambiguous` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-17 · 旧号 4.9

- **原因对象**：算子/内核 / tt-metal / ttnn.addcmul 双实现路径
- **原因行为**：LLK SFPU 内核先乘标量（与 torch.addcmul 一致），composite fallback（_addcmul）先算张量积 b*c 再乘标量；浮点乘法不可结合，由 dispatch 条件二选一
- **影响对象**：数值结果；测试判定（静默错误）
- **影响详情**：无 crash、无 NaN；同一语义 op 在不同配置下走不同路径、输出系统性不同，仅其一匹配注册 golden，测试只覆盖其一时另一路径的偏差静默通过
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/53282

### INF-0326 `sglang_dsa_topk_v2_tie_overflow_arbitrary_subset` SGLang DSA top-k v2 tie bin 溢出静默丢弃真实候选

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `var` · 发现 2026-08-18 · 旧号 4.10

- **原因对象**：算子/内核 / sglang / DSA top-k v2 内核 tie 收集
- **原因行为**：粗 bin 元素超过 kMaxNumTie=2048 时 collect pass 按 atomicAdd 到达序保留前 2048 个、静默丢弃其余，exact tie-break 只在该任意子集上运行；count_eq 已知溢出却未利用
- **影响对象**：数值结果；输出内容（静默错误）
- **影响详情**：无 crash、无错误标志；sparse attention 读错 KV 位置，长上下文检索质量静默退化（GLM-5.2-FP8、~1M 上下文实测 42 行中 2–3 行不符），需与 torch.topk fp32 multiset 对拍才能发现
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/35257
- **关联**：var:INF-0322

### INF-0327 `vllm_moe_index_add_atomic_nondeterministic_reduction` vLLM MoE index_add_ 原子散射归约跨 run 非确定

`故障` · `active` · `ambiguous` · `yes` · `verified` · `var` · 发现 2026-08-23 · 旧号 4.11

- **原因对象**：算子/内核 / vllm / MoE 归约器 index_add_ 散射
- **原因行为**：TopKWeightAndReduceNaiveBatched.apply 用 output.index_add_ 把多专家贡献原子散射回源 token；top_k>1 时同一行被并发原子加，浮点累加顺序跨 run 不定
- **影响对象**：数值结果；输出内容（不可复现）
- **影响详情**：无 crash、无 NaN；temperature=0/固定 seed 下同一输入跨 run 输出非确定，BF16 低位差异在近 tie logits 处翻转 greedy token；单次采样评测会误标为 clean，需 run-to-run 对拍
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/53430
- **关联**：var:INF-0317

### INF-0328 `sglang_vlm_greedy_cross_build_near_tie_flip_worker_nondeterminism` SGLang VLM temp=0 跨 build 与跨会话输出不可复现

`故障` · `active` · `ambiguous` · `other_stack` · `documented` · `var` · 发现 2026-08-26 · 旧号 4.12

- **原因对象**：算子/内核 / sglang / 待定
- **原因行为**：新 build 中存在 worker 级 run-to-run 数值非确定（具体算子未定位）；在 top-1/top-2 logprob 完全并列的分叉点，argmax 由低位数值差决定
- **影响对象**：输出内容；输出分布（不可复现）
- **影响详情**：同一 byte-identical 多模态请求 temp=0：v0.5.12 与 v0.5.17.dev 输出大幅分歧（相似度 0.28–0.58）；新 build 跨会话 342/342/342 vs 282/134/134 不一致，router 后 4 个相同 worker 亦不一致
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/36480
- **关联**：var:INF-0317

### INF-0333 `vllm_batch_invariant_matmul_self_test_violated_and_tf32_leak` vLLM 批不变 matmul 自测容许跨批差异且放行 TF32

`故障` · `active` · `infra_off` · `config` · `documented` · `var` · 发现 2026-09-04 · 旧号 4.17

- **原因对象**：算子/内核 / vllm / 批不变 matmul 内核及 TF32 开关
- **原因行为**：VLLM_BATCH_INVARIANT 路径的 batch-invariant matmul 实现及其自测基线本身不满足逐 batch 位一致，且该路径未强制 allow_tf32=False，fp32 matmul 仍走 TF32。
- **影响对象**：数值结果；测试判定（不可复现）
- **影响详情**：开 VLLM_BATCH_INVARIANT=1 跑 fp32 matmul 时，temp=0/逐位对照在跨 batch 维度不可复现，精度被 TF32 静默压低；官方 CI 在自家环境即复现差异，依赖该开关的量化验证、golden diff 结论失真。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/55131
- **关联**：var:INF-0318

### INF-0340 `vllm_gfx1151_rocm_attn_decode_output_depends_on_previous_requests` vLLM gfx1151 ROCM_ATTN decode 输出依赖先前请求

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-09-18 · 旧号 4.24

- **原因对象**：算子/内核 / vllm / ROCM_ATTN decode 内核
- **原因行为**：ROCM_ATTN decode 内核疑在请求间保留设备态（如工作区/tiling 状态未清零或按历史长度参数化），具体载体待上游定位。
- **影响对象**：跨请求/跨样本数据；输出内容（不可复现）
- **影响详情**：gfx1151 上 max_num_seqs=1、eager、关前缀缓存的完全串行配置下，20 次变长 filler 后同一 greedy probe 的 decode 第 5 步 logprob 出现 4 个值、completion 2 种，prefill 首 token 20/20 恒定。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/57493

### INF-0344 `vllm_qwen36_hybrid_gdn_multisequence_logprobs_nondeterministic_run_to_run` vLLM Qwen3.6 混合 GDN 多序列 logprobs 跨运行非确定

`故障` · `active` · `infra_off` · `config` · `documented` · `var` · 发现 2026-10-09 · 旧号 4.28

- **原因对象**：算子/内核 / vllm / 多序列路径内核（未定位）
- **原因行为**：多序列同 step 路径中存在跨运行非确定的内核，具体未定位（候选：默认 config 的 Triton fused MoE、GDN chunked prefill varlen、FA2 varlen）；单序列路径确定。
- **影响对象**：输出分布；输出内容（不可复现）
- **影响详情**：vLLM 0.30.0 Qwen3.6-35B-A3B 同一两条批重复提交 logprobs 差至 0.168，单跑与批内差至 0.198；272 个真实 prompt 批内评分 10–17% 所选选项改变；max_num_seqs=1 时差为 0。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/59764
- **关联**：var:INF-0328

### INF-0345 `vllm_triton_mla_batch_invariant_violated_under_chunked_prefill` vLLM TRITON_MLA 分块 prefill 违反批不变承诺

`故障` · `active` · `infra_off` · `config` · `documented` · `var` · 发现 2026-10-09 · 旧号 4.29

- **原因对象**：算子/内核 / vllm / TRITON_MLA 分块上下文 LSE 合并
- **原因行为**：cached context 分块分别 attend 后经 LSE 重缩放合并（疑为 merge_attn_states），非 bitwise 结合；分块边界由调度决定。
- **影响对象**：输出分布；输出内容（不可复现）
- **影响详情**：VLLM_BATCH_INVARIANT=1 下 DeepSeek-V2-Lite/V3.1 仅改 token 预算，131 chunk 时 logprobs 首步差约 1e-3、decode 第 9 步起 token 翻转；enforce_eager 仍失败，CI 未覆盖该测试。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/60007
- **关联**：var:INF-0333

### INF-0365 `sglang_triton_devptr_table_int64_xpu_usm_overflow` SGLang Triton 指针表用 int64，XPU 高位地址溢出

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-17 · 旧号 5.18

- **原因对象**：算子/内核 / sglang / Triton kernel 指针表构造
- **原因行为**：多个 Triton kernel 把 data_ptr() 打包进 int64 指针表，隐式假设地址 < 2^63；XPU USM 设备地址最高位置位，无法表示为有符号 int64。
- **影响对象**：可用性/进度（报错）
- **影响详情**：SGLang 在 Intel XPU 上调用构建指针表的 Triton kernel 时，kernel 尚未 launch 即在 host 侧抛 ValueError: Overflow when unpacking long long；NVIDIA CUDA 地址当前 < 2^63，不触发。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/35047

### INF-0374 `sglang_kda_split_grid_stale_program_id_overwrite_state_corruption` SGLang KDA kernel 陈旧赋值覆写轴映射致状态损坏

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `var` · 发现 2026-08-31 · 旧号 5.27

- **原因对象**：算子/内核 / sglang / KDA kernel program-id 映射
- **原因行为**：按 SPLIT_N_HV_GRID 建立 (i_k, i_n, i_hv) 映射后，残留的旧扁平 grid 赋值立即覆写这三个变量；NV>1 时 axis 0 被当作 i_k，读写越界 K tile。
- **影响对象**：KV状态；数值结果（数值异常）
- **影响详情**：GLM-5.3-Flash 在 AMD MI355X（gfx950）上服务时 recurrent state 被损坏，输出 BF16 NaN，KPool page-table 出现非法值，可能进一步越界访问或 crash。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/pull/37250
- **关联**：var:INF-0552

### INF-0375 `vllm_flashinfer_trtllm_moe_dp_ep_worker_stuck_wedge` vLLM TRTLLM MoE 后端 DP/EP 下启动静默卡死

`故障` · `active` · `infra_off` · `partial` · `documented` · `var` · 发现 2026-09-01 · 旧号 5.28

- **原因对象**：算子/内核 / vllm / FlashInfer TRTLLM MoE 后端
- **原因行为**：DP/EP 拓扑下 FlashInfer TRTLLM MoE 路径在 dummy run 期停滞，栈停在 _apply_quant_method，下方疑为 collective 或 kernel 启动等待。
- **影响对象**：可用性/进度（挂起）
- **影响详情**：GB200 TP8 + DP/EP 启动时 worker 无进展，无 NaN、无错误日志，引擎永远到不了 ready；Kimi-K2-base 与 DeepSeek-v3-base 均复现，v0.27.1-cu129 镜像同样命中，换 triton/cutlass moe_backend 可规避。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/54723
- **关联**：var:INF-0372

### INF-0378 `vllm_kimik3_lowm_tp8_kda_projection_cutlass_dsl_compile_fail` vLLM Kimi-K3 KDA 投影 kernel sm_90a 编译失败

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-09-04 · 旧号 5.31

- **原因对象**：算子/内核 / vllm / kda_skinny_gemm 内核
- **原因行为**：#54697 新增的 low-M TP8 KDA projection-overlap 路径中，kda_skinny_gemm 的 CUTLASS DSL 生成代码在 sm_90a 上无法通过 NVVM 编译。
- **影响对象**：可用性/进度（报错）
- **影响详情**：Kimi-K3（MXFP4）在 32×H100（TP8/PP4）启用 CUDA graphs 时，profiling 期 kda_skinny_gemm.run_n() 抛 NVVM_ERROR_COMPILATION，启动即崩溃；两个独立 4 节点部署复现，回退到 #54697 之前即正常。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/55350

### INF-0387 `vllm_flashinfer_autotune_config_cache_rank0_only_dump_multi_rank_launch_deadlock` FlashInfer autotune 缓存仅 rank0 落盘致启动死锁

`故障` · `active` · `infra_off` · `other_stack` · `verified` · `var` · 发现 2026-09-17 · 旧号 5.40

- **原因对象**：算子/内核 / flashinfer / JIT autotuner 配置缓存
- **原因行为**：autotune 配置缓存只为 rank 0 dump：rank0 命中缓存跳过调优、退出调优同步的 all-reduce，其余 rank 未命中进入调优并等待该 all-reduce，成员资格约定不一致。
- **影响对象**：可用性/进度（挂起）
- **影响详情**：多 rank 引擎（如 8×H100 TP8 或 TP7+EP7 MoE、NVFP4）存有 rank0 缓存时启动无限挂起：worker 全存活但零推进，无超时无报错；仅 TP0 打印 Config cache hit，其余停在 Tuning ...gemm1: 0%。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/57423
- **关联**：var:vllm_flashinfer_trtllm_moe_dp_ep_launch_wedge

### INF-0391 `vllm_flashinfer_autotune_sm103_prolonged_no_progress_boot` GB300 上 FlashInfer fp4 MoE 调优长时间无进展

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `var` · 发现 2026-09-22 · 旧号 5.44

- **原因对象**：算子/内核 / flashinfer / fp4 MoE kernel autotune
- **原因行为**：待定：根因未定。报告者已撤回「SM103 缺对应 cubin、候选永不终止」的归因；倾向为极慢调优而非真正卡死，尚待进程栈或完成时间验证；cache 只在调优完成时写出。
- **影响对象**：可用性/进度；性能/资源（挂起）
- **影响详情**：vLLM 在 GB300 上 autotune trtllm_fp4_block_scale_moe 时启动停留 40+ 分钟：GPU 利用率 100%，按行采集日志看不到 tqdm 进度，配置文件未写入，无 error；跳过该 op 调优约 8 分钟就绪。另一台 GB300 换模型可复现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/58031；https://github.com/vllm-project/vllm/issues/58031#issuecomment-5788562400
- **关联**：var:INF-0387

### INF-0392 `sglang_trtllm_mha_prefill_decode_both_sm90_silent_wrong_completions_gate_regression` SGLang 0.5.20 在 SM90 放开 trtllm_mha 致错误补全

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-09-23 · 旧号 5.45

- **原因对象**：算子/内核 / sglang / trtllm_mha 架构守卫与组合路径
- **原因行为**：v0.5.17 中 prefill 用 trtllm_mha 仅限 SM100 的守卫在 v0.5.20 消失或被绕过，SM90 接受双侧 trtllm_mha；该组合路径在 SM90 上数值不成立，内核层根因待定。
- **影响对象**：输出内容（静默错误）
- **影响详情**：H200（SM90）上 --attention-backend trtllm_mha 正常启动、/server_info 报告该后端、吞吐正常，但返回系统性错误补全（gpt-oss-120b 实测）；单侧 trtllm_mha 另一侧 fa3 均正确；升级 0.5.17→0.5.20 即触发。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/40921

### INF-0401 `mamba_ssm_stride_uint32_overflow` Mamba-1 selective-scan stride 溢出破坏状态缓存

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 待定 · 旧号 6.8

- **原因对象**：算子/内核 / 待定 / selective-scan stride 字段
- **原因行为**：selective-scan CUDA kernel 用 uint32 存放 stride 字段，SSM state 的 batch-stride 乘法越过 UINT32_MAX 时静默回绕，写到错误的 state-cache slot。
- **影响对象**：KV状态；跨请求/跨样本数据（静默错误）
- **影响详情**：batch 规模使 stride 超过 UINT32_MAX 时，约 31% 的 state-cache slot 被破坏，出现周期性的跨 batch 状态写入；仅影响 SSM（Mamba-1）模型。
- **改进措施**：检测 待定；修复 待定
- **来源**：引用：AI21 "Stride and Prejudice" blog。

### INF-0407 `sglang_grok_xai_temperature_scaling_backend_mismatch` SGLang 确定性 extend kernel 的 Grok 温度缩放不一致

`故障` · `active` · `model_on` · `other_stack` · `documented` · `NEW` · 发现 2026-07-30 · 旧号 6.14

- **原因对象**：算子/内核 / sglang / 确定性 extend attention
- **原因行为**：xai_temperature_len 缩放在阈值后，常规 extend 与 decode kernel 用对数缩放，统一 deterministic extend kernel 却用倒数缩放 len/(position+1)，公式不一致。
- **影响对象**：数值结果；输出内容（静默错误）
- **影响详情**：启用统一 deterministic extend attention 时，Grok 系列模型的 attention logits 与常规 Triton 路径不同，输出静默差异但仍连贯，无 crash、无 NaN；需跨后端重算对比才能发现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/32942

### INF-0410 `vllm_dsv4_sm12x_nan_mqa_topk_illegal_access` vLLM DSv4 在 SM120 上 NaN 致 top-k 内核非法访存

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-07-31 · 旧号 6.17

- **原因对象**：算子/内核 / vllm / top_k_per_row_prefill 内核
- **原因行为**：top_k_per_row_prefill 内核被 NaN 的 MQA logits 喂入时，把未初始化的 shared memory 当作 index 输出，这些垃圾 index 随后被用于地址计算。
- **影响对象**：可用性/进度；数值结果（报错）
- **影响详情**：DeepSeek-V4-Flash 在 SM120、TP=4 下，首个含超过约 2048 token prefill 的批次（如 16 个并发结构化输出请求）触发 CUDA illegal memory access，杀死全部 TP rank，报 EngineDeadError。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/49896

### INF-0414 `vllm_rocm_kimi_k3_kda_mixed_batch_nan_autotune_race` Kimi-K3 ROCm KDA 混合批 NaN 与 autotune 数据竞争

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-08-01 · 旧号 6.21

- **原因对象**：算子/内核 / vllm / KDA 混合批分流与 autotune 配置
- **原因行为**：_forward 把混合批的 decode 序列送入 chunk 内核，该内核对 length=1 序列返回 NaN；recompute_w_u_fwd_kernel 的 autotune 选中有数据竞争的 num_stages=4。
- **影响对象**：数值结果；输出内容（数值异常）
- **影响详情**：MI355X 上服务 Kimi-K3（gsm8k，VLLM_ROCM_USE_AITER=1）时，混合批中的 decode 序列得到 NaN；竞争配置（尤其配 num_warps=4）静默产生约 1e38 量级垃圾值，污染下游输出。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/50649

### INF-0445 `vllm_dsv4_flashmla_zero_query_len_chunk_crash` vLLM DSV4 稀疏 prefill 遇零 query 分块崩溃

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-08-08 · 旧号 6.52

- **原因对象**：算子/内核 / vllm / FlashMLA sparse 分块计划
- **原因行为**：get_prefill_chunk_plan 未保证每个 prefill 请求 ≥1 query token，可产出仅含 query_len==0 请求的 chunk，切出空张量传给 flash_mla_sparse_fwd。
- **影响对象**：可用性/进度（报错）
- **影响详情**：DeepSeek-V4 配 KVTransfer/SimpleCPUOffloadConnector、KV 加载期间 prefill 含 query_len==0 请求时触发；TMA 描述符 s_q==0 abort（CUDA error 700）。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/51489

### INF-0454 `vllm_rocm_aiter_fused_moe_nan_logits_gfx942` vLLM AITER fused-MoE 第二步起 NaN logits

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-09 · 旧号 6.61

- **原因对象**：算子/内核 / vllm / AITER fused-MoE 路径
- **原因行为**：AITER fused-MoE 路径在 decode 阶段算出 NaN，prefill 正常；组件隔离已排除 KV dtype 与 attention，具体代码缺陷原文未给出；main 已修复，v0.26.0 release 镜像仍存在。
- **影响对象**：输出分布；输出内容（数值异常）
- **影响详情**：gfx942（8x MI325X、TP8）以 VLLM_ROCM_USE_AITER=1 服务 Qwen3.5-397B-A17B-FP8 时触发；首 token 正常，此后每步对 NaN logits 取 argmax 恒输出 token 0，请求 logprobs 返回 400。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/51580

### INF-0467 `vllm_deepseek_v4_fused_qnorm_rope_kv_insert_large_batch_invisible_alloc_crash` vLLM DSV4 融合 kv-insert 算子大 batch 致崩溃

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-08-11 · 旧号 6.74

- **原因对象**：算子/内核 / vllm / DSV4 融合 kv-insert 算子
- **原因行为**：DeepSeek-V4 的 fused qnorm/rope/kv-insert 算子在运行时临时分配显存（new_empty），该分配绕过 memory profiler，大 batch 下分配失败
- **影响对象**：可用性/进度（报错）
- **影响详情**：--max-num-batched-tokens 设为 24576/32768 时，初始化、readiness 检查与单请求均正常，真实流量组装大 batch 持续几分钟后 EngineCore 崩溃（EngineDeadError），易漏到生产。H100 TP=4、fp8 KV + MTP。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/51743

### INF-0474 `vllm_deepgemm_paged_mqa_metadata_oob_zero_seq_lens` vLLM DeepGEMM paged-MQA 全零 seq_lens 越界读

`故障` · `active` · `infra_off` · `partial` · `verified` · `NEW` · 发现 2026-08-12 · 旧号 6.81

- **原因对象**：算子/内核 / vllm / indexer metadata 输入
- **原因行为**：indexer metadata 把 dummy seq_lens(=1) 除以 compress_ratio(=4) 得全零长度；DeepGEMM SM90 paged-MQA 调度在全零且 batch 为 32 倍数时越界读
- **影响对象**：可用性/进度（报错）
- **影响详情**：仅 compress_ratio=4 的 DeepSeek-V4-Flash 受影响：FULL cudagraph 捕获（capture size 总是 32 的倍数）时出现 CUDA illegal memory access，engine 崩溃；4× GH200 复现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/51972

### INF-0475 `vllm_rocm_gfx942_deepseek_v4_flash_sparse_sw_retrieval_collapse` ROCm gfx942 上 DSV4-Flash 长 prompt 检索静默崩溃

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-13 · 旧号 6.82

- **原因对象**：算子/内核 / rocm / AITER sparse indexer
- **原因行为**：gfx942 版 AITER sparse indexer 在 prompt 超过约 4–5k tokens 时给出错误结果，是 gfx950 上 #40018（>20K 输出乱码）的同类问题，阈值更低
- **影响对象**：输出内容；数值结果（静默错误）
- **影响详情**：8×MI325X、DEEPSEEK_SPARSE_SWA 后端下，needle 检索在 ≤3,611 tokens 时 3/3 正确，≥5,294 tokens 时 0/3：模型称 needle 不存在并跑到 max_tokens；无报错，GSM8K 短 prompt 不受影响。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/52109

### INF-0489 `vllm_triton_alignment_specialization_shared_cache_write_race_all_nan` causal_conv1d 对齐特化致 Triton 缓存竞态全 NaN

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-08-15 · 旧号 6.96

- **原因对象**：算子/内核 / vllm / causal_conv1d 元数据指针特化
- **原因行为**：_causal_conv1d kernel 对逐标量读取、对齐性随 batch 布局翻转的元数据指针做 16 字节对齐特化，使 unaligned 变体必在推理期 JIT；多引擎并发写共享 cache 条目产生竞态。
- **影响对象**：数值结果；性能/资源（数值异常）
- **影响详情**：单节点表现为推理期 JIT 延迟尖峰；多节点共享 TRITON_CACHE_DIR 时损坏的编译二进制被静默加载，服务不 crash 但持续返回全 NaN（GRPO RL，6×GB200 实测）。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/52413

### INF-0503 `sglang_dsv4_indexer_context_logits_alloc_unbudgeted_oom` DSV4 indexer logits 分配不计入预算致长上下文 OOM

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-08-17 · 旧号 6.110

- **原因对象**：算子/内核 / sgl-deep-gemm / indexer logits 张量分配
- **原因行为**：DeepSeek-V4 sparse indexer 在 host 侧为每层每个 prefill chunk 分配 fp32 logits 张量，大小随累积上下文线性增长，且对 mem_fraction_static 预算器不可见。
- **影响对象**：可用性/进度；指标/日志（报错）
- **影响详情**：约 800K 上下文的单请求在 prefill 中途确定性 OOM，indexer 复制而非 TP 分片，故每个 rank 同时 OOM；此前预算指标仍显示有余量。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/35201

### INF-0516 `trtllm_batched_prefill_gt3_progressive_corruption` TRT-LLM 批处理 prefill 在 batch>3 时输出逐步损坏

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-08-19 · 旧号 6.123

- **原因对象**：算子/内核 / tensorrt-llm / 批处理 prefill（batch>3 分支）
- **原因行为**：批处理 prefill/生成路径在 batch>3 时的数值或索引处理有误，原文推测为内部 kernel/调度按 batch size 分岔，根因未定位。
- **影响对象**：输出内容（静默错误）
- **影响详情**：Qwen3.5-2B 20 并发同 prompt 时 19/20 输出幻觉重复，batch≤3 正确；批 prefill 的请求输出彼此相同；损坏随生成长度累积，短输出看似正确，smoke test 会通过；关 CUDA graph 无效，vLLM 同模型正确。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/NVIDIA/TensorRT-LLM/issues/17964

### INF-0547 `sglang_sm121_cookbook_quad_silent_garbage_kv_holes_boot_crash` SGLang SM121 单卡缺 arch guard 致乱码与 NaN 洞

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-27 · 旧号 6.154

- **原因对象**：算子/内核 / sglang / QSA decode 选择、KV 压缩内核
- **原因行为**：decode 选择器可导入即用 SM100-only 的 trtllm-gen kernel，缺 arch 检查；compact KV 抽取按原始列偏移写行，读侧按紧凑布局读；另有 TMA-O 与稀疏 prefill fp8 缺陷。
- **影响对象**：输出内容；数值结果（静默错误）
- **影响详情**：cookbook 镜像在单卡 GB10（SM121）、fp4 + fp8 KV 下：首 token 后永久输出“!!!!”（NaN logits→token 0），/health 正常无报错；FA fallback 读到 NaN 洞致输出退化；TMA-O 启动 crash。TP2 不复现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/36716

### INF-0553 `vllm_flashinfer_sm120_nvfp4_fp8kv_16token_ima` vLLM sm_120 FlashInfer×NVFP4×fp8 KV 非法访存

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-08-29 · 旧号 6.160

- **原因对象**：算子/内核 / vllm / FlashInfer 注意力后端（sm_120）
- **原因行为**：sm_120 上 FlashInfer 注意力后端与 ModelOpt NVFP4、fp8 KV cache 组合时发生非法访存；错误码漂移且首个 decode step 即触发，指向底层内存损坏而非确定性越界读，具体位置未定位。
- **影响对象**：可用性/进度（报错）
- **影响详情**：RTX PRO 6000 Blackwell 上 16-token 单请求首个 decode step 即 CUDA IMA（三次运行分别在 63/74/1 个请求后），traceback 止于异步同步点而非故障内核；换 TRITON_ATTN 后 413 请求 1h45m 零 crash。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/54225

### INF-0579 `vllm_rocm_sparse_mla_indexer_decode_logits_workspace_prefill_bound_locked` vLLM ROCm indexer 缓冲按 prefill 上界锁定

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `var` · 发现 2026-09-03 · 旧号 6.186

- **原因对象**：算子/内核 / vllm / ROCm indexer decode 缓冲行数
- **原因行为**：decode-logits workspace 行数在 profiling 时取 hidden_states.shape[0]（即 prefill 上界），而非 batch_size*next_n，随后被 lock 终身冻结。
- **影响对象**：性能/资源（性能退化）
- **影响详情**：ROCm（gfx942/gfx950）+ sparse-MLA indexer 模型 + 1M 上下文 + MTP 时保留 8192 行而 decode 实需 256 行，每 GPU 白锁 32 GiB、占 KV cache 21%，并发/吞吐静默缩水，无 crash。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/55132
- **关联**：var:INF-0566

### INF-0648 `vllm_dsv41_sparse_mqa_nan_row_stale_col_indices_silent_topk_selection` vLLM DSV4.1 indexer NaN 行后复用陈旧 top-k

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-09-24 · 旧号 6.255

- **原因对象**：算子/内核 / vllm / MQA indexer 与 DeepSelect
- **原因行为**：DeepSelect 遇 NaN 行只写首槽哨兵即返回，其余 topk−1 槽不写；vLLM 跨步复用且不清零 col_indices_buffer，陈旧索引通过越界检查被当作真实位置。
- **影响对象**：数值结果（静默错误）
- **影响详情**：DeepSeek-V4.1 sparse MQA 中出现 NaN logits 行后，该步 sparse attention 静默读取陈旧位置集，无 crash、无 IMA；报告者已撤回服务层影响的演示，生产接受率下降归因待验证。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/58627

### INF-0653 `sglang_dsa_kpool_fp8_indexer_corrupts_16k_niah_needle_digits` SGLang DSA fp8 k-pool indexer 16K 长上下文错值

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-09-28 · 旧号 6.260

- **原因对象**：算子/内核 / sglang / DSA k-pool fp8 indexer
- **原因行为**：未逐行定位；三引擎对照把缺陷收窄到 SGLang k-pool lightning indexer 实现（kernel 语义、key-scale 处理或池化 logits 上的 top-k 选择），needle 所在 key 被丢弃或损坏。
- **影响对象**：输出内容（静默错误）
- **影响详情**：GLM-5.3-Flash fp8 在 16K 上下文 RULER niah_single_2 准确率降至 55%（4K 100%、8K 95%），逐字引用正确 needle 句却输出错误数字；vLLM 与 transformers 同输入正确，--dtype bfloat16 仅部分缓解。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/41494

### INF-0671 `flashinfer_fmha_v2_prefill_leaves_counters_shared_workspace_poisons_xqa_decode` flashinfer fmha_v2 遗留计数器污染 XQA decode

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-10-12 · 旧号 6.278

- **原因对象**：算子/内核 / flashinfer / fmha_v2/XQA 共享 workspace
- **原因行为**：fmha_v2 prefill 把 tile_id_counter 等计数器放在 workspace 开头，原子递增后留下非零值；XQA decode 却把同一 workspace 前 8 MB 当作须零初始化的 semaphore 区。
- **影响对象**：数值结果（静默错误）
- **影响详情**：SM90/SM120 上同一引擎共享 workspace 时，每次 prefill 后的首个 decode 静默产生错误注意力输出，无 crash；SGLang 已传 workspace[8MB:] 切片规避，vLLM 共享 workspace 的 trtllm 路径同样暴露。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/flashinfer-ai/flashinfer/issues/6151

### INF-0684 `trtllm_fp16_fails_fp8_normal` TRT-LLM Llama-3-70B FP16/BF16 失败而 FP8 正常

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-07-29 · 旧号 7.11

- **原因对象**：算子/内核 / tensorrt-llm / FP16/BF16 路径自定义 kernel
- **原因行为**：FP16/BF16 推理路径所用的自定义 kernel 存在缺陷，FP8 路径使用另一 kernel 实现而恰好规避；具体缺陷未定位（原文推测为数值范围保护缺失或累加精度问题）。
- **影响对象**：数值结果；输出内容（数值异常）
- **影响详情**：TRT-LLM 0.21.c-rc0（8×RTX 6000D、CUDA 13.0）上 Llama-3-70B 以 FP16/BF16 推理时输出错误、NaN 或 crash，而 FP8 量化推理正常。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/NVIDIA/TensorRT-LLM/issues/16899

### INF-0701 `vllm_gemma4_nvfp4_fp8_kv_blackwell_fa_version_selection_mismatch` Gemma-4 fp8 KV 后端选择与运行时 FA 版本不一致致崩溃

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-08-11 · 旧号 7.28

- **原因对象**：算子/内核 / vllm / FlashAttention 后端选择与版本解析
- **原因行为**：选择时不带 local-attention 信息解析 FA 版本，SM100+fp8 KV 得 FA4 而选 FLASH_ATTN；运行时带该信息，head_size=256 降为不支持 fp8 KV 的 FA2。
- **影响对象**：可用性/进度（报错）
- **影响详情**：Gemma-4 NVFP4 + fp8 KV cache 在 Blackwell SM100 上 warmup 崩溃，报 FlashAttention only support fp16 and bf16 data type；0.26 同配置选 TRITON_ATTN 正常服务。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/51828

### INF-0766 `vllm_gfx942_aiter_fused_moe_blockscale_fp8_1stage_engine_freeze_silent_death` gfx942 aiter FP8 fused MoE 内核致引擎冻结或无错死亡

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `var` · 发现 2026-09-17 · 旧号 7.93

- **原因对象**：算子/内核 / rocm / aiter FP8 fused MoE 内核
- **原因行为**：aiter blockscale-FP8 fused MoE 1stage 内核的 gfx942 路径在新 batch 宽度上卡死于内核内或引发无信号 worker 终止；分发记录含 kernelName2=nan。
- **影响对象**：可用性/进度（挂起）
- **影响详情**：MI325X TP8+MTP 服务 GLM-5.3 FP8 时，新 batch 宽度首次出现后吞吐跌至 0、请求驻留、仅 2/8 GPU 满载；部分 75s–12.8min 后自愈，部分 worker 无任何错误退出，仅留 shm_broadcast 超时日志；gfx950 无此问题。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/57429
- **关联**：var:vllm_rocm_quickreduce_bf16_fp16_no_saturation_inf_poisoning

### INF-0770 `sglang_deepseek_chunked_prefix_lse_base2_merged_as_natural_log_silent_overweight_cached_prefix` SGLang MLA 分块前缀按自然对数合并 base-2 LSE 致前缀过权

`故障` · `active` · `infra_off` · `other_stack` · `verified` · `NEW` · 发现 2026-09-23 · 旧号 7.97

- **原因对象**：算子/内核 / sglang / merge_state_v2 LSE 底数转换
- **原因行为**：trtllm_mla 等 ragged attention 内核产出 base-2 的 per-chunk LSE，merge_state_v2 及其 Triton fallback 按自然对数公式合并，中间缺少底数转换。
- **影响对象**：数值结果；输出分布（静默错误）
- **影响详情**：DeepSeek MLA 每次 prefix-cache 命中走 chunk-and-merge 时，缓存前缀注意力按 2^(0.4427·Δ) 被过权、新 token 权重被压，输出偏向历史上下文；Blackwell 默认后端即触发，无报错无日志，前缀越长越严重。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/40903

### INF-0789 `gqa_repeat_kv_rope_order` GQA repeat_kv 与 RoPE 顺序或头组映射错误致注意力偏差

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 待定 · 旧号 8.5

- **原因对象**：算子/内核 / 通用 / GQA repeat_kv 与头组映射
- **原因行为**：repeat_kv 相对 RoPE 的执行顺序不一致，或 query head→KV group 映射写错（repeat 与 repeat_interleave 混用），使注意力头对着错误的 KV group 计算。
- **影响对象**：隐状态；输出内容（质量退化）
- **影响详情**：顺序变体只造成极小的 logit 漂移；错组变体造成静默质量下降，退化局限于受影响的 head，只看整层聚合隐状态时信号被稀释。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/deepspeedai/Megatron-DeepSpeed/issues/294；https://github.com/huggingface/transformers/issues/25420

### INF-0798 `vllm_dspark_swa_index_width_flashinfer_mla_sparse_crash` vLLM DSpark 与 FlashInfer SWA 索引宽度不符致崩溃

`故障` · `active` · `infra_off` · `yes` · `verified` · `NEW` · 发现 2026-08-04 · 旧号 8.14

- **原因对象**：算子/内核 / vllm / DSV4 sparse MLA SWA 元数据
- **原因行为**：DSpark draft 按更宽的 noncausal_index_width 分配 decode_swa_indices，FlashInfer DSV4 sparse MLA 建索引元数据时仍按 window_size reshape。
- **影响对象**：可用性/进度（报错）
- **影响详情**：DeepSeek-V4-Pro-DSpark + FLASHINFER_MLA_SPARSE_DSV4 下抛 RuntimeError（如 shape '[7, 128]' invalid for input of size 1792），非静默。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/51042

### INF-0800 `sglang_cpu_backend_silent_full_attn_fallback_ignores_sliding_window` SGLang CPU 后端忽略滑窗静默回退全局注意力

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-08-06 · 旧号 8.16

- **原因对象**：算子/内核 / sglang / CPU 注意力后端滑窗路径
- **原因行为**：CPU attention 后端未实现 sliding window 路径，上下文超过 local_attention 窗口时静默改用全局 attention。
- **影响对象**：输出内容；数值结果；隐状态（静默错误）
- **影响详情**：--device cpu 运行 ModernBERT 等 encoder 模型、长上下文超过局部窗口时，embedding 与 HF Transformers 不一致，cosine similarity 降到约 0.86（GPU 约 1.0），无警告、无 NaN。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/33603

### INF-0811 `sglang_flashinfer_nosplit_cudagraph_uninitialized_grid_entries` FlashInfer no-split 填充 grid 含未初始化项

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-08-10 · 旧号 8.27

- **原因对象**：算子/内核 / flashinfer / FA2 planner grid-x 填充
- **原因行为**：no-split 下 planner 把 grid-x 填充到 occupancy 上界，但 entries 只为真实 work 构造，padded tail 无 validity mask，可能启动未初始化 entry 的 block。
- **影响对象**：数值结果；性能/资源（静默错误）
- **影响详情**：SGLang pin FlashInfer 0.6.15.post1、确定性 CUDA-graph decode 走 FA2 no-split 路径时：tail 为零只产生冗余工作；复用的非零 tail 数据带来条件性 stale-index 风险；无 crash。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/pull/34190

### INF-0813 `vllm_amd_kda_packed_decode_unit_stride_assumption_spec_decode` vLLM AMD KDA decode kernel 假设单位步幅致推测解码崩溃

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-10 · 旧号 8.29

- **原因对象**：算子/内核 / vllm / AMD KDA packed decode 内核
- **原因行为**：AMD 版 fused_recurrent_kda_packed_decode 假设 state_indices 单位步幅并预先拒绝非连续 tensor（NVIDIA 版已支持 stride 参数）。
- **影响对象**：可用性/进度（报错）
- **影响详情**：AMD 上 KDA 线性注意力开启推测解码时，GDNAttentionMetadataBuilder 取 block_table_tensor[:, 0] 得到 strided view，warmup 阶段抛 ValueError 崩溃；无推测解码时不暴露。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/51682

### INF-0819 `sglang_aiter_attention_sinks_silently_dropped_positional_arg_truncation` SGLang aiter 后端调用点静默丢弃 attention sinks

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-08-12 · 旧号 8.35

- **原因对象**：算子/内核 / sglang / aiter_backend.py 调用点
- **原因行为**：extend_attention_fwd 以 16 个位置参数+1 个 keyword 调用，callee 签名增长后 sliding_window_size 与 sinks 未被传入，静默回退默认值。
- **影响对象**：输出分布；数值结果（静默错误）
- **影响详情**：ROCm aiter backend 的 extend attention 运行带 attention sinks / 滑动窗口的模型时，无 crash、无 NaN，attention 不受 sinks 与滑动窗口约束，logits 静默错误。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/pull/34463

### INF-0824 `ttm_sdpa_fw_no_mask_deadlock_softmax_scale_mismatch_masked_nan` tt-train SDPA 无 mask 挂死、scale 不一致与反向 NaN

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-15 · 旧号 8.40

- **原因对象**：算子/内核 / tt-metal / sdpa_fw/sdpa_bw kernel
- **原因行为**：sdpa_fw 的 None mask 分支无条件 apply/pop 一个 host 从不创建的 mask CB，validation 却放行；fw/bw softmax scale 不一致；sdpa_bw 对全掩码行分母为 0。
- **影响对象**：可用性/进度；梯度；数值结果（挂起）
- **影响详情**：ring-attention 以 sdpa_fw(..., None) 调用时前向死锁或产出 garbage；Wormhole 上 fw/bw scale 不一致造成训练静默数值漂移；fully-masked 行（score 全 -inf）反向产生 NaN。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/53207

### INF-0826 `sglang_dsa_sparse_mla_griddim_z_zero_launch_silent` SGLang DSA sparse MLA gridDim.z 溢出致静默零发射

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-08-18 · 旧号 8.46

- **原因对象**：算子/内核 / sglang / DSA sparse MLA kernel 启动
- **原因行为**：prefill=trtllm 路径把 token 摊平进 batch 维，numCtasZ 超过 gridDim.z 上限 65535 时发射 0 个 CTA，cuErrCheck 只打印不抛，该路径缺 65535 guard。
- **影响对象**：数值结果；输出内容；指标/日志（静默错误）
- **影响详情**：SM100 上单次 unchunked extend ≥65536 token 时 attention 输出保持未初始化 sentinel 原样返回并进入残差流，garbage 输出且无异常无告警；bench_one_batch 报出虚假地快的 prefill 基准数字。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/34947

### INF-0827 `trtllm_sm120_e4m3_headsize256_kernel_gap_fallback` TRT-LLM sm_120 e4m3 缺 head 256 内核致静默回退

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-08-18 · 旧号 8.47

- **原因对象**：算子/内核 / tensorrt-llm / fmha_v2 sm_120 e4m3 内核枚举
- **原因行为**：fmha_v2/setup.py 的 sm_120 e4m3→bf16 内核枚举缺 head_size 256，head_dim=256 + FP8 KV 无 fused paged-context FMHA，静默回退 unfused。
- **影响对象**：性能/资源（性能退化）
- **影响详情**：RTX 5090（sm_120）上 GQA head_dim 256 模型（如 Qwen3.6-35B-A3B）+ FP8 KV 时无报错，回退路径请求约 626GB workspace，KV block 复用失效，表现为 prefix caching 不工作，且无指向根因的诊断。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/NVIDIA/TensorRT-LLM/issues/17911

### INF-0831 `vllm_dsv4_flashmla_sparse_prefill_phase1_crash_161k` vLLM FlashMLA sparse prefill 超长上下文崩溃

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-08-17 · 旧号 8.45

- **原因对象**：算子/内核 / vllm / FlashMLA sparse prefill
- **原因行为**：超长上下文下 sparse 索引/分块参数越过 FlashMLA sparse prefill 内核的假设边界，在 phase1.cuh:614 处非法访存。
- **影响对象**：可用性/进度（报错）
- **影响详情**：DeepSeek-V4-Flash w4a16、TP=8、FP8 KV、chunked prefill 8192、DSpark 推测解码，在 8×H20-3e 上约 161K 上下文 sparse prefill 时 CUDA illegal memory access 崩溃，非静默。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/52339

### INF-0832 `vllm_xpu_qnorm_rope_unordered_twin_store_race` vLLM XPU RoPE 融合 kernel 双 store 竞态损坏 KV

`故障` · `active` · `infra_off` · `partial` · `verified` · `NEW` · 发现 2026-08-19 · 旧号 8.48

- **原因对象**：算子/内核 / vllm / qnorm/RoPE 融合 kernel
- **原因行为**：同一程序先整体 store 未旋转的 rope 区，再以 stride-2 地址映射覆写旋转值，两次 store 写相同地址且无顺序保证，落地值取决于后端指令调度。
- **影响对象**：KV状态；输出内容（不可复现）
- **影响详情**：MTT S5000（MUSA）上 4096 token×64 head 中 30 个 token 的 cache RoPE 区与 golden 不符（29 个为未旋转原值），同输入再跑为 29 个；NVIDIA 调度恰好掩盖竞态；注意力漂移、生成质量退化，无 crash。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/52415

### INF-0833 `vllm_aiter_sparse_pa_spec_decoding_corruption` vLLM ROCm AITER sparse PA 原型路径叠加推测解码输出损坏

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-19 · 旧号 8.49

- **原因对象**：算子/内核 / vllm / AITER sparse PA 原型路径
- **原因行为**：AITER 与 shuffled KV layout 同开时自动选中 AITER sparse paged-attention 原型路径（无 opt-in），该路径与推测解码组合时计算有误，具体根因待定。
- **影响对象**：输出内容；输出分布（静默错误）
- **影响详情**：MiniMax-M3 FP8 TP8（MI325X）+ DSpark 推测解码时，输出从正确开头退化为重复/mojibake，smoke test 易漏检；acceptance 从 44% 降至 7%；改走 Triton sparse 路径即正确，无错误日志。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/52860

### INF-0834 `vllm_dsv4_sm120_constant_argmax_logprob_degenerate` vLLM DSV4 SM120 sparse MLA decode 恒定输出退化

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-19 · 旧号 8.50

- **原因对象**：算子/内核 / vllm / SM120 sparse MLA decode
- **原因行为**：SM120 上 FLASHINFER_MLA_SPARSE_DSV4 路径的 decode 步获得的注意力上下文恒定或被清零，前向计算断裂；具体根因待定。
- **影响对象**：输出分布；输出内容（静默错误）
- **影响详情**：SM120 TP4 + FP8 KV 服务 DeepSeek-V4-Flash，HTTP 200 正常，greedy 下每个 decode 位置都返回同一特殊 token 且 logprob 完全一致，temperature=1 为噪声；TP、DP+EP、enforce-eager 均复现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/52938

### INF-0835 `sglang_cpu_amx_extend_forced_causal_mask_bidirectional` SGLang CPU AMX 后端对双向模型强制 causal mask

`故障` · `active` · `infra_off` · `other_stack` · `verified` · `NEW` · 发现 2026-08-22 · 旧号 8.51

- **原因对象**：算子/内核 / sglang / extend_attention_cpu
- **原因行为**：Intel AMX CPU attention 后端的 extend_attention_cpu 对 extend 阶段恒定施加 causal mask，未按模型 mask 类型分派，双向 encoder-only 自注意力也被因果掩码。
- **影响对象**：数值结果；隐状态；输出内容（静默错误）
- **影响详情**：CPU 后端运行 bge-reranker 等 BERT 族 reranker/embedder 时无 crash，输出静默错误；rerank 中内容不同的文档得分完全相同（如均为 2.611328125）。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/pull/35434

### INF-0836 `vllm_dsv4_sm120_sparse_mla_page_size_advertise_mismatch` vLLM DSV4 SM120 页大小声明错配致专用 decode 内核被跳过

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-08-23 · 旧号 8.52

- **原因对象**：算子/内核 / vllm / DSV4 sparse 后端页大小声明
- **原因行为**：DSV4 SparseMLA/Indexer 后端硬编码声明 [256]，而 FlashInfer 在 SM120 按 64-token 页编译 DSV4 decode，块大小协商无法把 256 拆成内核页。
- **影响对象**：性能/资源（性能退化）
- **影响详情**：SM120/GB10（DGX Spark）+ --block-size 256 时专用 sparse decode 内核被静默跳过、退回非专用路径，吞吐/延迟退化；无 crash、无警告日志，无信号说明专用内核未被使用。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/53425

### INF-0839 `sglang_fast_topk_v2_threshold_bucket_overflow_silent_candidate_drop` sgl-kernel fast_topk_v2 候选缓冲溢出静默错选

`故障` · `active` · `infra_off` · `config` · `verified` · `NEW` · 发现 2026-08-28 · 旧号 8.55

- **原因对象**：算子/内核 / sglang / sgl_kernel.fast_topk_v2
- **原因行为**：两遍 radix top-k 在 threshold bin 候选超过 4096 条 shared-memory buffer 上限时静默丢弃超出部分，不置溢出标记，后续 threshold 选择基于截断总体。
- **影响对象**：数值结果（静默错误）
- **影响详情**：长行（≥约 100K 列）× 大 k（如 2048）时输出恰好 k 个合法去重索引但混入非 top-k 成员；64×256K、k=2048 时 64 行全错、每行 6–42 条误选，哪些行出错 run-to-run 不确定；无 crash、无形状异常。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/36807

### INF-0841 `ttmetal_sdpa_zero_padded_mask_columns_read_as_allowed` tt-metal SDPA 将零填充 mask 列读作允许注意力

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-08-29 · 旧号 8.57

- **原因对象**：算子/内核 / tt-metal / SDPA 显式 attn_mask 消费路径
- **原因行为**：SDPA 消费显式 mask 时不区分逻辑列与物理 padding 列，而 from_torch 对 TILE 布局默认以 0 填充，key 长度非 32 倍数时 padding 的 0 被当作“允许注意力”而非屏蔽。
- **影响对象**：数值结果（静默错误）
- **影响详情**：Wormhole Galaxy 上显式 attn_mask + 默认 pad + Sk%32≠0（如 S=16/40）时 attention 输出严重错误且确定，无 crash；S=32 正常，改 pad_value=-inf 即与 is_causal=True 对照逐位一致。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/54833

### INF-0842 `vllm_hy4_sm120_flashinfer_mla_sparse_degenerate_output` vLLM SM120 hy_v4 sparse MLA 后端输出退化

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-30 · 旧号 8.58

- **原因对象**：算子/内核 / vllm / SM120 sparse MLA 后端
- **原因行为**：嫌疑在该 SM120 sparse-MLA 后端：fp8_ds_mla KV 数值，或 sparse kernel 对 hy_v4 gated DSA/IndexCache 的处理（此前仅验证 DeepSeek/GLM 形状）；根因待定。
- **影响对象**：输出内容（质量退化）
- **影响详情**：SM120（RTX PRO 6000）上 hy_v4/Hy4-preview 任意 prompt（含 5-token）temp=0 都产出 " of of of…" 退化输出且可复现，无 crash；SM120 上没有替代 sparse 后端，模型在该硬件上不可用。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/54434

### INF-0844 `ttmetal_gemma4_wh_length_gated_deterministic_garbled_100k` tt-metal Wormhole 上 Gemma4 特定编码长度确定性输出乱码

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-09-01 · 旧号 8.60

- **原因对象**：算子/内核 / tt-metal / Gemma4 prefill 前向（具体算子待定）
- **原因行为**：Wormhole 上 Gemma4 共享 prefill 前向在特定长度窗口损坏；与 serving 层和 prefill chunk 配置无关，原文推测为 WH 特定的分块/寻址边界（如 tile 对齐或 index 回绕），根因未定位
- **影响对象**：输出内容；隐状态（静默错误）
- **影响详情**：Wormhole T3K 上 Gemma4 12B/31B 在特定编码长度（如 100,793、109,857、127,574）确定性输出 garbled 长文，长度非单调（相邻长度干净）；Blackhole 同长度正常，metal demo 与 vLLM serving 输出字节一致，无报错
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/55102

### INF-0851 `trtllm_flashinfer_splitkv_scratch_stale_partials_nonfinite_merge` TRT-LLM split-KV workspace 残留致非有限注意力输出

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-09-05 · 旧号 8.67

- **原因对象**：算子/内核 / tensorrt-llm / split-KV workspace 分配与复用
- **原因行为**：共享 FlashInfer workspace 用 torch.empty 分配未清零、运行前不重置；split-KV merge 读取 kernel 未写的 tile，混入早前 plan 残留 partials（LSE 可为 −inf）
- **影响对象**：数值结果；隐状态（数值异常）
- **影响详情**：TRT-LLM 使用 FlashInfer split-KV 注意力（paged prefill/decode、ragged prefill）并开 CUDA graph 时，batch 组合变化后间歇出现非有限（NaN/inf）注意力输出行；损坏内容取决于此前请求的计划形状历史
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/NVIDIA/TensorRT-LLM/pull/18755

### INF-0853 `vllm_rocm_aiter_mla_qseqlen8_gfx942_missing_kernel_abort` AITER MLA 超支持域 abort 杀测试进程且容差低于 ULP

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `var` · 发现 2026-09-06 · 旧号 8.69

- **原因对象**：算子/内核 / vllm / AITER MLA query_len 支持声明；观测/评测 / vllm / chunked-context MLA 测试容差
- **原因行为**：ROCM_AITER_MLA 宣称支持多 token 查询，但 gfx942 无 gqa_ratio≥16、qseqlen>4 的 kernel，守卫只覆盖 heads<16；另测试容差 1e-2 低于 1 bf16 ULP
- **影响对象**：可用性/进度；测试判定（报错）
- **影响详情**：gfx942 上 spec decode query_len>4 + bf16 KV 时 kernel 查找失败，修复前 aiter 以 std::abort 杀死 pytest worker 及同 shard 测试；另有 2 ULP 正常舍入差被判失败（假阳性）
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/55609
- **关联**：var:vllm_fp8_kv_sparse_mla_sm90_storage_dtype_startup_reject

### INF-0858 `sglang_minicpm_fused_topk_causal_and_sync_selection_errors` MiniCPM fused top-k 选择器三处错致选错块

`故障` · `active` · `infra_off` · `other_stack` · `verified` · `var` · 发现 2026-09-06 · 旧号 8.74

- **原因对象**：算子/内核 / sglang / MiniCPM fused top-k 选择器
- **原因行为**：fused top-k 选择器禁用了共享内存同步，跨 warp MMA 可能在共享 Q/K tile 写同步前读取；重叠压缩窗可给未来块打分、挤掉因果候选；先按块 ID 排序再截断候选缓冲会丢高分块
- **影响对象**：数值结果；输出内容（静默错误）
- **影响详情**：SGLang 服务 MiniCPM-SALA 长上下文时无报错，sparse attention 选中未来块（如 query 位 8895 选中块 139）、漏掉最近/forced 块，长上下文检索质量静默退化
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/pull/38232
- **关联**：var:sglang_fast_topk_threshold_bucket_overflow_silent_candidate_drop

### INF-0859 `sglang_infllm_v2_pooling_wrapper_logical_stride_vs_padded_width` InfLLM-v2 池化 wrapper 按逻辑长度算 stride 读错

`故障` · `active` · `infra_off` · `other_stack` · `verified` · `var` · 发现 2026-09-06 · 旧号 8.75

- **原因对象**：算子/内核 / sglang / InfLLM-v2 压缩分池化 wrapper
- **原因行为**：压缩分池化 wrapper 按逻辑 context 长度推导行 stride（8500//16=531），而 score 张量实际 padded 到 128 倍数宽（640），后续行与注意力头的读 offset 全错
- **影响对象**：数值结果；输出内容（静默错误）
- **影响详情**：SGLang 跑 MiniCPM-SALA（max_context_len=8500）时无报错，块选择分数读错行/头（实测 39 个有限条目错、最大绝对误差 0.136，inf 掩码行同样错位），sparse 块选择静默偏差，长上下文检索退化
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/pull/38231
- **关联**：var:INF-0748

### INF-0860 `sglang_ascend_attn_page16_kernel_tiling_silent_garbage` SGLang Ascend 后端 page_size=16 输出连贯但语义垃圾

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `var` · 发现 2026-09-06 · 旧号 8.76

- **原因对象**：算子/内核 / sglang / ascend 注意力 page16 路径
- **原因行为**：ascend attention 后端在 page_size=16 时通过 aclnnFusedInferAttentionScoreV3 的 tiling 检查，但其后执行语义出错；原文未定位具体根因
- **影响对象**：输出内容（静默错误）
- **影响详情**：Ascend 910C 上 SGLang 用 ascend 后端、--page-size 16 时，每个请求从首个 decode 步起输出连贯却语义垃圾的文本，无错误无警告、指标健康；page 1 被拒、page 128 正确，无推测解码也复现
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/38196
- **关联**：var:INF-0836

### INF-0861 `vllm_flashinfer_sm90_sparse_mla_tp_ep_mtp_ima_h20` H20 上 GLM-5.3-Flash TP+EP+MTP 运行期非法显存访问

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `var` · 发现 2026-09-07 · 旧号 8.77

- **原因对象**：算子/内核 / vllm / 待定（IMA 浮出于 sparse MLA）
- **原因行为**：根因未定位：某个更早的 kernel 发生非法显存访问，异步报错在 FlashInfer SM90 sparse MLA 后端构建 attention metadata 的 _kv_lens_host 同步点浮出
- **影响对象**：可用性/进度（报错）
- **影响详情**：8×H20 上 vLLM nightly 服务 GLM-5.3-Flash（TP+EP、MTP k=5），65,536 in/out × batch 32 负载下普通 chat 请求中途 Worker 死于 CUDA illegal memory access，服务崩溃
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/55626
- **关联**：var:INF-0848

### INF-0864 `flashinfer_split_kv_partial_output_rounded_to_output_dtype_before_merge` split-KV partial 在 merge 前被舍入到输出 dtype

`故障` · `active` · `infra_off` · `yes` · `verified` · `NEW` · 发现 2026-09-10 · 旧号 8.80

- **原因对象**：算子/内核 / flashinfer / split-KV tmp_v partial 存储
- **原因行为**：split-KV 的 tmp_v 按 float 分配，却以输出 dtype 写入和读回，cast_store 在 merge 前就把 fp32 partial 累加器舍入到 bf16/fp16；prefill 调度器同型
- **影响对象**：数值结果（静默错误）
- **影响详情**：split-KV 激活时（如 KV_LEN 16384 的长上下文 decode），1/3–1/2 的 decode 输出相对 fp32 参考偏差 ≥1 个 bf16 步长，关闭 split 后降至 0.2%；无报错
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/flashinfer-ai/flashinfer/issues/5107

### INF-0865 `ttmetal_sdpa_decode_k_chunk_size_32_second_chunk_garbage_default_path` sdpa_decode chunk 32 第二块起输出垃圾

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `var` · 发现 2026-09-10 · 旧号 8.81

- **原因对象**：算子/内核 / tt-metal / sdpa_decode 多 chunk 归并
- **原因行为**：sdpa_decode 在 k_chunk_size=32 时对第二个及之后 chunk 的边界/累加处理错误；默认 get_chunk_size() 取 KV 长度最大 2 的幂因子，32·奇数长度会自动选 32
- **影响对象**：数值结果；输出内容（静默错误）
- **影响详情**：Wormhole T3K 上 k_chunk_size=32 且 cur_pos≥32 时 decode 注意力输出垃圾（PCC 崩塌）；无 program_config 时 96/160/224 等 KV 长度默认中招，如 Qwen3-VL 96 位 KV 产生垃圾 token，128 位缓存正确
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/56171
- **关联**：var:INF-0063

### INF-0866 `flashinfer_sm120_sparse_mla_prefill_gt64q_topk_lane_unbounded_corruption` SM120 sparse-MLA prefill 超 64 query 输出损坏

`故障` · `active` · `infra_off` · `partial` · `verified` · `NEW` · 发现 2026-09-10 · 旧号 8.82

- **原因对象**：算子/内核 / flashinfer / SM120 sparse-MLA prefill
- **原因行为**：sparse-MLA prefill 内核不对 indices/extra_indices 中位置 ≥ topk_length[i] 的 lane 设界，陈旧或垃圾索引被当作有效 KV block 参与注意力
- **影响对象**：数值结果；输出内容（静默错误）
- **影响详情**：SM120（RTX PRO 6000）上 DSv4 提示超过 64 query 走 prefill 内核时，temperature=0 输出垃圾/无关文本/重复 CJK 片段，18–62 token 正常；EP1+marlin MoE 下同请求两次输出不同；把越界 lane 掩成 −1 即恢复
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/flashinfer-ai/flashinfer/issues/5095

### INF-0868 `sglang_glm5x_nope_mla_sm120_no_dsa_backend_available_unrunnable` SGLang GLM NoPE MLA 在 SM120 无可用 DSA 后端

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `var` · 发现 2026-09-13 · 旧号 8.84

- **原因对象**：算子/内核 / sglang / sparse-MLA 后端 smem/维度特化
- **原因行为**：各 sparse-MLA 后端的 smem 预算与维度特化只按 Hopper/B200（576 维）编写：tilelang 需 202KB smem 超 SM120 上限，FlashInfer 内核拒绝 NoPE 512 维
- **影响对象**：可用性/进度（报错）
- **影响详情**：SGLang 在 SM120（RTX PRO 6000 Blackwell）上服务 GLM-5.x（qk_rope_head_dim=0）时启动或首 token 崩溃、kernel launch 被拒，模型完全不可服务；fp8 KV 下按构造不可运行，bf16 KV 下其余后端各自受阻
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/39302
- **关联**：var:INF-0846

### INF-0872 `flashinfer_ragged_prefill_plan_noncontiguous_indptr_stride_ignored_wrong_attention` flashinfer plan() 忽略非连续 indptr 的 stride

`故障` · `active` · `infra_off` · `partial` · `verified` · `NEW` · 发现 2026-09-16 · 旧号 8.88

- **原因对象**：算子/内核 / flashinfer / Ragged prefill plan()
- **原因行为**：plan() 对 indptr 做 host 拷贝或透传设备指针时按 data_ptr 线性读取 numel 个元素，忽略 stride/storage_offset，也不拒绝或归一化非连续张量
- **影响对象**：数值结果（静默错误）
- **影响详情**：调用方传非连续 indptr view（如 indptr_full[::2]）时序列边界错位，部分 token 的 attention 在错误范围内计算，输出部分错误且无报错；连续拷贝结果正确，cuDNN 开关两态均复现；现有框架传连续缓冲，生产面窄
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/flashinfer-ai/flashinfer/issues/5260

### INF-0873 `vllm_dsv41_sm120_cuda_graph_dummy_capture_poisons_null_kv_block_nan` DSV4.1 图捕获毒化 null KV 块致 attention 输出 NaN

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `var` · 发现 2026-09-16 · 旧号 8.89

- **原因对象**：算子/内核 / vllm / decode_dsv4 内核无效索引读
- **原因行为**：decode 内核对无效索引钳位读 slot 0 再靠 mask 丢弃，只有 scale load 有 idx_raw>=0 守卫，NoPE/RoPE 数据 load 没有；图捕获期 dummy batch 把 NaN 写进 block 0
- **影响对象**：数值结果；输出内容（数值异常）
- **影响详情**：DeepSeek-V4.1-Flash 在 GB10/SM120 上开启 CUDA graphs（FULL/PIECEWISE/FULL_DECODE_ONLY 均中招）时首层起 attention 输出 NaN，服务返回空串或乱码；--enforce-eager 完全正确
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/57156
- **关联**：var:INF-0630

### INF-0874 `vllm_gfx1151_navi_paged_attention_silently_accepts_unsupported_alibi_and_block32` gfx1151 分页注意力静默忽略 ALiBi、block32 错值

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-09-18 · 旧号 8.90

- **原因对象**：算子/内核 / vllm / gfx1151 paged_attention
- **原因行为**：Navi（gfx1151）paged attention 内核的 ALiBi 斜率分支与 block32 分页路径未实现或未启用，入口校验只查支持项、不拒绝不支持项，不支持的组合沿默认路径按错误布局/偏移计算
- **影响对象**：数值结果（静默错误）
- **影响详情**：无 assert、fallback 或 warning：给 ALiBi 时输出等同 no-ALiBi 参考（误差约等于输出全幅）；block size 32 从 seq_len 17 起错值；只有 block16 且无 ALiBi 时正确；v0.26.0 与 main 均复现
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/57614

### INF-0875 `vllm_kimi_k3_fp8_latent_cache_nan_bytes_pv_mask_bypass_mla_decode_detonation` KDA 内核 NaN 污染 latent cache，P·V 穿透掩码

`故障` · `active` · `infra_off` · `partial` · `verified` · `var` · 发现 2026-09-19 · 旧号 8.91

- **原因对象**：算子/内核 / vllm / KDA chunk 内核、MLA P·V
- **原因行为**：混合批中 chunk_kda_with_fused_gate 对 decode 行返回 NaN（NVIDIA 副本缺 #50649 修复），写进 latent cache 新页头行；decode 的 mask 只管 QK，P·V 仍读坏字节
- **影响对象**：KV状态；输出分布（数值异常）
- **影响详情**：并发长序列负载下 3–30 分钟间歇触发：verify 步 target logits 全 NaN、argmax 退化为 0，该请求 drafter 分数永久 NaN，无守卫时 live-lock 到 max-token 并输出垃圾；NaN 首现于输入全有限的 MLA 层，bf16 KV 同样触发
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/57716
- **关联**：var:INF-0414

### INF-0883 `vllm_rocm_qwen38_27b_fp8_aiter_accuracy_collapse_high_concurrency` ROCm AITER 下 Qwen3.8-27B-FP8 高并发准确率坍塌

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `var` · 发现 2026-10-09 · 旧号 8.99

- **原因对象**：算子/内核 / vllm / AITER fp8 注意力/GEMM 路径
- **原因行为**：待定：根因未定位，单变量对照只锁定到 VLLM_ROCM_USE_AITER=1 启用的 AITER 路径，原文推测为并发相关的 fp8 注意力/GEMM 路径
- **影响对象**：输出内容（质量退化）
- **影响详情**：Qwen3.8-27B-FP8 在 MI355X TP4 上高并发服务时准确率坍塌、无 crash；VLLM_ROCM_USE_AITER=1 触发，关闭 AITER 恢复正常
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/60160
- **关联**：var:vllm_dsv4_flash_fp8_mi300x_length_smooth_degradation

### INF-0899 `sglang_qwen38_nvfp4_fused_qknorm_rope_outgate_grounding` SGLang 融合 RoPE 路径致 Qwen3.8 NVFP4 坐标偏移

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `var` · 发现 2026-08-22 · 旧号 9.16

- **原因对象**：算子/内核 / sglang / 融合 QK-norm+RoPE+gate 路径
- **原因行为**：CUDA 且 attn_output_gate 时 qwen3_5.py 走的 QK-RMSNorm+RoPE+output-gate 融合路径数值错误，native 路径正确。
- **影响对象**：数值结果；输出内容（质量退化）
- **影响详情**：Qwen3.8-27B-NVFP4 在 SGLang CUDA 上 2D grounding 坐标严重偏移（中心误差 178.3px、IoU 0），同 checkpoint/图/prompt 在 vLLM 0.27.1 与 native 路径准确；生成文本表观正常，文本质量评测难以暴露。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/35949
- **关联**：var:INF-0891

### INF-1086 `vllm_qwen38_flash_next_mtp_episodic_zero_acceptance_repetition_collapse` vLLM Qwen3.8-Flash-Next+MTP 偶发接受率归零与重复

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `var` · 发现 2026-09-04 · 旧号 11.63

- **原因对象**：算子/内核 / vllm / PLE short-conv 内核索引加载
- **原因行为**：PR #55467 定位：PLE short-conv kernel 把带步幅的 state_indices 列视图当连续加载，prefill 行 r≥1 落入 request 0 的 spec 块，conv state 被覆写
- **影响对象**：KV状态；输出内容；性能/资源（质量退化）
- **影响详情**：NVFP4 + MTP(k=3) + SM120 偶发持续数小时的 episode：acceptance 跌到 0、吞吐 66 vs 健康 95–135 tok/s、thinking 内无尽重复（一请求烧满 1M token），随后自愈；同镜像 Gemma 4 同签名
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/55357
- **关联**：var:INF-0571

### INF-1137 `vllm_punica_lora_long_seq_nan_token0` vLLM punica LoRA 长序列出 NaN，输出全为 "!"

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-08-19 · 旧号 12.21

- **原因对象**：算子/内核 / vllm / punica LoRA kernel 长序列路径
- **原因行为**：vLLM 动态 LoRA serving 的 punica Triton kernel 在长序列（>~400 token）路径上出错，原文推测为序列分块/累积路径越界或数值爆掉；base、merged LoRA、HF PEFT 对照均正常。
- **影响对象**：输出分布；输出内容（数值异常）
- **影响详情**：hybrid-GDN 模型 + 动态 LoRA，prompt 超过约 400 token 时 logits 为 NaN，argmax 得 token 0，输出全为 "!"；HTTP 200 无 crash，短 prompt 完全正常，smoke test 不可见。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/52568

### INF-1165 `vllm_legacy_fused_moe_lora_kernel_unconditionally_casts_operands_to_bf16` vLLM legacy fused MoE LoRA 内核无条件转 bf16

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-10-09 · 旧号 12.49

- **原因对象**：算子/内核 / vllm / legacy fused MoE LoRA 内核
- **原因行为**：legacy 内核 _fused_moe_lora_kernel 硬编码把两个操作数 cast 到 bfloat16 再做 tl.dot，不保留原生 dtype；本为 ROCm 类型推断的 workaround，被泛化为无条件。
- **影响对象**：数值结果；输出内容（质量退化）
- **影响详情**：fp16 MoE + LoRA 在 fully_sharded=True 或 max_lora_rank>128 时走该内核，乘加尾数从 10-bit 截到 7-bit，输出质量静默劣化；rank≤128 非分片的 one-shot 内核保留原生 dtype，两路径精度不同、难诊断。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/59920

### INF-1166 `vllm_gptoss_mixed_moe_lora_format_gate_up_slice_layout_mismatch_silent_wrong` vLLM GPT-OSS mixed LoRA gate/up 布局错位

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-10-09 · 旧号 12.50

- **原因对象**：算子/内核 / vllm / MoE LoRA gate/up 双切片写布局
- **原因行为**：enable_mixed_moe_lora_format=True 时 fused gate/up LoRA 切成两片，内核按连续区域写 delta，而 GPT-OSS 基座 GEMM 输出为 gate/up 交错布局。
- **影响对象**：数值结果；输出内容（静默错误）
- **影响详情**：GPT-OSS + mixed format LoRA（如 gpt-oss-20b-lora-text2sql）推理时 delta 加到错误通道再进 SwiGLU-OAI，shape 全程合法、无报错，结果静默错误；默认 3D wrapper 单融合切片路径不受影响。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/60811

### INF-1172 `bug_attn_backend_eager_sdpa_flash` eager/SDPA/flash 注意力后端切换使同一权重输出数值不一致

`故障` · `active` · `ambiguous` · `config` · `verified` · `NEW` · 发现 待定 · 旧号 13.4

- **原因对象**：算子/内核 / transformers / attention 后端 mask 与累加实现
- **原因行为**：attn_implementation 的三种后端数值实现不同：eager 加 float mask；sdpa 融合 softmax 与 matmul 并以 fp32 计算；flash 使用不同的 masking 与累加顺序。
- **影响对象**：数值结果；输出分布（静默错误）
- **影响详情**：切换后端后同权重 per-token logit 漂移，从约 1e-3（良性）到带 mask 病态情形下 argmax 翻转（transformers #36585 allclose 失败、#31787 FA2 不确定性）；幅度与运行时噪声底同量级，更适合作检测器的标定 null。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/huggingface/transformers/issues/36585；https://github.com/huggingface/transformers/issues/31787

### INF-1179 `trtllm_gdn_flashinfer_decode_index_alignment_noop` GDN decode 未查 index 对齐致 FlashInfer 崩溃

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-07-30 · 旧号 13.11

- **原因对象**：算子/内核 / tensorrt-llm / GDN decode FlashInfer 分派
- **原因行为**：initial_state_indices.int() 对已是 int32 的张量是 no-op，保留未对齐指针；_can_use_flashinfer_gdn_decode 不检查对齐且无 fallback。
- **影响对象**：可用性/进度（报错）
- **影响详情**：caller 传入元素 offset 不能被 8 整除的 index 切片视图时，FlashInfer CuTe-DSL GDN kernel 报 Misaligned Tensor data 硬错误崩溃；默认 zero-offset 视图不触发。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/NVIDIA/TensorRT-LLM/issues/17027

### INF-1180 `trtllm_kimiK3AttnRes_cuda_arch_guard_misadmit_sm120` kimiK3AttnRes 架构守卫误纳 sm_120 致编译失败

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-07-30 · 旧号 13.12

- **原因对象**：算子/内核 / tensorrt-llm / attnResFwd.cu 架构守卫
- **原因行为**：为 sm_100 家族定制、使用 tcgen05/TMEM PTX 的 attention kernel 用 __CUDA_ARCH__ >= 1000 守卫，sm_120 也满足；对象目标继承全局架构列表而非声明自己的。
- **影响对象**：可用性/进度（报错）
- **影响详情**：任何含 sm_120f 的 build 在 ptxas 报错 Instruction 'tcgen05.st' not supported，消费级 Blackwell 编译失败；datacenter Blackwell 不受影响。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/NVIDIA/TensorRT-LLM/pull/17085

### INF-1181 `sglang_gemma3_rmsnorm_highrank_skip_fused_mixed_dtype_nan` Gemma3RMSNorm 融合门控缺陷，dtype 不符时出 NaN

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-07-31 · 旧号 13.13

- **原因对象**：算子/内核 / sglang / Gemma3RMSNorm 融合分派
- **原因行为**：forward_cuda 仅对 2-D 输入分派融合 kernel，3-D 的 q_norm/k_norm 走 eager 路径；两条融合路径直接传 self.weight.data，不检查 dtype。
- **影响对象**：数值结果；性能/资源（数值异常）
- **影响详情**：权重 dtype 与融合 kernel 不匹配时返回 NaN 而非报错，NaN 静默传播到下游；gemma-3-1b 每次前向 157 次 norm 调用中有 52 次走未融合路径。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/32807

### INF-1186 `vllm_fused_mtp_rmsnorm_branch_dtype_mismatch_compile_fail` fused MTP RMSNorm 分支 dtype 不一致致编译失败

`故障` · `active` · `infra_off` · `yes` · `verified` · `NEW` · 发现 2026-08-04 · 旧号 13.18

- **原因对象**：算子/内核 / vllm / fused_mtp_input_rmsnorm
- **原因行为**：kernel 的两个运行时分支对局部变量 x 推断出不同 dtype：masked tl.where 路径提升到 FP32，hnorm 路径保留输入 dtype（如 bf16）。
- **影响对象**：可用性/进度（报错）
- **影响详情**：DeepSeek V4 使用 fused MTP RMSNorm kernel 时，Triton 因控制流分支合并处局部变量类型不一致而编译失败，属显式崩溃、非静默。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/50987

### INF-1187 `vllm_dsl_kernel_fp32_dtype_unsupported_qwen_gdn` GDN DSL kernel 不支持 float32 致服务启动失败

`故障` · `active` · `infra_off` · `yes` · `documented` · `NEW` · 发现 2026-08-06 · 旧号 13.19

- **原因对象**：算子/内核 / flashinfer / delta_rule_dsl_sm90
- **原因行为**：FlashInfer CUTLASS DSL 的 GDN delta rule kernel 只支持 fp16/bf16 输入，dtype 覆盖不全；用户显式请求 float32 时无优雅降级或自动精度转换。
- **影响对象**：可用性/进度（报错）
- **影响详情**：vLLM 以 dtype=torch.float32 部署 Qwen3.5-4B（含 GDN/linear attention）时，kernel 抛 RuntimeError: DSL kernel only supports fp16/bf16 inputs，服务启动失败。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/51305

### INF-1188 `vllm_sm120_flashinfer_sparse_mla_false_dense_prefill_claim` SM120 sparse MLA 误继承 dense prefill 能力声明

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-07 · 旧号 13.20

- **原因对象**：算子/内核 / vllm / SM120 sparse MLA 后端能力声明
- **原因行为**：FlashInferMLASparseSM120Impl 从 MLAAttentionImpl 继承 supports_dense_mha_prefill = True，但只实现 sparse forward_mqa 路径。
- **影响对象**：可用性/进度（报错）
- **影响详情**：SM120 + FlashInfer sparse MLA 下长 prompt 选 dense/masked MHA prefill 时，MLA router 访问未定义的 dense-path state（如 masked_mha_available），抛 AttributeError。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/51395

### INF-1194 `vllm_cute_fa4_flash_fwd_nameerror_first_call_sm120` vLLM CuTe FA4 前向在 sm_120 首次调用即 NameError

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-08-11 · 旧号 13.26

- **原因对象**：算子/内核 / vllm / cute/flash_fwd.py 前向类
- **原因行为**：FlashAttentionForwardSm80（及继承的 Sm120）引用未定义变量 mDynamicCausal，且缺少 self.is_split_kv 属性；该路径在 SM80 上不被选用，看起来从未被执行过。
- **影响对象**：可用性/进度（报错）
- **影响详情**：sm_120（RTX PRO 6000/RTX 5090）上任何路由到 flash_attn_varlen_func 的调用在 kernel 运行前即抛 NameError；单 GPU、batch size 1、25 行脚本即可复现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/51776

### INF-1197 `ttmetal_embedding_bw_rank4_gradient_shape_contract_violation` ttnn.embedding_bw 返回 rank-4 梯度违反形状契约

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-17 · 旧号 13.29

- **原因对象**：算子/内核 / tt-metal / ttnn.embedding_bw
- **原因行为**：ttnn.embedding_bw 对 rank-2 的 weight 输入返回 rank-4 的 gradient，违反梯度须与参数同形的 autograd 形状契约。
- **影响对象**：梯度；训练轨迹（静默错误）
- **影响详情**：训练/微调路径的 optimizer 拿到错误 rank 的梯度；视消费方而定，可能在广播规则下静默通过并污染更新方向，微调产物模型质量劣化且难以归因。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/53379

### INF-1210 `hf_transformers_metal_linear_wrong_outputs_rows_after_first_3d` Apple MPS 上 MetalLinear 3D 输入批维寻址错误

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-10-09 · 旧号 13.42

- **原因对象**：算子/内核 / transformers / MetalLinear 3D 批维寻址
- **原因行为**：Metal 线性内核对 3D 输入的批维寻址错误，只有第 0 个 batch 行的输出正确。
- **影响对象**：数值结果（静默错误）
- **影响详情**：transformers 在 Apple Metal（MPS）后端对 batch>1 的 3D 输入做线性前向时，第 0 行之后的每行输出静默错值，无 crash。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/huggingface/transformers/issues/49337

### INF-1216 `rdna_mamba_paged_attention_fallback_collapse` RDNA 上混合 Mamba 模型回退 Triton 慢速注意力路径

`故障` · `active` · `model_on` · `hw_specific` · `documented` · `NEW` · 发现 2026-07-29 · 旧号 14.6

- **原因对象**：算子/内核 / vllm / ROCm paged attention 路由
- **原因行为**：use_rocm_custom_paged_attention 要求 head_size==128 且 block_size==16，head_dim=256 的混合 Mamba 模型不满足，回退到 Triton 慢路径。
- **影响对象**：性能/资源（性能退化）
- **影响详情**：AMD RDNA（Radeon）上 Jamba 等混合 Mamba 模型长上下文 decode 从 518 context 的 82.5 ms/token 退化到 32K context 的 235.3 ms/token（2.85×），计算结果正确，只能靠延迟/吞吐监控发现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/50264

### INF-1229 `vllm_cascade_attn_missing_sink` vLLM cascade 注意力漏传 sink，softmax 分母错误

`故障` · `active` · `ambiguous` · `partial` · `documented` · `NEW` · 发现 待定 · 旧号 K1.1

- **原因对象**：算子/内核 / vllm / flash_attn cascade 路径
- **原因行为**：cascade attention 的共享前缀 pass 与后缀 pass 都没有传入学到的 sink 张量，经 merge_attn_states/LSE 合并后的 softmax 分母缺少 sink 项
- **影响对象**：数值结果；输出内容（质量退化）
- **影响详情**：开启 prefix caching 走 cascade 路径且模型带 attention sink（GPT-OSS）时出现：GPT-OSS-20B 的 MMLU-Pro 约 0.68→0.58，并出现退化的重复循环；非 cascade 路径正常
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/26203

### INF-1230 `vllm_cascade_attn_noncontig_lse_merge` vLLM cascade 注意力 LSE 非连续致前后缀合并错误

`故障` · `active` · `ambiguous` · `partial` · `verified` · `NEW` · 发现 待定 · 旧号 K1.2

- **原因对象**：算子/内核 / vllm / merge_attn_states LSE 合并
- **原因行为**：flash_attn_varlen_func 可能返回非连续的 suffix_lse，merge_attn_states 却假定连续布局，按错误的 stride 取元素，破坏前缀/后缀的加权组合
- **影响对象**：数值结果；输出内容（不可复现）
- **影响详情**：共享前缀负载上间歇出现乱码 token，准确率约 0.55；概率随 TP 度与并发度上升，batch size 为 1 时消失，--disable-cascade-attn 可恢复
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/22003；https://github.com/vllm-project/vllm/issues/22103

### INF-1274 `mla_absorb_vs_naive_materialization_mismatch` MLA absorb 路径折叠错误致与 naive 路径数值不一致

`故障` · `active` · `model_on` · `partial` · `documented` · `NEW` · 发现 待定 · 旧号 K6.9

- **原因对象**：算子/内核 / 通用 / MLA absorb 的 W^UK/W^UV 折叠
- **原因行为**：MLA absorb 路径把 W^UK 折进 query 投影、W^UV 折进输出，直接在潜空间做注意力；matmul 重排改变舍入，折叠 bug（RoPE/content 维度切分错、因子转置、投影过期）使其偏离 naive 解压路径。
- **影响对象**：数值结果；输出内容（静默错误）
- **影响详情**：原文按机制推断：naive prefill 与 absorb decode 之间出现无告警的数值不一致，潜缓存本身看似有效，长序列上逐渐放大为可见的输出偏差。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://arxiv.org/abs/2509.21081

### INF-1276 `vllm_prefix_cache_hit_recompute_divergence` vLLM ROCm 后端前缀缓存命中与重算结果不一致

`故障` · `active` · `ambiguous` · `config` · `documented` · `NEW` · 发现 待定 · 旧号 K7.1

- **原因对象**：算子/内核 / vllm / ROCm 注意力后端缓存 KV 归约
- **原因行为**：cache-HIT 路径加载缓存前缀 KV、只算新 token，部分后端对缓存 KV 的归约以与 cache-MISS 全序列重算不同的顺序/精度累加，两条路径数值不等价。
- **影响对象**：数值结果；输出内容（不可复现）
- **影响详情**：同一 prompt 在缓存命中与未命中两次运行下输出不同，偏离最早在位置 3 翻转 argmax 后级联放大，无报错；仅 AMD MI355X/MI325X 上的 TRITON_ATTN、ROCM_AITER_FA 受影响，CUDA 不受影响。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/33123

### INF-1304 `attn_kernel_nonpow2_seqlen_identical_output` MLA 解码 ASM 内核在 seqlen 非 2 的幂时各位置输出相同

`故障` · `active` · `ambiguous` · `partial` · `verified` · `NEW` · 发现 待定 · 旧号 K9.13

- **原因对象**：算子/内核 / rocm / mla_decode_stage1_asm_fwd
- **原因行为**：当 max_seqlen_q 不是 2 的幂时，ROCm/aiter 的 MLA 解码 ASM 内核对所有 query 位置产生逐位相同的注意力输出。
- **影响对象**：数值结果；输出内容（静默错误）
- **影响详情**：仅在 max_seqlen_q 非 2 的幂时触发；无崩溃、无 NaN、服务日志干净，生成内容退化，推测解码接受率变成常数。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/ROCm/aiter/issues/2720

### INF-1306 `flashinfer_trtllm_cudagraph_longctx_collapse` TRTLLM 注意力+CUDA graph 在超长上下文下输出错误

`故障` · `active` · `ambiguous` · `partial` · `verified` · `NEW` · 发现 待定 · 旧号 K9.15

- **原因对象**：算子/内核 / flashinfer / TRTLLM 注意力（CUDA graph 路径）
- **原因行为**：TRTLLM 注意力与 full CUDA graph 组合在 Blackwell 上处理 >128k 长上下文时产生错误输出；原文未给出具体根因。
- **影响对象**：数值结果；输出内容（质量退化）
- **影响详情**：需同时启用 cudagraph 与 TRTLLM 内核且上下文 >128k；短上下文正常，Qwen3-VL 的 GSM8K 准确率从 86% 崩到 1%，无崩溃提示。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/flashinfer-ai/flashinfer/issues/1968

### INF-1309 `mtp_dcp_cudagraph_silent_corruption` MTP+DCP+cudagraph 下注意力后端缺 varlen 解码致输出损坏

`故障` · `active` · `infra_off` · `partial` · `verified` · `NEW` · 发现 待定 · 旧号 K9.18

- **原因对象**：算子/内核 / vllm / 注意力后端 DCP varlen-decode
- **原因行为**：注意力后端在 DCP（decode context parallel）下缺少 varlen-decode 支持，却在 MTP 与 FULL_AND_PIECEWISE cudagraph 同时启用的组合中被使用。
- **影响对象**：输出内容（静默错误）
- **影响详情**：MTP、DCP、FULL_AND_PIECEWISE cudagraph 三者同时启用时触发：输出在 1-2 个字符后被截断或陷入重复循环，而健康检查、日志、指标均正常。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/45425

### INF-1323 `fa3_varlen_kvcache_cross_sequence_contamination` FA3 varlen 分页 KV 路径上批内序列互相干扰（Hopper）

`故障` · `active` · `ambiguous` · `partial` · `documented` · `NEW` · 发现 待定 · 旧号 K10.12

- **原因对象**：算子/内核 / flash-attention / FA3 varlen 分页 KV 路径
- **原因行为**：Hopper 上 FlashAttention-3 的 varlen 分页 KV 路径中，一个序列的 cache_seqlen 会扰动同批另一序列的计算，批内独立性被打破；怀疑源自共享 tile/LSE 归约，未确认。
- **影响对象**：数值结果；跨请求/跨样本数据（不可复现）
- **影响详情**：同一 prompt 在不同批次组成下得到数值不同乃至 token 不同的输出，单序列运行正常；只在 FA3 varlen 路径出现，FA2 不受影响。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/Dao-AILab/flash-attention/issues/1570

### INF-1326 `flashinfer_variable_blocksparse_output_layout_transpose` FlashInfer 块稀疏 wrapper 输出布局与文档不符致轴错配

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 待定 · 旧号 K10.15

- **原因对象**：算子/内核 / flashinfer / 块稀疏注意力 wrapper 输出布局
- **原因行为**：FlashInfer 块稀疏 wrapper 返回 [H, M, D] 而非文档所述 [M, H, D]，且忽视 kv_layout 设置；下游按文档形状 reshape，头轴与 token 轴被打乱后送进 o_proj。
- **影响对象**：数值结果；输出内容（静默错误）
- **影响详情**：使用块稀疏注意力路径时输出无意义，切回稠密路径则正常；上下游恰好 H == M 时不产生任何形状告警。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/flashinfer-ai/flashinfer/issues/2879

### INF-1327 `fp8_attn_pv_longctx_accumulation_collapse` FA3 FP8 注意力 P·V 长上下文累加求和淹没致精度崩溃

`故障` · `active` · `ambiguous` · `config` · `verified` · `NEW` · 发现 待定 · 旧号 K10.16

- **原因对象**：算子/内核 / vllm / FA3 FP8 内核 P·V 累加
- **原因行为**：FA3 的 FP8 内核沿序列长度累加 Softmax(P)·V，虽累加到 FP32 寄存器，但超过约 100k token 时小加数被大的运行和吞掉低位（求和淹没）。
- **影响对象**：数值结果；输出内容（质量退化）
- **影响详情**：短上下文 QA 完全正常；128k 长上下文检索（NIAH）准确率从 91% 崩到 13%，两级累加修复后回到约 89%。
- **改进措施**：检测 待定；修复 待定
- **来源**：引用：`gh`：vLLM flash-attention PR #104 + FP8-KV 博客（91%→13% NIAH → 89%）。

## 编译器

### INF-0009 `kernel_sync_semantic_silent_error` Triton 多 warpgroup 同步与内联汇编复制执行致静默错值

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-07-29 · 旧号 1.9

- **原因对象**：编译器 / triton / warpgroup_mma_wait 与内联汇编
- **原因行为**：warpgroup_mma_wait 被多 warpgroup 调用时不按多 warpgroup 同步，后续读到未完成的 MMA 结果；元素数少于线程数时 tl.inline_asm_elementwise 对每个复制元素各执行一次。
- **影响对象**：数值结果（静默错误）
- **影响详情**：Hopper/Blackwell 上手写多 warpgroup Triton 内核（含 vLLM/SGLang 自定义 attention/MoE kernel）可能静默产出错误值；带副作用的内联汇编（如 red.global.add）在不满 warp 的 tail 场景被重复累加。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/triton-lang/triton/issues/11047；https://github.com/triton-lang/triton/issues/11065

### INF-0012 `triton_fp8_e5m2_transposed_store_codegen_corruption` Triton fp8 转置 store 误编译写出 Inf/NaN 垃圾

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-07-30 · 旧号 1.12

- **原因对象**：编译器 / triton / fp8 转置 store lowering
- **原因行为**：值由 PropagateNan.ALL 的 tl.maximum/minimum 产生、lowering 为 min.NaN/max.NaN PTX 时，转置布局的 fp8 store 被错误 codegen，指向寄存器分配或向量化。
- **影响对象**：数值结果（数值异常）
- **影响详情**：同一值的 contiguous store 正确，transposed store 却写出解码为 Inf/NaN 的垃圾字节，损坏模式随 XBLOCK/num_warps 变化；用转置 fp8 e5m2 store 的自定义 kernel（MoE 专家输出、量化 attention）静默损坏。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/triton-lang/triton/issues/11111；https://gist.github.com/vkuzo/c7410babadd904c93beb3fd3638ffc75

### INF-0016 `triton_dual_dot_single_accumulator_sm103_wrong` Triton 在 sm_103 上双 tl.dot 共享累加器结果错误

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-07-31 · 旧号 1.16

- **原因对象**：编译器 / triton / 双 tl.dot 共享累加器 MMA 调度
- **原因行为**：K-loop 内两条 tl.dot 串联同一累加器时，编译器/MMA 调度在 sm_103 的 tcgen05 MMA 路径（BLOCK_M≥64）上算错。
- **影响对象**：数值结果；输出分布（静默错误）
- **影响详情**：B300（sm_103）上该路径的 GEMM 结果确定性错误，最大绝对误差约 50–70（参考值量级约 25），同一输入反复得到同一错值；无 crash、无 NaN，下游 logits 被静默污染。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/triton-lang/triton/issues/10821

### INF-0017 `triton_blackwell_rmw_accumulator_mma_miscompile` Blackwell 上 Triton 误编译读改写 TMEM 累加器

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-01 · 旧号 1.17

- **原因对象**：编译器 / triton / AssignLatencies/pipeliner
- **原因行为**：loop-carried TMEM 累加器每轮被读改写且又作同一循环内另一 MMA 的操作数时，AssignLatencies 给该 MMA 赋 latency=1，pipeliner 将累加器多缓冲化，消费侧 MMA 读到去同步的副本。
- **影响对象**：KV状态；数值结果（静默错误）
- **影响详情**：sm_100 上 num_warps≥4（async tcgen05）且 num_stages≥2 时触发，chunked simple-GLA/Mamba-2（SSD）前向内核的递归 [K,V] 状态静默出错、污染下游 logits；num_warps≤2 或 num_stages=1 不受影响。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/triton-lang/triton/pull/10993

### INF-0020 `triton_blocksize_dependent_ima_codegen` Triton 特定 BLOCK_SIZE 下 codegen 产生非法内存访问

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-08-02 · 旧号 1.20

- **原因对象**：编译器 / triton / tile 形状相关 codegen/调度
- **原因行为**：特定 tile 形状下 codegen/调度路径产生地址越界，而非算法定义错误；对源码改动高度敏感。
- **影响对象**：可用性/进度（报错）
- **影响详情**：BLOCK_SIZE 为 512 或 1024 时 kernel launch 即报 CUDA illegal memory access，32/64/128/256/2048 时完全正确；移除一个注入块、死跳转块里的 return 或把 tl.trans 换成 tl.reshape，崩溃即消失。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/triton-lang/triton/issues/11148

### INF-0021 `triton_autotuner_invalid_topk_silent_config_selection` Triton autotuner 不校验 top_k 致候选集被静默改变

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-08-02 · 旧号 1.21

- **原因对象**：编译器 / triton / autotuner top_k 剪枝
- **原因行为**：autotuner 不校验 prune_configs_by["top_k"] 的值域：负整数成为负 slice 边界，0.0 或负 float 经 max(1, …) 处理，无效值不报错直接参与剪枝。
- **影响对象**：性能/资源（静默错误）
- **影响详情**：负整数 top_k 静默移除一个候选，0.0 或负 float 静默只保留一个 config，最终选出的 kernel 配置与开发者意图不符且无错误指示；top_k=0 在 min() 处失败，NaN/无穷抛出误导性 TypeError。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/triton-lang/triton/issues/11072

### INF-0022 `ttmetal_quasar_lut2_wrong_lreg_class` sfpi 在 Quasar 上沿用 WH/BH 寄存器映射致 lut2 读错表源

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-02 · 旧号 1.22

- **原因对象**：编译器 / tt-metal / sfpi::lut2/lut2_sign
- **原因行为**：Quasar 上 SFPLUTFP32 的表源已迁到 SFPU 常量 LReg（slopes→LReg9-11、intercepts→LReg12-14），sfpi 仍按 WH/BH 映射把 6 个表字写入通用 LReg。
- **影响对象**：数值结果（静默错误）
- **影响详情**：Quasar 上 lut2()/lut2_sign() 静默读取错误的表源寄存器，得到错误的分段查表值，污染任何依赖 LUT 的数值计算；无 crash，不一定出现 NaN。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/51346

### INF-0039 `triton_compiled_fma_silent_wrong_vs_interpreter` Triton 编译 FMA kernel 算错而 interpreter 正确

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-08-06 · 旧号 1.39

- **原因对象**：编译器 / triton / Triton codegen 路径
- **原因行为**：Triton 编译路径对 FMA kernel y = A @ x 产生错误结果，语义等价的重写 kernel 正确，指向 codegen 路径特定缺陷而非算法错误；具体根因待定。
- **影响对象**：数值结果（静默错误）
- **影响详情**：ones(4,4) @ ones(4) 编译后返回 [2,4,4,4]（期望 [4,4,4,4]），interpreter 结果正确，3 次运行一致；无 crash、无 NaN，只有 interpreter 与编译结果交叉对照才能发现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/triton-lang/triton/issues/11153

### INF-0047 `triton_optimize_thread_locality_miscompile` OptimizeThreadLocality 误重排 reduction 外元素

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-08-07 · 旧号 1.47

- **原因对象**：编译器 / triton / OptimizeThreadLocality
- **原因行为**：Triton 的 OptimizeThreadLocality pass 引入带 can_reorder 的 reshape，可能重排 reduction 维度之外的元素，导致 miscompile。
- **影响对象**：数值结果（静默错误）
- **影响详情**：含 reduction + reshape 组合的 Triton kernel 经该 pass 编译后输出数值静默错误；原文未描述 crash，也未给出具体复现数值。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/triton-lang/triton/pull/11186

### INF-0056 `triton_amd_bf16_multiply_truncation_not_rne` Triton AMD 后端 bf16 乘法截断而非就近舍入

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-08-10 · 旧号 1.56

- **原因对象**：编译器 / triton / AMD 后端 bf16 fmul 降低
- **原因行为**：AMD 后端把标量 fmul bfloat 降低为第二 lane 置零的 fdot2.bf16.bf16（发射 v_dot2_bf16_bf16），该指令在 f32 累加后写 bf16 时不做舍入，低位尾数直接丢弃。
- **影响对象**：数值结果（静默错误）
- **影响详情**：AMD GPU（gfx1200）上含 bf16 乘法的 Triton kernel 每次乘法系统性截断最多 1 ULP，累积后与 torch/CUDA 分叉；bf16 频率的 RoPE kernel 约 41% 输出超出 rtol=1e-3（eager 约 14%）；bf16 加法不受影响。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/triton-lang/triton/issues/11217

### INF-0081 `triton_layout_conflict_arbitrary_fallback_lane_degenerate` Triton 布局冲突任意回退选中 lane 退化布局致 miscompile

`故障` · `active` · `infra_off` · `yes` · `documented` · `NEW` · 发现 2026-08-16 · 旧号 1.82

- **原因对象**：编译器 / triton / 布局传播 resolveConflicts
- **原因行为**：LayoutPropagation::resolveConflicts 在 blocked 与 MMA 编码均不适用时回退取插入序第一个候选，可能选中 lane base 全零的 layout，使 warp 每个 lane 持有同一元素。
- **影响对象**：数值结果（静默错误）
- **影响详情**：命中该回退路径的 kernel（layout 冲突且 blocked/MMA 均不适用，常见于 fused/reshape 密集模式）输出静默错误数值，同一 IR 确定性复现；无 crash。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/triton-lang/triton/pull/11309

### INF-0082 `triton_rdna_atomic_memory_order_barrier_address_space` Triton RDNA 原子操作 barrier 地址空间错选

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-16 · 旧号 1.83

- **原因对象**：编译器 / triton / AMD 后端原子 barrier 地址空间
- **原因行为**：RDNA 上同一 workgroup 的 wave 可在不同 CU 运行、各有独立 L0 cache，带内存序的 scalar tl.atomic_* 需跨 wave 可见的 barrier，但地址空间选择未区分该情形。
- **影响对象**：数值结果（静默错误）
- **影响详情**：RDNA + 带内存序的 tl.atomic_* 时跨 wave 可见性被破坏，data race 表现为静默错误结果或偶发错误；此前 #10906 以全面加 barrier 掩盖，代价是 LDS-only 场景性能回退。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/triton-lang/triton/pull/11289

### INF-0087 `triton_cluster_barrier_memdesc_view_alias_miss` Triton 集群 barrier 插入漏查 view 别名致 race

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-17 · 旧号 1.88

- **原因对象**：编译器 / triton / ClusterBarrierInsertion
- **原因行为**：效应收集循环用 getBufferIds 只读 valueBuffer，看不到记录在 aliasBuffer 的 memdesc view（同文件他处用含别名版），经 view 写入的 multicast TMA 漏插 barrier。
- **影响对象**：数值结果（不可复现）
- **影响详情**：NVIDIA 后端 cluster + multicast TMA 经 memdesc view 写入时，共享内存字节被重用前缺少同步，产生跨 CTA race，表现为依赖时序的间歇性错误结果；无 crash。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/triton-lang/triton/issues/11328

### INF-0088 `triton_membar_call_entry_state_unchecked_race` Triton membar 对 tt.call 漏查入口交叠致共享内存 race

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-08-17 · 旧号 1.89

- **原因对象**：编译器 / triton / Membar 分析 tt.call 入口
- **原因行为**：Membar 分析对 tt.call 只取 callee 的 exit 状态，不核对 caller pending 共享访问与 callee 首访；getScratchBufferId 又排除 CallOp，交叠时漏插 barrier。
- **影响对象**：数值结果（静默错误）
- **影响详情**：带共享内存读写的 tt.call，且 caller pending 访问与 callee 首访交叠时，调用边界前后发生共享内存 race，表现为间歇性静默错误结果；无 crash。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/triton-lang/triton/issues/11326

### INF-0094 `triton_amd_gfx1151_runtime_trip_count_loop_miscompile` Triton AMD gfx1151 运行时循环 miscompile

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-20 · 旧号 1.95

- **原因对象**：编译器 / triton / AMD 后端运行时 trip-count 循环
- **原因行为**：AMD 后端对唯一动态控制流为运行时 trip-count scf.for 的内核生成错误代码；TTGIR 语义正确，改写为 static_range 或 grid-per-token 变体后逐位正确，缺陷在该循环的代码生成/调度路径。
- **影响对象**：数值结果（静默错误）
- **影响详情**：Triton 3.7.0 + ROCm 7.14 gfx1151 上：num_warps=1 确定性错值（约 2/3 packed 输出错），2/4 先错后 illegal-address fault，8 首次 launch 即 fault；NVIDIA 后端不复现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/triton-lang/triton/issues/11378

### INF-0110 `triton_softmax_keepdims_false_reduction_axis_misbroadcast` tl.softmax keep_dims=False 时归约维广播错位

`故障` · `active` · `infra_off` · `partial` · `verified` · `NEW` · 发现 2026-08-27 · 旧号 1.111

- **原因对象**：编译器 / triton / tl.softmax keep_dims 分支
- **原因行为**：tl.softmax(dim=1, keep_dims=False) 丢弃归约维后，归约结果与原 shape 的广播对齐错位，行归一化被广播到错误的轴；编译与解释路径同错，缺陷在语言前端。
- **影响对象**：数值结果（静默错误）
- **影响详情**：keep_dims=False 时 softmax 输出不再逐行归一：2×2 fp32 复现行和为 [0.3979, 5.4493] 而非 [1, 1]；TRITON_INTERPRET=1 同样错误，keep_dims=True 正确；组合该原语的 kernel 数值全错。已修复关闭。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/triton-lang/triton/issues/11406

### INF-0126 `triton_fuse_nested_loops_nondominating_assume_miscompile` Triton 嵌套循环融合误用非支配 assume 致错误编译

`故障` · `active` · `infra_off` · `partial` · `verified` · `NEW` · 发现 2026-08-30 · 旧号 1.127

- **原因对象**：编译器 / triton / fuse-nested-loops pass
- **原因行为**：tritongpu-fuse-nested-loops 的 matchPositiveTripCount 在函数内任意位置匹配 llvm.assume，不检查 dominance/可达性，据此认定内层循环至少执行一次而跳过零迭代版本分裂。
- **影响对象**：数值结果（静默错误）
- **影响详情**：合法 kernel 合法调用（m=4、n=0）下 flatten=True 输出 [4,8,12,16]（应全 0）并越界写入 256 元素，flatten=False 全对；RTX PRO 6000 Blackwell 与 L4、Triton 3.6.0/3.7.1 均复现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/triton-lang/triton/issues/11519

### INF-0167 `triton_ptxas_129_s16x2_second_immediate_dropped_miscompile` ptxas 12.9.86 丢弃 s16x2 立即数致 Triton 错值

`故障` · `active` · `infra_off` · `config` · `verified` · `NEW` · 发现 2026-09-04 · 旧号 1.168

- **原因对象**：编译器 / cuda / ptxas 12.9.86 s16x2 指令选择
- **原因行为**：ptxas 12.9.86 指令选择时把第二个 packed add.s16x2 的立即数硬编码为 0x0（VIADD.16x2 R4, R4, 0x0），使该加法变为 no-op。
- **影响对象**：数值结果（静默错误）
- **影响详情**：Triton 编译的整数运算被 LLVM 窄化为 packed s16x2 时触发，H100 上 256 个输出中 16 个错 1；TRITON_INTERPRET=1 与 ptxas 13.3.33 下同一源码结果正确，正确性随 ptxas 版本翻转。无 crash、无 NaN。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/triton-lang/triton/issues/11581

### INF-0168 `triton_fuse_nested_loops_zero_trip_bound_load_hoist_ima` FuseNestedLoops 提升内层 bound load 致非法访存

`故障` · `active` · `infra_off` · `partial` · `verified` · `var` · 发现 2026-09-05 · 旧号 1.169

- **原因对象**：编译器 / triton / FuseNestedLoops flatten
- **原因行为**：flatten=True 时把内层循环 bound 的标量 tt.load 提升到外层循环之上，pass 后在函数作用域无条件执行该 load，与源控制流脱钩。
- **影响对象**：可用性/进度（报错）
- **影响详情**：内层 bound 来自无 mask 的指针标量读取且外层零次迭代（如 m=0）时，源程序从不访问的指针被解引用，launch 期 CUDA illegal memory access（exit=3）；同 kernel flatten=False 正常。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/triton-lang/triton/issues/11601
- **关联**：var:INF-0126

### INF-0172 `triton_fuse_nested_loops_scf_while_boundary_pass_crash` FuseNestedLoops 遇 scf.while 时 pass 崩溃

`故障` · `active` · `infra_off` · `partial` · `documented` · `var` · 发现 2026-09-06 · 旧号 1.173

- **原因对象**：编译器 / triton / FuseNestedLoops 循环嵌套切分
- **原因行为**：constructLoopNest 把 scf.while 当作分析边界并在该处重启，但 while 仍是外层循环体操作，fuseOneLevel 将其切入 prologue/epilogue 时切分点出错。
- **影响对象**：可用性/进度（报错）
- **影响详情**：flatten=True 外层循环体同时含 scf.while 与携带值的 scf.for 时，MLIR 断言失败致 PassManager::run failed；更大 kernel 上产出与 region successor 不匹配的 scf.if 被 verifier 拦截，阻断构建。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/triton-lang/triton/issues/11612
- **关联**：var:INF-0126

### INF-0173 `triton_interpreter_fma_fp32_double_rounding_fp64_intermediate` Triton interpreter fp32 FMA 双重舍入致结果分歧

`故障` · `active` · `infra_off` · `partial` · `verified` · `NEW` · 发现 2026-09-06 · 旧号 1.174

- **原因对象**：编译器 / triton / 解释器 create_fma
- **原因行为**：interpreter 的 create_fma 对 fp32 用 np.add(product, z, dtype=float64) 求和再 astype(float32) 两步舍入；fp64 精度足够的前提只对 fp16 成立。
- **影响对象**：数值结果；测试判定（静默错误）
- **影响详情**：TRITON_INTERPRET=1 下 tl.math.fma 精确值落在 fp32 半途点时与 compiled 的 fma.rn.f32 差 1 ULP（4097²+2⁻³⁰ 得 16785408.0，应为 16785410.0）；以 interpreter 作 golden 会把错值当基准。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/triton-lang/triton/issues/11610

### INF-0174 `triton_interpreter_sum_integer_bitpattern_addition_loses_nan` Triton interpreter 对 bf16/fp8 求和吞掉 NaN

`故障` · `active` · `infra_off` · `partial` · `verified` · `var` · 发现 2026-09-06 · 旧号 1.175

- **原因对象**：编译器 / triton / 解释器 ReduceOps.sum
- **原因行为**：interpreter 把 bf16/float8 张量存为原始无符号整数位模式，ReduceOps.sum() 直接对该整数数据 np.sum 后再重解释为 bf16，归约在错误的数据表示上执行。
- **影响对象**：数值结果；测试判定（静默错误）
- **影响详情**：bf16/float8 输入含 NaN 时，interpreter 的 tl.sum 静默返回有限值而非 NaN，compiled 路径则正确传播 NaN；CPU 即可复现，以 interpreter 作黄金基准会被误导。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/triton-lang/triton/issues/11604
- **关联**：var:ttmetal_argmax_comparator_nan_semantics_dtype_split

### INF-0180 `vllm_cpu_moe_torch_compile_nan_logits_quantization_independent` vLLM CPU MoE 经 inductor 编译产出 NaN logits

`故障` · `active` · `infra_off` · `partial` · `documented` · `var` · 发现 2026-09-07 · 旧号 1.181

- **原因对象**：编译器 / vllm / CPU MoE 内核 inductor 编译路径
- **原因行为**：torch.compile/inductor 为 Qwen3_5Moe 架构生成的 CPU MoE 内核路径（或其与 Inductor 分解的组合）计算出错，与权重量化方法无关。
- **影响对象**：输出分布；输出内容（数值异常）
- **影响详情**：从首个生成 token 起全词表 logits 为 NaN，输出为重排的垃圾文本，无 crash；bf16 与 GPTQ-Int4 checkpoint 均复现，--enforce-eager 即愈。vLLM 0.28.0 CPU 镜像、AMX CPU、TP=1 即触发。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/55560
- **关联**：var:INF-0012

### INF-0191 `triton_fuse_nested_loops_empty_dynamic_inner_loop_miscompile` Triton FuseNestedLoops 对空动态内层循环误编译

`故障` · `active` · `infra_off` · `partial` · `verified` · `var` · 发现 2026-09-09 · 旧号 1.192

- **原因对象**：编译器 / triton / FuseNestedLoops 循环扁平化
- **原因行为**：flatten 变换未对动态 trip count 下上界低于下界的空内层区间做守卫，扁平化后的循环在每轮外层仍执行一次内层体。
- **影响对象**：数值结果（静默错误）
- **影响详情**：tl.range(5, ub) 且 flatten=True、ub=3 时 flatten 版返回 4，普通嵌套版正确返回 0；合法 kernel 合法调用下静默多算，依赖空内层做条件跳过的 kernel 受影响（sm89、Triton 3.8.0 复现）。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/triton-lang/triton/issues/11656
- **关联**：var:INF-0126

### INF-0198 `triton_gfx942_block_pingpong_sbarrier_before_ds_read_waitcnt_stale_next_kstep_a` Triton gfx942 pingpong 调度竞争致 matmul 错值

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `var` · 发现 2026-09-10 · 旧号 1.199

- **原因对象**：编译器 / triton / AMD block-pingpong 调度
- **原因行为**：调度器在本组 ds_read 的 waitcnt 之前发射 s_barrier，另一 warp group 下一 stage 的 ds_write 写入同一单级 LDS 缓冲，与本组未完成的读竞争。
- **影响对象**：数值结果（静默错误）
- **影响详情**：MI300X 上特定配置 fp32 matmul 200/200 次在 4×32 patch 读到下一 k 步的 A 操作数；关 block-pingpong 后 0/200，gfx950 不触发；inductor max-autotune 中表现为 flaky 失败。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/triton-lang/triton/issues/11696
- **关联**：var:ttmetal_sdpa_decode_mask_seed_scalar_store_no_drain_race

### INF-0218 `ttmetal_sfpi_vfloat_eq_lowered_to_bitwise_identity_nan_eq_semantics_version_flip` sfpi ≥7.75.0 把 vFloat 相等比较下推为位模式等同

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-09-15 · 旧号 1.219

- **原因对象**：编译器 / sfpi / vFloat ==/!= lowering
- **原因行为**：sfpi ≥7.75.0 把 vFloat ==、!= 及与非零常量比较编译为整数减法加 SFPSETCC LREG_EQ0，谓词变成位模式等同而非 IEEE 754 相等；7.70–7.74 的旧下推同样不符合 IEEE。
- **影响对象**：数值结果（静默错误）
- **影响详情**：NaN==NaN 为 true、isnan 惯用法 x!=x 为 false、+0.0==−0.0 为 false，编译无警告；依赖 ==/!= 做 NaN/Inf 守卫的 SFPU 内核升级工具链后守卫方向翻转，同源码跨版本行为分歧；BH/WH 硅片实测，代码生成级可复现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/56659

### INF-0228 `triton_optimize_thread_locality_accumulator_extra_use_miscompile` Triton OptimizeThreadLocality 漏检累加器额外使用

`故障` · `active` · `infra_off` · `partial` · `verified` · `NEW` · 发现 2026-09-17 · 旧号 1.229

- **原因对象**：编译器 / triton / OptimizeThreadLocality
- **原因行为**：该 pass 把 acc = acc + tl.sum(...) 重写为线程本地累积时，只检查归约及其 update 的用途数，未检查循环携带累加器的其他使用者；createYield 把原累加器改为恒等携带初值。
- **影响对象**：数值结果（静默错误）
- **影响详情**：循环内对累加器的额外读取（如 tl.store 快照）持续读到初值 0，而最终结果正确，无 crash 无错误信号；sm89 复现 OUT 全 0、FIN 正确，仅当消费者依赖循环内中间值时显形，端到端单断言测试发现不了。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/triton-lang/triton/issues/11836

### INF-0252 `ttmetal_v_if_non_last_block_sfputt_comp_state_negation_early` SFPU v_if 非末 block 中 v_and 的状态取反过早

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-09-25 · 旧号 1.253

- **原因对象**：编译器 / tt-metal / SFPU v_if/v_and lowering
- **原因行为**：v_and 位于非最后一个 v_if block 时，编译器在 block 末尾而非开头插入 SFPCOMPC（状态取反），后续装载/编码指令执行时谓词状态已被取反。
- **影响对象**：数值结果（静默错误）
- **影响详情**：以 v_and 组合谓词编写的多 block 条件 SFPU 内核静默产出反逻辑结果（错误掩码/分支），反汇编可自证，WH/BH 均适用；既有 LLK 内核受影响面未普查。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/57889

### INF-0274 `triton_argmax_reshape_ravel_flip_allow_reorder_permutes_value_not_index` Triton argmax 在 reshape 后返回错误索引

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `var` · 发现 2026-10-09 · 旧号 1.275

- **原因对象**：编译器 / triton / reshape-reduce 合并 pass
- **原因行为**：value 来自 reshape/ravel/flip 时，CombineReshapeReducePatterns 将 reshape 标记 allow_reorder，value 可被重排而 index（tl.arange）不重排。
- **影响对象**：数值结果（静默错误）
- **影响详情**：tl.argmax/tl.argmin（return_indices=True）作用于 reshape/ravel/flip 结果时，返回索引静默指向错误 lane；依赖 argmax 做 gating/检索的 kernel 选错元素。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/triton-lang/triton/issues/12099
- **关联**：var:INF-0150

### INF-0275 `triton_bf16_scalar_constants_cut_to_6_decimals_by_to_string` Triton bf16 标量常量经 to_string 截断为 6 位小数

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-10-09 · 旧号 1.276

- **原因对象**：编译器 / triton / get_bf16 常量构造
- **原因行为**：get_bf16 以 std::to_string（固定 6 位小数）打印 float 再解析文本构造 bf16 常量，超过 6 位小数的部分被截断，而非位级舍入到 bf16。
- **影响对象**：数值结果（静默错误）
- **影响详情**：tl.full([16],1e-8,tl.bfloat16) 与 x_bf16*1e-8 得 0，1.5e-6 得 2e-6；小幅度 bf16 常量（衰减系数、epsilon）静默归零或偏离，下游数值静默漂移。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/triton-lang/triton/issues/12102

### INF-0276 `triton_loop_carry_detection_skips_prev_cur_same_initial_value` Triton _find_carries 漏掉初值相同的循环 carry 变量

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-10-09 · 旧号 1.277

- **原因对象**：编译器 / triton / _find_carries 迭代参数判定
- **原因行为**：CodeGenerator._find_carries 以一次 dry run 后比较 IR 值决定循环 iter_args，而非分析值能否跨迭代变化；prev=cur 起始同值时 prev 不进 iter_args。
- **影响对象**：数值结果（静默错误）
- **影响详情**：for/while 循环中以同值初始化的 carry 变量静默丢失，循环体沿用 pre-loop 初值；依赖状态递推的 kernel（滑动统计、递归滤波）静默用初值计算。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/triton-lang/triton/issues/12104

### INF-0277 `triton_fuse_nested_loops_flatten_advances_outer_carry_on_inner_iterations` Triton FuseNestedLoops 扁平化错推外层 carry

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `var` · 发现 2026-10-09 · 旧号 1.278

- **原因对象**：编译器 / triton / FuseNestedLoops pass
- **原因行为**：flatten=True 扁平化时未区分内/外层边界对 carried 变量的更新时机，外层 loop-carried 索引在内层迭代上也被推进，按总迭代步进。
- **影响对象**：数值结果（静默错误）
- **影响详情**：tl.range(flatten=True) 嵌套循环携带外层索引时，循环中的地址/索引静默错位，无报错。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/triton-lang/triton/issues/12192
- **关联**：var:INF-0126

### INF-0330 `vllm_batch_invariant_m_bucket_dynamo_constant_folding_collapse` vLLM batch-invariant 按 M 查表被 Dynamo 常量折叠

`故障` · `active` · `ambiguous` · `config` · `verified` · `NEW` · 发现 2026-08-29 · 旧号 4.14

- **原因对象**：编译器 / vllm / batch-invariant 调参查表
- **原因行为**：按 (arch,N,K,M bucket) 选 Triton launch config 的查表在 Dynamo trace 期求值并烘焙为 constexpr，M 仍是运行时参数，每个 (N,K) 调用点只有一个静态 tile
- **影响对象**：性能/资源（性能退化）
- **影响详情**：VLLM_BATCH_INVARIANT=1 + torch.compile 下 decode M=8 与 prefill M=2048 无法各取表项，tile 配置系统性错配致性能退化、调参意图失效；无任何告警，tracing 也不报 graph break
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/54243

### INF-0334 `sglang_h3_ref2va_profiling_graph_switch_nondeterminism` SGLang Ref2VA 音频编码器 JIT 图切换致跨调用非确定

`故障` · `active` · `ambiguous` · `other_stack` · `documented` · `var` · 发现 2026-09-06 · 旧号 4.18

- **原因对象**：编译器 / sglang / 音频编码器 scripted snake 激活
- **原因行为**：MiniMax-H3 音频编码器的 snake 激活经 TorchScript 脚本化，profiling 后执行图由初始图切换为融合 TensorExprGroup 图，浮点求值顺序改变；默认 warmup 用图像参考，未预热该路径。
- **影响对象**：输出内容；数值结果（不可复现）
- **影响详情**：同一服务器同种子同请求首次与后续输出不同（视频 SSIM 0.73–0.83、音频 PCM 最大差 0.185、MP4 字节不同）；参考音频 conditioning 的微小扰动经联合音视频去噪逐步放大，在参考音频请求首次执行时出现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/pull/38225
- **关联**：var:INF-0328

### INF-0335 `vllm_rope_kvcache_fusion_mla_only_zero_pattern_side_effects_paid` vLLM rope+KV 融合 pass 对 MLA 模型零匹配仍付副作用

`故障` · `active` · `infra_off` · `config` · `documented` · `var` · 发现 2026-09-07 · 旧号 4.19

- **原因对象**：编译器 / vllm / rope_kvcache 融合 pass 层收集
- **原因行为**：融合 pass 用 Attention 类收集目标层，而 MLAAttention 是兄弟类，MLA-only 模型收集恒空；但扩编译端点、强插 +rotary_embedding、pass 无条件执行三项副作用未被同一条件守卫。
- **影响对象**：性能/资源；数值结果（性能退化）
- **影响详情**：MLA-only 模型（如 GLM-5.3，ROCm gfx950 TP4）开两个 fusion flag 时零 pattern 注册，编译端点由 [5461,8192] 变为 [256,256,5461,8192]，吞吐下降，rope 改走 custom op，日志无提示。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/55720
- **关联**：var:INF-0600

### INF-0342 `vllm_glm5x_replicated_indexer_key_norm_per_rank_autotune_nondeterministic_tp_greedy_forks` vLLM GLM-5.x 逐 rank autotune 致 TP 输出分叉

`故障` · `active` · `infra_off` · `config` · `documented` · `var` · 发现 2026-09-24 · 旧号 4.26

- **原因对象**：编译器 / vllm / indexer k_norm autotune
- **原因行为**：各 rank 复制计算 indexer key 路径须 bit-identical，但 _fused_indexer_k_norm 的 Inductor autotune 逐 rank 自选 XBLOCK，候选含归约顺序不同的不等价配置。
- **影响对象**：数值结果；输出内容（不可复现）
- **影响详情**：2×GB10 TP2 16 次启动出现三种状态，部分 rank 选 XBLOCK=1 致 indexer top-512 候选集分歧，temperature 0 输出随启动分叉；单次启动内自一致，重放验证失效。TORCHINDUCTOR_DETERMINISTIC=1 因 torch bug 无效。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/58636
- **关联**：var:INF-0328

### INF-0343 `vllm_v2_spec_prefill_rmsnorm_autotune_config_flips_qwen3_greedy_first_token` vLLM spec 模式 RMSNorm 启动配置不同致首 token 翻转

`故障` · `active` · `infra_off` · `config` · `documented` · `var` · 发现 2026-09-28 · 旧号 4.27

- **原因对象**：编译器 / vllm / 三个 fused RMSNorm 启动配置
- **原因行为**：三个 fused RMSNorm 变体的 torch.compile/autotune 工件在同源下按 spec 与 AR 的 dispatch/padding 形态选了不同 reduction-block 启动配置，两者数值不等价。
- **影响对象**：数值结果；输出内容（不可复现）
- **影响详情**：Qwen3-4B BF16 greedy 开 DFlash2/DSpark 推测解码后首 token 即与 AR 不同（334→95456），分歧发生在初始 target prefill、任何 draft 决策之前；对齐 launch 配置后 prefill logits bitwise 复原。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/58973
- **关联**：var:INF-0342

### INF-0458 `sglang_dsa_topk_cicc131_drops_nonprimary_ranks` CUDA 13.1+ cicc 误编译 DSA top-k 致输出置零

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-08-09 · 旧号 6.65

- **原因对象**：编译器 / cuda / cicc 编译 TopKCluster
- **原因行为**：phase-3 把 DSMEM 与 shared::cta 两个目的合并为单指针后，cicc 13.1+ 在 sm_90a 上错误 lowering 该分支，丢弃非 primary rank 输出，tmp_out 留零。
- **影响对象**：数值结果；可用性/进度（静默错误）
- **影响详情**：DeepSeek-V4 DSA decode 用 CUDA 13.1–13.3 构建时触发（同 H200 上 12.9/13.0 正确）；phase 3.5 把零拷到正确地址，512 槽中 383 槽错，下游解引用错误 KV 索引而非法访存崩溃（#33835），或静默输出错误。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/pull/34167

### INF-0491 `vllm_inductor_pad_alias_view_frozen_capture_zeros_cuda_graph_moe` Inductor alias 化 pad 致 graph 模式 MoE 输入全零

`故障` · `active` · `infra_off` · `yes` · `documented` · `NEW` · 发现 2026-08-15 · 旧号 6.98

- **原因对象**：编译器 / vllm / moe_forward 调用前的 F.pad
- **原因行为**：MoE 激活的 F.pad 在 torch.compile trace 区执行后喂给 opaque custom op，Inductor 将其优化为 alias view，内核读到 graph replay 不刷新的 buffer。
- **影响对象**：隐状态；输出内容（静默错误）
- **影响详情**：所有 CUDA graph 模式（PIECEWISE、FULL_ALL 等）下 hidden_states 读到冻结的捕获期全零，输出确定性地流畅但语义损坏；eager 正常，易误判为模型或量化问题。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/52383

### INF-0535 `sglang_dsa_triton_prefill_autotune_nan_config_rocm` Triton 3.7.0 gfx950 编出 NaN 内核致 DSA 损坏

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `var` · 发现 2026-08-22 · 旧号 6.142

- **原因对象**：编译器 / triton / Triton 3.7.0 gfx950 代码生成
- **原因行为**：Triton 3.7.0 在 gfx950 上把 DSA sparse-MLA prefill 的 BLOCK_N=128、num_warps=1、num_stages=2 config 编成 NaN/inf 内核，SGLang 未排除。
- **影响对象**：数值结果；输出内容（数值异常）
- **影响详情**：SGLANG_DSA_TRITON_PREFILL=1 时 autotune 可在某 shape 首次 prefill 选中坏 config 并缓存，prefill 输出 NaN/inf，decode 全程 garbage tokens，重启前同 shape 持续出错；CUDA 不受影响。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/pull/35274
- **关联**：var:INF-0414

### INF-0569 `flashmla_sparse_decode_tmem_alloc_spin_wedge_ptxas` FlashMLA TMEM 分配无界自旋致 rank 静默挂起

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-31 · 旧号 6.176

- **原因对象**：编译器 / flashmla / FlashMLA tcgen05 TMEM 分配
- **原因行为**：FlashMLA sparse-decode 的 tcgen05.alloc.cta_group::2 被 ptxas 降低为等待对侧 CTA 的 TMEM 分配自旋循环，无超时；该循环由 ptxas 生成，源级无法加界。
- **影响对象**：可用性/进度（挂起）
- **影响详情**：vLLM Blackwell tcgen05 路径上对侧 CTA 因任何原因停滞时，本 rank 以 100% SM 利用率无界空转、看似在计算；约 300s 后才在其他 rank 不相关的 collective 超时中报错，报错位置与肇事 rank 无关。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/54638

### INF-0570 `xpu_inductor_decode_pieces_spec_decode_livelock_32k` XPU inductor decode 分段与 MTP 在长上下文活锁

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-09-01 · 旧号 6.177

- **原因对象**：编译器 / vllm / inductor 编译的 decode 分段
- **原因行为**：inductor piecewise（FULL_DECODE_ONLY）编译的 decode 分段对缓冲复用/同步的假定，与 MTP 多 draft token 的形状冲突，首个 decode chunk 即互锁。
- **影响对象**：可用性/进度（挂起）
- **影响详情**：Intel XPU（2×Arc Pro B70）+ MTP + ≥32k 上下文冷 prefill 并发时双卡计算/拷贝引擎 100%、host 停在 device sync，ttft 飙至约 200s 后无响应、吞吐归零，利用率指标显示健康；禁用编译或 spec 即规避。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/54796

### INF-0892 `vllm_qknorm_rope_fusion_b200_fp8_silent_split_fail` B200+FP8 下 QK Norm+RoPE 融合因重复 split 静默失效

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-07 · 旧号 9.9

- **原因对象**：编译器 / vllm / QKNorm+RoPE 融合 pass 匹配
- **原因行为**：Inductor CSE 未合并相同的 split_with_sizes，图中为 3 个各 1 user 的 split 节点，而 pattern matcher 期望单个 split 带 q/k/v 三个 getitem user
- **影响对象**：性能/资源（性能退化）
- **影响详情**：B200 + FP8 模型（如 Qwen3-30B-A3B-FP8）上 QK Norm+RoPE 融合静默不生效、回退到非融合路径；输出正确，无 crash、无 NaN，仅性能退化
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/33967

## 运行时/执行

### INF-0015 `ttmetal_dram_sharded_dup_xcoord_buffer_corruption` TensorAccessor 只按 x 坐标选 bank 致 buffer 损坏

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-07-31 · 旧号 1.15

- **原因对象**：运行时/执行 / tt-metal / TensorAccessor bank 映射
- **原因行为**：对 DRAM buffer，get_noc_addr 只用 core 的 x 坐标作 bank id、忽略 y 坐标，host 侧也不做 DRAM 坐标虚拟化；shard grid 含重复 x 坐标时多个 core 映射到同一 bank。
- **影响对象**：数值结果（静默错误）
- **影响详情**：ND-sharded DRAM buffer 的 shard grid 含重复 x 坐标（如 CoreRange((0,0),(1,1)) 只有 2 个不同 x）时写入地址重叠，读回被覆盖的错误值，一旦触发整块 buffer 持续错乱。无 crash。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/51503

### INF-0019 `ttmetal_erisc_byte_vs_word_count_overcopy` Blackhole ERISC 固件按字节传字长参数致本地内存 4 倍越界

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-01 · 旧号 1.19

- **原因对象**：运行时/执行 / tt-metal / AERISC 固件本地内存初始化
- **原因行为**：initialize_local_memory() 把 .data 段大小以字节传给 len 单位为 32 位字的 l1_to_local_mem_copy()，拷贝量放大 4 倍。
- **影响对象**：可用性/进度；待定（静默错误）
- **影响详情**：固件 boot 时从 .data 起写 4 倍字节，越界覆盖 .bss/栈并越界读 L1 init image；AERISC 本地内存窗口仅 6400 字节，.data/.rodata 非空时必然破坏，active-ethernet 控制流状态静默损坏。原文未说明对上层计算的影响。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/51238

### INF-0037 `ttmetal_dispatcher_paged_write_unaligned_stride` dispatcher paged write 未按对齐步进致写入错位

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-06 · 旧号 1.37

- **原因对象**：运行时/执行 / tt-metal / dispatcher 分页写 bank 遍历
- **原因行为**：#51812 用手写 bank cycle walk 替换 TensorAccessor::get_noc_addr，walk 按原始 page size 步进，而 bank 内行间步幅应为向上对齐到 allocator 对齐后的大小。
- **影响对象**：数值结果（静默错误）
- **影响详情**：page size 未按 allocator 对齐时，每行之后的首行偏短、数据写到错误位置；test_dispatcher 的 PagedWrite DRAM 用例自 2026-08-04 起在 Blackhole 与 Wormhole 的 runtime_fd2 上验证失败。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/pull/52227

### INF-0058 `sglang_sm103_sm100_family_gate_kernel_misadmit` SGLang 把 sm_103 误判为 SM100 致 Xid 13 与挂起

`故障` · `active` · `待定` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-11 · 旧号 1.58

- **原因对象**：运行时/执行 / sglang / is_sm100_supported 家族检查
- **原因行为**：is_sm100_supported() 以 major==10 做家族级检查，sm_103（compute 10.3）因此被放行进入 SM100 专属 kernel 路径。
- **影响对象**：可用性/进度（挂起）
- **影响详情**：B300（sm_103）上 cutedsl TGV BF16 GEMM 抛 Xid 13 CGA「CTA Not Present」，trtllm-gen MoE finalize 静默挂起、无输出无日志；两者是通过 CUDA coredump 而非运行时日志发现的。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/34340

### INF-0076 `ttm_interleaved_to_sharded_dispatch_cache_hash_domain_wider_than_params` interleaved_to_sharded 缓存命中沿用错误 stride

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-15 · 旧号 1.77

- **原因对象**：运行时/执行 / tt-metal / interleaved_to_sharded
- **原因行为**：dispatch cache 自定义 hash 以 padded_shape 为键，但 RM 分支的行单元数由 logical shape 计算；op 无 runtime args override，cache 命中后不重新应用。
- **影响对象**：数值结果（静默错误）
- **影响详情**：padded shape 相同而 logical shape 不同的两次 dispatch 合法命中同一 cache 项（validator 不约束 logical=padded），第二次静默沿用第一次的 row stride，数据布局被错读，无 crash、无报错。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/53196

### INF-0080 `triton_kernel_cache_group_nonatomic_mixed_artifact_load` Triton kernel cache 组发布非原子，并发编译装载混组产物

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-08-16 · 旧号 1.81

- **原因对象**：运行时/执行 / triton / Triton kernel cache 组发布
- **原因行为**：kernel cache 条目是一组文件（IR、cubin、metadata、__grp__ manifest），单文件经 os.replace 原子写入，但整组发布非原子；并发编译同一 key 的多 writer 向同一目录写同名文件。
- **影响对象**：数值结果（静默错误）
- **影响详情**：共享 TRITON_CACHE_DIR（多进程 worker 或多主机 NFS）且同一 kernel 并发首次编译时，读者可能装载 A 的 cubin 配 B 的 metadata/IR，导致静默错误结果或运行时崩溃，依赖编译时序、难复现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/triton-lang/triton/pull/11319

### INF-0089 `ttmetal_program_cache_hit_cb_region_l1_overlap_silent` program cache 命中不重验 CB 区域，静默覆写 L1 张量

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-18 · 旧号 1.90

- **原因对象**：运行时/执行 / tt-metal / program cache 命中路径 CB 校验
- **原因行为**：静态 CB 区域只在 program cache miss 时经 validate_circular_buffer_region() 校验；cache-hit 路径只 patch buffer 地址、不重验，L1 分配也不触发缓存失效。
- **影响对象**：数值结果（静默错误）
- **影响详情**：两次运行间 L1 占用增长越过缓存的 CB 区域（如新分配大 L1 tensor）后重跑同一 op，CB 静默覆写 resident tensor；无 crash、无诊断，只断言 op 自身输出的测试会空洞通过。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/53537

### INF-0090 `ttmetal_get_max_l1_space_hybrid_percore_blind` get_max_l1_space 在 HYBRID 下高估 CB 空间

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-18 · 旧号 1.91

- **原因对象**：运行时/执行 / tt-metal / get_max_l1_space L1 预算查询
- **原因行为**：get_max_l1_space() 依赖的设备级查询在 HYBRID allocator 下只问 bank 0 分配器，看不到其他 bank 的 per-core 分配；验证器却逐核遍历全部分配器，口径分叉。
- **影响对象**：数值结果；可用性/进度（静默错误）
- **影响详情**：HYBRID allocator + per-core 分配下，走该预算的 op（untilize/transpose/matmul 等）构建出超额 CB 计划：有验证路径时被拒绝而 crash，无验证路径时 CB 与数据静默重叠。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/53538

### INF-0091 `ttmetal_quasar_noc_overlay_cmd_buff_reinit_missing` Quasar noc_init 未重置 overlay 缓冲致 L1 损坏

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-18 · 旧号 1.92

- **原因对象**：运行时/执行 / tt-metal / Quasar NOC 初始化 noc_init
- **原因行为**：Quasar 的 noc_init() 缺少 overlay_cmd_buff_init 调用，per-core NOC overlay command buffer 在程序边界未重新初始化，残留状态带入下一个程序。
- **影响对象**：数值结果（静默错误）
- **影响详情**：同进程背靠背运行多个程序（sub-device 测试序列）时第二个程序出现 L1 corruption：Watcher 开启时 TT_THROW 报错，关闭时为静默损坏；单独运行正常。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/53566

### INF-0099 `ttmetal_kimi_k26_dense_path_intermittent_pcc0_flap` Kimi-K2.6 dense 路径 prefill 间歇性输出全错

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-21 · 旧号 1.100

- **原因对象**：运行时/执行 / tt-metal / Blaze prefill dense 公共路径
- **原因行为**：根因未定位。缺陷位于 Blaze prefill pipeline 的 dense 公共路径（DSA indexer 路径不可达仍失败、pristine main 亦复现），且为非确定性。
- **影响对象**：隐状态（不可复现）
- **影响详情**：Kimi-K2.6-1T padded chunked prefill（15k@5k、no-trace、BH）间歇性返回与 golden 零相关的 decoder 输出（min PCC −0.0012，阈值 0.88）；retry 可通过，形似 CI flake；无 crash。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/53883

### INF-0133 `xpu_graph_capture_mtp_k4_persistent_state_nondeterministic_logits` XPU graph capture 下 MTP k=4 输出非确定

`故障` · `active` · `ambiguous` · `hw_specific` · `documented` · `var` · 发现 2026-09-01 · 旧号 1.134

- **原因对象**：运行时/执行 / xpu / XPU graph capture 持久显存状态
- **原因行为**：Battlemage XPU graph capture 的持久显存/上下文状态在推测解码第 4 条 draft 通道的读写模式下被损坏；关闭 capture 即确定，batch 回退路径干净，排除内核本身。
- **影响对象**：输出内容；输出分布（不可复现）
- **影响详情**：temperature=0 下同一请求输出在 2–3 个变体间循环，top-1 logprob 摆幅约 1.5 nats、top token 可翻转，约 40 请求后退化为重复循环；k≤3、eager、compile 无 capture 均干净，边界恰在 k=4。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/54785
- **关联**：var:battlemage_xpu_persistent_vram_state_corruption_token0

### INF-0135 `xpu_battlemage_enginecore_graphs_replay_indefinite_hang_concurrent` Battlemage XPU graph replay 并发下无限挂起

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `var` · 发现 2026-09-01 · 旧号 1.136

- **原因对象**：运行时/执行 / xpu / xpu graphs replay 并发路径
- **原因行为**：XPU graph replay 在并发多请求下于驱动/运行时层挂起；与推测解码无关，graph replay 本身在并发下即挂，且无 traceback。
- **影响对象**：可用性/进度（挂起）
- **影响详情**：Arc B70 单卡并发负载下 EngineCore 卡在 torch.xpu.graphs.replay()，无 crash、无日志推进，引擎停止服务，客户端超时是唯一外部信号；GPTQ Int4 + fp8 KV + 110k 上下文形态，MTP 开/关均复现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/54698
- **关联**：var:battlemage_xpu_persistent_vram_state_corruption_token0

### INF-0141 `ttmetal_quasar_dfb_chain_dm_outnumbers_tensix_wrong_data` Quasar DFB 链 DM 线程多于 Tensix 时静默交付错数据

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-09-02 · 旧号 1.142

- **原因对象**：运行时/执行 / tt-metal / Metal 2.0 DFB 链式同步
- **原因行为**：dataflow buffer 链式路径存在线程关系缺陷：任一 DFB 的 DM endpoint 多于 Tensix 侧（违反 R<=C 且 W<=C）时，级间 credit/同步假设被破坏，tile 缺失或重复。
- **影响对象**：数值结果（静默错误）
- **影响详情**：DM→DFB→Tensix→DFB→DM 管线 45.83%–70.83% 的 word 出错，无 hang、无报错、无降速；最大线程配置（6 DM + 4 Neo）反而 bit-exact，gtest 与 TTNN 两套仪器在 31/31 配置上一致。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/55276

### INF-0145 `ttmetal_quasar_cq_wraparound_modulo_address_basis_miscompute` Quasar 完成队列读指针回写取模算错地址，多 CQ 下停顿

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-09-03 · 旧号 1.146

- **原因对象**：运行时/执行 / tt-metal / SystemMemoryManager 读指针回写
- **原因行为**：以 completion_q_rd_ptr % completion_tlb_size 把 dispatch-core L1 地址折算成 TLB 窗口偏移，取模不是有效换算；多 CQ 下各 CQ 地址平移，回写可落到错误 L1 地址。
- **影响对象**：可用性/进度（挂起）
- **影响详情**：Quasar 多 CQ 拓扑下设备看到陈旧读指针、认为完成队列仍满（尤其 wraparound 附近），导致停顿/挂起类活性退化；无错数据、无 crash 信号。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/55406

### INF-0159 `ttmetal_binary_ng_program_cache_key_omits_alignment_tile` binary_ng 缓存哈希键漏 Alignment/Tile

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-09-03 · 旧号 1.160

- **原因对象**：运行时/执行 / tt-metal / binary_ng to_hash 缓存键
- **原因行为**：binary_ng 的 tensor_args_t::to_hash 与 attribute_values() 只哈希 dtype/memory_config/分页 shape，漏掉 Alignment 与 Tile 两项语义输入。
- **影响对象**：数值结果（静默错误）
- **影响详情**：同 dtype/memory_config 但 Alignment/Tile 不同的张量共享同一 program cache 条目，存在命中并复用错误程序的风险，当前版本静默出错；Metal 2.0 合法性检查下会显形为 legality check failure。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/55414

### INF-0235 `quasar_trisc3_mailbox_read_before_write_lands_stale_value` Quasar TRISC3 mailbox 读早于写，取到陈旧值

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-09-20 · 旧号 1.236

- **原因对象**：运行时/执行 / tt-metal / Quasar mailbox 读写同步
- **原因行为**：mailbox_write 与对侧 mailbox_read 之间缺生产者-消费者同步，Quasar 新增 IsolateSfpu 线程（TRISC3）的读窗口落在 Unpack 侧写入生效之前，读到上一轮或初始值。
- **影响对象**：数值结果（静默错误）
- **影响详情**：DataflowBufferReadTileValue 单测返回 incorrect results，DPRINT 显示各线程寄存器值错位；DataflowBuffer tile 寻址走该路径时会读取错误 tile，无 NaN 或崩溃特征。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/57066

### INF-0236 `ttmetal_program_spec_cache_alias_primed_binding_freeze_first_match_misbind` ProgramSpec 缓存冻结别名绑定，后续调用错绑输入

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-09-21 · 旧号 1.237

- **原因对象**：运行时/执行 / tt-metal / ProgramSpec 缓存绑定解析
- **原因行为**：按 MeshTensor 地址 find_if 取首个匹配解析参数位置，别名调用使多个参数槽解析到同一索引，该绑定表被冻结进程序缓存条目，cache hit 时只刷新张量指针不重算绑定。
- **影响对象**：数值结果（静默错误）
- **影响详情**：同进程先以别名调用（如 concat([x,x])）预热、再以不同张量调用同 spec 时，concat([a,b]) 返回 concat(a,a)；反序调用正确，结果依赖调用历史；多输入已移植 op 均暴露。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/57248

### INF-0286 `ttmetal_trace_replay_overwrites_buffers_allocated_after_capture` ttnn trace 回放静默覆写捕获后分配的设备 buffer

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-10-09 · 旧号 1.287

- **原因对象**：运行时/执行 / tt-metal / ttnn trace 捕获/回放
- **原因行为**：trace 只持有输入输出张量，捕获期中间 buffer 释放后，每次 execute_trace 仍按捕获期地址写入，与之后在同区域的分配重叠；仅 docstring 警告，无强制。
- **影响对象**：数值结果；输出内容（静默错误）
- **影响详情**：捕获后分配且落入该区域的设备 buffer 每次回放都被覆写，覆写块集合对给定分配历史确定；MiniMax-H3 开启 adaLN 调制表缓存后生成样本比无缓存管线差 12.6 dB PSNR。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/59645

### INF-0329 `sglang_deterministic_flag_silent_coverage_gap_quantized_hybrid` SGLang 确定性 flag 对量化+hybrid+drafter 静默失效

`故障` · `active` · `ambiguous` · `other_stack` · `documented` · `NEW` · 发现 2026-08-26 · 旧号 4.13

- **原因对象**：运行时/执行 / sglang / 确定性推理 flag 覆盖范围
- **原因行为**：确定性推理 flag 的 batch-invariant 调度/内核选择路径未覆盖量化反量化、hybrid recurrent state 或 drafter 中至少一维，flag 仍被接受且无未覆盖告警
- **影响对象**：输出内容（不可复现）
- **影响详情**：GB10 上 NVFP4 W4A4 + hybrid Mamba + DFlash2 drafter 组合：ServerArgs 显示 enabled，但 temp=0 并发相同请求探针持续失败；同环境 dense BF16 开 flag 后从 8 个 completion 坍缩为 1
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/36291

### INF-0341 `vllm_hybrid_single_token_prompt_full_cudagraph_nondeterministic_garbled` vLLM 混合 Mamba 单 token prompt 全图模式下非确定乱答

`故障` · `active` · `infra_off` · `config` · `verified` · `NEW` · 发现 2026-09-19 · 旧号 4.25

- **原因对象**：运行时/执行 / vllm / 全图 CUDA graph 单 token 捕获
- **原因行为**：full CUDA graph 捕获 prefill 长度 1 的形状时，把 Mamba conv/recurrent 状态初始化或 attention padding 按错误布局固化进图；逐段执行语义正确，损坏特定于全图捕获的跨段融合。
- **影响对象**：输出内容（不可复现）
- **影响详情**：granite-4.0-h、Falcon-H1 恰好 1 token 的 prompt 在 temperature 0 下重复 6 次得 4–6 种 completion，多数与 prompt 无关；2–6 token 全稳定，enforce-eager 或 PIECEWISE 下 0/40 失败。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/57720

### INF-0346 `vllm_batch_invariant_cold_model_info_cache_leaves_attention_unwrapped_in_graphs` vLLM 批不变模式冷缓存下 attention 未包裹即被捕获进图

`故障` · `active` · `infra_off` · `config` · `documented` · `var` · 发现 2026-10-09 · 旧号 4.30

- **原因对象**：运行时/执行 / vllm / breakable cudagraph 装饰器时序
- **原因行为**：@eager_break_during_capture 在 import 时求值，早于 breakable cudagraph 使能；冷 model-info cache 下 attention op 未包裹即被捕获进图，fork 后继承。
- **影响对象**：输出内容（静默错误）
- **影响详情**：attention 以捕获期 FlashAttention 参数回放，默认调度下 16-token prefill 输出错 token（max_num_seqs=16 时 2–16 token 均错），无警告；进程内与默认多进程 LLM 引擎中招，vllm CLI（spawn）免疫。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/59940
- **关联**：var:INF-0333

### INF-0359 `vllm_pp_decode_full_cudagraph_captures_gloo_recv_object_silent_hang` vLLM FULL graph 捕获 PP gloo recv 致静默挂起

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-08-12 · 旧号 5.12

- **原因对象**：运行时/执行 / vllm / FULL decode CUDA graph 捕获
- **原因行为**：自动启用 breakable cudagraph 时捕获的 full-decode graph 含 PP 边界的 host 端 gloo recv_object，重放时无 host 参与，该 recv 永远得不到服务。
- **影响对象**：可用性/进度（挂起）
- **影响详情**：PP>1 多节点且模型无 @support_torch_compile 时 graph 捕获成功，首个 decode 请求即静默挂起，约 300s 后 RPC sample_tokens 超时；--enforce-eager 或关闭 breakable cudagraph 可工作。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/51887

### INF-0362 `sglang_kfd_peerdirect_multinic_reregistration_memory_accounting_corruption` KFD 多 NIC 注册致显存 free 虚增，SGLang 超配 OOM

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-15 · 旧号 5.15

- **原因对象**：运行时/执行 / rocm / amdgpu/KFD 驱动显存记账
- **原因行为**：同一 GPU buffer 经 ibv_reg_mr 在多个 NIC 上注册为 RDMA MR 时，KFD 驱动每多一个 NIC 少计一份用量，hipMemGetInfo 的 free 虚增、used 可下溢。
- **影响对象**：指标/日志；可用性/进度（报错）
- **影响详情**：MI355X 上 8GB buffer 注册 4 个 NIC 后 free=325.58GB，超过物理总量 309GB；Mooncake 分离式部署中 SGLang 按该值确定 KV pool 大小，过度分配导致服务启动期 OOM crash。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/pull/34641

### INF-0368 `vllm_shm_broadcast_startup_wedge_models_endpoint_healthy` vLLM shm_broadcast 启动竞态致引擎死而 API 仍健康

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-08-20 · 旧号 5.21

- **原因对象**：运行时/执行 / vllm / shm_broadcast 启动期块分配；调度/服务 / vllm / 引擎死亡后的 HTTP API server
- **原因行为**：启动时 shm_broadcast 共享内存块饥饿约 60s 的竞态使 rank 0 worker 意外退出并 EngineDeadError；引擎已死时 HTTP server 仍照常启动并响应。
- **影响对象**：可用性/进度；指标/日志（报错）
- **影响详情**：8-rank TP + Ray 启动时 7 次中 5 次复现；受控试验中 2/3 次 GET /v1/models 返回 200 而生成已死，基于 models 端点的 readiness 检查报告健康，只有真实 completion 请求能暴露。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/53157

### INF-0379 `vllm_dbo_ubatch_early_leaver_ring_handoff_none_deref_and_parked_hang` vLLM DBO 微批早退撕裂 handoff 致崩溃或挂起

`故障` · `active` · `infra_off` · `partial` · `documented` · `var` · 发现 2026-09-06 · 旧号 5.32

- **原因对象**：运行时/执行 / vllm / UBatchContext 微批 handoff
- **原因行为**：UBatchContext.__exit__ 在其他微批仍运行时清空离开者的上下文槽位；recv-hook 查找只判全局表非空而非 peer 仍在；异常微批不再 yield，exit 信号只释放一次。
- **影响对象**：可用性/进度（挂起）
- **影响详情**：DBO/ubatch slicing 下某微批抛异常或提前完成时，幸存微批或因 recv_hook 查找得 None 抛 AttributeError 崩溃，或 parked 在 yield 等待永不到来的 handoff，step 无限挂起。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/55586
- **关联**：var:sglang_pdmux_mixed_batch_deadlock

### INF-0408 `vllm_gb10_enginecore_silent_warmup_death` vLLM 在 GB10 上 EngineCore 于 warmup 静默死亡

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-07-30 · 旧号 6.15

- **原因对象**：运行时/执行 / vllm / EngineCore warmup 阶段
- **原因行为**：待定：原文未定位根因，仅知设置 CUDA_LAUNCH_BLOCKING=1 可避开。
- **影响对象**：可用性/进度（报错）
- **影响详情**：GB10（sm_121）上启动 Qwen3-0.6B 时，EngineCore 在 flashinfer autotune 结束后、startup complete 前死亡，无 traceback、CUDA 错误或 OOM；API server 报 EngineDeadError 后关闭。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/50067

### INF-0411 `vllm_uva_gpu_cc_silent_zero_token_input` CC 模式下 is_pinned 误判致 vLLM 静默喂入全零 token

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-08-01 · 旧号 6.18

- **原因对象**：运行时/执行 / vllm / cuda_view.cu UVA 视图回退路径
- **原因行为**：CC 模式下 is_pinned 对 pinned 内存返回 false，get_cuda_view_from_cpu_tensor 走 detached-buffer 回退，破坏 UVA buffer 的 live-alias 契约。
- **影响对象**：输入编码；输出内容；可用性/进度（静默错误）
- **影响详情**：CC 模式下每个请求向模型喂入全零 token 却返回 HTTP 200，服务层与正常输出不可区分；退化生成使 async scheduler 的 num_output_placeholders 下溢，单个长上下文请求即以 AssertionError 崩溃 EngineCore。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/50671

### INF-0422 `vllm_minimax_m3_fp8_query_alloc_unstable_cudagraph_replay` MiniMax-M3 query_fp8 分配跨 graph 回放不稳定

`故障` · `active` · `infra_off` · `yes` · `verified` · `NEW` · 发现 2026-08-05 · 旧号 6.29

- **原因对象**：运行时/执行 / vllm / MiniMax-M3 query_fp8 分配条件
- **原因行为**：query_fp8 在 eager CUDA-graph break 之外分配，但分配条件依赖在 eager attention 段内重新评估的动态请求 metadata，跨 capture/replay 不稳定。
- **影响对象**：可用性/进度（报错）
- **影响详情**：FP8 + CUTLASS MSA + CUDA graph 下，graph 在无 CUTLASS plan 时捕获、replay 时请求已可走 CUTLASS，eager CUTLASS 路径收到 query_fp8=None，assert query_fp8 is not None 失败崩溃。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/51203

### INF-0423 `sglang_qwen3vl_deepstack_input_dropped_bcg_replay` SGLang BCG 回放丢弃 Qwen3-VL DeepStack 输入

`故障` · `active` · `infra_off` · `other_stack` · `verified` · `NEW` · 发现 2026-08-05 · 旧号 6.30

- **原因对象**：运行时/执行 / sglang / Prefill BCG replay 输入槽拷贝
- **原因行为**：Prefill breakable CUDA graph（BCG）replay 未把请求专属的 input_deepstack_embeds 复制进稳定的 graph input slot，该 slot 保留默认零值。
- **影响对象**：隐状态；输出分布；输出内容（静默错误）
- **影响详情**：Qwen3-VL-8B-Instruct（H200、BF16、greedy）走 BCG 时请求正常完成，但与 eager 仅前 7 个 token 相同、最大 logprob 差 1.154；bcg_normal 与 bcg_zero_deepstack 差 0.000。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/pull/33726

### INF-0424 `vllm_dsv4_c128a_topk_stride_capture_vs_runtime_mismatch` vLLM DSv4 C128A stride 捕获与运行时不一致

`故障` · `active` · `infra_off` · `yes` · `verified` · `NEW` · 发现 2026-08-06 · 旧号 6.31

- **原因对象**：运行时/执行 / vllm / _build_c128a_metadata
- **原因行为**：_build_c128a_metadata 由 max_seq_len 派生行 stride；cudagraph 捕获按 max_model_len 烘焙最宽 stride，runtime 按当前 batch 实际值排列行。
- **影响对象**：输出内容；数值结果（静默错误）
- **影响详情**：DeepSeek-V4-Flash 并发长上下文负载下，除 row 0 外各行 stride 不匹配，生成中途出现 <|begin_of_sentence|> burst 和多语言 token salad，每 batch 一个请求出错、从不是第一个；无 crash、无 NaN。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/51318

### INF-0447 `sglang_overlap_batch_freed_while_forward_reads` SGLang overlap 调度在 forward 仍读时释放张量

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-08-08 · 旧号 6.54

- **原因对象**：运行时/执行 / sglang / overlap 调度器的张量生命周期
- **原因行为**：_forward_isolation 退出时丢掉 forward 内重绑的 input_ids/out_cache_loc 唯一引用；mix_running_indices 在 forward 流仍读时被释放。
- **影响对象**：输出内容（静默错误）
- **影响详情**：overlap 调度配推测解码（Spec-V2/EAGLE）时触发；缓存分配器把已释放块分给同流重分配，造成 in-range 损坏，server 无 crash 继续运行但发出错误 token。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/pull/34076

### INF-0448 `sglang_relay_staging_unfenced_tail_write` SGLang relay 跨流写无 fence，新请求读前任状态

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-08-08 · 旧号 6.55

- **原因对象**：运行时/执行 / sglang / FutureMap relay 跨流写
- **原因行为**：FutureMap relay 由 forward stream 的 tail relay 写与 schedule stream 的 batch-prep staging 写同一 pool 行，二者之间没有排序/fence。
- **影响对象**：跨请求/跨样本数据；输出内容（静默错误）
- **影响详情**：overlap 调度配 PD-disagg/EAGLE disagg/hisparse 且前任最终 forward 仍在跑时触发；stale tail 写落在 staging 后，新请求首次 resolve 静默读到前任的 bonus token/topk/seq_lens，无 crash。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/pull/34077

### INF-0461 `sglang_cache_dit_silent_disable_with_breakable_graph` SGLang Cache-DiT 遇 breakable graph 静默禁用

`故障` · `active` · `model_on` · `config` · `documented` · `NEW` · 发现 2026-08-09 · 旧号 6.68

- **原因对象**：运行时/执行 / sglang / _maybe_enable_cache_dit
- **原因行为**：启用 --enable-breakable-cuda-graph 时 DenoisingStage._maybe_enable_cache_dit 提前返回、不启用 Cache-DiT，但既不报错也不警告。
- **影响对象**：性能/资源（性能退化）
- **影响详情**：breakable CUDA graph 与 SGLANG_CACHE_DIT_ENABLED=1 同启时触发；无错误，唯一症状是 1.00x speedup，易误读为 Cache-DiT 对该模型无效；MiniMax-H3 按请求动态 mount 时也可能逐请求静默失效。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/pull/34177

### INF-0464 `sglang_inference_mode_mismatch_lazy_buffers` SGLang warmup 的 inference_mode 不一致致报错

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-08-10 · 旧号 6.71

- **原因对象**：运行时/执行 / sglang / warmup/autotune 上下文
- **原因行为**：warmup/autotune 在 inference_mode 下运行，而真实 forward 与 graph capture 不在；后端懒分配的 buffer 若首次分配于 warmup，即成为 inference tensor
- **影响对象**：可用性/进度（报错）
- **影响详情**：首次分配于 FlashInfer autotune/warmup 的懒分配 buffer，在真实 forward 中 inplace 更新时报 RuntimeError（Inplace update to inference tensor outside InferenceMode），非启动阶段。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/33470

### INF-0465 `sglang_dspark_cudagraph_correctness_robustness_roadmap` SGLang DSpark 推测解码 CUDA-Graph 正确性五层分解

`结论` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-08-10 · 旧号 6.72

- **原因对象**：运行时/执行 / sglang / DSpark CUDA-Graph 执行路径
- **原因行为**：DSpark CUDA-Graph 执行有五层独立缺陷：多卡控制面一致、verifier 状态/写窗口、kernel plan/index、capture 初始化顺序、replay 的 ownership/stream/lifetime
- **影响对象**：可用性/进度（报错）
- **影响详情**：DSpark 推测解码 + CUDA-Graph + 多卡下，各层缺陷都表现为相似的 CUDA illegal memory access 或 graph replay 失败（crash），原先难以区分来源；关联 #31195、#32183、#32467、#33795。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/34297

### INF-0477 `vllm_kimi_k3_kda_recurrent_kernel_full_cudagraph_capture_fault` vLLM KDA 算子被 FULL cudagraph 内联致 GPU 故障

`故障` · `active` · `infra_off` · `partial` · `verified` · `NEW` · 发现 2026-08-13 · 旧号 6.84

- **原因对象**：运行时/执行 / vllm / FULL 模式 eager-break 快路径
- **原因行为**：eager_break_during_capture 的 FULL 快路径把被装饰调用内联进 graph，使非 capture-safe 的 KDA recurrent kernel 被捕获进 FULL graph
- **影响对象**：可用性/进度（报错）
- **影响详情**：Kimi-K3 在 FULL_AND_PIECEWISE 下跑 FULL uniform-decode batch（如 spec-decode verify）时，capture/replay 出现 GPU Memory access fault；target-only cudagraph 正常。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/52205

### INF-0485 `trtllm_autotuner_profiling_cache_cross_device_silent_load` TRT-LLM AutoTuner 缓存不校验设备致跨设备复用 tactic

`故障` · `active` · `ambiguous` · `config` · `documented` · `NEW` · 发现 2026-08-13 · 旧号 6.92

- **原因对象**：运行时/执行 / tensorrt-llm / AutoTuner 缓存键与元数据
- **原因行为**：AutoTunerProfilingCache 写入 device_name/device_capability 但从不读回校验，cache key 也不含设备，其他 GPU 生成的 cache 可命中并返回本设备未测过的 tactic
- **影响对象**：数值结果；性能/资源（静默错误）
- **影响详情**：多设备环境复用 profiling cache 文件时，静默选用其他设备测得的 kernel tactic，日志无任何提示；原文称结果可能次优，也可能数值不正确。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/NVIDIA/TensorRT-LLM/pull/17664

### INF-0492 `vllm_kimi_k3_multinode_tp_cudagraph_three_mode_corruption` Kimi-K3 多机 EFA TP 下 CUDA graph 静默输出损坏

`故障` · `active` · `infra_off` · `yes` · `documented` · `NEW` · 发现 2026-08-16 · 旧号 6.99

- **原因对象**：运行时/执行 / vllm / K3 MoE/KDA 路径 graph 捕获
- **原因行为**：根因未定位。该拓扑禁用 FlashInfer fused all-reduce、collective 走 NCCL over EFA；报告者怀疑 K3 MoE/KDA 路径有捕获外分配的 buffer 在此路径上被触及。
- **影响对象**：输出内容；可用性/进度（静默错误）
- **影响详情**：TP16 双节点 B200 首请求即复现：FULL_AND_PIECEWISE 输出流利但语义随机、第 3 次请求 SIGSEGV；PIECEWISE 塌缩为单 token 重复；另一模式报 illegal memory access；enforce_eager 始终正确。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/52531

### INF-0493 `flashinfer_gemm_workspace_resize_graph_captured_uaf` FlashInfer workspace 扩容搬址致 graph 用后释放

`故障` · `active` · `infra_off` · `yes` · `documented` · `NEW` · 发现 2026-08-16 · 旧号 6.100

- **原因对象**：运行时/执行 / flashinfer / gemm_base 共享 workspace 扩容
- **原因行为**：cuDNN GEMM 路径在执行计划需更大空间时对共享 32 MiB workspace 调 resize_()，buffer 搬到新地址，已入队 launch 与此前捕获的 CUDA graph 仍解引用旧地址。
- **影响对象**：可用性/进度；数值结果（报错）
- **影响详情**：graph 捕获中途 workspace 增长时触发（SM120 NVFP4/FP8 实测）：旧地址被 allocator 复用，GEMM 先静默读到陈旧内容，sustained 负载下数分钟内 SIGSEGV/GPU coredump 或 wedge。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/52553

### INF-0521 `vllm_prefill_shape_misdispatch_spec_cudagraph_gdn_state_loss` 3-token prefill 误判为 decode 致 GDN 状态丢失

`故障` · `active` · `infra_off` · `yes` · `documented` · `NEW` · 发现 2026-08-20 · 旧号 6.128

- **原因对象**：运行时/执行 / vllm / _is_uniform_decode 形状判据
- **原因行为**：_is_uniform_decode 纯按形状判定，MTP k=2 时 3-token 首次 prefill 被误判为 decode、派入 spec FULL cudagraph，GDN recurrent-state 写被跳过。
- **影响对象**：KV状态；输出内容（静默错误）
- **影响详情**：MTP k=2 + Qwen3-Next/Qwen3.5-MoE 等 hybrid GDN 模型下，3-token prompt（如“中国的首都是”）首步即毒化请求，后续 decode 从零状态产出确定性垃圾输出，draft 接受率坍塌；无 crash。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/53051

### INF-0522 `vllm_empty_splitting_ops_guard_skip_piecewise_cudagraph_spec_reject` vLLM 早退绕过 splitting_ops 防护致 cudagraph 错值

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-08-20 · 旧号 6.129

- **原因对象**：运行时/执行 / vllm / CompilationConfig 防护早退
- **原因行为**：set_splitting_ops_for_v1 在非 VLLM_COMPILE 模式早退，绕过“splitting_ops 为空则强制 cudagraph_mode=NONE”防护，attention 进入 piecewise 捕获区。
- **影响对象**：数值结果；性能/资源（静默错误）
- **影响详情**：经 VLLM_USE_BREAKABLE_CUDAGRAPH 可在 stock vLLM 到达；query length 与捕获值不同的 batch 静默产出错误结果，叠加推测解码时 draft 被全部静默拒绝、退化为无加速逐 token 解码；无 crash、无日志。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/53030

### INF-0542 `vllm_sleep_wake_lora_state_silent_corruption` vLLM sleep level=1 唤醒后 LoRA 状态静默损坏

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-08-26 · 旧号 6.149

- **原因对象**：运行时/执行 / vllm / sleep level=1 wake_up 恢复
- **原因行为**：sleep mode level=1（保留权重、释放 KV cache 内存）的 wake_up 未完整恢复引擎状态，有状态未被推回，恢复路径既不覆盖全部状态，也没有恢复后的完整性探针。
- **影响对象**：模型权重；输出内容（静默错误）
- **影响详情**：NemotronH NVFP4、TP=2、单 LoRA 下 sleep/wake 后，base 与 adapter 路由固定 seed 输出均偏离 sleep 前基线且彼此完全相同（adapter 不再生效），一张卡显存驻留少约 3GB；无错误、HTTP 200。level=2 正常。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/53888

### INF-0545 `vllm_hybrid_gdn_mtp_async_ima_exit0_supervisor_undeath` vLLM hybrid 异步调度 IMA 后 API server 退出码为 0

`故障` · `active` · `infra_off` · `partial` · `documented` · `var` · 发现 2026-08-27 · 旧号 6.152

- **原因对象**：运行时/执行 / vllm / 异步调度下 GDN+MTP 路径；调度/服务 / vllm / API server 致命错误退出码
- **原因行为**：async scheduling 下小请求完成与 mid-MTP-decode 长生成交织时，gdn_attn build 路径出现 CUDA IMA（根因未定位）；API server 随后以退出码 0 退出而非报告失败。
- **影响对象**：可用性/进度（报错）
- **影响详情**：Qwen3.5 hybrid GDN 27B + MTP k=3 + 异步调度 + fp8 KV，混合长短流量下 30–50% 复现，死亡请求已算 token 约 23.6K–26.2K；exit 0 使 systemd Restart=on-failure 不拉起，服务静默下线。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/53726
- **关联**：var:INF-1075

### INF-0560 `sglang_gb10_long_prefill_unified_memory_silent_worker_kill` SGLang GB10 统一内存长 prefill 静默杀死 worker

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-29 · 旧号 6.167

- **原因对象**：运行时/执行 / sglang / 统一内存下长 prefill 内存分配
- **原因行为**：GB10 统一内存上长 prefill 的激活峰值耗尽内存，worker 进程被杀（推测为内核 OOM-kill 或驱动路径），死亡路径绕过 Python 异常处理与内存记录。
- **影响对象**：可用性/进度（报错）
- **影响详情**：SGLang 在 GB10 上处理 >40k token 长 prefill 时 worker rank 消失，无 traceback、无 OOM 记录、无日志，需依赖 dmesg 等系统层证据；另一推理栈同输入 54k tokens 可通过。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/36941

### INF-0566 `vllm_kv_pool_first_boot_capacity_under_allocation` vLLM 首次启动 KV 容量探测偏低约 5%，容量依赖启动历史

`故障` · `active` · `infra_off` · `config` · `verified` · `NEW` · 发现 2026-08-30 · 旧号 6.173

- **原因对象**：运行时/执行 / vllm / 启动期 KV 容量显存探测
- **原因行为**：某 shape 首次启动时，compile 与 CUDA graph capture 的临时分配先于容量探测且部分驻留，探测在 warmup 未回收时采样可用显存，结果偏低。
- **影响对象**：性能/资源（性能退化）
- **影响详情**：每机每个 (max-num-batched-tokens, max-num-seqs) shape 首启或 compile cache 失效后，KV cache 容量比后续同参启动少约 5%（如 82,688 vs 88,096 token），首启慢 30–35s；无报错，部署容量取决于启动历史。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/54383

### INF-0567 `sglang_qwen38_qsa_nextn_decode_graph_silent_semantic_corruption` SGLang QSA+NEXTN decode graph 静默产出无效输出

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-08-30 · 旧号 6.174

- **原因对象**：运行时/执行 / sglang / decode CUDA graph 捕获/重放
- **原因行为**：decode CUDA graph 捕获/重放路径对 QSA 稀疏注意力 + NEXTN 推测解码组合的状态处理有缺陷，具体出错的捕获变量原文未定位。
- **影响对象**：输出内容（质量退化）
- **影响详情**：GB10（SM121）TP2 + Qwen3.8-Flash-Next 开 decode CUDA graph 时 HTTP 200 且吞吐更高，但 1,024-token essay 退化为重复「!」、~25K-token marker 检索全失败；关闭 CUDA graph 则全过。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/37111

### INF-0576 `sglang_prefill_breakable_cuda_graph_weakref_break_inputs_cross_bucket_alias` SGLang BCG 弱引用 break 输入致跨 bucket 内存复用

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-09-02 · 旧号 6.183

- **原因对象**：运行时/执行 / sglang / breakable graph 输入弱引用
- **原因行为**：breakable CUDA graph 以弱引用保存 graph-break inputs，误以为共享 mempool 的 use_count 能保活存储，实际只保活 pool 对象，blocks 被分给后捕获的 bucket。
- **影响对象**：输出内容；数值结果（静默错误）
- **影响详情**：SGLang CUDA 默认 prefill 路径多 bucket 捕获后，replay 早 bucket 读写另一 bucket 拥有的内存：依分配布局表现为 temp=0 下静默输出错误 token，或 CUDA illegal memory access 崩溃；非确定，默认配置即触发。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/37606

### INF-0582 `vllm_rocm_arch_string_overinclusion_glm53_mrv1_accuracy_collapse` vLLM ROCm 按架构字符串把 GLM-5.3 切到 MRV1 致精度塌方

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `var` · 发现 2026-09-04 · 旧号 6.189

- **原因对象**：运行时/执行 / vllm / ROCm 默认 MRV1 架构列表
- **原因行为**：为让 GLM-5.2 留在 MRV1，ROCM_DEFAULT_MRV1_ARCHITECTURES 加入 GlmMoeDsaForCausalLM；同架构字符串的 GLM-5.3 被一并切到 MRV1，而 MRV1 对其算错。
- **影响对象**：输出内容（质量退化）
- **影响详情**：vLLM ROCm MI350/MI355 上运行 GLM-5.3（Quark-MXFP4-AttnFP8）时服务正常返回 200，但 GSM8K 5-shot 从 91.6% 跌到 14.9%；bisect 定位 #53155，main 仍存在，不跑评测则无法发现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/54924
- **关联**：var:sglang_envs_declared_runtime_defaults_split

### INF-0583 `vllm_mrv2_workspace_unlocked_postcapture_growth_invalidates_graphs` vLLM MRV2 workspace 未加锁致 graph 写已释放内存

`故障` · `active` · `infra_off` · `partial` · `verified` · `NEW` · 发现 2026-09-04 · 旧号 6.190

- **原因对象**：运行时/执行 / vllm / MRV2 WorkspaceManager 扩容
- **原因行为**：V2 的 _ensure_workspace_size 在需求超出时替换整个 arena 并释放旧缓冲，但 V2 重写丢失了 V1 在 capture 后 lock workspace 的保护，已捕获 graph 仍引用旧地址。
- **影响对象**：跨请求/跨样本数据；输出内容（静默错误）
- **影响详情**：Model Runner V2（DSV4 等默认 V2）+ CUDA graphs，捕获后出现更大 workspace 需求（长上下文/大 batch 晚到）时，replay 写入已转交他用的内存：多为无 crash 的跨请求静默数据损坏，偶发 IMA/输出损坏，易误归因于硬件或模型。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/55336

### INF-0584 `vllm_mig_nVML_permission_swallowed_smallest_gpu_batch_defaults` vLLM MIG 下 NVML 探测失败被吞，batch 默认值静默落入最小档

`故障` · `active` · `infra_off` · `config` · `verified` · `NEW` · 发现 2026-09-04 · 旧号 6.191

- **原因对象**：运行时/执行 / vllm / NVML 显存探测与 batch 默认值
- **原因行为**：MIG 容器内 NVML 取父设备 handle 查询被权限拒绝，get_batch_defaults() 的 catch-all 吞异常并以 device_memory=0 兜底，未回退 torch total_memory。
- **影响对象**：性能/资源（性能退化）
- **影响详情**：H200 2g.35gb MIG 引擎依赖默认 batch 参数时静默拿到最小档 max_num_batched_tokens=2048/max_num_seqs=256（整卡档 8192/1024），吞吐降 28–53%；输出正确、默认日志级无告警，显式设置该参数可绕过。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/55381

### INF-0610 `vllm_dual_instance_concurrent_start_device_delta_memory_misattribution` vLLM 同卡并发启动两实例时显存 profiling 误记他进程分配

`故障` · `active` · `infra_off` · `yes` · `verified` · `var` · 发现 2026-09-08 · 旧号 6.217

- **原因对象**：运行时/执行 / vllm / Worker 可用显存 profiling
- **原因行为**：Worker.determine_available_memory 用整设备 free-memory 差值推导本实例 non-torch 内存，无法区分本进程与 profiling 窗口内他进程的分配；分配方向无任何检测。
- **影响对象**：性能/资源；可用性/进度（静默错误）
- **影响详情**：同 GPU 同时启动两个实例时 KV cache 静默缩水（如 4.72→1.84 GiB），或算出负值以 ValueError: No available memory for the cache blocks 启动失败、报错误导指向 gpu_memory_utilization；顺序启动正常。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/55827
- **关联**：var:INF-0529

### INF-0621 `vllm_mtp_pp_bcast_drafts_idx_mapping_missing_record_stream_ima` vLLM MTP+PP 广播缺 record_stream 致 IMA

`故障` · `active` · `infra_off` · `yes` · `verified` · `var` · 发现 2026-09-11 · 旧号 6.228

- **原因对象**：运行时/执行 / vllm / broadcast_drafts 侧流读取
- **原因行为**：broadcast_drafts 在侧流上按 idx_mapping 索引 draft_tokens，但每步新分配的 idx_mapping 未 record_stream，allocator 可在读完成前把该 block 复用给下次分配。
- **影响对象**：可用性/进度（报错）
- **影响详情**：MTP 推测解码 + PP>1 在负载压力下（TP2 PP3 A100、16 并发流约 90s）gather 读到已复用内存，出现 illegal memory access 与 EngineDeadError；崩溃前窗口内 gather 可能读到部分陈旧的映射。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/56470
- **关联**：var:vllm_mapped_layer_courier_missing_record_stream_storage_reuse

### INF-0631 `vllm_rocm_pd_disagg_remote_prefilled_sparse_mla_decode_cudagraph_garbage` vLLM ROCm PD 远端 prefill 请求 graph 解码出垃圾

`故障` · `active` · `infra_off` · `other_stack` · `verified` · `NEW` · 发现 2026-09-15 · 旧号 6.238

- **原因对象**：运行时/执行 / vllm / decode CUDA graph 路径
- **原因行为**：根因未定位到单点；KV 传输已证 bit-exact，故障收窄到 CUDA graph 捕获/replay 的元数据或 buffer 生命周期对 remote-prefilled KV 布局的某项假设不成立，eager 路径不经过该假设。
- **影响对象**：输出内容（静默错误）
- **影响详情**：ROCm 2×MI300X 1P1D（MoRIIO WRITE、fp8 KV）下 remote-prefilled 请求在任一 decode CUDA graph 模式输出重复垃圾，本地 prefill 请求与 eager 正常；只能全局关 decode CUDA graph 规避。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/57064

### INF-0635 `vllm_async_scheduler_chunked_prefill_premature_sampled_token_placeholder_underflow` vLLM async 调度分块 prefill 提前交付 token 致下溢

`故障` · `active` · `infra_off` · `config` · `verified` · `NEW` · 发现 2026-09-18 · 旧号 6.242

- **原因对象**：运行时/执行 / vllm / discard_request_mask 判据
- **原因行为**：async scheduling 下 runner 的 discard_request_mask 未丢弃未完成 prefill chunk 步的采样 token，scheduler 却未为其预留 placeholder，两侧判据不对称。
- **影响对象**：输出内容；可用性/进度（报错）
- **影响详情**：默认 async scheduling + chunked prefill + 并发多块长 prompt（约 9k+）时 num_output_placeholders 下溢、断言杀死 EngineCore；此前伪 token 已写入用户可见响应。关 async scheduling 可规避。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/57562

### INF-0713 `vllm_rocm_dsv4_mrv2_full_decode_only_graph_accuracy_drop` DSv4 在 MRV2 FULL_DECODE_ONLY 图模式下准确率下降

`故障` · `active` · `ambiguous` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-17 · 旧号 7.40

- **原因对象**：运行时/执行 / vllm / MRV2 FULL_DECODE_ONLY 图
- **原因行为**：根因尚在定位：MRV2 model runner 在 ROCm 上以 FULL_DECODE_ONLY 把 decode 全路径 graph 化后出现数值回归，原文推测为 capture/replay 边界或 kernel 选择差异。
- **影响对象**：输出内容（质量退化）
- **影响详情**：DeepSeek-V4-Pro-0813 在 MI350/MI355 上用 FULL_DECODE_ONLY 时 lm_eval 准确率显著下降，eager 与 PIECEWISE 正常；无 crash、无 NaN。上游已把 DSv4 移回 MRV1，复现需 pin commit。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/52644

### INF-0762 `vllm_dsv2lite_online_fp8_triton_mla_cudagraph_batch3_nan_logits` DSV2-Lite 在线 FP8 在 graph ≥3 序列批出 NaN

`故障` · `active` · `infra_off` · `config` · `verified` · `var` · 发现 2026-09-15 · 旧号 7.90

- **原因对象**：运行时/执行 / vllm / CUDA graph ≥3 序列 decode 批
- **原因行为**：根因未定位到内核；隔离收敛到 CUDA graph replay 的 ≥3 序列 decode 批，inductor 无罪；批尺寸阈值指向 graph 批维 padding/分组假设与在线 fp8 路径的交互。
- **影响对象**：输出分布；输出内容（数值异常）
- **影响详情**：RTX 4090（sm89）默认配置下 graph 批 decode ≥3 序列时每请求 NaN logits（请求 logprobs 时 HTTP 400，否则输出重复垃圾），1–2 序列正常，日志无输出；eager、关 cudagraph 或 max-num-seqs 2 可规避。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/57017
- **关联**：var:INF-0553

### INF-0822 `vllm_rocm_glm52_fp8_piecewise_cudagraph_cold_replay_poisons_own_prefill` vLLM ROCm CUDA graph 冷重放污染空闲后首请求 KV

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-14 · 旧号 8.38

- **原因对象**：运行时/执行 / vllm / piecewise CUDA graph 冷重放
- **原因行为**：引擎空闲后 piecewise CUDA graph 的首次重放以陈旧或未正确初始化的捕获状态执行该请求的 prefill，把 garbage 写入其 KV/indexer cache。
- **影响对象**：KV状态；输出内容（静默错误）
- **影响详情**：ROCm gfx942 + MoE AITER blockscale-FP8 路径，空闲 ≥20s 后首请求从第一个 token 起输出噪声且整请求不自愈，约 0.5s 后的下一请求正常；空闲 ≥50s 时复现率 50–100%，背靠背请求从不失败，无 crash。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/52150

### INF-0854 `vllm_flashinfer_cudagraph_support_target_heads_over_draft_kv_heads` 用 target 头数判 draft 组致 CUDA graph 全关

`故障` · `active` · `infra_off` · `partial` · `verified` · `NEW` · 发现 2026-09-06 · 旧号 8.70

- **原因对象**：运行时/执行 / vllm / get_cudagraph_support 判定
- **原因行为**：get_cudagraph_support() 对 draft 的 KV 组仍用 target 的每 rank 查询头数除以该组 num_kv_heads，未按 is_eagle_group 区分，头比算错判为不支持
- **影响对象**：性能/资源（性能退化）
- **影响详情**：spec decode 加入 draft KV 组且 target/draft 头比不同时（如 TP=3：target 66 头/draft 36 头），全引擎 CUDA graph 被静默关闭，输出正确但性能塌方；TP=2 同镜像正常捕获，难以归因
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/55581

### INF-0898 `sglang_session_mrope_table_partial_slice_crash` SGLang session 续聊 M-RoPE 位置表切片无防护致引擎崩溃

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-08-19 · 旧号 9.15

- **原因对象**：运行时/执行 / sglang / M-RoPE 位置表切片（fill_from）
- **原因行为**：session 续聊第二请求的 mm-input M-RoPE 位置表与静态 buffer 注册尺寸不一致，cuda_graph_buffer_registry.fill_from 对其做 partial slice 时没有边界防护。
- **影响对象**：可用性/进度（报错）
- **影响详情**：Qwen3.5-family 模型 session 第二请求（任意模态、CUDA graph 开或关）在 torch._foreach_copy_ 处抛 RuntimeError（1842 vs 1816），全部 TP scheduler 退出，在飞请求全部失败。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/35482

### INF-1044 `sglang_dspark_large_decode_cudagraph_nondeterministic_illegal_memory` SGLang DSpark 大 decode 图捕获随机非法访存

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-08-06 · 旧号 11.20

- **原因对象**：运行时/执行 / sglang / 大 decode CUDA Graph 捕获
- **原因行为**：大 decode CUDA-Graph capture 存在 capture 时序/内存竞态（强根因分析指向此处），导致非法内存访问。
- **影响对象**：可用性/进度（报错）
- **影响详情**：v0.5.16 + DSpark + TP8 启动时大 decode capture 以 CUDA illegal memory 或 host SIGSEGV 失败，失败 shape/rank/kernel 跨启动不同；跳过失败 shape 会迁移到更大 shape，H200/TP4 也复现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/33356

### INF-1075 `vllm_async_sched_hybrid_spec_update_states_postprocess_race` vLLM 异步调度+spec 下状态重写与 mamba 后处理竞态

`故障` · `active` · `infra_off` · `partial` · `verified` · `NEW` · 发现 2026-08-25 · 旧号 11.52

- **原因对象**：运行时/执行 / vllm / spec 输入准备与后处理同步
- **原因行为**：input prep 同步只等 forward 前的事件；_update_states 先于 _prepare_inputs 执行且无等待，改写上一步 postprocess_mamba_align_gpu 仍在读写的持久缓冲
- **影响对象**：KV状态（静默错误）
- **影响详情**：async scheduling + hybrid（Mamba align）+ spec decode 下，postprocess 读到半新半旧的 block table/状态索引，状态物化或计数出错，可发展为 cache 污染或崩溃；原文未给终端症状，窗口随并发批次增大
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/53613

### INF-1103 `vllm_v2_encoder_multi_budget_graphs_shared_pool_zero_init_nan_embeddings` vLLM encoder CUDA graph 共享池致大图 NaN

`故障` · `active` · `ambiguous` · `yes` · `verified` · `var` · 发现 2026-09-16 · 旧号 11.80

- **原因对象**：运行时/执行 / vllm / encoder CUDA graph 共享池
- **原因行为**：九档 token 预算的 encoder CUDA graph 共享同一 graph pool，varlen attention 输出缓冲未零初始化，各档图对其填充范围假设不一致，replay 大预算图时读到残留或未写区域。
- **影响对象**：隐状态；输出内容（数值异常）
- **影响详情**：开启 cudagraph_mm_encoder 时，1280×720 大图（1196 视觉 token、预算 2048）产出 NaN embedding 与错误回答，224×224 小图正常；只捕获单档、每档独立 pool 或零初始化输出均正确。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/57136
- **关联**：var:INF-1074

### INF-1113 `vllm_mrv2_lm_head_lora_request_level_mapping_misapplies_delta_to_wrong_logits_rows_spec` vLLM MRV2 lm_head LoRA 推测解码下按请求映射致错行

`故障` · `active` · `infra_off` · `partial` · `verified` · `var` · 发现 2026-09-29 · 旧号 11.90

- **原因对象**：运行时/执行 / vllm / MRV2 make_lora_inputs 映射
- **原因行为**：make_lora_inputs 的 LoRA 映射长度为 num_reqs，未按 cu_num_logits 把请求 id 重复到每请求 1+num_draft_tokens 个 logits 行，过短且对齐到前 num_reqs 行。
- **影响对象**：输出分布；跨请求/跨样本数据（静默错误）
- **影响详情**：MRV2 + 推测解码 + 针对 lm_head 的 LoRA 时，单请求批的 delta 只落在每步 row 0，混合批中 delta 打到其他请求的 logits；adapter 对本请求无效，行为随批内组成漂移；V1 runner 正确。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/59161
- **关联**：var:INF-1144

### INF-1145 `vllm_kda_chunked_scan_transient_buffers_outside_memory_profiling_oom` vLLM KDA 瞬态 buffer 漏计 profiling 致运行中 OOM

`故障` · `active` · `ambiguous` · `partial` · `documented` · `var` · 发现 2026-09-01 · 旧号 12.29

- **原因对象**：运行时/执行 / vllm / 启动期 memory profiling
- **原因行为**：启动 profiling 的 dummy 前向未覆盖 KDA chunked-scan 随 max-num-batched-tokens 线性增长的瞬态 buffer，KV cache 却按这部分内存空闲定容。
- **影响对象**：可用性/进度（报错）
- **影响详情**：GLM-5.3-Flash（MI350X MXFP4）在 --max-num-batched-tokens≥81920、16 并发 ~38k 请求时运行中 OOM，EngineCore 关闭，在途请求全部失败；8192–65536 稳定。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/54775
- **关联**：var:INF-0529

### INF-1153 `trtllm_prompt_embedding_async_h2d_outlives_pinned_pool_source_midcopy_overwrite` TRT-LLM pinned 池复用覆写在途 prompt 表拷贝

`故障` · `active` · `infra_off` · `partial` · `verified` · `NEW` · 发现 2026-09-15 · 旧号 12.37

- **原因对象**：运行时/执行 / tensorrt-llm / prompt 表 pinned 池异步 H2D
- **原因行为**：prompt embedding 表经 copyFrom() 异步 H2D 拷贝，host 源缓冲借自可复用 pinned pool；movePromptEmbeddingTableToGpu() 归还缓冲前未等待拷贝完成事件。
- **影响对象**：输入编码；输出内容（不可复现）
- **影响详情**：高并发且同 GPU 有竞争 CUDA context 时，新分配者可在拷贝进行中覆写同一 pinned 页，设备端收到半旧半新的 embedding 表，prompt-tuning 请求生成质量随损坏比例退化、无日志；低并发下几乎不可复现。1.1.0/1.2.1 均复现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/NVIDIA/TensorRT-LLM/issues/19172

### INF-1189 `vllm_gemma4_moe_v1_model_runner_mtp_dimension_mismatch` Gemma 4 MoE 回退 V1 后 MTP drafter 输入维度错误

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-08-10 · 旧号 13.21

- **原因对象**：运行时/执行 / vllm / Gemma4 MTP drafter 输入
- **原因行为**：MoE 变体回退 V1 runner，其路径给 MTP drafter 喂 backbone_hidden+draft_hidden（3840），而 pre_projection 需 2×backbone_hidden（5632）。
- **影响对象**：可用性/进度（报错）
- **影响详情**：以 MTP drafting 部署 Gemma 4 MoE（26B A4B）时，Dynamo fake tensor 调用抛 reduction dim 不匹配的 RuntimeError（[s47, 3840] × [5632, 1024]），engine 初始化失败。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/51493

### INF-1284 `cudagraph_padded_batch_stale_block_table` CUDA graph 填充行过期 block table 污染 KV

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 待定 · 旧号 K8.7

- **原因对象**：运行时/执行 / vllm / CUDA graph padding 输入准备
- **原因行为**：完整 CUDA graph 捕获把 block-table/slot_mapping 缓冲区地址烘焙进图，batch 被 pad 到已捕获尺寸，padding 行保留上次 replay 的过期 block-table 值，未指向保留哑页。
- **影响对象**：KV状态；跨请求/跨样本数据（静默错误）
- **影响详情**：paged attention 可能从存活序列的物理块读取甚至写入，污染同批存活请求的 KV，造成跨请求内容串扰；只在需要 pad 的特定 batch 尺寸下出现。
- **改进措施**：检测 待定；修复 待定
- **来源**：引用：vLLM CUDA Graphs 设计文档 + flex-nano-vllm 分析（机制已佐证；

### INF-1311 `cudagraph_fullcachehit_skip_kv_transfer_sync` vLLM CUDA graph 在完整缓存命中时跳过 KV 传输同步

`故障` · `active` · `infra_off` · `no` · `verified` · `NEW` · 发现 待定 · 旧号 K9.20

- **原因对象**：运行时/执行 / vllm / 逐层异步 KV 传输同步
- **原因行为**：完整缓存命中场景下 CUDA graph 重放跳过了 KV 传输同步，一个逐层异步 KV 操作未被等待，decode 读取尚未传输完成的 KV。
- **影响对象**：KV状态；输出内容（不可复现）
- **影响详情**：跨节点 KV 传输且完整缓存命中时偶发输出错误，损坏集中在命中前缀段、后缀正常；因是竞态，复现依赖时序，服务无报错。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/29608；https://github.com/vllm-project/vllm/pull/31057

## 集合通信

### INF-0103 `ttmetal_all_broadcast_writer_accessor_cross_buffer_space` all_broadcast 输出 accessor 误用输入 buffer

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-25 · 旧号 1.104

- **原因对象**：集合通信 / tt-metal / all_broadcast 写端地址生成
- **原因行为**：all_broadcast interleaved 路径的 writer 地址生成器从输入 tensor 的 buffer 构建，却用于解析输出 buffer；输入/输出分属 L1 与 DRAM 时 bank 解析落在错误地址空间。
- **影响对象**：数值结果（静默错误）
- **影响详情**：输入/输出分属不同存储空间（L1→DRAM 或 DRAM→L1）时 broadcast 数据静默损坏，PCC≈0.03；同空间组合正常，测试矩阵未覆盖跨空间组合。sharded 路径与 transfer size 的同源问题未修。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/pull/54229

### INF-0221 `vllm_rocm_custom_all_reduce_start_barrier_fully_relaxed_torn_reads_xgmi` vLLM ROCm 自定义 all-reduce 起点屏障全松弛致撕裂读

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-09-16 · 旧号 1.222

- **原因对象**：集合通信 / vllm / barrier_at_start（ROCm 分支）
- **原因行为**：ROCm barrier_at_start 的 flag store/load 均为 RELAXED，数据写读间无 release/acquire；带 release 的变体无内核调用，1stage/2stage 均用全松弛变体。
- **影响对象**：数值结果；输出内容（静默错误）
- **影响详情**：多 GPU ROCm 经 XGMI P2P（4×MI100 实测）时，消费者可读到对端仍在途的 staged buffer，撕裂 bf16 值引发间歇性单层激活损坏（NaN/巨大有限值）与长上下文垃圾 token；随负载时序变化、难复现，CUDA 路径不受影响。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/57059

### INF-0348 `moe_allreduce_topology_fp_reorder` TP all-reduce 累加顺序不同致 logprob 漂移

`故障` · `active` · `model_on` · `yes` · `verified` · `NEW` · 发现 待定 · 旧号 5.1

- **原因对象**：集合通信 / 通用 / TP all-reduce 累加顺序
- **原因行为**：TP 推理每层 all-reduce 中，NCCL reduce-scatter+all-gather 各 rank 累加顺序与自定义 Lamport/IPC 统一顺序不一致；浮点加法不结合，比特级差异逐层复合。
- **影响对象**：隐状态；输出分布；训练信号（静默错误）
- **影响详情**：相对参考实现出现 sub-1e-3 的 KL 漂移，输出仍连贯但逐 token logprob 不同，静默污染以 KL 为锚的 RLHF/GRPO reward；单机多卡 NCCL 可复现，量级通常低于批不变性噪声底。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://fireworks.ai/blog/when-faster-not-identical-moe-numerics

### INF-0351 `deepep_dispatch_imbalance_corruption` DeepEP 负载不均时计数翻倍致 token 错发 expert

`故障` · `active` · `infra_off` · `no` · `verified` · `NEW` · 发现 待定 · 旧号 5.4

- **原因对象**：集合通信 / deepep / low-latency dispatch 计数
- **原因行为**：low-latency dispatch 在专家负载严重不均且 token 数较大时，NVSHMEM ibgda 对 per-expert token count 的 atomic-fetch-add 结果被翻倍/三倍送达。
- **影响对象**：隐状态；输出内容（静默错误）
- **影响详情**：token 被 dispatch 到或从错误 expert combine，MoE 输出被破坏；仅在特定不均衡度与 token 数组合下间歇触发，需跨节点/RDMA 拓扑。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/deepseek-ai/DeepEP/issues/427

### INF-0367 `vllm_custom_allreduce_cross_node_ipc_handle_abort` vLLM CustomAllreduce 误准入跨节点组致 abort

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-08-18 · 旧号 5.20

- **原因对象**：集合通信 / vllm / CustomAllreduce 启用准入
- **原因行为**：CustomAllreduce 未校验组内 rank 是否在同一节点就对跨节点 TP 组启用，并用 cudaIpcOpenMemHandle 打开仅在单节点内有效的 peer IPC handle。
- **影响对象**：可用性/进度（报错）
- **影响详情**：GB200 NVL72（MNNVL）双节点每节点 1 GPU、Ray TP=2 时返回 invalid resource handle，CUDACHECK 内 exit 直接杀进程，启动期双 rank abort 且无 Python traceback。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/52778

### INF-0372 `sglang_symm_mem_pool_growth_nccl_window_register_silent_deadlock` SGLang symm-mem 池扩容卡在 NCCL 注册致静默死锁

`故障` · `active` · `infra_off` · `待定` · `documented` · `NEW` · 发现 2026-08-29 · 旧号 5.25

- **原因对象**：集合通信 / sglang / symm-mem 池扩容的 window 注册
- **原因行为**：--enable-symm-mem 下对称内存池达容量后运行时扩容，进入 ncclCommWindowRegister 注册路径并阻塞在无超时的 UDS recvmsg，依赖注册的调度循环整体卡死。
- **影响对象**：可用性/进度（挂起）
- **影响详情**：负载持续到池扩容时全部 TP scheduler 静默 wedge，无 traceback、无 watchdog、无日志；启动期注册正常，冒烟测试通过而生产长跑命中，无推理探针的健康检查仍全绿。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/36943

### INF-0376 `vllm_dp8_p2p_tier_secondary_nixl_ucx_agent_startup_hang` vLLM DP8 二级 tier NIXL/UCX agent 构造挂起

`故障` · `active` · `infra_off` · `partial` · `documented` · `var` · 发现 2026-09-02 · 旧号 5.29

- **原因对象**：集合通信 / vllm / 二级 tier NixlTransport 构造
- **原因行为**：TieringOffloadingSpec 的 p2p 二级 tier 为每个 engine-core 进程单独构造 NixlTransport agent，某个 rank 上的 UCX 初始化/内存注册路径无限阻塞，且无超时。
- **影响对象**：可用性/进度（挂起）
- **影响详情**：fresh pod 上 DP8（8×H200、TP1+EP、MultiConnector）启动时 primary NIXL agent 均成功，某二级 agent 卡在 UCX 实例化之前，serving port 不开、pod 永不 Ready，8 卡被占而闲置。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/55000
- **关联**：var:INF-0375

### INF-0386 `vllm_sparse_nccl_weight_patch_noncontiguous_view_stride_ignored` vLLM 稀疏 NCCL 权重补丁忽略非连续 view stride

`故障` · `active` · `infra_off` · `partial` · `verified` · `var` · 发现 2026-09-16 · 旧号 5.39

- **原因对象**：集合通信 / vllm / PyNccl broadcast 封装
- **原因行为**：trainer 把 SparseWeightPatch 的 indices/values 直接交给 PyNccl broadcast；遇非连续 view 时按 data_ptr()+numel 发送相邻存储元素，不应用 stride。
- **影响对象**：模型权重；输出内容（静默错误）
- **影响详情**：indices/values 为非连续 view（如 stride-2 切片）时，推理 worker 静默加载索引与值都错位的补丁（期望 [10,-1,20,-1]，实得 [10,11,-1,-1]），无报错，在线更新后模型质量退化；发送侧正确，难定位。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/57130
- **关联**：var:INF-0385

### INF-0557 `vllm_envs_typechecking_default_runtime_split_hybrid_mode_off` vLLM DeepEP hybrid mode env 声明默认与运行时默认分裂

`故障` · `active` · `ambiguous` · `config` · `verified` · `var` · 发现 2026-08-29 · 旧号 6.164

- **原因对象**：集合通信 / vllm / DeepEP 混合模式 env 默认值
- **原因行为**：envs.py 中 DeepEP hybrid mode 环境变量在 TYPE_CHECKING 块声明默认 True，运行时 lambda 却默认 "0" 解析为 False 并传给 DeepEP，偏离 DeepEP 上游默认 True。
- **影响对象**：性能/资源（性能退化）
- **影响详情**：多机 EP 部署未显式设置该 env 时静默以 direct mode 运行，而非 hierarchical RDMA+NVLink 路径，多机多轨网络下性能/行为不同，无 crash 无告警；查阅类型声明或文档会得到相反结论。显式设为 1 可规避。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/54281
- **关联**：var:INF-0412

### INF-0741 `vllm_rocm_quickreduce_bf16_to_fp16_no_saturation_inf_poison` QuickReduce bf16 转 fp16 不饱和致归约出 inf

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-09-04 · 旧号 7.68

- **原因对象**：集合通信 / vllm / QuickReduce bf16→fp16 转换
- **原因行为**：quick_reduce_impl.cuh 在 pack 前用 __float22half2_rn 把 bf16 收窄为 fp16 且不饱和，超 65504 的值静默变 inf；位于所有 codec 公共路径，无数值检查。
- **影响对象**：数值结果；隐状态（数值异常）
- **影响详情**：MI355X 上 TP2/TP4、bf16 激活超出 ±65504 时，单个超限元素即让各 rank 的归约输出含 inf，并沿残差流污染整层/整序列；无 crash、无 warning，下游 NaN/乱码易被误归因于模型或量化。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/55345

### INF-0810 `sglang_deterministic_allreduce_tp_gt1_nondeterministic` SGLang 确定性模式 TP>1 时 NCCL channel 数致结果不确定

`故障` · `active` · `ambiguous` · `partial` · `documented` · `NEW` · 发现 2026-08-09 · 旧号 8.26

- **原因对象**：集合通信 / sglang / 确定性模式 NCCL all-reduce 配置
- **原因行为**：确定性推理只把 NCCL 算法 pin 为 allreduce:tree，未固定 channel 数；NCCL 按消息大小选 channel 数，同一元素的归约顺序随 batch 中共享 token 数变化。
- **影响对象**：输出分布（不可复现）
- **影响详情**：--enable-deterministic-inference 且 tp>1 时，同一 token 的 prefill 与 decode logprob 不一致（KL 达 0.15–0.18），违反确定性承诺；pin channel 数后 Inkling-Small-NVFP4 tp=4 逐位一致
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/pull/34159

### INF-1257 `mooncake_rdma_mr_silent_truncation` Mooncake 内存注册超 max_mr_size 被截断致 KV 部分传输

`故障` · `active` · `infra_off` · `no` · `documented` · `NEW` · 发现 待定 · 旧号 K5.1

- **原因对象**：集合通信 / mooncake / registerLocalMemory 内存注册
- **原因行为**：registerLocalMemory() 以超过网卡 max_mr_size 的长度调用时只注册 max_mr_size 字节，却让 BufferDesc.length 继续宣告原始大小。
- **影响对象**：KV状态；输出内容（静默错误）
- **影响详情**：在 MR 上限较小的网卡（如 AMD Pensando 2 GiB）上，大于上限的逐层 KV 缓冲区只被部分传输，越界 RDMA 读写命中 IBV_WC_REM_ACCESS_ERR；消费端拿到被截断的上下文，仍能生成通顺文本。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/kvcache-ai/Mooncake/issues/2017

### INF-1261 `dynamo_nixl_async_kv_overwrite_after_recv` Dynamo 自定义 NIXL 传输在接收后 KV 被覆写

`故障` · `active` · `infra_off` · `no` · `documented` · `var` · 发现 待定 · 旧号 K5.5

- **原因对象**：集合通信 / dynamo / 自定义 NIXL/NCCL KV 传输后端
- **原因行为**：decode 端接收 KV 之后远端 KV 内存仍被改动，疑为描述符/通知顺序问题或缓冲区生命周期竞态（写入落地前接收即被标记完成）；issue 被定性为自定义后端用法错误，根因未解决。
- **影响对象**：KV状态；输出内容（不可复现）
- **影响详情**：decode 使用的最终 KV 与 prefill 发出的不一致，表现为与传输时序相关、不可复现的输出偏差（原文按机制推断）。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/ai-dynamo/dynamo/issues/1107
- **关联**：var:INF-0870

## 并行切分

### INF-0353 `distributed_seed_not_broadcast` 分布式 seed 未广播或 TP 切分错误致采样不一致与输出错乱

`故障` · `active` · `infra_off` · `yes` · `documented` · `NEW` · 发现 待定 · 旧号 5.6

- **原因对象**：并行切分 / 通用 / 跨 rank 采样 RNG seed 广播；并行切分 / 通用 / TP 张量维度切分
- **原因行为**：PP/MoE 场景未把 RNG seed 广播到各 rank，随机采样在 rank/stage 间不一致；另一变体是 TP 对某张量维度切分错误，all-reduce 求和了错误切片。
- **影响对象**：输出内容；隐状态（不可复现）
- **影响详情**：seed 变体隐状态健康、仅各 rank 采样结果不同，需输出级跨 rank 对拍才能发现；切分变体 all-reduce 求和错误切片，输出为垃圾。多进程初始化即可构造。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://arxiv.org/abs/2506.09713

### INF-0356 `sglang_dsv4_nonep_tbo_attn_tp_replica_misshard` SGLang DSV4 TBO 误把 attention 副本当分片

`故障` · `active` · `infra_off` · `other_stack` · `verified` · `NEW` · 发现 2026-08-02 · 旧号 5.9

- **原因对象**：并行切分 / sglang / TBO 全 TP 变长 collective
- **原因行为**：DP attention + moe_a2a_backend=none + attention TP>1 时，TBO 把全 TP 变长 collective 应用于 attention-TP 组内已复制的张量，把副本当作独立分片处理。
- **影响对象**：输出内容；隐状态（静默错误）
- **影响详情**：DeepSeek-V4 non-EP TBO 在上述组合下 gather 与 combine 布局损坏，产生非法模型输出，无 crash、无 NaN。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/pull/33250

### INF-0357 `sglang_minimax_h3_fsdp_inference_silent_av_corruption` SGLang FSDP 推理服务 MiniMax-H3 静默损坏音视频

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-08-11 · 旧号 5.10

- **原因对象**：并行切分 / sglang / --use-fsdp-inference 推理路径
- **原因行为**：--use-fsdp-inference 启用的 FSDP 推理路径未对 diffusion/多模态模型验证，与 MiniMax-H3（fl2va）数据流不兼容；具体出错代码原文未给出。
- **影响对象**：输出内容（静默错误）
- **影响详情**：8×H20 服务 MiniMax-H3 时开启该 flag，视频帧被压成暗色低对比图像、音频为噪声，日志无错误或警告；去掉 flag 同请求输出正确，帧灰度统计显著偏离。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/34227

### INF-0371 `sglang_mhc_communicator_missing_dp_reduce_scatterv_pair` SGLang MHC communicator 缺 DP 归约分支致输出损坏

`故障` · `active` · `infra_off` · `other_stack` · `verified` · `NEW` · 发现 2026-08-29 · 旧号 5.24

- **原因对象**：并行切分 / sglang / MHC communicator 的归约分支
- **原因行为**：MHC communicator 从 PR #24785 前的快照改编、未回映上游修复：scatter 缺 MoE partial combine，should_use_reduce_scatter 仍返回 False。
- **影响对象**：隐状态；输出内容（静默错误）
- **影响详情**：dp-attention 且 TP=DP=EP 时 SUM_LEN 轮每 token 只得 1/4 routed experts，输出 "A: A: A:" 式重复坍缩；只修前者则 dense MLP 输出 ×4，流畅但 tool-call schema 仅 49/63。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/36879

### INF-0373 `sglang_gptoss_dp_attention_double_reduction_allreduce_scatterv` SGLang GPT-OSS DP attention 双重归约致准确率塌零

`故障` · `active` · `infra_off` · `other_stack` · `verified` · `var` · 发现 2026-08-31 · 旧号 5.26

- **原因对象**：并行切分 / sglang / GptOssSparseMoeBlock 归约
- **原因行为**：forward_normal 先做 TP all-reduce 使各 rank 持完整 expert 和，DP-attention 后处理又对同一 TP 组 reduce_scatterv 再求和，结果约为 TP 倍。
- **影响对象**：隐状态；输出内容（质量退化）
- **影响详情**：GPT-OSS-120B MXFP4 8 卡开 dp-attention 时激活整体放大 4 倍（TP4），输出充满重复 Harmony channel marker、空答案、打满 8192 token，GSM8K 124/128→0/128、MMLU 110/128→4/128，HTTP 层正常。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/37187
- **关联**：var:INF-0371

### INF-0381 `sglang_minicpm_lightning_attn_tp_partial_rms_norm_denominator` SGLang MiniCPM TP 下 RMS 归一化分母只算本 rank

`故障` · `active` · `infra_off` · `other_stack` · `verified` · `var` · 发现 2026-09-06 · 旧号 5.34

- **原因对象**：并行切分 / sglang / Lightning attn 的 RMS 归一化
- **原因行为**：Lightning attention 层对全头组合输出做 RMS 归一化，TP 下各 rank 只持头子集，却用本 rank 子集计算归一化分母，未跨 TP rank 归约均方统计。
- **影响对象**：隐状态（静默错误）
- **影响详情**：TP>1 且各 rank 激活量级不同时输出偏离全宽归一化：TP2 MiniCPM-SALA NVFP4 激活对比 FP32 参考，prefill 最大绝对误差 2.3125、decode 1.1875，无 crash；TP=1 不受影响，小规模验证不显形。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/pull/38230
- **关联**：var:INF-0373

### INF-0383 `sglang_hyv4_a2a_backend_attn_tp_gather_moe_full_width_token_mismatch` SGLang HYV4 deepep 下 MoE 静默只算部分 token

`故障` · `active` · `infra_off` · `other_stack` · `verified` · `NEW` · 发现 2026-09-09 · 旧号 5.36

- **原因对象**：并行切分 / sglang / require_attn_tp_gather()
- **原因行为**：require_attn_tp_gather() 仅凭设置了 a2a backend 就把 num_token_non_padded 按 attn-TP 分片推导，而 HYV4 无 LayerCommunicator、按全宽调 MoE。
- **影响对象**：隐状态；输出分布（静默错误）
- **影响详情**：HYV4 + deepep a2a + attn_tp_size>1（无 DP attention）时，超出分片宽度的行 topk_ids=-1、部分 EP rank routed_out 为 0；输出流畅、无报错、吞吐反升 16.7%，仅 logprob 对比可发现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/38606

### INF-0393 `sglang_dit_tp2_modulation_norm_sharded_as_ordinary_ffn_chroma_speckle` SGLang DiT 在 TP=2 下图像出现 speckle 噪点

`故障` · `active` · `ambiguous` · `other_stack` · `documented` · `NEW` · 发现 2026-09-25 · 旧号 5.46

- **原因对象**：并行切分 / sglang / DiT 调制/QK-norm 的 TP 切分
- **原因行为**：待定（报告者假设）：Qwen-Image-2.1 单流 DiT 的 AdaLN 调制输出或 per-head QK-norm 状态被按普通 FFN 语义切分/all-reduce，部分和求和造成轻度保结构损坏；精确张量位点未定位。
- **影响对象**：输出内容；隐状态（质量退化）
- **影响详情**：--tp-size 2 时输出图像被致密多色 speckle 噪点覆盖（平滑区尤甚）、构图正确，TP=1 干净；vLLM-Omni 独立 TP 实现 TP=2 同样损坏。已排除 flash-attn、rank 互换、NCCL 精度与 offloader；无 crash。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/41192

### INF-0459 `sglang_cp_roundrobin_single_row_inflated_pitch_memory_fault` SGLang CP 单行 shard 保留膨胀 pitch 致访存越界

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-08-09 · 旧号 6.66

- **原因对象**：并行切分 / sglang / CP round-robin 切分函数
- **原因行为**：CP round-robin 切分对 view 切片调 .contiguous()；chunk 恰为 cp_size token 时 shard 仅 1 行，被判为连续不拷贝，保留 cp_size 倍膨胀的 row pitch。
- **影响对象**：可用性/进度；数值结果（报错）
- **影响详情**：DSV4 CP prefill 且 chunk 恰为 cp_size token 时触发；下游按 stride(0) 取 pitch 的 aiter mhc_pre 越界寻址，报 Memory access fault by GPU；越界地址恰好已映射时可能静默损坏。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/pull/34142

## KV缓存

### INF-0055 `ttmetal_update_cache_batch_gt32_tile_padding_overwrite` update_cache 在 batch>32 且非 32 倍数时覆写用户行

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-08-09 · 旧号 1.55

- **原因对象**：KV缓存 / tt-metal / update_cache padding 循环边界
- **原因行为**：update_cache 多核 factory 的 tile padding 循环边界对 input 与 cache 的 padded_shape 处理不一致，把输入的 tile padding 写到真实用户行上。
- **影响对象**：KV状态（不可复现）
- **影响详情**：缓存 batch Bcache>32 且非 32 倍数时受影响用户行静默变为零，无警告；部分尺寸非确定，同输入每次 dispatch 损坏行数不同、偶尔完全正确（Bcache=33 时损坏 0–31 行），测试极易漏检；Bcache=32/64/96 正确。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/52671

### INF-0096 `ttmetal_layerack_enqueue_time_semantics_kv_stream_race` gpt-oss 逐层 ack 早于 KV 写完成，逐层迁移会读到半写 cache

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-20 · 旧号 1.97

- **原因对象**：KV缓存 / tt-metal / gpt-oss 逐层 ack 与 KV 迁移
- **原因行为**：GPT-OSS 的 per-layer ack（layer_ack_channel.inject）在 host 完成层 enqueue 后立即注入，无任何 device 同步；且从不调用 zero_padded_kv_cache。
- **影响对象**：KV状态；输出内容（静默错误）
- **影响详情**：当前全序列化迁移下无害；启用 ack 驱动的逐层 streaming 迁移（PREFILL_MIGRATION_INTERLEAVED=1）后 decode 读到部分写入的 KV，迁移 PCC 校验失败，无报错；迁移尾部还携带陈旧 pad 数据。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/53815

### INF-0127 `vllm_gdn_triton_fp32_ssm_readout_fp16_store_overflow_collapse` vLLM GDN 读出按 fp16 物化，溢出致重复 token

`故障` · `active` · `infra_off` · `partial` · `verified` · `NEW` · 发现 2026-08-30 · 旧号 1.128

- **原因对象**：KV缓存 / vllm / GDN Triton 内核读出输出 dtype
- **原因行为**：fused_sigmoid_gating_delta_rule_update 读出 o=state@q 以 fp32 累加，输出却按激活 dtype 分配（q.new_empty 继承 fp16）；chunk/prefill 内核同此模式。
- **影响对象**：数值结果；输出内容（数值异常）
- **影响详情**：fp32 SSM cache 且走 Triton 路径（非 Blackwell）时，长对话状态增长使读出超 fp16 上限 65504、store 溢出为 inf，经 RMSNormGated 变 NaN，decode 坍缩为单 token 无限重复（“!!!!”）；短 prompt 正常。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/54308

### INF-0339 `vllm_hybrid_mamba2_prefix_cache_all_mode_chunked_prefill_cross_call_nondeterminism` vLLM 混合 Mamba2 all 缓存模式下分块 prefill 非确定

`故障` · `active` · `infra_off` · `config` · `verified` · `var` · 发现 2026-09-17 · 旧号 4.23

- **原因对象**：KV缓存 / vllm / Mamba 状态 checkpoint 写入/恢复
- **原因行为**：mamba_cache_mode=all 的状态 checkpoint 写入/恢复与跨 step 的 chunked prefill 分割交互存在未定序路径，疑为状态恢复读取序或 checkpoint 绑定序非确定。
- **影响对象**：KV状态；输出内容（不可复现）
- **影响详情**：granite-4.0-h-tiny、Falcon-H1-Tiny 同序 8 prompt 四次 greedy generate 跨调用分叉（prompt 1 第 5 token 即 16 vs 24）；需 all 模式、跨步分块且 ≥3 并发序列，align/none/不分块均恒等。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/57267
- **关联**：var:sglang_vlm_temp0_cross_build_near_tie_flip_worker_nondeterminism

### INF-0358 `vllm_cudahostregister_mixed_pinned_unpinned_tp_ranks` vLLM KV offload 各 rank 独立 pin 致混合状态

`故障` · `active` · `ambiguous` · `partial` · `documented` · `NEW` · 发现 2026-08-11 · 旧号 5.11

- **原因对象**：KV缓存 / vllm / KV offload 的 pinned 注册
- **原因行为**：CPU KV offload 各 TP rank 独立对共享 mmap 调用 cudaHostRegister，失败时本地 warn-and-continue，缺少全 rank 协调，可形成 pinned/unpinned 混合状态。
- **影响对象**：性能/资源（性能退化）
- **影响详情**：TP=2 主机重复启动时注册状态随启动变化；混合启动下同步传输被 unpinned rank 限速，pinned rank 仍付页表 bookkeeping 成本（48 GiB 区域每 GPU 96 MiB PTE），无 crash。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/51762

### INF-0360 `hf_nemotron_h_multigpu_devicemap_silent_wrong_output` HF NemotronH 多卡 device_map 下静默近随机输出

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-08-13 · 旧号 5.13

- **原因对象**：KV缓存 / transformers / hybrid scan cache 跨设备传递；模型定义 / transformers / NemotronH 状态设备放置
- **原因行为**：use_cache=True 时 hybrid scan cache 跨设备传递 SSM 状态不一致；部分 device_map 切分边界切断了须留在同一设备的 Mamba/latent 状态。
- **影响对象**：KV状态；输出内容（质量退化）
- **影响详情**：Nemotron-3.5-Lightning-30B 用 device_map="auto" 多卡时无 crash、无 warning，输出近随机（loss≈10）；use_cache=False 时也仅特定切分正确（2 卡对、3/4 卡错）。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/huggingface/transformers/issues/47942

### INF-0364 `vllm_mooncake_hetero_tp_replicated_gqa_head_transfer_gap` vLLM Mooncake 异构 TP 只传 25% KV 却标记就绪

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-08-16 · 旧号 5.17

- **原因对象**：KV缓存 / vllm / Mooncake 异构 TP 传输规划器
- **原因行为**：规划器把 replicated 当作每个 rank 持整块 KV；num_kv_heads < producer_tp_size 时只传 rank 0 的一个 head 切片，仍按成功回执标记 KV-ready。
- **影响对象**：KV状态；输出内容（静默错误）
- **影响详情**：PD 分离 + 异构 TP（如 TP8 prefill→TP1 decode、4 个 KV head）时 75% 字节未传，decode 基于缺 3/4 KV head 的 cache 生成静默错误输出，无 crash 无报错、传输层显示成功；Qwen3-235B-A22B-FP8 官方模型可复现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/52516

### INF-0369 `sglang_mooncake_pd_mamba_ssm_dtype_item_length_mismatch_state_corruption` SGLang PD 两端 SSM dtype 不一致致状态静默错位

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-08-27 · 旧号 5.22

- **原因对象**：KV缓存 / sglang / Mooncake PD SSM state 传输
- **原因行为**：PD 两端各自解析 Mamba SSM state dtype，传输层用源端 item length 同时定拷贝长度和目的 slot stride 且无校验；两端 dtype 不同时目的 state 布局错位。
- **影响对象**：KV状态；输出内容（质量退化）
- **影响详情**：Kimi-K3 + ReplaySSM 部署中 prefill 默认 BF16、decode 开 --enable-linear-replayssm-spec 强制 FP32 时无错误、HTTP 200，多轮/长上下文输出质量严重退化，短输出可能看似正常；两端对齐 dtype 即恢复。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/36371

### INF-0370 `sglang_dflash_mamba_state_cache_silent_concurrency_cap_misread_as_bandwidth` SGLang DFLASH 下 Mamba 状态池静默把并发钳到 2

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-08-28 · 旧号 5.23

- **原因对象**：KV缓存 / sglang / Mamba state 池容量与并发钳制
- **原因行为**：hybrid KDA + DFLASH 下每请求需 1+D 个 Mamba state slot，内存紧张时推导的 max_mamba_cache_size 过小，调度器据此静默压低有效并发，仅打 info 日志。
- **影响对象**：性能/资源；指标/日志（性能退化）
- **影响详情**：GLM-5.3-Flash + DFLASH（D=5）在 2×DGX Spark TP2 上 --max-running-requests 8 被接受、/get_server_info 报 8，实际并发仅 2；c8 吞吐 47.3 vs 修正后 80.3 tok/s，平台期易被误读为带宽上限。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/36889

### INF-0377 `vllm_heterogeneous_tp_mamba2_conv_state_corruption_disagg` vLLM 异构 TP 分离式 Mamba2 conv state 传输损坏

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-09-04 · 旧号 5.30

- **原因对象**：KV缓存 / vllm / 异构 TP 的 conv state 传输
- **原因行为**：异构 TP 传输假设所有状态张量都按 token 维可切，而 conv state 沿 channel 维分布；跨 TP 度传输时未按目标侧拓扑重分片或重排，decode 实例拿到语义错位的 state。
- **影响对象**：KV状态；输出内容（质量退化）
- **影响详情**：prefill/decode TP 度不同的 disaggregated serving 运行 Mamba2 模型时，decode 侧状态与 prefill 不一致，从 P/D 切换点起输出质量静默下降，无 crash、无 NaN；仅回到同构 TP 或直连模式才可对照确认。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/55208

### INF-0380 `sglang_dspark_draft_kv_pool_budget_tp_size_vs_attn_tp_size` SGLang draft KV 池按 tp_size 少计预留致 OOM

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `var` · 发现 2026-09-06 · 旧号 5.33

- **原因对象**：KV缓存 / sglang / draft KV 池预留预算
- **原因行为**：DFLASH/DSPARK draft KV 池的预留按原始 tp_size 计算，而 draft 池实际按 attn_tp_size 分片；DP attention 下两者相差 attn_dp_size 倍，预留被少计。
- **影响对象**：可用性/进度（报错）
- **影响详情**：Kimi-K3 PD 解码节点开 DP attention + DSPARK 时，target KV 池已吃掉几乎全部 --mem-fraction-static 预算，draft worker 在 _build_mha_kv_pool 再分配 3.50 GiB 即启动期 OOM。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/38202
- **关联**：var:INF-0529

### INF-0384 `ttmetal_kv_chunk_address_table_proto_import_sorts_config_ids_by_name` tt-metal KV 地址表导入按名字重排致 KV 错位

`故障` · `active` · `infra_off` · `other_stack` · `verified` · `NEW` · 发现 2026-09-11 · 旧号 5.37

- **原因对象**：KV缓存 / tt-metal / KvChunkAddressTable 导入/配对
- **原因行为**：from_proto_message 经名字键构造器按 std::map 字典序重排 config id，未保留序列化顺序；发送端按下标配对两端 config，只校验数量不校验名字。
- **影响对象**：KV状态（静默错误）
- **影响详情**：两端 config 命名排序不同（如 MiniMax-M3 prefill 用 "0".."8"、decode 用 k_h0..index_k）时 PD 迁移后每个 config 错位一格，decode 读错 head；无报错，loopback 自检全过，仅跨端 golden PCC 可见。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/56263

### INF-0394 `vllm_apc_dirty_block_hash` vLLM APC 非满尾块 hash 失真致复用脏 KV

`故障` · `active` · `ambiguous` · `config` · `verified` · `NEW` · 发现 待定 · 旧号 6.1

- **原因对象**：KV缓存 / vllm / APC 前缀缓存非满尾块 hash
- **原因行为**：APC 以 token ID 的 hash 作为 KV block 的 key，但非满尾块的 hash 未反映真实内容，hash 到 token 的映射出错，使后续请求命中并复用过期/脏 KV。
- **影响对象**：KV状态；跨请求/跨样本数据；输出内容（不可复现）
- **影响详情**：开启 enable_prefix_caching 时，greedy（temp=0）补全在多次运行之间、以及与关闭缓存的服务之间结果不同，评测分数下降，不崩溃；TinyLlama 上可复现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://arxiv.org/abs/2506.09713

### INF-0395 `vllm_multimodal_cachekey_collision` vLLM 多模态前缀缓存 key 忽略图像且缺租户隔离

`故障` · `active` · `model_on` · `partial` · `verified` · `NEW` · 发现 待定 · 旧号 6.2

- **原因对象**：KV缓存 / vllm / 前缀缓存 key 与租户隔离
- **原因行为**：prefix/radix cache 的 key 只 hash 文本 token、忽略多模态输入（或用截断 SHA256 + NCHF），且 radix/APC 缺少租户隔离，相同文本 + 不同图片/租户共用 key 而复用错误 KV。
- **影响对象**：KV状态；跨请求/跨样本数据（静默错误）
- **影响详情**：多模态或共享前缀负载下静默、间歇地产生「错图/错租户」输出；文本 token 的隐状态计算本身正确，只是 cross-attention 读到错误的图像或序列 KV；缺少租户隔离还构成跨租户侧信道。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/20261

### INF-0399 `radix_kvcache_cross_contamination` SGLang radix 缓存在 prefill/decode 重叠时跨序列污染

`故障` · `active` · `infra_off` · `partial` · `documented` · `var` · 发现 待定 · 旧号 6.6

- **原因对象**：KV缓存 / sglang / radix 缓存 block 边界
- **原因行为**：prefill 与 decode 重叠执行（overlap prefill+decode）时，radix 缓存在 block 边界处发生跨序列 KV 污染。
- **影响对象**：KV状态；跨请求/跨样本数据（静默错误）
- **影响详情**：开启 overlap prefill+decode 的并发服务中，attention 读到另一个序列的 K/V，输出静默发散；污染覆盖整个 block 而非单点。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/22819；https://github.com/vllm-project/vllm/issues/39146
- **关联**：var:INF-0870

### INF-0402 `vllm_pd_preemption_race_empty_kv` vLLM P/D 抢占竞态使 connector 向 decode 传输空 KV

`故障` · `active` · `infra_off` · `no` · `documented` · `var` · 发现 2026-07-29 · 旧号 6.9

- **原因对象**：KV缓存 / vllm / 异步 KV connector 传输取消
- **原因行为**：请求被 preempt 时 KV blocks 已被 purge，但异步 connector 的 request_finished() 看到空 block 列表仍继续传输、未取消；竞态窗口在最终步 prefill 完成与传输取消之间。
- **影响对象**：KV状态；输出内容（静默错误）
- **影响详情**：P/D 分离部署下请求被抢占时，decode 节点收到空 KV 或残留元数据，从空/过期 KV 开始解码，输出垃圾或无关内容；无 crash。单进程路径无法复现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/50297
- **关联**：var:INF-0399

### INF-0403 `vllm_block_table_width_mismatch` vLLM 块表宽度在多路径间分叉致 DSA indexer 缓冲不匹配

`故障` · `active` · `infra_off` · `partial` · `documented` · `var` · 发现 2026-07-29 · 旧号 6.10

- **原因对象**：KV缓存 / vllm / DSA indexer 块表宽度
- **原因行为**：块表（block table）宽度在多个路径间分叉，MRV1/MRV2/DSA indexer 各用不同的宽度假设，在对齐 + DCP 下造成 indexer buffer mismatch。
- **影响对象**：KV状态（静默错误）
- **影响详情**：DSA（DeepSeek Attention）/indexer 路径在对齐 + DCP 下 KV 块索引错位，attention 读取错误的 KV block。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/50302
- **关联**：var:INF-1238

### INF-0404 `vllm_mamba_dcp_block_table_not_preserved` vLLM Mamba 混合架构块表容量在 DCP 下未保留

`故障` · `active` · `infra_off` · `partial` · `documented` · `var` · 发现 2026-07-29 · 旧号 6.11

- **原因对象**：KV缓存 / vllm / Mamba 层块表容量
- **原因行为**：混合架构（Mamba）的块表容量在 DCP（Decoder Checkpoint 恢复）下未被保留，检查点恢复后 Mamba 块表状态不一致。
- **影响对象**：KV状态；输出内容（质量退化）
- **影响详情**：DCP 恢复后 Mamba 层的状态缓存（block table）被截断或不一致，decode 质量退化或输出垃圾。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/50287
- **关联**：var:INF-1174

### INF-0406 `vllm_pd_mooncake_client_disconnect_orphan_kv` vLLM P/D Mooncake 断连未通知 P 致孤儿 KV 堆积

`故障` · `active` · `infra_off` · `no` · `documented` · `NEW` · 发现 2026-07-30 · 旧号 6.13

- **原因对象**：KV缓存 / vllm / MooncakeConnector 取消传播
- **原因行为**：客户端断连或网关超时在请求提交到 D 节点 EngineCore 前取消请求时，取消不传播到 P；P 侧为 D 保留的 KV blocks 一直占用，直到 VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT 过期。
- **影响对象**：性能/资源；可用性/进度（挂起）
- **影响详情**：P/D 分离部署持续流量、高并发下孤儿 pending transfer 累积，耗尽 P 节点 KV cache，后续请求 stall/超时；无 crash。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/50425

### INF-0416 `sglang_hicache_storage_key_missing_kv_dtype` SGLang HiCache 存储键缺 KV dtype 致跨运行复用错格式缓存

`故障` · `active` · `ambiguous` · `other_stack` · `verified` · `NEW` · 发现 2026-08-02 · 旧号 6.23

- **原因对象**：KV缓存 / sglang / HiCacheStorageConfig 存储键
- **原因行为**：HiCache 持久化存储键只含模型名与 TP/PP/CP 几何，不含 KV cache dtype；根因是 HiCacheStorageConfig 本身没有 dtype 字段，任何后端都无法把 dtype 纳入键。
- **影响对象**：KV状态；输出内容（静默错误）
- **影响详情**：同一模型先后以不同 --kv-cache-dtype（如先 bf16 再 fp8_e4m3）运行时键相同，第二次运行静默复用字节格式错误的缓存条目，attention 读到格式错误的 KV，输出退化；无 crash、无 NaN、无报错。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/33268

### INF-0417 `vllm_pd_nixlpush_failed_write_notification_undercount` vLLM PD push transfer 启动失败未通知致请求挂起

`故障` · `active` · `infra_off` · `no` · `verified` · `var` · 发现 2026-08-02 · 旧号 6.24

- **原因对象**：KV缓存 / vllm / NIXL push connector 通知计数
- **原因行为**：push 模式下 decode 节点靠计数到达的 notification 确认收齐 KV，但 prefill 节点无法启动某个 transfer 时只记日志后继续，不通知 decode 节点，计数无法匹配。
- **影响对象**：可用性/进度；KV状态（挂起）
- **影响详情**：PD 分离 push 模式下任一 transfer 启动失败，decode 节点等待永远到不了的计数，请求挂到 lease 超时，期间持有覆盖半写缓存的 block；无 crash。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/50754
- **关联**：var:INF-1265

### INF-0419 `vllm_kimi_k3_packed_kda_prefill_nan_cross_request_infection` Kimi-K3 packed KDA prefill NaN 跨请求永久感染

`故障` · `active` · `infra_off` · `yes` · `documented` · `NEW` · 发现 2026-08-04 · 旧号 6.26

- **原因对象**：KV缓存 / vllm / kda.py packed prefill 分支
- **原因行为**：疑似 kda.py:658：batch 含 prefill 时把全部请求（含 decode）打包进单次 prefill launch（共享 cu_seqlens），NaN 请求经共享 launch 与 state 写回感染其他请求。
- **影响对象**：跨请求/跨样本数据；输出分布；输出内容（数值异常）
- **影响详情**：vLLM 0.26.0 服务 Kimi-K3，约 240K 长上下文预填充后，所有后续请求每步返回 pad token（id=0）、不出 EOS 直到 max_tokens；感染随并发从 2 增至 8，持续 19 小时直至重启；带 logprobs 的请求会杀死 Rust frontend。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/51039

### INF-0420 `sglang_dspark_pagesize1_req_to_token_row_overflow_leak` SGLang page_size=1 下推测 slot 预留超行宽致跨请求溢出

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-08-04 · 旧号 6.27

- **原因对象**：KV缓存 / sglang / req_to_token 行宽分配与释放
- **原因行为**：推测解码每步每请求预留 committed+2*num_draft_tokens 个 slot，req_to_token 行宽仅在 page_size>1 时按此分配，page_size=1 时小于预留；释放切片行尾静默 clamp。
- **影响对象**：KV状态；跨请求/跨样本数据；可用性/进度（静默错误）
- **影响详情**：page_size=1、DSPARK gamma=16 时请求接近上下文边界即触发：写入溢出覆盖邻居请求行头部，每个边界命中请求永久泄漏 11-27 个 slot；长 eval 正常完成后服务器被空闲自检以 token_to_kv_ 错误 kill。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/33579

### INF-0421 `vllm_prefix_cache_silent_noop_mamba_gdn_hybrid` vLLM prefix caching 对混合 SSM 架构静默 no-op

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-08-05 · 旧号 6.28

- **原因对象**：KV缓存 / vllm / APC 前缀缓存的 recurrent 状态追踪
- **原因行为**：APC 只追踪 attention KV block，混合架构的 recurrent conv+SSM 状态没有 APC 缓存条目、无法被追踪，--enable-prefix-caching 实为 no-op 且不给 warning。
- **影响对象**：性能/资源（性能退化）
- **影响详情**：Qwen/Qwen3.6-35B-A3B 等 Mamba-2/GDN 混合模型上，共享相同长前缀的请求命中率 0%（vllm:prefix_cache_hits_total 恒为 0），prefill 计算量未减少；无 crash、无 error（vLLM 0.26.0，GB10/sm_121a）。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/51198

### INF-0426 `vllm_prefix_cache_silent_noop_mamba_gdn_hybrid` prefix caching 对 Qwen3.6 混合架构静默 0% 命中

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-08-06 · 旧号 6.33

- **原因对象**：KV缓存 / vllm / APC 前缀缓存的 recurrent 状态追踪
- **原因行为**：APC 只跟踪 attention KV block，Mamba-2/GDN 混合架构的 recurrent conv+SSM 状态不存入 APC，不创建任何 prefix-cache 条目，prefix caching 实为 no-op。
- **影响对象**：性能/资源（性能退化）
- **影响详情**：Qwen3.6-35B-A3B 开启 --enable-prefix-caching 后，相同长前缀请求 prefix_cache_queries_total 1131、hits_total 0，TTFT 与吞吐同未启用；错峰时序排除缓存未填充竞态，无警告。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/51250

### INF-0428 `sglang_dsv4_spec_draft_gt4_silent_compress_ring_underwrite` SGLang DSV4 draft>4 压缩 ring 欠写致 KV 损坏

`故障` · `active` · `infra_off` · `other_stack` · `verified` · `NEW` · 发现 2026-08-06 · 旧号 6.35

- **原因对象**：KV缓存 / sglang / c_plan.cuh 压缩 ring pad
- **原因行为**：compress write-plan 按乐观 seq_len 构建，提交 token 须靠 pad 留在 ring 中，c_plan.cuh 却把 pad 硬编码为 kMaxMTPDraftTokens=4，host 路径缺 pad。
- **影响对象**：KV状态；输出内容（质量退化）
- **影响详情**：speculative_num_draft_tokens > 4 时部分已提交 token 不被持久化，后续压缩窗口从陈旧 ring slot 计算；无 illegal access、无 NaN、无 assert，仅表现为安静的精度回归。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/pull/33872

### INF-0429 `sglang_pd_mtp_metadata_buffer_schema_mismatch` SGLang PD 分离 MTP 不对称致元数据 schema 不一致

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-08-06 · 旧号 6.36

- **原因对象**：KV缓存 / sglang / PD 分离 MTP 元数据 buffer
- **原因行为**：PD 两端各自分配 output_hidden_states 元数据 buffer：无 MTP 的 Prefill 用 16 元素 FP32 padding，有 MTP 的 Decode 用 MTP hidden 尺寸与 dtype。
- **影响对象**：可用性/进度（报错）
- **影响详情**：GLM-5.2-FP8 PD 分离、Prefill 无 MTP（tc_piecewise）、Decode 有 MTP 时，两节点正常启动，但 Mooncake 传输报 segment descriptor 获取失败 / KVTransferError。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/pull/33877

### INF-0432 `vllm_dsv4_mtp_eagle_non256_prompt_full_cache_miss` DSv4 MTP+Eagle 非对齐 prompt 前缀缓存全 miss

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-08-07 · 旧号 6.39

- **原因对象**：KV缓存 / vllm / prefix cache 命中长度块对齐
- **原因行为**：hybrid sparse attention 与 Eagle 推测解码的块对齐在非对齐 prompt 长度上失效，prefix caching 计算出的 cache_hit_length 为 0。
- **影响对象**：性能/资源（性能退化）
- **影响详情**：DSv4-Pro + fp8 KV + --block-size 256 + MTP（num_speculative_tokens=3）开启 prefix caching，prompt 长 16384、16385 命中 0，20000 命中 16128；无 crash，重算而非用错缓存。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/51441

### INF-0433 `vllm_offloading_connector_chunk_boundary_silent_wrong_output` vLLM Mamba 卸载在 chunk 边界重取后静默输出错误

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-08-07 · 旧号 6.40

- **原因对象**：KV缓存 / vllm / resolve_mamba_align_size
- **原因行为**：resolve_mamba_align_size 仅对 align 模式做命中窗口对齐，all 模式也按 block 存 recurrent 状态；精确边界 refetch 返回 N-1 token，恢复状态却含 token N。
- **影响对象**：KV状态；输出内容（静默错误）
- **影响详情**：Nemotron-Nano-9B-v2 + mamba_cache_mode=all + OffloadingConnector CPU 卸载，prompt 恰为 Mamba chunk 整数倍时，refetch 输出与 cold 在第 5 个 token 分叉（1270 vs 1556），无异常。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/51094

### INF-0434 `sglang_dflash_specdec_sconv_state_zero_corruption` SGLang 推测池 radix 命中时 sconv 状态全零

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-08-07 · 旧号 6.41

- **原因对象**：KV缓存 / sglang / 推测池 mamba tracking gate
- **原因行为**：DFlash/DSpark 推测池的 mamba tracking gate 挂在 prefill-graph capture 上，radix cache 命中路径绕过它，命中请求的 sconv 状态解析为全零 cache 张量。
- **影响对象**：KV状态；输出内容（质量退化）
- **影响详情**：Inkling/Inkling-Small（hybrid-SWA）开启 radix cache 与 DFlash/DSpark 推测时，长上下文 rollout 遗忘、卡住；SWEBench 70 题子集 pass rate 从 55% 降至 35%；无 crash、无 NaN。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/pull/34043

### INF-0435 `sglang_dsv4_dp_attention_progressive_corruption` SGLang DSV4 + DP attention 并发下渐进式输出损坏

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-08-07 · 旧号 6.42

- **原因对象**：KV缓存 / sglang / DSA 路径 KV/page-index（疑似）
- **原因行为**：根因未定位；原文疑似 dsv4 attention backend 的 DSA（HiSparse）注意力路径中 KV/page-index 出错，vLLM 0.26.0 同节点同 checkpoint 同 fp8 KV 不复现。
- **影响对象**：输出内容（静默错误）
- **影响详情**：2× H200 以 dsv4 attention + DP attention 服务 DeepSeek-V4-Flash-0731，输出损坏随 batch size 加剧：细微语言退化→畸形 tool-call markup→token soup；两种几乎不共享组件的配置均复现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/33397

### INF-0436 `vllm_free_blocks_lifo_to_fifo_reversal` vLLM 关闭前缀缓存时 free_blocks 由 LIFO 变 FIFO

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-08-08 · 旧号 6.43

- **原因对象**：KV缓存 / vllm / free_blocks 释放入队顺序
- **原因行为**：#48017 的条件含 enable_caching，caching 关闭时释放块走 append_n（FIFO）而非 prepend_n（LIFO），块重用由热块复用变为循环整个 block-id 空间。
- **影响对象**：可用性/进度；性能/资源（报错）
- **影响详情**：以 --no-enable-prefix-caching 运行时块重用顺序反转，TPU v7e 上确定性触发 E0200 RuntimeUnexpectedCoreHalt（SparseCore UserFatal）；该行为与 PR 描述“纯 no-op”及代码注释矛盾。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/51482

### INF-0437 `vllm_extract_hidden_proposer_wrong_slot_mapping` vLLM 隐藏态抽取 proposer 忽略分组 slot_mapping

`故障` · `active` · `infra_off` · `yes` · `documented` · `NEW` · 发现 2026-08-08 · 旧号 6.44

- **原因对象**：KV缓存 / vllm / 隐藏态抽取 proposer.propose()
- **原因行为**：propose() 忽略按组 slot_mappings，改用主组（block_size=1024）的 slot_mapping；CacheOnly 层 block_size=28 时得出错误 block/offset，散射写入未分配块。
- **影响对象**：隐状态；KV状态（数值异常）
- **影响详情**：hybrid 线性/全注意力模型多 KV cache group、CacheOnly 层 block_size 不同时触发；无 crash，抽取出的隐藏态 70%+ 为零值并含 NaN/inf，会毒化下游训练（若用于 RL proposer）或推测解码 draft。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/51328

### INF-0441 `vllm_kimi_k3_stateless_first_chunk_reads_stale_kda` vLLM KDA 单 token 首块误判 decode 读陈旧状态

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-08-08 · 旧号 6.48

- **原因对象**：KV缓存 / vllm / kda_metadata 的 decode 分类
- **原因行为**：非 spec 分支用 decode_threshold=1，单 token 首 chunk 被当 decode，直接读 conv/recurrent state；无状态 slot 掩码只在 prefill 路径，Mamba 块重分配不清零。
- **影响对象**：KV状态；跨请求/跨样本数据（静默错误）
- **影响详情**：Kimi-K3 KDA、非 spec、batch 全为单 token 首 chunk 行时触发；无 crash、无 NaN，请求静默继承前一拥有者回收块的 KDA state，输出语义错乱但表面正常。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/51483

### INF-0443 `vllm_truncate_computed_blocks_mamba_assert_kill` vLLM hybrid 模型块数断言杀死 prefill worker

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-08-08 · 旧号 6.50

- **原因对象**：KV缓存 / vllm / truncate_computed_blocks
- **原因行为**：truncate_computed_blocks() 断言每个 KV group 至少持 num_computed_tokens//block_size 块，但 Mamba group 块是整个 recurrent state，可合法更少。
- **影响对象**：可用性/进度（报错）
- **影响详情**：hybrid（full-attention+Mamba）模型配 KV connector、外部 hit 在本地 group 不再覆盖的边界提供 state 时触发；engine-fatal AssertionError 杀死 prefill worker。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/51468

### INF-0446 `sglang_batched_kv_free_aliasing` SGLang 批量 KV free 别名可变行，释放错误索引

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-08-08 · 旧号 6.53

- **原因对象**：KV缓存 / sglang / 延迟 KV free 队列
- **原因行为**：free group 持有可变 req_to_token 行的 view；cache_unfinished_req 在 free_group_end 前经 radix rematch 覆写该行，致释放替换后的 KV 索引而非原索引。
- **影响对象**：KV状态；输出内容（静默错误）
- **影响详情**：启用 radix cache、批量 KV free 期间发生 cache_unfinished_req rematch 时触发，token/paged/SWA/NPU 等全部分配器受影响；无 crash，活跃 KV 块被释放复用，后续输出静默错乱。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/pull/34067

### INF-0450 `sglang_sm120_page_split_int32_overflow_stray_write` SGLang page-split int32 偏移溢出致越界写

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-08 · 旧号 6.57

- **原因对象**：KV缓存 / sglang / _page_split_kernel 偏移计算
- **原因行为**：_page_split_kernel 的 page_idx 取自 tl.program_id(0)（int32），与 stride 的乘积按 int32 计算，page_idx≈57359 时回绕，偏移越出 buffer 约 2GiB。
- **影响对象**：KV状态；模型权重；可用性/进度（静默错误）
- **影响详情**：SM120 大 KV 池（约 3.67M token/rank 以上）时触发；越界地址未映射则 illegal-memory-access 崩溃，已映射则静默 stray write 进相邻的权重/KV/scratch 分配，无错误信号。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/pull/34027

### INF-0452 `sglang_hicache_write_through_host_mirror_starvation` SGLang HiCache write_through 致存储层饥饿

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-08-08 · 旧号 6.59

- **原因对象**：KV缓存 / sglang / HiCache host 池回收
- **原因行为**：write_through 下每个进 device radix cache 的节点立即 pin host 镜像；backup ack 只减引用计数不释放内存；唯一回收器 evict_host 跳过 device 常驻节点。
- **影响对象**：性能/资源（性能退化）
- **影响详情**：HiCache write_through 配存储后端（如 mooncake）且 host 池小于 device 池时触发；host slot 被 device 常驻节点永久占满，存储层静默饥饿、无法写入新条目，表现为缓存命中率逐渐下降，无 crash。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/pull/33862

### INF-0457 `sglang_prefill_graph_stale_track_rows_corrupt_conv_checkpoint` SGLang prefill graph 残留 track 污染前请求状态

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-08-09 · 旧号 6.64

- **原因对象**：KV缓存 / sglang / PrefillInputBuffers 清零守卫
- **原因行为**：replay 的 prefill graph 中 mamba_track_mask/indices 残留前次 replay 的行；清零守卫 bs != raw_bs 因按 token padding 永不成立。
- **影响对象**：KV状态；跨请求/跨样本数据（静默错误）
- **影响详情**：hybrid-SWA 模型在 prefill CUDA graph+chunked prefill+并发+prefix hit 时触发，禁用 prefill graph 即消失；track scatter 写入前一请求的 conv checkpoint，之后恢复该前缀的请求整个生成错误。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/pull/34184

### INF-0463 `vllm_mooncake_connector_request_id_aba_kv_block_leak` vLLM Mooncake 抢占后请求 ID 复用致 KV block 泄漏

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-08-10 · 旧号 6.70

- **原因对象**：KV缓存 / vllm / Mooncake 异步存储作业记账
- **原因行为**：MooncakeStoreConnector 以 Request-ID 为 key 跟踪异步存储作业；请求被抢占后重新调度时复用同一 Request-ID，旧作业仍在飞行中即被新作业覆盖记账（ABA），破坏异步存储作业记账
- **影响对象**：性能/资源；KV状态（性能退化）
- **影响详情**：无 crash；抢占频繁时 KV blocks 无人释放、静默累积泄漏，global segment 逐渐耗尽。复现于 vLLM 0.25/0.26 + Mooncake 0.3.12 + Qwen3-8B，TP2 多副本。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/51637

### INF-0466 `vllm_hybrid_kv_fine_grained_prefix_cache_incompatible_layout_assertion` vLLM 混合模型细粒度前缀命中与 KV manager 粒度不符致断言

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-08-11 · 旧号 6.73

- **原因对象**：KV缓存 / vllm / prefix-cache 细粒度命中启用判断
- **原因行为**：含 Mamba align 组的混合模型启用 fine-grained prefix-cache hit，未检查其他 KV 组（如 SWA DSpark drafter）的 manager 只支持 block 对齐 lookup
- **影响对象**：可用性/进度；性能/资源（报错）
- **影响详情**：Mamba + sliding-window drafter 混合模型（如 Kimi K3 + DSpark SWA）开启 prefix caching 时 assertion 崩溃；修复回退到 scheduler block 对齐命中后，短/中 prompt 无前缀复用、命中率下降。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/51843

### INF-0468 `vllm_multiconnector_duplicate_recving_completion_scheduler_crash` vLLM MultiConnector 重复上报 req_id 完成致断言崩溃

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-08-11 · 旧号 6.75

- **原因对象**：KV缓存 / vllm / MultiConnector 接收完成事件聚合
- **原因行为**：MultiConnector 聚合多个 consumer connector 的接收完成结果时不去重，同一 req_id 在两个 scheduler step 分别由 Connector A、B 上报
- **影响对象**：可用性/进度（报错）
- **影响详情**：PD 部署使用 NixlConnector + LMCacheMPConnector 时，第二次 completion 到达时请求已是 RUNNING，命中 assert req.status == RequestStatus.WAITING_FOR_KV，服务器崩溃。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/51786

### INF-0469 `vllm_mooncake_store_token_len_block_hashes_race_permanent_gap` vLLM Mooncake store 在 MTP+异步调度下崩溃或留永久间隙

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-08-11 · 旧号 6.76

- **原因对象**：KV缓存 / vllm / Mooncake 存储 token 记账
- **原因行为**：scheduler 在 block_hashes 扩展前乐观推进 token 数，process_tokens 未处理越界；clamp 后 _record_saved 仍记 clamp 前计数，resume offset 越过实际保存范围
- **影响对象**：KV状态；可用性/进度（报错）
- **影响详情**：Mooncake KV store + MTP 推测解码 + async scheduling 下：未 clamp 时 process_tokens 断言崩溃；clamp 后被跳过的 token 永不重试，外部 store 留下永久小间隙，影响 cache 命中率，生成正确性不受影响。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/51820

### INF-0470 `vllm_dsv4_flash_0731_b300_1m_kv_concurrency_regression` vLLM DSV4-Flash-0731 在 B300 并发退化 3.5 倍

`故障` · `active` · `ambiguous` · `config` · `documented` · `NEW` · 发现 2026-08-11 · 旧号 6.77

- **原因对象**：KV缓存 / vllm / B300 page size 下 KV 内存布局
- **原因行为**：根因未证实；原文推测检查点格式/权重变更改变了 KV 布局或 alignment，使 B300 特定 page size 下每 token KV 占用增加、利用率下降
- **影响对象**：性能/资源（性能退化）
- **影响详情**：B300 上 DeepSeek-V4-Flash-0731 1M 上下文最大并发由 preview 的 27.08x 降至 7.73x（含 DSpark），远超 B200 上 DSpark 的正常开销（20.43x→17.12x）；无报错，不跑 benchmark 难以察觉。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/51799

### INF-0473 `vllm_prefix_caching_mamba2_gdn_hybrid_silent_zero_hit_noop` vLLM 前缀缓存在 Mamba-2/GDN 混合模型上静默 0% 命中

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-08-12 · 旧号 6.80

- **原因对象**：KV缓存 / vllm / APC 前缀缓存（混合模型）
- **原因行为**：APC 只跟踪 attention KV blocks，无法缓存 recurrent conv+SSM 状态，但在 Mamba-2/GDN 混合模型上仍接受 --enable-prefix-caching 且不告警
- **影响对象**：性能/资源（性能退化）
- **影响详情**：Qwen3.6-35B-A3B 等混合模型开启前缀缓存后，即使请求共享长前缀也 0% 命中，prefill 无复用；无 crash、无 warning，/metrics 中 vllm:prefix_cache_hits_total 为 0。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/51198

### INF-0476 `vllm_gdn_non_spec_state_indices_strided_view_non_contiguous` vLLM GDN 向 kernel 传非连续 state_indices

`故障` · `active` · `infra_off` · `partial` · `verified` · `NEW` · 发现 2026-08-13 · 旧号 6.83

- **原因对象**：KV缓存 / vllm / GDN metadata non-spec 分支
- **原因行为**：non-spec decode 分支把 strided 列视图 block_table_tensor[:, 0] 直接当 state_indices 传给要求连续 1-D 的 kernel；单元素时 .contiguous() 不纠正
- **影响对象**：可用性/进度（报错）
- **影响详情**：GDN 混合线性注意力模型在非 FULL cudagraph 路径的 non-spec decode（如单 decode batch）warmup/decode 时报错 state_indices must be contiguous and one-dimensional。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/52207

### INF-0478 `vllm_diskbackend_kv_offload_double_buffer_race_corrupts_gpu_cache` vLLM DiskBackend 双缓冲竞态静默损坏 GPU KV cache

`故障` · `active` · `infra_off` · `config` · `verified` · `NEW` · 发现 2026-08-13 · 旧号 6.85

- **原因对象**：KV缓存 / vllm / DiskBackend._do_load 双缓冲
- **原因行为**：_do_load 双缓冲流水线把 pending H2D event 记在局部列表，返回前从不 drain；下一次调用的 preadv 可能覆盖上一次仍在被 H2D 拷贝读取的 buffer slot
- **影响对象**：KV状态（静默错误）
- **影响详情**：启用 DiskBackend KV 卸载并连续 _do_load（驱逐后 reload）时，reload 进 GPU 的 KV cache 静默损坏，无 crash；对应回归测试在 main 上失败、修复后通过。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/51667

### INF-0479 `vllm_eagle_block_drop_untested_ge3_attention_groups_prefix_cache_zero` vLLM EAGLE block-drop 使 ≥3 注意力组模型前缀缓存失效

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-08-13 · 旧号 6.86

- **原因对象**：KV缓存 / vllm / 混合 KV 前缀命中收敛循环
- **原因行为**：use_eagle() 为真时走通用 convergence loop，但 SWA 不满足 downward-closed（j 处命中不保证 i<j 命中），该循环对 ≥3 个 attention 组失效；CI 只覆盖 ≤2 组
- **影响对象**：性能/资源（性能退化）
- **影响详情**：DeepSeek-V4-Flash + DSpark（1 个 full MLA + 3 个 SWA 族共 4 组）开启前缀缓存时命中率为 0，无报错；含 #42258 但缺 #44082 的分支完全失去前缀缓存，主线可能正常。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/51771

### INF-0480 `vllm_mamba_running_cow_misclassified_after_external_hits` vLLM 外部命中后 Mamba CoW 被误分类致状态不一致

`故障` · `active` · `infra_off` · `partial` · `verified` · `NEW` · 发现 2026-08-13 · 旧号 6.87

- **原因对象**：KV缓存 / vllm / Mamba 块分配早退路径
- **原因行为**：外部命中填充 Mamba block table 后，continuation 仍在同一 block 时 allocate_new_blocks() 提前返回、未登记该请求，后续 CoW 被误判为首次 prefill CoW
- **影响对象**：KV状态（静默错误）
- **影响详情**：Mamba 混合模型（如 Kimi-K3）外部命中后注册 partial-tail cache 条目时，Core 替换 live block 而 worker 视请求为 running 并追加返回的 block ID，两侧对 live Mamba state 列不一致，后续可能读错状态。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/51766

### INF-0481 `sglang_mamba_state_donation_misalignment_dcp_logprob_divergence` SGLang DCP 下 Mamba 状态网格未加宽致 logits 偏离

`故障` · `active` · `infra_off` · `partial` · `verified` · `NEW` · 发现 2026-08-13 · 旧号 6.88

- **原因对象**：KV缓存 / sglang / Mamba radix cache 状态跟踪网格
- **原因行为**：DCP 把 tree page 加宽 dcp_size 倍，Mamba state 跟踪网格未加宽；donated state 可领先节点达 tree_page-1 token，匹配时仍视节点深度等于状态深度，中间 token 重复喂入
- **影响对象**：KV状态；输出分布（静默错误）
- **影响详情**：hybrid 线性注意力模型（Kimi-K3）以 --dcp-size>1 + extra_buffer 策略服务时，从缓存前缀 resume 的 logits 确定性偏离 fresh prefill：|dlogprob| 最大 6.9，157/1081 位置 >2，约 700 token 后衰减。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/pull/34760

### INF-0483 `trtllm_kvcm_v2_page_tables_confidential_compute_host_mapping_fault` TRT-LLM KVCM V2 在 CC 下向 kernel 传 host 张量

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-08-13 · 旧号 6.90

- **原因对象**：KV缓存 / tensorrt-llm / KVCM V2 page table 传递
- **原因行为**：KVCM V2 把四个 host tensor 直接传给从 GPU 解引用它们的 CUDA kernel，依赖 host allocation 可映射到 GPU 地址空间，映射不可用时未改用 H2D 传输
- **影响对象**：可用性/进度（报错）
- **影响详情**：Confidential Compute 模式下 host 内存无法映射到 GPU 地址空间，kernel 访问时 fault；非 CC 模式不受影响。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/NVIDIA/TensorRT-LLM/pull/17658

### INF-0484 `trtllm_minimax_m3_eagle_draft_subpage_pool_bound_underreported` TRT-LLM MiniMax-M3 draft 池上界少报致读错 KV

`故障` · `active` · `infra_off` · `config` · `verified` · `NEW` · 发现 2026-08-13 · 旧号 6.91

- **原因对象**：KV缓存 / tensorrt-llm / DraftSubpageView 池上界
- **原因行为**：draft 视图把 blocks_in_primary_pool 委托给 target manager，该值以 target 128-token page 计、相对另一 pool base，TMA descriptor 少报 page 数
- **影响对象**：KV状态；性能/资源（性能退化）
- **影响详情**：MiniMax-M3 Eagle3 分离式服务走 FlashInfer TRTLLM-Gen draft 路径时，draft block-table 指向 descriptor 范围外的 slot、读到错误 draft KV；target 输出正确，冷启动接受长度从 3.3+ 降至 1.8–2.0。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/NVIDIA/TensorRT-LLM/pull/17656

### INF-0486 `vllm_mooncake_preemption_frees_blocks_under_async_store_kv_key_mismatch` vLLM Mooncake 抢占释放在飞 store 的 block 致键值错配

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-08-14 · 旧号 6.93

- **原因对象**：KV缓存 / vllm / Mooncake 异步 store/ledger
- **原因行为**：异步 KV store 任务在飞行中时，抢占路径已释放其正在读取的 GPU block 并可复用给其他请求；store 随后写入可能已被覆盖的内容，且 ledger 键缺少序列维度区分
- **影响对象**：KV状态；跨请求/跨样本数据；输出内容（静默错误）
- **影响详情**：Mooncake 分离式部署发生抢占时，store 中 key 与 KV 内容错配；后续命中该 key 的 prefill 读到其他请求/内容的 KV，从 prefill 起消费错误上下文，输出静默退化，无 crash。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/52360

### INF-0488 `sglang_swa_tombstone_blocks_prefix_match_pool_accounting_crash` SGLang SWA tombstone 被拒匹配致 pool 记账漂移崩溃

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-08-14 · 旧号 6.95

- **原因对象**：KV缓存 / sglang / SWA 前缀匹配校验器 tombstone 处理
- **原因行为**：未启用 HiCache 时 SWA 匹配校验器拒绝 tombstone 节点，insert 后 rematch 返回 0，cache_protected_len 不前进，dup_start 算错致 pool 记账漂移。
- **影响对象**：KV状态；可用性/进度（报错）
- **影响详情**：hybrid SWA + EAGLE + 统一 radix cache（默认配置）下触发：前缀缓存命中先静默塌到 0，随后每个 TP rank 报 pool memory leak detected! 崩溃。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/pull/34870

### INF-0497 `sglang_abort_inflight_kv_transfer_freed_page_overwrite` SGLang abort 立即释放 KV 页遭在飞传输覆写

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-08-16 · 旧号 6.104

- **原因对象**：KV缓存 / sglang / decode 侧 abort KV 页释放时机
- **原因行为**：PD 分离下 decode 请求被 abort 时，其 KV pages 与 req-pool slot 被立即释放，但不取消仍在飞行的 prefill→decode KV 传输，也不等其完成。
- **影响对象**：KV状态；跨请求/跨样本数据（质量退化）
- **影响详情**：超时、客户端断连等 abort 频繁的负载下，下一个分到同一批 pages 的无关请求被迟到写入覆写 KV；无 crash、无报错，受害请求输出质量劣化。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/pull/35049

### INF-0498 `trtllm_cluster_storage_get_prefix_ttl_bypass` TRT-LLM cluster storage 前缀读不查 TTL 返回过期项

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-08-16 · 旧号 6.105

- **原因对象**：KV缓存 / tensorrt-llm / cluster storage 前缀读路径
- **原因行为**：HTTP cluster storage 的点读路径已用过期谓词拒绝过期条目，但 get_prefix() 返回匹配条目时不检查 TTL 状态。
- **影响对象**：KV状态；待定（静默错误）
- **影响详情**：条目已过期而后台清扫尚未执行的窗口内，get_prefix() 仍返回该条目，依赖 TTL 语义做缓存一致性的调用方拿到过期数据；无 crash。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/NVIDIA/TensorRT-LLM/pull/17769

### INF-0500 `vllm_mooncake_multiconnector_pd_silent_output_corruption` vLLM Mooncake store 与 NIXL 组合静默污染 KV

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-08-17 · 旧号 6.107

- **原因对象**：KV缓存 / vllm / Mooncake store 组合路径
- **原因行为**：根因未精确定位。MultiConnector 同时挂 NixlConnector 与 MooncakeStoreConnector 时，Mooncake store 路径与该组合交互在持续负载下污染 KV（关 Mooncake 即不复现）。
- **影响对象**：KV状态；输出内容（静默错误）
- **影响详情**：Kimi-K3 TP=8 两节点 1P1D 持续负载下，受影响请求生成长串无意义混合语言/重复 token，部分流以 HTTP 502 中断；引擎不 crash。NIXL-only 稳定，关推测解码仍复现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/52627

### INF-0501 `vllm_lmcache_pd_full_hit_token0_divergence` vLLM LMCache PD 报告完整命中但解码自 token 0 发散

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-08-17 · 旧号 6.108

- **原因对象**：KV缓存 / vllm / LMCache P/D receiver 恢复路径
- **原因行为**：receiver 侧命中判定只校验元数据（prefix 匹配/长度）、不校验载荷字节，报告 full KV-cache hit 时恢复出的 KV 载荷与 prefill 真值不符。
- **影响对象**：KV状态；输出内容（静默错误）
- **影响详情**：vLLM 0.26.0 + LMCache P/D 分离走前缀命中路径时，decode 自 token 0 起即与 monolithic 基线发散；无 crash、无 NaN，命中率指标一切正常。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/52617

### INF-0502 `vllm_offloading_connector_gt64g_silent_crash_restart` OffloadingConnector 超 64GB CPU 卸载时静默崩溃

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-08-17 · 旧号 6.109

- **原因对象**：KV缓存 / vllm / Tiering offload 大容量路径
- **原因行为**：TieringOffloadingSpec 在 cpu_bytes_to_use 超过 64GB 的大容量路径上出错（原文未给具体机制），同容量下 SimpleCPUOffloadConnector 正常。
- **影响对象**：可用性/进度（报错）
- **影响详情**：GLM 5.2 FP8 下 cpu_bytes_to_use=128GB 时服务数秒后进程静默消失并重启，256GB 时无法进入 serving；无错误日志。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/52656

### INF-0504 `sglang_dspark_hicache_long_session_prefix_hit_zero` SGLang DSV4 HiCache 长会话每轮前缀命中为 0

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-08-17 · 旧号 6.111

- **原因对象**：KV缓存 / sglang / hybrid pool radix 前缀复用
- **原因行为**：DSV4 hybrid-pool 的 radix 前缀复用路径在长多轮会话分支上失效，前缀稳定的长会话请求匹配不到已缓存 token（客户端抓包已排除请求侧问题）。
- **影响对象**：性能/资源；指标/日志（性能退化）
- **影响详情**：DSV4-Flash + DSPARK + HiCache 生产流量中，长 agent 会话每轮 #cached-token 为 0，同窗口短请求命中约 98%；输出正确但长会话全量重新 prefill，整体命中率 83.8% 掩盖了该损失。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/35129

### INF-0505 `vllm_pd_multitier_producer_held_blocks_supply_nothing` vLLM PD 多层 offload：producer 已持块反而零供给

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-08-18 · 旧号 6.112

- **原因对象**：KV缓存 / vllm / Multi Tier p2p 供给路径
- **原因行为**：OffloadingConnector Multi Tier 流（secondary_tiers type:p2p）中，producer 已持有请求块的本地前缀缓存命中路径不向 consumer 供给任何数据。
- **影响对象**：性能/资源（性能退化）
- **影响详情**：consumer 的 fetch 一直未满足直至 30s 超时，再按 recompute 策略本地重算 prompt；输出正确但尾延迟 +30s，producer 缓存越好失败越多，表面仅 recompute 计数上涨。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/52808

### INF-0506 `vllm_offloading_connector_spec_decode_store_never_serve` vLLM OffloadingConnector 遇推测解码只存不取

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-08-18 · 旧号 6.113

- **原因对象**：KV缓存 / vllm / OffloadingConnector 加载路径
- **原因行为**：启用 MTP/EAGLE 推测解码时，OffloadingConnector 的 serve 路径不命中：KV 持续 offload 到 CPU tier，但后续请求从不从 offload 缓存恢复。
- **影响对象**：性能/资源（性能退化）
- **影响详情**：hybrid GDN 模型（XPU 首报）+ MTP/EAGLE 下触发：无 crash、输出正确，前缀复用收益归零、每轮全量重算，并额外消耗 CPU 与带宽。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/52735

### INF-0508 `sglang_radix_evictable_shared_page_double_count_idle_die` SGLang radix 共享页重复计数致空闲检查杀死 scheduler

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-08-18 · 旧号 6.115

- **原因对象**：KV缓存 / sglang / radix evictable_size()
- **原因行为**：radix cache 的 evictable_size() 按节点引用累计而 total 按物理页计量，被多节点共享的页被重复计入，与池不变量检查的计量单位不一致。
- **影响对象**：可用性/进度（报错）
- **影响详情**：默认配置下，全部请求成功完成后的下一次空闲时 _check_all_pools() 不变量检查失败，抛 ValueError: pool memory leak detected! 杀死 scheduler；无数据损坏，dLLM 路径 page_size=32 放大触发。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/35270

### INF-0510 `trtllm_kvcache_v2_avg_seqlen_unset_1m_segfault` KvCacheManagerV2 未设 avg_seq_len 长上下文崩溃

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-08-18 · 旧号 6.117

- **原因对象**：KV缓存 / tensorrt-llm / KvCacheManagerV2 容量回退
- **原因行为**：未设 kv_cache_config.avg_seq_len 时回退为 max_seq_len/2，KvCacheManagerV2 在该默认回退下与真实请求长度组合时违反内部容量假设。
- **影响对象**：可用性/进度（报错）
- **影响详情**：Nemotron-3-Ultra NVFP4 + max_seq_len 1048576 且不设 avg_seq_len 时，约 256k token 的请求在 KvCacheManagerV2 内 segfault；补 avg_seq_len: 9000 后 800k 请求正常。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/NVIDIA/TensorRT-LLM/issues/17926

### INF-0511 `trtllm_block_radix_tree_clear_childless_root_hang` TRT-LLM BlockRadixTree.clear() 遇空根块死循环

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-08-18 · 旧号 6.118

- **原因对象**：KV缓存 / tensorrt-llm / BlockRadixTree.clear()
- **原因行为**：clear() 只靠 detach_next() 摘除 RootBlock 最后一个子节点时的副作用收缩 self.next；已无子节点的 RootBlock 无子可摘，外层循环永远重选它。
- **影响对象**：可用性/进度（挂起）
- **影响详情**：KV cache radix 树持有 childless RootBlock 时调用 clear() 触发：clear() 永不返回，尾部 assert not self.next 不可达。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/NVIDIA/TensorRT-LLM/issues/17908

### INF-0513 `vllm_kimi_k3_ngram_spec_strided_state_indices_gpu_fault` vLLM ngram 推测下 GDN state_indices 非连续致崩溃

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `var` · 发现 2026-08-18 · 旧号 6.120

- **原因对象**：KV缓存 / vllm / spec 掩码切片的 state_indices
- **原因行为**：ngram 推测路径以 block_table_tensor[~spec_mask, 0] 切片得到 stride≠1 的非连续 state_indices（数据对、布局错）；aiter/ROCm 对混合批 GDN 状态索引布局另有缺陷。
- **影响对象**：可用性/进度（报错）
- **影响详情**：ROCm/aiter + GDN hybrid 启用 ngram 推测时 KDA decode kernel 报 must be contiguous；紧致化修复后单流 spec-only 稳定，混合 spec/non-spec 并发批仍触发 GPU memory access fault。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/52792
- **关联**：var:INF-0476

### INF-0515 `vllm_nixl_hma_recv_fail_open_corrupted_200` vLLM NIXL KV 接收失败 fail-open 返回损坏 token

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-08-19 · 旧号 6.122

- **原因对象**：KV缓存 / vllm / NIXL HMA connector 接收失败处理
- **原因行为**：decode 节点 KV receive 失败时未终止请求，继续从 invalid/uninitialized KV cache 解码；KV load failure policy 声明为 fail，实现却是 fail-open。
- **影响对象**：输出内容；KV状态（静默错误）
- **影响详情**：2×B200 P/D 分离 + NIXL HMA 下，网络故障或对端崩溃导致接收失败时无 crash、返回 HTTP 200 携带损坏 token；DeepSeek-V4 变体中损坏内容进 reasoning 字段、content 为空，客户端看到空响应。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/52234；https://github.com/vllm-project/vllm/issues/52276

### INF-0517 `sglang_mooncake_shared_ssd_path_multi_rank_kv_corruption` SGLang 多 rank 共用 Mooncake SSD 路径致 KV 损坏

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-08-19 · 旧号 6.124

- **原因对象**：KV缓存 / sglang / MooncakeStore SSD 卸载路径
- **原因行为**：MooncakeStore 把同一 ssd_offload_path 传给每个 rank 的 Mooncake client，而 LOCAL_DISK bucket 层假设一目录一 client，无跨 client 互斥或命名空间隔离。
- **影响对象**：KV状态；输出内容（静默错误）
- **影响详情**：TP8 + HiCache L3 卸载到 Mooncake bucket backend 时各 rank 的 KV 对象静默互相覆盖，命中率正常但恢复内容错误，生成质量不可预测退化；无 crash、无错误日志；重启后无法区分对象所属 rank。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/35484

### INF-0519 `vllm_align_prefix_cache_priority_sched_never_hit` vLLM align 模式前缀缓存在 priority 调度下命中率恒为 0

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-08-19 · 旧号 6.126

- **原因对象**：KV缓存 / vllm / priority 调度切分与 align 快照
- **原因行为**：align-mode Mamba 快照只在精确块边界缓存，要求非最终 prefill chunk 块对齐；SchedulingPolicy.PRIORITY 切分 token budget 时不保块对齐，快照从不落在精确边界。
- **影响对象**：性能/资源（性能退化）
- **影响详情**：Qwen3.8-27B hybrid GDN + MTP + prefix caching，仅换 --scheduling-policy priority 后 99.6 万查询 0 命中（FCFS 为 17.5%），第二请求无延迟收益；无报错、无日志，只有命中率计数器能暴露。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/52897

### INF-0520 `vllm_mooncake_null_block_kv_key_publication_silent_divergence` vLLM Mooncake 为 NULL 状态块发布缓存键致输出静默分歧

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-08-20 · 旧号 6.127

- **原因对象**：KV缓存 / vllm / Mooncake Store 保存路径键发布
- **原因行为**：Mooncake Store 保存路径把 NULL_BLOCK_ID=0 的空 recurrent-state 块解析为地址并照常发布 Store 键，不拒绝空块，为无有效 GDN/Mamba 状态的边界也发布键。
- **影响对象**：KV状态；输出分布（静默错误）
- **影响详情**：hybrid Mamba/GDN + align 模式 + prefix caching 时，消费端成功加载全零状态并跳过重算，与冷启动基线静默分歧；无 Put/Get 错误、无 crash，部分情形 token 不变但 logprob 显著漂移，传输完整性校验全过。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/53084

### INF-0524 `vllm_mamba_align_precopy_state_idx_wrong_block_size_ima` vLLM Mamba 状态列误用 attention 块大小致越界崩溃

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-08-20 · 旧号 6.131

- **原因对象**：KV缓存 / vllm / _mamba_state_idx_gpu 播种
- **原因行为**：add_request 用 attention 的 cache_config.block_size 而非 mamba 组块大小播种 _mamba_state_idx_gpu，致 state_idx=781（应为 45）。
- **影响对象**：KV状态；可用性/进度（报错）
- **影响详情**：hybrid GDN + prefix caching + align 模式下第二请求续传时，precopy kernel 读越界 block_table，确定性报 CUDA illegal memory access；越界落在映射内时则静默读错 recurrent state。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/53142

### INF-0526 `vllm_nixlpush_kv_lease_expiry_zero_transfer` vLLM PD 分离 KV lease 续期失效致 decode 请求卡死

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-08-21 · 旧号 6.133

- **原因对象**：KV缓存 / vllm / NixlPush KV lease 心跳续期
- **原因行为**：跨节点 P/D 高并发下 heartbeat 未能续期 KV lease，prefill worker 在默认 30s lease 到期后单方面释放尚未被远端取走的 KV block。
- **影响对象**：可用性/进度（挂起）
- **影响详情**：UCX 跨节点 P/D、Batch=128 时受影响请求成功传输数为 0，decode 侧请求永久 stuck；无 crash，prefill 侧仅一条 info 级释放日志，只查进程/引擎健康的 liveness 探针不可见。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/53095

### INF-0527 `vllm_tiering_unbound_store_60s_reap_recompute` vLLM offload 未绑定 store 固定 60s 回收致消费者落空重算

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-08-21 · 旧号 6.134

- **原因对象**：KV缓存 / vllm / tiering p2p 未绑定 store 回收
- **原因行为**：producer 的 parked KV 在 store 提交后固定 60s（UNBOUND_STORE_TIMEOUT_S）内无 FetchMsg 绑定即被 reap；定时器锚点在 store 时刻，与消费者请求是否仍存活或排队无关。
- **影响对象**：性能/资源（性能退化）
- **影响详情**：consumer 因队列积压晚于 60s 发出 fetch 时找到空，等满 LOAD_TIMEOUT_S=30s 后回退全量 prompt 重算；输出正确、无 crash，表现为 +30s 尾延迟，表面仅 recompute 计数上涨。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/53128

### INF-0529 `vllm_hidden_states_kv_budget_overestimate_startup_reject` vLLM 开启 hidden-state 提取后 KV 容量记账高估致启动被拒

`故障` · `active` · `ambiguous` · `partial` · `documented` · `NEW` · 发现 2026-08-21 · 旧号 6.136

- **原因对象**：KV缓存 / vllm / hidden-state 提取的 KV 容量记账
- **原因行为**：启用 extract_hidden_states 时其显存记账使 KV-cache 容量计算高估所需显存。
- **影响对象**：可用性/进度（报错）
- **影响详情**：DeepSeek-V4-Flash TP4 H100 提取层 40–43 hidden states（fp8 KV、prefix caching、CUDA graph）时，模型加载成功后引擎启动阶段被拒；0.26.0 失败，仅换含 793ca6998 修复的 wheel 即恢复。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/53243

### INF-0530 `vllm_kv_offload_supply_demand_contract_drift` vLLM PD 多层 KV offload 供需 key 契约漂移致超时重算

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-08-22 · 旧号 6.137

- **原因对象**：KV缓存 / vllm / p2p tier 供给/需求 key 计算
- **原因行为**：producer 供给集与 consumer 需求集由两台引擎的两条独立路径计算、无一致性校验；#51840 把 lookup 返回改为 HIT_PENDING，静默改变了 consumer 需求的 key。
- **影响对象**：性能/资源（性能退化）
- **影响详情**：PD 流程经 P2P secondary tier 且供需分歧时，consumer 未匹配需求 park 至 _LOAD_TIMEOUT_S 后重算 prompt；输出正确，+30s 尾延迟，仅 recompute 计数上涨，测试套件未察觉；#52808、#53062 为同一弱点的独立实例。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/53083

### INF-0531 `vllm_tiering_cascade_hit_pending_key_permanent_drop` vLLM tiering 级联丢弃 HIT_PENDING key 致超时重算

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-08-22 · 旧号 6.138

- **原因对象**：KV缓存 / vllm / tiering 级联、prepare_store
- **原因行为**：cascade 只保留 lookup 为 HIT 的 key；在途的 HIT_PENDING 被 prepare_store 排除、被 cascade 丢弃，scheduler 又推进 chunk 索引越过它，无路径再供给。
- **影响对象**：性能/资源（性能退化）
- **影响详情**：prefix 命中而写入在途时，P2P producer 腿上无该 kv_request_id 下的 parked 条目，peer 等满 _LOAD_TIMEOUT_S 后全量重算；输出正确、无 crash，表现为 +30s 超时与重算开销。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/53062

### INF-0532 `vllm_kv_load_reject_partial_rollback_state_divergence` vLLM 外部 KV load 被拒后部分回卷致结果与全量重算不一致

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-08-22 · 旧号 6.139

- **原因对象**：KV缓存 / vllm / KV load 失败回卷路径
- **原因行为**：外部 KV load 被拒后 scheduler 状态回卷，但 Model Runner V2 保留旧 device 侧 computed-token offset；overlap 批次在途 output 可改写已重置状态或迟到。
- **影响对象**：输出内容（静默错误）
- **影响详情**：外部 KV connector + kv_load_failure_policy=recompute 且 load 失败、overlap 调度（含多 token 推测异步帧）时，最终结果与全量重算不等价；无 crash，隐藏在罕见失败回退路径中。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/53298；https://github.com/vllm-project/vllm/pull/53730

### INF-0533 `vllm_kv_offload_sub_block_ptr_padded_page_oob` vLLM KV offload 裸指针假设无 padding 布局致启动崩溃

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-08-22 · 旧号 6.140

- **原因对象**：KV缓存 / vllm / offload 子块指针推导与 mask
- **原因行为**：compute_sub_block_ptrs 由 data_ptr()+stride 推导裸地址，假设无 padding 行布局；swap kernel 的 mask 只对 declared size、不对 allocation 边界。
- **影响对象**：可用性/进度（报错）
- **影响详情**：DeepSeek-V4 MLA 页 padding 到 576B（fp8_ds_mla）或 584B 时，启用 OffloadingConnector + fs_python tier 即在启动期以 CUDA error 杀死 worker；错误表面与布局假设根因相距很远。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/53271

### INF-0537 `vllm_mamba_align_state_materialization_lookup_cap_mismatch` vLLM align 模式 Mamba 前缀缓存物化点与查找上限错配致零复用

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-08-23 · 旧号 6.144

- **原因对象**：KV缓存 / vllm / align 模式状态物化与查找上限
- **原因行为**：align 模式状态仅在 chunk 末物化，而查找上限为 num_tokens−1，唯一深位状态常越界；推测解码 back-off 按 16-token block 设计，align block 却为 544–2128。
- **影响对象**：性能/资源（性能退化）
- **影响详情**：sibling 共享 4278 token 中 2000 时复用为 0，精确块对齐的逐字节重复请求也零命中；eagle/mtp 等推测家族每次额外放弃一整块前缀，GB10 实测 warm prefill 慢约 9×；无错误、无日志，只有命中率计数器能暴露。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/53479

### INF-0538 `vllm_mamba_full_local_hit_n_minus_1_truncation_zero_new_tokens_crash` vLLM hybrid 前缀缓存全命中截断致 scheduler 断言崩溃

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-08-25 · 旧号 6.145

- **原因对象**：KV缓存 / vllm / hybrid 前缀缓存命中截断与查找顺序
- **原因行为**：hybrid（Mamba/GDN）路径把块对齐 prompt 的本地前缀缓存全量命中按 N−1 截断，且查找先于 get_num_new_matched_tokens() 执行，使待 prefill 的 num_new_tokens 归零。
- **影响对象**：可用性/进度（报错）
- **影响详情**：块对齐 prompt 首跑返回 200，重放时 scheduler 的 assert num_new_tokens > 0 失败，prefill EngineCore 终止、HTTP 500、K8s 容器重启；实证于 Dynamo 1P1D + NixlConnector、TP4。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/53706

### INF-0539 `vllm_eagle_last_block_drop_hybrid_alignment_unit_recompute` vLLM EAGLE 命中扣除按 hybrid 大对齐单元重算致吞吐下降

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-08-25 · 旧号 6.146

- **原因对象**：KV缓存 / vllm / KV 协调器 eagle 组标记
- **原因行为**：模型无 is_eagle_group 注解时把所有 KV 组标为 eagle，命中扣除一个 scheduler 对齐单元；为 16-token 块设计的保护在 hybrid 的 1,648-token 对齐单元上代价放大约 100 倍。
- **影响对象**：性能/资源（性能退化）
- **影响详情**：hybrid GDN/Mamba + prefix caching + 任意 EAGLE 族推测解码（含 K=0）时每次命中重算整个对齐单元：warm 命中率 97.5% 降至 86.9–91.7%，c=8 吞吐 327 降至约 200 tok/s；无报错，仅缓存计数器可见。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/53670

### INF-0540 `vllm_hybrid_state_registration_n_floor_lookup_cap_mismatch` vLLM hybrid 状态注册与查找 n−1 边界错配致命中归零

`故障` · `active` · `infra_off` · `config` · `verified` · `NEW` · 发现 2026-08-25 · 旧号 6.147

- **原因对象**：KV缓存 / vllm / hybrid 前缀缓存状态注册站点
- **原因行为**：查找只搜 ≤n−1 的位置（eagle drafter 再下移一个 hash 单元），而 mamba tail split、tail-boundary 位置、SWA retention 三个注册点从 n 取 floor，块对齐时永远查不到。
- **影响对象**：性能/资源（性能退化）
- **影响详情**：hybrid（Mamba/GDN+SWA）模型在块对齐 prompt 或启用 eagle 族 drafter 时 prefix 命中归零，共享文档的分叉尾重提交全部从零 prefill；无报错，仅 vllm:prefix_cache_hits_total 类计数器可见。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/53802

### INF-0541 `vllm_example_connector_token_keyed_external_kv_partition_drop` vLLM ExampleConnector 存储键丢隔离维度致跨租户 KV 复用

`故障` · `active` · `infra_off` · `config` · `verified` · `var` · 发现 2026-08-25 · 旧号 6.148

- **原因对象**：KV缓存 / vllm / ExampleConnector 存储键派生
- **原因行为**：外部存储键只由 prompt token bytes 派生，丢弃 lora_request、不用 block_hashes；prompt_embeds 请求无 token ids，对齐得 -16。HF3FS 连接子复制同一模式。
- **影响对象**：跨请求/跨样本数据；KV状态（静默错误）
- **影响详情**：cache_salt 或 LoRA 不同的同 token 请求互相命中他人 KV（salt-y 请求拿到 salt-x 存的 48 token），返回基于他人缓存的结果且无报错；prompt_embeds 请求全部键到同一值，并向 scheduler 返回 -16 的负外部 token 计数。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/53495
- **关联**：var:INF-1301

### INF-0543 `vllm_swa_store_skip_eagle_tail_plus_one_starvation` vLLM offload SWA 存储跳过与 eagle 查找差 1 致零命中

`故障` · `active` · `infra_off` · `partial` · `verified` · `var` · 发现 2026-08-26 · 旧号 6.150

- **原因对象**：KV缓存 / vllm / KV offload 的 SWA 存储跳过逻辑
- **原因行为**：_build_store_jobs 按非 eagle 查找窗口只存每段末尾 tail 个 SWA 块，而 _lookup 对 eagle 未验证组要求 tail+1 个连续块，存储侧不变式与 eagle 查找窗口差 1。
- **影响对象**：性能/资源（性能退化）
- **影响详情**：KV offloading + hybrid SWA + eagle/MTP 推测解码组配齐时，eagle SWA 组每个请求恒为 0 命中并传播为整请求零命中；offload 计数器 bytes out 持续增长、bytes in 恒为零，卸载收益归零且存储无限增长，无报错无警告。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/53951
- **关联**：var:INF-0530

### INF-0546 `vllm_align_prefix_cache_retention_interval_default0_unlogged_dead_states` vLLM hybrid align 前缀缓存保留间隔默认 0 致命中静默失效

`故障` · `active` · `infra_off` · `config` · `verified` · `NEW` · 发现 2026-08-27 · 旧号 6.153

- **原因对象**：KV缓存 / vllm / MambaManager 保留间隔默认值与校验
- **原因行为**：hybrid align 模式下 retention_interval 默认 0，只保留 semantic checkpoints，其余 boundary state 照常写入但从不 hash；该默认无任何日志，校验只对无害设值报错。
- **影响对象**：性能/资源；指标/日志（性能退化）
- **影响详情**：hybrid GDN 模型启用 --enable-prefix-caching（align 模式）且用默认值时，prefix cache 大部分失效：存储在写、查找永空；Prefix cache hit rate 指标仍虚报非零，无报错无告警。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/53595

### INF-0550 `vllm_kv_offload_finished_request_unwritten_final_slot_cross_request_poisoning` vLLM KV offload 水位线含未写入末槽致跨请求 KV 投毒

`故障` · `active` · `infra_off` · `config` · `verified` · `NEW` · 发现 2026-08-28 · 旧号 6.157

- **原因对象**：KV缓存 / vllm / KV offload 完成请求存储水位线
- **原因行为**：正常完成分支用 req.num_tokens 作存储水位线（中止分支正确），含从未被 forward 写入的末位 KV slot；offload_prompt_only=False 时不 clamp，块对齐时该块按内容 hash 存为完整。
- **影响对象**：KV状态；跨请求/跨样本数据；输出内容（静默错误）
- **影响详情**：offload_prompt_only=False 且请求结束恰在 block 边界（约 1/block_size）时，后续同内容请求（如多轮重发 transcript）命中该块，末槽为 stale/未初始化 KV，推测解码下为被拒 draft 的 KV、看似合法；无 crash，续写质量静默退化。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/54193

### INF-0551 `sglang_dcp_index_k_cache_oob_write_crash_and_residual_quality_degradation` SGLang DCP 下 index-K cache 越界写崩溃及残留质量退化

`故障` · `active` · `infra_off` · `other_stack` · `verified` · `NEW` · 发现 2026-08-28 · 旧号 6.158

- **原因对象**：KV缓存 / sglang / DSA index-K cache 分配尺寸
- **原因行为**：DSA index-K cache 按 rank-local 尺寸分配，读写却用 DCP 虚拟位置（大 dcp_size 倍），累计分配越过 max_total_num_tokens 后越界写；尺寸修复后仍有无 crash 的静默质量退化。
- **影响对象**：输出内容；可用性/进度（质量退化）
- **影响详情**：GLM-5.3-Flash（DSA）+ --dcp-size 8：触发点确定、受害内核随机的 crash（6 种签名）；修复尺寸后温度 0 输出仍退化：CoT 膨胀 4–9×且自相矛盾、翻译丢实体、超 token 预算返回空内容；仅去掉 DCP flag 即恢复。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/36886

### INF-0552 `vllm_glm53flash_multi_kernel_random_ima_single_corrupter` vLLM GLM-5.3-Flash 多内核族随机 IMA，单点损坏者未定位

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-08-29 · 旧号 6.159

- **原因对象**：KV缓存 / vllm / 待定（疑在线性注意力状态路径）
- **原因行为**：待定：推测为负载/状态依赖的单点上游内存损坏者（最可能在 linear-attention/state 路径），异步 CUDA 错误在同步点才浮出；硬件已排除（ECC 0、无 Xid、无 page retirement）。
- **影响对象**：可用性/进度（报错）
- **影响详情**：GLM-5.3-Flash 4×B200 TP4 生产流量下约 6 小时 5 次 CUDA IMA（EngineDeadError）；故障帧分布于 Triton KDA、TileLang MHC、TRT-LLM fused MoE 三个内核族，同次 crash 各 rank 不同；禁用 PDL 无效
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/54317

### INF-0556 `sglang_kimi_k3_high_pressure_cross_request_context_corruption` SGLang Kimi-K3 高并发多轮对话跨请求上下文污染

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `var` · 发现 2026-08-29 · 旧号 6.163

- **原因对象**：KV缓存 / sglang / 待定
- **原因行为**：待定：原文嫌疑层为 radix/prefix cache 索引、hierarchical 路径外的 KV 复用或调度器 batch 上下文绑定，禁用 DCP 仍复现，根因未定位。
- **影响对象**：跨请求/跨样本数据；输出内容（静默错误）
- **影响详情**：Kimi-K3 双 8×B200、fp8 KV、高并发（尤其多轮）下后续轮次回答其他请求的问题（任务 000045 答成 000030/000011），输出流畅无 crash；DCP+hierarchical-cache 与 EP8+marlin 两套正交配置均命中。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/36859
- **关联**：var:INF-0399

### INF-0558 `sglang_kda_triton_padding_remap_to_allocatable_last_slot` SGLang KDA Triton padding 哨兵重映射到可分配末槽

`故障` · `active` · `infra_off` · `other_stack` · `verified` · `NEW` · 发现 2026-08-29 · 旧号 6.165

- **原因对象**：KV缓存 / sglang / KDA Triton padding 槽位重映射
- **原因行为**：kda_triton.py 把 padding 哨兵 -1 重映射到 shape[0]-1，假设尾槽是保留哨兵，但分配器保留的是 slot 0、尾槽可分配；重映射发生在内核见到索引之前，绕过 index>=0 守卫。
- **影响对象**：KV状态；输出内容（静默错误）
- **影响详情**：存在 DP/CP padding 行且 mamba 池满压使末槽被活请求占用时，padded 行读写该活请求（或 radix checkpoint）的循环状态，输出错误且无任何错误信号；审计发现，生产未观测到。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/36880

### INF-0559 `sglang_hybrid_branch_reuse_lru_pressure_silent_collapse` SGLang hybrid Mamba 槽 LRU 驱逐致分支复用静默坍塌

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-08-29 · 旧号 6.166

- **原因对象**：KV缓存 / sglang / mamba 槽 LRU 驱逐与分支复用
- **原因行为**：分支复用要求 branch 链上 mamba 槽全部存活，但 KV 池 token 数 / chunk 大小超过 mamba cache 容量时 LRU 驱逐从链中间打断；启动期对该派生不等式无任何推导或警告。
- **影响对象**：性能/资源（性能退化）
- **影响详情**：hybrid Mamba/GDN 模型多轮/分支场景前缀命中率骤降、复用退化为重算，吞吐静默劣化；无日志无告警，监控只见命中率缓降，易被当作预期缓存行为。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/36935

### INF-0561 `vllm_hybrid_page_alignment_giant_blocks_fixed_footprint_collapse` vLLM hybrid 页对齐巨块致每请求固定足迹与并发坍塌

`故障` · `active` · `infra_off` · `config` · `verified` · `NEW` · 发现 2026-08-30 · 旧号 6.168

- **原因对象**：KV缓存 / vllm / hybrid attention 页大小对齐
- **原因行为**：为使 attention 页字节 ≥ mamba 页字节，block size 被向上对齐（GLM-5.3-Flash 达 7,808 token/页），未重审 KV 组数 × 每组每请求至少一页，约 23 组使每请求固定占约 23 巨页。
- **影响对象**：性能/资源（性能退化）
- **影响详情**：TP2/DCP2 实测 457,880-token 池仅 70 页，12-token 请求占 32.9%，并发上限 3；35k 缓存 prompt 交错 admission 后命中 23,808→0、TTFT 2.3s→7.1s；收紧 --max-model-len 无效，无报错。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/54458

### INF-0562 `sglang_lmcache_storage_keys_drop_cache_salt_tenant_isolation` SGLang LMCache 存储键丢弃 cache_salt 致跨租户共享

`故障` · `active` · `infra_off` · `config` · `verified` · `var` · 发现 2026-08-30 · 旧号 6.169

- **原因对象**：KV缓存 / sglang / LMCache 集成 lookup/store 键
- **原因行为**：LMCache 集成的 LOOKUP 与 STORE 路径都不转发 cache_salt：调 lookup_kv 时未传入作用域内的 salt，StoreMetadata 也无 salt 字段，外部键仅由 token 决定。
- **影响对象**：跨请求/跨样本数据；KV状态（静默错误）
- **影响详情**：启用 --enable-lmcache 且多租户以 cache_salt 隔离时，不同租户同前缀请求共享外部缓存前缀：第二账户同前缀请求 1,198 ms 对比重算基线 3,340 ms，形成跨租户计时侧信道；命中本身无任何异常信号。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/37160
- **关联**：var:INF-1301

### INF-0564 `vllm_spec_decode_nightly_silently_disables_hybrid_prefix_cache` vLLM nightly 推测解码使 hybrid GDN 前缀缓存静默零命中

`故障` · `active` · `infra_off` · `partial` · `verified` · `NEW` · 发现 2026-08-30 · 旧号 6.171

- **原因对象**：KV缓存 / vllm / hybrid GDN align 模式前缀缓存
- **原因行为**：推测解码（mtp/dflash）与 hybrid GDN 的 align 模式 prefix caching 组合时缓存永不命中，无 warning/日志；属 v0.24.0 之后 nightly 引入的回归，原文未定位具体代码。
- **影响对象**：性能/资源（性能退化）
- **影响详情**：hybrid GDN 模型 + spec decode + prefix caching 时 prefix_cache_hits_total 恒为 0，重复长 prompt 全额重 prefill、TTFT 不降；输出正确、无 crash，v0.24.0 同机同模型正常命中。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/54360

### INF-0568 `vllm_mooncake_multi_connector_group_index_hybrid_save_crash` vLLM MultiConnector 计数转发缺陷致 save 崩溃

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-08-31 · 旧号 6.175

- **原因对象**：KV缓存 / vllm / MultiConnector token 数转发
- **原因行为**：update_state_after_alloc 只把真实 token 数转发给首个子 connector，其余收到 0/陈旧计数；sender 线程仍按全局 g_idx 索引 per-group block_ids 而越界。
- **影响对象**：可用性/进度（报错）
- **影响详情**：MooncakeStoreConnector（MultiConnector 组合）+ hybrid 多 KV-group 模型（如 Kimi-K3）在请求完成时的 save 路径，后台 sender 线程抛 IndexError；主请求可能已完成，表现为完成后爆 traceback。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/54643

### INF-0572 `vllm_glm53_dsa_indexer_tail_cache_block_size_offload_align_impossible` vLLM GLM-5.3 indexer 块尺寸致 offload 无法启用

`故障` · `active` · `infra_off` · `config` · `documented` · `var` · 发现 2026-09-01 · 旧号 6.179

- **原因对象**：KV缓存 / vllm / KV offload 配置对齐断言
- **原因行为**：indexer tail_cache 组 block_size=4 无法被 hash 粒度 8704 整除，offloading 配置断言拒绝；--mamba-cache-mode align 不触及该组，报错还推荐已默认开启的 flag。
- **影响对象**：可用性/进度（报错）
- **影响详情**：vLLM + GLM-5.3-Flash（DSA indexer）配置 OffloadingConnector 或 LMCache 时启动期 assert 拒绝，该模型族无法启用 KV offload；按报错提示加 --enable-prefix-caching 无效，误导排障方向。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/54831
- **关联**：var:INF-0561

### INF-0575 `vllm_qwen38_flash_next_mooncake_hybrid_offload_staged_startup_failure` vLLM hybrid GDN 模型启用 Mooncake 逐级启动失败

`故障` · `active` · `infra_off` · `config` · `documented` · `var` · 发现 2026-09-01 · 旧号 6.182

- **原因对象**：KV缓存 / vllm / Mooncake 连接器 hybrid 支持
- **原因行为**：hybrid GDN 模型多 KV 组的 block 尺寸/组织与连接器统一假设冲突，mamba_cache_mode 的 align 支持矩阵未覆盖 MooncakeStoreConnector，而官方 recipe 声称支持。
- **影响对象**：可用性/进度（报错）
- **影响详情**：Qwen3.8-Flash-Next 按官方 recipe 启用 Mooncake 即启动失败：先报 block_size 不能被 hash_block_size 整除，加 --prefix-match-unit 8 后又报不支持 mamba_cache_mode != 'align'。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/54726
- **关联**：var:INF-0572

### INF-0577 `vllm_kv_offload_block_not_found_multi_day_uptime_assert_engine_dead` vLLM KV offload 数天后查不到 block 致引擎死亡

`故障` · `active` · `infra_off` · `partial` · `documented` · `var` · 发现 2026-09-03 · 旧号 6.184

- **原因对象**：KV缓存 / vllm / offload prepare_load 查找
- **原因行为**：CPU 侧 KV offload 的 prepare_load 按 block key 查 cache 并断言存在；长期运行中某 key 条目被逐出/未写入/生命周期错配（原文未定位根因），查找落空。
- **影响对象**：可用性/进度（报错）
- **影响详情**：TP8+DP3、DSV4-Pro 1M 上下文、OffloadingConnector 的生产环境运行数天后 assert block is not None 失败，EngineDeadError 使整个 DP rank 服务死亡、在途请求全失败；短测不可见。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/54914
- **关联**：var:INF-0550

### INF-0578 `vllm_glm53_flash_sparse_mla_indexer_prefill_workspace_token_vs_pool_granule` vLLM GLM-5.3 indexer workspace 漏除压缩比

`故障` · `active` · `infra_off` · `partial` · `documented` · `var` · 发现 2026-09-03 · 旧号 6.185

- **原因对象**：KV缓存 / vllm / GLM indexer prefill 缓冲大小
- **原因行为**：GLM 路径按 get_max_prefill_buffer_size() 的 token 数申请 indexer workspace，未除以 compress_ratio（KV 为 pool 粒度，deepseek_v4 已除）。
- **影响对象**：性能/资源（性能退化）
- **影响详情**：GLM-5.3-Flash + 1M 级 --max-model-len 时每 GPU 多锁约 2.6 GiB，KV 池少约 8%（修复前→后 KV 可用 29.44→32.06 GiB、池 4.50M→4.90M token），长上下文吞吐/并发下降；无 crash，needle 检索不变。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/55221
- **关联**：var:INF-0566

### INF-0580 `sglang_linear_replayssm_forces_no_buffer_prefix_cache_collapse_ttft` SGLang replayssm 强制 no_buffer 致前缀命中坍塌

`故障` · `active` · `ambiguous` · `other_stack` · `documented` · `NEW` · 发现 2026-09-03 · 旧号 6.187

- **原因对象**：KV缓存 / sglang / mamba radix no_buffer 策略
- **原因行为**：--enable-linear-replayssm 强制 --mamba-radix-cache-strategy no_buffer，该策略把 mamba checkpoint 存在后续请求够不到的 radix 树深度。
- **影响对象**：性能/资源（性能退化）
- **影响详情**：hybrid/mamba 模型 + 共享前缀 workload：rag 2048-tok 前缀命中 88.2%→65.9%、TTFT 均值 226→1063ms；fewshot 512-tok 命中 62.2%→0%；吞吐不变、输出正确。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/37834

### INF-0581 `sglang_dsv4_kv_pool_mem_usage_metric_zero_missing_finalize` SGLang DeepSeek-V4 KV 内存用量指标恒为 0

`故障` · `active` · `infra_off` · `other_stack` · `verified` · `NEW` · 发现 2026-09-03 · 旧号 6.188

- **原因对象**：KV缓存 / sglang / DSV4 KV 池 mem_usage 统计
- **原因行为**：DSV4 的 SingleKVPool/IndexerPool 分配缓冲后从不调 _finalize_allocation_log()，顶层 TokenToKVPool 也不聚合子池，mem_usage 停留在初始值 0。
- **影响对象**：指标/日志（静默错误）
- **影响详情**：SGLang 部署 DeepSeek-V4 族模型（任意配置）时 sglang:kv_cache_memory_usage_gb 恒报 0，推理正确、无报错；依赖该指标的 KV 内存监控、告警与容量规划全部失明。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/37852

### INF-0586 `sglang_glm53flash_dsa_hicache_host_roundtrip_corruption_no_spec` SGLang HiCache 往返未恢复 DSA index 缓冲致生成损坏

`故障` · `active` · `infra_off` · `other_stack` · `verified` · `var` · 发现 2026-09-04 · 旧号 6.193

- **原因对象**：KV缓存 / sglang / HiCache 往返 DSA index 缓冲
- **原因行为**：HiCache 往返只恢复 KV 与 recurrent state，不备份/恢复 DSA target+draft index 缓冲；prefix-cache 分裂落在压缩组内时，不同后缀共享的压缩 index 行互相覆写。
- **影响对象**：KV状态；输出内容（静默错误）
- **影响详情**：SGLang + GLM-5.3-Flash（DSA）开启 --hicache-size/--hicache-ratio 后，命中 host 往返 prefix 的请求静默生成损坏：掉 tool call、退化重复循环，cache 却报恢复成功；关闭 EAGLE/MTP 仍复现，只影响部分请求。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/38031
- **关联**：var:vllm_kv_offload_days_uptime_block_not_found_crash

### INF-0590 `vllm_nixlpush_post_producer_end_write_wrong_kv_http200` vLLM NixlPushConnector 过期回调写错 KV

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-09-05 · 旧号 6.197

- **原因对象**：KV缓存 / vllm / NixlPushConnector 延迟回调
- **原因行为**：握手回调、deferred-inbox drain、transfer sink 三处延迟路径持有发起时的逻辑 block id 且无“请求已放弃”检查；生产者请求结束、源 block 被复用后仍提交 WRITE。
- **影响对象**：跨请求/跨样本数据；KV状态（静默错误）
- **影响详情**：PD 分离部署中握手延迟或 lease 到期窗口内，仍在运行的 decode 请求在被另一请求内容污染的错误 KV 上继续生成；无 crash、API 返回 HTTP 200，输出看似正常但语义错误。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/55470

### INF-0591 `sglang_hicache_host_mem_guard_double_charge_rank_desync` SGLang HiCache host 池守卫跨 rank 重复计扣

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-09-05 · 旧号 6.198

- **原因对象**：KV缓存 / sglang / HiCache host 内存预算守卫
- **原因行为**：host 池预算把 psutil available 按 rank 数均分，但各 TP rank 读取时刻不同且无同步，后到者在已扣除先到者池的余量上仍除以全部 rank 数；sidecar 池守卫同病。
- **影响对象**：可用性/进度（报错）
- **影响详情**：TP>1 时物理上放得下的 host 池被 ValueError: Not enough host memory 拒绝，同配置同主机间歇启动失败，取决于 rank 到达顺序；tp8/1772 GiB 主机 --hicache-size 176 实测复现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/38156

### INF-0592 `vllm_mamba_state_retirement_stops_at_null_gap_pool_exhaustion` vLLM Mamba align 状态退休遇 null gap 即停

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-09-05 · 旧号 6.199

- **原因对象**：KV缓存 / vllm / mamba state 退休扫描
- **原因行为**：align 模式 prefill 在 committed prefix 内留下未映射的 null gap 槽，退休逻辑把第一个 null 当终止哨兵，gap 之后仍在分配账上的更老 state 永不释放。
- **影响对象**：性能/资源；可用性/进度（性能退化）
- **影响详情**：hybrid Mamba + --mamba-cache-mode align 长 uptime 下 mamba state 池慢性泄漏式耗尽，新请求无法获得 state 槽，出现容量型拒绝或退化；无 crash。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/55450

### INF-0593 `sglang_mamba_slot_headroom_overeviction_checkpoint_churn` SGLang mamba 槽按常量过量驱逐 checkpoint

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-09-05 · 旧号 6.200

- **原因对象**：KV缓存 / sglang / alloc_req_slots 驱逐下界
- **原因行为**：alloc_req_slots 按每请求固定 headroom 常量（num_reqs×3）预留并驱逐 mamba checkpoint，而非按本 batch 实际取数（prefix 命中取 2、chunk 续传取 0）。
- **影响对象**：性能/资源（性能退化）
- **影响详情**：hybrid 模型 + 紧 mamba 池 + chunked prefill 下，盈余驱逐在少数 slot 间循环，最热 checkpoint 也保不住，前缀缓存命中率静默坍塌、TTFT 退化；无错误，仅 mamba 命中率指标偏低。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/pull/38151

### INF-0596 `vllm_arc_offload_t2_only_reclaim_store_atomic_failure` vLLM ARC KV 卸载只扫 T2 致 store 失败

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-09-05 · 旧号 6.203

- **原因对象**：KV缓存 / vllm / CPU KV offload ARC 驱逐
- **原因行为**：T1 低于自适应目标时只扫描 T2，T2 无合格候选（pin/保护/耗尽）即返回 None，把偏好序实现成硬性单侧，不回退 T1 中的可驱逐条目。
- **影响对象**：性能/资源（性能退化）
- **影响详情**：T2 条目受 pin/保护压力或在 batch 中途耗尽时，尽管 T1 有足够可驱逐条目，store 仍原子性失败；CPU offload 吞吐间歇塌陷、KV 驻留 GPU 加重显存压力，失败报给调用方但根因不可见。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/55461

### INF-0597 `vllm_hybrid_mamba_state_idx_engine_lowered_block_size_oob` vLLM mamba 状态列以降级 block_size 播种

`故障` · `active` · `infra_off` · `config` · `documented` · `var` · 发现 2026-09-06 · 旧号 6.204

- **原因对象**：KV缓存 / vllm / add_request mamba 状态列播种
- **原因行为**：add_request 用 cache_config.block_size 播种 mamba 状态列，而该值已被引擎降为最小可缓存组 block size、远小于 mamba_block_size，以错误单位索引 block table。
- **影响对象**：KV状态；可用性/进度（静默错误）
- **影响详情**：hybrid mamba（如 GLM-5.3-Flash）+ prefix caching + 小 block 可缓存 attention 组时：长前缀（≥8 mamba 块）越界致 Xid 31 IMA；短前缀静默恢复无关 block 的 recurrent state，毒化后续生成。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/55600
- **关联**：var:INF-0524

### INF-0598 `sglang_hybrid_mla_kv_read_before_hicache_loadback` SGLang MLA KV 读取未等 HiCache load-back

`故障` · `active` · `infra_off` · `other_stack` · `verified` · `var` · 发现 2026-09-06 · 旧号 6.205

- **原因对象**：KV缓存 / sglang / HybridLinearKVPool 读路径
- **原因行为**：wrapper 包裹 MLATokenToKVPool 时禁用了内层的 HiCache load-back 等待，但 get_mla_kv_buffer() 转发读取前不做外层 wait，异步 load-back 未完成即读 MLA KV。
- **影响对象**：KV状态；隐状态（静默错误）
- **影响详情**：hybrid MLA + HiCache host offload 下读到未完成传输的陈旧/零值 KV，注意力输出静默损坏；受控实验修复前 20/20 次读错；端到端服务中投影先跑留出时间，故间歇性显形。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/pull/38217
- **关联**：var:INF-0586

### INF-0599 `sglang_unified_radix_cache_finished_empty_key_insert_root_repoint` SGLang 统一 radix 缓存 finished 路径缺空键早退

`故障` · `active` · `infra_off` · `other_stack` · `verified` · `NEW` · 发现 2026-09-06 · 旧号 6.206

- **原因对象**：KV缓存 / sglang / cache_finished_req 早退守卫
- **原因行为**：cache_finished_req 缺 effective_cache_len<=0 早退：空键仍 insert() 并把 last_node 指向未持锁 root；ReplaySSM 下 cache_len 可为负，负切片插入错键。
- **影响对象**：KV状态（静默错误）
- **影响详情**：短 mamba 请求未达 track boundary，或 ReplaySSM no-buffer + session radix cache 时，无 crash：缓存树锁语义被静默破坏并插入错键，为后续驱逐/复用路径埋下状态分歧。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/pull/38191

### INF-0605 `vllm_hybrid_gdn_align_boundary_checkpoint_nan_prefix_hit` vLLM GDN 块边界 checkpoint 损坏致 NaN 输出

`故障` · `active` · `infra_off` · `config` · `verified` · `NEW` · 发现 2026-09-07 · 旧号 6.212

- **原因对象**：KV缓存 / vllm / GDN align 块边界 checkpoint
- **原因行为**：前一个 prefill 恰在 N*816+r（r 为 4–10 的偶数）处结束时写下的块边界 GDN 状态 checkpoint 是坏的，后续请求前缀命中后从该边界恢复；具体根因待定。
- **影响对象**：KV状态；输出分布（数值异常）
- **影响详情**：Qwen3.5/3.8 hybrid GDN + prefix caching（align）下，后续请求从首个未缓存步起全词表 NaN logits，持续采样“!”至 max_tokens；坏块再入缓存使重试确定性失败；vllm:corrupted_requests_total 递增。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/55766

### INF-0607 `vllm_glm53flash_pd_disagg_repetitive_offtopic_outputs` vLLM GLM-5.3-Flash PD 分离部署输出离题或无限重复

`故障` · `active` · `infra_off` · `partial` · `documented` · `var` · 发现 2026-09-07 · 旧号 6.214

- **原因对象**：KV缓存 / vllm / 待定
- **原因行为**：待定（open issue 未附根因定位；原文仅称不排除 kv-connector 传输或混合架构状态缺陷）。
- **影响对象**：输出内容（质量退化）
- **影响详情**：8×H100 上 PD 分离 + kv-connector 部署时，同 prompt 频繁产出完全无关内容或同短语无限重复（“好的好的好的…”）；非分离单节点部署也出现但较轻；无 crash。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/55689
- **关联**：var:INF-0556

### INF-0609 `vllm_hybrid_mamba_mtp_drafter_backend_reorder_threshold_min_corruption` vLLM Mamba2+MTP 重排阈值取 min 致状态槽损坏

`故障` · `active` · `infra_off` · `partial` · `verified` · `var` · 发现 2026-09-08 · 旧号 6.216

- **原因对象**：KV缓存 / vllm / reorder 阈值跨组 min 聚合
- **原因行为**：runner 对各注意力组 reorder_batch_threshold 取 min，drafter 后端报 1 压低 Mamba2 所需的 4，draft-decode 行被切入 prefill 内核，推测状态槽未写入。
- **影响对象**：KV状态；输出内容（静默错误）
- **影响详情**：Nemotron-3.5-Lightning 等 hybrid Mamba2 + MTP 在混合 prefill/decode 负载下 0.2–1% 请求（定向触发最高 7%）从约第 5 个 token 起输出 word salad 或特殊 token 空循环直至结束；无 crash、无日志。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/55894
- **关联**：var:INF-0605

### INF-0611 `sglang_hicache_backup_thread_dies_on_backend_exception_leak` SGLang HiCache 备份线程遇后端异常即死

`故障` · `active` · `infra_off` · `other_stack` · `verified` · `var` · 发现 2026-09-08 · 旧号 6.218

- **原因对象**：KV缓存 / sglang / HiCache 备份线程异常边界
- **原因行为**：backup_thread_func（含 HybridCacheController 覆盖）只捕获 queue.Empty，后端写页抛出的任何异常都会杀死 daemon 线程；batch_set 软失败也不回传失败信息。
- **影响对象**：性能/资源；指标/日志（静默错误）
- **影响详情**：任一存储后端（Mooncake/NIXL/3FS/file 等）一次写页异常后，备份操作永远滞留 backup_queue、不再写 L3，host 内存永不释放直至池耗尽；无日志指向死线程，hicache_* 指标仍显示备份在途。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/38428
- **关联**：var:INF-0531

### INF-0612 `sglang_unified_radix_backuped_stub_prefix_never_queries_l3` SGLang backuped stub 前缀从不查询 L3

`故障` · `active` · `infra_off` · `other_stack` · `verified` · `var` · 发现 2026-09-08 · 旧号 6.219

- **原因对象**：KV缓存 / sglang / backuped stub 锚定路径
- **原因行为**：对 backuped-only stub 节点，prefetch 路径视其跨度为已匹配而不发存储查询，load-back 路径因 stub 无 host 索引拒绝该跨度，两路径语义矛盾。
- **影响对象**：性能/资源（性能退化）
- **影响详情**：前缀 KV 被 L1/L2 双双驱逐但存储备份已完成后，重发同前缀时全量重算 prefill、L3 命中率静默归零、TTFT 退化；结果正确；flush_cache 或重启后同请求可命中 L3。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/38452
- **关联**：var:INF-0530

### INF-0613 `sglang_blockstored_kv_event_cache_salt_undecodable_public_types` SGLang KV 事件公共类型丢弃 cache_salt

`故障` · `active` · `infra_off` · `other_stack` · `verified` · `var` · 发现 2026-09-08 · 旧号 6.220

- **原因对象**：KV缓存 / sglang / KV 事件公共 msgspec 类型
- **原因行为**：导出的 BlockStoredWithMetadata 把 metadata 设为必填且与基类同 tag，混合流在类型层不可表达；经 KVEventBatch 解码时 salted 事件被当作普通 BlockStored，salt 被丢弃。
- **影响对象**：KV状态；跨请求/跨样本数据（静默错误）
- **影响详情**：无 crash；外部 KV 事件消费者（监控、审计、分布式缓存协调器）用公共类型解码时看不到 cache_salt，依赖 salt 做键隔离的下游会把 salted 与 unsalted 条目混入同一命名空间，且无文档提示。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/38561
- **关联**：var:INF-0562

### INF-0615 `sglang_pd_disagg_dcp_decode_retraction_widened_index_cpu_backup_oob` SGLang DCP>1 下 retract 备份用错 KV 索引

`故障` · `active` · `infra_off` · `other_stack` · `verified` · `var` · 发现 2026-09-09 · 旧号 6.222

- **原因对象**：KV缓存 / sglang / retract KV CPU 备份/恢复索引
- **原因行为**：decode retraction 把 req_to_token 中 DCP 扩宽的逻辑 KV 索引直接用作 MLA KV 池 get_cpu_copy/load_cpu_copy 的物理行索引，缺少按 owner rank 局部化的变换。
- **影响对象**：KV状态；可用性/进度（静默错误）
- **影响详情**：PD 分离 decode 节点 DCP>1、容量压力触发 retract 时：索引越界则 CUDA device-side assert 崩溃；索引碰巧在界内则静默备份/恢复错误物理行的 KV 数据。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/38645
- **关联**：var:INF-0550

### INF-0618 `sglang_swa_mamba_branch_checkpoint_minlen_mismatch_wrong_state_attach` SGLang 混合分支缓存取 min 长度把晚期 Mamba 状态挂错键

`故障` · `active` · `infra_off` · `other_stack` · `verified` · `NEW` · 发现 2026-09-10 · 旧号 6.225

- **原因对象**：KV缓存 / sglang / Full/SWA/Mamba 分支点缓存插入
- **原因行为**：SWA 与 Mamba 的分支边界独立计算，调度器只按 Mamba 目标选 checkpoint；插入取各组件长度的 min，对不可切片的 Mamba 状态也照用，把 state192 挂到 key96 上。
- **影响对象**：KV状态；输出分布（静默错误）
- **影响详情**：后续请求命中共享前缀键时从含未见 token 的 Mamba 状态恢复；H200 + Inkling checkpoint 复现，warm-cache logprobs 三组输入全部偏离、greedy 输出一组改变，冷缓存重复完全一致。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/38815

### INF-0622 `vllm_mamba_state_blocks_unzeroed_reallocation_cross_tenant_poisoning` vLLM mamba/KDA 状态块分配不清零致跨租户毒化

`故障` · `active` · `infra_off` · `partial` · `verified` · `NEW` · 发现 2026-09-11 · 旧号 6.229

- **原因对象**：KV缓存 / vllm / Mamba/KDA 状态块分配零化
- **原因行为**：BlockPool 分配的 mamba/KDA 状态块不清零，零化机制的类型守卫只覆盖 AttentionSpec：MambaManager 绕过新块记录，KVBlockZeroer 跳过非 attention 层且不解 tuple 绑定。
- **影响对象**：KV状态；跨请求/跨样本数据（静默错误）
- **影响详情**：abort 中断长 prefill（约 40–240k）后立即重试，1–2/36 概率以上一租户半写状态起步，输出无界多语言垃圾直至引擎重启；请求正常完成还会污染后续缓存提交。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/56524

### INF-0623 `vllm_mamba_prefix_hash_published_before_commit_fence_abort_retry_poison` vLLM Mamba 前缀 hash 先于写入发布致重试毒化

`故障` · `active` · `infra_off` · `partial` · `verified` · `NEW` · 发现 2026-09-11 · 旧号 6.230

- **原因对象**：KV缓存 / vllm / Mamba 前缀缓存 hash 发布时序
- **原因行为**：MambaManager.cache_blocks 在调度时就把状态块 hash 发布进前缀缓存，写该状态的 chunk 稍后才执行（abort 则永不执行）；一步有效的守卫在写入仍在途时已被清除。
- **影响对象**：KV状态；跨请求/跨样本数据（静默错误）
- **影响详情**：abort 长 prefill 后 1–2 秒内重试同前缀，命中已发布未写入的 hash，以垃圾 recurrent state 起步输出无界垃圾；重试正常完成后毒化 hash 被提交为有效，同前缀后续请求持续中毒至重启。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/56525

### INF-0627 `sglang_hicachefile_hybrid_pool_existence_min_endpoint_misreport` SGLang HiCache 文件后端混合池存在性查询误报

`故障` · `active` · `infra_off` · `yes` · `verified` · `NEW` · 发现 2026-09-13 · 旧号 6.234

- **原因对象**：KV缓存 / sglang / HiCache 文件后端存在性查询
- **原因行为**：batch_exists_v2 对混合池取各池最长可用前缀的 min，而非各池合法端点集合的交集，对 SWA/Mamba trailing 池合法端点中的洞失明，把无对象的端点报为可命中。
- **影响对象**：KV状态（静默错误）
- **影响详情**：混合池前缀存在性查询误报可恢复长度（如 SWA {3}/Mamba {2} 报 2，应为 0），恢复路径可能拿到截断或缺失的状态；报告为空文件契约级复现，未声称端到端输出错误。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/39147

### INF-0629 `vllm_mixed_prompt_embeds_selection_mask_omitted_from_prefix_cache_key` vLLM prompt embeds 选择 mask 未纳入前缀缓存键

`故障` · `active` · `infra_off` · `partial` · `verified` · `NEW` · 发现 2026-09-15 · 旧号 6.236

- **原因对象**：KV缓存 / vllm / prompt embeds 前缀缓存键计算
- **原因行为**：前缀缓存键只哈希归一化后的 token ids 与供给 embedding 张量，未纳入逐位置选择 mask prompt_is_token_ids；embedding 位被归一化为合法 token id 0，不同 mask 的请求键碰撞。
- **影响对象**：跨请求/跨样本数据；KV状态（静默错误）
- **影响详情**：开启 prompt embeds 与前缀缓存时，token ids 与供给张量相同但 mask 不同的请求错误命中缓存（实测 32 token），输出由 `[9,0]*8` 变为 `[753]*16`；temp=0 下结果取决于谁先写缓存，无报错。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/56655

### INF-0632 `vllm_hisparse_host_resident_kv_decode_token_noise_first_token_correct` vLLM HiSparse host KV 解码自第 2 token 起噪声

`故障` · `active` · `infra_off` · `other_stack` · `verified` · `NEW` · 发现 2026-09-15 · 旧号 6.239

- **原因对象**：KV缓存 / vllm / HiSparse host KV 读路径
- **原因行为**：根因未定位；已隔离到 decode 侧读取 host-mirror KV 的路径（prefill 完好），原文候选为 host 池取回块的布局/尺度错配，或异步取回与 decode 调度的可见性时序。
- **影响对象**：KV状态；输出内容（静默错误）
- **影响详情**：GLM-5.2 NVFP4-FP8 等 DSA 模型开 HiSparse（TP2+）时，每个 completion 自第 2 个生成 token 起全为标点噪声，首 token 与对照一致，max_tokens=1 链式请求输出正确；确定性复现，GSM8K 0.0000 vs 0.9409，无日志。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/57019

### INF-0633 `sglang_hierarchical_cache_host_tier_hit_gdn_mamba_wrong_output_other_pass_voice` SGLang host 层缓存命中后 hybrid GDN 模型答非所问

`故障` · `active` · `infra_off` · `other_stack` · `verified` · `var` · 发现 2026-09-16 · 旧号 6.240

- **原因对象**：KV缓存 / sglang / HiCache host 层取回（GDN 状态）
- **原因行为**：hierarchical cache 从 host tier 取回时需同时恢复注意力 KV 与 GDN/Mamba 状态，host 路径的取回/重组或 state 绑定与 device 路径不一致，恢复出另一遍计算的状态。
- **影响对象**：KV状态；输出内容（静默错误）
- **影响详情**：prefix 被 device pool 驱逐后从 host tier 命中时，byte-identical 请求输出流畅但答非所问（20/20 全错），cached_tokens 与 finish_reason 均正常；冷启动与 device 命中全对。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/39830
- **关联**：var:INF-0586

### INF-0634 `vllm_mamba_block_size_explicit_conv_ssm_forward_strided_slice_oob_engine_death` vLLM 显式 mamba-block-size 对齐切片为空致引擎死亡

`故障` · `active` · `infra_off` · `config` · `verified` · `var` · 发现 2026-09-17 · 旧号 6.241

- **原因对象**：KV缓存 / vllm / MambaMixer2 块对齐状态切片
- **原因行为**：conv_ssm_forward 在显式 mamba block size 下修正 first_aligned_chunk 的对齐换算在 chunk 边界算出越界起点，varlen_states 步长切片退化为零长度，与索引形状不匹配。
- **影响对象**：可用性/进度（报错）
- **影响详情**：hybrid Mamba2 模型（granite-4.0-h-tiny、Falcon-H1-Tiny）开显式 --mamba-block-size 与前缀缓存，8 并发长 prompt 首次调度即杀死 EngineCore，此后所有请求 HTTP 500 EngineDeadError。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/57266
- **关联**：var:vllm_hybrid_mamba_engine_block_size_post_division_seeding_oob

### INF-0636 `vllm_nixl_pd_base_addr_dedup_folds_hybrid_layers_shared_storage_silent_garble` vLLM NIXL 区域仅按 base_addr 去重致传输花屏

`故障` · `active` · `infra_off` · `partial` · `verified` · `NEW` · 发现 2026-09-19 · 旧号 6.243

- **原因对象**：KV缓存 / vllm / NIXL base_worker 区域注册去重
- **原因行为**：传输区域注册仅按 base_addr 去重，折叠分支只在 is_mla_region 为真时刷新 block_len/stride/num_blocks；共享存储基址但块长不同的 hybrid 层被折叠进先注册区域的几何。
- **影响对象**：KV状态；输出内容（静默错误）
- **影响详情**：DeepSeek-V4.1-Flash 走 NIXL P/D 分离（2P×TP8+2D×TP8）时，compressor state 等按错误块长截断错位传输，输出全量乱码或 content/reasoning 为空，无错误日志；全注意力模型同拓扑正常。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/57661

### INF-0637 `vllm_reset_prefix_cache_success_cpu_offload_tier_keeps_serving_blocks` vLLM reset_prefix_cache 不清 CPU 卸载层

`故障` · `active` · `infra_off` · `config` · `verified` · `NEW` · 发现 2026-09-19 · 旧号 6.244

- **原因对象**：KV缓存 / vllm / reset_prefix_cache 跨层失效
- **原因行为**：reset_prefix_cache 端点只调用引擎内 KVCacheManager 的失效路径，不向已注册的 KV connector 广播失效，CPU 卸载层的块生存期独立于引擎侧缓存。
- **影响对象**：指标/日志；测试判定（静默错误）
- **影响详情**：配置 SimpleCPUOffloadConnector 时 /reset_prefix_cache 返回 success，但同 prompt 下一请求由 CPU 层命中（1024-token 实测 external hits +1008）；以 reset 为冷基线的 TTFT/吞吐测量测到暖路径。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/57727

### INF-0639 `transformers_dynamic_cache_offloading_hybrid_gdn_state_roundtrip_wrong_logits` transformers 缓存卸载往返损坏 GDN 状态致 logits 错

`故障` · `active` · `infra_off` · `config` · `verified` · `var` · 发现 2026-09-19 · 旧号 6.246

- **原因对象**：KV缓存 / transformers / DynamicCache 卸载 GDN 状态
- **原因行为**：DynamicCache offloading 路径对 linear_attention 层 recurrent/conv state 的 CPU 卸载→读回往返引入数值损坏：prefill 写入一致，坏在读回。
- **影响对象**：KV状态；输出分布（静默错误）
- **影响详情**：hybrid GDN 模型开 offloading=True 时，prefill 与关闭时逐位一致，首次 cache 复用起 logits max_abs_diff 达 2.12 并逐步放大（第 3 步 17.4），argmax 从第 1 步分歧，确定性复现；单步验证全绿。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/huggingface/transformers/issues/48947
- **关联**：var:INF-0633

### INF-0643 `vllm_engine_resolved_prefix_match_unit_not_propagated_to_workers_mamba_checkpoint_dropped` vLLM 匹配单位未下发致 mamba 前缀 checkpoint 丢弃

`故障` · `active` · `infra_off` · `config` · `verified` · `NEW` · 发现 2026-09-21 · 旧号 6.250

- **原因对象**：KV缓存 / vllm / mamba checkpoint 位置换算
- **原因行为**：EngineCore 把前缀匹配单位解析为各组 block size 的 GCD（16）却不传给 worker；worker 回退用本地 block_size（1600）算 checkpoint 位置，floor 到 0 被判无效而丢弃。
- **影响对象**：性能/资源（性能退化）
- **影响详情**：混合 KV 组模型（attention 16 + mamba 1600，如 Qwen3.6-27B）在默认 prefix_match_unit=None 下，mamba 前缀 checkpoint 全部失效、命中退化为重算；无错误无日志，仅命中率异常低，弹性 EP 新建 worker 同样受影响。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/58020

### INF-0644 `vllm_lmcache_kv_load_failure_hetero_attention_assert_kills_enginecore` vLLM LMCache KV 加载失败触发 assert 杀死引擎

`故障` · `active` · `infra_off` · `other_stack` · `verified` · `var` · 发现 2026-09-21 · 旧号 6.251

- **原因对象**：KV缓存 / vllm / KV load 失败处理与 xfer 完成回调
- **原因行为**：KV load 失败时调度器先移除请求，随后异步到达的传输完成回调对已删除请求执行裸 assert；失败处理与完成事件之间没有取消或序号屏障，也未走降级重算。
- **影响对象**：可用性/进度（报错）
- **影响详情**：LMCache 后端读错误（NVMe/hipFile I/O，failure_policy=fail）在 Gemma 4 等多 KV 组异构注意力模型上触发 AssertionError，EngineCore 死亡，在飞与后续请求全部 EngineDeadError，须重启服务。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/57938
- **关联**：var:INF-0532

### INF-0645 `vllm_partial_block_kv_event_parent_hash_never_emitted_chain_unreconstructable` vLLM partial 块 KV 事件引用从未上报的 parent hash

`故障` · `active` · `infra_off` · `config` · `verified` · `var` · 发现 2026-09-22 · 旧号 6.252

- **原因对象**：KV缓存 / vllm / BlockPool partial 块 KV 事件
- **原因行为**：满块事件按 block_size（1536）发布，而哈希链按 hash_block_size（128）步进，中间哈希节点从未发事件，partial 块的 BlockStored 却引用 128 粒度的父 hash。
- **影响对象**：KV状态（静默错误）
- **影响详情**：开 KV cache events 且 block_size≠hash_block_size 时，1800-token 请求只发 H1536 与 H1792 两个事件，H1792 的 parent H1664 悬空，外部消费者无法重建前缀链，复用退化或错判；服务本身无信号。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/58119
- **关联**：var:INF-0613

### INF-0646 `vllm_partial_prefix_cache_promotion_events_parent_removed_in_same_batch` vLLM 部分前缀提升事件的 parent 指向同批已移除 hash

`故障` · `active` · `infra_off` · `config` · `verified` · `var` · 发现 2026-09-23 · 旧号 6.253

- **原因对象**：KV缓存 / vllm / cache_partial_block 提升事件
- **原因行为**：部分前缀条目提升到更晚边界时，同一事件批先发旧边界 BlockRemoved，再发新边界 BlockStored，其 parent_block_hash 恰指向刚被移除的 hash。
- **影响对象**：KV状态（静默错误）
- **影响详情**：开 KV cache events 时，外部消费者在提升场景收到自相矛盾的事件批，父引用无法解析、前缀链断裂；生成正常、无 crash，CPU 单元级可复现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/58394
- **关联**：var:INF-0645

### INF-0647 `vllm_mooncake_connector_min_collapsed_block_size_misread_as_physical_hybrid_pd_transfer_break` vLLM Mooncake 误把塌缩逻辑 block_size 当物理尺寸

`故障` · `active` · `infra_off` · `other_stack` · `verified` · `var` · 发现 2026-09-23 · 旧号 6.254

- **原因对象**：KV缓存 / vllm / Mooncake connector 传输块尺寸
- **原因行为**：hybrid 模型 KV 初始化后 core.py 把 cache_config.block_size 塌缩为各 group 最小值，Mooncake connector 却把该逻辑值当作物理传输块尺寸，块 id 换算粒度错误。
- **影响对象**：可用性/进度；KV状态（报错）
- **影响详情**：混合 block size 模型 + Mooncake PD 分离：GPU 上 worker 初始化即 assert；Ascend 910B 上首次 P2P KV 传输 assert 杀死 decode 侧 EngineCore、API 返回 500；若绕过 assert，传输内容会按错误边界切分。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/58362
- **关联**：var:INF-0604

### INF-0649 `pd_disagg_mla_target_gqa_draft_kv_asymmetric_tp_mistransfer_vllm_nixl_sglang_mooncake` PD 分离 MLA target+GQA draft 异构 TP 传输错位

`故障` · `active` · `infra_off` · `other_stack` · `verified` · `NEW` · 发现 2026-09-24 · 旧号 6.256

- **原因对象**：KV缓存 / vllm / Nixl/Mooncake draft KV 映射
- **原因行为**：传输层把 MLA 复制假设套到按头切分的 GQA draft 区域：vLLM Nixl 全区域按 kv_heads=1 映射，SGLang Mooncake 把 P 侧 item length 当 D 侧 stride。
- **影响对象**：KV状态；可用性/进度；性能/资源（报错）
- **影响详情**：prefill TP < decode TP 时：vLLM 握手拒绝正确几何、接受错误几何，7/8 decode rank 读错 draft 头致接受率下降但输出正确；SGLang draft 按错误步长落址，B300 实测 28/256 请求 KV 传输失败。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/58470；https://github.com/sgl-project/sglang/issues/41038

### INF-0650 `ttmetal_gemma4_shared_scheduler_chunk_multi_request_ragged_offset_cross_request_kv_mismatch` tt-metal Gemma4 多请求共享 prefill 步时 KV 错配

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-09-25 · 旧号 6.257

- **原因对象**：KV缓存 / tt-metal / 共享 prefill 步每请求 KV 偏移
- **原因行为**：多个请求共享 prefill 步时，后续请求 chunk 边界非 2048 倍数（ragged 偏移），每请求 K/V 定位在共享调度路径上错配；精确内核位点未定位。
- **影响对象**：跨请求/跨样本数据；KV状态；输出内容（静默错误）
- **影响详情**：Gemma4-31B 经 vLLM TT plugin 在 P150x8 开 chunked prefill，多请求同步 prefill 时 needle 测试 12 请求错 5–6 个（错答或退化重复）；单请求跨步 split-prefill 正确，非 speculation 引入。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/57873

### INF-0651 `vllm_simplecpu_offload_misaligned_explicit_block_size_write_only_offload` vLLM SimpleCPU offload 块尺寸不对齐致只写不读

`故障` · `active` · `infra_off` · `config` · `documented` · `var` · 发现 2026-09-26 · 旧号 6.258

- **原因对象**：KV缓存 / vllm / SimpleCPU offload 块映射
- **原因行为**：mamba 页对齐使实际 attention block size（如 896）≠ 显式 --block-size；SimpleCPU connector 按显式值构建读写映射，写路径能存入，读路径块查找永远 miss。
- **影响对象**：性能/资源（性能退化）
- **影响详情**：hybrid GDN 模型设 VLLM_USE_SIMPLE_KV_OFFLOAD=1 且显式 --block-size 不对齐时，CPU RSS 持续增长、external_kv_transfer 恒 0，逐出后重问耗时约等于全冷 prefill；无错误无警告，省略该 flag 即正常。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/58653
- **关联**：var:INF-0620

### INF-0652 `vllm_dflash2_prefix_cache_hit_permanent_acceptance_zero_hybrid_mamba_state_boundary_misregistration` vLLM DFlash2 前缀命中后 mamba 状态索引错位致接受率永久为 0

`故障` · `active` · `infra_off` · `config` · `documented` · `var` · 发现 2026-09-27 · 旧号 6.259

- **原因对象**：KV缓存 / vllm / mamba state 索引播种
- **原因行为**：running mamba state 索引用已降为 DFlash2 SW 组 64 的 cache_config.block_size 而非 mamba block size 播种，前缀命中时恢复错误 recurrent state。
- **影响对象**：KV状态；性能/资源；可用性/进度（性能退化）
- **影响详情**：v0.30.0 + hybrid GDN + DFlash2 + 前缀缓存：首个前缀命中后 acceptance 塌到 0% 且永不恢复，drafter 照常出 draft 白付 verify 成本；x86 上长命中直接 IMA 杀引擎；v0.29.0 正常。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/58894
- **关联**：var:INF-0597

### INF-0654 `sglang_strip_thinking_cache_retraction_release_kv_cache_frees_radix_owned_slots_double_alloc` SGLang --strip-thinking-cache 致 KV 槽双分配

`故障` · `active` · `infra_off` · `other_stack` · `verified` · `NEW` · 发现 2026-09-29 · 旧号 6.261

- **原因对象**：KV缓存 / sglang / owned_kv_len 与 KV 释放
- **原因行为**：owned_kv_len 在有 reasoning token 后封顶于 prompt 长度；被 retract 的请求重 prefill 后整段插入 radix tree，释放时把属于 tree 节点的槽当超额分配释放，造成双释放。
- **影响对象**：KV状态；跨请求/跨样本数据（静默错误）
- **影响详情**：开 --strip-thinking-cache 且发生 retraction 时，节点驱逐二次释放使 available_size 可超池大小，alloc 可把同一 KV slot 发给两个请求，静默跨请求上下文污染；本应捕获的断言被跳过。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/41617

### INF-0656 `vllm_nixlconnector_heterogeneous_tp_split_kv_slot_layout_silent_misread` vLLM NixlConnector 异构 TP 字节切分忽略页内布局致静默乱码

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-09-29 · 旧号 6.263

- **原因对象**：KV缓存 / vllm / NixlConnector KV 块切分与握手
- **原因行为**：NixlConnector 按字节切分 KV block，假设块内为 head-first 布局；ROCM_AITER_FA shuffle、ROCM_ATTN、B12X 的 K/V 为双 slot 布局，握手也看不到页内字节排布。
- **影响对象**：KV状态；输出内容（静默错误）
- **影响详情**：MI300X 实测 standard 页 prefill → shuffled 128-token 页 decoder 时握手与传输全通过，P→D 30/30 输出 gibberish（本地 30/30 正确）且无任何错误；异构 TP 字节切分形态仅读源码定位。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/59110

### INF-0657 `vllm_v2_runner_preemption_updates_overwrite_pages_before_connector_callback` vLLM V2 runner 抢占回调前更新请求致页被覆写

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-10-09 · 旧号 6.264

- **原因对象**：KV缓存 / vllm / execute_model 回调顺序
- **原因行为**：execute_model() 先执行 update_requests()（可清零新块或做 COW 拷贝）再调 connector 的 handle_preemptions()，覆写已提交，回调内等待也保护不了旧页。
- **影响对象**：KV状态（静默错误）
- **影响详情**：抢占/驱逐时，KV connector 异步 save 仍在读取的旧物理页可被覆写，形成潜在静默 KV 损坏窗口；CPU mock 确定性复现，尚无观测到的输出损坏。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/59409

### INF-0658 `vllm_p2p_kv_offload_store_timeout_unpins_slots_under_inflight_transfer_reports_success` vLLM P2P kv_offload store 超时解钉在途传输槽仍报成功

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-10-09 · 旧号 6.265

- **原因对象**：KV缓存 / vllm / P2P kv_offload store 超时清理
- **原因行为**：超时 sweep 把超 30 s 的 store job 判失败并解钉主槽，不检查它是否挂在在途 RDMA 传输上；传输完成时 job 已移除被跳过，round 仍报 success=True。
- **影响对象**：KV状态；输出内容（静默错误）
- **影响详情**：传输超过 30 s（如降级 RDMA 路径）时，传输期间主槽可被复用覆写，消费者从被覆写块解码且传输报成功，静默 KV 损坏，无 crash；由代码评审推导。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/59569

### INF-0659 `vllm_lhbnc_native_cpu_offload_stride_heuristic_offloads_head0_layer0_only` vLLM LHBNC 布局 CPU 卸载误判 packed 仅存 head0

`故障` · `active` · `infra_off` · `other_stack` · `verified` · `var` · 发现 2026-10-09 · 旧号 6.266

- **原因对象**：KV缓存 / vllm / register_kv_caches 布局判定
- **原因行为**：register_kv_caches 按“block stride < page size 即 packed”启发式把 LHBNC（恰为 packed 的反面）误判为 packed，每块只 offload layer 0 的 head 0。
- **影响对象**：KV状态；输出内容（静默错误）
- **影响详情**：设 VLLM_KV_CACHE_LAYOUT=LHBNC 并开 native CPU offload 时，GPU→CPU 字节仅为 1/384；驱逐后重发同 prompt 命中回读，输出 gibberish，块仍计为 stored+hit，无告警。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/60004
- **关联**：var:INF-0651

### INF-0660 `vllm_glm52_fp8_prefix_cache_reuse_empty_content_incoherent_reasoning` vLLM GLM-5.2-FP8 前缀命中返回空正文与乱语 reasoning

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-10-09 · 旧号 6.267

- **原因对象**：KV缓存 / vllm / fp8 KV 前缀缓存复用路径
- **原因行为**：根因未定位；单变量对照只表明缺陷在 fp8 KV + 前缀缓存复用路径上，冷 prefill 正常。
- **影响对象**：输出内容（静默错误）
- **影响详情**：GLM-5.2-FP8 TP8 重复发送约 43k token 请求，命中前缀缓存后返回 content 为 null、reasoning 为无条理多语文本、finish_reason=stop，HTTP 200 无 traceback；强制冷 prefill 0/50 失败。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/60494

### INF-0661 `vllm_dsv4_flash_prefix_cache_reuse_tool_call_format_failures_replica_local` vLLM DSV4-Flash 前缀缓存复用请求 tool-call 格式失败

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-10-09 · 旧号 6.268

- **原因对象**：KV缓存 / vllm / 前缀缓存复用路径（机制未知）
- **原因行为**：机制未知；失败集中于前缀缓存复用请求且限于个别 replica，可见偏差始于 </think> 后首 token，“缓存头块漂移”假设已被日志否定。
- **影响对象**：输出内容（静默错误）
- **影响详情**：v0.27.1 生产中命中前缀缓存的 agentic 请求以非 DSML 语法输出 tool call 而丢失，同刻两 replica 0/35 vs 17/19；短共享头 + 长未命中后缀失败率 63%，cache_salt 隔离后 44/55→0/37（变量混杂）。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/59926

### INF-0665 `vllm_recoverssm_align_exact_block_boundary_commits_state_to_unwritten_block` vLLM RecoverSSM align 块边界处把 state 提交到未写块

`故障` · `active` · `infra_off` · `partial` · `verified` · `var` · 发现 2026-10-09 · 旧号 6.272

- **原因对象**：KV缓存 / vllm / RecoverSSM align 状态提交列号
- **原因行为**：align 模式把最终提交的 state 写入 floor(x/B) 列，而 align 路径其余部分用 ceil(x/B)−1；两约定仅在 x 为 B 整倍数时分歧，此时 state 提交到本请求从未写过的块。
- **影响对象**：KV状态；输出内容（静默错误）
- **影响详情**：spec decode 下 verify 步恰在块边界全接受后，下一步读到垃圾或他人的 recurrent/conv state；GLM-5.3 KDA 长 prompt 4 并发实测 10–11/16 输出损坏（mid-word 丢上下文后单 token 重复到 max_tokens）。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/59933
- **关联**：var:INF-0652

### INF-0666 `trtllm_pytorch_fp8_cross_kv_pool_overrun_ima_or_silent_wrong_tokens` TRT-LLM fp8 cross-KV 池越界：崩溃或静默错 token

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-10-12 · 旧号 6.273

- **原因对象**：KV缓存 / tensorrt-llm / cross-attention KV 池
- **原因行为**：cross-attention KV 池的 block 数按 fp8 字节尺寸计算，但每块访问宽度按 bf16 步长，访问越出池边界。
- **影响对象**：KV状态；输出内容（静默错误）
- **影响详情**：PyTorch backend encoder-decoder + KvCacheConfig(dtype="fp8") 下，池小于 2 倍在飞需求时进程 IMA 崩溃，或越界落入其他映射内存时静默返回错误 token；池余量充足时被掩盖，同配置随负载在崩溃与静默损坏间漂移。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/NVIDIA/TensorRT-LLM/issues/19861

### INF-0667 `vllm_qwen38_27b_nvfp4_dflash2_prefix_cache_hit_single_token_loop` vLLM NVFP4+DFlash2 前缀命中后单 token 无限重复

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-10-12 · 旧号 6.274

- **原因对象**：KV缓存 / vllm / 前缀命中×spec 状态恢复路径
- **原因行为**：0.30.0 起 prefix 缓存命中与 spec-decode 状态恢复交互的路径发生静默损坏；具体内核未定位，报告者的五因子排除矩阵完整。
- **影响对象**：输出内容；KV状态（质量退化）
- **影响详情**：Qwen3.8-27B NVFP4 + DFlash2 下第三次请求起（prompt 从缓存块续起）输出单 token（duct）无限重复，drafter 接受长度恒为 1.00；前两次冷 prefill 正常；v0.29.0 正确，v0.30.0/0.31.0 出现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/60174

### INF-0670 `sglang_nvfp4_hicache_host_pool_element_vs_byte_geometry_draft_kv_garbage` SGLang NVFP4 HiCache 回载几何错位致 draft KV 损坏

`故障` · `active` · `infra_off` · `config` · `verified` · `NEW` · 发现 2026-10-12 · 旧号 6.277

- **原因对象**：KV缓存 / sglang / pool_host/mha.py host 池几何
- **原因行为**：pool_host/mha.py 按 device 元素数而非字节数推算 host 池行几何，未考虑 NVFP4 每字节 packed 2 元素，缺少 FP4-aware host pool；KV 块逐出后回载时几何错位。
- **影响对象**：KV状态；性能/资源（性能退化）
- **影响详情**：NVFP4 KV + hierarchical cache + spec decode 下，大前缀逐出到 host 池再回载后 draft KV 为垃圾：spec_accept_length 钉在 1.00（健康 2.0–3.0），target 输出仍正确，吞吐反降 25%，仅重启可愈。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/43239

### INF-0673 `sglang_gdn_cutedsl_decode_state_pool_transposed_k_equals_v_masks` SGLang GDN cutedsl decode 按转置布局读写 state

`故障` · `active` · `infra_off` · `config` · `verified` · `NEW` · 发现 2026-10-12 · 旧号 6.280

- **原因对象**：KV缓存 / sglang / CuTe DSL GDN decode 内核
- **原因行为**：#20283 起 GDN state pool 与 Triton 内核统一为 [N,HV,V,K] 布局，CuTe DSL decode 内核仍按 [N,HV,K,V] 读写；K==V 时形状校验失效，每步读到转置的 state。
- **影响对象**：KV状态；输出内容（静默错误）
- **影响详情**：GDN 模型用 --linear-attn-decode-backend cutedsl 时全量垃圾输出（"What is the capital of France?"→"The. be Ha人 II"）；默认 Triton 后端正确；双内核对照测试自 #20436 起被 skip。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/43331

### INF-0682 `nvfp4_kv_block_scale_sm_arch_swizzle_mismatch` vLLM NVFP4 KV 的 V scale 在 SM120 布局错位

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `var` · 发现 待定 · 旧号 7.9

- **原因对象**：KV缓存 / vllm / NVFP4 KV 写入 V block scale
- **原因行为**：reshape_and_cache_nvfp4_kernel 不分架构按 SM100 的 4×4 swizzle 写 V scale，而 SM120 的 XQA decode 按线性 NHD stride 读取、无 de-swizzle。
- **影响对象**：KV状态；数值结果（静默错误）
- **影响详情**：仅 SM120/SM121（如 RTX 5090）+ NVFP4 KV + XQA decode 触发：反量化 V 值错误，attention 输出偏离，表现为轻微精度退化到明显输出劣化；无 NaN、无 crash、无 shape error，SM100 上不可见。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/50084；https://github.com/vllm-project/vllm/pull/50085
- **关联**：var:INF-1325

### INF-0785 `attn_sink_eviction_collapse` 滑窗/FIFO KV 驱逐 attention sink 致长生成崩溃

`故障` · `active` · `model_on` · `yes` · `verified` · `NEW` · 发现 待定 · 旧号 8.1

- **原因对象**：KV缓存 / 通用 / 滑窗/FIFO KV 驱逐策略
- **原因行为**：朴素 sliding-window/FIFO KV cache 把最前约 4 个吸收 softmax 多余概率质量的 sink token 驱逐，或滑窗实现未保留 sink。
- **影响对象**：KV状态；输出内容（质量退化）
- **影响详情**：生成长度超过窗口后输出变得不连贯/乱码，perplexity 跳升数个数量级；驱逐边界之前完全正常。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://arxiv.org/abs/2309.17453；https://arxiv.org/abs/2504.02732

### INF-0786 `h2o_kv_eviction_misrank` H2O/SnapKV 分数式 KV 驱逐误排序丢弃关键 needle

`故障` · `active` · `model_on` · `yes` · `verified` · `NEW` · 发现 待定 · 旧号 8.2

- **原因对象**：KV缓存 / 通用 / 分数式 KV 驱逐（H2O/SnapKV）
- **原因行为**：按 attention 累计得分 top-k 加最近 token 保留 KV，预算过小或启发式排序误判时，把当前得分低但对后续关键的 needle 的 K/V 驱逐。
- **影响对象**：KV状态；输出内容（静默错误）
- **影响详情**：被驱逐区段发生静默检索失败：模型无法回答相关问题并可能编造，不崩溃、无数值离群；需与完整 cache 对照才能发现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://arxiv.org/abs/2306.14048；https://arxiv.org/abs/2404.14469；https://arxiv.org/abs/2510.13334；https://arxiv.org/abs/2510.00231

### INF-0788 `cache_position_off_by_one` cache_position 偏移致 RoPE 位置整体错位

`故障` · `active` · `infra_off` · `yes` · `verified` · `var` · 发现 待定 · 旧号 8.4

- **原因对象**：KV缓存 / transformers / cache_position 计算
- **原因行为**：带 past_key_values 时 cache_position 算错使 position_ids 整体偏移常数、RoPE 角度错；孪生形态：右填充错位、左填充配 chunked attention 致首 chunk 全为 PAD。
- **影响对象**：隐状态；输出分布（静默错误）
- **影响详情**：偏移小时 logits 静默漂移，偏移大时输出乱码；padding 类变体只在 batch>1 时出现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://arxiv.org/abs/2510.01238
- **关联**：var:9.25 / fault_rope offset

### INF-0791 `paged_attention_block_corruption` vLLM block-table 映射错误致跨请求 KV 泄漏

`故障` · `active` · `infra_off` · `partial` · `documented` · `var` · 发现 待定 · 旧号 8.7

- **原因对象**：KV缓存 / vllm / block-table 映射
- **原因行为**：block-table 映射出错，使一个序列的注意力读到属于另一个序列或上一轮请求的过期 KV block。
- **影响对象**：KV状态；跨请求/跨样本数据（静默错误）
- **影响详情**：静默发生、无报错，造成跨请求内容泄漏，隐私后果严重；整段位置的表示被替换成另一语境的表示，换入值数值上健康，逐通道离群检测容易漏。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/9231
- **关联**：var:INF-0870

### INF-0792 `sglang_eagle_spec_decode_defeats_radix_prefix_reuse` SGLang EAGLE 推测解码使 radix 前缀复用率崩溃

`故障` · `active` · `ambiguous` · `other_stack` · `verified` · `NEW` · 发现 2026-08-01 · 旧号 8.8

- **原因对象**：KV缓存 / sglang / radix 前缀复用与 EAGLE 推测路径
- **原因行为**：启用 EAGLE 推测解码后缓存的 prefix 未被复用，疑为推测路径对 radix 填充的 KV 做了防御性绕过或静默重算（原文未确诊）。
- **影响对象**：性能/资源（性能退化）
- **影响详情**：v0.5.16、GLM-DSA NVFP4、TP8 多轮 agentic 流量下，≥20K token 深提示的 cache hit 从 97% 降到 40–53%，TTFT 与吞吐等同完整重新 prefill；任何 draft 长度均触发，无 crash，输出 token 正确。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/32459

### INF-0793 `trtllm_vanilla_attention_layer_specific_kv_index_misresolution` TRT-LLM 无 layer_idx 解析 KV 块索引致层特定布局读错块

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-08-01 · 旧号 8.9

- **原因对象**：KV缓存 / tensorrt-llm / Vanilla 注意力元数据 KV 索引解析
- **原因行为**：Vanilla attention 元数据一次性准备并共享给所有注意力层，在无 layer_idx 的阶段解析 KV block 索引，该无参 lookup 对层特定布局无效。
- **影响对象**：KV状态；输出内容（静默错误）
- **影响详情**：在 VSWA、混合线性/注意力缓存、多物理池或 KVCacheManagerV2 同池多 page-index scale 布局下，普通与 MLA 生成读到错误窗口/池/scale 的 block 索引，输出静默污染，或因歧义 lookup 抛异常。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/NVIDIA/TensorRT-LLM/issues/16801

### INF-0794 `trtllm_reuse_tree_block_silent_detach_no_removed_event` TRT-LLM 复用 trie 脱离 block 时不发 removed 事件

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-08-01 · 旧号 8.10

- **原因对象**：KV缓存 / tensorrt-llm / WindowBlockManager 事件
- **原因行为**：WindowBlockManager 两条路径把 block 脱离复用 trie 却不发 removed 事件；其一 blockInRadixTree() 用 getPrevBlock()!=nullptr 判断，测错了对象。
- **影响对象**：KV状态；性能/资源（静默错误）
- **影响详情**：无 crash、无 NaN；依赖 stored/removed 事件镜像 worker 复用树的外部消费者（如做前缀重叠路由的分布式 KV router）永久泄漏已脱离的 block hash，路由误差长期累积、渐进退化。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/NVIDIA/TensorRT-LLM/pull/16335

### INF-0795 `vllm_minimax_m3_fp8e5m2_kv_python_scatter_cuda_graph_corruption` vLLM MiniMax-M3 fp8 KV 写入不可图捕获致输出垃圾

`故障` · `active` · `infra_off` · `yes` · `verified` · `NEW` · 发现 2026-08-03 · 旧号 8.11

- **原因对象**：KV缓存 / vllm / MiniMax-M3 fp8 _insert_kv
- **原因行为**：fp8 KV 路径的 _insert_kv 是纯 Python 实现，每次新分配 tensor 并做 advanced-index scatter，非 graph-safe，在 CUDA graph 捕获下写坏 index-K cache。
- **影响对象**：KV状态；输出内容（静默错误）
- **影响详情**：SM80 + MiniMax-M3 + fp8_e5m2 KV + TRITON_ATTN 时，任何 CUDA graph 模式下服务正常启动却流畅生成纯垃圾，无报错；--enforce-eager 输出连贯但仅 5.2 tok/s，bf16+graph 正常。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/50881

### INF-0797 `sglang_dsv4_flash_dsa_hisparse_progressive_corruption_under_batching` SGLang DSV4 稀疏注意力在 batching 下输出渐进损坏

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-08-03 · 旧号 8.13

- **原因对象**：KV缓存 / sglang / dsv4 DSA（HiSparse）页索引
- **原因行为**：DSA（HiSparse）稀疏注意力路径在 batching 下疑似 KV/page-index 出错（报告者怀疑，截至扫描日未确诊、未修复）。
- **影响对象**：输出内容（静默错误）
- **影响详情**：DeepSeek-V4-Flash、dsv4 backend + DP attention 下，输出损坏随 batch size 放大：从轻微语言退化到错通道 tool-call markup，再到 token soup；单请求/低并发正常，同节点同 checkpoint 的 vLLM 无此问题。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/33397

### INF-0801 `sglang_dsv4_hierarchical_cache_swa_position_corruption_nan_crash` SGLang 分层缓存 SWA KV 写位置错误致 NaN 崩溃

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-08-07 · 旧号 8.17

- **原因对象**：KV缓存 / sglang / 分层缓存 SWA 组件 KV 写位置
- **原因行为**：hierarchical cache 的 SWA 组件在 KV write_position 上确定性写错（kv-canary 捕获存储 token 与预期不符）。
- **影响对象**：KV状态；输出分布；可用性/进度（数值异常）
- **影响详情**：DeepSeek-V4-Flash + --enable-hierarchical-cache（v0.5.16）下，轻载（1 请求、约 24K 上下文）即可复现：采样概率张量含 NaN，assertion 失败杀掉所有 TP rank。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/33656

### INF-0802 `sglang_dspark_dflash_page_size1_kv_slot_leak_boundary` SGLang spec v2 在 page_size=1 时 KV 行过窄

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-08-07 · 旧号 8.18

- **原因对象**：KV缓存 / sglang / req_to_token 行宽分配
- **原因行为**：spec v2 每步按 committed+2×draft 数预留 KV slot，req_to_token 行却只在 page_size>1 时按此分配，page_size=1 时行宽低于预留水位，release 静默 clamp。
- **影响对象**：KV状态；性能/资源；跨请求/跨样本数据（静默错误）
- **影响详情**：DSPARK gamma=16 等大 draft、请求接近 context 边界时，行写入溢出到邻居请求的行，溢出 tail slot 永不释放，KV slot 渐进泄漏、资源耗尽，并有跨请求损坏风险；无 crash、无 NaN。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/33579

### INF-0803 `sglang_hicache_storage_key_omits_kv_dtype` SGLang HiCache 存储键缺 KV dtype 致跨运行缓存碰撞

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-08-07 · 旧号 8.19

- **原因对象**：KV缓存 / sglang / HiCache 存储键 config_suffix
- **原因行为**：HiCache 存储键只由 model_name 与 TP/PP/CP geometry 构建，HiCacheStorageConfig 没有 KV cache dtype 字段，不同 dtype 的运行产生相同 key。
- **影响对象**：KV状态；输出内容（静默错误）
- **影响详情**：同一模型先后以 bf16 与 fp8_e4m3 运行时，第二次静默复用 dtype 不符的缓存字节，输出错误；hf3fs/flexkv/lmcache/aibrix 所有后端均受影响，无 crash、无 NaN。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/33268

### INF-0805 `vllm_hybrid_attn_eagle_drop_full_cache_miss` vLLM 混合模型 + Eagle 非块对齐 prompt 前缀缓存全 miss

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-08-08 · 旧号 8.21

- **原因对象**：KV缓存 / vllm / 混合模型前缀缓存 checkpoint 位置
- **原因行为**：Eagle 下 prefix cache 要求匹配点之后再多一块（128）才 drop；非块对齐 prompt 时 Mamba 两个边界 checkpoint 都坍缩到同一位置，无 checkpoint 能满足全注意力 +128 条件。
- **影响对象**：性能/资源（性能退化）
- **影响详情**：Kimi K3 + Eagle（block_size=1536、PMU=128）下，prompt_length=24577 等非块对齐 prompt 的 prefix cache 全 miss（0 hit），每请求全量重算，吞吐严重损失；无 crash，输出正确。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/51295

### INF-0806 `sglang_unified_memory_dspark_page_zero_int32_wrap_nan` SGLang 统一内存池+DSPARK：页未清零与 int32 回绕致 NaN

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-08-08 · 旧号 8.22

- **原因对象**：KV缓存 / sglang / 统一内存池页分配与 CuTe slot 索引
- **原因行为**：统一内存池分配页时未正确清零 recurrent state slot；CuTe kernel 的 slot-stride 索引用 int32，大 batch 下回绕导致读取错误 state。
- **影响对象**：KV状态；数值结果（数值异常）
- **影响详情**：开启 --enable-unified-memory 并使用 DSPARK chain 推测解码时产出 NaN，最终 crash 或静默 NaN 传播；Kimi-K3（TP8）与 Kimi-Linear-48B 端到端验证。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/pull/33974

### INF-0807 `sglang_deterministic_triton_extend_unified_pool_nan` SGLang 确定性 Triton extend 读未转译槽位致 NaN

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-08-08 · 旧号 8.23

- **原因对象**：KV缓存 / sglang / out_cache_loc 虚拟→物理转译
- **原因行为**：确定性推理下 Triton extend 走 _forward_extend_unified，该 kernel 被传入未转译的 out_cache_loc；统一内存池上它存的是虚拟 id，kernel 按其读错位置。
- **影响对象**：输出分布（静默错误）
- **影响详情**：统一内存 + triton backend + 确定性推理三标志同启时 logits 出现 NaN；仅当 SGLANG_ENABLE_ASYNC_ASSERT 开启时 abort，否则静默返回有限的 top-k logprobs；任二标志组合无害。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/pull/33517

### INF-0808 `sglang_gdn_chunked_extend_neg1_sentinel_arithmetic` SGLang GDN chunked extend 未守卫 -1 索引

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-08-08 · 旧号 8.24

- **原因对象**：KV缓存 / sglang / GDN chunked extend kernel
- **原因行为**：padding 行状态索引被置为 -1；decode kernel 有 idx>=0 守卫，chunked GDN extend kernel 却把 -1 直接用于初始状态读取与最终状态写回的指针算术。
- **影响对象**：KV状态；可用性/进度（静默错误）
- **影响详情**：GDN 模型（如 Qwen3.5-397B-A17B-FP8）在 breakable prefill CUDA graph + DP attention 下，padded 行携带 sentinel，在状态池之前寻址，可能 crash 或静默读写池前内存；boundary_check 不捕获。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/pull/33810

### INF-0809 `sglang_dsv4_spec_draft_gt4_compress_ring_underwrite` SGLang DSV4 draft>4 时压缩状态 ring 欠写

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-08-08 · 旧号 8.25

- **原因对象**：KV缓存 / sglang / c_plan.cuh 压缩写入计划
- **原因行为**：compress write-plan 按乐观 seq_len 规划，pad 却硬编码为 4（host 路径完全缺 pad），num_draft_tokens>4 时欠写 ring，部分 committed token 未持久化。
- **影响对象**：KV状态；输出内容（静默错误）
- **影响详情**：DSV4 + 推测解码且 speculative_num_draft_tokens>4 时，后续压缩窗口从 stale ring slot 计算，输出语义静默偏差、表面正常；无越界、NaN 或 assert。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/pull/33872；https://github.com/sgl-project/sglang/pull/34189

### INF-0812 `trtllm_window_vector_layer_indexing_model_global_vs_stage_local` TRT-LLM PP 下窗口向量全局/本地层索引混淆

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-08-10 · 旧号 8.28

- **原因对象**：KV缓存 / tensorrt-llm / V1/V2 管理器窗口向量层索引
- **原因行为**：max_attention_window 是模型全局的重复模式，而 KV pool 与 attention 元数据按流水线 stage 本地组织；V1 管理器部分分片、V2 保持全局，消费者需猜用本地还是全局层索引。
- **影响对象**：隐状态；输出内容（静默错误）
- **影响详情**：流水线并行（含非零或不均匀 stage）+ VSWA/滑窗注意力下，层索引不一致可使 attention 采用错误的窗口值，输出静默退化、无 crash；masked hybrid cache 的启用层约定加剧此问题。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/NVIDIA/TensorRT-LLM/pull/17466

### INF-0814 `vllm_rocm_dspark_kv_cache_layout_overlap_silent_nan` vLLM ROCm KV 布局与混合模型 page pool 重叠致 NaN

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-11 · 旧号 8.30

- **原因对象**：KV缓存 / vllm / ROCm KV 布局、混合模型 page pool
- **原因行为**：ROCm KV 形状 (2, num_blocks, …) 让 K/V 分居两端，block b 跨两个不相邻半页，违反 page pool“block b 占 page b”协议，attention 与 Mamba state 页重叠。
- **影响对象**：KV状态；输出内容（数值异常）
- **影响详情**：MI300 上混合模型 + DSpark 推测解码输出垃圾（GSM8K 0.20 vs 0.95）；持偶数 block id 的序列在首个 full-attention 层输入干净而输出 NaN，单个 NaN key 毒化整行 softmax 致文本塌缩；无 crash。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/51837

### INF-0815 `vllm_kv_block_zeroing_only_full_attention_fp8_swa_stale_data` vLLM KV 块清零漏掉 FP8 滑窗页致 acceptance 塌缩

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-08-11 · 旧号 8.31

- **原因对象**：KV缓存 / vllm / KVBlockZeroer 清零范围
- **原因行为**：KV 清零对混合 Mamba/混合精度 cache 启用，但 scheduler 只上报 full/MLA block 类型、worker zeroer 只处理 full-attention 组，回收的 FP8 滑窗页残留旧数据。
- **影响对象**：KV状态；输出内容；性能/资源（质量退化）
- **影响详情**：Kimi-K3 DSpark drafter SWA GQA（FlashInfer/TRTLLM-gen）上，后续 GSM8K batch 的 acceptance length 从约 4.8 渐进塌缩、准确率退化；强制 FlashAttention 或 BF16 KV 则稳定，无 crash。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/51749

### INF-0816 `vllm_sliding_window_from_kv_group_spec_global_layers_lose_tokens` vLLM 组级 window 套用到全局层致旧 token 丢失

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-08-11 · 旧号 8.32

- **原因对象**：KV缓存 / vllm / FA/CPU attention 窗口来源
- **原因行为**：FlashAttention 从 KV cache group spec 读取 window 并应用到组内每一层，组内同时含滑窗层与全局层时全局层也被套用窗口；CPU attention 的组级元数据同样如此。
- **影响对象**：KV状态；隐状态；输出内容（静默错误）
- **影响详情**：Gemma-3 + --disable-hybrid-kv-cache-manager（滑窗层提升为 full-attention 存储）时，全局层静默丢失窗口以外的旧 token，上下文在窗口边界被截断，输出退化但不报错、无 NaN。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/51756

### INF-0817 `vllm_eagle_volatile_tail_boundary_collapse_hybrid_swa_zero_recovery` vLLM EAGLE 尾部回退致 CPU KV offload 零命中

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-08-11 · 旧号 8.33

- **原因对象**：KV缓存 / vllm / CPU KV offload 存储/查找边界
- **原因行为**：EAGLE volatile-tail pop 把共享 replay 边界收紧到 full-attention 对齐以下，而 store 只保留 prompt 末尾 tail、未存小滑窗组验证窗口所需 chunk，lookup 级联下探。
- **影响对象**：性能/资源（性能退化）
- **影响详情**：混合 EAGLE/MTP + 多个小滑窗组 + CPU KV offload 下，非对齐 prompt（如 30,208 token）存储成功但 replay 时恢复 0 个外部 token，等同无缓存；对齐的 30,272-token prompt 正常恢复；无 crash。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/51761

### INF-0820 `sglang_mamba_radix_cache_extra_buffer_wrong_ssm_state` SGLang mamba radix cache 缓存错误位置的 SSM 状态

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-08-12 · 旧号 8.36

- **原因对象**：KV缓存 / sglang / _init_track_ssm_indices
- **原因行为**：_init_track_ssm_indices 以 cumsum 计算 track_ssm_src_offset，对未对齐 chunk 边界的请求指向错误 packed state，缓存了错误 token 位置的 SSM state。
- **影响对象**：KV状态；输出内容（质量退化）
- **影响详情**：NemotronH 等 Mamba 混合模型开启 extra_buffer 策略且多请求 prefill 时，batch 中多数请求缓存的 SSM state 位置错误；prefix 命中后恢复错误状态，输出静默退化，无 crash、无 assert。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/pull/33450

### INF-0821 `heterogeneous_attention_memory_runtime_observability` 异构 attention memory 运行时观测契约与损坏定位

`方法` · `active` · `infra_off` · `not_target` · `documented` · `NEW` · 发现 2026-08-13 · 旧号 8.37

- **原因对象**：KV缓存 / 通用 / attention memory 压缩与驱逐
- **原因行为**：所针对的故障：plain KV、latent、sparse selector、recurrent state 四类 attention memory 在压缩、驱逐或 slot 复用下各以不同方式失效，造成静默损坏。
- **影响对象**：KV状态（静默错误）
- **影响详情**：论文在带 packed compressed-KV 原型的 DeepSeek-V4 服务栈上把一次 silent corruption 定位到精确结构边界；所有观察到的故障都出现在 eviction 或 slot-reuse regime。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/metask-ai/witprobe-attention-memory；https://arxiv.org/abs/2608.05863

### INF-0825 `dlm_bidirectional_attention_breaks_kv_immutability_prefix_cache` 扩散语言模型双向注意力使共享前缀 KV 缓存损坏

`故障` · `active` · `model_on` · `other_stack` · `documented` · `NEW` · 发现 2026-08-15 · 旧号 8.41

- **原因对象**：KV缓存 / 通用 / 共享前缀 KV 缓存复用
- **原因行为**：把 LLM 的共享前缀 KV 缓存直接用于扩散语言模型，沿用“KV 一旦计算即不变”的假设复用前缀 KV，而 DLM 双向注意力下更新任一 token 都会改变整个上下文的 KV。
- **影响对象**：KV状态；输出内容（静默错误）
- **影响详情**：DLM serving 套用标准 prefix caching/paged KV 时，复用的前缀 KV 对应旧去噪步的上下文，共享前缀缓存被 corrupt；无 crash，输出静默语义漂移。
- **改进措施**：检测 待定；修复 待定
- **来源**：http://arxiv.org/abs/2606.07571

### INF-0830 `vllm_indexer_kv_dtype_dual_knob_silent_bool_resolution` vLLM indexer KV dtype 双旋钮同传时静默按 bool 解析

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-08-16 · 旧号 8.44

- **原因对象**：KV缓存 / vllm / indexer KV dtype 配置旋钮
- **原因行为**：indexer K-cache dtype 由 bool use_fp4_indexer_cache 与 enum indexer_kv_dtype 两个旋钮分别控制，两者同传时被静默接受并按 bool 解析，enum 被忽略。
- **影响对象**：KV状态（静默错误）
- **影响详情**：DeepSeek V3.2/V4、MiniMax M3 等 sparse indexer 模型同时设置两个 dtype 旋钮时无报错，indexer KV cache 拿到非预期 dtype（FP4 而非 enum 指定值），精度/正确性静默偏离。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/52550

### INF-0837 `vllm_nvfp4_kv_cache_hnd_layout_probe_inversion` vLLM NVFP4 KV 写内核 HND 布局误判致 cache 静默损坏

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-08-25 · 旧号 8.53

- **原因对象**：KV缓存 / vllm / NVFP4 KV-cache 写内核
- **原因行为**：写内核假设逻辑视图，从 key_cache.size(1) 取 block_size、以硬编码 stride(2)>stride(1) 探测布局；收到 HND 物理形状时把 num_heads 误读为 block_size 并反转布局判定。
- **影响对象**：KV状态；输出内容（静默错误）
- **影响详情**：FlashInfer 等 HND 物理形状调用方使用 NVFP4 KV cache 且 num_heads%4==0 时，唯一下游整除检查碰巧通过，cache 静默损坏、decode 输出垃圾文本，无 crash 无告警；内核单测仍全绿。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/53681

### INF-0840 `vllm_rocm_glm53flash_kpool_block_size_advertising_mismatch_self_overwrite` vLLM ROCm kpool 索引器块大小声明错配致 KV 自覆写

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-08-29 · 旧号 8.56

- **原因对象**：KV缓存 / vllm / kpool 索引器 block size 声明
- **原因行为**：ROCm 上 indexer 声明支持任意 16 倍数，640 被当作原生块而不建 kernel 粒度 block table；slot_mapping 与 gather 按页寻址却查 spec-block id，5× 过订阅。
- **影响对象**：KV状态；输出内容（静默错误）
- **影响详情**：8×MI350X 上 GLM-5.3-Flash（index_kpool=4、block_size=640）4096 个 pool 条目写进 864 个 slot、79% 索引键错误；文本仍流畅但长上下文 recall 退化，temp=0 不可复现，无 crash。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/54359

### INF-0847 `vllm_encoder_attention_quantized_kv_request_time_reject` vLLM encoder 注意力配量化 KV 启动不校验、首个请求才报错

`故障` · `active` · `infra_off` · `config` · `verified` · `NEW` · 发现 2026-09-03 · 旧号 8.63

- **原因对象**：KV缓存 / vllm / encoder 注意力量化 KV 校验
- **原因行为**：encoder-only attention 对量化 KV 的能力检查只存在于请求处理路径，preflight 与引擎构建期没有该组合校验
- **影响对象**：可用性/进度；指标/日志（报错）
- **影响详情**：embedding 模型（如 bge-large-en）配 --kv-cache-dtype fp8_e4m3 时服务正常启动、健康检查通过，首个 /v1/embeddings 请求才抛 NotImplementedError 返回 500；同类组合的错误在构建期、选择期或请求期随机显形
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/55048

### INF-0848 `vllm_rocm_glm53flash_kpool_mtp_decode_memory_access_fault` ROCm kpool 页粒度错配致 MTP decode 显存 fault

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `var` · 发现 2026-09-04 · 旧号 8.64

- **原因对象**：KV缓存 / vllm / kpool indexer 页/块粒度
- **原因行为**：hybrid 对齐器落定 block_size=1152 后，kpool indexer 把 288-entry 存储虚拆为 32-entry paged-MQA 页，存储页粒度与块粒度错配，MTP decode 路径越界访问
- **影响对象**：可用性/进度（报错）
- **影响详情**：ROCm gfx942（MI325X、TP4）上 GLM-5.3-Flash 开 MTP（k=2）decode 时报 GPU Memory access fault，单 worker abort 致引擎下线；非 OOM，加 slot-mapping 守卫后仍复现
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/55280
- **关联**：var:INF-0840

### INF-0850 `vllm_hybrid_block_alignment_semantic_vs_packed_mla_width` hybrid 对齐按语义 head size 定 MLA 页大小漏算打包位宽

`故障` · `active` · `infra_off` · `config` · `documented` · `var` · 发现 2026-09-05 · 旧号 8.66

- **原因对象**：KV缓存 / vllm / hybrid 块对齐 MLA 页大小计算
- **原因行为**：hybrid block 对齐器按语义 head size 计算 MLA 物理页大小，未计入 FP8/NVFP4 下的 packed 存储位宽，与注意力侧对物理页大小的推导不一致
- **影响对象**：KV状态（静默错误）
- **影响详情**：vLLM 运行 hybrid MLA+Mamba 模型且 KV 为 FP8/NVFP4 时，注意力块与 Mamba padding 的物理页尺寸错配；原文推断显形为静默错位读写或硬件 fault（取决于越界落点），未给出具体观测
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/55449
- **关联**：var:INF-0840

### INF-0855 `vllm_rocm_aiter_mla_kv_dtype_global_cache_config_vs_group_spec` AITER MLA 误用全局 KV dtype 致 graph 捕获崩溃

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `var` · 发现 2026-09-06 · 旧号 8.71

- **原因对象**：KV缓存 / vllm / AITER MLA 元数据 KV dtype
- **原因行为**：AiterMLAMetadataBuilder 从全局 cache_dtype 推导 KV dtype 而非组级 spec，bf16 draft 组被当作 fp8，据此定元数据尺寸、选 fp8 kernel 读 bf16 缓冲
- **影响对象**：可用性/进度（报错）
- **影响详情**：MI355X 上 Kimi-K3 + MTP draft（auto dtype）+ --kv-cache-dtype fp8 + TP8 时，8 个 rank 在 cudagraph capture 期同步 memory access fault 崩溃；eager 走 Gluon 路径不显形
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/55591
- **关联**：var:INF-0854

### INF-0856 `vllm_kv_sharing_layers_default_scale_dequant_mismatch` KV-sharing 层未继承 target scale，按 1.0 反量化

`故障` · `active` · `infra_off` · `config` · `verified` · `var` · 发现 2026-09-06 · 旧号 8.72

- **原因对象**：KV缓存 / vllm / KV cache 初始化 scale 传播
- **原因行为**：KV-sharing 层读 target 层以其 k/v_scale 写入的量化 KV，checkpoint 无 sharing 层 scale，绑定共享缓存时未传播 target scale，sharing 层按默认 1.0 反量化
- **影响对象**：隐状态；输出内容（质量退化）
- **影响详情**：Gemma-3n/Gemma-4-E 系列配量化 KV（NVFP4/fp8）时无报错，20 个 sharing 层注意力输出相对误差 0.40–0.91（正常层 0.10–0.26）；长文 needle 检索 2/9（修复后 7/9），chat 格式不受影响
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/55559
- **关联**：var:INF-0843

### INF-0863 `flashinfer_recurrent_kda_cute_dsl_negative_ssm_index_wraparound_overwrites_pool_tail` FlashInfer recurrent_kda 负索引回绕覆写状态池尾槽

`故障` · `active` · `infra_off` · `partial` · `verified` · `NEW` · 发现 2026-09-09 · 旧号 8.79

- **原因对象**：KV缓存 / flashinfer / recurrent_kda dense 路径索引
- **原因行为**：cute-dsl 后端 dense T=1 路径在 Python 层用普通索引 gather/scatter 循环状态池，未跳过 ssm_state_indices 中表示 padding 的负条目，-1 回绕到池末槽并被写回
- **影响对象**：KV状态；跨请求/跨样本数据（静默错误）
- **影响详情**：B200 上 ssm_state_indices=[-1,3]、5-slot 池时池末 slot 4 出现 delta 77.0/61.2 的意外写入；占用池尾槽的活请求后续 decode 状态被静默覆写，无报错；cake 后端与 cu_seqlens 路径均正常
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/flashinfer-ai/flashinfer/issues/5042

### INF-0867 `vllm_glm53flash_rocm_indexer_block_table_granularity_mismatch_kpool_generic_slotmap` GLM-5.3-Flash 块表粒度错配与 slot 映射越组

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `var` · 发现 2026-09-11 · 旧号 8.83

- **原因对象**：KV缓存 / vllm / 块大小推导、KpoolTail 映射
- **原因行为**：prepare_kernel_block_sizes 不遵守 storage_block_size，块表存 manager 块 ID 而非物理页 ID；KpoolTailSpec 未排除于通用 slot mapping，循环块被按位置映射
- **影响对象**：KV状态；输出内容；可用性/进度（静默错误）
- **影响详情**：MI308X TP8 上 160K 长 prompt 请求正常完成但末段检索静默失效（表宽外写零 scale、表宽内错页读）；chunked prefill 下越行读出野物理 ID，触发 GPU memory access fault / EngineDeadError
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/56380
- **关联**：var:INF-0857

### INF-0870 `kv_corrupt` KV cache 单位置 K/V 行损坏（zero/noise/swap）

`故障` · `active` · `infra_off` · `yes` · `verified` · `NEW` · 发现 待定 · 旧号 8.85

- **原因对象**：KV缓存 / 通用 / KV cache 第L层位置j的K/V行
- **原因行为**：第 L 层 KV cache 中位置 j 的 K、V 行被改写：zero 整行清零、noise 加按该位置 K/V 范数定标的高斯噪声、swap 换成同序列另一位置的 K/V；破坏的是该 token 对其他 token 的可见内容
- **影响对象**：KV状态；输出内容（静默错误）
- **影响详情**：后续解码 token 基于被污染的上下文，表现为忘记或记错前文某个词；zero 模式下该位置对后续 query 近乎无信息，swap 模式像把相邻位置内容重复读一遍；打到 attention-sink 位置时破坏最大
- **改进措施**：检测 待定；修复 待定
- **来源**：引用：本项目早期实测条目（原型阶段实测，量值不跨栈继承；

### INF-0871 `ttmetal_gemma4_12b_tp8_warmup_trace_nkv1_doubled_sliding_block_step1_divergence` Gemma4 TP8 warmup trace 固化错误滑窗几何致发散

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `var` · 发现 2026-09-15 · 旧号 8.87

- **原因对象**：KV缓存 / tt-metal / sliding 块计数、warmup 页表
- **原因行为**：12B 在 TP=8 下 nkv_local=1，sliding block 块数被按 2 倍计入，warmup 捕获的 trace 固化该几何；warmup 还用全零 page table，reset_batch 仅 step 0 传入
- **影响对象**：KV状态；输出内容（质量退化）
- **影响详情**：vLLM warmup + hybrid-KV 服务路径下 decode 从第 2 步起与参考完全发散（step0 PCC 0.9995，之后约 0.06），输出单字或空响应、gpqa 0 分，无 crash；非 warmup 的 text_demo_v2 正常，31B 不触发
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/56577
- **关联**：var:INF-0844

### INF-0877 `ttmetal_gemma4_exact_window_sliding_ring_spec_verify_loses_in_window_kv_warning_only` Gemma4 spec verify 覆写滑窗 ring 中仍在窗内的 KV

`故障` · `active` · `infra_off` · `config` · `verified` · `NEW` · 发现 2026-09-24 · 旧号 8.93

- **原因对象**：KV缓存 / tt-metal / spec verify 写环与槽 mask
- **原因行为**：sliding KV ring 默认等于窗口（31B 为 1024），spec verify 一次写 K+1 行，写 p+j 即覆写仍在前序 query 窗内的 p+j−1024；服务类只打 warning 仍照常 speculate
- **影响对象**：KV状态；输出内容（静默错误）
- **影响详情**：上下文超过约 1019 token 且 K>1 时 speculative 输出自首 token 起静默损坏：sliding 层 committed token 来自缩短窗口，被拒 draft 行被后续 decode 当历史读；唯一痕迹是启动 warning，262144 上下文的生产配置均自动开启
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/57701

### INF-0878 `vllm_rocm_kpool_indexer_block_table_unconverted_640_token_table_as_32pool_pages` ROCm kpool indexer 非整除时跳过块表转换致检索失效

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `var` · 发现 2026-09-26 · 旧号 8.94

- **原因对象**：KV缓存 / vllm / MLA indexer 块表粒度转换
- **原因行为**：indexer metadata builder 仅在存储块整除 kernel 块时转换块表；ROCm 上 kernel 块 640、存储块 128 不整除，块表被原样当作 32-pool 页号交给 writer 与 gather
- **影响对象**：KV状态；数值结果（静默错误）
- **影响详情**：GLM-5.3-Flash 在 ROCm（gfx1030 实测）上 74 页中仅 3 页正确：写入落 padding、越块位置别名 block 0、超列宽被 mask，sparse attention 的 top-k 检索基于错或零 keys，输出退化为无索引注意力；NVIDIA 走整除分支不受影响
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/58858
- **关联**：var:INF-0867

### INF-0880 `ttmetal_update_cache_fp32_corrupts_31_neighbouring_rows_of_updated_position` update_cache 写 FP32 cache 截断同 tile 邻行

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-10-09 · 旧号 8.96

- **原因对象**：KV缓存 / tt-metal / (paged_)update_cache 写回
- **原因行为**：update 路径把中间结果按输出 tile 格式物化后整块写回，同一 32 行 tile block 中未被更新的行也被按该格式截断
- **影响对象**：KV状态（静默错误）
- **影响详情**：FLOAT32 cache 写入 1 个 token 时，同 tile block 其余 31 行静默变为 bf16 截断（8192 元素中 2762 个越更新行变化），fp32_dest_acc_en=True 时为 TF32 截断；块外行保持精确，同张量 fill_cache 精确
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/59561

### INF-0881 `ttmetal_fill_cache_multihead_work_straddles_head_boundary_wrong_head_pages` ttnn.fill_cache 多核切分跨 head 边界写错 head 页

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-10-09 · 旧号 8.97

- **原因对象**：KV缓存 / tt-metal / fill_cache 多核工作切分与 writer
- **原因行为**：interleaved 且 kv_heads>1 时 work 按 head-major 枚举，split_work_to_cores 可跨头切分，writer 只按 Wt 步进、不知道头边界；TODO 承诺的 assert 不存在
- **影响对象**：KV状态（静默错误）
- **影响详情**：core 分块跨越 head 边界处，head h+1 的 token 被写进 head h 的页区间，head h+1 的真实页从未写入；多 head prefill 填充后注意力基于错位 KV，无报错
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/58683

### INF-0882 `vllm_glm53flash_rocm_kpool_always_select_tail_covers_only_incomplete_pool` GLM-5.3-Flash kpool 尾部保证仅覆盖不完整尾池

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `var` · 发现 2026-10-09 · 旧号 8.98

- **原因对象**：KV缓存 / vllm / kpool indexer 尾池保证
- **原因行为**：always_select_tail=True 只保证 ≤3 token 的不完整尾池入选，完整的 recent 池与其他候选一同竞争 top-k，可能被排在 index_topk=2048 截断之外
- **影响对象**：数值结果；输出内容（静默错误）
- **影响详情**：GLM-5.3-Flash 在 ROCm aiter kpool 路径（gfx1151 TP2 AWQ-W4A16 实测）上，长 prompt 的尾部上下文检索非确定失败，输出流畅但答错；block table 正确、无 crash
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/59741
- **关联**：var:INF-0867

### INF-0906 `vllm_kv_offload_reuses_blocks_across_rope_scaling_change` vLLM KV offload 缓存键缺 RoPE 配置致跨配置复用

`故障` · `active` · `infra_off` · `config` · `verified` · `NEW` · 发现 2026-09-10 · 旧号 9.23

- **原因对象**：KV缓存 / vllm / OffloadingConnector 外部缓存键
- **原因行为**：OffloadingConnector + TieringOffloadingSpec 的外部缓存键不含 RoPE scaling/theta 等位置配置；RoPE 参数改变后同一 token 的 KV 语义已变，但键不变仍命中旧块。
- **影响对象**：KV状态；输出内容（静默错误）
- **影响详情**：同模型同 offload 目录下把 rope_parameters.factor 由 1 改为 2 重启，新进程复用旧配置写入的 73 个 KV 文件，首个生成 token 即与空目录对照分歧（top-20 仅 3 个共同候选），且确定性复现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/56311

### INF-1067 `sglang_teacache_cfg_lifecycle_double_reset_state_leak` SGLang TeaCache CFG 状态重复 reset 致跨请求泄漏

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-08-17 · 旧号 11.44

- **原因对象**：KV缓存 / sglang / TeaCache CFG 状态生命周期
- **原因行为**：TeaCache 的 CFG 生命周期管理有两处缺陷：串行 CFG 下重复 reset 导致跨步状态泄漏；CFG parallel 下负分支状态未隔离、forecaster 状态跨请求复用。
- **影响对象**：输出内容；跨请求/跨样本数据（质量退化）
- **影响详情**：SGLang diffusion 启用 TeaCache 与 CFG（串行或 parallel）时，时间步缓存命中/跳过判断基于受污染状态，生成质量静默退化；无 crash。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/35054

### INF-1068 `sglang_spectrum_cfg_parallel_negative_branch_init_skip` SGLang Spectrum CFG parallel 跳过负分支初始化

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `var` · 发现 2026-08-17 · 旧号 11.45

- **原因对象**：KV缓存 / sglang / Spectrum CFG 负分支初始化
- **原因行为**：Spectrum 的 CFG parallel 路径不初始化负分支，forecaster 状态可跨请求复用。
- **影响对象**：跨请求/跨样本数据；输出内容（质量退化）
- **影响详情**：SGLang diffusion 启用 Spectrum + CFG parallel 时，跨请求状态污染使生成质量静默退化；无 crash，与 TeaCache 属同一 CFG 分支状态隔离缺口。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/35053
- **关联**：var:INF-1067

### INF-1087 `vllm_mamba_spec_context_stale_batch_mapping_poisons_state` vLLM Mamba spec context 块表映射陈旧致状态毒化

`故障` · `active` · `infra_off` · `partial` · `verified` · `NEW` · 发现 2026-09-05 · 旧号 11.64

- **原因对象**：KV缓存 / vllm / Mamba spec context 块表映射
- **原因行为**：MambaSpecDecodeGPUContext 只捕获一次 block table，align kernel 却按每步重收集的 batch 序取行；PP 下非末 rank 晚 pp_size 步后处理，读写别的请求的块
- **影响对象**：KV状态；输出内容（静默错误）
- **影响详情**：hybrid + PP≥2 + MTP + prefix caching + ≥3 并发：14–33% 请求退化为 prompt 无关的常量 token 1023 循环，acceptance 50%→26%；sm_80 静默，sm_121 表现为 illegal access 崩溃
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/55506

### INF-1088 `vllm_gdn_mixed_batch_state_recovery_partition_discards_nonspec_counts` vLLM GDN 混合 batch 接受计数分区恢复丢弃非 spec 计数

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-09-05 · 旧号 11.65

- **原因对象**：KV缓存 / vllm / V1 GDN 混合 batch 接受计数分区
- **原因行为**：spec 与非 spec 请求的接受计数需独立分区恢复，合并实现在全部非 spec 计数均为 1 时走丢弃分支，conv caller 回退到 spec 分区的计数，offset 与条目数双错
- **影响对象**：KV状态；输出内容（静默错误）
- **影响详情**：GDN hybrid 模型在 spec 解码关闭后的转换期或 spec/非 spec 混合 batch 中，非 spec 接受计数全为 1（常规 decode 的常见形态）即触发；conv state 错位读写，decode 输出错值或后续崩溃，取决于落点
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/55504

### INF-1091 `vllm_hybrid_gdn_mtp_scheduler_three_wide_concurrency_window` vLLM GDN hybrid+MTP 并发≥4 时只调度 3 条序列

`故障` · `active` · `infra_off` · `partial` · `documented` · `var` · 发现 2026-09-06 · 旧号 11.68

- **原因对象**：KV缓存 / vllm / mamba 状态池 MTP slot 预算
- **原因行为**：原文推断：mamba/GDN 状态池在 target 递归状态与 MTP draft slot 间的预算划分，把有效 spec 并发压到与状态容量相关的窄窗，max_num_seqs/batched_tokens 均未触及
- **影响对象**：性能/资源（性能退化）
- **影响详情**：27B Qwen3.5 族 hybrid GDN（48 GDN+16 full attn）+ MTP 在 batch≥4 时每次 decode 迭代只调度 3/8 序列，其余等待轮换，MTP 比无 spec 慢 2–3×，输出正确；bs≤3 健康，24 层 GDN 的 9B 模型 bs=20 仍健康
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/55533
- **关联**：var:INF-0529

### INF-1092 `sglang_chunked_prefill_radix_insert_race_dangling_kv_pages_qsa` SGLang 分块 prefill 与 retract 竞态致悬垂 KV 页

`故障` · `active` · `infra_off` · `partial` · `verified` · `var` · 发现 2026-09-07 · 旧号 11.69

- **原因对象**：KV缓存 / sglang / MambaRadixCache 分块插入与释放
- **原因行为**：cache_unfinished_req 在 chunked prefill 中途把各 chunk 的 KV 页插入 radix 树；retract/abort 时这些页被释放回分配器，radix 节点留下悬垂引用
- **影响对象**：KV状态；跨请求/跨样本数据；输出内容（静默错误）
- **影响详情**：QSA 压缩注意力模型 + prefix caching 在 retract/abort 压力下，页被重分配后命中该前缀的请求首个 decode 步即读垃圾，确定性输出 token 248319 循环、accept_len 恒 1；压缩寻址放大为跨请求污染；flush_cache 可暂愈
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/38319
- **关联**：var:INF-0556

### INF-1095 `vllm_pp_spec_mamba_align_stale_idx_mapping_wrong_block_state_copy` vLLM PP+spec 下 mamba align 用陈旧映射拷错状态块

`故障` · `active` · `infra_off` · `partial` · `verified` · `var` · 发现 2026-09-08 · 旧号 11.72

- **原因对象**：KV缓存 / vllm / mamba align 块表索引
- **原因行为**：PP>1 时 align 内核拿到 pp_size 步前的 idx_mapping，却用来索引 T−1 batch 的 block table，recurrent state 被拷到错误的物理 mamba 块
- **影响对象**：KV状态；输出分布（数值异常）
- **影响详情**：PP>1 + spec + hybrid Mamba/GDN 且批组成变化时，仅 rank 0..pp−2 受影响：8×3090 TP2 PP4 k=3 混合 4 路并发出现 NaN logits，945K RULER 召回 12/20、14/20（spec off 16/20）；单请求不触发
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/55951
- **关联**：var:INF-1087

### INF-1098 `vllm_ngram_spec_hybrid_recurrent_state_layout_convention_switch_stale_slot` vLLM ngram 推测下 hybrid 循环状态槽约定分歧致读陈旧状态

`故障` · `active` · `infra_off` · `partial` · `verified` · `NEW` · 发现 2026-09-11 · 旧号 11.75

- **原因对象**：KV缓存 / vllm / GDN/Mamba 循环状态槽布局约定
- **原因行为**：推测与非推测路径对循环状态的存放约定不同（固定槽 0 vs 按接受数选槽），runner 与 GDN builder 可在相邻步间切换约定，读槽 0 得到陈旧状态；num_accepted_tokens 初始为 0 会成索引 −1。
- **影响对象**：KV状态；输出内容（静默错误）
- **影响详情**：接受 draft 的步紧接 ngram 无匹配的零 draft 步时触发（EAGLE/MTP 不触发），Qwen3-Next、Nemotron Mamba2、Kimi-Linear 等 hybrid 模型间歇性掉字/重复 token，负载依赖、无报错。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/56531

### INF-1102 `vllm_dflash_drafter_kv_group_unidentified_mamba_prefix_reuse_silently_disabled` vLLM dflash KV 组未识别致 hybrid 前缀缓存失效

`故障` · `active` · `ambiguous` · `config` · `verified` · `var` · 发现 2026-09-15 · 旧号 11.79

- **原因对象**：KV缓存 / vllm / drafter KV cache 组识别规则
- **原因行为**：KV 组识别规则只对 method=mtp 生效，dflash drafter 复用 target 层架构、spec 规则也认不出；回退把所有组（含 Mamba/GDN）当 draft 组，Mamba 组满足不了放宽的 lookup 窗口。
- **影响对象**：性能/资源（性能退化）
- **影响详情**：dflash 推测 + hybrid 模型 + --enable-prefix-caching 时，字面前缀子请求不复用（36.4s≈全冷 prefill），只有完全相同文本重发才复用；输出正确，启动警告每次打印但易被当作正常日志。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/57032
- **关联**：var:vllm_spec_decode_prefix_cache_hybrid_silent_disable

### INF-1227 `kv_compression_evidence_eviction_hallucination_abstention` KV 压缩逐出答案证据致模型幻觉作答而非弃权

`故障` · `active` · `model_on` · `not_target` · `documented` · `NEW` · 发现 2026-09-01 · 旧号 15.9

- **原因对象**：KV缓存 / 通用 / KV 压缩/驱逐（证据 token 逐出）
- **原因行为**：驱逐/掩码类 KV 压缩逐出了承载答案证据的 token，模型失去判断上下文不足、选择弃权的依据。
- **影响对象**：输出内容（静默错误）
- **影响详情**：证据被逐出后模型不自知地 on-manifold 编造答案而不是弃权；仅看输出无法区分“没证据也答”与“有证据答错”，幻觉率随压缩策略与证据分布联合变化。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://arxiv.org/abs/2608.29934

### INF-1231 `llamacpp_context_shift_rope_kshift_corruption` llama.cpp context shift 致 RoPE 相位非均匀漂移

`故障` · `active` · `infra_off` · `other_stack` · `verified` · `NEW` · 发现 待定 · 旧号 K2.1

- **原因对象**：KV缓存 / llama.cpp / context shift cell delta
- **原因行为**：context shift 时对存活缓存键施加 RoPE K-shift 重新编号，但每个 cell 的 delta 只反映最后一次 shift，pos 与 delta 失同步，多次 shift 后各位置残留未修正的相位
- **影响对象**：KV状态；输出内容（质量退化）
- **影响详情**：长流式上下文中渐进丧失连贯、出现逐字重复式乱码；单次 shift 无碍，损坏随累积 shift 距离扩大，加 --no-context-shift 可消除
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/ggml-org/llama.cpp/issues/3825；https://github.com/ggml-org/llama.cpp/issues/4097；https://github.com/ggml-org/llama.cpp/pull/13194

### INF-1232 `swa_sliding_cache_premature_roll` SlidingWindowCache 守卫差一，提前 roll 留下零值槽

`故障` · `active` · `infra_off` · `yes` · `documented` · `NEW` · 发现 待定 · 旧号 K2.2

- **原因对象**：KV缓存 / transformers / HybridCache 滑窗更新守卫条件
- **原因行为**：_sliding_update 守卫 cache_position >= max_cache_len - 1 在 seq_len 首次达到窗口时提前一步触发，缓存未满就被 torch.roll 平移，旧 token 被移出并留下未初始化零槽
- **影响对象**：KV状态；输出内容（质量退化）
- **影响详情**：长上下文解码中 seq_len 越过 SWA 窗口后，窗口边界附近的注意力被破坏，零值 K/V 槽与错位一格的旧 token 使生成持续劣化
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/huggingface/transformers/issues/37574

### INF-1233 `streamingllm_position_not_recompacted_after_eviction` StreamingLLM 驱逐后未重新压紧位置，RoPE 旋转越出预训练窗口

`故障` · `active` · `infra_off` · `partial` · `documented` · `var` · 发现 待定 · 旧号 K2.3

- **原因对象**：KV缓存 / 通用 / 驱逐后的缓存内位置分配
- **原因行为**：驱逐中间 token 后没有按 token 在缓存内部的位置连续重新编号，而是保留原始带间隙的位置，形成逐 token 可变的位置间隙
- **影响对象**：KV状态；隐状态（静默错误）
- **影响详情**：解码时 RoPE 把保留下来的键旋转到预训练位置窗口之外的相位区；源文档未给出外部可观察症状
- **改进措施**：检测 待定；修复 待定
- **来源**：https://arxiv.org/abs/2309.17453
- **关联**：var:INF-0788

### INF-1234 `llamacpp_self_extend_grpattn_position_fold` llama.cpp self-extend 平移回收出错，破坏位置→KV 映射

`故障` · `active` · `ambiguous` · `other_stack` · `documented` · `var` · 发现 待定 · 旧号 K2.4

- **原因对象**：KV缓存 / llama.cpp / self-extend 缓存平移回收
- **原因行为**：self-extend 在 grp_attn_w 窗口内按 grp_attn_n 整除折叠位置并据此平移缓存，其 shift/reclaim 逻辑有误，位置→KV 映射被下取整折叠破坏
- **影响对象**：KV状态；输出内容（静默错误）
- **影响详情**：启用 self-extend 且上下文超过 grp_attn_w 时，上下文过早耗尽（平移后 KV-cache 满），或在折叠位置下输出错误（#7742 还带架构门控）
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/ggml-org/llama.cpp/issues/9171；https://github.com/ggml-org/llama.cpp/issues/7742
- **关联**：var:INF-0788

### INF-1235 `overlap_scheduler_committed_len_off_by_one` SGLang overlap 调度致前缀缓存多提交一个 token

`故障` · `active` · `model_on` · `partial` · `documented` · `NEW` · 发现 待定 · 旧号 K2.5

- **原因对象**：KV缓存 / sglang / radix 前缀缓存提交长度记账
- **原因行为**：overlap 调度器在批次 N 的 EOS-finish 前执行下一批 prepare_for_decode（kv_committed_len+1），使 cache_finished_req 向 radix 树多提交一个 token
- **影响对象**：KV状态；跨请求/跨样本数据（静默错误）
- **影响详情**：共享该前缀的后续请求复用了本应重算的边界 token 的 KV，仅在复用接缝处有单位置索引偏移；源文档未给出外部症状
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/24590；https://github.com/sgl-project/sglang/pull/24745

### INF-1236 `trtllm_chunked_context_reuse_slice_overflow` TRT-LLM 分块上下文 KV slice 索引越界，输出截断或崩溃

`故障` · `active` · `ambiguous` · `other_stack` · `documented` · `NEW` · 发现 待定 · 旧号 K2.6

- **原因对象**：KV缓存 / tensorrt-llm / 分块上下文 KV slice 索引
- **原因行为**：use_paged_context_fmha 与 enable_chunked_context 同时开启时，由 chunk 位置推出的 block/slice 索引超出每请求 KV 缓冲区
- **影响对象**：输出内容；可用性/进度（报错）
- **影响详情**：高并发或小 max_num_tokens 下输出被静默截断，部分请求抛出缓冲区溢出异常（slice 2044 exceeds buffer size 420）而崩溃；开启 enable_kv_cache_reuse 会更早触发
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/NVIDIA/TensorRT-LLM/issues/1377；https://github.com/NVIDIA/TensorRT-LLM/issues/1480

### INF-1237 `encdec_cross_attention_cache_stale_or_misordered` 编解码模型交叉注意力缓存未随 beam 重排或缓冲区尺寸错误

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 待定 · 旧号 K2.7

- **原因对象**：KV缓存 / transformers / 交叉注意力 KV 缓存重排与尺寸
- **原因行为**：交叉注意力 K/V 算一次后每步复用：beam search 按 beam_idx 重排了自注意力缓存却没重排交叉注意力缓存；静态 EncoderDecoderCache 的 cross 缓冲区相对 encoder_seq_len 尺寸不当
- **影响对象**：KV状态；隐状态（静默错误）
- **影响详情**：源文档未给出外部症状；重排缺失在 batch>1 时暴露，会复用过期或长度错误的 cross KV，交叉注意力子层输出偏离而自注意力子层正常
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/huggingface/transformers/issues/35444；https://github.com/huggingface/transformers/issues/11788；https://github.com/vllm-project/vllm/pull/5964

### INF-1238 `static_cache_seqlen_dependent_gibberish` StaticCache 掩码未限到已填充前缀，贪心结果随缓存长度变化

`故障` · `active` · `infra_off` · `yes` · `documented` · `NEW` · 发现 待定 · 旧号 K3.1

- **原因对象**：KV缓存 / transformers / StaticCache 路径因果掩码构造
- **原因行为**：注意力掩码没有把注意力限制在已填充前缀上，cache_position 与掩码在 StaticCache 预分配的零缓冲区上错位，query 会注意到零值的未初始化槽
- **影响对象**：输出内容；隐状态（不可复现）
- **影响详情**：同一 prompt 的贪心解码结果随 max_new_tokens/max_cache_len 改变而改变，间或输出乱码；切到 DynamicCache 路径即消失
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/huggingface/transformers/issues/30400；https://github.com/huggingface/transformers/issues/30670

### INF-1239 `vllm_apc_samestep_uninitialized_block_read` vLLM APC 同一步命中未写入的满块，读到无效前缀 KV

`故障` · `active` · `infra_off` · `config` · `documented` · `var` · 发现 待定 · 旧号 K3.2

- **原因对象**：KV缓存 / vllm / cache_full_blocks 注册时机
- **原因行为**：同一步内 cache_full_blocks 在 GPU 前向写入 KV 之前，就把新分配的满块插入 APC 哈希表；同一步稍后的 get_cached_block 命中该块，把未初始化/清零或邻居序列的显存当作有效前缀 KV 读取。
- **影响对象**：KV状态；跨请求/跨样本数据（静默错误）
- **影响详情**：开启 APC 且同一步内有请求命中刚注册的块时，第一个共享前缀边界块携带清零或另一条序列的 KV；对输出的外部可见影响原文未给出。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/37152；https://github.com/vllm-project/vllm/issues/37076
- **关联**：var:INF-0394

### INF-1240 `llamacpp_kv_defrag_cell_move_corruption` llama.cpp KV defrag 搬移偏移错误，首次整理后持续乱码

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 待定 · 旧号 K3.3

- **原因对象**：KV缓存 / llama.cpp / KV defrag cell 搬移逻辑
- **原因行为**：llama_kv_cache_defrag_impl 搬移被占用的 cell 并重写 cell→位置映射时算出错误的 src/dst 偏移（量化缓存叠加并行槽时更严重），KV 落到错误位置
- **影响对象**：KV状态；输出内容（质量退化）
- **影响详情**：首次 defrag 之后输出持续乱码，直到进程重启；拷贝路径还可能直接段错误
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/ggml-org/llama.cpp/issues/12253；https://github.com/ggml-org/llama.cpp/issues/12259；https://github.com/ggml-org/llama.cpp/issues/3380；https://github.com/ggml-org/llama.cpp/pull/10873

### INF-1241 `offload_decode_block_index_desync` vLLM CPU 卸载解码块索引失步，重载后提供过期 KV

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 待定 · 旧号 K3.4

- **原因对象**：KV缓存 / vllm / CPU 卸载连接器 store 索引
- **原因行为**：_get_reqs_to_store 计算 num_blocks 时，解码期新块的哈希尚未进入 req.block_hashes，索引静默失败、_next_stored_block_idx 漂移，解码区 KV 从未被卸载
- **影响对象**：KV状态；输出内容（静默错误）
- **影响详情**：swap-in 重载之后，靠后的解码区位置被提供过期或缺失的 KV；源文档未单列外部症状
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/33864；https://github.com/vllm-project/vllm/issues/35507；https://github.com/vllm-project/vllm/pull/25856；https://github.com/vllm-project/vllm/pull/31341

### INF-1242 `kv_pool_v_head_dim_undersize` SGLang KV 池按 K 维度分配 V，v_head_dim 不同时写错槽

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 待定 · 旧号 K3.5

- **原因对象**：KV缓存 / sglang / get_size_per_token 尺寸计算
- **原因行为**：get_size_per_token() 按 head_dim 统一计算 K 与 V 的每 token 尺寸；v_head_dim≠head_dim 时 V 按 K 维度分配，set_kv_buffer 把 V 写进形状错误的槽
- **影响对象**：KV状态；隐状态（待定）
- **影响详情**：在 v_head_dim≠head_dim 的模型（MiMo V2.5）开启 HiCache 时触发；源文档未给出外部症状
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/26734

### INF-1243 `hybrid_cache_sliding_window_misroute` Gemma2 漏传 sliding_window，滑动层被误派静态更新

`故障` · `active` · `infra_off` · `partial` · `verified` · `var` · 发现 待定 · 旧号 K3.6

- **原因对象**：KV缓存 / transformers / Gemma2 HybridCache 滑动层分派
- **原因行为**：Gemma2Attention.forward 的 cache_kwargs 缺 sliding_window，HybridCache.update 把滑动层误派给 _static_update，按绝对索引写入，越出较小的滑动缓冲区
- **影响对象**：KV状态；隐状态（静默错误）
- **影响详情**：长上下文下交错的滑动（局部）层发生错误槽写入或索引越界，全注意力层不受影响；源文档未单列外部症状
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/huggingface/transformers/issues/35711；https://github.com/huggingface/transformers/pull/35681；https://github.com/huggingface/transformers/pull/35691
- **关联**：var:INF-0790

### INF-1244 `trtllm_int8_kv_reuse_plugin_abort` TRT-LLM 强开 INT8 KV 复用，约 3000 请求后 SIGABRT

`故障` · `active` · `ambiguous` · `other_stack` · `documented` · `var` · 发现 待定 · 旧号 K3.7

- **原因对象**：KV缓存 / tensorrt-llm / paged-context INT8 KV 路径
- **原因行为**：KV 复用依赖不支持 INT8 KV 的 paged-context FMHA（有断言守卫）；强行启用后，该 INT8 路径在复用期间错误处理量化块布局与 scale
- **影响对象**：KV状态；可用性/进度（报错）
- **影响详情**：强行启用后能运行，约 3000 个请求后在 GPTAttentionPlugin::enqueue 中 SIGABRT 中止；源文档称崩溃前复用块的 INT8 反量化给出粗糙的 KV
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/NVIDIA/TensorRT-LLM/issues/2166；https://github.com/NVIDIA/TensorRT-LLM/issues/2112
- **关联**：var:INF-0675

### INF-1245 `llamacpp_unified_cache_seqid_stream_mismap` llama.cpp 统一缓存 seq_to_stream 映射错位致断言崩溃

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `var` · 发现 待定 · 旧号 K3.8

- **原因对象**：KV缓存 / llama.cpp / 统一缓存 seq_to_stream 映射
- **原因行为**：b5913 拆分 K/V 缓冲区后统一缓存经 seq_to_stream 表索引；加载已缓存 prompt 或喂入意外 seq_id 时索引到错误 stream，序列的 cell 被归属到另一 stream。
- **影响对象**：KV状态；可用性/进度（报错）
- **影响详情**：有文档记录的主要模式是 GGML_ASSERT(seq_id < seq_to_stream.size()) 硬断言崩溃；不触发断言的路径上推测会出现跨 stream 的非致命 KV 泄漏。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/ggml-org/llama.cpp/issues/14847；https://github.com/ggml-org/llama.cpp/issues/15215；https://github.com/abetlen/llama-cpp-python/issues/2045
- **关联**：var:INF-0791

### INF-1246 `vllm_kimi_k3_prefix_cache_boundary_miss` vLLM Kimi-K3 前缀缓存在 1536-token 边界未命中

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-07-29 · 旧号 K3.9

- **原因对象**：KV缓存 / vllm / 前缀缓存块哈希边界对齐
- **原因行为**：prompt 在 1536-token（混合注意力块大小）边界上被切分时，缓存哈希计算与实际块边界不对齐，缓存键失效
- **影响对象**：性能/资源（性能退化）
- **影响详情**：Kimi-K3 的 prompt 长度恰为 1536-token 边界时，本应命中的前缀缓存未命中，导致重复 prefill；计算结果正确
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/50235

### INF-1247 `vllm_flexattention_paged_kv_offset_int32_overflow` vLLM FlexAttention 分页 KV 块偏移 int32 溢出回绕

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-07-30 · 旧号 K3.10

- **原因对象**：KV缓存 / vllm / FlexAttention 块偏移打包
- **原因行为**：FlexAttention 后端把 paged K/V 的 block offset 打包为 int32，num_gpu_blocks 大到偏移超过 2**31 时回绕，指向无效地址或错误的合法 block
- **影响对象**：KV状态；输出内容；可用性/进度（静默错误）
- **影响详情**：高显存卡（H800/H200/B200）加大 KV 池（长上下文）使 num_gpu_blocks 逼近阈值时触发：回绕到无效地址则崩溃，落到错误的合法 block 则读错位 KV、静默输出错误
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/50427

### INF-1248 `vllm_lora_prefix_cache_name_id_collision` vLLM 前缀缓存哈希用 lora_name，同名适配器复用他人 KV

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 待定 · 旧号 K4.1

- **原因对象**：KV缓存 / vllm / LoRA 前缀缓存额外哈希键
- **原因行为**：_gen_lora_extra_hash_keys 用 lora_name 而非唯一的 lora_int_id 生成块哈希，LoRARequest 的 __eq__/__hash__ 也只看 lora_name，同名不同权重适配器哈希相同
- **影响对象**：KV状态；跨请求/跨样本数据（静默错误）
- **影响详情**：前缀缓存开启且同名适配器被热切换/重载后，适配器 B 的请求命中 A 权重下算出的 KV 块，输出语法正常但内容偏向另一适配器，无一致性校验报警
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/30931；https://github.com/vllm-project/vllm/issues/42125

### INF-1250 `cacheblend_chunkboundary_crossattn_miss` CacheBlend 式 RAG chunk KV 全量复用缺失跨块交叉注意力

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 待定 · 旧号 K4.3

- **原因对象**：KV缓存 / 通用 / 非前缀 RAG chunk KV 复用
- **原因行为**：各 RAG chunk 的 KV 独立预计算后拼接复用，非前缀 chunk 的 K/V 从未编码对前序 chunk 的 cross-attention；跳过 CacheBlend 选择性重算时缺失全部 chunk 间交叉项
- **影响对象**：KV状态；输出内容（质量退化）
- **影响详情**：多 chunk RAG 回答相对完整预填充参考质量下降，错误集中在需要跨 chunk 整合信息的问题上；对 batch 结构不敏感
- **改进措施**：检测 待定；修复 待定
- **来源**：https://arxiv.org/abs/2405.16444

### INF-1251 `llamacpp_slot_restore_stale_reuse_contamination` llama.cpp 槽位复用过期 KV，请求被他人会话前缀污染

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `var` · 发现 待定 · 旧号 K4.4

- **原因对象**：KV缓存 / llama.cpp / 槽位 KV 恢复与复用
- **原因行为**：源文档未给出机制字段；已知先前请求在同一槽上留下的 KV 被当前请求复用，使请求被条件化在另一会话的前缀上
- **影响对象**：跨请求/跨样本数据；KV状态（静默错误）
- **影响详情**：输出流畅但内容属于另一会话的对话；只在先前请求于同一槽上留下 KV 时触发
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/ggml-org/llama.cpp/issues/24746；https://github.com/ggml-org/llama.cpp/issues/17200；https://github.com/mybigday/llama.rn/issues/258
- **关联**：var:INF-0399

### INF-1252 `trtllm_ptuning_fake_id_reuse_collision` TRT-LLM p-tuning fake id 键碰撞，复用他人软提示 KV

`故障` · `active` · `model_on` · `other_stack` · `documented` · `var` · 发现 待定 · 旧号 K4.5

- **原因对象**：KV缓存 / tensorrt-llm / KV 复用块匹配键（p-tuning）
- **原因行为**：KV 复用只按 token id 匹配块；p-tuning 虚拟 token 用超出词表的 fake id，真正条件在 prompt-embedding 表里被缓存键忽略，省略 extra_ids 时不同 embedding 撞到同一节点
- **影响对象**：KV状态；跨请求/跨样本数据（静默错误）
- **影响详情**：embedding 不同但 fake id 相同的请求 B 复用 A 的虚拟 token KV，被 A 的软提示条件化，输出看似正常但遵循另一套 prompt-tuning 的指令倾向
- **改进措施**：检测 待定；修复 待定
- **来源**：引用：TensorRT-LLM KV-cache-reuse 文档（p-tuning / extra_ids 章节）。
- **关联**：var:INF-0395

### INF-1253 `lmcache_chunk_key_seed_role_mismatch` LMCache 调度与 worker 角色 chunk 键种子不一致

`故障` · `active` · `model_on` · `partial` · `documented` · `var` · 发现 待定 · 旧号 K4.6

- **原因对象**：KV缓存 / lmcache / TokenDatabase chunk 键种子
- **原因行为**：SCHEDULER 与 WORKER 算 per-chunk 键时用了不同的 TokenDatabase 子类/init_none_hash 种子（种子根因为维护者假设），调度器认为已缓存的 chunk 在 worker 侧查不到
- **影响对象**：KV状态；可用性/进度（静默错误）
- **影响详情**：CacheBlend 的 compute_layer(0) 直接崩溃报 "Layer 0 not loaded"；非崩溃分支引擎照常出词，但实际用到的 chunk 与调度器记账不一致
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/LMCache/LMCache/issues/3229
- **关联**：var:INF-0399

### INF-1255 `hybrid_mamba_ssm_attention_state_desync` 混合 Mamba 模型 SSM 状态与 attention KV 失步

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 待定 · 旧号 K4.8

- **原因对象**：KV缓存 / vllm / 混合模型 Mamba conv/SSM 状态管理
- **原因行为**：Mamba 层 conv+SSM 状态是不受 block manager 管理的逐请求张量；前缀缓存+推测解码下 attention KV 按 block gather/回滚，SSM 状态不知 gather 索引、不回滚被拒 token。
- **影响对象**：KV状态；隐状态（静默错误）
- **影响详情**：Jamba/NemotronH 类混合模型同时开启 prefix cache 与 MTP 推测解码时，vLLM #39809 记录为崩溃；非崩溃路径下 attention 层与 Mamba 层对同一 token 索引持有不同历史，两条状态流推进到不同位置。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/17140；https://github.com/vllm-project/vllm/issues/39809

### INF-1258 `vllm_pd_disagg_mm_kv_size_keyed_reuse` vLLM PD 分离多模态 KV 存储键不含图像内容致跨请求复用

`故障` · `active` · `infra_off` · `no` · `documented` · `var` · 发现 待定 · 旧号 K5.2

- **原因对象**：KV缓存 / vllm / SharedStorageConnector存储键
- **原因行为**：V1 P/D 分离中 SharedStorageConnector 的存储键仅由 prompt_token_ids（文本+随图像尺寸变化的占位符数）派生，不含图像内容，同尺寸不同内容的图像哈希到已有条目。
- **影响对象**：KV状态；跨请求/跨样本数据；输出内容（静默错误）
- **影响详情**：带同尺寸、不同内容图像的后续请求，decode 端加载前一请求的 KV，输出与前一请求逐字相同；见于 Qwen2.5-VL-3B 1P1D 部署。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/21175
- **关联**：var:INF-0395

### INF-1259 `pd_disagg_dside_page_reuse_stale_kv` PD 分离 decode 端页复用早于 KV 落地致读到过期 KV

`故障` · `active` · `infra_off` · `no` · `speculative` · `var` · 发现 待定 · 旧号 K5.3

- **原因对象**：KV缓存 / sglang / decode 端 KV 页释放与分配
- **原因行为**：（源 issue 未确认的假设机制）decode 端经 Mooncake RDMA 收 KV，已完成请求释放的页可能在新请求入站传输落地前分配给它，调度器读到回收页上的过期字节；bootstrap_room 元数据匹配，故无告警。
- **影响对象**：KV状态；输出内容；跨请求/跨样本数据（质量退化）
- **影响详情**：输出随请求计数渐进坍缩为乱码：准确率从第 1–50 个请求约 94–96% 降到第 151 个以后 25–41%，损害跨请求累积；重启 PD 可恢复。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/23020
- **关联**：var:INF-0399

### INF-1260 `trtllm_reuse_block_evict_before_kvtransfer` TRT-LLM PD 分离 block reuse 在 KV 发送前驱逐块

`故障` · `active` · `infra_off` · `no` · `documented` · `NEW` · 发现 待定 · 旧号 K5.4

- **原因对象**：KV缓存 / tensorrt-llm / reuse 树驱逐与异步 CacheSender
- **原因行为**：开启 enable_block_reuse 时 prefill 侧经 UCX 调度异步 CacheSender，executor 却视请求已完成，抢先把其 block 从 reuse 树驱逐，sendSync 找不到该 block。
- **影响对象**：KV状态；可用性/进度（静默错误）
- **影响详情**：4P1D UCX 部署下两种表现：发送根本未发生时请求死锁挂起；槽位已被重新分配时 decode 侧静默使用过期或外来的前缀 KV。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/NVIDIA/TensorRT-LLM/issues/12542

### INF-1262 `lmcache_partial_retrieval_token_block_mismatch` LMCache 部分检索致 token/block 计数不符引擎崩溃

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 待定 · 旧号 K5.6

- **原因对象**：KV缓存 / lmcache / retrieve 与 wait_for_save
- **原因行为**：慢速或跨服务器链路上 retrieve() 返回的 token 少于预期（如 4980 个只回来 512 个），token 计数与 block 计数随之不一致。
- **影响对象**：可用性/进度；KV状态（报错）
- **影响详情**：wait_for_save() 抛 AssertionError，引擎以 "Ref count of MemoryObj negative / Double free" 关闭退出；崩溃前已加载的那部分上下文是被截断的。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/LMCache/LMCache/issues/2204

### INF-1263 `hicache_pp_hosttree_rank_divergence` PP+HiCache 下各 rank host-tree 发散取到错误 KV

`故障` · `active` · `infra_off` · `no` · `documented` · `NEW` · 发现 待定 · 旧号 K5.7

- **原因对象**：KV缓存 / sglang / HiCache host-tree 预取与驱逐
- **原因行为**：PP+HiCache L3 下各 PP rank 各跑异步 prefetch 线程，PP0 可能先于 PP1 插入 host-tree 致 radix 树与哈希链发散；各自基于 time.monotonic() 的 LRU 驱逐放大发散。
- **影响对象**：KV状态；输出内容（不可复现）
- **影响详情**：不同 PP rank 持有互不一致的 KV 视图并取到不同/错误的 KV，输出随部署拓扑与预取时序不可复现地变化（原文按机制推断）。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/22607

### INF-1264 `disagg_stale_restart_metadata_deadlock` PD 分离 KV connector 元数据过期致请求挂起

`故障` · `active` · `no` · `documented` · `NEW` · 发现 待定 · 旧号 K5.8

- **原因对象**：KV缓存 / vllm / KV connector 对端元数据
- **原因行为**：Mooncake 分支为过期 engine_id；NIXL 分支触发层数断言。
- **影响对象**：可用性/进度（挂起）
- **影响详情**：NIXL 分支抛断言直接失败；Mooncake 分支请求永久挂在 WAITING_FOR_REMOTE_KVS，既不出词也不报错。需 P/D 分离部署。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/38840；https://github.com/vllm-project/vllm/issues/37745

### INF-1265 `pd_disagg_transfer_inflight_page_release_overwrite` PD 分离 abort 后在途 KV 传输覆写新请求的 KV 页

`故障` · `active` · `infra_off` · `no` · `verified` · `NEW` · 发现 待定 · 旧号 K5.9

- **原因对象**：KV缓存 / sglang / abort 时 KV 页释放与在途传输
- **原因行为**：请求因 abort/timeout 中止时 KV 页归还分配器，但 Mooncake 后端在途的 KV 传输未取消；页释放与传输完成间无同步（TOCTOU），新请求分到同批页后被旧写入覆盖。
- **影响对象**：跨请求/跨样本数据；KV状态；输出内容（静默错误）
- **影响详情**：无报错、无 crash、无 NaN，唯一症状是答案略错：与旧请求前缀重叠大时近似轻微偏置，完全不同时语义偏离但仍流畅。触发窗口窄，生产中频繁 abort 慢请求时实际命中。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/pull/32564

### INF-1266 `quantized_cache_first_token_sink_quantized` QuantizedCache 立即量化首个 sink token

`故障` · `active` · `model_on` · `yes` · `verified` · `var` · 发现 待定 · 旧号 K6.1

- **原因对象**：KV缓存 / transformers / QuantizedCache 全精度残差窗口
- **原因行为**：QuantizedCache 本应保留近期 token 的全精度残差窗口、只量化更旧的 token，实现却立即量化了第一个 token（attention sink），与 sink/近期保全精度的做法相反。
- **影响对象**：KV状态；输出分布（质量退化）
- **影响详情**：准确率损失集中在 sink token 上并在长生成中累积：Llama-3.2-3B WikiText-2 2-bit ppl 损坏 12.22 vs 修复 11.46。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/huggingface/transformers/issues/35185；https://github.com/huggingface/transformers/pull/35760
- **关联**：var:INF-0675

### INF-1268 `pyramidkv_layer_uniform_budget_misalloc` PyramidKV：逐层统一 KV 预算致低层关键 token 被丢弃

`故障` · `active` · `model_on` · `yes` · `documented` · `NEW` · 发现 待定 · 旧号 K6.3

- **原因对象**：KV缓存 / 通用 / KV 驱逐的逐层预算分配
- **原因行为**：对各层采用统一 cache 预算而非 PyramidKV 的算术衰减分配，在注意力收敛的稀疏高层保留无用 token，在注意力弥散的低层丢弃集体重要的 token。
- **影响对象**：KV状态；输出内容（质量退化）
- **影响详情**：原文按机制推断：给定总预算下长上下文检索/聚合能力下降，低层被饿死，答案遗漏早期上下文中的分散证据，且不报错。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://arxiv.org/abs/2406.02069；https://arxiv.org/abs/2411.18077

### INF-1269 `adakv_uniform_headbudget_misalloc` Ada-KV：逐头均匀 KV 预算致弥散头被饿死

`故障` · `active` · `model_on` · `yes` · `documented` · `NEW` · 发现 待定 · 旧号 K6.4

- **原因对象**：KV缓存 / 通用 / KV 驱逐的逐头预算分配
- **原因行为**：驱逐方法把一层的预算在各注意力头间均匀切分，忽略各头注意力集中度的差异，使稀疏头被过度服务、弥散头被饿死。
- **影响对象**：KV状态；输出内容（质量退化）
- **影响详情**：原文按机制推断：同等总预算下注意力输出偏离全注意力，损失主要落在弥散头上，表现为静默的长上下文质量下降而非崩溃。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://arxiv.org/abs/2407.11550

### INF-1270 `duoattention_retrieval_head_misclassified_streaming` DuoAttention 误判 retrieval 头为 streaming 头

`故障` · `active` · `model_on` · `yes` · `documented` · `NEW` · 发现 待定 · 旧号 K6.5

- **原因对象**：KV缓存 / 通用 / DuoAttention 逐头类型门控
- **原因行为**：DuoAttention 的 per-head 门控把需要完整 KV 的 retrieval head 误分类为只保留 sink+近期窗口的 streaming head，裁掉其中段上下文 KV，摧毁该头的长程检索。
- **影响对象**：KV状态；输出内容（静默错误）
- **影响详情**：原文按机制推断：长上下文中段信息的检索静默失败，该头无法把中段证据带入输出；上下文短于 sink+近期窗口时无可见影响。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://arxiv.org/abs/2410.10819

### INF-1271 `quest_page_minmax_criticality_underestimate` Quest 页 Min/Max 重要性估计漏选关键页

`故障` · `active` · `ambiguous` · `other_stack` · `documented` · `NEW` · 发现 待定 · 旧号 K6.6

- **原因对象**：KV缓存 / 通用 / Quest 页重要性 Top-K 选择
- **原因行为**：Quest 在 decode 时只加载 Top-K 页，用 query 与每页 Min/Max Key 的逐通道乘积估计页重要性；若页内唯一关键 token 不主导 min/max 包络，该页可能落在 Top-K 外被跳过。
- **影响对象**：隐状态；输出内容（静默错误）
- **影响详情**：原文按机制推断：特定 query 上出现静默检索失败（针尖 token 所在页被跳过），注意力低估真正的 argmax token；其余 decode 步与其他 query 完全正常。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://arxiv.org/abs/2406.10774

### INF-1273 `llamacpp_quantized_kshift_precision_break` llama.cpp 量化 K-cache 的 context shift 损坏

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `var` · 发现 待定 · 旧号 K6.8

- **原因对象**：KV缓存 / llama.cpp / 量化 K-cache 的 K-shift 路径
- **原因行为**：context swap/shift 对缓存 K 就地施加 RoPE 旋转；量化 K-cache（type_k=q4_0）下需逐 cell 反量化→旋转→重量化，这条路径被破坏或不受支持，旋转后的 key 出错。
- **影响对象**：KV状态；可用性/进度（报错）
- **影响详情**：实际部署中先触发 GGML_ASSERT 断言崩溃；若跳过断言，上下文平移后旧 token 的位置编码错乱，输出逐渐脱轨。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/ggml-org/llama.cpp/issues/5652；https://github.com/ggml-org/llama.cpp/issues/9743
- **关联**：var:INF-0675

### INF-1275 `reasoning_think_token_cache_pollution` reasoning 模式 think token 污染前缀缓存产生死 block

`故障` · `active` · `partial` · `documented` · `NEW` · 发现 待定 · 旧号 K6.10

- **原因对象**：KV缓存 / vllm / APC/radix 前缀缓存哈希链
- **原因行为**：启用 reasoning parser 时 vLLM APC/SGLang radix 对所有发出 token（含 <think>）做哈希并链到父节点；下一轮按 API 惯例剥离 thinking，在共享问题后首个 block 偏离缓存链。
- **影响对象**：性能/资源（性能退化）
- **影响详情**：第 1 轮答案的 KV 成为永久不可达的死 block，仍占用 pool 并参与 LRU 驱逐竞争；多轮 reasoning 会话前缀命中率持续下降，吞吐/延迟恶化，但输出正确。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/39321；https://github.com/sgl-project/sglang/issues/22373；https://github.com/sgl-project/sglang/pull/23315

### INF-1277 `nemotron_cudagraph_track_buffer_stale_tail` SGLang 混合 Mamba CUDA graph track 缓冲尾部过期

`故障` · `active` · `infra_off` · `partial` · `verified` · `var` · 发现 待定 · 旧号 K7.2

- **原因对象**：KV缓存 / sglang / CUDA graph mamba track 缓冲
- **原因行为**：CUDA graph 的 _replay_metadata 只重填 mamba_track_indices_buf 的 [:bs]，却把整个 max_bs 缓冲交给 ForwardMetadata，尾部残留旧 replay 的 track。
- **影响对象**：KV状态；输出分布（静默错误）
- **影响详情**：Mamba2 混合模型开 CUDA graph 且前缀命中 decode-seeded mamba 条目时读到损坏 track，logits 明显偏离（mean logprob diff≈0.61、KL≈1.55，关 CUDA graph 为≈0.002），输出语义漂移但流畅，无报错。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/pull/32555
- **关联**：var:INF-1284

### INF-1278 `cudagraph_cache_position_captured_constant` CUDA graph 捕获 cache_position 为常量致缓存锁死

`故障` · `active` · `infra_off` · `partial` · `verified` · `NEW` · 发现 待定 · 旧号 K8.1

- **原因对象**：KV缓存 / transformers / cache_position 长度计数器
- **原因行为**：在 StaticCache 上对 model.forward 用 torch.compile 的 CUDA graph 模式时，cache_position/累积长度计数器被捕获为常量，每次 replay 都读写同一批槽位。
- **影响对象**：KV状态；输出内容；跨请求/跨样本数据（静默错误）
- **影响详情**：缓存锁定在编译时使用的第一个 prompt：第二个 prompt 返回第一个 prompt 的内容，decode 输出无视新输入。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/huggingface/transformers/issues/30351；https://github.com/huggingface/transformers/pull/30476

### INF-1281 `specdec_eagle_lookahead_slot_uncontamination` vLLM V1 未给 EAGLE draft token 预留 KV 槽致覆盖

`故障` · `active` · `infra_off` · `partial` · `verified` · `NEW` · 发现 待定 · 旧号 K8.4

- **原因对象**：KV缓存 / vllm / V1 KV 槽分配（lookahead）
- **原因行为**：V1 只为 target 验证 token 分配 KV 槽，未为 EAGLE draft head 提议的 token 分配；因分页 block_table 跨层共享，draft head 把 K/V 写进了从未为它保留的槽。
- **影响对象**：KV状态；输出内容（静默错误）
- **影响详情**：已验证位置的 KV 被覆盖，后续 token 基于被污染的上下文生成，输出静默错误、无报错；仅在 vLLM V1 启用 EAGLE 推测解码时出现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/16370；https://github.com/vllm-project/vllm/issues/15901

### INF-1282 `specdec_radix_prefix_draft_topk_corruption` RadixCache 批内重复 prompt 致 draft 读到损坏上下文

`故障` · `active` · `infra_off` · `partial` · `documented` · `var` · 发现 待定 · 旧号 K8.5

- **原因对象**：KV缓存 / sglang / RadixCache 批内前缀共享
- **原因行为**：开启 RadixCache 前缀共享时，同一批次中重复 prompt 的 KV 由兄弟条目提供，推测 draft model 因此读到被损坏的上下文。
- **影响对象**：KV状态；输出内容（静默错误）
- **影响详情**：批内放入相同 prompt 时第一个的提议正确、重复者发出错误 topk_index 提议，接受率骤降甚至输出错误；单独发送同一 prompt 正常，disable_radix_cache=True 时消失。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/8726
- **关联**：var:INF-0399

### INF-1283 `specdec_tree_rejected_branch_kv_leak` 树注意力推测解码未裁剪被拒分支 KV 致幽灵位置泄漏

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 待定 · 旧号 K8.6

- **原因对象**：KV缓存 / 通用 / 树注意力被拒分支 KV 裁剪记账
- **原因行为**：Medusa/EAGLE 树注意力为每个候选分支写入 KV，只有一条根到叶路径存活；retrieve_indices/tree_indices 记账差一或跳过裁剪，使被拒分支的 KV 留在 cache 中。
- **影响对象**：KV状态；输出内容（静默错误）
- **影响详情**：后续 query 会注意到不属于任何被接受 token 的幽灵位置，输出静默偏移且泄漏随步数累积；cache 长度与占用看起来合法。仅在树注意力推测解码下触发。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://arxiv.org/abs/2401.10774

### INF-1292 `kv_online_scale_calc_hybrid_gdn_corruption` 混合 GDN 模型在线计算 FP8 KV scale 得到垃圾值

`故障` · `active` · `infra_off` · `partial` · `verified` · `NEW` · 发现 待定 · 旧号 K9.1

- **原因对象**：KV缓存 / vllm / calculate-kv-scales 哑前向
- **原因行为**：--calculate-kv-scales 用一次哑前向在线派生每层 FP8 KV scale；在 GDN+Attention 混合架构上 GDN 层在该 pass 中带着未初始化的循环状态，统计出的 scale 为垃圾值。
- **影响对象**：KV状态；输出内容（静默错误）
- **影响详情**：服务健康、无 error/warning，但生成出现幻觉输入、话题锁死重复乃至乱码；仅在启用 --calculate-kv-scales 且模型为 GDN+Attention 混合架构（Qwen3.5）时出现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/37554

### INF-1296 `fp8_kv_pd_disagg_arithmetic_error` PD 分离下 FP8 KV 跨节点传输无法往返致算术错误

`故障` · `active` · `infra_off` · `no` · `verified` · `NEW` · 发现 待定 · 旧号 K9.5

- **原因对象**：KV缓存 / vllm / PD 分离 FP8 KV 传输与反量化
- **原因行为**：FP8 KV 在 prefill 侧量化、跨节点传输后在 decode 侧反量化时无法往返，量化元数据与数值在传输边界上不一致。
- **影响对象**：KV状态；输出内容（静默错误）
- **影响详情**：NVFP4 的 PD 分离服务开启 --kv-cache-dtype fp8 时，连 2+2 这类基础算术都答错（得 5），服务无报错；改用 BF16 KV 则正常。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/42179

### INF-1297 `prefix_cache_hash_collision_no_validation` 前缀缓存哈希碰撞缺 token 校验而复用错误 KV 块

`故障` · `active` · `infra_off` · `config` · `verified` · `NEW` · 发现 待定 · 旧号 K9.6

- **原因对象**：KV缓存 / openvino-genai / 前缀缓存哈希碰撞处理
- **原因行为**：不同 token 序列哈希到同一前缀缓存键时，代码缺少 token 级校验（源码留有 TODO），直接复用该键下的 KV 块；属真正的哈希碰撞而非哈希输入不完整。
- **影响对象**：跨请求/跨样本数据；输出内容（静默错误）
- **影响详情**：受害请求的输出带上另一条无关请求的前缀语义，表现为答非所问或内容串台，无报错；发生概率取决于哈希碰撞率，极难复现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/openvinotoolkit/openvino.genai/issues/3722

### INF-1298 `prefix_cache_kv_block_use_after_free` KV 块分配器在驱逐压力下 use-after-free

`故障` · `active` · `infra_off` · `config` · `verified` · `NEW` · 发现 待定 · 旧号 K9.7

- **原因对象**：KV缓存 / vllm / KV 块分配器驱逐路径
- **原因行为**：KV 块分配器在驱逐压力下存在 use-after-free（TOCTOU 竞态）：块仍被引用时即被释放并重新分配。
- **影响对象**：KV状态；跨请求/跨样本数据（静默错误）
- **影响详情**：高负载下偶发输出错乱或内容串台，注意力读到已释放内存或另一序列的 KV；低负载复现不出。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/37076

### INF-1299 `prefix_cache_recompute_path_divergence` 前缀缓存命中/重算路径与完整 prefill 数值不等价

`故障` · `active` · `ambiguous` · `config` · `verified` · `var` · 发现 待定 · 旧号 K9.8

- **原因对象**：KV缓存 / vllm / 前缀缓存命中/重算路径（含 SGLang）
- **原因行为**：前缀缓存的重算/命中路径与完整 prefill 路径在数值上不等价；改变输入切分点会改变命中边界，使相同前缀走上数学等价但实际偏离的计算路径。
- **影响对象**：输出内容；数值结果（质量退化）
- **影响详情**：vLLM 为 recomputation 类负载启用前缀缓存后 GSM8K 78.4→60.1；SGLang 中相同前缀因切分点不同得到不同输出。单条输出看起来完全合理。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/18055；https://github.com/sgl-project/sglang/issues/7857
- **关联**：var:INF-1276

### INF-1300 `cpu_offload_highconcurrency_kv_corruption` 高并发下 CPU offload 恢复的 KV 块损坏

`故障` · `active` · `ambiguous` · `partial` · `verified` · `var` · 发现 待定 · 旧号 K9.9

- **原因对象**：KV缓存 / vllm / KV CPU offload 恢复路径
- **原因行为**：高并发下从 CPU offload 恢复回显存的 KV 块的路径上存在竞态，恢复的块偶尔含有损坏数据。
- **影响对象**：KV状态；输出内容（静默错误）
- **影响详情**：低并发下完全正常；高并发下偶发整段输出切换到另一种语言或另一段内容，无报错。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/31210
- **关联**：var:INF-1241

### INF-1301 `lmcache_cache_salt_isolation_missing` LMCache 连接器缺失 cache_salt 致跨租户 KV 复用

`故障` · `active` · `infra_off` · `partial` · `verified` · `NEW` · 发现 待定 · 旧号 K9.10

- **原因对象**：KV缓存 / lmcache / LMCacheConnectorV1 缓存身份
- **原因行为**：LMCacheConnectorV1 在构造 LMCache 缓存身份时省略了 vLLM 的 cache_salt，本应按 salt 相互隔离的请求因此共享缓存身份。
- **影响对象**：跨请求/跨样本数据（静默错误）
- **影响详情**：本应互不可见的租户/会话请求之间发生 KV 复用，隔离失效，既是正确性问题也是隔离性（安全）问题；数值上无任何离群。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/LMCache/LMCache/issues/2878

### INF-1302 `prefix_cache_xxhash_missing_silent_empty` 未安装 xxhash 时前缀缓存静默返回空输出

`故障` · `active` · `ambiguous` · `config` · `documented` · `NEW` · 发现 待定 · 旧号 K9.11

- **原因对象**：KV缓存 / vllm / 前缀缓存 xxhash 哈希依赖检查
- **原因行为**：设置 prefix_caching_hash_algo='xxhash' 但未安装 xxhash 时，缺失依赖既不被检测也不抛异常，属控制平面缺陷，前向计算本身未被扰动。
- **影响对象**：输出内容（静默错误）
- **影响详情**：generate() 正常返回、状态码正常、无异常日志，但内容字段为空；仅在选用 xxhash 哈希算法且未安装该依赖时出现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/39338

### INF-1303 `mla_prefixcache_chunkedprefill_runtime_defect` MLA 前缀缓存与禁用 chunked prefill 组合运行时错误

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 待定 · 旧号 K9.12

- **原因对象**：KV缓存 / vllm / MLA 缓存与前缀缓存调度交互
- **原因行为**：MLA 模型的缓存与调度器优化之间的交互不正确，在启用前缀缓存且禁用 chunked prefill 的组合下命中运行时错误。
- **影响对象**：可用性/进度（报错）
- **影响详情**：DeepSeek-R1 类 MLA 模型在该配置组合下服务报运行时错误（大体是崩溃），属响亮失败而非静默故障。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/14069

### INF-1307 `mtp_prefixcache_kv_pollution_acceptance_collapse` vLLM 前缀缓存 + MTP 推测解码组合下用被污染的前缀 KV 做验证

`故障` · `active` · `ambiguous` · `partial` · `verified` · `NEW` · 发现 待定 · 旧号 K9.16

- **原因对象**：KV缓存 / vllm / 前缀缓存与 MTP 草稿验证的前缀 KV 共享
- **原因行为**：前缀缓存与 MTP 推测解码组合时，被接受的 token 是相对错误共享或被污染的前缀 KV 做的验证；已合并的调度器修复 #51113 未覆盖 hybrid Mamba/GDN 场景。
- **影响对象**：KV状态；输出内容；性能/资源（质量退化）
- **影响详情**：两开关单独启用均正常，组合后准确率降约 20%；多轮下草稿接受率坍缩到 0、吞吐坍缩、输出成乱码；hybrid Mamba/GDN 模型上部分响应退化为 !!!! 或空 completion，v0.28.0 仍复现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/43559；https://github.com/vllm-project/vllm/issues/36872；https://github.com/vllm-project/vllm/issues/53912；https://github.com/vllm-project/vllm/pull/51113

### INF-1310 `specdec_kv_slot_boundary_faults` 推测解码 KV 槽/索引边界故障（合并：槽越界、形状不匹配、接受率坍缩）

`故障` · `active` · `infra_off` · `partial` · `verified` · `NEW` · 发现 待定 · 旧号 K9.19

- **原因对象**：KV缓存 / vllm / 草稿路径 KV 槽索引与 seq_rm
- **原因行为**：KV 槽/索引边界出错：(a) 长序列 MTP 的 KV 槽越界；(b) DeepSeek-V4 EAGLE 草稿路径 req_pool_indices 与 out_cache_loc 形状不符；(c) GDN 不允许部分 seq_rm。
- **影响对象**：可用性/进度；性能/资源（报错）
- **影响详情**：(a)(b) 表现为非法内存访问或形状断言崩溃；(c) 在恰落于 KV 槽边界的特定上下文长度上，MTP 草稿接受率突然归零、吞吐坍塌，而输出仍正常。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/40756；https://github.com/sgl-project/sglang/issues/24747；https://github.com/ggml-org/llama.cpp/issues/23658；https://github.com/ggml-org/llama.cpp/pull/22400

### INF-1312 `tgi_prefix_cache_discontiguous_slot_misindex` TGI paged-v3 误设槽连续，KV 写进其他请求的槽

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 待定 · 旧号 K10.1

- **原因对象**：KV缓存 / tgi / paged-v3 KV 槽索引计算
- **原因行为**：paged-v3 用 slots.expand + arange_int 计算 KV 槽索引，隐含假设槽序号连续，但 RadixAllocator 发放非连续槽，追加的 KV 按错误序号写进其他请求的缓存槽。
- **影响对象**：KV状态；跨请求/跨样本数据；输出内容（静默错误）
- **影响详情**：启用 prefix caching 的多请求服务上，命中前缀后生成的尾部内容被污染，出现跨请求内容串味；单请求难以复现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/huggingface/text-generation-inference/pull/2711

### INF-1313 `vllm_kv_sharing_target_layer_not_before_current` vLLM 跨层 KV 共享目标层被推导到尚未计算的层

`故障` · `active` · `ambiguous` · `partial` · `documented` · `NEW` · 发现 待定 · 旧号 K10.2

- **原因对象**：KV缓存 / vllm / 跨层 KV 共享目标层名推导
- **原因行为**：YOCO/Gemma-3n 类跨层 KV 共享中，kv_sharing_target_layer_name 被错误推导到并非在当前层之前计算的层，共享层读取的是尚未写入的 KV。
- **影响对象**：KV状态；隐状态（静默错误）
- **影响详情**：仅 KV 共享层输出异常，读到未初始化内存或上一请求残留的 KV；生成早期最明显，随解码推进缓存被真实写入后自行恢复。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://arxiv.org/abs/2405.05254

### INF-1314 `flashinfer_append_paged_kv_varlen_cudagraph_oob_write` FlashInfer KV 追加内核在 CUDA graph 下写入填充槽

`故障` · `active` · `ambiguous` · `partial` · `documented` · `NEW` · 发现 待定 · 旧号 K10.3

- **原因对象**：KV缓存 / flashinfer / AppendPagedKVCacheKernel
- **原因行为**：固定批次 CUDA-graph 捕获下，AppendPagedKVCacheKernel 按 positions[i] 写入却不按真实序列长度做边界约束，批内对齐用的零填充槽被写进分页 KV cache，覆盖真实 KV。
- **影响对象**：KV状态；输出内容（质量退化）
- **影响详情**：仅固定批 CUDA-graph 路径触发、非 graph 路径正常；≥真实序列长度的位置出现零 KV 条目，输出退化且与批内其他序列的长度分布相关。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/flashinfer-ai/flashinfer/issues/919

### INF-1315 `lckv_warmup_layer_boundary_misclassified` LCKV warmup 层分类差一致误复用顶层 KV

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 待定 · 旧号 K10.4

- **原因对象**：KV缓存 / 通用 / LCKV warmup/condensed 层分类
- **原因行为**：LCKV 的 warmup 与 condensed 层分类存在差一错误，使某个本应自行计算 KV 的 warmup 层错误复用 condensed 顶层的 KV。
- **影响对象**：KV状态；隐状态（质量退化）
- **影响详情**：模型在 warmup/condensed 接缝层附近出现难以归因的质量下降，接缝层之外行为基本正常。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://arxiv.org/abs/2405.10637

### INF-1317 `gemma3n_sharedkv_batched_rotating_cache_geometry_desync` mlx-lm Gemma-3n 共享 KV 层忽视旋转缓存几何致读位错位

`故障` · `active` · `ambiguous` · `partial` · `verified` · `NEW` · 发现 待定 · 旧号 K10.6

- **原因对象**：KV缓存 / mlx-lm / Gemma-3n 共享 KV 层缓存读取
- **原因行为**：Gemma-3n 共享 KV 层把缓存当 (keys, values) 读取，忽视 BatchRotatingKVCache 的 offset、left_padding 与环形旋转，读位置偏移 offset − left_padding。
- **影响对象**：KV状态；输出内容（质量退化）
- **影响详情**：mlx-lm 上 Gemma-3n 输出乱码（returns garbage output），只在批处理带左填充且缓存已回绕之后出现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/ml-explore/mlx-lm/issues/1384

### INF-1318 `cachedattention_truncation_stale_positional_kv_reuse` CachedAttention 式 KV 持久化烘焙 RoPE 相位致截断后错位

`故障` · `active` · `infra_off` · `yes` · `documented` · `var` · 发现 待定 · 旧号 K10.7

- **原因对象**：KV缓存 / 通用 / 分层 KV 存储的会话复用与截断
- **原因行为**：CachedAttention 式分层 KV 存储把 RoPE 相位烘焙进持久化的 K；恢复的会话历史被截断后存活 token 逻辑位置整体下移，但存储的 K 仍保留旧相位，键侧角度偏离一个截断长度。
- **影响对象**：KV状态；输出内容（静默错误）
- **影响详情**：会话恢复并截断历史后，模型对早段上下文的指代与距离判断错乱；首次会话完全正常。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://arxiv.org/abs/2403.19708
- **关联**：var:streamingllm_position_not_recompacted

### INF-1320 `paged_kv_last_page_len_zero_at_page_boundary` 分页 KV 的 last_page_len 在页边界取 0 致最后一整页被丢弃

`故障` · `active` · `ambiguous` · `partial` · `documented` · `NEW` · 发现 待定 · 旧号 K10.9

- **原因对象**：KV缓存 / flashinfer / 分页 KV last_page_len 计算
- **原因行为**：用 seq_len % page_size 计算 last_page_len，序列长度恰落在页边界时得到 0 而非 page_size，有效注意力长度据此计算，最后一整页被排除在可见上下文外。
- **影响对象**：KV状态；输出内容（质量退化）
- **影响详情**：仅在长度为 page_size 整数倍时触发，周期恰为 page_size：生成质量随长度周期性忽好忽坏，页边界步上模型忘掉最近一整页上下文。
- **改进措施**：检测 待定；修复 待定
- **来源**：引用：来源：FlashInfer append/kv_layout 文档 *（候选原引用 #1353 有误；

### INF-1322 `radix_prefix_reuse_overmatch_or_boundary_stale` 前缀复用过度匹配或复用过期边界块致跨请求 KV 串味（合并）

`故障` · `active` · `infra_off` · `partial` · `documented` · `var` · 发现 待定 · 旧号 K10.11

- **原因对象**：KV缓存 / tgi / RadixTrie::find 与边界块复用
- **原因行为**：(a) TGI RadixTrie::find 返回父节点/过度递归，报告出过长的共享前缀；(b) SGLang 在 prefix_len == block_size 时复用过期边界块；均使接缝后 KV 与当前请求不对应。
- **影响对象**：跨请求/跨样本数据；KV状态；输出内容（不可复现）
- **影响详情**：高缓存命中率场景下偶发内容串味，生成中掺入另一请求前缀的语义，且随流量模式而不可复现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/huggingface/text-generation-inference/pull/3067；https://github.com/sgl-project/sglang/issues/22819
- **关联**：var:INF-0399

### INF-1324 `crosslayer_shared_kv_wrong_source_layer_or_headgroup` 跨层 KV 共享锚层/owner-map 映射错误致跟随层读错源 KV

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 待定 · 旧号 K10.13

- **原因对象**：KV缓存 / 通用 / 跨层 KV 共享锚层与 owner-map
- **原因行为**：跨层 KV 共享（CLA/YOCO/LCKV/MLKV）中锚层 (layer // sf) * sf 差一、sharing_factor 取错或 owner-map 写错，使跟随层对另一层或头组的 K/V 做注意力。
- **影响对象**：KV状态；隐状态（质量退化）
- **影响详情**：模型在特定深度出现表示断层，整体输出质量下降但无任何报错；owner 层行为与基线一致，只有被错映射的跟随层受影响。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://arxiv.org/abs/2405.12981

### INF-1325 `paged_kv_nhd_hnd_layout_axis_transpose` 分页 KV 的 NHD/HND 布局错配致头轴与序列轴被静默交换（合并）

`故障` · `active` · `infra_off` · `partial` · `verified` · `NEW` · 发现 待定 · 旧号 K10.14

- **原因对象**：KV缓存 / vllm / TRITON_ATTN 等后端 KV 布局硬编码
- **原因行为**：按错误 NHD/HND 布局读写分页 KV：TRITON_ATTN 在 NIXL/异构 TP 用 HND 时仍硬编码 NHD；FlashInfer trtllm_batch_decode 默认 HND 而其他 wrapper 用 NHD。
- **影响对象**：KV状态；输出内容（静默错误）
- **影响详情**：heads == page_size 时两种布局字节数相同、不报任何 shape 错误；输出跨位置近乎恒定、低熵，典型为重复吐 is is is… 而非正确答案 Paris。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/37703

### INF-1328 `exllamav2_paged_async_attention_accuracy_divergence` ExLlamaV2 分页 + 异步批处理路径准确率偏离单序列路径

`故障` · `active` · `ambiguous` · `other_stack` · `documented` · `NEW` · 发现 待定 · 旧号 K10.17

- **原因对象**：KV缓存 / exllamav2 / 分页 + 异步批处理注意力路径
- **原因行为**：ExLlamaV2 的分页（256-token 页）+ 异步批处理路径与连续单序列路径计算结果偏离；根因未确立，候选为页对齐填充、LSE 归约顺序或 block-table 索引方式。
- **影响对象**：隐状态；输出内容（质量退化）
- **影响详情**：同一模型同一权重，开启分页 + 异步批处理后任务准确率从 82.2% 降到 69.2%，单序列连续路径不受影响。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/turboderp-org/exllamav2/issues/676

### INF-1329 `sepllm_separator_set_tokenizer_mismatch_segment_drop` SepLLM 分隔符集与 tokenizer 不匹配致整段 KV 丢失

`故障` · `active` · `ambiguous` · `yes` · `documented` · `NEW` · 发现 待定 · 旧号 K10.18

- **原因对象**：KV缓存 / 通用 / SepLLM 分隔符保留式 KV 压缩
- **原因行为**：SepLLM 只保留各段尾部分隔符 token 的 KV；分隔符 id 列表取自错误 tokenizer 或输入分隔符稀疏时，某段没有任何被保留的分隔符，窗口溢出时整段被丢弃且不留摘要。
- **影响对象**：KV状态；输出内容（静默错误）
- **影响详情**：模型表现得像 prompt 中段从未出现，代码、中文、base64 等分隔符稀疏输入上尤易触发；cache 长度坍缩到预算以下，中段注意力质量约为 0，存活 KV 范数不变。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://arxiv.org/abs/2412.12094

### INF-1330 `pyramidinfer_prefill_early_layer_irrecoverable_prune` PyramidInfer 预填充期逐层剪枝 KV，早期误判不可逆

`故障` · `active` · `ambiguous` · `other_stack` · `documented` · `NEW` · 发现 待定 · 旧号 K10.19

- **原因对象**：KV缓存 / 通用 / PyramidInfer 预填充逐层 KV 剪枝
- **原因行为**：PyramidInfer 在预填充期间逐层剪枝 KV，早期层判为非关键而丢弃的 token 从整个上层堆栈消失；与事后预算不同，晚绑定的 token 无法被重新考虑。
- **影响对象**：KV状态；输出内容（质量退化）
- **影响详情**：关键信息出现在早期、到后期推理才用得上的任务质量骤降，普通短问答基本无感；上层 KV 长度小于深层注意力目标。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://arxiv.org/abs/2405.12532

### INF-1331 `infinigen_layerahead_speculative_misprefetch_drops_critical_kv` InfiniGen 提前一层推测预取失准致关键 KV 不在工作集

`故障` · `active` · `ambiguous` · `other_stack` · `documented` · `NEW` · 发现 待定 · 旧号 K10.20

- **原因对象**：KV缓存 / 通用 / InfiniGen 提前一层 KV 推测预取
- **原因行为**：InfiniGen 用第 i−1 层偏斜的注意力分数预测第 i 层所需 token 并从 CPU 预取到 GPU 工作集；真实注意力偏离推测时，关键 token 虽在 CPU 上却不在工作集中。
- **影响对象**：KV状态；输出内容（静默错误）
- **影响详情**：偶发、与输入强相关的信息丢失：多数请求正常，少数注意力模式跨层跳变的请求答错；本应落在缺失 token 上的 softmax 质量被重新分配，剩余 KV 均健康。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://arxiv.org/abs/2406.19707

### INF-1332 `shadowkv_stale_lowrank_key_projection_decode_tokens` ShadowKV 解码期沿用过期低秩键基致误差累积

`故障` · `active` · `model_on` · `other_stack` · `documented` · `NEW` · 发现 待定 · 旧号 K10.21

- **原因对象**：KV缓存 / 通用 / ShadowKV 预填充固定的低秩键基
- **原因行为**：ShadowKV 在预填充阶段用 SVD 固定一组低秩键基，解码期新产生的键仍用这组过期基投影，产生重建误差，进而导致 landmark top-k 误选、从 CPU 取回错误的 value。
- **影响对象**：KV状态；输出内容（质量退化）
- **影响详情**：生成越长质量越差：前几十个 token 正常，长回答后段逐渐跑偏；误差随生成累积，重建的解码 K 残差与预填充子空间正交。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://arxiv.org/abs/2410.21465

### INF-1333 `fastgen_prefill_profiled_head_policy_decode_drift` FastGen 预填充时冻结逐头压缩策略，解码期错配

`故障` · `active` · `ambiguous` · `other_stack` · `documented` · `NEW` · 发现 待定 · 旧号 K10.22

- **原因对象**：KV缓存 / 通用 / FastGen 逐头 KV 压缩策略剖析
- **原因行为**：FastGen 在预填充时为每个注意力头剖析一次压缩策略（局部/全局/特殊 token 等）并冻结；预填充时呈局部、解码时转为全局注意的头所需的 KV 槽早已被丢弃。
- **影响对象**：KV状态；输出内容（质量退化）
- **影响详情**：长生成中后段出现遗忘远处上下文的错误，仅在部分头行为随 prefill/decode 切换的模型或输入上出现；落在已剪枝位置的注意力质量随解码步单调上升。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://arxiv.org/abs/2310.01801

### INF-1337 `mlx_rotating_kvcache_swa_prefix_trim_break` mlx-lm RotatingKVCache 修剪忽略环形回绕致读错位 KV

`故障` · `active` · `ambiguous` · `hw_specific` · `documented` · `var` · 发现 待定 · 旧号 K10.26

- **原因对象**：KV缓存 / mlx-lm / RotatingKVCache 前缀修剪
- **原因行为**：SWA 的环形 RotatingKVCache 无法修剪到公共前缀；不安全变体按逻辑长度修剪而忽略环形回绕，回绕后物理槽索引不等于逻辑位置模窗口长度，解码读到过期或位置错误的窗口条目。
- **影响对象**：KV状态；输出内容（静默错误）
- **影响详情**：仅不安全变体且在回绕之后出现：多轮会话中指代错乱、出现陈旧内容；安全路径（整块抹除重算）只表现为前缀复用完全失效。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/ml-explore/mlx-lm/issues/980
- **关联**：var:INF-1232

### INF-1338 `ollama_context_shift_nkeep_zero_sink_bos_drop` Ollama 上下文平移在 n_keep=0 时驱逐 sink/BOS

`故障` · `active` · `infra_off` · `yes` · `documented` · `var` · 发现 待定 · 旧号 K10.27

- **原因对象**：KV缓存 / ollama / context-shift 驱逐与位置重基
- **原因行为**：Ollama 的 context-shift 在 n_keep=0 时把 BOS/attention-sink token 一并驱逐，且不重新基准化位置。
- **影响对象**：KV状态；输出内容（质量退化）
- **影响详情**：长会话触发 context shift 后突然开始复读或输出退化，重启会话即恢复；sink 消失后首层注意力熵出现尖峰。
- **改进措施**：检测 待定；修复 待定
- **来源**：引用：`gh`：ollama #2805、#3176 *（误引用的 #5303 已弃用）*。
- **关联**：var:INF-0785

### INF-1340 `block_retrieval_representative_misselection_block_unretrieved` Landmark/InfLLM 块代表元误选致含答案块未被检索

`故障` · `active` · `ambiguous` · `partial` · `documented` · `var` · 发现 待定 · 旧号 K10.29

- **原因对象**：KV缓存 / 通用 / 块检索的代表元相关性估计；位置编码 / 通用 / 检索块共享的均匀块距离 RoPE
- **原因行为**：块检索用块内代表元估计块相关性，代表元选得差（如被高范数 sink token 主导）时选错块或漏掉含答案的块；另对所有块共享的均匀块距离会错误偏移 RoPE。
- **影响对象**：KV状态；输出内容（静默错误）
- **影响详情**：长上下文检索任务偶发找不到明明存在的答案，同一文档的其他问题正常；含答案的块排在 top-k 之下、其 KV 从未参与注意力，块边界处可见位置扰动。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://arxiv.org/abs/2305.16300
- **关联**：var:INF-1271

### INF-1341 `sirllm_entropy_metadata_cache_desync` SirLLM 驱逐后熵数组与 KV 槽错位致驱逐决策失效

`故障` · `active` · `infra_off` · `other_stack` · `speculative` · `var` · 发现 待定 · 旧号 K10.30

- **原因对象**：KV缓存 / 通用 / SirLLM 每 token 熵数组与衰减
- **原因行为**：SirLLM 维护与 KV 张量并行的每 token 熵数组用于驱逐排序；驱逐后熵数组与 KV 的重新索引不一致，或衰减按步而非按轮应用，使槽位熵值不再对应实际 KV。
- **影响对象**：KV状态；输出内容（质量退化）
- **影响详情**：驱逐退化为近似随机：重要 token 被丢弃、无关 token 被保留，长上下文质量不稳定地下滑；来源标注为可信的推测性分层 bug，非已记录缺陷。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://arxiv.org/abs/2405.12528
- **关联**：var:INF-0786

### INF-1342 `gemini_gcs_object_mutation_stale_context_cache` Gemini 显式上下文缓存不追踪 GCS 对象变更致用旧 KV

`故障` · `active` · `model_on` · `other_stack` · `documented` · `NEW` · 发现 待定 · 旧号 K10.31

- **原因对象**：KV缓存 / gemini-api / 显式上下文缓存的缓存 id 失效
- **原因行为**：显式上下文缓存按 TTL（默认 60 分钟）存储内容的 KV，缓存 id 不追踪可变内容；TTL 内源 GCS 对象被改动后，缓存资源仍指向由过期字节算出的 KV。
- **影响对象**：KV状态；输出内容（静默错误）
- **影响详情**：用户更新源文件后继续提问，回答仍基于编辑前版本，最长持续一个 TTL 且无任何提示；隐状态完全健康，只有对比一次全新重算才能检测。
- **改进措施**：检测 待定；修复 待定
- **来源**：引用：来源：ai.google.dev/gemini-api/docs/caching（原文警告）。

## 调度/服务

### INF-0323 `hf_whisper_feature_extractor_single_nan_propagates_full_matrix` WhisperFeatureExtractor 单个 NaN 污染整张特征矩阵

`故障` · `active` · `infra_off` · `yes` · `documented` · `NEW` · 发现 2026-08-11 · 旧号 4.7

- **原因对象**：调度/服务 / transformers / WhisperFeatureExtractor
- **原因行为**：log_spec = np.maximum(log_spec, log_spec.max() - 8.0) 中 max 遇单个 NaN 返回 NaN，np.maximum 再把 NaN 传播到整个 80×3000 特征矩阵
- **影响对象**：输入编码；输出内容（静默错误）
- **影响详情**：波形含单个非有限样本（故障麦克风、损坏文件、预处理 bug）即令下游 ASR 从全 NaN 特征产出自信的垃圾转写，无错误、无警告
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/huggingface/transformers/issues/47885

### INF-0355 `sglang_multinode_tp_rank_divergence_nccl_deadlock` SGLang 多节点 TP rank 间 batch 分歧致 NCCL 死锁

`故障` · `active` · `infra_off` · `no` · `verified` · `NEW` · 发现 2026-08-02 · 旧号 5.8

- **原因对象**：调度/服务 / sglang / scheduler 跨 rank batch 同步
- **原因行为**：两个 scheduler rank 的 batch/step 循环分歧：一侧持续投递 logits all-gather，另一侧已退出循环停在 _broadcast_reqs_across_ranks，不会加入该 collective。
- **影响对象**：可用性/进度（挂起）
- **影响详情**：2 节点 TP=2 跑 DeepSeek-V4-Flash + DSPARK 时间歇性挂死（分钟到小时级），NCCL proxy 被楔住、GPU stream 不完成，最终 scheduler watchdog 杀服务；关闭请求超时仍发生。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/33289

### INF-0390 `sglang_pd_retract_rebootstrap_recompute_not_pinned_to_bound_prefill_dp_rank` SGLang P/D 回撤重算未绑定原 prefill DP rank

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-09-22 · 旧号 5.43

- **原因对象**：调度/服务 / sglang / retraction 重算的 DP 路由
- **原因行为**：retraction 后的 rebootstrap 重算只取客户端提示的 prefill DP rank（多为 None），由普通负载均衡派发；而 decode 侧 KV receiver 按 init 时 DP rank 绑定且复用。
- **影响对象**：可用性/进度（挂起）
- **影响详情**：多 prefill DP 且客户端未 pin 时，重算落到其它 DP rank：sender 与 receiver 各等 300s 超时后请求失败，无 crash；错路由 sender 还覆写 bootstrap 表 room→dp_rank 映射。dp_size==1 或显式 pin 不受影响。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/40730

### INF-0405 `sglang_pd_warmup_decode_ep32_dp_attn` SGLang P/D 预热在 EP32+DP attention 下配置损坏

`故障` · `active` · `infra_off` · `no` · `documented` · `NEW` · 发现 2026-07-29 · 旧号 6.12

- **原因对象**：调度/服务 / sglang / P/D 分离 decode 预热路径
- **原因行为**：P/D 分离式预热路径未正确处理 EP 与 DP attention 的组合，warmup 后的 CUDA graph / KV 传输配置不匹配。
- **影响对象**：KV状态（静默错误）
- **影响详情**：decode 端 EP32 + DP attention 的 P/D 部署预热后，decode 阶段的 attention 计算可能使用错误的 KV 或 routing；单进程路径不可复现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/32182

### INF-0425 `vllm_batch_invariant_not_disable_async_scheduling` vLLM batch-invariant 未约束 async 调度致不可复现

`故障` · `active` · `infra_off` · `config` · `verified` · `var` · 发现 2026-08-06 · 旧号 6.32

- **原因对象**：调度/服务 / vllm / async scheduling 组批时序
- **原因行为**：VLLM_BATCH_INVARIANT=1 只固定 kernel 逐行结果，不约束 scheduler 组批；默认开启的 async scheduling 提前组下一 batch，成员随 wall-clock 时序变化，无 guard。
- **影响对象**：数值结果；输出内容（不可复现）
- **影响详情**：VLLM_BATCH_INVARIANT=1 配默认 async scheduling（大 batch、变长长请求）时结果仍随时序变化，服务正常启动运行、无警告、无文档，确定性承诺静默失效；--no-async-scheduling 可消除，约损 3% 吞吐。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/51287
- **关联**：var:INF-0317

### INF-0427 `vllm_encoder_decoder_cache_slot_release_non_idempotent` vLLM 编码器-解码器缓存槽位每步重复释放致预算静默失效

`故障` · `active` · `infra_off` · `yes` · `verified` · `NEW` · 发现 2026-08-06 · 旧号 6.34

- **原因对象**：调度/服务 / vllm / 编码器缓存 free_encoder_input
- **原因行为**：allocate 每请求只跑一次，_free_encoder_inputs 每步调用 free_encoder_input，后者无引用守卫累加 num_free_slots；get_cached_input_ids 无条件返回全部 id。
- **影响对象**：性能/资源（静默错误）
- **影响详情**：Whisper/Voxtral 等 encoder-decoder 模型在首个 decode step 之后，num_free_slots 每步被重复计入，编码器缓存预算静默停止执行；无 crash、无 NaN。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/51228

### INF-0444 `vllm_check_stop_min_tokens_skip_max_len_hang` vLLM check_stop 因 min_tokens 跳过长度检查致挂起

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-08-08 · 旧号 6.51

- **原因对象**：调度/服务 / vllm / check_stop() 停止条件判断
- **原因行为**：check_stop() 在 num_output_tokens < min_tokens 时提前返回，同时跳过了其后的 max_model_len/max_tokens 长度检查。
- **影响对象**：可用性/进度；性能/资源（挂起）
- **影响详情**：请求在达到 min_tokens 前已撞上 max_model_len 时触发（显式或隐式 clamp 的 max_tokens，参数校验能通过）；调度器每步算出 0 个新 token 并跳过，请求永留 running queue，llm.generate() 不返回，KV 块泄漏。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/51299

### INF-0460 `sglang_chunked_prefill_final_abort_commit_race` SGLang 末块 prefill abort 竞态暴露 1 个 token

`故障` · `active` · `ambiguous` · `other_stack` · `documented` · `NEW` · 发现 2026-08-09 · 旧号 6.67

- **原因对象**：调度/服务 / sglang / overlap 循环 abort 与结果提交
- **原因行为**：abort 在末块 EXTEND 启动后、延迟结果提交前到达：结果处理在消费 pending finish 前已 append token；overlap 循环先调度下一 batch 再处理结果，缺 admission guard。
- **影响对象**：输出内容；KV状态（静默错误）
- **影响详情**：chunked prefill+overlap 调度下请求被 abort 时触发；无 crash，被取消请求先暴露 1 个 token 再以 abort 结束，pending-finish 请求进入乐观 decode，KV watermark 与 decode slot 多分配一个。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/pull/34153

### INF-0490 `vllm_inference_v1_generate_silent_choice_drop_n_gt_1` vLLM generate 端点 n>1 时静默丢弃部分 choices

`故障` · `active` · `infra_off` · `yes` · `documented` · `NEW` · 发现 2026-08-15 · 旧号 6.97

- **原因对象**：调度/服务 / vllm / generate 端点 choices 组装
- **原因行为**：/inference/v1/generate 端点在请求 n > 1 时只把部分 choices 放进响应，其余候选被丢弃。
- **影响对象**：输出内容（静默错误）
- **影响详情**：请求 n>1 时客户端收到的候选少于请求数量；无 crash、无报错，客户端无法区分是模型只生成了这些还是服务端丢弃。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/52398

### INF-0494 `vllm_hybrid_mamba_align_near_ceiling_requeue_livelock` vLLM hybrid Mamba 近池上限请求被接纳后永不重调度

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-08-16 · 旧号 6.101

- **原因对象**：调度/服务 / vllm / scheduler 准入与重调度
- **原因行为**：长度接近 KV 池上限的请求仍被接纳，chunk-prefill 到约 99% 占用后最后一块分配失败被去调度，此后永不重新调度，也没有“池永远装不下”的失败通道。
- **影响对象**：可用性/进度；指标/日志（挂起）
- **影响详情**：hybrid Mamba + --mamba-cache-mode align 下触发：请求零输出直至客户端超时，prefill 被整体丢弃，仅 vllm:num_requests 可见停滞；关联问题中 prefix cache 把丢弃的命中报成干净 miss。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/52530

### INF-0499 `vllm_qwen36_gdn_scheduler_admission_hard_wedge` Qwen3.6 GDN 引擎永久 wedge 而健康检查仍通过

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-08-16 · 旧号 6.106

- **原因对象**：调度/服务 / vllm / V1 scheduler 请求准入
- **原因行为**：根因未定位。V1 引擎 scheduler 进入永久不接纳任何请求的状态（py-spy 显示 MainThread idle 在 _process_engine_step），两种触发模式是否同根因未知。
- **影响对象**：可用性/进度；指标/日志（挂起）
- **影响详情**：Qwen3.6-27B（GDN、align、prefix caching）下：A 每请求 2 张大图且 max-num-batched-tokens=8192 首批即冻结（32768 正常）；B 并发约 180k 长多轮 prefill，调参无效。/health 仍 200，仅重启可恢复。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/52551

### INF-0509 `sglang_anthropic_adapter_pd_bootstrap_field_drop` SGLang Anthropic 适配层丢弃 PD bootstrap 字段

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-08-18 · 旧号 6.116

- **原因对象**：调度/服务 / sglang / AnthropicServing 请求转换
- **原因行为**：Anthropic 路径先解析为无 PD 字段的 AnthropicMessagesRequest，再在 _convert_to_chat_completion_request() 只用原生字段重建，bootstrap 字段被丢弃。
- **影响对象**：可用性/进度（报错）
- **影响详情**：PD 分离 + 外部 router 经 /v1/messages 发请求时，decode worker 报 without bootstrap room id 拒绝；同 body 走 /v1/chat/completions 成功，count_tokens 不受影响。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/35229

### INF-0512 `vllm_streaming_input_zero_token_turn_engine_die` vLLM streaming input 零输出轮非法重入队杀死引擎

`故障` · `active` · `infra_off` · `yes` · `documented` · `NEW` · 发现 2026-08-18 · 旧号 6.119

- **原因对象**：调度/服务 / vllm / streaming input 续写轮重入队
- **原因行为**：streaming-input 请求某续写轮生成零个输出 token（首个采样即 stop token）时，被重新放回 waiting 队列但状态仍为 RUNNING，scheduler 遇此非法状态直接抛 RuntimeError。
- **影响对象**：可用性/进度（报错）
- **影响详情**：可恢复会话 + temperature>0 + 非首轮即可命中，普通聊天流量也会触发：EngineCore 死亡并带走全部并发请求，5 实例 4 配置观测到 9 次引擎死亡。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/52693

### INF-0514 `grief_llm_serving_fuzzer_methodology` GRIEF：以定时多请求 trace 模糊测试 LLM 服务引擎

`方法` · `active` · `infra_off` · `not_target` · `documented` · `NEW` · 发现 2026-08-18 · 旧号 6.121

- **原因对象**：调度/服务 / 通用 / 服务引擎并发共享状态处理
- **原因行为**：服务引擎在并发、缓存与前缀共享、推测解码、adapter 与多租户调度中对共享状态的处理缺陷（如 KV-cache 隔离失效）。
- **影响对象**：跨请求/跨样本数据；性能/资源；可用性/进度（静默错误）
- **影响详情**：此类故障只在真实并发负载下显形，标准 model/safety/API 测试全部漏检；GRIEF 以轻量 oracle 检测 crash、hang、性能病与静默输出损坏，在 vLLM 与 SGLang 上发现 15 个漏洞（10 个确认、含 2 个 CVE）。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://arxiv.org/abs/2605.11202

### INF-0518 `vllm_voxtral_zero_embed_fallback_infinite_decode_hang` vLLM Voxtral 零嵌入 fallback 致无限解码与全引擎 hang

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-08-19 · 旧号 6.125

- **原因对象**：调度/服务 / vllm / Voxtral 空嵌入零填充 fallback
- **原因行为**：Realtime 模型收到空 multimodal embeddings 时，防御性 fallback 以零嵌入替代输入以避免 crash，且该路径缺乏解码终止条件。
- **影响对象**：可用性/进度（挂起）
- **影响详情**：Voxtral-Mini-4B-Realtime 经 /v1/audio/transcriptions 处理某特定 WAV 时，模型永不输出 EOS、无限解码（GPU 100%），/health 停止响应，只能 kill 进程；同文件确定性复现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/52926

### INF-0523 `vllm_clip_pooling_mm_load_empty_prompt_embedding_collapse` vLLM CLIP pooling 文本 embedding 在混合负载下变空串

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-08-20 · 旧号 6.130

- **原因对象**：调度/服务 / vllm / pooling 路径文本请求处理
- **原因行为**：pooling 路径的 text 请求处理被多模态批处理状态污染，疑似空 prompt 占位 embedding 被错误复用；具体根因 issue 仍 open。
- **影响对象**：输出内容；跨请求/跨样本数据（静默错误）
- **影响详情**：DFN5B-CLIP pooling 在持续图像批混合偶发 text 请求下，文本请求均返回空串 embedding（不同文本 cos=1.0），响应 200 良构，kNN 返回相同邻居；图像 embedding 与 /tokenize 正常，重启 pod 接流量即坏。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/53091

### INF-0525 `vllm_kimi_k3_chunked_prefill_hidden_state_divergence` Kimi-K3 分块 prefill 隐状态与一次性 prefill 分歧

`故障` · `active` · `ambiguous` · `partial` · `documented` · `NEW` · 发现 2026-08-20 · 旧号 6.132

- **原因对象**：调度/服务 / vllm / 跨 chunk 边界的 prefill 计算
- **原因行为**：chunked prefill 跨 chunk 边界的数值或上下文处理与 one-shot 不等价，分歧自第二个 chunk 引入、非 PP 通信所致；issue 已关闭但未给出最终机制。
- **影响对象**：隐状态；输出内容（不可复现）
- **影响详情**：Kimi-K3 TP8/PP2 上同一 32K 请求，4×8192 chunked 与 one-shot prefill 的隐状态在首个 PP stage 即超出 BF16 容差，生成随 --max-num-batched-tokens 变化；首个 8K chunk 逐位一致；无 crash。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/52909

### INF-0528 `vllm_scheduler_deferred_wedge_healthy_engine` vLLM scheduler deferred 计数泄漏致准入永久停止

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-08-21 · 旧号 6.135

- **原因对象**：调度/服务 / vllm / v1 scheduler 准入计数
- **原因行为**：准入计数把 skipped_waiting（deferred）请求与 running 一起计入 max_num_seqs 上限；deferred 请求被跳过后既不释放名额也不被重排，计数永久钉在上限。
- **影响对象**：可用性/进度（挂起）
- **影响详情**：deferred 请求累积至 max_num_seqs 后新请求不再被接纳（排队堆积或超时），而进程、指标、健康检查正常、KV cache 大量空闲，仅重启可恢复；无需特殊模型，#53868 在 CPU offload TP=2 下独立复现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/53130；https://github.com/vllm-project/vllm/issues/53868

### INF-0536 `vllm_xdrope_mm_metadata_strip_prefix_cache_hit_crash` vLLM XD-RoPE 模型前缀缓存命中时 grid 元数据被剥离致崩溃

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-08-23 · 旧号 6.143

- **原因对象**：调度/服务 / vllm / strip_covered_mm_data 豁免
- **原因行为**：strip_covered_mm_data 的元数据豁免只覆盖 M-RoPE（scheduler 只传 uses_mrope），XD-RoPE 模型计算位置所需的 image_grid_thw 在 prefix-cache 覆盖时被剥离。
- **影响对象**：可用性/进度（报错）
- **影响详情**：HunyuanOCR/HunyuanVL 等 XD-RoPE 模型开 prefix caching 时，相同 prompt 二次请求命中缓存即在 get_xdrope_input_positions 报 IndexError，EngineCore fatal、API server 关闭。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/53456

### INF-0548 `sglang_streaming_session_disconnect_silent_context_drop` SGLang streaming session 断开后多轮上下文静默丢失

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-08-27 · 旧号 6.155

- **原因对象**：调度/服务 / sglang / streaming session 断开回滚处理
- **原因行为**：客户端在流式 turn 中途断开时，断开处理路径把 session 增量状态（含 turn 0 之后继承的 KV）整体丢弃，而不是回滚到最后完成的 turn，且不向客户端报告该 turn 已损坏。
- **影响对象**：KV状态；输出内容（静默错误）
- **影响详情**：开启 streaming session 时 turn 1 在首个 SSE chunk 后断开并立即发 turn 2：HTTP 200 但 prompt_tokens=1、cached_tokens=0（期望 624），多轮上下文蒸发且无报错；再次触发可使 scheduler crash。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/36475

### INF-0554 `vllm_async_llm_streaming_session_inplace_prompt_mutation` vLLM AsyncLLM 流式输入原地修改调用方 prompt list

`故障` · `active` · `infra_off` · `yes` · `verified` · `NEW` · 发现 2026-08-29 · 旧号 6.161

- **原因对象**：调度/服务 / vllm / AsyncLLM 流式输入处理
- **原因行为**：AsyncLLM.generate 的客户端侧 streaming 输入处理把后续 chunk 的 token 原地追加到调用方传入的 prompt list 对象上，而不是使用副本；污染发生在引擎边界之外。
- **影响对象**：跨请求/跨样本数据；输入编码（静默错误）
- **影响详情**：调用方复用同一 list 对象时，64-token 会话后该 list 变为 76 个 token，后续请求静默携带未发送的 token（演示中注入的 token 888 出现在输出中）；服务端看来请求内容即如此，无从检测。OpenAI/HTTP serving 路径不受影响。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/54215

### INF-0563 `trtllm_kvv2_suspend_retains_peft_adapter_device_ownership` TRT-LLM 挂起请求不释放 PEFT adapter 致资源死锁

`故障` · `active` · `infra_off` · `other_stack` · `verified` · `NEW` · 发现 2026-08-30 · 旧号 6.170

- **原因对象**：调度/服务 / tensorrt-llm / KVCacheV2Scheduler 挂起/恢复
- **原因行为**：_suspend_request() 挂起时只释放 KV 页，不调 PeftCacheManager.mark_request_done(pause=True)；恢复路径也缺 add_request_peft。
- **影响对象**：可用性/进度；性能/资源（挂起）
- **影响详情**：PyTorch 后端 + PEFT/LoRA + 满载驱逐时，被挂起请求的 adapter 仍 device-active、页不可驱逐，另一 adapter 的 ensure_batch 无法从满 device cache 加载，形成资源死锁、吞吐塌缩；输出不受损。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/NVIDIA/TensorRT-LLM/issues/18407

### INF-0565 `vllm_pd_mamba_spec_padding_window_survives_align_split_truncation` vLLM 推测 padding 窗口被 Mamba 裁剪后未同步致引擎终止

`故障` · `active` · `infra_off` · `other_stack` · `verified` · `NEW` · 发现 2026-08-30 · 旧号 6.172

- **原因对象**：调度/服务 / vllm / waiting 路径推测 padding 窗口
- **原因行为**：新准入请求只剩 1 个 prompt token 时被 pad 到 1+K=8；hybrid Mamba align 块边界裁剪到 5 个物理 token 后，pad_spec_decode 仍为 true、仍发布 7 个推测占位。
- **影响对象**：可用性/进度（报错）
- **影响详情**：PD分离、hybrid Mamba align、推测解码且批内已有decode请求时准入1-token请求（调度序列触发）：proposer按7-token窗口解释5-token行，Ascend DSpark position/slot mapping非法致引擎终止；crash表象在后端。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/54392

### INF-0573 `sglang_responses_api_prompt_cache_key_silent_drop_global_namespace` SGLang 静默丢弃 prompt_cache_key 致缓存隔离失效

`故障` · `active` · `infra_off` · `other_stack` · `verified` · `var` · 发现 2026-09-01 · 旧号 6.180

- **原因对象**：调度/服务 / sglang / ResponsesRequest 字段适配
- **原因行为**：ResponsesRequest 未声明 OpenAI 标准字段 prompt_cache_key（仅有原生 cache_salt），Pydantic 忽略未知字段将其丢弃，适配器以 cache_salt=None 构造请求。
- **影响对象**：跨请求/跨样本数据（静默错误）
- **影响详情**：SGLang /v1/responses 收到 Codex 等客户端的 prompt_cache_key 时请求成功、无告警，但不同 key 的请求全部落入同一全局 radix-cache 命名空间，跨任务/租户缓存隔离失效，客户端无法察觉。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/37263
- **关联**：var:INF-0562

### INF-0585 `sglang_single_malformed_mm_request_engine_suicide_retry_cluster_takedown` SGLang 单个畸形多模态请求致引擎自杀并经重试击穿集群

`故障` · `active` · `infra_off` · `other_stack` · `verified` · `NEW` · 发现 2026-09-04 · 旧号 6.192

- **原因对象**：调度/服务 / sglang / 多模态 embedding 校验与异常隔离
- **原因行为**：缺少输入校验，embedding 短于 placeholder 数的多模态请求进入前向，_adjust_embedding_length() 抛 RuntimeError；调度器把单请求错误升级为 SIGQUIT 整引擎退出。
- **影响对象**：可用性/进度（报错）
- **影响详情**：SGLang 多模态 VLM 服务收到单个畸形 mm 请求即整台 8-GPU 引擎退出；请求未被消费，客户端重试 × 粘性路由把它送到下一节点，2 小时内 5 节点集群的 5 个引擎实例全部被击穿。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/38022

### INF-0588 `sglang_pdmux_mixed_batch_deadlock_after_long_prefill_hybrid_gdn` SGLang PDMux 长 prefill 后混合 batch 死锁

`故障` · `active` · `infra_off` · `other_stack` · `verified` · `NEW` · 发现 2026-09-04 · 旧号 6.195

- **原因对象**：调度/服务 / sglang / PDMux 调度器事件循环
- **原因行为**：≥4K-token prefill 改变事件顺序后，混合 decode+split-prefill batch 的多 stream 交错在 decode_stream.synchronize() 处形成循环等待。
- **影响对象**：可用性/进度（挂起）
- **影响详情**：hybrid GDN 模型在 PDMux 下经 ≥4K prefill 后的混合 batch 上调度器挂起，无 crash 无日志，吞吐归零、请求堆积；时序敏感，单轮 30–70% 触发，CI/冒烟可全过。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/37904

### INF-0594 `sglang_dp_attention_running_batch_pinned_max_minus_one` SGLang DP attention 下运行请求数恒为上限减一

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-09-05 · 旧号 6.201

- **原因对象**：调度/服务 / sglang / 调度器准入判断
- **原因行为**：chunked prefill 与 DP attention 组合下，调度器准入判断保留一个永不释放的 slot，每 DP rank 实际运行数恒比上限少 1（准入 off-by-one）；具体机制原文未详述。
- **影响对象**：性能/资源（性能退化）
- **影响详情**：GLM-5-FP8、DP=8、--max-running-requests 128 时 8 个 rank 的 #running-req 恒为 15（上限 16）且队列 ≥1；输出无误，集群容量损失约 1/D（约 6% 吞吐）并增加排队。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/38118

### INF-0595 `trtllm_responses_history_trim_nonyielding_loop_over_capacity` TRT-LLM Responses API 历史超容量且无完成轮次时裁剪死循环

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-09-05 · 旧号 6.202

- **原因对象**：调度/服务 / tensorrt-llm / ConversationHistoryStore
- **原因行为**：历史裁剪条件依赖 completed-turn 边界；输入超容量但无已完成 turn 可裁（纯 user 消息流或无 final 的 Harmony 历史）时，循环既不前进也不让出。
- **影响对象**：可用性/进度（挂起）
- **影响详情**：默认容量下存储 65 条连续 user 消息（无 completed turn），或 Harmony 历史无 final 消息时，请求挂起、服务端忙循环，无 crash；长会话 agent 客户端可触发，组件级复现无需 GPU。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/NVIDIA/TensorRT-LLM/issues/18759

### INF-0601 `sglang_b12x_w4a8_arena_launcher_value_vs_aggregate_prefill_undersize` SGLang B12X arena 定容小于聚合 prefill 需求

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `var` · 发现 2026-09-06 · 旧号 6.208

- **原因对象**：调度/服务 / sglang / B12X W4A8 共享 arena 定容
- **原因行为**：W4A8 共享 arena 按 launcher 的 token 值与 sparse decode/幂次阶梯定容，未计入 scheduler 聚合 prefill 与多模态 span 对齐过冲，准入请求所需 workspace 超出预留。
- **影响对象**：可用性/进度（报错）
- **影响详情**：B12X Vision 分支在首批聚合 prefill 时运行期双 rank 终止，启动自检干净；实测 launcher 供 8,192 行而聚合 prefill 准入 16,384 行，一请求需 412,183,756 B 仅预留 400,008,864 B。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/pull/38235
- **关联**：var:INF-0529

### INF-0604 `vllm_mm_uuid_verbatim_cache_key_cross_item_staleness_engine_teardown` vLLM 多模态 uuid 逐字作缓存键，跨 item 碰撞服错内容或拆引擎

`故障` · `active` · `infra_off` · `config` · `documented` · `var` · 发现 2026-09-06 · 旧号 6.211

- **原因对象**：调度/服务 / vllm / 多模态 uuid 缓存键
- **原因行为**：无 hash factors 时把客户端提供的多模态 item uuid 逐字用作 P0 processor cache 与 encoder-output cache 的唯一键，未将 uuid 绑定到首次缓存的 payload。
- **影响对象**：跨请求/跨样本数据；可用性/进度（静默错误）
- **影响详情**：同 uuid 不同内容复用时：处理后长度匹配则模型静默消费错误 item 的特征、输出合理但错；长度不匹配则 _merge_multimodal_embeddings 抛异常，V1 拆掉整个引擎树（DP8 部署全灭）。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/55549
- **关联**：var:INF-1301

### INF-0606 `vllm_encoder_decoder_cache_manager_per_step_double_free_budget_inflation` vLLM encoder-decoder 缓存每步重复释放致空槽虚增与超额接纳

`故障` · `active` · `infra_off` · `config` · `verified` · `var` · 发现 2026-09-07 · 旧号 6.213

- **原因对象**：调度/服务 / vllm / encoder-decoder 缓存释放记账
- **原因行为**：调度器每个 decode 步释放请求全部 encoder inputs，get_cached_input_ids 恒返回全部 id，free_encoder_input 缺引用计数守卫，无条件累加 num_free_slots。
- **影响对象**：可用性/进度；性能/资源（报错）
- **影响详情**：Whisper 类 ASR 服务中，对 E 个 embedding 的输入 num_free_slots 每个 decode 步虚增 E（100 槽 5 步后读数 300）；can_allocate 据此超额接纳并发 encoder prefill，运行期 OOM 或分配冲突 crash。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/55733
- **关联**：var:INF-0529

### INF-0614 `sglang_chunked_prefill_splits_encoder_region_negative_extend_num_tokens_crash` SGLang 分块 prefill 截入 encoder 区致计数为负崩溃

`故障` · `active` · `infra_off` · `other_stack` · `verified` · `NEW` · 发现 2026-09-09 · 旧号 6.221

- **原因对象**：调度/服务 / sglang / PrefillAdder、encoder 记账
- **原因行为**：准入 PrefillAdder 不感知 encoder 区，截断点可落在其内；prepare_encoder_info_extend 仍无条件减去完整 encoder_len，使 extend_num_tokens 为负。
- **影响对象**：可用性/进度（报错）
- **影响详情**：encoder-decoder 模型（Whisper、mllama 等）开 chunked prefill，encoder 区大于剩余 chunk 预算或多请求并发挤占时，out_cache_loc 长度断言失败，scheduler SIGQUIT、服务器被杀。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/38627

### INF-0617 `trtllm_pyexecutor_multimodal_metadata_failure_kills_all_ranks` TRT-LLM 多模态 metadata 失败逃逸事件循环杀死全部 rank

`故障` · `active` · `infra_off` · `other_stack` · `verified` · `var` · 发现 2026-09-09 · 旧号 6.224

- **原因对象**：调度/服务 / tensorrt-llm / PyExecutor 错误分类与事件循环
- **原因行为**：PyExecutor 错误分类未把 multimodal metadata/encoder 构建失败归为请求级错误，异常逃逸进 executor 事件循环，按默认语义对全部 rank 致命。
- **影响对象**：可用性/进度（报错）
- **影响详情**：单个 multimodal 请求 metadata 构建失败（如 hybrid-Mamba 缺 vocab_size，或 disable_mm_encoder 部署收到 image_url）时全部 rank 被 hard-kill、容器 exit 137；任何客户端可远程触发，重试会连杀 pod。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/NVIDIA/TensorRT-LLM/issues/18971
- **关联**：var:INF-0585

### INF-0625 `sglang_dp_scheduler_batch_is_full_sticky_wedge_after_cancel` SGLang DP 调度 batch_is_full 粘滞致 rank 永久拒收

`故障` · `active` · `infra_off` · `yes` · `verified` · `var` · 发现 2026-09-13 · 旧号 6.232

- **原因对象**：调度/服务 / sglang / batch_is_full 标志与准入判据
- **原因行为**：请求验证与 PrefillAdder 接纳判据相差一个预留页，近容量请求通过验证却永不被接纳；调度器随即置 batch_is_full=True，该标志无复位路径，请求取消后仍保持。
- **影响对象**：可用性/进度（挂起）
- **影响详情**：DP attention 下 input 长度落在近容量窗口（如 190720–190777）的请求被取消后，该 DP rank 静默停摆、永久拒收新请求，无错误日志，仅重启可恢复。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/39271
- **关联**：var:INF-0528

### INF-0642 `sglang_embedding_overlength_request_missing_abort_enqueued_blocks_to_timeout` SGLang 超长 embedding 请求漏调 abort 致客户端挂起至超时

`故障` · `active` · `infra_off` · `other_stack` · `verified` · `var` · 发现 2026-09-20 · 旧号 6.249

- **原因对象**：调度/服务 / sglang / embedding 请求长度校验分支
- **原因行为**：handle_embedding_request 的 validate_input_length 错误分支只构造错误消息，漏调 req.set_finish_with_abort，请求以全长度入队且没有 finish reason。
- **影响对象**：可用性/进度；性能/资源（挂起）
- **影响详情**：超过 max-total-tokens 的 /v1/embeddings 请求收不到预期的 HTTP 400，客户端挂起直到超时，同时占用全长度调度资源；无 crash、无错误响应。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/40459
- **关联**：var:INF-0585

### INF-0663 `sglang_gpu_jpeg_decode_overwrites_memory_queued_gpu_work_still_reads` SGLang GPU JPEG 解码覆写在途 GPU 工作仍在读取的内存

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-10-09 · 旧号 6.270

- **原因对象**：调度/服务 / sglang / GPU JPEG 解码输出 buffer
- **原因行为**：GPU JPEG 解码输出 buffer 的生命周期与异步消费 kernel 未同步，解码可覆写仍被排队 GPU 工作读取的内存，形成写读竞态。
- **影响对象**：输入编码（静默错误）
- **影响详情**：SGLang 多模态服务开启 GPU JPEG 解码（配置触发）时，多模态输入图像数据静默损坏，无 crash。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/43075

### INF-0668 `sglang_concurrent_start_zmq_rebind_port_plus_42_routes_requests_to_other_instance` SGLang 端口冲突时静默改绑 port+42 致请求串到其他实例

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-10-12 · 旧号 6.275

- **原因对象**：调度/服务 / sglang / get_zmq_socket 端口绑定重试
- **原因行为**：worker 侧 get_zmq_socket(bind=True) 绑定失败后静默改绑 port+42（仅一行日志），客户端、HTTP、metrics、disagg 仍连配置端口；settle_port 只在 parse 时查端口空闲。
- **影响对象**：跨请求/跨样本数据；可用性/进度（静默错误）
- **影响详情**：并发启动两实例且 scheduler 端口冲突时，请求被送达并执行于先占端口的另一实例，结果写入错误实例（拿到别的模型/会话结果），对端已退出时本实例永久等待；默认端口配置下多任务同启约 3/40 命中。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/43731

### INF-0818 `sglang_lfm2vl_per_image_cache_hit_wrong_length_scheduler_crash` SGLang 多模态缓存命中嵌入长度错误致 scheduler 崩溃

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-08-11 · 旧号 8.34

- **原因对象**：调度/服务 / sglang / MultiModalStaticCache 缓存
- **原因行为**：per-image cache hit 返回错误长度嵌入（7,126 而非 7,135），set() 不覆盖已存在 hash 致 poisoned 条目持续；HIP 上 LFM2-VL 未注册为 capable model。
- **影响对象**：输入编码；可用性/进度；性能/资源（报错）
- **影响详情**：H100 C=32 混合前缀 batch 下间歇落入 _adjust_embedding_length 致命分支致 scheduler 崩溃；该 image 每个后续 batch 重新 encode；HIP 上 vision encoder 重复调用 771 次。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/pull/33751

### INF-0902 `sglang_pretokenized_image_retokenized_mask_length_drift_mrope_crash` SGLang GLM 预分词请求 ids 与 mask 不等长致崩溃

`故障` · `active` · `infra_off` · `partial` · `verified` · `NEW` · 发现 2026-08-30 · 旧号 9.19

- **原因对象**：调度/服务 / sglang / GLM processor ids/mask 来源
- **原因行为**：pretokenized 请求中 input_ids 被换成客户端精确 ID，GLM 消费端仍读对 decode(input_ids) 重分词得到的 attention_mask；重分词对非规范序列不保长。
- **影响对象**：输入编码；可用性/进度（报错）
- **影响详情**：GLM-4V/GLM-5.3 的 pretokenized 多模态请求在 token 边界非规范时，MRoPE 用 N+1 长 ID 索引 N 长 mask 抛 IndexError；RL 多轮复用采样 ID 时最易命中，短单轮 VLM 测试不可见，表现为间歇崩溃。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/37097

### INF-0937 `vllm_batched_chat_echo_prepends_last_user_prompt` batched chat 的 echo 把用户 prompt 拼进回答

`故障` · `active` · `infra_off` · `yes` · `documented` · `NEW` · 发现 2026-08-16 · 旧号 10.21

- **原因对象**：调度/服务 / vllm / 批量 chat 路径 echo 拼接
- **原因行为**：batched 路径中 echo=true 把 conversation[-1] 的 content 前置到模型回答，未检查末消息 role；默认 add_generation_prompt=true 时末消息是 user prompt。
- **影响对象**：输出内容（静默错误）
- **影响详情**：/chat/completions batched 路径 + echo=true + 默认 add_generation_prompt 时触发；返回“用户 prompt + 助手回答”拼接文本，无 crash，下游以回答文本做解析/评估的管线（eval 基准、RAG 引用对齐等）被静默污染。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/52529

### INF-0948 `vllm_find_mm_placeholders_ambiguous_match_position` vLLM 多模态占位符重复 token 歧义匹配致嵌入错位

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-08-19 · 旧号 10.32

- **原因对象**：调度/服务 / vllm / find_mm_placeholders
- **原因行为**：find_mm_placeholders 按内容扫描定位占位符而非依据插入记账；PromptInsertion 插入的 token 与原 prompt 相邻 token 相同时，扫描命中更早的歧义匹配，返回错误的 start_idx。
- **影响对象**：输入编码；输出内容（静默错误）
- **影响详情**：插入的 placeholder 与原 prompt 相邻 token 重叠（如同 id 图像占位符相邻）时，多模态嵌入被放到偏移 1+ 个 token 的位置，视觉/音频特征绑定错位，理解质量静默下降且难以从输出归因。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/52924

### INF-0949 `hf_convert_to_rgb_alpha_as_foreground` transformers convert_to_rgb 误把 alpha 当前景

`故障` · `active` · `infra_off` · `config` · `verified` · `NEW` · 发现 2026-08-21 · 旧号 10.33

- **原因对象**：调度/服务 / transformers / convert_to_rgb 视频通道混合
- **原因行为**：video_utils.py 的 RGBA→RGB 混合行以 video[..., 3, :, :]（alpha 通道）作前景参与计算，正确应为 video[..., :3, :, :]（RGB 三通道）。
- **影响对象**：输入编码（静默错误）
- **影响详情**：输入 RGBA（4 通道）视频帧时，输出张量形状错误（channels-last 得 2,2,1,1 而非 2,3,1,1）且像素值错误，随后送入视觉编码器；无报错无告警，管线内无法发现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/huggingface/transformers/issues/48051

### INF-0950 `sglang_kimi_k3_dp_vision_shard_local_grid_index` SGLang Kimi-K3 DP 视觉预处理索引错位致崩溃

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-08-22 · 旧号 10.34

- **原因对象**：调度/服务 / sglang / Kimi-K3 deferred 视觉预处理索引
- **原因行为**：DP 分片视觉的 deferred GPU 预处理路径以 shard-local 位置索引按全局图片位置组织的 grid_thws_host，缺少 shard-local→全局位置的换算（回归自 #34404）。
- **影响对象**：可用性/进度（报错）
- **影响详情**：Kimi-K3 + cpu transport + DP-sharded vision 下，batch 含 ≥2 张不同分辨率图片即抛 ValueError（wrong grids）使服务崩溃；各图 grid 相同时缺陷不可见，原单测用相同 grid 样例因而长期漏检。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/pull/35305

### INF-0955 `vllm_deepseek_vl_ocr_bare_assert_input_corruption_python_o` vLLM deepseek_vl2/ocr 裸 assert 校验输入

`故障` · `active` · `infra_off` · `config` · `verified` · `NEW` · 发现 2026-08-26 · 旧号 10.39

- **原因对象**：调度/服务 / vllm / deepseek_vl2/ocr 处理器校验
- **原因行为**：deepseek_vl2/ocr 处理器用裸 assert 校验 image token 数等于图片数；这是 zip(text_splits, images) 对齐的唯一防线，且 AssertionError 不被识别为输入错误。
- **影响对象**：输入编码；可用性/进度（静默错误）
- **影响详情**：图像计数失配输入时：默认模式 AssertionError 落入 else 分支返回 HTTP 500（应为 400）；python -O 启动时 assert 被剥除，文本段被静默丢弃或复制、图文错位，模型照常生成。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/53850

### INF-0957 `vllm_whisper_chunk_decoder_early_eot_silent_audio_drop` vLLM Whisper chunk 内提前 EOT 后残余音频静默不转写

`故障` · `active` · `infra_off` · `partial` · `verified` · `NEW` · 发现 2026-08-27 · 旧号 10.41

- **原因对象**：调度/服务 / vllm / transcriptions 逐 chunk 解码
- **原因行为**：/v1/audio/transcriptions 按 30s chunk 每块只 decode 一次；decoder 在 chunk 中途发出 EOT 后，该 chunk 剩余音频既不重试也不续解码，直接丢弃。
- **影响对象**：输出内容（静默错误）
- **影响详情**：decoder 提前 EOT 时转写尾部语音缺失：29.42s 单 chunk 丢 7.9s，58.85s 双 chunk 共丢约 1/3；duration 报全长、text 看似完整、无 warning，只能靠 segment end 与音频时长的差值发现，确定性复现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/53144

### INF-0974 `vllm_routed_experts_prompt_start_boundary_assert_engine_core_fatal` vLLM routed_experts 起点边界值致 EngineCore 崩溃

`故障` · `active` · `infra_off` · `config` · `verified` · `var` · 发现 2026-09-07 · 旧号 10.58

- **原因对象**：调度/服务 / vllm / routed_experts 起点校验与断言
- **原因行为**：API 校验层放行 prompt_start 等于 prompt 长度，scheduler 却断言 prompt_start < num_prompt_tokens，合法域定义不一致；断言失败被当作 EngineCore fatal。
- **影响对象**：可用性/进度（报错）
- **影响详情**：--enable-return-routed-experts 下 prompt_start 等于 rendered prompt 长度（如 len 14、offset 14）时请求 HTTP 500，EngineCore 死亡，后续无关普通请求也持续 500；offset 13 正常。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/55750
- **关联**：var:INF-0585

### INF-0983 `vllm_truncate_prompt_tokens_applied_before_multimodal_expansion_final_prompt_overrun` vLLM 截断早于多模态占位符展开致 prompt 超限

`故障` · `active` · `ambiguous` · `config` · `verified` · `var` · 发现 2026-09-15 · 旧号 10.67

- **原因对象**：调度/服务 / vllm / prompt 截断与多模态展开次序
- **原因行为**：渲染管线在多模态占位符扩展之前执行 truncate_prompt_tokens，按扩展前 token 数计算限额，单 token 占位符扩展为数百音频 token 后不再校正。
- **影响对象**：输入编码（静默错误）
- **影响详情**：Qwen3-ASR 请求截断到 96 token 实际 prompt 为 147 token，请求被接受但超限；完整音频 span 本可放入 96 窗口。按上下文/KV 预算或计费的下游假设静默失准。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/57038
- **关联**：var:rope_offload_pretokenized_mask_length_drift

### INF-0984 `vllm_qwen3_asr_audio_preprocessing_token_count_divergence_hf_native` vLLM Qwen3-ASR 音频预处理与 HF 原生 token 数不一致

`故障` · `active` · `ambiguous` · `partial` · `verified` · `var` · 发现 2026-09-15 · 旧号 10.68

- **原因对象**：调度/服务 / vllm / Qwen3-ASR 音频特征提取与padding
- **原因行为**：硬编码 WhisperFeatureExtractor 而非 checkpoint 声明的 Qwen3ASRFeatureExtractor（缺最小长度 padding），并沿用 Omni 波形 padding 到 hop 倍数。
- **影响对象**：输入编码；输出内容（静默错误）
- **影响详情**：同一波形 vLLM 与 HF 原生 processor 音频 token 数不同：16001 样本 14 vs 13，321 样本 1 vs 7（丢 6 个）；短音频误差比例大，ASR 转写质量静默退化，与 HF 管线无法对拍。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/56981
- **关联**：var:trtllm_tokenizer_alias_table_drift

### INF-0986 `sglang_anthropic_messages_streaming_thinking_block_missing_signature_sdk_cold_concurrent_crash` SGLang Anthropic 流式 thinking 缺签名致 SDK 崩溃

`故障` · `active` · `infra_off` · `other_stack` · `verified` · `var` · 发现 2026-09-17 · 旧号 10.70

- **原因对象**：调度/服务 / sglang / /v1/messages thinking 签名
- **原因行为**：Anthropic 适配层对无原生签名的后端流式发出不带 signature 的 thinking block，captured_thinking_signature 被初始化却从未赋值，缺 fallback signature。
- **影响对象**：输出内容；可用性/进度（报错）
- **影响详情**：Anthropic SDK 冷进程并发流初始化时 content_block 保持 raw dict，accumulator 抛 AttributeError，整条流失败；新进程首批双请求同时到达触发，同进程重试持续失败、串行重放不复现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/39752
- **关联**：var:INF-1154

### INF-0987 `vllm_anthropic_deferred_tools_tool_addition_block_400_silent_client_retry` vLLM /v1/messages 拒收 tool_addition 致静默重试

`故障` · `active` · `infra_off` · `other_stack` · `verified` · `var` · 发现 2026-09-17 · 旧号 10.71

- **原因对象**：调度/服务 / vllm / Anthropic 内容块类型枚举
- **原因行为**：Anthropic 兼容 schema 的 content block 类型枚举未跟上 deferred tools 协议扩展，缺 tool_addition，Pydantic 校验拒绝含该块的请求。
- **影响对象**：性能/资源（性能退化）
- **影响详情**：Claude Code 开启 ENABLE_TOOL_SEARCH 时每会话首请求返回 400 + 约 20KB 错误体，客户端吞掉后带全量 schema 重发成功；每会话多一次往返，仅服务端日志可见（实测 29×200/3×400）。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/57324
- **关联**：var:sglang_kimi_k3_parser_channel_terminator_assumption_tool_call_silent_loss

### INF-0992 `vllm_multimodal_placeholder_token_value_binding_collision_grounding_mismatch` vLLM 多模态占位符按 token 值绑定，用户文本可劫持图像槽

`故障` · `active` · `infra_off` · `yes` · `verified` · `NEW` · 发现 2026-09-20 · 旧号 10.76

- **原因对象**：调度/服务 / vllm / PromptReplacement 占位符绑定
- **原因行为**：PromptReplacement 在渲染分词后的平坦 token 序列上按 token 值匹配占位符，不保留来源；chat_utils 不为模板媒体占位符做预留，用户可拼写的特殊 token 先被当作媒体槽消费。
- **影响对象**：输入编码；输出内容（静默错误）
- **影响详情**：用户正文含字面 <|image_pad|> 时图像特征 span 被插入用户正文、真槽只剩 1 个未展开 pad，总长与 token 多重集不变；Qwen2.5-VL 视觉安全裁决由 UNSAFE 翻转为 SAFE（BLOCK→ALLOW）。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/57740

### INF-0993 `vllm_responses_api_parallel_builtin_tool_calls_only_last_dispatched` vLLM Responses API 并行内建工具调用只执行最后一个

`故障` · `active` · `infra_off` · `config` · `verified` · `NEW` · 发现 2026-09-20 · 旧号 10.77

- **原因对象**：调度/服务 / vllm / Responses API 内建工具分发循环
- **原因行为**：非 Harmony Responses API 路径的 built-in 工具执行循环只处理消息序列中最后一个 function call 项，未遍历全部调用。
- **影响对象**：输出内容（静默错误）
- **影响详情**：模型一轮 emit 多个 built-in tool call（如 web_search_preview + code_interpreter）时只有最后一个被执行，其余静默漏执行，与默认开启的 parallel_tool_calls 语义矛盾；CPU-only pytest 可复现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/57791

### INF-0997 `vllm_completions_derender_drops_requested_prompt_logprobs` vLLM derender 端点丢弃请求的 prompt_logprobs

`故障` · `active` · `infra_off` · `config` · `verified` · `var` · 发现 2026-09-21 · 旧号 10.81

- **原因对象**：调度/服务 / vllm / OnlineDerenderer 响应组装
- **原因行为**：derender 步从 GenerateResponse 重组 OpenAI 形状响应时未映射 prompt_logprobs 字段，该字段在管线前三级正确存活、在最后一环形状转换中被丢掉。
- **影响对象**：输出内容；输出分布（静默错误）
- **影响详情**：scale-out 管线经 /v1/completions/derender 请求 prompt_logprobs 时，HTTP 200、生成文本与 usage 正常，但 choices[0].prompt_logprobs 为 null；客户端不显式断言非空则无感知。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/57929
- **关联**：var:INF-1074

### INF-1010 `vllm_request_level_chat_template_kwargs_silently_drops_all_images` vLLM 请求级 chat_template_kwargs 致图像全被丢弃

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-10-09 · 旧号 10.94

- **原因对象**：调度/服务 / vllm / chat_template_kwargs 合并路径
- **原因行为**：请求级 chat_template_kwargs 的合并路径绕过或破坏多模态输入绑定，即使取值与 server 默认值完全相同也会触发。
- **影响对象**：输入编码；输出内容（静默错误）
- **影响详情**：vLLM 0.30.0 /v1/chat/completions 带任意请求级 chat_template_kwargs 时，prompt_tokens 约 20（正常约 276），渲染 prompt 只含文本，模型答没有图像，200 无告警；去掉 kwargs 即正常。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/59876

### INF-1023 `sglang_images_edits_mask_upload_accepted_never_read_inpainting_ignored` SGLang images/edits 接收 mask 但从不读取

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-10-12 · 旧号 10.107

- **原因对象**：调度/服务 / sglang / images/edits mask 处理
- **原因行为**：image_api.py 声明并接收 mask 上传，但 handler 从不读取该字段，API 契约声明与实现脱节。
- **影响对象**：输出内容（静默错误）
- **影响详情**：发送 mask 的 inpainting 请求返回 200，却按全图无掩码编辑执行，调用方无法感知 mask 被忽略；SGLang 自家 ComfyUI 集成（server_api.py 附 mask）即会触发。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/43409

### INF-1024 `sglang_comfyui_cond_key_shape_scalar_collision_swaps_t5_context` SGLang ComfyUI 条件缓存键漏 T5 上下文致碰撞

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-10-12 · 旧号 10.108

- **原因对象**：调度/服务 / sglang / ComfyUI 插件 _cond_key 缓存键
- **原因行为**：_cond_key 只用 shape 加 prompt_embeds[0] 首尾标量作键；Flux 下该张量仅是 pooled CLIP-L，T5 上下文与 image_latent 均不入键，键窄于载荷身份。
- **影响对象**：跨请求/跨样本数据；输出内容（静默错误）
- **影响详情**：Flux 下共享前约 75 个 CLIP token 的两个 prompt（或同 clip_l 不同 t5xxl 文本）键碰撞，第二个 prompt 静默用第一个的 T5 上下文生成；Qwen-Image-Edit 下同文不同参考图同样碰撞。无 crash。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/43107

### INF-1051 `vllm_prompt_logprobs_zero_truthiness_bypass_guard_engine_crash` vLLM prompt_logprobs=0 绕过校验杀死 EngineCore

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-08-11 · 旧号 11.27

- **原因对象**：调度/服务 / vllm / logprob 参数 truthiness 校验
- **原因行为**：多处用 truthiness 而非 is not None 检查 logprob 参数，合法值 prompt_logprobs=0 绕过 kv-sharing-fast-prefill 不兼容 guard 与 stream 拒绝。
- **影响对象**：可用性/进度（报错）
- **影响详情**：prompt_logprobs=0 等 falsy 值配合 echo/stream/kv-sharing 配置时，GPUModelRunner 下游 is not None 检查命中 bare assert，整个 EngineCore 进程退出，所有在途与后续 client 均受影响。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/51846

### INF-1079 `sglang_prefill_logprob_cursor_skip_misalignment_cross_request` SGLang skip 分支不推进 logprob 游标致跨请求错位

`故障` · `active` · `infra_off` · `other_stack` · `verified` · `NEW` · 发现 2026-08-29 · 旧号 11.56

- **原因对象**：调度/服务 / sglang / prefill 结果处理 logprob 游标
- **原因行为**：skip 分支对 retracted/finished 请求 continue 时跳过 logprob_pt 推进，生产者仍为其计入切片；非 disaggregated 路径漏了 advance_logprob_pt()
- **影响对象**：跨请求/跨样本数据；输出分布（静默错误）
- **影响详情**：请求 input logprobs 且同 batch 有 retracted/finished 请求时，后续请求拿到别人的 prompt logprobs；数值合法，断言只查长度、移位不越界，无异常；默认 overlap scheduling 下 KV 压力即触发
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/36938

### INF-1144 `vllm_unload_lora_adapter_frontend_only_engine_state_leak` vLLM unload_lora_adapter 只删前端名单、引擎未卸载

`故障` · `active` · `infra_off` · `config` · `verified` · `var` · 发现 2026-09-01 · 旧号 12.28

- **原因对象**：调度/服务 / vllm / unload_lora_adapter 接口
- **原因行为**：unload_lora_adapter 只删前端 lora_requests 条目并记“Removed LoRA adapter”日志，从不调用 AsyncLLM.remove_lora；Rust 前端同样只删名称。
- **影响对象**：性能/资源；输出内容（静默错误）
- **影响详情**：调用 /v1/unload_lora_adapter 返回成功，adapter 仍驻留 worker GPU slot/CPU cache 直到 LRU 逐出，显存不释放；“已卸载”的 adapter 可能仍在服务，同名 reload 时新旧权重竞争。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/54839
- **关联**：var:vllm_sleep_wake_lora_adapter_state_corruption

### INF-1155 `sglang_glm53_flash_pinned_transformers_missing_glm5_next_processor_degrades_tokenizer` SGLang GLM-5.3 图片输入静默失效（processor 降级）

`故障` · `active` · `infra_off` · `other_stack` · `verified` · `var` · 发现 2026-09-16 · 旧号 12.39

- **原因对象**：调度/服务 / sglang / 多模态 processor 加载与能力探测
- **原因行为**：pin 的 transformers 5.12.1 缺 glm5_next 模块，AutoProcessor 静默回退为裸 tokenizer；SGLang 以其充当多模态 processor，能力探测只看 config。
- **影响对象**：输入编码；输出内容（静默错误）
- **影响详情**：/model_info 报支持图像理解，但每张图被缩为 3 个 token（placeholder 不展开、无 pixel_values），模型凭空作答（纯红图答“穿亮黄夹克的人”）；文本路径正常，无报错无警告。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/39831
- **关联**：var:INF-0549

### INF-1167 `vllm_qwen3_asr_realtime_session_update_silently_drops_prompt_and_language` vLLM realtime ASR 丢弃会话 prompt/language

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-10-12 · 旧号 12.51

- **原因对象**：调度/服务 / vllm / SessionUpdate 协议模型
- **原因行为**：/v1/realtime 的 SessionUpdate 只有 model 字段，prompt 与 language 被丢弃；realtime 路径硬编码 chat 模板、不走 get_generation_prompt()。
- **影响对象**：输入编码；输出内容（质量退化）
- **影响详情**：Qwen3-ASR 实时会话经 session.update 设的上下文 prompt 与语言强制均无效，客户端无报错，而 REST /v1/audio/transcriptions 同参数生效；LiveKit 语音 agent 的 per-call 领域词/姓名上下文全部丢失，识别质量退化。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/61026

### INF-1168 `sglang_update_weights_from_disk_params_declared_never_consumed` SGLang update_weights_from_disk 参数未实现

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-10-12 · 旧号 12.52

- **原因对象**：调度/服务 / sglang / update_weights_from_disk
- **原因行为**：is_async、keep_pause、token_step 请求字段与 num_paused_requests 响应字段已声明并写入文档，但全树没有任何读取方；响应字段由生产者硬编码为 0 并透传到 HTTP/gRPC。
- **影响对象**：输出内容；指标/日志（静默错误）
- **影响详情**：调用方收到恒为 0 的暂停请求数，DP>1 时返回 per-rank 零列表而文档类型为 int；RL 训练侧依赖的异步语义、暂停行为与记账全部静默落空（如误判更新已完成、记账恒零），无任何报错。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/42544

### INF-1192 `hf_asr_pipeline_stereo_channels_last_mean_axis0_destroys_audio` ASR pipeline 对立体声沿时间轴求均值，静默销毁音频

`故障` · `active` · `infra_off` · `yes` · `documented` · `NEW` · 发现 2026-08-11 · 旧号 13.24

- **原因对象**：调度/服务 / transformers / ASR pipeline 立体声转单声道
- **原因行为**：AutomaticSpeechRecognitionPipeline 对 soundfile 返回的 (samples, channels) channels-last 立体声波形做 mean(axis=0)，沿时间轴而非通道轴平均。
- **影响对象**：输入编码；输出内容（静默错误）
- **影响详情**：以 soundfile 读入的立体声语音为输入时，3 秒音频被压缩为极短片段，转写结果为 ' you'；无错误无警告，仅有误导性的“converted to mono”日志。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/huggingface/transformers/issues/47886

### INF-1254 `pd_router_first_message_prefix_misroute` SGLang PD router 以首条消息做前缀匹配致误路由重算

`故障` · `active` · `model_on` · `no` · `verified` · `NEW` · 发现 待定 · 旧号 K4.7

- **原因对象**：调度/服务 / sglang / PD router 缓存感知前缀提取
- **原因行为**：cache-aware PD router 取 body.messages.first()（通常即 system prompt）做 radix 前缀匹配与 worker 选择，路由被共享 system prompt 主导，而非可复用的前缀。
- **影响对象**：性能/资源（性能退化）
- **影响详情**：输出完全正确，但多轮请求被路由到不持有其 KV 的 worker 而被迫完整重算，重算 token 数与延迟升高；修复后前缀缓存命中率由 69% 升至 96%。需 P/D 分离部署。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/26263；https://github.com/sgl-project/sglang/pull/27430

## 输出解析

### INF-0574 `vllm_qwen3_xml_tool_choice_required_silently_acts_as_auto` vLLM qwen3 parser 强制 tool_choice 被静默降级

`故障` · `active` · `infra_off` · `config` · `documented` · `var` · 发现 2026-09-01 · 旧号 6.181

- **原因对象**：输出解析 / vllm / qwen3 parser tool_choice
- **原因行为**：qwen3_coder/qwen3_xml parser 未实现 tool_choice 强制：required/具名函数本应触发 guided decoding 限定 tool-call 输出，实际未施加约束。
- **影响对象**：输出内容（静默错误）
- **影响详情**：tool_choice 为 required 或具名函数时请求返回 HTTP 200，但行为如同 auto，模型可不发 tool call；strict: true 与默认开启的 VLLM_ENFORCE_STRICT_TOOL_CALLING 均无效。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/54808
- **关联**：var:INF-0956

### INF-0927 `vllm_kimi_k3_close_marker_stream_leak` vLLM Kimi K3 流式路径把 turn 关闭标记泄漏进 content

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-08-08 · 旧号 10.11

- **原因对象**：输出解析 / vllm / Kimi K3 tool parser 流式路径
- **原因行为**：流式 _extract_response_content 只在 tools open/response close 处停止 content，未像非流式路径那样剥离 <|close|>message<|sep|> 关闭标记。
- **影响对象**：输出内容（静默错误）
- **影响详情**：Kimi K3 流式且回合无 response close 时触发；客户端收到 Hello<|close|>message<|sep|> 而非 Hello，原始 XTL 标记进入用户可见 content，导致客户端解析错误；非流式正常，无 crash。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/51400

### INF-0930 `sglang_dsv4_reasoning_toolcall_streaming_parser_six_bugs` SGLang DSV4 流式解析丢 tool call、泄漏特殊 token

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-08-09 · 旧号 10.14

- **原因对象**：输出解析 / sglang / DSV4 reasoning/tool 解析器
- **原因行为**：DSV4 detector 缺 tool_start_token，DSML 块滞留 reasoning_content；基类在 chunk 边界无条件清空 buffer、销毁部分 token；边界切分与多 token 批处理另有解析错误。
- **影响对象**：输出内容（静默错误）
- **影响详情**：DSV4 流式 reasoning + tool call 时触发；无 crash，tool call 被静默丢弃（功能失效），或 <｜DSML｜tool_calls> 等特殊 token 泄漏进 API content 输出。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/pull/34178

### INF-0932 `trtllm_harmony_silent_discard_malformed_toolcall_empty_200` Harmony adapter 静默丢弃格式错误的 tool call

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-08-10 · 旧号 10.16

- **原因对象**：输出解析 / tensorrt-llm / HarmonyAdapter 不完整消息剥离
- **原因行为**：_strip_incomplete_messages 在最后一个 <|start|> 后没有 <|message|> 时删除其后全部内容，格式错误（如缺 <|message|>）的 tool-call 消息被整条删除且不记任何日志。
- **影响对象**：输出内容（静默错误）
- **影响详情**：trtllm-serve + Harmony adapter（如 gpt-oss-120b + Eagle3）遇格式错误 tool-call 时触发；返回空 tool_calls、空 content、HTTP 200、finish_reason=stop，无日志，调用方无法与模型拒绝调用工具区分。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/NVIDIA/TensorRT-LLM/pull/17464

### INF-0936 `vllm_muse_glimmer_parser_drops_completed_reasoning_blocks_on_truncation` vLLM Muse Glimmer 推理解析器截断时丢弃已完成块

`故障` · `active` · `infra_off` · `yes` · `documented` · `NEW` · 发现 2026-08-15 · 旧号 10.20

- **原因对象**：输出解析 / vllm / Muse Glimmer 推理解析器
- **原因行为**：extract_reasoning 在 collapse 串（已吃掉早先块的 <|eom|>）上做闭合块匹配，却在未 collapse 原文上做截断 fallback；末块未终止时闭合匹配为空，fallback 只取最后开放块。
- **影响对象**：输出内容（静默错误）
- **影响详情**：多 reasoning 块且末块因 max_tokens/abort 未终止时触发；如 A/B/C 三块在 C 内截断，客户端只收到 C，A、B 整块静默消失；流式与非流式返回也不一致。无 crash。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/52465

### INF-0938 `trtllm_deepseekr1_parser_finish_discards_unawaited_tag_buffer` TRT-LLM DeepSeekR1Parser 流结束时丢弃尾部推理内容

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-08-16 · 旧号 10.22

- **原因对象**：输出解析 / tensorrt-llm / DeepSeekR1Parser.finish
- **原因行为**：finish() 在 buffer 恰为 <think> 或 </think> 时将其当作标记丢弃，未区分是否为 parser 正在等待的标签，buffer 内残留的推理文本随该判断被一并丢弃。
- **影响对象**：输出内容（静默错误）
- **影响详情**：DeepSeek-R1 reasoning parser 下生成在推理块中途结束（max_tokens/abort/EOS）时触发；无 crash，尾部推理内容整块静默丢失。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/NVIDIA/TensorRT-LLM/pull/17774

### INF-0939 `trtllm_qwen_toolcall_eot_prefix_separator_missplit` TRT-LLM Qwen3 eot 与分隔符前缀重叠致流式内容被吞

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-08-16 · 旧号 10.23

- **原因对象**：输出解析 / tensorrt-llm / 基础流式 tool parser
- **原因行为**：Qwen3 的 eot_token 与 tool_call_separator（换行）共享前缀，调用完成后保留的 eot 被误判为新工具调用的开始，parser 滞留在 JSON/tool 分支并 buffer 住后续内容。
- **影响对象**：输出内容（静默错误）
- **影响详情**：TRT-LLM 流式 serving 中 Qwen3 完成工具调用后继续生成文本时触发；无 crash，客户端只收到 tool call，其后的助手文本滞留 buffer 不下发。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/NVIDIA/TensorRT-LLM/pull/17748

### INF-0940 `vllm_minimax_m3_mxfp8_streaming_tool_call_early_stop` vLLM MiniMax-M3 流式 tool call 被提前终止

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-08-17 · 旧号 10.24

- **原因对象**：输出解析 / vllm / minimax_m3 流式解析状态机
- **原因行为**：streaming 路径对 reasoning→tool-call 转换处理有误：reasoning/tool 双 parser 的流式状态机在边界处提前判停。
- **影响对象**：输出内容（静默错误）
- **影响详情**：MiniMax-M3 MXFP8 + tools + stream:true 且推理较长后决定调用工具时触发；reasoning 通道关闭后 SSE 立即以空 delta + finish_reason=stop 结束，tool_calls 为空；同 payload 非流式正常，无报错。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/52613

### INF-0942 `vllm_gemma4_parser_template_thinking_default_mismatch` Gemma4 parser 与模板 thinking 默认值矛盾

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-08-17 · 旧号 10.26

- **原因对象**：输出解析 / vllm / Gemma4 parser 默认值
- **原因行为**：Gemma4 parser 在未传 enable_thinking 时默认 True，而 bundled chat template 默认 false（文档亦称默认关），同一 flag 在两层默认值矛盾。
- **影响对象**：输出内容（静默错误）
- **影响详情**：请求省略 enable_thinking 时触发：模板产出非 thinking prompt，parser 却按 thinking 解析，reasoning_content/content 分桶错位或 reasoning 为空；无 crash，parser-only 即可复现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/52410

### INF-0943 `sglang_reasoning_parser_pretoken_routing_nway_divergence` SGLang reasoning 标记判定与剥离条件不对称

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-08-17 · 旧号 10.27

- **原因对象**：输出解析 / sglang / reasoning 检测器基类
- **原因行为**：检测器判定用 think_start_token in text（任意位置），剥离却用 startswith（仅开头），标记前内容落入缝隙；流式仅在整个 buffer 为标记前缀时识别部分标记。
- **影响对象**：输出内容（不可复现）
- **影响详情**：任意 reasoning parser 下输出中起始标记前有内容（哪怕一个空格）时触发；同一字符串在 main 上有四种解析结果，reasoning/content 分桶随 chunk 边界变化，流式与 one-shot 不一致。无 crash。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/35083

### INF-0945 `trtllm_deepseekr1_parser_enable_thinking_request_ignored` TRT-LLM R1 parser 忽略 thinking 开关致正文为空

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `var` · 发现 2026-08-18 · 旧号 10.29

- **原因对象**：输出解析 / tensorrt-llm / DeepSeekR1Parser 推理切分
- **原因行为**：DeepSeekR1Parser 以类级常量 reasoning_at_start=True 注册，不读请求级 enable_thinking；缺 </think> 时仍按推理开头切分（子类 NemotronV3 已修，父类未回移）。
- **影响对象**：输出内容（静默错误）
- **影响详情**：enable_thinking=false 时模板不注入 <think>、模型不产 </think>，整个回复进 reasoning_content、content 为空；读 message.content 的客户端对成功请求拿到空，同服务要/不要 thinking 的客户端无法共存。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/NVIDIA/TensorRT-LLM/issues/17916
- **关联**：var:INF-0942

### INF-0946 `trtllm_tool_parser_zero_extraction_200_silent` TRT-LLM tool parser 零提取时静默返回空 tool_calls

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-08-18 · 旧号 10.30

- **原因对象**：输出解析 / tensorrt-llm / TRT-LLM tool parser 提取逻辑
- **原因行为**：文本匹配到 tool call marker 但参数解析/提取产出零个调用时，parser 把它当作“无 tool call”正常返回，既不报错也不透传原文。
- **影响对象**：输出内容（静默错误）
- **影响详情**：模型输出格式错误的 tool call 时，请求 200 成功但 tool_calls 为空、原文消失；agent 层无法区分“模型没想调用”与“解析器丢了调用”。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/NVIDIA/TensorRT-LLM/issues/17917

### INF-0947 `sglang_toolcall_parsers_streaming_vs_detect_mismatch` SGLang 8 个 tool parser 流式与一次性解析不一致

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-08-19 · 旧号 10.31

- **原因对象**：输出解析 / sglang / 8 个 tool parser 流式增量解析
- **原因行为**：8 个 parser（minimax-m2、mistral、gemma4 等）的 parse_streaming_increment 状态机遇空参数 {} 等极简参数时走空分支处理错误，与 detect_and_parse 结果不一致。
- **影响对象**：输出内容（静默错误）
- **影响详情**：stream=true 且 tool call 含空/极简参数时，流式 200 成功但调用静默丢失或残缺：minimax-m2 整个调用消失、mistral 丢第 2 个调用、cohere_command4/gemma4 被截断；同一文本流式与非流式结果矛盾。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/35564

### INF-0951 `vllm_gemma4_tool_parser_bare_opener_fsm_dead_state` vLLM Gemma 4 tool parser FSM 死态吞掉调用

`故障` · `active` · `infra_off` · `config` · `verified` · `NEW` · 发现 2026-08-23 · 旧号 10.35

- **原因对象**：输出解析 / vllm / gemma4 tool parser FSM
- **原因行为**：FSM 只定义 TOOL_PREAMBLE→TOOL_NAME 的 "call:" 前缀转移，未覆盖模型实际发出的 bare 形式 <|tool_call>:name；TOOL_PREAMBLE 态无 content 出口，成为死态。
- **影响对象**：输出内容（静默错误）
- **影响详情**：Gemma 4 发出省略 call 的 bare tool call 时，FSM 滞留 TOOL_PREAMBLE 直到块结束，整个 tool call 被丢弃：既无提取也无 content，无错误日志，客户端无从得知调用被吞。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/53444

### INF-0954 `vllm_dsv4_flash_tool_choice_auto_deep_effort_empty_drop` vLLM DSV4-Flash 深推理档+auto 工具静默空响应

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-08-26 · 旧号 10.38

- **原因对象**：输出解析 / vllm / deepseek_v4 推理/工具解析边界
- **原因行为**：深推理档拉长推理段后，推理→动作的解码边界处理与 tool-calling 语法未对齐：模型在推理中选定工具后流即终止，未产出可提取的 tool call 或 content；具体代码位置原文未定位。
- **影响对象**：输出内容（静默错误）
- **影响详情**：deepseek_v4 推理/工具 parser + tool_choice=auto + reasoning_effort xhigh/max 时高概率返回 finish_reason=stop、空 content、无 tool_calls；两条件缺一则不高频，live 流量可复现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/53831

### INF-0958 `vllm_kimi_k3_parser_channel_terminator_assumption_tool_call_silent_drop` vLLM Kimi-K3 parser 终止符假设不全丢 tool call

`故障` · `active` · `infra_off` · `partial` · `verified` · `NEW` · 发现 2026-08-28 · 旧号 10.42

- **原因对象**：输出解析 / vllm / kimi_k3 reasoning/响应解析
- **原因行为**：is_reasoning_end() 只认 think 开/闭标记对，模型不发 close-think 直接开 tools 通道时恒假，tool parser 收不到 delta；流式路径未处理 close-message 终止符。
- **影响对象**：输出内容（静默错误）
- **影响详情**：Kimi-K3 跳过 close-think 的轮次 tool call 静默蒸发（finish_reason=stop、零输出 token）或裸 XTML 漏进可见内容；以 close-message 结尾的回答流式把终止符漏给客户端而非流式不漏；agentic 流量间歇触发。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/54273

### INF-0960 `sglang_tool_call_array_nonobject_entry_discards_valid_neighbors` SGLang tool 数组含非 object 项时整批丢弃

`故障` · `active` · `infra_off` · `other_stack` · `verified` · `var` · 发现 2026-09-01 · 旧号 10.44

- **原因对象**：输出解析 / sglang / parse_base_json
- **原因行为**：parse_base_json 对 action 数组逐项调用 act.get("name") 而不做 isinstance 检查，非 object 条目抛 AttributeError，上层异常处理丢弃整批结果而非只跳过畸形项。
- **影响对象**：输出内容（静默错误）
- **影响详情**：模型输出的 tool-call 数组含 "junk"、[1,2]、null 等非 object 条目时，旁边格式良好的调用一并消失（零 tool call），异常被吞、对话照常继续，agent 表现为该调用时没调用。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/37283
- **关联**：var:INF-0958

### INF-0962 `vllm_tool_choice_none_delegates_tool_shaped_content_silent_drop` vLLM tool_choice=none 仍删除工具形状 content

`故障` · `active` · `infra_off` · `config` · `verified` · `var` · 发现 2026-09-03 · 旧号 10.46

- **原因对象**：输出解析 / vllm / DelegatingParser 工具提取分支
- **原因行为**：enable_auto_tools 下 engine-based streaming parser 分支不检查 tool_choice 语义，tool_choice=none 时仍走消费检测文本的路径，返回解析剩余物而非模型原文。
- **影响对象**：输出内容（静默错误）
- **影响详情**：开启 auto tool choice、用 glm45/glm47/qwen3_coder 等 engine-based parser 且 tool_choice=none 时，输出中 <tool_call>…</tool_call> 形状的文本被删、content 为 None，HTTP 200。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/55080
- **关联**：var:INF-0958

### INF-0965 `vllm_qwen3_parser_malformed_parameter_element_streamed_json_poison` vLLM Qwen3 XML parser 参数正则过宽致流式 JSON 中毒

`故障` · `active` · `infra_off` · `config` · `documented` · `var` · 发现 2026-09-05 · 旧号 10.49

- **原因对象**：输出解析 / vllm / qwen3 XML parser 参数正则
- **原因行为**：参数名正则 [^>]+ 过宽，畸形 <parameter>（缺名、缺 >、多余 =）使 partial converter 把直到闭合标签的整段当 key 流出，而 final 转换不产生该 key。
- **影响对象**：输出内容；可用性/进度（静默错误）
- **影响详情**：流式下模型输出近错 XML tool call 时，流式前缀不变量被破坏、_flush_arg_converter 静默返回 None，客户端持有未终结 JSON；逐字回传 tool call 的客户端（Responses API/Codex CLI）此后该会话每个请求被 400 拒绝。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/55497
- **关联**：var:INF-0958

### INF-0971 `sglang_dsv4_dsml_incremental_parser_partial_tool_transaction_escape` SGLang DeepSeek V4 增量 parser 截断时泄出部分工具调用

`故障` · `active` · `infra_off` · `other_stack` · `verified` · `var` · 发现 2026-09-06 · 旧号 10.55

- **原因对象**：输出解析 / sglang / DeepSeek V4 DSML 增量解析器
- **原因行为**：增量 parser 在 invoke 开标签出现即发布函数名、DSML 信封闭合前发布临时参数，发布粒度与事务完整性不匹配；流末不丢弃不完整/畸形 DSML。
- **影响对象**：输出内容（静默错误）
- **影响详情**：DeepSeek V4 function calling 生成被截断（max_tokens 边界）时，客户端收到永不完整的部分 tool call；非流式异常路径把畸形 DSML 当普通 assistant content 返回。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/pull/38233
- **关联**：var:INF-0958

### INF-0973 `vllm_qwen3_xml_parser_whitespace_separator_before_first_tool_call_content_delta` vLLM qwen3_xml parser 把空白分隔符发成 content

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `var` · 发现 2026-09-07 · 旧号 10.57

- **原因对象**：输出解析 / vllm / qwen3_xml 流式 tool parser
- **原因行为**：增量 parser 的通道状态机把非标签文本一律归 content 通道，thinking 结束标签与首个 tool call 开标签之间的空白分隔符落在过渡间隙被误发布。
- **影响对象**：输出内容（静默错误）
- **影响详情**：Qwen3 流式输出 thinking 块 + 纯空白分隔符 + 首个 XML tool call 时，客户端收到只含 "\n\n" 的幽灵 content 事件，污染以 content 事件驱动 UI/落库的下游（见于 vllm-ascend 镜像）。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/55633
- **关联**：var:INF-0958

### INF-0976 `vllm_tool_parser_nonstreaming_truncation_drops_post_tool_call_text` vLLM 非流式 tool parser 丢弃工具调用之后的文本

`故障` · `active` · `infra_off` · `yes` · `verified` · `var` · 发现 2026-09-10 · 旧号 10.60

- **原因对象**：输出解析 / vllm / hermes/deepseekv3 非流式解析
- **原因行为**：非流式 extract_tool_calls 取 tool-call 起始 token 之前的文本作 content 后不再恢复，丢弃其后文本；流式实现会发出 post-tool 文本，两路径语义未对齐。
- **影响对象**：输出内容（静默错误）
- **影响详情**：模型在 tool call 后继续生成文本时，非流式请求（含 /v1/messages 非流式转换）静默丢失该文本（如丢 It is 30 degrees.），流式请求正常返回，同一 generation 两种交付模式内容不一致。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/56263
- **关联**：var:INF-0958

### INF-0978 `vllm_glm53flash_reasoning_exit_tool_call_token_bleed_word_salad` vLLM GLM-5.3 退出推理段时 tool token 泄漏致乱码

`故障` · `active` · `infra_off` · `yes` · `verified` · `NEW` · 发现 2026-09-13 · 旧号 10.62

- **原因对象**：输出解析 / vllm / reasoning parser 退出边界处理
- **原因行为**：reasoning 解析器跳出思维链段时未检查 token 是否为 tool call 起始，tool-call 控制串泄漏进正文解码上下文，污染后续自回归生成。
- **影响对象**：输出内容（静默错误）
- **影响详情**：GLM-5.3-Flash 多轮 agentic 长会话中输出间歇性坍缩为重复 token/shell 片段 word salad，HTTP 200、finished_reason 正常；单轮/短请求干净，合成请求无法复现，仅真实 agentic 客户端触发，build A/B 回滚可消除。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/56605

### INF-0981 `vllm_responses_api_include_reasoning_flag_ignored` vLLM /responses 端点不过滤 reasoning 内容

`故障` · `active` · `infra_off` · `yes` · `verified` · `var` · 发现 2026-09-13 · 旧号 10.65

- **原因对象**：输出解析 / vllm / /responses 响应组装
- **原因行为**：/responses 端点的响应组装未接入 include_reasoning 过滤（reasoning.effort=none 亦无效），该过滤只在 /chat/completions 与 /completions 实现。
- **影响对象**：输出内容（静默错误）
- **影响详情**：对无关闭 thinking 开关的模型（GLM-5.3+）设置 include_reasoning=false 或 effort=none 时，/responses 返回的 text content 仍混入 reasoning 内容；其它端点同参数行为正确。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/56428
- **关联**：var:INF-0968

### INF-0985 `sglang_qwen3_coder_detector_duplicated_parameter_tag_regex_truncates_overwrites` SGLang Qwen3Coder 参数正则遇重复标签截断并覆写

`故障` · `active` · `infra_off` · `other_stack` · `verified` · `var` · 发现 2026-09-16 · 旧号 10.69

- **原因对象**：输出解析 / sglang / Qwen3CoderDetector 参数正则
- **原因行为**：参数提取正则含 (?=<parameter=) lookahead，值内出现同名字面 <parameter= 时被切成新匹配，findall 返回的后一个匹配覆写 parsed_params 中的前值。
- **影响对象**：输出内容（静默错误）
- **影响详情**：模型在数组/对象参数值闭合前重复输出 <parameter= 标签时，tool call 参数被截断并静默覆写早先正确值；生产 4/3678 请求（0.11%）全在数组值参数（edits、changes、todos 等），流式路径同形。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/39836
- **关联**：var:sglang_kimi_k3_parser_channel_terminator_assumption_tool_call_silent_loss

### INF-0988 `vllm_internlm_tool_parser_opening_marker_split_across_stream_chunks_drops_call` vLLM internlm parser 开标记跨 chunk 时丢弃工具调用

`故障` · `active` · `infra_off` · `partial` · `documented` · `var` · 发现 2026-09-18 · 旧号 10.72

- **原因对象**：输出解析 / vllm / internlm 流式 tool parser
- **原因行为**：流式 parser 的标记匹配只看当前增量缓冲，不保留跨增量边界的部分前缀，开标记跨 chunk 分裂时状态机停在未进入 tool call 态。
- **影响对象**：输出内容（静默错误）
- **影响详情**：流式解码中 <tool_call> 等开标记恰被 chunk 边界劈开时，整个 tool call 被静默丢弃，无错误无部分结果；非流式整段解析正常，是否触发取决于服务端增量切分。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/57490
- **关联**：var:vllm_kimi_k3_parser_channel_terminator_assumption_tool_call_silent_loss

### INF-0991 `vllm_tool_arg_converters_unclosed_last_parameter_silent_drop_four_family` vLLM 四族 tool arg converter 静默丢弃未闭合末参数

`故障` · `active` · `infra_off` · `partial` · `verified` · `var` · 发现 2026-09-20 · 旧号 10.75

- **原因对象**：输出解析 / vllm / qwen3_coder 等四族参数转换器
- **原因行为**：尾部参数值提取只在 if partial: 分支执行，终态 partial=False 转换只认闭合标签；该结构复制在 qwen3_coder 等四个 converter 中，#57707 只修了 qwen3。
- **影响对象**：输出内容（静默错误）
- **影响详情**：模型省略末参数闭合标签直接闭函数时，非流式 arguments 少最后一个键（如丢 unit），流式 flush 产出比已流出更短的 JSON，客户端得到无效 JSON；BFCL live_simple 温度 0 可触发。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/57699；https://github.com/vllm-project/vllm/issues/57827
- **关联**：var:vllm_kimi_k3_parser_channel_terminator_assumption_tool_call_silent_loss

### INF-0995 `vllm_functiongemma_hyphenated_tool_name_streaming_nonsplit_double_decode_divergence` vLLM functiongemma 连字符工具名非流式解析失败并泄漏控制帧

`故障` · `active` · `infra_off` · `config` · `verified` · `NEW` · 发现 2026-09-21 · 旧号 10.79

- **原因对象**：输出解析 / vllm / functiongemma 非流式解析
- **原因行为**：流式与非流式 parse 对工具名的假设不同：非流式整段 parser 的名字白名单/正则拒绝含连字符的函数名，拒绝后不报错，把未消费的控制帧放入 content。
- **影响对象**：输出内容（静默错误）
- **影响详情**：声明函数名含连字符（get-weather）时非流式请求返回无 tool call，并把完整 <start_function_call>…<end_function_call> 控制帧作为 assistant content 暴露；流式正常，改名 get_weather 两模式都正常。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/57950

### INF-0996 `vllm_batch_completions_include_reasoning_false_logprobs_token_ids_leak_hidden_chain` vLLM batch 端点 token 级字段泄漏隐藏思维链

`故障` · `active` · `infra_off` · `config` · `verified` · `var` · 发现 2026-09-21 · 旧号 10.80

- **原因对象**：输出解析 / vllm / batch 响应组装 reasoning 剥离
- **原因行为**：include_reasoning=false 的剥离只作用于 message 结构化字段，logprobs 的 token 字符串与 choice.token_ids 等 token 级通道仍携带未过滤的原始生成序列。
- **影响对象**：输出内容（静默错误）
- **影响详情**：batch 端点同时设 include_reasoning=false 与 logprobs=true 或 return_token_ids=true 时，HTTP 200、message 体干净，但客户端可从 token 级字段完整恢复隐藏的 <think> 推理内容。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/57935
- **关联**：var:INF-0981

### INF-0998 `vllm_qwen_tool_call_ordinary_token_delimiter_spelling_bypasses_parser_streaming` vLLM Qwen parser 漏识普通 token 拼写的分隔符

`故障` · `active` · `infra_off` · `config` · `verified` · `NEW` · 发现 2026-09-22 · 旧号 10.82

- **原因对象**：输出解析 / vllm / Qwen parser 分隔符识别
- **原因行为**：文本 grammar 允许用普通词表 token 序列拼出 </think>、<tool_call>、</tool_call>，而 Qwen parser 只按专属 token ID 识别边界，不认这些普通 token 拼写。
- **影响对象**：输出内容（静默错误）
- **影响详情**：流式 + thinking + tool_choice=auto 下模型按 grammar 用普通 token 拼出分隔符时，整个合法调用留在 reasoning 字段、tool_calls 为空；解码文本逐字正常，仅 token-ID 级检查可发现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/58105

### INF-1000 `vllm_hunyuan_a13b_streaming_answer_close_marker_leaked_into_content` vLLM hunyuan_a13b 流式解析把 </answer> 泄漏进正文

`故障` · `active` · `infra_off` · `config` · `verified` · `var` · 发现 2026-09-22 · 旧号 10.84

- **原因对象**：输出解析 / vllm / hunyuan_a13b parser 流式收尾
- **原因行为**：reasoning parser 的流式聚合路径把 answer 块的闭合标记当内容累积、终态未剥离，与非流式整体解析对同一 token 序列的处理不一致。
- **影响对象**：输出内容（静默错误）
- **影响详情**：格式良好的 <think>…</think><answer>answer</answer> 输出，非流式 content 为 "answer"，流式聚合为 "answer\n</answer>"；两请求均 HTTP 200、token IDs 相同，流式与非流式客户端结果分裂。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/58127
- **关联**：var:INF-0995

### INF-1003 `vllm_llama3_json_tool_parser_streaming_swallows_leading_brace_plain_content` vLLM llama3_json 流式 parser 吞掉以 { 开头的正文

`故障` · `active` · `infra_off` · `config` · `verified` · `var` · 发现 2026-09-26 · 旧号 10.87

- **原因对象**：输出解析 / vllm / Llama3JsonToolParser 流式判定
- **原因行为**：extract_tool_calls_streaming 仅凭首字符是否为 { 或 <|python_tag|> 判定 tool call 且永不回退，而非流式先完整解析 JSON 再判 name 键。
- **影响对象**：输出内容（静默错误）
- **影响详情**：流式下以 { 开头的普通回答返回 content=None 且零 tool call；空参数调用流式得 arguments=''；调用前有文字时流式把全部内容当 content。均 HTTP 200，与非流式结果不同。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/58824
- **关联**：var:INF-0958

### INF-1006 `vllm_parser_engine_nonstreaming_drops_post_tool_call_text_streaming_keeps` vLLM ParserEngine 非流式丢弃 tool call 后的正文

`故障` · `active` · `infra_off` · `config` · `verified` · `var` · 发现 2026-09-29 · 旧号 10.90

- **原因对象**：输出解析 / vllm / _single_pass_parse 非流式汇总
- **原因行为**：_events_to_delta 把跟在 tool 事件后的文本 defer，流式由下一个 delta flush，但非流式的 _single_pass_parse 缺少 flush 步骤。
- **影响对象**：输出内容（静默错误）
- **影响详情**：qwen3_coder、qwen3_xml、glm45、minimax_m2 等 parser 的非流式响应中，tool call 之后或之间的正文（如 "Done."）丢失、content 为 None，流式正确；/v1/completions 原样输出证明模型生成一致。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/59218
- **关联**：var:INF-0958

### INF-1011 `vllm_olmo3_reasoning_buffer_never_flushes_final_submarker_delta` vLLM Olmo3 流式 parser 尾部缓冲永不 flush

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-10-09 · 旧号 10.95

- **原因对象**：输出解析 / vllm / Olmo3 parser 流式缓冲
- **原因行为**：string_overlap 把包含关系（而非前缀重叠）当作重叠，恰为 <think>/</think> 子串的 delta 被缓冲，且流结束时没有 flush。
- **影响对象**：输出内容（静默错误）
- **影响详情**：流式生成若以 "think"、"in"、">" 这类 delta 结束，末尾文本永不 emit，reasoning/content 末尾静默丢词；非流式 extract_reasoning 返回完整内容。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/60340

### INF-1012 `vllm_internlm2_tool_parser_drops_content_after_action_start_and_streaming_chars` vLLM InternLM2 tool parser 静默丢正文与参数字符

`故障` · `active` · `infra_off` · `config` · `documented` · `var` · 发现 2026-10-09 · 旧号 10.96

- **原因对象**：输出解析 / vllm / InternLM2 tool parser 状态机
- **原因行为**：状态机遇到 <|action_start|> 进入 action 态后，若无 <|plugin|> 跟随也没有回退路径；流式增量处理丢失 function arguments 字符。
- **影响对象**：输出内容（静默错误）
- **影响详情**：流式与非流式下，<|action_start|> 后无 <|plugin|> 时其后全部 content 被丢弃；流式 tool call 参数丢字符。无报错。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/60341
- **关联**：var:INF-0958

### INF-1013 `vllm_tool_parser_batch_multiline_json_nested_args_and_newline_corruption` vLLM 三个 tool parser 对多行/深嵌套参数丢失或损坏

`故障` · `active` · `infra_off` · `config` · `documented` · `var` · 发现 2026-10-09 · 旧号 10.97

- **原因对象**：输出解析 / vllm / deepseek_v3、xlam、olmo3解析器
- **原因行为**：DeepSeek-V3、xLAM、Olmo3 parser 的增量 JSON 与括号深度处理不支持跨多行 arguments、两层以上嵌套，以及字符串内的 } 或换行。
- **影响对象**：输出内容（静默错误）
- **影响详情**：DeepSeek-V3 对跨多行 JSON arguments 的调用返回空；xLAM 流式丢超两层嵌套参数并在字符串内 } 处截断；Olmo3 破坏含换行的字符串参数并丢跨行调用。均无报错。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/60342
- **关联**：var:INF-0958

### INF-1014 `vllm_glm47_streaming_parser_stray_tool_call_token_swallows_rest_of_stream` vLLM glm47 流式解析遇 <tool_call> 字样吞掉后续

`故障` · `active` · `infra_off` · `config` · `documented` · `var` · 发现 2026-10-09 · 旧号 10.98

- **原因对象**：输出解析 / vllm / glm47 parser 流式状态机
- **原因行为**：流式 parser 遇到开标记 <tool_call> 即进入 tool 态，之后没有退回 prose 的路径。
- **影响对象**：输出内容（静默错误）
- **影响详情**：正文中零散出现 <tool_call> 字样时，其后的整条流式响应被静默截断，包括后面真正的 tool call；无报错。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/60703
- **关联**：var:INF-0958

### INF-1015 `vllm_dots_tool_parser_ignores_init_tools_and_functiontool_attribute_error` vLLM DotsToolParser 忽略构造时 tools 致调用全丢

`故障` · `active` · `infra_off` · `config` · `documented` · `var` · 发现 2026-10-09 · 旧号 10.99

- **原因对象**：输出解析 / vllm / DotsToolParser 工具来源
- **原因行为**：parser 只从 request.tools 读取已知工具名，忽略构造时传入的 tools；同时未适配 Responses API 的 FunctionTool 形状（无 .function 属性）。
- **影响对象**：输出内容（静默错误）
- **影响详情**：request.tools 为 None 时 _known_tool_names 为空，全部合法 <dots_function_call> 被当普通文本静默丢弃；Responses API 下抛 AttributeError 返回 500。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/60865
- **关联**：var:INF-0958

### INF-1016 `sglang_detector_argument_value_coercion_ignores_declared_schema_type_batch` SGLang 多个 detector 无视声明类型强转工具参数值

`故障` · `active` · `infra_off` · `config` · `documented` · `var` · 发现 2026-10-09 · 旧号 10.100

- **原因对象**：输出解析 / sglang / 四个 detector 参数值类型转换
- **原因行为**：PoolsideV1、dots、Muse Glimmer、Qwen3Coder 转换参数值时不查声明 type：只认裸标量 type、直接 json.loads、null 文本转 null、非 true 布尔一律 false。
- **影响对象**：输出内容（静默错误）
- **影响详情**：字符串型工具参数被静默改型（"123"→123、"null"→null、对象变 dict），带空白的布尔值翻为 false，下游工具调用语义漂移；无报错。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/43149
- **关联**：var:INF-0985

### INF-1017 `vllm_deepseek_r1_reasoning_streaming_deltas_raw_bpe_glyphs_nonascii` vLLM deepseek_r1 流式 reasoning 未解码致乱码

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-10-09 · 旧号 10.101

- **原因对象**：输出解析 / vllm / reasoning 流式增量解码
- **原因行为**：reasoning 通道的流式增量路径跳过 UTF-8 增量解码器，直接发出字节级 BPE token glyph。
- **影响对象**：输出内容（静默错误）
- **影响详情**：--reasoning-parser deepseek_r1 + 流式时，非 ASCII（如 CJK）reasoning 文本乱码；同流 delta.content 正确、非流式正常。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/59738

### INF-1020 `sglang_glm4_moe_parse_arguments_literal_eval_comma_scalar_to_json_array` SGLang glm4_moe detector 把含逗号标量参数解析成数组

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-10-12 · 旧号 10.104

- **原因对象**：输出解析 / sglang / glm4_moe parse_arguments
- **原因行为**：parse_arguments 的 Strategy 3 用 ast.literal_eval 解析参数，裸逗号表达式被解析为 Python tuple 再序列化为 JSON 数组，且无标量回退路径。
- **影响对象**：输出内容（静默错误）
- **影响详情**：GLM4 MoE tool call 中 "1,000" 作为 number 参数时工具收到 [1, 0]，"a","b" 同理；千分位或逗号标量静默变数组，无报错。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/43147

### INF-1021 `sglang_harmony_parser_commentary_prefix_filter_deletes_answer_head` SGLang Harmony 解析器工具调用后误删流式答案开头

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-10-12 · 旧号 10.105

- **原因对象**：输出解析 / sglang / HarmonyParser 注释过滤状态
- **原因行为**：工具调用后进入“过滤 commentary 填充”状态，把 strip/lower 后以 "commentary" 为前缀的 normal 事件静默丢弃且不重发；前缀匹配在任意 chunk 边界上不可判定，状态机无缓冲回退。
- **影响对象**：输出内容（静默错误）
- **影响详情**：Harmony parser + 流式 + 工具调用后，答案首 chunk 恰为 "c"/"Co"/"comment" 或空串时开头被删，如 "Co"+"mpare prices." 只返回 "mpare prices."；无 crash，随 chunk 边界概率性出现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/43156

### INF-1022 `sglang_deepseekv3_detector_greedy_last_sep_match_drops_first_tool_call` SGLang deepseekv3 tool 解析贪心匹配丢失首个调用

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-10-12 · 旧号 10.106

- **原因对象**：输出解析 / sglang / deepseekv3 tool 检测器
- **原因行为**：detector 正则 (.*)<｜tool▁sep｜> 贪心地从最后一个分隔符匹配，且对单个增量内的多个调用不做迭代扫描，只解析出最后一个 tool call。
- **影响对象**：输出内容（静默错误）
- **影响详情**：流式下 _coalesce_streaming_chunks 把积压 chunk 合成单增量或 --stream-interval 较大时必现：同一增量含两个完整 tool call 时只发第二个（tool_index 0），第一个永不到达客户端。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/43523

### INF-1034 `sglang_top_logprobs_decode_text_collision_collapse` SGLang top_logprobs 以解码文本为键致高 rank 被覆盖

`故障` · `active` · `infra_off` · `other_stack` · `verified` · `NEW` · 发现 2026-08-01 · 旧号 11.10

- **原因对象**：输出解析 / sglang / append_top_logprobs
- **原因行为**：append_top_logprobs 用解码后的 token 文本而非 token id 作 top_logprobs 字典的 key，两个 token id 解码为同一字符串时，后出现的低 rank 项覆盖高 rank 项。
- **影响对象**：输出内容；输出分布（静默错误）
- **影响详情**：byte-fallback 分词器下多个 token id 解码为同一替换字符片段时，返回给客户端的 top_logprobs 中最高 rank 候选的 logprob 被低 rank 替换；主 token 正确，无 crash、无 NaN。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/pull/32387

### INF-1035 `sglang_mimo_toolcall_streaming_silent_text_drop` SGLang MiMo 流式 tool call 解析静默丢文本

`故障` · `active` · `infra_off` · `other_stack` · `verified` · `NEW` · 发现 2026-08-02 · 旧号 11.11

- **原因对象**：输出解析 / sglang / MiMoDetector 流式增量解析
- **原因行为**：current_tool_id>0 后文本被持续缓冲，stream 结束时从不清空 buffer；<tool_call> 被增量切分时，no-marker 分支把 buffer 当 normal text flush 并清空。
- **影响对象**：输出内容（静默错误）
- **影响详情**：流式模式下最后一个 tool call 之后的文本（解释、总结）永远不被发出；标记被切分时整个 tool call 识别失败，原始 markup 作为 content 流给用户。one-shot 与 chunked 差异测试可暴露。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/33186

### INF-1037 `vllm_base_thinking_reasoning_parser_midstream_think_silent_text_drop` vLLM 推理解析基类遇中段 <think> 静默丢文本

`故障` · `active` · `infra_off` · `yes` · `documented` · `NEW` · 发现 2026-08-03 · 旧号 11.13

- **原因对象**：输出解析 / vllm / 推理解析基类 extract_reasoning
- **原因行为**：基类 extract_reasoning 在 <think> 出现于流中段时丢弃部分模型输出，被丢文本既不进 content 也不进 reasoning；继承该基类且未覆写此方法的 parser 均受影响。
- **影响对象**：输出内容（静默错误）
- **影响详情**：使用 deepseek_r1、poolside_v1 等 reasoning parser 时，模型输出中段出现 <think> 后，响应缺一段文本且无任何指示；minimax_m3/ernie45 因自行覆写不受影响，与硬件/后端无关。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/50901

### INF-1038 `vllm_gemma4_tool_parser_empty_tool_calls_concurrent_silent_loss` vLLM Gemma4 tool parser 并发返回空 tool_calls

`故障` · `active` · `infra_off` · `yes` · `documented` · `NEW` · 发现 2026-08-03 · 旧号 11.14

- **原因对象**：输出解析 / vllm / Gemma4 tool-call parser
- **原因行为**：非流式 chat completions 下 Gemma4 tool parser 在并发负载时间歇丢失解析出的调用，却仍置 finish_reason=tool_calls，原始文本也不回填 content；具体根因原文未给出。
- **影响对象**：输出内容（静默错误）
- **影响详情**：并发 agentic 负载下间歇返回 finish_reason="tool_calls" 但 tool_calls 为空数组、content 为空或短回显；约 2 天 5 次，每次打断 agent 回合（gemma-4-31B W4A16，TP8）。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/50889

### INF-1040 `vllm_minimax_m3_structured_output_silently_ignored_thinking` vLLM MiniMax-M3 thinking 下结构化输出被静默忽略

`故障` · `active` · `infra_off` · `yes` · `verified` · `NEW` · 发现 2026-08-06 · 旧号 11.16

- **原因对象**：输出解析 / vllm / MiniMax-M3 推理结束检测
- **原因行为**：推理结束检测在 _reasoning_active_streaming 置位时早返回 False，该 flag 只由前端 detokenizer 路径清除、scheduler 从不清除，thinking 下 grammar 永不启用。
- **影响对象**：输出内容（静默错误）
- **影响详情**：MiniMax-M3 + thinking_mode: enabled + response_format json_schema/json_object 时，服务器返回 HTTP 200 和无约束 prose，structured output 被静默忽略，无警告。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/51230

### INF-1049 `vllm_streaming_stop_buffer_desync_reasoning_content_misroute` vLLM 流式 stop 缓冲致 </think> 泄漏进 content

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-08-10 · 旧号 11.25

- **原因对象**：输出解析 / vllm / detokenizer stop buffer
- **原因行为**：V1 detokenizer 的 stop buffer 暂存最多 max(len(stop))-1 个字符的 delta_text，却不暂存对应 delta_token_ids，reasoning parser 的两路输入因此不同步。
- **影响对象**：输出内容（静默错误）
- **影响详情**：流式请求带非空 stop 且 include_stop_str_in_output=false，在 MTP/推测解码/异步输出下单个 token batch 跨越 </think> 边界时，reasoning 尾部与 </think> 标记泄漏进 content；无 crash、无 NaN。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/51641

### INF-1055 `vllm_stop_string_scope_crosses_reasoning_segment_truncates_cot` vLLM stop 串在 think 段内命中致 CoT 截断

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-08-14 · 旧号 11.32

- **原因对象**：输出解析 / vllm / V1 detokenizer stop 串判定
- **原因行为**：V1 detokenizer 把客户端 stop 串对整个输出流（含 reasoning 段）求值，未把作用域限定在 content 段，CoT 自然复述的 stop 串在 think 段内即命中终止。
- **影响对象**：输出内容（质量退化）
- **影响详情**：think-in-prompt 模板加客户端 stop 串（如 lm-eval 的 Question:）时，</think> 不出现，返回 content=None、finish_reason=stop；GSM8K n=50 有 8–15 条被截断，分数 0.66–0.84，抑制后 0.98。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/52393

### INF-1097 `vllm_ngram_spec_decode_qwen3_tool_parser_argument_corruption` vLLM ngram 推测下 qwen3 tool parser 损坏工具参数

`故障` · `active` · `infra_off` · `partial` · `documented` · `var` · 发现 2026-09-09 · 旧号 11.74

- **原因对象**：输出解析 / vllm / Qwen3EngineToolParser
- **原因行为**：ngram 推测解码的 draft/verify/回退边界与增量 tool parser 的流式解析窗口错位：回退重放时 parser 的部分事务状态与重放 token 流不一致，参数字段被跨事务重组。
- **影响对象**：输出内容（静默错误）
- **影响详情**：同时启用 ngram 推测解码、qwen3_coder 类 tool parser 与 qwen3 reasoning parser 时，tool call 参数为空、键值折叠、函数名损坏或泄漏 XML 标签；模型推理已正确决定参数，关闭 ngram 即恢复。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/56077
- **关联**：var:INF-1052

### INF-1154 `sglang_anthropic_streaming_multi_tool_arguments_merged_single_block` SGLang Anthropic 流式多 tool call 参数并入同一块

`故障` · `active` · `infra_off` · `other_stack` · `verified` · `var` · 发现 2026-09-16 · 旧号 12.38

- **原因对象**：输出解析 / sglang / Anthropic 流式块索引路由
- **原因行为**：Anthropic 流式适配器把 OpenAI 风格的 tool-call index 翻译为 content_block index 时未跟踪多调用的块边界，所有 input_json_delta 都路由到当前打开的第一个块。
- **影响对象**：输出内容（静默错误）
- **影响详情**：/v1/messages stream=true 且单响应含多个 tool call 时，三个调用的参数全标 index=0 灌入同一块，聚合成拼接的非法 JSON，工具调用全部失效，HTTP 200 无错；OpenAI 端点流式/非流式与 Anthropic 非流式均正确。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/39782
- **关联**：var:INF-0971

## 解码/采样

### INF-0123 `trtllm_blackwell_gvr_topk_decode_barrier_divergence_race` TRT-LLM GVR top-k 内核 barrier 分歧竞态

`故障` · `active` · `infra_off` · `other_stack` · `verified` · `NEW` · 发现 2026-08-29 · 旧号 1.124

- **原因对象**：解码/采样 / tensorrt-llm / GVR top-k decode 内核
- **原因行为**：hint 依赖内核假设 hint 集合有区分度：全零 pre_idx 使全部线程走同一 gather 分支，tie 行使 hint 值逐位碰撞，均破坏 barrier 到达率假设，引发 barrier 分歧竞态。
- **影响对象**：数值结果；可用性/进度（静默错误）
- **影响详情**：B200（sm100a）上冷启动全零 pre_idx 或量化 logits 的 tie 密集行即触发输出损坏或 CUDA_ERROR_ILLEGAL_ADDRESS 类崩溃；生产首个 decode step 可命中，完美 hint 也复现，cutlass-dsl 三版本均复现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/NVIDIA/TensorRT-LLM/issues/18338

### INF-0260 `ttmetal_dflash_shard_argmax_dealloc_freed_buffer_view_rows1` dFlash 单行 _shard_argmax 读取已释放 buffer

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-09-26 · 旧号 1.261

- **原因对象**：解码/采样 / tt-metal / dFlash 草稿 _shard_argmax
- **原因行为**：rows==1 时 ttnn.reshape 返回与 sel_u 共享设备 buffer 的视图，紧接着 sel_u.deallocate(True) 绕过共享引用检测强制释放该 buffer，返回的视图指向已释放内存。
- **影响对象**：输出内容（不可复现）
- **影响详情**：开启 GEMMA4_DFLASH_SHARD_ARGMAX=1、TP>1 且 drafter 仅 1 行时，选出的 draft token id 来自已释放内存，内容随后续分配时序漂移（最小重现中被 7 填充），无报错；verify 调用（≥2 行）与默认配置不受影响。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/57989

### INF-0361 `vllm_nemotronh_mtp_speculator_v1_path_bypasses_sampling_chain` vLLM XPU speculator 绕过验证链致 MTP 接受率为 0

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-08-14 · 旧号 5.14

- **原因对象**：解码/采样 / vllm / XPU 后端 V1 speculator 执行路径
- **原因行为**：XPU 后端 V1 speculator 路径完全绕过标准采样/验证链（sample、RejectionSampler 等均不执行），draft token 照常提出但从不验证。
- **影响对象**：性能/资源（性能退化）
- **影响详情**：Intel Arc Pro B70 上 NemotronH native MTP 推测解码无 crash，spec 计数器递增但 draft acceptance 恒为 0%，推测解码静默完全失效、无加速。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/52107

### INF-0366 `sglang_eagle_nextn_xpu_tp2_warmup_hang` SGLang EAGLE/NEXTN XPU TP=2 warmup 死锁

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-17 · 旧号 5.19

- **原因对象**：解码/采样 / sglang / verify 决策的 TP broadcast
- **原因行为**：#34238 把 verify-decision 的 TP broadcast 移出 sampler 后，EAGLE/NEXTN 推测解码 verify 路径的 collective 顺序在 XPU 后端上不再匹配。
- **影响对象**：可用性/进度（挂起）
- **影响详情**：Intel XPU 上 EAGLE/NEXTN + TP=2 时 warmup 阶段死锁、永不完成，进程存活但不服务；NVIDIA 路径正常，属 #34238 之后的回归。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/35144

### INF-0396 `vllm_chunked_prefill_logprob_norecompute` vLLM 分块 prefill 致 prompt_logprobs 偏移

`故障` · `active` · `model_on` · `config` · `verified` · `NEW` · 发现 待定 · 旧号 6.3

- **原因对象**：解码/采样 / vllm / prompt_logprobs 处理器
- **原因行为**：V1 始终启用 chunked prefill；prompt_logprobs 处理器把逐 chunk 的 logprob 聚合后即丢弃，较早或已缓存的 chunk 不会针对最终全上下文 attention 重新计算。
- **影响对象**：输出分布；训练信号（静默错误）
- **影响详情**：两个共享前缀、仅最后一个 token 不同的 prompt，在相同的共享前缀位置得到不同的 prompt_logprob；GRPO/PPO 的重要性采样 ratio 随之静默偏移，生成的 token 文本通常不变。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/25262

### INF-0397 `vllm_apply_temperature_inf_nan` vLLM apply_temperature 遇 -inf 产生 NaN

`故障` · `active` · `infra_off` · `partial` · `verified` · `NEW` · 发现 待定 · 旧号 6.4

- **原因对象**：解码/采样 / vllm / apply_temperature 与温度校验
- **原因行为**：apply_temperature 用温度去除 logits，含 -inf 时得 +inf，softmax 后整个概率向量变 NaN；温度校验用 </> 比较，对 NaN/Inf 静默为 False（CVE-2026-54235）。
- **影响对象**：输出分布；输出内容（数值异常）
- **影响详情**：每当 -inf logit 遇上 temperature scaling，就偶发吐出垃圾、重复或无意义 token；隐状态不受影响，故障只在采样阶段出现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/22180；https://github.com/vllm-project/vllm/pull/24734

### INF-0398 `shared_static_sampler_concurrency` llama.cpp sampler 静态状态被并发序列共享致采样互相破坏

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 待定 · 旧号 6.5

- **原因对象**：解码/采样 / llama.cpp / sampler RNG 与 Mirostat mu
- **原因行为**：sampler 的 RNG state、Mirostat 的 mu 等 static/全局变量被并发的 worker/sequence 共享，并发请求相互破坏对方的采样状态。
- **影响对象**：跨请求/跨样本数据；输出内容（不可复现）
- **影响详情**：并发服务时采样状态在序列间泄漏，选出错误 token；结果不可复现且与负载相关，模型 forward 本身健康。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/ggml-org/llama.cpp/issues/3537；https://github.com/ggml-org/llama.cpp/issues/12220

### INF-0400 `tgi_guidance_structured_output` TGI 受约束解码与流式反分词产生畸形结构化输出

`故障` · `active` · `ambiguous` · `other_stack` · `documented` · `NEW` · 发现 待定 · 旧号 6.7

- **原因对象**：解码/采样 / tgi / guidance/outlines 约束施加；分词/模板 / tgi / special token 与流式反分词
- **原因行为**：合并条目：受约束解码的约束施加有误；special token 在约束/流式路径上被切开；stream 模式下的 detokenization 产生空白/边界伪影。
- **影响对象**：输出内容（静默错误）
- **影响详情**：TGI 搭配 guidance/outlines 做结构化输出并流式返回时，出现畸形 JSON/markdown、被切开的 special token 与空白字符伪影。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/4577

### INF-0415 `vllm_topk_per_row_nan_uninit_smem_index` top_k_per_row 内核对 NaN 行输出未初始化 index

`故障` · `active` · `infra_off` · `partial` · `verified` · `var` · 发现 2026-08-01 · 旧号 6.22

- **原因对象**：解码/采样 / vllm / sampler.cu topKPerRowJob
- **原因行为**：extractBinIdx 把正 NaN 分到 +inf 之上，NaN 主导行的阈值 bin 落在 NaN bin；插入排序的 NaN 比较全为 false，其余 smemOutput 槽保留未初始化 smem 并被当作 index 输出。
- **影响对象**：数值结果；可用性/进度（报错）
- **影响详情**：NaN 主导的行（非 NaN 候选少于 top_k 时还会未初始化读取 smemThresholdBinIdx）输出垃圾 index，下游 gather 触发设备端 assert 或非法内存访问；为 6.17 同一故障链的内核级根因。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/50201
- **关联**：var:INF-0410

### INF-0418 `vllm_override_generation_config_silent_param_drop` vLLM override-generation-config 静默丢弃惩罚参数

`故障` · `active` · `infra_off` · `config` · `verified` · `var` · 发现 2026-08-02 · 旧号 6.25

- **原因对象**：解码/采样 / vllm / available_params 采样参数白名单
- **原因行为**：override-generation-config 的 available_params 白名单不含 presence_penalty 与 frequency_penalty，传入的这两个参数被默默丢弃，无警告或错误。
- **影响对象**：输出分布；输出内容（静默错误）
- **影响详情**：vLLM 0.26.x 通过 override-generation-config 设置 presence/frequency penalty 后，得到的是无惩罚的输出，与配置及文档承诺不符；无 crash、无 NaN、无报错。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/50767
- **关联**：var:INF-0412

### INF-0440 `vllm_gdn_kda_stale_zero_accept_silent_neg1_corruption` vLLM 零接受计数致 GDN/KDA 状态 -1 索引

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-08-08 · 旧号 6.47

- **原因对象**：解码/采样 / vllm / GDN/KDA 后端的接受计数索引
- **原因行为**：async scheduling 丢弃的步骤把 valid_sampled_token_count=0 写入 num_accepted_tokens，消费者假设计数 ≥1，用 count-1 索引得到 -1。
- **影响对象**：KV状态；输出内容（静默错误）
- **影响详情**：GDN/KDA 模型开 async scheduling 与推测解码时触发；Triton kernel 路径越界崩溃（illegal memory access），CPU 对齐模式下 -1 是合法索引，静默从 state 末尾读取，输出损坏且无报错。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/51508

### INF-0442 `vllm_dsd_k_lookup_counts_prefills_k_thrash` vLLM DSD K-lookup 误计 prefill 致吞吐崩溃

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-08-08 · 旧号 6.49

- **原因对象**：解码/采样 / vllm / scheduler 的 DSD K-lookup
- **原因行为**：K-lookup 用 len(num_scheduled_tokens) 计 batch size，计入 mid-prefill/chunked-prefill 请求而非只计当步采样请求，K 在值间反复跳变（如 2→0→2）。
- **影响对象**：性能/资源（性能退化）
- **影响详情**：DSD 下 prefill 与 decode 混合 batch 时触发；无 crash，每次 K 翻转都改变 verify-token 数、forward shape 与 CUDA graph dispatch，吞吐灾难性下降，是 #49548 吞吐崩溃的诱因。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/51466

### INF-0456 `vllm_tli_heterogeneous_vocab_local_argmax_silent_skip` vLLM TLI 异构词表与 local-argmax 同启跳过 ID 映射

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-08-09 · 旧号 6.63

- **原因对象**：解码/采样 / vllm / _greedy_sample 提前返回
- **原因行为**：use_heterogeneous_vocab 与 use_local_argmax_reduction 无互斥校验；同启时 local-argmax 分支提前返回，跳过交集 mask 与 draft→target 映射，输入侧却仍映射。
- **影响对象**：输出内容；性能/资源（静默错误）
- **影响详情**：TLI 异构词表与 local-argmax 同启时触发，文档配对 Qwen3-8B + SmolLM2-135M 即可触达；无 crash，spec acceptance 下降，或向 rejection sampling 暴露未映射 token id。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/51561

### INF-0534 `vllm_compile_cache_key_omits_spec_tokens_stale_graph` vLLM 编译缓存键漏推测 token 数致复用 stale 图崩溃

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-08-22 · 旧号 6.141

- **原因对象**：解码/采样 / vllm / 推测配置 compute_hash
- **原因行为**：SpeculativeConfig.compute_hash() 不含 num_speculative_tokens，而 VllmConfig.compute_hash() 委托它生成 torch.compile 缓存键，不同 k 值同键。
- **影响对象**：可用性/进度（报错）
- **影响详情**：同一模型扫 k=7/9/11 并复用 cache 目录时，后续 k 加载为其他 k 编译的 Inductor 图，运行期报 expected size 7==9 断言；干净 cache 上正常，症状随 cache 历史漂移、跨机器难归因。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/53366

### INF-0571 `sglang_nextn_draft_acceptance_uptime_decay_zero_restart_heals` SGLang NEXTN 接受率随 uptime 衰减至 0，重启恢复

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-09-01 · 旧号 6.178

- **原因对象**：解码/采样 / sglang / NEXTN 推测解码累积状态
- **原因行为**：随服务 uptime 累积的某个服务端状态（原文推测为 drafter 缓存/计数器/KV 组织，未定位）使 draft 与 target 分布逐渐脱钩。
- **影响对象**：性能/资源（性能退化）
- **影响详情**：SGLang + Qwen3.8-Flash-Next + NEXTN/MTP 长时间运行，accept rate 从 fresh boot 的 0.47–0.76 在约 24h 内降到 0.00、吞吐降 2.9×；无 crash、错误或警告，完全重启可复现地恢复。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/37326

### INF-0589 `sglang_spec_custom_mask_torch_cat_unbounded_growth_stale_mask` SGLang EAGLE/DFlash custom_mask 只增不减

`故障` · `active` · `infra_off` · `other_stack` · `verified` · `NEW` · 发现 2026-09-04 · 旧号 6.196

- **原因对象**：解码/采样 / sglang / EAGLE/DFlash custom_mask
- **原因行为**：generate_attn_arg_prefill 在 batch 增大时用 torch.cat 追加 True padding 扩展 custom_mask 并存回 self，只增不减；超出当前 batch 的陈旧条目留在张量中。
- **影响对象**：性能/资源；数值结果（性能退化）
- **影响详情**：batch 尺寸波动时 GPU 显存随峰值 batch 无界增长，长时间运行最终 OOM；若注意力内核读越当前 batch 边界则吃到过期掩码；无即时错误，波动越大越快显形。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/37553

### INF-0619 `ttmetal_gptoss_galaxy_host_sampled_structured_decode_first_token_garbage` tt-metal GPT-OSS 结构化输出 host 采样路径乱码

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-09-10 · 旧号 6.226

- **原因对象**：解码/采样 / tt-metal / host-logits decode 变体路径
- **原因行为**：根因待定；structured 请求被强制 host 采样，跑 host-logits decode 变体（每步回读全 padded-vocab logits），首选候选是该路径损坏，另有请求更替设备状态、EP 行分歧等候选。
- **影响对象**：输出内容（静默错误）
- **影响详情**：Galaxy 上 GPT-OSS-120B structured-output 测试三周未过闸（HT 90.6–96.9%、LL 15.6–37.5%）：失败样本 prefill 正确，自首个 decode token 起生成单词循环或空响应；同 build 的 device 采样 demo 正常。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/56100

### INF-0638 `vllm_prompt_embeds_any_penalty_scatter_gather_oob_assert_engine_death_all_tenants` vLLM prompt_embeds 带 penalty 越界致引擎永久死亡

`故障` · `active` · `infra_off` · `config` · `verified` · `NEW` · 发现 2026-09-19 · 旧号 6.245

- **原因对象**：解码/采样 / vllm / penalty 应用（token ids 索引）
- **原因行为**：penalty 应用假设请求必有 token ids 并以其为 scatter/gather 索引源；prompt_embeds 请求的 ids 通道为空或未初始化，导致索引越界触发 device-side assert。
- **影响对象**：可用性/进度（报错）
- **影响详情**：开 --enable-prompt-embeds 时，带任一 penalty 的 embeds 请求批首轮即 EngineDeadError，引擎永不恢复，在途与后续请求全部 HTTP 500 须重启；16 并发混合长度 10/10 必死，普通请求即可对全体租户 DoS。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/57719

### INF-0655 `vllm_dspark_lookahead_underallocation_overwrites_target_kv_cross_request` vLLM DSpark 预留少一槽致 drafter 覆写他请求 KV

`故障` · `active` · `infra_off` · `partial` · `verified` · `NEW` · 发现 2026-09-29 · 旧号 6.262

- **原因对象**：解码/采样 / vllm / DSpark lookahead 预留
- **原因行为**：sample_from_anchor=false 时 DSpark drafter 用 N+1 个 query 位置，num_lookahead_tokens 只预留 N 槽；越界 slot 经 block table 尾部读到陈旧块号。
- **影响对象**：KV状态；跨请求/跨样本数据；输出内容（静默错误）
- **影响详情**：如 137 token prompt + 7 draft + 16-token 块时调度分配 144 槽而 drafter 写位置 144，覆写另一请求的 target KV，受害请求生成重复畸形输出；GPU instrumentation 确认覆写，无 crash。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/59104

### INF-0669 `sglang_hybrid_gdn_dflash_grammar_traffic_batchwide_token0_loop_collapse` SGLang DFLASH 遇 grammar 流量整批坍塌为 token 0

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-10-12 · 旧号 6.276

- **原因对象**：解码/采样 / sglang / grammar bitmask×spec 批状态
- **原因行为**：hybrid GDN 下 grammar bitmask 与 spec-decode（DFLASH）批状态交互，污染整批 logits 路径；根因未定位。
- **影响对象**：输出内容；跨请求/跨样本数据（质量退化）
- **影响详情**：批内有 grammar 约束请求（tools+tool_choice required）时，整批运行中请求（含无 grammar 请求）一步坍塌为 token id 0 重复到 max_tokens，accept 率三步内 0.48→0.00；新准入请求健康，触发请求离开后自愈。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/43514

### INF-0706 `sglang_dspark_draft_quantized_lm_head_shape_dtype_mismatch` DSpark draft 未处理量化 lm_head，启动崩溃

`故障` · `active` · `infra_off` · `partial` · `verified` · `NEW` · 发现 2026-08-13 · 旧号 7.33

- **原因对象**：解码/采样 / sglang / DSpark draft lm_head 投影
- **原因行为**：draft 借用 target 的 lm_head，直接对 .weight 做 dense matmul；lm_head 为 uint8 packed 时 K 维错误、激活被转成 uint8，采样 buffer 也取 uint8。
- **影响对象**：可用性/进度（报错）
- **影响详情**：target 检查点 lm_head 已量化（如 ModelOpt NVFP4）且启用 DSpark 推测解码时，CUDA graph capture 与 eager 均报 mat1 and mat2 shapes cannot be multiplied，server 启动即 crash。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/pull/34742

### INF-0718 `sglang_dspark_base_logits_packed_lmhead_matmul_mismatch` SGLang DSpark 直接乘打包 NVFP4 lm_head 致崩溃

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-08-18 · 旧号 7.45

- **原因对象**：解码/采样 / sglang / DSpark base-logits 计算
- **原因行为**：DSpark draft 直接用 torch.matmul 乘 lm_head.weight 算 base logits，绕过 quant_method.apply，假设权重为 dense 形状，未处理 NVFP4 打包后的物理形状。
- **影响对象**：可用性/进度（报错）
- **影响详情**：DSpark draft + ModelOpt NVFP4 打包 lm_head 目标模型时，CUDA graph capture 阶段抛形状 RuntimeError（245×5120 与 2560×248320 不可乘）；降低 mem-fraction 或 graph batch 无效。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/35354

### INF-0799 `trtllm_onemodel_mtp_kv_cache_overcount_illegal_page_access` TRT-LLM one-model MTP 首步高估 KV 长度致非法页访问

`故障` · `active` · `infra_off` · `other_stack` · `verified` · `NEW` · 发现 2026-08-04 · 旧号 8.15

- **原因对象**：解码/采样 / tensorrt-llm / MTP 首步 KV 长度与 draft 计数
- **原因行为**：MTP 首个 overlapped generation step 在缺 kv_lens_cuda 时用了超出 prompt KV 边界的 cached length；KV scheduler 缺 Python 侧 draft 计数。
- **影响对象**：可用性/进度（报错）
- **影响详情**：one-model MTP 推测解码 + FlashInfer paged KV 下，首步访问无效 FlashInfer page，触发 CUDA illegal memory access 崩溃，KV scheduler 容量规划同时出错；非静默。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/NVIDIA/TensorRT-LLM/pull/17264

### INF-0845 `xpu_gdn_attention_ragged_ngram_draft_length_assert_engine_death` XPU gdn_attention 遇 ragged draft 断言致引擎死亡

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `var` · 发现 2026-09-01 · 旧号 8.61

- **原因对象**：解码/采样 / vllm / gdn_attention 推测 token 计数
- **原因行为**：XPU gdn_attention 的 AOT 编译子模块按 batch 内 draft 长度均匀的假定计算 spec_token 总数，ngram 部分匹配产生的 ragged draft 长度使 shape 不符而触发内核断言
- **影响对象**：可用性/进度（报错）
- **影响详情**：Intel XPU 上 hybrid GDN 模型启用 ngram 推测解码时，常规部分匹配即触发 assert（单流即可、非混合 batch/空 draft），EngineDeadError 致引擎关闭、请求返回 500，需重启服务
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/54740
- **关联**：var:vllm_pd_mamba_spec_padding_window_align_clip_mismatch

### INF-0925 `transformers_generation_config_value_precedence_wrong` generation_config 优先级解析错误致参数被覆盖

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-08-06 · 旧号 10.9

- **原因对象**：解码/采样 / transformers / generation_config 优先级解析
- **原因行为**：多处指定同一生成参数（model.generation_config、调用时 GenerationConfig(...)、generate(**kwargs)）时优先级解析错误，低优先级来源的值覆盖了应生效的值。
- **影响对象**：输出分布；输出内容（静默错误）
- **影响详情**：多处指定 temperature/top_p 等生成参数时触发；无 crash、无 NaN、无警告，如 temperature 不生效，输出分布偏离用户意图。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/huggingface/transformers/issues/47752

### INF-0928 `vllm_v2_cpu_runner_structured_output_cuda_assume` V2 CPU runner 结构化输出假设 CUDA 致 engine 终止

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-08-08 · 旧号 10.12

- **原因对象**：解码/采样 / vllm / CPU runner 结构化输出路径
- **原因行为**：V2 CPU model runner 继承 StructuredOutputsWorker，其结构化输出路径假设 CUDA 可用：创建 CUDA stream、分配 pinned memory、调用 Triton kernel。
- **影响对象**：可用性/进度（报错）
- **影响详情**：CPU-only 部署收到结构化输出（grammar/约束解码）请求时触发；engine 终止，一个结构化输出请求即可杀掉整个 CPU-only 部署。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/51306

### INF-0941 `vllm_muse_glimmer_schema_silent_skip_with_reasoning` muse_glimmer 下 JSON schema 默认静默不强制

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-08-17 · 旧号 10.25

- **原因对象**：解码/采样 / vllm / muse_glimmer 下结构化输出约束
- **原因行为**：结构化输出与 reasoning parser 的互斥未被建模：enable_in_reasoning=True 时 grammar mask 从 token 0 生效；默认 False 时不强制 JSON schema 且无警告。
- **影响对象**：输出内容（静默错误）
- **影响详情**：muse_glimmer 下请求 JSON schema 且用默认 enable_in_reasoning=False 时触发；请求正常返回、无警告，输出实为自由生成，下游 JSON 解析才可能报错；设为 True 则模型打不开 to=self 通道、message.reasoning 为空。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/52594

### INF-0952 `vllm_bad_words_empty_tokenization_index_crash` vLLM bad_words 编码为空时 IndexError 返回 500

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-08-23 · 旧号 10.36

- **原因对象**：解码/采样 / vllm / bad_words 分词更新逻辑
- **原因行为**：update_from_tokenizer 假设非空字符串至少编码出一个 token；分词器把含非空白控制字符的 bad_words 条目编码为空列表时，处理空格前缀形式仍对空列表索引，抛 IndexError。
- **影响对象**：可用性/进度（报错）
- **影响详情**：SmolVLM 类分词器 + 含非空白控制字符的 bad_words 条目（fuzzing 可稳定触发）时，请求以 HTTP 500 + traceback 失败，而非结构化的 4xx invalid-request 响应。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/53433

### INF-0953 `hf_stop_string_criteria_vocab_probe_roundtrip_failure` transformers StopStringCriteria 词表探针报错

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-08-25 · 旧号 10.37

- **原因对象**：解码/采样 / transformers / StopStringCriteria 词表清洗
- **原因行为**：_clean_tokenizer_vocab 用“ASCII 探针前缀 + token 解码后剥前缀”验证词表，假设任意 token 都能与前缀拼接 round-trip；word-level 非拉丁词表不满足，剥前缀失败。
- **影响对象**：可用性/进度（报错）
- **影响详情**：word-level 非拉丁词表（如中文 WordLevel + Whitespace）构造 StopStringCriteria 即抛裸 ValueError: substring not found，错误信息指向内部实现，用户无法归因到词表问题。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/huggingface/transformers/issues/48320

### INF-0956 `vllm_legacy_guided_json_silently_dropped_unconstrained_200` vLLM legacy guided_json 字段被静默丢弃，返回无约束文本

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-08-27 · 旧号 10.40

- **原因对象**：解码/采样 / vllm / legacy guided_json 字段处理
- **原因行为**：请求解析层无声忽略 legacy 顶层 guided_json 字段：既未路由到已配置的 xgrammar 结构化输出后端，也未按已移除字段拒绝请求。
- **影响对象**：输出内容（静默错误）
- **影响详情**：vLLM 0.26.1rc1+ 上携带 guided_json 的 /v1/chat/completions 请求返回 200 与非 JSON 自由文本，无客户端 warning、无服务端 error；依赖该字段结构保证的存量客户端下游 JSON 解析失败或误读。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/53975

### INF-0963 `sglang_xgrammar_pattern_minlength_combo_drops_length_bound` SGLang xgrammar pattern+minLength 丢长度约束

`故障` · `active` · `infra_off` · `other_stack` · `verified` · `var` · 发现 2026-09-03 · 旧号 10.47

- **原因对象**：解码/采样 / xgrammar / JSON schema→CFG 编译
- **原因行为**：xgrammar 的 JSON schema→CFG 编译不完整支持 pattern 与 minLength 联合语义，缺失时既不拒绝 schema 也不施加长度约束；outlines 后端则互补地丢弃 pattern 与数值界。
- **影响对象**：输出内容（静默错误）
- **影响详情**：JSON schema 同时含 pattern 与 minLength（如 ^[a-z]+$ + minLength 5）时，结构化输出接受 "ab" 等违反约束的字符串且无报错；换 outlines 后端丢的约束子集不同，同一 schema 在不同部署语义不一致。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/37707
- **关联**：var:INF-0956

### INF-0964 `vllm_guidance_backend_tool_structural_tags_triggers_keyerror` vLLM guidance 后端拒 triggers 键致工具请求 400

`故障` · `active` · `infra_off` · `config` · `verified` · `NEW` · 发现 2026-09-04 · 旧号 10.48

- **原因对象**：解码/采样 / vllm / guidance 后端 tool grammar
- **原因行为**：vLLM 组装的 tool-call grammar 含 triggers 结构标签，llguidance 1.7.6 不认识该键（接口版本漂移）而抛 KeyError，被包装为 ValidationError 透出。
- **影响对象**：可用性/进度（报错）
- **影响详情**：guidance 后端下，产生结构标签的 tool 请求（auto+strict、required、named）在首 token 前全部被拒（400 或流内 error）；无 strict 的 auto 不受影响，与模型/GPU 无关。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/55152

### INF-0970 `vllm_tool_choice_required_streaming_thinking_off_not_enforced` vLLM 流式无思考模式下 tool_choice=required 不被强制

`故障` · `active` · `infra_off` · `partial` · `documented` · `var` · 发现 2026-09-06 · 旧号 10.54

- **原因对象**：解码/采样 / vllm / tool_choice=required 约束
- **原因行为**：流式 + reasoning-parser + 大工具集组合下，tool_choice=required 的结构化输出约束失联（疑似语法构造/advance 在 reasoning 段切换处丢失）。
- **影响对象**：输出内容（静默错误）
- **影响详情**：12 工具、流式且关闭 thinking 时 3 次中 2 次返回纯文本、tool_calls 为空；非流式、thinking on 或仅 2 工具时正常；TC-45 0/8（SGLang 8/8）；thinking on+MTP 时 xgrammar 报 FSM advance 失败。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/55552
- **关联**：var:INF-0574

### INF-0975 `sglang_kimik3_strict_tool_grammar_additional_properties_dilutes_named_property` SGLang Kimi-K3 strict tool 具名属性约束被稀释

`故障` · `active` · `infra_off` · `other_stack` · `verified` · `var` · 发现 2026-09-09 · 旧号 10.59

- **原因对象**：解码/采样 / sglang / get_kimik3_structural_tag
- **原因行为**：构建 structural tag 时未按 JSON Schema 语义把具名 property 从 additionalProperties matcher 中排除，两个 matcher 的接受集并集超出具名键的合法域。
- **影响对象**：输出内容（静默错误）
- **影响详情**：strict tool 参数 schema 同时含具名 property 与 schema 值 additionalProperties 时，声明为 integer 的属性可输出为 object 仍通过 grammar（HTTP 200），下游解析到类型不符的值；CPU 可确定性复现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/38587
- **关联**：var:INF-0963

### INF-0977 `vllm_beam_search_ignores_skip_special_tokens_control_tokens_exposed` vLLM beam search 不应用 skip_special_tokens

`故障` · `active` · `infra_off` · `yes` · `verified` · `var` · 发现 2026-09-10 · 旧号 10.61

- **原因对象**：解码/采样 / vllm / beam search 输出文本组装
- **原因行为**：beam search 输出组装（online 与 offline 共用）直接 decode 全部 token id，未把 skip_special_tokens 传给 detokenizer；常规采样路径链路完整。
- **影响对象**：输出内容（静默错误）
- **影响详情**：use_beam_search=true 且模型生成控制 token（如 Voxtral Realtime 转录的 [STREAMING_PAD]/[STREAMING_WORD]）时，返回文本暴露数百个内部 special token；同请求常规采样输出干净。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/56209
- **关联**：var:INF-0968

### INF-0980 `vllm_xgrammar_multibranch_allof_silently_unenforced_no_fallback` vLLM xgrammar 多分支 allOf 不强制且 auto 不回退

`故障` · `active` · `infra_off` · `yes` · `verified` · `var` · 发现 2026-09-13 · 旧号 10.64

- **原因对象**：解码/采样 / vllm / xgrammar allOf 编译与 gate
- **原因行为**：xgrammar 0.2.3 把多分支 allOf schema 编译成恒真文法；vLLM 的 has_xgrammar_unsupported_json_features 未列 allOf，auto 模式不回退到可用后端。
- **影响对象**：输出内容（静默错误）
- **影响详情**：默认 auto 后端下含多分支 allOf（如 Pydantic/OpenAPI 继承）的 json schema 请求返回 200 及违反 schema 的 JSON：违反 minimum、缺 required 字段乃至纯字符串均被接受；guidance 后端同 schema 正确强制。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/56556
- **关联**：var:INF-0963

### INF-0989 `vllm_reasoning_parser_completions_structured_output_bitmask_never_applied_unconstrained` vLLM reasoning-parser 下 completions 约束失效

`故障` · `active` · `infra_off` · `config` · `verified` · `var` · 发现 2026-09-19 · 旧号 10.73

- **原因对象**：解码/采样 / vllm / 结构化输出 reasoning_ended 门控
- **原因行为**：bitmask 以 reasoning_ended 门控，初值取 is_reasoning_end(prompt_token_ids)；裸 completion prompt 无 think 标记，门控恒假，bitmask 从不施加。
- **影响对象**：输出内容（静默错误）
- **影响详情**：配置 --reasoning-parser 后 /v1/completions 的 json/regex/choice 及 guided_json 请求返回 200 但完全无约束，零警告；去掉该参数或 prompt 后缀 think 块即受约束，chat 端点不受影响。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/57726
- **关联**：var:INF-0956

### INF-0990 `vllm_outlines_backend_finished_guide_rejects_allowed_eos_500_every_completed_request` vLLM outlines 后端拒收终止 EOS 致约束请求 HTTP 500

`故障` · `active` · `infra_off` · `config` · `verified` · `NEW` · 发现 2026-09-19 · 旧号 10.74

- **原因对象**：解码/采样 / vllm / outlines 后端 accept_tokens
- **原因行为**：outlines Guide 完成后拒绝一切 token，而 bitmask 终止步只放行 EOS；封装层仅 is_terminated 侧容忍，accept_tokens([eos]) 返回 False 被视作拒绝。
- **影响对象**：可用性/进度（报错）
- **影响详情**：outlines 后端下自然 emit EOS 完成的约束请求均返回 HTTP 500：180 秒 38,589 请求中 25,425 个 500（66%），全部 choice 与自然完成的 regex 失败；先撞 max_tokens 的存活；guidance/xgrammar 0 错。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/57723

### INF-1004 `vllm_xgrammar_structural_tag_nested_json_schema_skips_unsupported_feature_check_constraints_silent` vLLM structural tag 嵌套 schema 跳过不支持特性检查

`故障` · `active` · `infra_off` · `config` · `verified` · `var` · 发现 2026-09-27 · 旧号 10.88

- **原因对象**：解码/采样 / vllm / validate_xgrammar_grammar
- **原因行为**：校验按顶层约束类型分派，structural_tag 分支只编译 tag 本身，不对嵌套 json_schema 做 unsupported-feature 检查，本应回退 guidance/outlines 的请求留在 xgrammar。
- **影响对象**：输出内容（静默错误）
- **影响详情**：自供 structural_tag、strict tool 参数、Harmony（gpt-oss）response_format 三条路径中，含 multipleOf 等特性的 schema 通过验证却不被强制，输出可违反 schema；同 schema 走 plain json 会被拒绝。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/58930
- **关联**：var:INF-0956

### INF-1007 `vllm_thinking_budget_state_holder_stale_state_leak_and_row0_collapse_under_spec` vLLM thinking budget 状态容器行映射塌缩与陈旧态泄漏

`故障` · `active` · `infra_off` · `config` · `verified` · `var` · 发现 2026-09-29 · 旧号 10.91

- **原因对象**：解码/采样 / vllm / ThinkingBudgetStateHolder
- **原因行为**：(a) 零 draft 步 cu_num_tokens 全 0，各请求的 forced </think> 都写进 logits row 0；(b) sync_batch 单向 move 仅在源有 state 时清目标，残留前任 state。
- **影响对象**：输出分布；跨请求/跨样本数据（静默错误）
- **影响详情**：thinking_token_budget 批内：请求 0 被插入别人的 </think>，其余请求 budget 不再强制、超预算生成；槽位复用时无 budget 的新请求被按前任预算强插 </think> 截断推理。无任何报错。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/59272；https://github.com/vllm-project/vllm/issues/59273
- **关联**：var:INF-1109

### INF-1009 `vllm_xgrammar_combinator_beside_constraint_keywords_silent_unenforced` vLLM auto 后端漏检组合子旁约束关键字，xgrammar 静默不强制

`故障` · `active` · `infra_off` · `config` · `documented` · `var` · 发现 2026-10-09 · 旧号 10.93

- **原因对象**：解码/采样 / vllm / 结构化输出 auto 后端选择谓词
- **原因行为**：auto 后端选择的检查只覆盖 allOf≥2，漏掉组合子（anyOf/oneOf/单支 allOf）与 type/properties/required 同节点的 schema，仍交给会丢邻近关键字的 xgrammar。
- **影响对象**：输出内容（静默错误）
- **影响详情**：此类 schema 在 xgrammar 下编译为等价 {} 的 any-JSON 规则，实测接受全部 9 个非法实例，非法 JSON 以 200 返回且无错误；本应回退的 guidance 后端则拒绝全部。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/60237
- **关联**：var:INF-0980

### INF-1019 `vllm_xgrammar_lets_control_tokens_into_json_strings_muse_glimmer_truncates_answer` vLLM xgrammar 放行控制 token 进 JSON 字符串致答案截断

`故障` · `active` · `infra_off` · `config` · `documented` · `var` · 发现 2026-10-09 · 旧号 10.103

- **原因对象**：解码/采样 / vllm / xgrammar 字符串域与控制 token
- **原因行为**：xgrammar 后端的 grammar 未把 <|eom|>、<|start|>、<|message|> 等保留控制 token 排除出 JSON 字符串域；muse_glimmer tool parser 遇到这些 token 即截断。
- **影响对象**：输出内容（静默错误）
- **影响详情**：使用 xgrammar 与 muse_glimmer parser 时，模型在 JSON 字符串内生成控制 token 后答案被静默截断，无报错。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/60580
- **关联**：var:vllm_xgrammar_pattern_minlength_drops_length_bound

### INF-1025 `degeneration_repetition_loop` greedy/beam/低温解码坍缩进重复 n-gram 循环

`故障` · `active` · `model_on` · `config` · `verified` · `NEW` · 发现 待定 · 旧号 11.1

- **原因对象**：解码/采样 / 通用 / greedy/beam/低温解码策略
- **原因行为**：似然最大化的 greedy、beam 或低温采样会坍缩进高似然的重复 n-gram 循环并自我强化；问题在采样策略的选择环节，不在模型参数或前向计算。
- **影响对象**：输出内容（质量退化）
- **影响详情**：输出在 token/短语/句子级别重复打结，一直持续到 max_tokens；残差流保持健康，只在输出 token 轨迹上（n-gram 重复导致熵坍缩）可见。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://arxiv.org/abs/1904.09751

### INF-1026 `repetition_penalty_on_input_ids` transformers 重复惩罚误作用于 prompt token

`故障` · `active` · `model_on` · `config` · `verified` · `NEW` · 发现 待定 · 旧号 11.2

- **原因对象**：解码/采样 / transformers / repetition penalty 处理
- **原因行为**：enforce_repetition_penalty_（源自 CTRL）作用在完整 input_ids（prompt+已生成）上，而非只作用于已生成 token，每个 prompt token 都被乘性惩罚。
- **影响对象**：输出分布；输出内容（质量退化）
- **影响详情**：prompt 中出现的 token 在生成中被压制，表现为奇怪的同义词替换、实体缺失、话题漂移；隐藏态健康，只在最后一层 logits 上可见 prompt token 被乘性压低。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://arxiv.org/abs/1909.05858

### INF-1027 `specdec_acceptance_buggy` 推测解码接受逻辑缺陷使输出偏离 target 分布

`故障` · `active` · `ambiguous` · `config` · `documented` · `NEW` · 发现 待定 · 旧号 11.3

- **原因对象**：解码/采样 / vllm / 推测解码 acceptance 逻辑
- **原因行为**：推测解码的 acceptance 判定有缺陷：按 buggy 的接受准则（off-by-one 或错误 clamping）重采样输出 token 分布。
- **影响对象**：输出分布；输出内容（静默错误）
- **影响详情**：输出流畅可信，但聚合分布与未加速的 target 模型偏离，greedy 下偶尔给出不同 token；是否显形取决于草稿与 target 的一致程度及输入，签名在 acceptance-rate 分布与 n-gram 统计上。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://arxiv.org/abs/2401.10774

### INF-1028 `stop_string_token_boundary_miss` StopStringCriteria 漏检跨 token 的多字节停止串

`故障` · `active` · `model_on` · `config` · `verified` · `NEW` · 发现 待定 · 旧号 11.4

- **原因对象**：解码/采样 / transformers / StopStringCriteria
- **原因行为**：只检查新生成的 token 是否在 token 边界上恰好完成 stop string；byte-level BPE 下多字节/CJK stop string 被切散到多个 byte-fragment token，匹配器永远看不到。
- **影响对象**：输出内容（静默错误）
- **影响详情**：多字节/CJK stop string（如 Qwen2.5）不触发停止，生成越过应停位置继续追加多余文本；无隐藏态签名，可由 EOS 邻近度、答案后是否续写等 token 流统计发现。
- **改进措施**：检测 待定；修复 待定
- **来源**：引用：HF transformers #46519 (Qwen2.5)；

### INF-1029 `logit_bias_id_mismatch` logit_bias token id 错配扭曲采样分布

`故障` · `active` · `model_on` · `config` · `documented` · `NEW` · 发现 待定 · 旧号 11.5

- **原因对象**：解码/采样 / 通用 / logit_bias logits 处理
- **原因行为**：隐藏态健康，post-softmax 分布被扭曲。
- **影响对象**：输出分布；输出内容（静默错误）
- **影响详情**：隐藏态健康，post-softmax 分布被扭曲，表现为采样偏好异常、某些 token 被无故压制或抬升；签名只在 logits 空间，logits/输出分布级统计检查可发现。
- **改进措施**：检测 待定；修复 待定
- **来源**：引用：CTRL / OpenAI docs；

### INF-1030 `beam_search_hyperparam_sensitivity` beam search 对剪枝超参敏感导致输出改变

`故障` · `active` · `model_on` · `config` · `documented` · `NEW` · 发现 待定 · 旧号 11.6

- **原因对象**：解码/采样 / 通用 / beam search 剪枝超参
- **原因行为**：beam search 对 length penalty、repetition、early-stopping、patience factor 等超参敏感，取值变化会改变最终保留的 beam。
- **影响对象**：输出内容（静默错误）
- **影响详情**：前向计算无异常、隐藏态与健康基线不可区分，超参取值变化时最终输出随之改变；只能靠不同超参下的输出分歧或输出级统计发现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://arxiv.org/abs/2204.05424

### INF-1031 `sglang_nan_logits_uniform_random_sample_200` SGLang 全 NaN logits 经 sanitize 后变均匀随机采样

`故障` · `active` · `infra_off` · `other_stack` · `verified` · `NEW` · 发现 2026-08-01 · 旧号 11.7

- **原因对象**：解码/采样 / sglang / NaN logits sanitize 路径
- **原因行为**：启用 SGLANG_SANITIZE_NAN_LOGITS 后，整行 NaN logits 被 nan_to_num_(nan=-1e30) 映射为同一常量，softmax 后成全词表均匀分布，采样器抽取纯随机 token。
- **影响对象**：输出分布；输出内容；KV状态（静默错误）
- **影响详情**：整行 logits 为 NaN（如 Kimi-K3 瞬态 NaN）时以 HTTP 200 流式返回多语言 BPE 拼盘，P(EOS)≈1/163,840 几乎不终止直到 max_tokens；NaN 步写入的 KV 使该请求后续步骤全被污染。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/33187

### INF-1032 `sglang_rocm_spec_decode_verify_silent_greedy_forced_temp0` SGLang ROCm 推测解码验证静默退化为贪婪

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-08-01 · 旧号 11.8

- **原因对象**：解码/采样 / sglang / eagle_sample 门控、ROCm 构建
- **原因行为**：eagle_sample 对 _is_hip 等平台一律走 torch.argmax 贪婪验证；因分布保持所需内核未在 ROCm sgl-kernel 构建中编译（setup_rocm.py 缺失），代码选择静默退化而非报错。
- **影响对象**：输出分布；输出内容（静默错误）
- **影响详情**：ROCm 上启用推测解码后，所有 HIP 请求无论 temperature/top_p/top_k 都按贪婪输出，相当于强制 temperature=0，非贪婪采样设置静默无效；无 crash、无 NaN。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/pull/32922

### INF-1033 `trtllm_samplingparams_nan_top_p_min_p_temperature` TRT-LLM SamplingParams 校验放行 NaN 采样参数

`故障` · `active` · `infra_off` · `other_stack` · `verified` · `NEW` · 发现 2026-08-01 · 旧号 11.9

- **原因对象**：解码/采样 / tensorrt-llm / SamplingParams._validate
- **原因行为**：_validate 对 top_p、min_p、temperature 只用 <0、>1 正向范围检查，NaN 的比较结果均为 False，NaN 值因此通过校验进入采样逻辑。
- **影响对象**：输出分布；输出内容（静默错误）
- **影响详情**：请求传入 NaN 的 top_p/min_p/temperature 时不被拒绝，采样行为未定义，可静默产生异常 token 分布，不必然 crash。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/NVIDIA/TensorRT-LLM/issues/17158

### INF-1039 `vllm_dspark_acceptance_collapse_after_pos0_dsv4_flash` vLLM DSPark 推测解码 position 1+ 接受率链式崩溃

`故障` · `active` · `infra_off` · `yes` · `documented` · `NEW` · 发现 2026-08-04 · 旧号 11.15

- **原因对象**：解码/采样 / vllm / DSPark 推测解码草稿链
- **原因行为**：根因未定位。怀疑：parallel backbone 的 position 1+ 预测未以中间 draft token 为条件；_sample_sequential 给 markov head 的 prev/映射有误；mtp 权重命名交互。
- **影响对象**：性能/资源（性能退化）
- **影响详情**：DeepSeek-V4-Flash-0731 + 2×DGX Spark（GB10 SM121a）+ 5 个推测 token：position 0 接受率正常，position 1+ 条件接受率约 25%、3–4 位降为 0%，平均接受长度仅 1.46，加速几乎归零，输出仍正确。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/51009

### INF-1041 `vllm_cpu_seed_ignored_when_batch_has_unseeded_request` vLLM CPU 后端混合 batch 中 seed 不生效

`故障` · `active` · `infra_off` · `yes` · `documented` · `NEW` · 发现 2026-08-06 · 旧号 11.17

- **原因对象**：解码/采样 / vllm / CPU 采样路径 RNG 状态管理
- **原因行为**：CPU 采样路径在 batch 同时含 seed 与无 seed 请求时 RNG 状态管理错误，seeded 请求的随机性受同 batch 其他请求采样参数影响。
- **影响对象**：输出内容；跨请求/跨样本数据（不可复现）
- **影响详情**：CPU 后端同 batch 中一个请求 seed=42、另一个无 seed 时，seeded 请求每次输出不同；给另一个也加 seed 后即可复现。无 crash、无警告（vLLM 0.26.0+cpu）。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/51226

### INF-1042 `sglang_deterministic_sampling_rejects_min_p_with_seed` SGLang min_p 与 seed 组合被采样器 assert 拒绝

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-08-06 · 旧号 11.18

- **原因对象**：解码/采样 / sglang / PyTorch 采样路径 min_p+seed
- **原因行为**：SamplingParams 接受 min_p 与确定性采样 seed 的组合，但 PyTorch 采样路径未实现 min_p 过滤后的 seeded multinomial，以 assert 拒绝而非优雅降级。
- **影响对象**：可用性/进度（报错）
- **影响详情**：同时设置 min_p 与 deterministic sampling seed 的请求触发 AssertionError（With sampling seed, multinomial_with_seed will provide wrong results），配置层接受而采样层拒绝。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/33695

### INF-1043 `sglang_spec_decode_penalty_only_one_token_per_step` SGLang 推测解码下惩罚每步只累积一个 token

`故障` · `active` · `infra_off` · `other_stack` · `verified` · `NEW` · 发现 2026-08-06 · 旧号 11.19

- **原因对象**：解码/采样 / sglang / ScheduleBatch 惩罚累积
- **原因行为**：推测解码每 step 提交 accept_len 个 token，但 cumulate_penalty_output_tokens 只把 output_ids[-1] 喂给 penalizer，其余 token 不进入惩罚状态且永不回填。
- **影响对象**：输出分布；输出内容（静默错误）
- **影响详情**：平均接受长度为 L 时，frequency_penalty 有效强度约降为 1/L；min_new_tokens 按 step 计数使 EOS 被多屏蔽约 L 倍；presence/repetition_penalty 延迟激活。无 crash、无 NaN。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/pull/33643

### INF-1045 `sglang_seeded_sampling_empty_distribution_token0` SGLang seeded 采样遇空分布静默输出 token 0

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-08-07 · 旧号 11.21

- **原因对象**：解码/采样 / sglang / multinomial_with_seed
- **原因行为**：multinomial_with_seed 对 log 概率加确定性 Gumbel 噪声后取 argmax；整行为 -inf 时 argmax 返回 index 0，未像 torch.multinomial 那样拒绝空分布。
- **影响对象**：输出内容（静默错误）
- **影响详情**：seeded 采样遇到整行 log 概率全为 -inf 的空分布时，静默产出 token ID 0（如 BOS/padding）而非请求级失败，客户端看到无意义 token。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/33697

### INF-1046 `sglang_dflash_dspark_swa_eviction_unreachable_kill` SGLang DFLASH/DSPARK 漏调 SWA 驱逐致请求中途被撤回

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-08-08 · 旧号 11.22

- **原因对象**：解码/采样 / sglang / DFlashDraftInputV2 解码准备
- **原因行为**：DFLASH/DSPARK 的 prepare_for_decode 从不调用 maybe_evict_swa，也不推进 decode_batch_idx（驱逐门要求 ≥1），而 EAGLE 路径两者都做，SWA 驱逐因此不可达。
- **影响对象**：可用性/进度；性能/资源（报错）
- **影响详情**：hybrid-SWA 模型 + DFLASH/DSPARK 长生成时，SWA KV 逐 token 保留，约 swa_full_tokens_ratio×max_total_num_tokens 个 token 后池耗尽，请求被中途 retract；实测 77,824 token 处触发。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/pull/33805

### INF-1047 `sglang_dspark_compact_silent_block_size_ignore` SGLang DSpark compact 路径静默忽略 block size

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-08-08 · 旧号 11.23

- **原因对象**：解码/采样 / sglang / DSpark compact verify 路径
- **原因行为**：compact/confidence verify 路径用 checkpoint 原生 gamma 而非运行时 draft window 来 view 张量，静默忽略 --speculative-dspark-block-size。
- **影响对象**：可用性/进度；性能/资源（报错）
- **影响详情**：设置非 checkpoint 原生 block size 时 decode-graph capture 形状错误而崩溃；原生深度下参数被静默锁定，而 DSV4-Flash 原生深度 5 恰是 SM120 上已损坏的深度，两 bug 互相掩盖。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/pull/34023

### INF-1048 `sglang_dspark_acc_additive_penalties_field_mismatch` SGLang DSPARK 读取错误的惩罚字段名

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-08-08 · 旧号 11.24；11.30

- **原因对象**：解码/采样 / sglang / DSPARK verify 惩罚字段读取
- **原因行为**：dflash_utils 与 dspark_verify 读取不存在的 acc_linear_penalties（应为 acc_additive_penalties），恒得 None，加性惩罚分支与 no-op 判断无法经此字段生效。
- **影响对象**：输出分布；输出内容（静默错误）
- **影响详情**：DSPARK 请求设 min_new_tokens=50 时返回 HTTP 200、finish_reason=stop、completion_tokens=18；该症状经哪条惩罚分支传播未经运行期证实，且存在 fallback 分支，不能断言所有惩罚请求都被忽略。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/33493

### INF-1050 `vllm_sampler_logprob_token_ids_shared_batch_silent_neg_inf` vLLM V1 Sampler 混批致其他请求 logprob 变 -inf

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-08-11 · 旧号 11.26

- **原因对象**：解码/采样 / vllm / V1 Sampler.forward
- **原因行为**：logprob_token_ids 请求与 logprobs=k 请求同批时，Sampler 用指定 token 的 gather 结果替换整个 batch 的 top-k 结果，缺少按请求隔离。
- **影响对象**：输出分布；跨请求/跨样本数据（静默错误）
- **影响详情**：两个并发请求即可触发：其他请求返回的 top-k logprob 变为 -inf 行（API 中为单个 <pad>: -9999.0 条目），生成文本不受影响，完全静默；仅 V1 sampler，V2 runner 已按请求合并。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/51789

### INF-1052 `trtllm_spec_decode_interior_control_token_rejection_silent_toolcall_drop` TRT-LLM 推测解码接受路径截断控制块致 tool call 静默丢失

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-08-11 · 旧号 11.28

- **原因对象**：解码/采样 / tensorrt-llm / 推测解码 acceptance 路径
- **原因行为**：单模型推测解码的 acceptance 路径只返回 accepted_tokens 与数量并直接写入 new_tokens，不检查 accepted token 流的结构完整性，也无下游 rollback。
- **影响对象**：输出内容（静默错误）
- **影响详情**：draft 提议 tool-call 控制前导而 target 拒绝其中某个内部 control token 时，控制块打开后永不闭合，返回空 tool_calls、空 content、finish_reason=stop、HTTP 200；非推测解码下同一 draft 产出完整调用。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/NVIDIA/TensorRT-LLM/issues/17437

### INF-1053 `ttmetal_ttsampling_topk_iota_half_width_wrong_tokens` TTSampling top-k 索引宽度减半返回错误 token

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-08-12 · 旧号 11.29

- **原因对象**：解码/采样 / tt-metal / TTSampling top-k iota 索引
- **原因行为**：decode_forward 传给 ttnn.topk 的 iota indices_tensor 宽度只有 logits（padded vocab）的一半，多核 top-k 因此从截断的 logits 行中选 token。
- **影响对象**：输出内容（静默错误）
- **影响详情**：temperature≠0 或 p∉{0,1} 且 vocab≥32768 时触发：Mistral-7B Top-1 从 96.4% 降到 83.2%，vocab 32768 时 256 token 错 126 个，部分超出词表；CI 只走 argmax 路径而未暴露。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/52928

### INF-1054 `hf_spec_decode_candidate_generator_q_distribution_mismatch` transformers 推测解码 q 分布错用 raw logits

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-08-13 · 旧号 11.31

- **原因对象**：解码/采样 / transformers / block drafter 候选生成器
- **原因行为**：DFlash 等三个 block drafter 从 processor 处理后的 logits 采样，却返回 raw candidate_logits，被 _speculative_sampling 当作 q。
- **影响对象**：输出分布（静默错误）
- **影响详情**：do_sample=True 且配置 logits processor 时，acceptance test 基于错误的 q 运行，推测解码失去无损性保证，输出分布静默偏离理论分布；无 crash。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/huggingface/transformers/issues/47932

### INF-1056 `sglang_npu_mtp_conv_window_front_end_layout_mismatch_corrupts_verify` SGLang NPU MTP conv 窗口布局错配致输出损坏

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-08-14 · 旧号 11.33

- **原因对象**：解码/采样 / sglang / NPU GDN verify conv 窗口布局
- **原因行为**：Ascend NPU 的 causal_conv1d 恒从 conv_states 的 FRONT 读 base conv 窗口，sglang 推测 verify 路径却按 END 布局写入，GDN verify 读到错误窗口。
- **影响对象**：输出内容；KV状态（静默错误）
- **影响详情**：Ascend 910B3 + NEXTN/MTP + GDN/线性注意力模型时，部分 prompt 持续答错（如 2+2 输出乱码），关闭 MTP 即正确；损坏随接受 token 数呈概率性变化，无 crash。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/pull/34794

### INF-1059 `vllm_async_scheduling_stateful_logits_processor_minus1_placeholder` vLLM 异步调度下有状态 logits processor 读到 -1

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-08-15 · 旧号 11.36

- **原因对象**：解码/采样 / vllm / logits processor 注册统计
- **原因行为**：async scheduling 下 output_tok_ids 用 -1 占位，修复机制的启用判据 logitsprocs_need_output_token_ids 只统计 CLI 注册的 processor，漏计代码注册的。
- **影响对象**：输出分布；输出内容（静默错误）
- **影响详情**：async scheduling（0.27 起默认开启）下，经 API/代码注册、依赖生成历史的 stateful logits processor（如 presence penalty 变体、自定义 ban-list）持续读到 -1 占位符并据此决策，采样语义静默偏离；无 crash。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/52461

### INF-1060 `vllm_spec_decode_structured_output_commit_before_grammar_validation` vLLM 推测解码在 grammar 校验前提交 token 破坏结构化输出

`故障` · `active` · `infra_off` · `yes` · `documented` · `NEW` · 发现 2026-08-15 · 旧号 11.37

- **原因对象**：解码/采样 / vllm / scheduler 推测 token 提交
- **原因行为**：scheduler 在 StructuredOutputGrammar.accept_tokens() 校验 accepted block 之前就把它追加进请求历史，破坏历史与 grammar 状态沿同一最长合法前缀推进的提交不变量。
- **影响对象**：输出内容；可用性/进度（静默错误）
- **影响详情**：推测解码 + 结构化输出在推理结束延迟激活 grammar 或 grammar 状态迁移边界处，请求历史与 grammar 状态失同步，输出可违反 schema，被拒后缀无法重新采样，表现为卡死或非法 JSON；DS4-Flash 复现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/52452

### INF-1061 `vllm_dspark_adaptive_verification_zero_draft_budget_grammar_mask_crash` vLLM DSpark 零草稿预算致 grammar mask 断言崩溃

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-08-15 · 旧号 11.38

- **原因对象**：解码/采样 / vllm / grammar bitmask 映射
- **原因行为**：structured outputs 的 grammar bitmask 映射未定义 draft_budget=0 的情形，而 DSpark 自适应验证控制器在收益不覆盖验证成本时会常规选择零草稿预算。
- **影响对象**：可用性/进度（报错）
- **影响详情**：启用 enable_adaptive_verification 的 structured outputs 请求，在 drafter 置信度低到控制器选零 draft 预算的瞬间触发 assert num_masks == len(mapping)，服务崩溃。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/52436

### INF-1062 `sglang_dspark_forced_reject_gdn_state_drift_divergence` SGLang DSpark 强制拒绝下 GDN 状态仍累积漂移

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-08-17 · 旧号 11.39

- **原因对象**：解码/采样 / sglang / DSpark verify GDN 状态读写
- **原因行为**：DSpark verify 路径对 GDN 循环状态的读写/回滚存在精度问题，即使强制拒绝全部 draft、只提交 target argmax，循环状态仍逐步累积漂移。
- **影响对象**：KV状态；输出内容（静默错误）
- **影响详情**：GDN/Mamba hybrid 模型 + DSpark，γ=1 强制拒绝全部 draft 时仍在后期与 Base decode 确定性发散，差异随上下文增长放大；状态提到 FP32 只推迟不消除；逐 token 对齐校验不可见，无 crash、无 NaN。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/35150

### INF-1063 `vllm_dspark_breakable_graph_concurrent_reasoning_loop` vLLM DSpark 可中断图并发致 reasoning 死循环

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-08-17 · 旧号 11.40

- **原因对象**：解码/采样 / vllm / DSpark 可中断 CUDA graph 交互
- **原因行为**：DSpark 推测解码与 breakable CUDA graph（nightly 默认开启）在高并发下交互出错，使真实 decode 循环无法离开 reasoning 通道；原文未定位到具体代码。
- **影响对象**：输出内容（静默错误）
- **影响详情**：DSpark + breakable CUDA graph + 并发≥10 时部分请求 reasoning 紧循环至 max_tokens，finish_reason=length、空 content；--enforce-eager 不复现，SGLang/TRT-LLM 同 prompt 无失败。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/52448

### INF-1064 `vllm_ngram_gpu_spec_decode_xgrammar_concurrent_500` vLLM ngram_gpu 推测解码与 xgrammar 并发返回 500

`故障` · `active` · `infra_off` · `partial` · `documented` · `var` · 发现 2026-08-17 · 旧号 11.41

- **原因对象**：解码/采样 / vllm / ngram_gpu grammar mask 映射
- **原因行为**：ngram_gpu 推测解码的 draft/verify 循环与 xgrammar 的 grammar mask 位分配在并发下发生映射冲突。
- **影响对象**：可用性/进度（报错）
- **影响详情**：ngram_gpu 推测解码 + xgrammar 结构化输出并发请求时以 HTTP 500 失败（请求级，非静默）；与 DSpark 零 draft 预算 grammar mask assert 同族。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/52620
- **关联**：var:INF-1061

### INF-1066 `sglang_inkling_mtp_draft_extend_graph_capture_crash` SGLang Inkling MTP draft-extend 图捕获崩溃

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-08-17 · 旧号 11.43

- **原因对象**：解码/采样 / sglang / Inkling MTP draft-extend
- **原因行为**：Inkling 模型 MTP draft 结构的动态形状/状态与 draft-extend CUDA graph capture 的静态化要求冲突。
- **影响对象**：可用性/进度（报错）
- **影响详情**：SGLang 上 Inkling 模型启用 MTP 并做 CUDA graph capture 时，draft-extend 捕获阶段崩溃。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/35039

### INF-1069 `sglang_dspark_verify_identifier_grounded_token_corruption` SGLang DSPARK verify 静默篡改 prompt 中的标识符

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-08-18 · 旧号 11.46

- **原因对象**：解码/采样 / sglang / DSPARK verify/提交路径
- **原因行为**：DSPARK 推测解码的 verify/提交路径非无损，存在 token 级替换，替换率随 verify 窗口加宽而上升，与采样随机性和循环状态无关；原文未定位到具体代码。
- **影响对象**：输出内容（静默错误）
- **影响详情**：DeepSeek-V4-Flash + DSPARK 下 prompt 原样给出的 identifier 被改写（kimi-k3→kimi-kw）；temp 0 时 block 6 损坏率 1.03%、block 5 为 0.32%、block 1 为 0，关闭推测 0/4561；输出流畅，易漏检。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/34959

### INF-1070 `vllm_mtp_maxlen_skip_zero_draft_ids_overlimit` vLLM MTP 超限 skip 返回全零 draft ID

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-08-18 · 旧号 11.47

- **原因对象**：解码/采样 / vllm / MTP drafter 超限 skip 路径
- **原因行为**：序列超过 drafter max_model_len 时，skip 路径返回全零 token ID 而非零条 draft，scheduler 当作真实 draft 送 verify；资格检查还用 batch 级 max_seq_len。
- **影响对象**：性能/资源；指标/日志（性能退化）
- **影响详情**：上下文超过 drafter 上限（如 115k 输入 vs 80k 限制）时输出正确，但 verify 恒拒绝，SpecDecoding 指标报 0% 接受、易被误读为 drafter 质量差；同 batch 内短请求也被一并停掉推测。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/52756

### INF-1071 `vllm_mtp_structural_tag_grammar_matcher_post_termination` MTP 推测下 grammar matcher 终止后仍被推进

`故障` · `active` · `infra_off` · `partial` · `documented` · `var` · 发现 2026-08-18 · 旧号 11.48

- **原因对象**：解码/采样 / vllm / 推测解码路径的 matcher 推进
- **原因行为**：structural tag 与 MTP 推测同时开启时，matcher 接受 stop token 终止后仍被喂 token；#44297 的 reasoning 边界 trim 未覆盖推测解码下推进 matcher 的全部路径
- **影响对象**：性能/资源；指标/日志（性能退化）
- **影响详情**：strict:true 或 tool_choice 非 auto/none 且开 MTP 时 grammar_matcher.cc:612 反复告警（tag+MTP 21 次，关 MTP 0 次）；tool call 结果 3/3 正确，仅日志噪声与无谓 matcher 工作，更大负载未验证
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/52767
- **关联**：var:INF-1060

### INF-1072 `sglang_dspark_dsv4_sampling_nan_weekly_device_assert` SGLang DSPARK 采样概率 NaN 致全 rank 断言崩溃

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `var` · 发现 2026-08-18 · 旧号 11.49

- **原因对象**：解码/采样 / sglang / DSPARK 推测解码路径（疑）
- **原因行为**：根因未定位：DSPARK + DeepSeek-V4-Flash 下 multinomial 的概率张量间歇含 inf/nan/负值；原文指向 DSPARK verify/draft 路径的共性 NaN 注入面
- **影响对象**：输出分布；可用性/进度（报错）
- **影响详情**：8×H100、sglang 0.5.16 生产流量约每周一次 CUDA device-side assert（TensorCompare.cu 的 multinomial 异步检查），8 个 TP rank 同时失败、整服退出；无稳定复现，与长上下文/并发的相关性未隔离
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/35324
- **关联**：var:INF-1291

### INF-1073 `vllm_rejection_sampler_nan_row_laundered_oov_token_id` vLLM rejection sampler 将 NaN 行洗成越界 id

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-08-20 · 旧号 11.50

- **原因对象**：解码/采样 / vllm / rejection sampler 重采样内核
- **原因行为**：verify 行 logits 全 NaN 时，_rejection_resample_kernel 的 residual 与 gumbel 噪声仍为 NaN，argmax 链产出等于 vocab_size 的越界 id，不报错也不回退
- **影响对象**：输出内容；可用性/进度（报错）
- **影响详情**：越界 id 经 input_ids 传播，1–3 步后以 indexSelectSmallIndex device assert 崩溃，异步浮现于 copy_event.synchronize()，远离根源、误导排查；引擎 NaN 检测器看不到 verify 行
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/53029

### INF-1074 `vllm_prompt_logprobs_drafter_graph_buffer_reuse_corruption` vLLM prompt_logprobs 静默取自 draft 模型

`故障` · `active` · `infra_off` · `partial` · `verified` · `NEW` · 发现 2026-08-25 · 旧号 11.51

- **原因对象**：解码/采样 / vllm / drafter 与 logprobs 计算执行序
- **原因行为**：drafter 在 _bookkeeping_sync() 前运行，复用共享 CUDA graph 内存池，覆写 target hidden_states 所在 buffer，prompt logits 再从中计算
- **影响对象**：输出分布（静默错误）
- **影响详情**：spec decode + piecewise CUDA graph 下请求 prompt_logprobs：不超 capture 限长的 prompt 得到 draft 模型的 logprobs，超出（eager）正确；generation 不受影响，无 crash、无 NaN、无日志
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/53520

### INF-1076 `vllm_dflash2_xgrammar_fsm_deterministic_draft_token_reject_retry_storm` vLLM DFlash2 结构化输出下 draft 确定性被拒重试风暴

`故障` · `active` · `infra_off` · `partial` · `documented` · `var` · 发现 2026-08-26 · 旧号 11.53

- **原因对象**：解码/采样 / vllm / DFlash2 draft 与 FSM 联动
- **原因行为**：DFlash2 在 grammar mask 下 draft 出 FSM 当前态不允许的 token 271；重试时 greedy drafter 确定性给出同一 token，drafter 分布未与 mask 联动，形成拒绝-重试循环
- **影响对象**：性能/资源；指标/日志（性能退化）
- **影响详情**：xgrammar response_format json_object 下固定 prompt 3/3 复现：单请求最多 7 个重试波、每波 2 条 Failed to advance FSM ERROR；HTTP 200、JSON 有效，但尾延迟升高、ERROR 日志淹没告警
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/53777
- **关联**：var:INF-1064

### INF-1080 `vllm_stale_draft_handoff_grammar_bitmask_fail_open` vLLM draft 回传丢失致 grammar mask 全放开

`故障` · `active` · `infra_off` · `partial` · `verified` · `NEW` · 发现 2026-08-30 · 旧号 11.57

- **原因对象**：解码/采样 / vllm / draft 回传槽与 bitmask 构建
- **原因行为**：async scheduler 装入 [-1]*K 占位，回填经无请求身份、每步覆写的单槽 DraftTokensHandler，竞态下占位残留；mask 构建把 -1 行写成全允许，而验证侧用 GPU 副本的真实 drafts
- **影响对象**：输出内容（静默错误）
- **影响详情**：spec + async 调度 + 结构化输出下，最多 num_speculative_tokens 个 token 不受约束：碰巧合法则静默带毒继续，不合法则下游 FSM 报错终止受害请求；main 上潜伏，PP 能力 drafter 落地即触发
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/54437

### INF-1081 `vllm_cpu_spec_decode_fp64_gumbel_flag_silent_downcast_fp32` vLLM CPU 后端静默忽略 --use-fp64-gumbel

`故障` · `active` · `ambiguous` · `yes` · `verified` · `var` · 发现 2026-09-01 · 旧号 11.58

- **原因对象**：解码/采样 / vllm / CPU shim 的 recovered 采样
- **原因行为**：CPU fallback shim 接受 USE_FP64_GUMBEL 后在 argmax 前把 inv_q 降转为 float32，flag 不生效；上方注释称按原 dtype 读取，与代码相反
- **影响对象**：输出分布；输出内容（静默错误）
- **影响详情**：CPU 后端 + spec decode recovery 采样 + --use-fp64-gumbel：flag 开关输出逐位相同、无警告；q≈1e-40 时 inv_q 降转溢出为 inf，微概率 token 窃取 argmax，与 GPU Triton 内核同 draw 发散
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/54843
- **关联**：var:INF-0329

### INF-1082 `sglang_dflash_target_layer_ids_fallback_guess_count_collision_silent` SGLang DFlash fallback 静默猜错 tap 层

`故障` · `active` · `infra_off` · `other_stack` · `verified` · `var` · 发现 2026-09-01 · 旧号 11.59

- **原因对象**：解码/采样 / sglang / DFlash 目标层 fallback 公式
- **原因行为**：draft config 缺 target_layer_ids 时 fallback 用等距公式猜 tap 层；tap 层是训练期决定的，猜出的层数相同但层号错（[1,16,31,46,61] vs [5,19,33,47,61]）
- **影响对象**：性能/资源（性能退化）
- **影响详情**：draft config 缺 dflash_config.target_layer_ids 时权重干净加载、形状检查通过、零报错，drafter 读错目标层，acceptance 系统性偏低、吞吐受损，唯一症状是 accept rate；20 个公开 checkpoint 中 14 个若触发会猜错
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/37475
- **关联**：var:INF-1064

### INF-1083 `vllm_speculative_kv_dtype_auto_flashinfer_startup_native_crash` vLLM draft KV dtype 透传 auto 致启动崩溃

`故障` · `active` · `infra_off` · `partial` · `documented` · `var` · 发现 2026-09-01 · 旧号 11.60

- **原因对象**：解码/采样 / vllm / draft KV dtype 归一化缺失
- **原因行为**：_create_draft_vllm_config 把未指定的 draft KV dtype 以 auto 下传，FlashInfer 后端 dtype 表无 auto 映射；第二级 native crash 原因未定位
- **影响对象**：可用性/进度（报错）
- **影响详情**：spec 配置设 kv_cache_dtype fp8 即报 ValueError: Unrecognized dtype: auto；显式设 target KV dtype 后，SM89 + hybrid GDN + DFlash2 在 attention init 无 traceback 崩溃
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/54690
- **关联**：var:vllm_compressed_tensors_kv_scale_dropped

### INF-1084 `vllm_mrv2_mtp_thinking_token_budget_ignored_reasoning_runaway` vLLM V2 runner+MTP 下 thinking 预算被忽略

`故障` · `active` · `infra_off` · `config` · `verified` · `NEW` · 发现 2026-09-04 · 旧号 11.61

- **原因对象**：解码/采样 / vllm / MTP 循环缺 thinking 预算检查
- **原因行为**：thinking budget 的截断逻辑只挂在标准 decode 路径；V2 runner 的 MTP 多 token draft/verify 循环绕开该 hook，逐 token 预算检查没有插入点
- **影响对象**：输出内容；性能/资源（静默错误）
- **影响详情**：V2 runner + MTP + thinking_token_budget：Qwen3.8-27B NVFP4 请求把 completion 配额全当推理烧完、不产出 </think> 或正文；去掉 spec 则预算生效；无 crash，成本与延迟放大到 max_tokens
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/54906

### INF-1085 `vllm_qwen_mtp_npredict_none_multimodal_wrapper_config_scope` 多模态 wrapper 下 MTP n_predict 解析为 None

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-09-04 · 旧号 11.62

- **原因对象**：解码/采样 / vllm / MTP n_predict 配置解析作用域
- **原因行为**：多模态 wrapper checkpoint 的 config 为嵌套结构，mtp_num_hidden_layers 从 wrapper 顶层而非 text_config 读取，顶层无此键，n_predict 解析为 None
- **影响对象**：性能/资源（性能退化）
- **影响详情**：Qwen3.5/3.6 多模态 wrapper checkpoint 开 MTP 时无 crash、无警告，drafter 不被构建，服务按普通 decode 运行，输出正确但慢，用户以为在用推测加速
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/55322

### INF-1089 `sglang_ngram_cache_reuse_stale_draft_longer_context` SGLang NGRAM 复用请求 ID 时缓存返回陈旧 draft

`故障` · `active` · `infra_off` · `other_stack` · `verified` · `NEW` · 发现 2026-09-05 · 旧号 11.66

- **原因对象**：解码/采样 / sglang / NgramCorpus 请求级缓存失效
- **原因行为**：NgramCorpus 的请求级缓存条目在语料插入后不失效：draft 生成读请求绑定时的 trie 快照/游标，synchronize() 只保证插入可见，不触发缓存条目重算
- **影响对象**：性能/资源（性能退化）
- **影响详情**：NGRAM spec 复用请求 ID 且插入更长上下文后，缓存请求得旧 draft [30,70,80,90]，同历史新请求得正确的 [30,40,50,60]，BFS/PROB 均复现；draft 被 verify 拒绝，输出正确但接受率与吞吐静默下降
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/38129

### INF-1090 `vllm_glm53_nonflash_mtp_draft_noise_pos0_acceptance_2pct` vLLM GLM-5.3 非 Flash MTP draft 接受率灾难性偏低

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `var` · 发现 2026-09-06 · 旧号 11.67

- **原因对象**：解码/采样 / vllm / GLM-5.3 MTP proposer
- **原因行为**：根因未定位：eager+MTP-1 排除 CUDA graph/dispatch，损坏在 proposer 自身；嫌疑：DSA sparse-indexer draft 迭代状态或 FP8 block-scale 下的 draft 权重
- **影响对象**：输出分布；性能/资源（性能退化）
- **影响详情**：官方 FP8、B300×8 TP8 + MTP：重复文本 pos-0 接受率 ≤2.4%（健康 60–80%），低于随机文本（0.62–0.94% vs 8.69–8.85%）；吞吐比无 spec 低 29–41%，输出正确；双 nightly、双后端恒定复现
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/55605
- **关联**：var:INF-1086

### INF-1093 `trtllm_eagle3_dynamic_tree_decoding_emits_out_of_grammar_tokens` TRT-LLM EAGLE3 树状推测绕过 grammar 约束

`故障` · `active` · `infra_off` · `other_stack` · `verified` · `NEW` · 发现 2026-09-08 · 旧号 11.70

- **原因对象**：解码/采样 / tensorrt-llm / dynamic-tree grammar 约束
- **原因行为**：dynamic-tree 解码为每分支维护 flat mask，grammar bitmask 只施加在主路径，树状 draft 分支扩展的 token 未经过滤即被提交，约束状态机与树状扩展的交集计算缺失
- **影响对象**：输出内容（静默错误）
- **影响详情**：EAGLE3 dynamic-tree + guided decoding：grammar aa(bc|de)f、max_tokens=4 时线性模式只出合法前缀 aabc/aade，tree 模式（depth 2、top-K 32）产出非法前缀；无 crash，单层 dummy 权重即复现
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/NVIDIA/TensorRT-LLM/issues/18892

### INF-1094 `trtllm_guided_decoding_grammar_restart_after_kv_recompute` TRT-LLM KV 重计算后 grammar 重启，guided 输出违反约束

`故障` · `active` · `infra_off` · `other_stack` · `verified` · `NEW` · 发现 2026-09-08 · 旧号 11.71

- **原因对象**：解码/采样 / tensorrt-llm / grammar matcher 重算恢复路径
- **原因行为**：KV 重计算路径重建 attention 状态，但不保存/恢复 grammar matcher 已消费 token 的游标；matcher 从头初始化，把恢复后的上下文当作新的约束会话。
- **影响对象**：输出内容（静默错误）
- **影响详情**：MAX_UTILIZATION 调度器因 KV 容量不足暂停请求并重算后，guided 输出从头重新满足约束：约束 a{31}b{32}c{32} 时在 64 个 token 后又输出 31 个 a，静默违反长度/顺序约束；只在容量压力下出现，冒烟测试不可见。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/NVIDIA/TensorRT-LLM/issues/18891

### INF-1099 `vllm_extract_hidden_states_proposer_first_column_feedback_multi_token` vLLM extract_hidden_states 提议器只读首列致重复

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-09-11 · 旧号 11.76

- **原因对象**：解码/采样 / vllm / extract_hidden_states 提议器
- **原因行为**：prepare_next_token_ids_padded() 只读 sampled_token_ids 第 0 列，假设每迭代仅 1 个有效 token；spec 步提交两个 token 时返回的有效计数与续接 token 均错。
- **影响对象**：输出内容（静默错误）
- **影响详情**：extract_hidden_states 推测解码 + async scheduling 下，生成文本出现可复现的数字与标点重复，后续 draft 对位错行；改用 eagle_prepare_next_token_padded_kernel 后恢复健康基线 token。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/56417

### INF-1100 `vllm_dspark_fillin_layout_checkpoint_missing_flag_silent_sub_one_percent_acceptance` vLLM DSpark drafter 缺布局旗标按默认值误解析致接受率<1%

`故障` · `active` · `infra_off` · `config` · `verified` · `NEW` · 发现 2026-09-14 · 旧号 11.77

- **原因对象**：解码/采样 / vllm / DSpark drafter 布局旗标默认值
- **原因行为**：drafter config 缺 sample_from_anchor 时 DSparkSpeculator 不探测布局、按默认 native 解析，fill-in 的 anchor 位被当作被预测 token，后续各位条件于错误上下文。
- **影响对象**：性能/资源（性能退化）
- **影响详情**：加载 fill-in 布局导出且缺该字段的 DeepSpec drafter（两条独立导出路径均中招）时，vLLM 0.27.1 接受率约 0.9%，零错误零警告；输出经 target 验证仍正确，只有接受率指标可见。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/56792

### INF-1101 `ttmetal_gemma4_dflash_spec_session_reseed_replay_skip_previous_request_prompt_bleed` tt-metal dFlash spec 会话跳过 replay 致答上一请求

`故障` · `active` · `infra_off` · `other_stack` · `verified` · `NEW` · 发现 2026-09-15 · 旧号 11.78

- **原因对象**：解码/采样 / tt-metal / spec 会话 reseed/replay 守卫
- **原因行为**：spec 会话 replay 与 capture 之间守卫时序有窗口：批式请求后 session 的 pending replay 未消费，下一请求跳过 replay 直接读 _out 残留，以上一请求 token 开场；所有权标志未覆盖。
- **影响对象**：跨请求/跨样本数据；输出内容（静默错误）
- **影响详情**：31B dFlash、P150x8、temperature 0：8 并发 baseline 突发后的 solo thinking 请求生成 400+ token、finish=stop，内容回答的是上一个请求的 prompt；依赖批式→solo 流量转换，复现 2/2。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/56680

### INF-1105 `sglang_dflash_anchor_first_checkpoint_layout_unvalidated_position_shift_acceptance_collapse` SGLang 不校验 DFLASH 布局约定致接受率坍塌

`故障` · `active` · `infra_off` · `config` · `verified` · `var` · 发现 2026-09-18 · 旧号 11.82

- **原因对象**：解码/采样 / sglang / DFlashDraftConfig 布局解析
- **原因行为**：SGLang 假设 DFLASH drafter 为 1+N 布局且不校验，配置解析不读 query_zero_predicts_next 声明，anchor-first checkpoint 被按 1+N 解析，draft 全体位移一位。
- **影响对象**：性能/资源（性能退化）
- **影响详情**：Qwen3.8-Flash-Next + 官方 DeepSpec block7 drafter，8×A800 实测 47,411 步 accept length 1.06（同 checkpoint 在 vLLM 为 2.76–2.98）；服务健康、零警告，输出仍正确，只有接受率指标可见。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/40144
- **关联**：var:INF-1100

### INF-1106 `flashinfer_topk_minp_sampling_indices_per_request_threshold_row_mismatch_oob_read` flashinfer top-k/min-p 采样阈值按错行读取

`故障` · `active` · `infra_off` · `yes` · `verified` · `NEW` · 发现 2026-09-19 · 旧号 11.83

- **原因对象**：解码/采样 / flashinfer / TopK/MinP 采样 kernel 阈值索引
- **原因行为**：top-k/min-p 采样 kernel 用 top_k_arr[bx]/min_p_arr[bx] 按输出位置取阈值，而非 indices[bx] 对应的数据行；长度校验只对照 probs 而非 indices。
- **影响对象**：输出分布（静默错误）
- **影响详情**：传入 indices（gather/重排批）且 top_k/min_p 为 per-request 张量时，每个输出行套用错误行的阈值，采样分布静默错误；len(indices) 大于 probs 行数时读越阈值张量末尾。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/flashinfer-ai/flashinfer/issues/5339

### INF-1107 `vllm_allowed_token_ids_bad_words_empty_intersection_silent_allowlist_break` vLLM allowlist+bad_words 空交集时采出非法 token

`故障` · `active` · `infra_off` · `config` · `verified` · `NEW` · 发现 2026-09-21 · 旧号 11.84

- **原因对象**：解码/采样 / vllm / allowlist/bad_words 约束组合
- **原因行为**：allowlist 掩码与 bad_words 屏蔽按独立 pass 施加，交集为空时得到全 -inf logits 行；采样器对“无合法 token”无拒绝/报错路径，请求校验层也不检查两约束的一致性。
- **影响对象**：输出内容（静默错误）
- **影响详情**：Qwen3-0.6B 上 allow [264] 同时 ban 该 token，请求返回 HTTP 200 却生成 4 次不在 allowlist 的 token 0；两字段单独使用及非空交集对照均正常，温度 0 确定复现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/57986

### INF-1108 `vllm_mtp_draft_model_dict_hf_overrides_not_inherited_length_gate_dummy_zero_acceptance_collapse` vLLM MTP draft 不继承上下文 override 致接受率归零

`故障` · `active` · `infra_off` · `config` · `verified` · `var` · 发现 2026-09-22 · 旧号 11.85

- **原因对象**：解码/采样 / vllm / MTP draft 配置继承与输入门
- **原因行为**：dict 形式 --hf-overrides 不应用到 MTP draft，draft 沿用原生 max_model_len；序列超限时 drafter 跳过提案、以普通 token 0 填充，与 sampler 的 -1 占位语义冲突。
- **影响对象**：性能/资源（性能退化）
- **影响详情**：长上下文部署在序列越过 draft 原生 max_position_embeddings 后，MTP 接受率坍塌至约 0，每步仍付 target verify 成本，延迟变差；target 输出正确、零日志，只有 spec-decode 指标可见。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/58080
- **关联**：var:INF-1086

### INF-1109 `vllm_thinking_token_budget_forced_end_prefix_realign_repetition_loop_at_exhaustion` vLLM thinking budget 耗尽时强制结束错位致重复循环

`故障` · `active` · `infra_off` · `config` · `verified` · `NEW` · 发现 2026-09-23 · 旧号 11.86

- **原因对象**：解码/采样 / vllm / thinking budget 强制结束逻辑
- **原因行为**：强制 end 序列全部进入尾部后 prefix-match 移位、end_prefix_len 重算为 0，每步重新强制首个 end token；forced/natural end 不同时早退守卫失效。
- **影响对象**：输出内容（质量退化）
- **影响详情**：使用 thinking_token_budget 且 forced/natural end 配置不同时，生成恰在预算耗尽点坍塌为逐字重复直至 max_tokens，无日志；RTX PRO 6000 Blackwell 生产部署确定性复现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/58390

### INF-1110 `vllm_v1_thinking_budget_spec_decoding_multi_token_end_forcing_desync_corrupts_end_sequence` vLLM V1 thinking budget 叠加推测解码致结束序列残缺

`故障` · `active` · `infra_off` · `config` · `documented` · `var` · 发现 2026-09-24 · 旧号 11.87

- **原因对象**：解码/采样 / vllm / V1 thinking budget 强制推进
- **原因行为**：强制结束每步至多一个 force_index，后续步按 spec_token_ids 匹配推进 end_count；draft 匹配位置不强制、交 target 自由采样，被拒时游标不回退，跳过的 end token 永不发出。
- **影响对象**：输出内容（静默错误）
- **影响详情**：V1 runner + 推测解码 + 14 token 的 reasoning_end_str 时结束短语被跳 token 残缺化；GB10+27B+MTP 32 请求中 17 例结束序列损坏、5 例烧到 max_tokens，同一 main 走 V2 全部完好；loop-breaker 同病。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/58485
- **关联**：var:INF-1109

### INF-1111 `sglang_spec_v2_workers_diverge_forward_batch_generation_signature_pp_proxy_tensors_typeerror` SGLang 四个 spec-v2 worker 签名缺参致 TypeError

`故障` · `active` · `infra_off` · `other_stack` · `verified` · `NEW` · 发现 2026-09-25 · 旧号 11.88

- **原因对象**：解码/采样 / sglang / spec-v2 worker 接口签名
- **原因行为**：非 overlap 推测路径调度器恒传 pp_proxy_tensors，但 BaseSpecWorker 未声明 forward_batch_generation 抽象方法锚定签名，dspark 等四个 v2 worker 签名缺该参数。
- **影响对象**：可用性/进度（报错）
- **影响详情**：经上述四个 worker 服务的模型在首个非 overlap 步抛 TypeError（unexpected keyword argument 'pp_proxy_tensors'），对应 spec 后端整体不可用；属 fail-loud，非静默错值。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/40155

### INF-1112 `flashinfer_chain_speculative_sampling_philox_offset_advance_mismatch_rng_draw_reuse` flashinfer 推测采样 Philox offset 步进不足致随机数复用

`故障` · `active` · `infra_off` · `other_stack` · `verified` · `NEW` · 发现 2026-09-28 · 旧号 11.89

- **原因对象**：解码/采样 / flashinfer / spec 采样 Philox offset 记账
- **原因行为**：chain_speculative_sampling 的 wrapper 按 B*(N+1) 上取整到 4 的倍数推进 offset 并返回推进后的值，而 kernel 每流在拒绝时最多消耗 N+2 个随机字，超出预付步进。
- **影响对象**：输出分布（静默错误）
- **影响详情**：同一 torch.Generator 连续两次等形调用且首次发生拒绝时，恢复抽取与下次接受判定共用同一随机坐标；208 种子研究中目标事件 208/208（独立应约 1/2），输出合法无异常。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/flashinfer-ai/flashinfer/issues/5637

### INF-1114 `vllm_cpu_apply_top_k_top_p_triton_drops_tokens_top_p_must_keep` vLLM CPU triton top-p 丢弃应保留的 token

`故障` · `active` · `infra_off` · `other_stack` · `verified` · `NEW` · 发现 2026-10-09 · 旧号 11.91

- **原因对象**：解码/采样 / vllm / apply_top_k_top_p_triton
- **原因行为**：triton 实现的 top-p 对累计概率质量边界的包含性判断错误，把 top-p 必须保留的 token 截掉；既有测试只比 kept 计数不比 kept 集合，且只覆盖 V≥1024。
- **影响对象**：输出分布（静默错误）
- **影响详情**：CPU 构建（triton-cpu）下用 top_p 采样：8-token 行中携 7.5% 质量的 token 被丢；引擎级 4000 请求该 token 采样 0 次，精确分布约 300 次；V≤4096 时 0–4% 行受影响。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/59785

### INF-1115 `vllm_eagle_prepare_inputs_index_underflows_past_query_start_pp_spec_decode` vLLM EAGLE prepare_inputs 采样索引下溢致跨请求串线

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-10-09 · 旧号 11.92

- **原因对象**：解码/采样 / vllm / EAGLE prepare_inputs 索引计算
- **原因行为**：eagle_prepare_inputs_padded_kernel 以 q_last_tok_idx-num_rejected_tokens 算采样索引且无下界钳制，query 无 bonus 行时前提不成立；CPU 实现同错。
- **影响对象**：跨请求/跨样本数据；输出内容（静默错误）
- **影响详情**：PP>1 下调度器只排 draft token 且全拒时，首请求负索引越界 gather，后续请求采到前一请求行的 token 并损坏此后每步 decode；单元级复现两请求全拒输出 [-1, 1]。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/60897

### INF-1116 `trtllm_beam_search_logits_processors_receive_per_slot_lists_not_hypotheses` TRT-LLM beam search 的 processor 收到槽位非假设

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-10-09 · 旧号 11.93

- **原因对象**：解码/采样 / tensorrt-llm / beam search 槽位↔假设映射
- **原因行为**：PyTorch 后端 beam search 按静态 admission-width 布局把 slot 的 token 列表传给 logits processor，slot↔hypothesis 映射未随 beam 消亡/重生维护。
- **影响对象**：输出内容（静默错误）
- **影响详情**：用 logits processor 做 ban/boost 引导时，两 beam 同源于同一候选后 processor 收到的 token_ids 与 beam 假设不符，引导作用于错误谱系，输出静默偏离意图（2-beam TinyLlama 复现）。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/NVIDIA/TensorRT-LLM/issues/19902

### INF-1132 `trtllm_group_greedy_sample_state_stale_across_iterations` TRT-LLM group greedy 采样决策跨迭代不复位致采样语义串扰

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-08-14 · 旧号 12.16

- **原因对象**：解码/采样 / tensorrt-llm / group greedy 采样决策状态
- **原因行为**：上一迭代同步的 group_all_greedy_sample 决策在推导当前 batch 的 greedy flag 前不清空，过期决策被后续 CUDA-graph lookup 与 sampling-buffer 填充持续复用。
- **影响对象**：输出分布；跨请求/跨样本数据（静默错误）
- **影响详情**：DSpark rejection sampling + attention DP + LM-head TP 下，GEN engine 用无采样参数的 CONTEXT_INIT 请求 warm-up 后，温度/rejection sampling 语义在迭代间串扰，无 crash。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/NVIDIA/TensorRT-LLM/pull/17710

### INF-1147 `vllm_dflash2_draft_fp16_xpu_zero_acceptance_bf16_works` vLLM XPU 上 DFlash2 fp16 draft 接受率归零

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-09-04 · 旧号 12.31

- **原因对象**：解码/采样 / vllm / DFlash2 draft fp16 路径
- **原因行为**：XPU 上 DFlash2 draft 模型在 fp16 下数值路径与 target 分歧（block_size-8 局部计算 fp16 累积误差更大），逐 token 分布偏离到 verify 永远拒绝。
- **影响对象**：性能/资源（性能退化）
- **影响详情**：XPU + DFlash2 + --dtype float16 时服务正常、输出正确，但 acceptance length 恒 1.00（0% 接受），吞吐回到普通 decode 且白占 draft 显存算力；bf16 正常，GPTQ/int4 常用 fp16 故默认踩坑。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/55250

### INF-1160 `vllm_dflash_dcp_slot_mapping_kernel_block_size_vs_kv_block_size_ownership_divergence` vLLM DFlash 在 DCP 下混用块尺寸致 slot 错位

`故障` · `active` · `infra_off` · `partial` · `verified` · `NEW` · 发现 2026-09-29 · 旧号 12.44

- **原因对象**：解码/采样 / vllm / DFlash 输入准备 slot 映射
- **原因行为**：_prepare_dflash_inputs_kernel 用单一 kernel block size 同时算 DCP 归属与 slot offset；通用 compute_slot_mappings() 则把两种块尺寸分开。
- **影响对象**：KV状态；输出内容（静默错误）
- **影响详情**：cp_size>1 且物理/KV 块大小≠kernel 块大小时，DFlash 读写落到错误 rank/错误槽，输出错误且无报错（测试例 B=32/K=16/CP=4 中 kernel 块 10 应得 slot 160）；默认同尺寸配置不受影响。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/59184

### INF-1279 `specdec_bonus_token_draft_kv_desync` 推测解码 bonus token 未写入 draft KV 致上下文滞后一位

`故障` · `active` · `model_on` · `partial` · `verified` · `NEW` · 发现 待定 · 旧号 K8.2

- **原因对象**：解码/采样 / vllm / 推测解码 bonus token 处理
- **原因行为**：提议全部被接受时会从 verifier 采样 bonus token，但对自持 KV 的 proposer（draft model/MTP），该 token 的 K/V 从未在 draft 中生成，draft KV 比被接受序列少一项。
- **影响对象**：KV状态；输出内容（静默错误）
- **影响详情**：draft 与 target 上下文永久错位，后续每次提议都基于落后一位的上下文，接受率下降、输出静默偏移；仅在提议全部被接受且 proposer 自持 KV 时触发。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/4212

### INF-1280 `beam_reorder_cache_index_dim_mismatch` 混合模型 beam search 按 beam 重排 KV 时维度不匹配

`故障` · `active` · `infra_off` · `yes` · `verified` · `NEW` · 发现 待定 · 旧号 K8.3

- **原因对象**：解码/采样 / transformers / Lfm2 混合 cache 的 beam 重排
- **原因行为**：beam search 每步用 index_select(0, beam_idx) 按 beam 重排各层 KV；Lfm2 的 conv 层与注意力层交错，各层 cache 不是统一的 [batch,...] 张量，重排维度不匹配。
- **影响对象**：KV状态；可用性/进度；输出内容（报错）
- **影响详情**：混合模型上 beam search 直接 IndexError 崩溃；若在错误维度或缺 beam 轴的层上静默重排，某个 beam 会沿用另一 beam 的历史继续生成，输出前后语义断裂。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/huggingface/transformers/issues/42257

### INF-1285 `frozen_kv_mtp_draft_depth_gated_corruption` Frozen-KV MTP 在 draft 深度≥5 且并发时输出损坏

`故障` · `active` · `ambiguous` · `partial` · `documented` · `NEW` · 发现 待定 · 旧号 K8.8

- **原因对象**：解码/采样 / sglang / Frozen-KV MTP 的 verify 路径
- **原因行为**：根因未确认（issue open）；推测为 Frozen-KV 策略在 verify 路径 CUDA graph 捕获/重放中，draft 链越长 KV slot 记账累积偏差越大，使 verify batch 读到错位 KV。
- **影响对象**：输出内容；KV状态（静默错误）
- **影响详情**：并发≥2 且 draft depth≥5 时，所有无 grammar 约束的请求返回退化 token 泥浆并跑到 max_tokens（8/8），无 crash/NaN；depth≤3 或串行干净，depth=4 时 verify CUDA graph 捕获非法访存、服务无法启动。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/32666

### INF-1286 `specdec_eagle_fused_topk1_nan_oob` EAGLE fused top-k=1 遇 NaN 选出越界 token

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-07-29 · 旧号 K8.10

- **原因对象**：解码/采样 / sglang / fused top-k=1 的 argmax
- **原因行为**：draft logits 含 NaN 时，fused top-k=1 的 argmax 因 NaN > -inf 比较为真而选中 masked tail lane；NaN 未先映射到有限 floor 就做 argmax。
- **影响对象**：输出内容；可用性/进度（报错）
- **影响详情**：推测解码返回越界 token_id（== vocab_size），引发越界访问或 crash；仅在 EAGLE fused top-k=1 路径且 draft logits 出现 NaN 时触发。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/pull/32396

### INF-1287 `specdec_sa_suffix_automaton_seqlen_crash` 后缀自动机推测解码长 prompt 下 seqlen 预算推导错误致崩溃

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-07-29 · 旧号 K8.11

- **原因对象**：解码/采样 / tensorrt-llm / SA 推测解码的 max_tokens 推导
- **原因行为**：后缀自动机（SA）推测解码按 draft 膨胀后的 max_seq_len 推导 default_max_tokens，而膨胀后的 seqlen 超过了实际分配的预算。
- **影响对象**：可用性/进度（报错）
- **影响详情**：prompt 接近 max_seq_len 且启用 SA 推测解码时 executor event loop 崩溃，属响亮失败而非静默错误。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/NVIDIA/TensorRT-LLM/issues/16826

### INF-1288 `specdec_vllm_short_conv_lfm2_fix` vLLM 上 LFM2 short_conv 推测解码控制流错误

`故障` · `active` · `infra_off` · `partial` · `documented` · `var` · 发现 2026-07-29 · 旧号 K8.12

- **原因对象**：解码/采样 / vllm / LFM2 推测解码接受/拒绝控制流
- **原因行为**：LFM2（short_conv 混合架构）上推测解码的接受/拒绝控制流存在 bug；原文未给出更具体机制。
- **影响对象**：输出内容（静默错误）
- **影响详情**：LFM2 模型启用推测解码时输出不正确；其他触发条件原文未给出。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/50272
- **关联**：var:INF-1027

### INF-1289 `specdec_dspark_structured_output_interaction` DSpark 推测解码与结构化输出交互缺陷

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-07-29 · 旧号 K8.13

- **原因对象**：解码/采样 / tensorrt-llm / DSpark 与结构化输出交互
- **原因行为**：DSpark speculative drafter 的 draft token 生成与结构化输出 grammar 约束之间的交互存在缺陷；原文未给出具体机制。
- **影响对象**：输出内容（静默错误）
- **影响详情**：推测解码与结构化输出组合使用时，输出不符合 grammar 约束或质量退化；单进程路径不可复现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/NVIDIA/TensorRT-LLM/issues/16967

### INF-1290 `batch_specdec_output_equivalence_violation` 批量推测解码普遍违反输出等价性

`结论` · `active` · `infra_off` · `partial` · `verified` · `NEW` · 发现 2026-07-29 · 旧号 K8.9

- **原因对象**：解码/采样 / 通用 / 批量推测解码的 batched verify
- **原因行为**：batch 场景下不同请求的 draft token 接受/拒绝决策在 batched verify 中互相耦合，改变了采样路径的概率结构；原文称所有现有实现均如此。
- **影响对象**：输出分布；输出内容（静默错误）
- **影响详情**：批量推测解码的输出分布与非批量自回归解码不恒等，偏差取决于 batch 组成与 draft 质量；对大多数 prompt 偏差可能很小，但在 acceptance 边界附近可能翻转 token，无 crash/NaN。
- **改进措施**：检测 待定；修复 待定
- **来源**：http://arxiv.org/abs/2510.22876v3

### INF-1291 `sglang_kimiK3_dspark_longctx_pad_nan_storm` Kimi-K3 DSPARK 长上下文 NaN 致 [PAD] 风暴与崩溃

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-07-30 · 旧号 K8.14

- **原因对象**：解码/采样 / sglang / DSPARK prefill/draft 路径
- **原因行为**：长上下文下 logits 被瞬态 NaN 污染，根因与 #32477 修复的写侧 bug 一致但修复未进入 kimi-k3 image；叠加 allowed_special="all" 且 pad 未被 stop/过滤。
- **影响对象**：输出分布；输出内容；可用性/进度（静默错误）
- **影响详情**：约 90k–237k token 长上下文 + DSPARK 时突发：(A) 持续输出字面 [PAD] 直到 max_tokens 或断连（最坏单流 17,256 个）；(B) DSPARK 路径抛 CUDA assert，杀死全部 TP rank。短上下文测试不可见。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/32968

### INF-1308 `specdec_kvquant_noncausal_incompat` DFlash 非因果草稿注意力与 KV cache 量化不兼容致循环

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 待定 · 旧号 K9.17

- **原因对象**：解码/采样 / vllm / DFlash 非因果草稿注意力
- **原因行为**：DFlash 式推测解码需要非因果草稿注意力，与 fp8、TurboQuant 等 KV cache 量化在原理上不兼容，草稿注意力在非因果读取下读到量化损坏的 KV。
- **影响对象**：输出内容（质量退化）
- **影响详情**：两个特性各自可用、组合即坏：稠密与混合注意力模型上生成都陷入 token 级重复循环；TurboQuant KV 与任意推测解码组合同样出现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/41559；https://github.com/vllm-project/vllm/issues/40831

## 量化

### INF-0109 `vllm_xpu_battlemage_persistent_vram_state_corruption_token0` vLLM XPU 并发 decode 后持久损坏，输出全为 token 0

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-27 · 旧号 1.110

- **原因对象**：量化 / vllm / W4A16 内核（head_dim 256）
- **原因行为**：根因未定位：持续并发 decode 后，权重或持久 workspace 等长寿命 VRAM 状态疑被原地损坏；两张物理卡复现排除单卡硬件故障，head_dim 256 × W4A16 内核组合可疑。
- **影响对象**：输出内容（静默错误）
- **影响详情**：Arc Pro B70 上 Qwen3.8-27B W4A16 持续并发 decode 数小时（或 12 并发 10 分钟）后输出全变 token 0（“!”），HTTP 200、健康检查与 dmesg 正常；负载清空后新请求仍坏，重启重载权重才恢复；生产 30 小时命中 4 次。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/53480

### INF-0217 `vllm_hadacore_transform_tail_warp_cp_async_wait_negative_switch_cross_launch_stale_smem` vLLM hadacore 漏发 cp.async 等待致读残留 smem

`故障` · `active` · `infra_off` · `yes` · `verified` · `NEW` · 发现 2026-09-15 · 旧号 1.218

- **原因对象**：量化 / vllm / hadacore_transform 尾 warp
- **原因行为**：尾部 warp 以 switch(k+diff_num_chunks) 选择等待组数，rows>8192 且 rows%8∈{1..4} 时下标为 −1、无对应 case，不发 wait 即读 shared memory。
- **影响对象**：数值结果；KV状态；输出内容（数值异常）
- **影响详情**：int4_per_token_head KV 或 compressed-tensors transform 且头数非 8 倍数时，Q 路径出 NaN logprob（请求 logprobs 则 HTTP 400，否则静默错 token），5 次 3 种输出；K/V 路径坏值持久驻留 cache。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/57014

### INF-0240 `ttmetal_host_bfp4_bfp2_packer_direct_mantissa_round_vs_device_bfp8_then_narrow` host BFP4/BFP2 打包舍入与设备路径不一致

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-09-22 · 旧号 1.241

- **原因对象**：量化 / tt-metal / host BFP packer 舍入
- **原因行为**：host packer 把 fp32 24 位尾数直接 RNE 舍入到 BFP4/BFP2 目标宽度，跳过 ISA 规定的先产 7 位 BFP8 幅值（ties-away）再截断的中间步。
- **影响对象**：数值结果（静默错误）
- **影响详情**：同一 bf16 输入经 ttnn.from_torch（host）与 ttnn.typecast（设备）转 bfloat4_b，32×32 N(0,3) 上 42.5% 元素位型不同，最坏差一个完整 mantissa step（值 ×2），含 host 非零→设备零。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/57410

### INF-0254 `ttmetal_requantize_tensor_spelled_scales_bf16_intermediate_up_to_80_lsb` requantize 以 Tensor 传 scale 时中间量 bf16 化

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `var` · 发现 2026-09-25 · 旧号 1.255

- **原因对象**：量化 / tt-metal / requantize composite 回退
- **原因行为**：任一 scale/zero_point 为 Tensor 时走 composite 回退，dequantize 中间量 dtype 缺省 BFLOAT16，经 7 位尾数舍入后再重量化；全标量路径走 fp32 融合 LLK。
- **影响对象**：数值结果（静默错误）
- **影响详情**：int32 输入、in_scale=0.05、out_scale=0.001 时 75.4% 元素偏差、最大 13 LSB，0.08/0.003 时 94.2% 偏差、80 LSB；torch 导出量化模型以 Tensor 携带 scale 必中此路径，Python 标量测试恒绿。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/57852
- **关联**：var:INF-0756

### INF-0289 `ttmetal_blackhole_custom_matmul_bfp2_b_plain_path_unrelated_output` custom matmul Bfp2_b plain path 输出无关值

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-10-09 · 旧号 1.290

- **原因对象**：量化 / tt-metal / custom matmul Bfp2_b 解码
- **原因行为**：Bfp2_b 权重在 plain path 上的解码实现错误，原文指向 LLK 审计报告，未给出细节。
- **影响对象**：数值结果（静默错误）
- **影响详情**：Blackhole LLK custom matmul 中 Bfp2_b 权重走 plain path 时输出与乘积无关的值，无报错；Bfp4_b/Bfp8_b 权重及压缩路径的 Bfp2_b tile 均正确。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/58785

### INF-0337 `vllm_moe_wna16_splitk_16bit_atomic_add_run_to_run_nondeterministic` vLLM wna16 MoE split-K 原子加合并致跨 run 非确定

`故障` · `active` · `ambiguous` · `partial` · `verified` · `var` · 发现 2026-09-14 · 旧号 4.21

- **原因对象**：量化 / vllm / moe_wna16_gemm split-K 合并
- **原因行为**：split-K 部分和按 CTA 到达顺序以 16-bit atomicAdd 合并进 bf16 输出，浮点加法不结合，低位比特随运行顺序分歧。
- **影响对象**：数值结果；输出内容（不可复现）
- **影响详情**：同一输入两次调用比特不同，top-k 1 亦命中，进程内与跨进程均复现（RTX PRO 6000 SM120）；temperature=0 下近 tie 处 greedy token 跨 run 翻转，单次评测无法暴露。flashinfer SM120 b12x_fused_moe 有同族现象。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/56887
- **关联**：var:INF-0327

### INF-0338 `vllm_batch_invariant_flag_ignored_by_triton_fused_moe_experts_batch_wide_dynamic_scale` vLLM 批不变 flag 未覆盖 Triton fused-MoE 专家

`故障` · `active` · `ambiguous` · `config` · `verified` · `var` · 发现 2026-09-15 · 旧号 4.22

- **原因对象**：量化 / vllm / Fp8MoEMethod 激活 scale 粒度；算子/内核 / vllm / Triton fused-MoE 批不变覆写
- **原因行为**：批不变覆写只含 HUMMING 后端，Triton fused-MoE 专家未被重定向也不告警；在线 fp8 专家漏设 per_act_token_quant，全批共用一个动态 scale；bf16 专家按 token 数选配置。
- **影响对象**：数值结果；输出内容；KV状态（不可复现）
- **影响详情**：开 VLLM_BATCH_INVARIANT=1 用 Triton fused-MoE 时无警告，greedy 输出随批伴内容翻转（logprob gap 0.07–0.53），前缀缓存写入的 KV 字节依赖写者后缀；同输入重放探针全过，只有跨批/跨写者对拍才暴露。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/57016
- **关联**：var:INF-0333

### INF-0350 `tp_marlin_gibberish` TP>1 下 GPTQ-Marlin 量化 GEMM 分片出错致乱码

`故障` · `active` · `infra_off` · `partial` · `verified` · `NEW` · 发现 待定 · 旧号 5.3

- **原因对象**：量化 / vllm / GPTQ-Marlin GEMM 的 TP 分片
- **原因行为**：TP>1 时 GPTQ-Marlin 量化 GEMM kernel 的分片计算（shard-split）产生与非量化路径不一致的错误 partial result。
- **影响对象**：数值结果；输出内容（静默错误）
- **影响详情**：TP=4 加载 Meta-Llama-3-70B-GPTQ 时服务正常启动、输出看似流畅，实为重复 token 和无意义内容；不与 TP=1 做 diff 很容易漏掉。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/6258；https://github.com/vllm-project/vllm/issues/8017

### INF-0438 `vllm_lfm2_shortconv_prefix_quant_ignore_miss` vLLM LFM2 ShortConv 前缀与量化忽略表不匹配

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-08-08 · 旧号 6.45

- **原因对象**：量化 / vllm / LFM2 ShortConv 的 prefix
- **原因行为**：属性名 .short_conv 与构造 prefix .conv 不一致；ignore 列表经 mapper 重写为 .short_conv. 后匹配不上，should_ignore_layer() 返回 False，conv 按量化构建。
- **影响对象**：可用性/进度（报错）
- **影响详情**：LFM2/LFM2-MoE 加载含 conv ignore list 的量化 checkpoint 时触发；checkpoint 中 conv 只有无量化 weight，加载因缺 weight_packed/weight_scale 而失败。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/51495

### INF-0439 `vllm_ue8m0_fusion_silent_scale_contract_loss` vLLM ue8m0 组量化融合 pass 静默丢失 2 的幂 scale 契约

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-08 · 旧号 6.46

- **原因对象**：量化 / vllm / rms/act 量化融合 pass（ue8m0）
- **原因行为**：融合 pass 为 is_e8m0=True 注册变体，但重写目标 rms_norm_per_block_quant/silu_and_mul_per_block_quant 无 ue8m0 参数，恒发 fp32 group scale。
- **影响对象**：数值结果；输出分布（静默错误）
- **影响详情**：SM120/Blackwell 上 ue8m0 组量化配融合 pass 时触发；无 crash、无 NaN，编译图产出非 2 的幂 scale（实测 128/128），DeepGEMM FP8 路径语义改变，输出静默偏差。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/51477

### INF-0453 `vllm_dflash_fused_kv_proj_bypasses_quant_weight_scale` vLLM DFlash 绕过量化方法、忽略 weight_scale

`故障` · `active` · `infra_off` · `yes` · `documented` · `NEW` · 发现 2026-08-09 · 旧号 6.60

- **原因对象**：量化 / vllm / Qwen3DFlash fused KV GEMM
- **原因行为**：构建 fused KV GEMM 时切 qkv_proj.weight 用裸 F.linear 计算，绕过量化方法；qkv_proj 为量化 Linear 时直接用 packed 权重，忽略 weight_scale。
- **影响对象**：数值结果；性能/资源（静默错误）
- **影响详情**：DFlash 推测解码配 weight-only 量化 drafter（fp8/AWQ/GPTQ/compressed-tensors）时触发；dtype 不对齐报 RuntimeError，对齐时静默产出数值错误的 draft，只表现为 acceptance 下降。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/51581

### INF-0455 `vllm_nvfp4_moe_cutlass_silent_empty_completions_sm120` vLLM SM_120 NVFP4 MoE cutlass 静默产空补全

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-08-09 · 旧号 6.62

- **原因对象**：量化 / vllm / NVFP4 MoE VLLM_CUTLASS 后端
- **原因行为**：显式 --moe-backend cutlass（即 VLLM_CUTLASS）服务 NVFP4 MoE 时，该后端在 SM_120 上产不出有效内容；内部机制原文未给出，auto 不选此后端。
- **影响对象**：输出内容（静默错误）
- **影响详情**：SM_120（RTX 5090 Laptop）+ NVFP4 MoE + 显式 cutlass 后端时触发；HTTP 200、无异常无警告，耗尽 max_tokens 后返回 content:null、finish_reason:length，短 prompt smoke 通过、吞吐正常。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/51525

### INF-0487 `trtllm_visualgen_static_quant_recipe_unquantized_checkpoint_silent_weight_corruption` TRT-LLM VisualGen 静态量化缺 scale 校验致权重损坏

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-08-14 · 旧号 6.94

- **原因对象**：量化 / tensorrt-llm / VisualGen 静态量化 recipe 校验
- **原因行为**：VisualGen 管线允许对缺 scale tensor 的高精度 checkpoint 应用静态 FP8/FP8_BLOCK_SCALES/NVFP4 recipe，缺 scale 的模块不校验也不报错，权重按 recipe 被改写。
- **影响对象**：模型权重；输出内容（静默错误）
- **影响详情**：VisualGen 加载未量化 checkpoint 并配置静态 FP8/NVFP4 recipe 时触发；无 crash、无 warning，生成输出质量不可预测地退化。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/NVIDIA/TensorRT-LLM/pull/17699

### INF-0507 `vllm_bnb_plugin_intree_duplicate_op_registration_crash` bnb 插件与树内实现重复注册 op 致启动崩溃

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-08-18 · 旧号 6.114

- **原因对象**：量化 / vllm / apply_bnb_4bit op 注册
- **原因行为**：vLLM 树内已含 bitsandbytes 量化实现，外部 vllm-bnb-plugin 又注册同一 apply_bnb_4bit op，两者之间没有注册冲突仲裁。
- **影响对象**：可用性/进度（报错）
- **影响详情**：安装 vllm-bnb-plugin 并使用 bitsandbytes 量化时，启动或首用即报 duplicate registration 崩溃，非静默；卸载插件可规避。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/52802

### INF-0603 `vllm_flashinfer_fp8_scaled_mm_sm120_sustained_load_xid13_ima` SM120 FlashInfer FP8 scaled MM 负载下 IMA

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `var` · 发现 2026-09-06 · 旧号 6.210

- **原因对象**：量化 / flashinfer / FlashInfer FP8 scaled MM
- **原因行为**：SM120 上 FlashInferFP8ScaledMMLinearKernel（或其与 CUDA graph 捕获的组合）在持续负载下发生非法内存访问；具体根因待定。
- **影响对象**：可用性/进度（报错）
- **影响详情**：Ministral-3-14B FP8 在 RTX PRO 5000（SM120）持续负载下 EngineCore 概率性死于 CUDA illegal memory access，同秒内核日志 Xid 13；禁用该 kernel 或 --enforce-eager 任一即消失。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/55571
- **关联**：var:INF-0553

### INF-0641 `vllm_sm120_per_tensor_static_fp8_w8a8_production_xid13_ima_024_stable` vLLM 0.28 在 SM120 静态 FP8 W8A8 下周期性崩溃

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `var` · 发现 2026-09-20 · 旧号 6.248

- **原因对象**：量化 / vllm / SM120 FP8 W8A8 相关内核（未定位）
- **原因行为**：0.24→0.28 间引入或变更的内核路径在 SM120 + per-tensor static FP8 W8A8 + 非对齐剩余 chunk 下越界访问或执行非法指令；肇事内核未定位。
- **影响对象**：可用性/进度（报错）
- **影响详情**：RTX PRO 6000 生产流量 3 小时内 EngineCore 三次死亡（illegal memory access/illegal instruction，伴 Xid 13），均在部分前缀命中的单请求 prefill 步；回滚 v0.24 或升级 v0.29 稳定，按记录形状重放未复现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/57822
- **关联**：var:INF-0603

### INF-0674 `massive_activation_quantize_sink` 量化/裁剪破坏 massive activation 承载的 sink

`故障` · `active` · `model_on` · `config` · `verified` · `var` · 发现 待定 · 旧号 7.1

- **原因对象**：量化 / 通用 / per-tensor RTN 激活量化/裁剪
- **原因行为**：massive activation 在固定 feature dim 达中位数 1000x 以上，充当隐式 attention bias；per-tensor RTN 量化时其抬高 scale 压扁普通激活，裁剪则抹掉该 bias。
- **影响对象**：隐状态；输出内容（质量退化）
- **影响详情**：sink 被彻底破坏时 perplexity 爆炸、输出灾难性乱码（仅置零 4 个 massive activation 即可触发）；仅被削弱时 attention pattern 轻微漂移，输出读起来仍正常。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://arxiv.org/abs/2402.17762
- **关联**：var:INF-0765

### INF-0675 `kv_cache_fp8_quantization_drift` FP8 KV cache 量化误差随上下文长度累积漂移

`故障` · `active` · `infra_off` · `config` · `documented` · `var` · 发现 待定 · 旧号 7.2

- **原因对象**：量化 / vllm / FP8 KV cache scale 与粒度
- **原因行为**：KV cache 以 FP8（e4m3/e5m2）存储，默认 scale=1.0 严重欠用动态范围；Key 未用 per-channel、Value 未用 per-token 粒度，误差随 token 累积并被点积放大。
- **影响对象**：KV状态；输出内容（质量退化）
- **影响详情**：长上下文退化并随长度复合放大：e4m3 且校准良好时较轻，e5m2 或未校准时严重，出现重复循环、约一句话后语无伦次；kv_cache_dtype=fp8 与 bf16 KV 对照可见。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://arxiv.org/abs/2402.02750；https://arxiv.org/abs/2401.18079
- **关联**：var:INF-0870

### INF-0676 `fp8_static_scale_overflow` FP8 静态 per-tensor scale 遇离群值饱和或溢出

`故障` · `active` · `infra_off` · `config` · `verified` · `NEW` · 发现 待定 · 旧号 7.3

- **原因对象**：量化 / 通用 / FP8 静态 per-tensor scale
- **原因行为**：采用由校准集导出的固定 per-tensor scale，遇 OOD 输入或 massive activation 时数值超出缩放后的 E4M3 范围（约 ±448），被钳到上限饱和或溢出为 inf→NaN。
- **影响对象**：数值结果；输出内容（数值异常）
- **影响详情**：通常间歇性，只在出现离群值的 prompt 上触发：静默饱和时输出漂移，溢出时 NaN 灾难性传播。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://arxiv.org/abs/2502.01070

### INF-0677 `mxfp4_blockscale_outlier` MXFP4/NVFP4 块内 outlier 抬高共享 scale 压缩精度

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 待定 · 旧号 7.4

- **原因对象**：量化 / 通用 / MXFP4/NVFP4 块共享 scale
- **原因行为**：微缩 FP4 以 32 元素（MXFP4）或 16 元素（NVFP4）块共享一个 exponent，块内一两个 outlier 抬高共享 scale，把块内其余数值压缩到 4-level 分辨率。
- **影响对象**：数值结果；输出内容（质量退化）
- **影响详情**：精度渐进下降，集中在 outlier 密集的层；偶尔出现 token 级翻转（Flips）。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://arxiv.org/abs/2509.23202

### INF-0678 `per_tensor_act_clamp_outlier` per-tensor 激活量化被 outlier 通道抬高 scale

`故障` · `active` · `infra_off` · `config` · `verified` · `var` · 发现 待定 · 旧号 7.5

- **原因对象**：量化 / 通用 / per-tensor 激活量化 absmax
- **原因行为**：per-tensor 激活量化只用一个 absmax scale，少数系统性 outlier 通道（约 0.01%–0.7% 维度，幅度为常规通道 6–20x）抬高 scale，压垮主体数值；朴素 W8A8 int8 即属此类。
- **影响对象**：数值结果；隐状态；输出内容（质量退化）
- **影响详情**：perplexity 中等到严重上升并出现 Flips；outlier 列占比较高时模型可能实质性崩溃。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://arxiv.org/abs/2208.07339；https://arxiv.org/abs/2211.10438
- **关联**：var:INF-0765

### INF-0679 `rotation_preprocess_missing` 量化前旋转预处理缺失或用错致 4-bit 精度下降

`故障` · `active` · `model_on` · `config` · `verified` · `NEW` · 发现 待定 · 旧号 7.6

- **原因对象**：量化 / 通用 / QuaRot/SpinQuant 旋转预处理
- **原因行为**：量化前应施加并融合进相邻层的正交/Hadamard 旋转缺失（退化为朴素 4-bit）或用错（随机旋转代替学到的 Cayley 旋转），outlier 通道能量未被摊开、尾部未压平。
- **影响对象**：数值结果；输出内容（质量退化）
- **影响详情**：无旋转时 W4A4 精度急剧下降；随机旋转只能部分恢复，学到的旋转才能完全恢复。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://arxiv.org/abs/2404.00456；https://arxiv.org/abs/2405.16406

### INF-0680 `gptq_awq_smoothquant_calibration_drift` GPTQ/AWQ/SmoothQuant 保住聚合精度但逐 token 输出翻转

`故障` · `active` · `model_on` · `config` · `documented` · `var` · 发现 待定 · 旧号 7.7

- **原因对象**：量化 / 通用 / GPTQ/AWQ/SmoothQuant 方案
- **原因行为**：GPTQ（act-order/Cholesky）、AWQ（salient-channel scaling）、SmoothQuant（alpha migration）引入的量化扰动只保住聚合精度，不保证逐 token 选择与 fp 基线一致。
- **影响对象**：输出内容；输出分布（静默错误）
- **影响详情**：基准分数不变、看不出退化，但逐 token 生成结果与 fp 基线不同（Flips 盲区）；隐状态几何/谱/轨迹类度量与健康量化不可分，只能靠逐 token logprob 与分布距离观测。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://arxiv.org/abs/2210.17323；https://arxiv.org/abs/2306.00978；https://arxiv.org/abs/2211.10438；https://arxiv.org/abs/2407.09141
- **关联**：var:INF-0765

### INF-0681 `quantized_kernel_bias_omission` 融合反量化 matmul kernel 丢 bias 等量化 kernel 缺陷

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 待定 · 旧号 7.8

- **原因对象**：量化 / 通用 / 融合 dequant-matmul kernel
- **原因行为**：融合 dequant-matmul kernel 实现中漏掉 bias 项；或 INT8 与 BF16 引擎不匹配；或 FP8 多层堆叠间的层间 requantization 引入误差（合并 GPTQ、EXL2、MLC 等子故障）。
- **影响对象**：数值结果；隐状态；输出内容（静默错误）
- **影响详情**：关键层 bias 被完全丢弃时输出灾难性乱码；部分或轻微时表现为细微的 per-channel 加性偏置（均匀 DC 偏移），与 bias_param_bit_flip 签名重叠。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/NVIDIA/TensorRT-LLM/issues/1303；https://github.com/Dao-AILab/flash-attention/issues/679

### INF-0683 `sglang_fp8_reciprocal_mul_glm52_regression` SGLang FP8 倒数乘法替代除法致 GLM-5.2 decode 回归

`故障` · `active` · `ambiguous` · `other_stack` · `verified` · `NEW` · 发现 2026-07-29 · 旧号 7.10

- **原因对象**：量化 / sglang / per-token group FP8 量化
- **原因行为**：PR #30924 把 FP32-scale 乘子由直接除法（x / scale）改为显式倒数乘法（x * (1/scale)），在 Hopper 上引入小的 FP8 舍入差异，作用于 per-token group 量化路径。
- **影响对象**：数值结果；输出内容（质量退化）
- **影响详情**：Hopper 上 GLM-5.2 decode 输出质量回归，DeepSeek-V3.2 MTP accept length 从 3.459 降到 2.462；无 crash、无 NaN，输出仍连贯。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/pull/32616

### INF-0685 `trtllm_nvfp4_gemma4_vproj_scale_missing` TRT-LLM Gemma4 NVFP4 漏复制 v_proj scale

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `var` · 发现 2026-07-29 · 旧号 7.12

- **原因对象**：量化 / tensorrt-llm / NVFP4 权重映射 v_proj scale
- **原因行为**：Gemma4 attention_k_eq_v（共享 K/V 投影）在 NVFP4 下，权重映射未把 k_proj 的 scale 张量复制给 v_proj，fused-QKV loader 取不到 v 的 scale。
- **影响对象**：可用性/进度（报错）
- **影响详情**：TRT-LLM 加载 Gemma4 NVFP4 权重时 crash，报 KeyError: 'v'。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/NVIDIA/TensorRT-LLM/issues/16867
- **关联**：var:INF-0682

### INF-0686 `sglang_nvfp4_sm100_moe_flashinfer` SGLang SM100 NVFP4 MoE 量化 kernel 路由错误

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-07-29 · 旧号 7.13

- **原因对象**：量化 / sglang / NVFP4 auto MoE 后端选择
- **原因行为**：SM100 上 NVFP4 auto MoE 量化未路由到所需的 flashinfer_cutlass backend，而是选到错误的量化 kernel。
- **影响对象**：数值结果；输出内容（静默错误）
- **影响详情**：Blackwell（SM100/B200）上 NVFP4 MoE 专家输出精度退化或错误；通用 Ampere/Ada/Hopper 上通常不可复现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/pull/32229

### INF-0687 `vllm_gemma4_nvfp4_fp4mm_shape_mismatch_sm120` vLLM Gemma-4 NVFP4 FP4 矩阵乘在 sm_120 形状不匹配

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `var` · 发现 2026-07-31 · 旧号 7.14

- **原因对象**：量化 / vllm / flashinfer_scaled_fp4_mm
- **原因行为**：NVFP4 scaled FP4 矩阵乘 kernel flashinfer_scaled_fp4_mm 在 sm_120 子架构上对 Gemma-4 权重布局存在兼容性缺陷，形状不匹配。
- **影响对象**：可用性/进度（报错）
- **影响详情**：RTX 5060 Ti（sm_120）上以 --quantization modelopt 服务 AxionML/Gemma-4-12B-NVFP4 时报 shape mismatch，服务启动失败；无静默错误输出窗口。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/50577
- **关联**：var:INF-0686

### INF-0688 `vllm_minimax_m3_nvfp4_illegal_mem_access` vLLM MiniMax-M3 NVFP4 kernel 非法内存访问

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `var` · 发现 2026-07-31 · 旧号 7.15

- **原因对象**：量化 / vllm / NVFP4 量化 kernel（未定位）
- **原因行为**：NVFP4 量化 kernel 在 MiniMax-M3 的权重/形状布局下发生越界访问；具体 kernel 原文未给出。
- **影响对象**：可用性/进度（报错）
- **影响详情**：vLLM 服务 MiniMax-M3-NVFP4 时触发 CUDA illegal memory access，服务崩溃。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/50557
- **关联**：var:INF-0686

### INF-0689 `onnxruntime_matmulnbits_accuracy_level4_int8_act_massive_activation_wrong_argmax` ORT MatMulNBits int8 激活量化在 CPU 选错 argmax

`故障` · `active` · `model_on` · `config` · `verified` · `NEW` · 发现 2026-08-01 · 旧号 7.16

- **原因对象**：量化 / onnxruntime / MatMulNBits int8 激活量化
- **原因行为**：CPU EP 上 MatMulNBits 在 accuracy_level=4 时先把激活 A 做 int8 量化再做 int8 matmul，对 massive activation/outlier 通道量化过粗。
- **影响对象**：输出内容；输出分布（静默错误）
- **影响详情**：ORT-GenAI 默认导出 accuracy_level=4；Qwen3-0.6B、Phi-3.5-mini 等在 CPU 上贪婪解码时于 near-tie logits 处选错 argmax，token 流与 accuracy_level 0/1/2/3 参考不一致；无 crash、无 NaN。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/microsoft/onnxruntime/issues/29849

### INF-0690 `sglang_dsv4_fused_wqa_wkv_packed_weight_silent_drop` SGLang DS-V4 融合投影加载静默丢弃量化 packed 权重

`故障` · `active` · `infra_off` · `other_stack` · `verified` · `var` · 发现 2026-08-02 · 旧号 7.17

- **原因对象**：量化 / sglang / DS-V4 融合 wq_a+wkv 加载路径
- **原因行为**：融合 wq_a+wkv 加载路径只映射未量化的 .weight 名，checkpoint 的 qweight/qzeros/scales 等 packed 张量无处映射而被静默丢弃，仅留一条无 error 关键字的日志。
- **影响对象**：模型权重；输出内容（静默错误）
- **影响详情**：SGLANG_OPT_FUSE_WQA_WKV=1（默认）下加载 GPTQ/AWQ 等 packed 格式的量化 DS-V4 checkpoint 时，服务正常启动并返回 HTTP 200，但输出是流畅的垃圾；无 crash、无 NaN、无报错，任何 GPU 都受影响。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/33245
- **关联**：var:INF-0681

### INF-0691 `sglang_autoround_gptq_moe_marlin_branch_inverted_silent_triton_fallback` auto-round MoE 委托缺 desc_act，静默回退 Triton

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-08-03 · 旧号 7.18

- **原因对象**：量化 / sglang / AutoRound→MoeWNA16 委托
- **原因行为**：委托给 MoeWNA16Config 的 config 缺 desc_act 键，Marlin 资格检查视缺键为不合格，FusedMoE 专家一律落到 Triton wna16；use_marlin=False 分支还引用未定义变量。
- **影响对象**：性能/资源（性能退化）
- **影响详情**：所有架构的 GPTQ-packed auto-round MoE checkpoint 均触发：无报错，各 rank 日志打 Using default MoE kernel config 而 dense 层走 Marlin，decode 性能静默退化。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/pull/33414

### INF-0692 `vllm_mxfp4_flashinfer_cutlass_weight_layout_w1w3_swap_crash` MXFP4 权重转换缺 FlashInfer CUTLASS 分支，加载崩溃

`故障` · `active` · `infra_off` · `config` · `verified` · `NEW` · 发现 2026-08-04 · 旧号 7.19

- **原因对象**：量化 / vllm / MXFP4 MoE 权重格式转换
- **原因行为**：backend selector 可为标准 MXFP4 checkpoint 选 FlashInfer CUTLASS，但权重格式转换缺对应分支，fused gate/up 仍为 [w1;w3]，而该 backend 期望 [w3;w1]。
- **影响对象**：可用性/进度（报错）
- **影响详情**：MXFP4 checkpoint 且选中 FlashInfer CUTLASS backend 时，模型加载后抛 ValueError，无法起服；非静默。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/51038

### INF-0693 `vllm_nongated_moe_w13_shard_hardcode2_uninit_nan` 非门控 MoE 量化方法硬编码 2 个 w13 shard 致 NaN

`故障` · `active` · `infra_off` · `config` · `verified` · `NEW` · 发现 2026-08-05 · 旧号 7.20

- **原因对象**：量化 / vllm / MoE 量化方法 w13 shard 数
- **原因行为**：非门控 MoE 的 w13 只有一个 shard，14 种 MoE 量化方法却硬编码为 2：第二槽为未初始化内存，scale/bias 过量分配，zero-padding 边界算错、pad 行未清零。
- **影响对象**：模型权重；数值结果（数值异常）
- **影响详情**：仅非门控 MoE（如 NemotronH relu2_no_mul）配受影响量化方法时触发：无即时 crash，未初始化值流入量化权重与 scale，GEMM 产出 NaN，pad 行还污染真实专家行的共享 per-block scale；门控模型 bit-for-bit 不受影响。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/51125

### INF-0694 `vllm_mxfp4_fp32_weight_scale_cast_error` MXFP4 FP32 scale 转换出错且 per-expert 映射缺失

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-08-07 · 旧号 7.21

- **原因对象**：量化 / vllm / MXFP4 量化权重加载
- **原因行为**：部分 MXFP4 checkpoint 把 weight_scale 存为 FP32，加载时 cast 到 uint8 出错；per-expert checkpoint 的权重映射缺失，未路由到 per-expert tensor。
- **影响对象**：模型权重（待定）
- **影响详情**：加载 FP32 weight_scale 或 per-expert 格式的 MXFP4 checkpoint 时出现权重加载错误；原文未说明是报错中断还是静默加载出错误权重。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/51419

### INF-0695 `vllm_int8_w8a8_moe_triton_experts_dtype_unrecognized` TritonExperts 不认 int8 激活，W8A8 MoE 启动崩溃

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-08-07 · 旧号 7.22

- **原因对象**：量化 / vllm / TritonExperts 输入 dtype 断言
- **原因行为**：fused_moe 的 TritonExperts 断言只列 bf16/fp16 等，未把 torch.int8 列为有效输入 dtype，INT8 W8A8 MoE 处理量化激活时断言失败。
- **影响对象**：可用性/进度（报错）
- **影响详情**：INT8 W8A8 MoE 模型走 TritonExperts 时 assert hidden_states.dtype 失败，报 RuntimeError: Engine core initialization failed，引擎无法启动；非静默。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/51411

### INF-0696 `sglang_mxfp4_per_expert_block_scale_zeroed` MXFP4 per-expert 加载静默清零 scale，输出流利乱码

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-08-07 · 旧号 7.23

- **原因对象**：量化 / sglang / FusedMoE per-expert 权重加载
- **原因行为**：per-expert 加载路径未经验证：name remap 把 .scale 改成 fp8 名、dtype 推断错、scale shape 不匹配被静默跳过，致每个 expert 的 block scale 被清零。
- **影响对象**：模型权重；输出内容（静默错误）
- **影响详情**：加载 DeepSeek-V4 MXFP4 per-expert checkpoint 时，模型正常启动、服务并生成流利文本，但不再跟随 prompt；无 crash、无 NaN，日志仅一行 weights not initialized 列表项。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/pull/33541

### INF-0697 `vllm_nvfp4_moe_cutedsl_nongated_tile128_misalign` CuTeDSL NVFP4 非门控 MoE 中间维未按 128 对齐致静默错误

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-08-09 · 旧号 7.24

- **原因对象**：量化 / vllm / 非门控 NVFP4 MoE 权重 pad
- **原因行为**：FlashInfer CuTeDSL NVFP4 MoE 按 128 行 tile 处理中间维，非门控路径未把权重与 scale pad 到 128 行就 swizzle scale，违反 kernel 对齐契约。
- **影响对象**：数值结果；输出内容（静默错误）
- **影响详情**：rank-local intermediate size（由模型架构与并行布局决定，非运行时 batching 参数）不是 128 行 tile 对齐时，MoE 输出静默错误；无 crash。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/51545

### INF-0698 `vllm_rocm_aiter_fp8_mla_short_prompt_rundependent_wrong_answer` AITER FP8 MLA prefill 致短 prompt 答案随运行变化

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-10 · 旧号 7.25

- **原因对象**：量化 / vllm / AITER FP8 MLA prefill 路由
- **原因行为**：ROCm 上 forward_mha() 对所有非 chunked-context 请求（含短 prompt）都选 AITER FP8 prefill，FP8 在短 prompt 上改变 borderline logits。
- **影响对象**：输出分布；输出内容（不可复现）
- **影响详情**：短 prompt（603–698 tokens，Kimi-K3 PP8 跑 GSM8K-100 5-shot）下出现错误答案，重跑同一失败 prompt 可能又答对；无 crash、无 NaN。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/51380

### INF-0699 `trtllm_static_fp8_fused_projection_requant_discards_calibration` TRT-LLM 静态 FP8 融合投影取 max scale 二次量化丢校准

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-08-10 · 旧号 7.26

- **原因对象**：量化 / tensorrt-llm / Cosmos3 静态 FP8 融合投影加载
- **原因行为**：Q/K/V、gate/up 被融合为单个 per-tensor FP8 Linear，只能持有一个 weight scale；loader 保留 max(scales) 并重新量化组内其他成员，在 ModelOpt 校准上多一次舍入。
- **影响对象**：模型权重；数值结果（静默错误）
- **影响详情**：加载静态 FP8 的 Cosmos3 Nano/Super 检查点时，约 56% 的 fused shard（Nano 141/252、Super 252/448）附加误差达数值底限的 7–18 倍；无 crash、无 NaN，输出仍连贯但精度降低。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/NVIDIA/TensorRT-LLM/pull/17476

### INF-0700 `sglang_w4a4_megamoe_fp4_acts_deterministic_nearby_integer_substitution` MegaMoE W4A4 FP4 激活致确定性答错算术并增多空回复

`故障` · `active` · `ambiguous` · `config` · `documented` · `NEW` · 发现 2026-08-10 · 旧号 7.27

- **原因对象**：量化 / sglang / MegaMoE FP4 激活 packing
- **原因行为**：开启 USE_FP4_ACTS=1 与 USE_MXF4_KIND=1 时 MegaMoE 以 FP4 打包激活（W4A4），相对默认 FP8 激活（W4A8）精度下降，在特定 prompt 形状上改变输出；原文未定位到具体代码缺陷。
- **影响对象**：输出内容；输出分布（静默错误）
- **影响详情**：DeepSeek-V4-Flash-0731 8×B200 A/B：12+23=? 时 W4A4 确定性答 36（35 被踢出 top-8），3/3 可复现；512K 上空 completion 10 个 vs 2 个；个别 prompt 出现 1.6% 非 ASCII；TTFT 无差异。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/34179

### INF-0702 `sglang_flashinfer_trtllm_nvfp4_moe_tile192_nan` FlashInfer NVFP4 MoE tile-192 路径输出 NaN

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-12 · 旧号 7.29

- **原因对象**：量化 / flashinfer / NVFP4 MoE tile-192 BMM
- **原因行为**：FlashInfer 0.6.16rc4 之后的版本在 Blackwell 上新暴露 tile-192 TRTLLM_GEN BMM tactic，产生非有限 MoE 输出，tile-128 正确；属依赖侧回归，非 SGLang 特性回归。
- **影响对象**：数值结果；输出内容（数值异常）
- **影响详情**：SGLang flashinfer_trtllm 在线 NVFP4 MoE 在 FlashInfer 0.6.16.post4/0.6.17 下于 SM100/SM103 输出 NaN，GSM8K 200 例得 0.0；0.6.15.post1/0.6.16rc4 正常，4×B200 CI 可复现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/34629

### INF-0703 `vllm_mistral_fp8_pixtral_image_input_nan_compiled` vLLM FP8-dynamic Pixtral 编译模式下图像请求返回 NaN

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-08-12 · 旧号 7.30

- **原因对象**：量化 / vllm / FP8 Pixtral 图像编译路径
- **原因行为**：根因未定位：仅 LM Linear 层为 FP8，视觉塔/projector/lm_head 为 bf16；原文推测 FP8 量化边界与 torch.compile/CUDA graph 在图像 forward 上交互产生 NaN。
- **影响对象**：数值结果；输出内容（数值异常）
- **影响详情**：FP8-dynamic Mistral-Small-3.1 检查点下，任何含图像的请求返回 NaN，API 序列化报 Out of range float values are not JSON compliant: nan；文本请求正常，--enforce-eager 可工作，引擎不 crash。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/52034

### INF-0704 `vllm_kimi_k3_nvfp4_b300_reasoning_channel_degenerate_output` Kimi-K3-NVFP4 于 B300 上 reasoning 通道输出退化

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-08-13 · 旧号 7.31

- **原因对象**：量化 / vllm / NVFP4 推理路径（根因待定）
- **原因行为**：根因未定位：原文仅称 NVFP4 量化与特定版本栈组合（v0.27.0、FlashInfer 0.6.16.post3 等）导致，推测涉及 NVFP4 dequant 路径与 reasoning 相关 MoE expert 的数值交互。
- **影响对象**：输出内容（质量退化）
- **影响详情**：RedHatAI/Kimi-K3-NVFP4 在 8×B300、TP8、--kv-cache-dtype=fp8 下 reasoning channel 输出退化、不连贯，无 crash；生产部署已因此回滚。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/51798

### INF-0705 `sglang_fp8_mla_kv_scale_propagation_inconsistent` FP8 MLA KV cache 的 BMM2 漏乘 V descale

`故障` · `active` · `infra_off` · `partial` · `verified` · `NEW` · 发现 2026-08-13 · 旧号 7.32

- **原因对象**：量化 / sglang / FP8 MLA KV descale 传播
- **原因行为**：MLA 同一 latent row 同时作 BMM1 的 K 与 BMM2 的 V；部分 MLA/DSA 路径写 cache 用 unit scale，读时 BMM1 只恢复 K scale，BMM2 省略 V scale。
- **影响对象**：数值结果（静默错误）
- **影响详情**：使用非 unit 校准 scale 的 FP8 MLA KV cache 并走 TRT-LLM MLA/DSA backend 时，attention 输出静默不一致；无 crash。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/pull/34765

### INF-0707 `trtllm_fp8_moe_zero_block_amax_unclamped_infinite_output` FP8 MoE 激活量化未 clamp 零 block amax 致 NaN

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-08-13 · 旧号 7.34

- **原因对象**：量化 / tensorrt-llm / FP8 MoE 激活量化 block amax
- **原因行为**：FP8 MoE 激活量化未对 block amax 设下限，全零激活 block 的 amax 为 0，按 amax/fp8_max 得零 scale，进而产生零 FP8 输出或 NaN/inf scale。
- **影响对象**：数值结果；输出内容（数值异常）
- **影响详情**：大型混合注意力 FP8 serving 中 MoE 出现全零激活 block 时，FP8 输出错误或为 NaN。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/NVIDIA/TensorRT-LLM/pull/17635

### INF-0708 `vllm_nvfp4_monolithic_moe_backend_non_gated_relu2_semantic_corruption_sm100` monolithic NVFP4 MoE 对非门控 ReLU² 语义错误

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-14 · 旧号 7.35

- **原因对象**：量化 / vllm / vLLM NVFP4 MoE 后端自动选择
- **原因行为**：monolithic NVFP4 MoE 内核 trtllm_fp4_block_scale_moe 对非门控 ReLU² 激活的 MoE 计算语义错误；vLLM 在 SM100 上默认自动选中该 FLASHINFER_TRTLLM 后端。
- **影响对象**：输出分布；输出内容（静默错误）
- **影响详情**：B200（SM100）跑 Nemotron-3-Nano-30B-A3B-NVFP4 时输出语法流利但语义损坏，greedy 下 100% 确定性复现；切 flashinfer_cutlass 后端即恢复，同 checkpoint 在 SM120 上正确；无 crash。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/52308

### INF-0709 `vllm_rocm_situ_moe_dual_flag_layout_mismatch_silent_garbage` SiTU MoE 库/框架双 flag 错配致静默乱码

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-08-15 · 旧号 7.36

- **原因对象**：量化 / vllm / SiTU MoE 权重布局与 kernel 选择
- **原因行为**：aiter 侧 AITER_SITUV2_A8W4 选 interleave 布局的 MoE kernel，权重 interleave shuffle 却由默认关闭的 vLLM 侧 flag 控制，两者无一致性校验。
- **影响对象**：输出内容；数值结果（静默错误）
- **影响详情**：MI350X 上 Kimi-K3 SiTU MoE 设 AITER_SITUV2_A8W4=1 而 VLLM_ROCM_USE_AITER_MOE_SITUV2_A8W4 未设时输出静默乱码（2+2= 得 0. 4. 0. 0. 4），无报错、无 warning。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/52442

### INF-0710 `hf_metalconfig_2d_kernel_3d_batched_input_corruption` MetalLinear 对 3D batched 输入只算首个样本

`故障` · `active` · `infra_off` · `yes` · `documented` · `NEW` · 发现 2026-08-15 · 旧号 7.37

- **原因对象**：量化 / transformers / MetalLinear.forward 输入展平
- **原因行为**：MetalConfig 量化的 MetalLinear.forward 把 batched 生成的 3D 输入直接传给按 2D 语义计算的 affine_qmm_t kernel，只算第一个 batch 元素，其余输出未写入。
- **影响对象**：数值结果；输出内容（静默错误）
- **影响详情**：Apple Silicon MPS 上 batch_size>1 的 batched 生成时，第一个序列正确，后续序列输出乱码或一串感叹号；无 crash，逐样本测试发现不了。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/huggingface/transformers/pull/47970

### INF-0711 `sglang_fp8_marlin_bias_tile_permutation_order_mismatch` FP8 Marlin bias 按 tile 序置换却按逻辑序外加

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-08-16 · 旧号 7.38

- **原因对象**：量化 / sglang / dense FP8 Marlin bias 处理
- **原因行为**：准备阶段把 bias permute 进 Marlin tile 布局，但 dense wrapper 不把 bias 传入 Marlin GEMM，而在 kernel 返回后外加，外加需要的是原始逻辑输出通道序，两者布局不一致。
- **影响对象**：数值结果；输出内容（静默错误）
- **影响详情**：所有带 bias 的 dense FP8 Marlin 投影（如 Qwen2 QKV projection）输出静默错误，量化模型与对应 FP16/BF16 模型输出系统性偏离；无 crash。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/pull/35020

### INF-0712 `vllm_batch_invariant_nvfp4_emulation_device_mismatch_crash` 批不变模式下 NVFP4 emulation 查找表设备不匹配崩溃

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-08-17 · 旧号 7.39

- **原因对象**：量化 / vllm / break_fp4_bytes 查找表
- **原因行为**：VLLM_BATCH_INVARIANT=1 关闭硬件 NVFP4 kernel、强制 EMULATION MoE 后端；该路径的共享工具 break_fp4_bytes 把查找表留在 CPU，却用 GPU 索引直接索引。
- **影响对象**：可用性/进度（报错）
- **影响详情**：VLLM_BATCH_INVARIANT=1 加任意 modelopt_fp4/NVFP4 模型时，engine init 报 device mismatch RuntimeError，容器 crash-loop；不设该 env 时同一模型正常加载。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/52407

### INF-0714 `sglang_dsv4_fp4_dequant_env_short_circuited_by_native_branch` ROCm 下 FP4 dequant 开关被 AITER 分支短路致起服失败

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `var` · 发现 2026-08-17 · 旧号 7.41

- **原因对象**：量化 / sglang / Fp8MoEMethod 块量化权重后处理
- **原因行为**：Fp8MoEMethod 块量化后处理在检查 dequant_fp4_to_fp8 前就从 AITER 原生 FP4 分支 return，原 dequant 块又因 _is_fp8_fnuz 在 ROCm 上不可达，开关不生效。
- **影响对象**：可用性/进度（报错）
- **影响详情**：ROCm + AITER 加载 FP4 checkpoint 并设 SGLANG_DSV4_FP4_DEQUANT=1 时，routed 专家权重仍为 FP4，起服报 Unsupported kernel config for moe heuristic dispatch。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/35122
- **关联**：var:INF-0709

### INF-0715 `vllm_mm_encoder_tp_mode_data_nvfp4_multi_image_crash` vLLM mm-encoder data 并行 × NVFP4 多图请求崩溃

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-08-17 · 旧号 7.42

- **原因对象**：量化 / vllm / encoder DP 分片多图拼接×NVFP4
- **原因行为**：多模态 encoder 以 --mm-encoder-tp-mode data 数据并行切分时，多图拼接路径上 shard 的激活布局与 NVFP4 kernel 期望的布局不匹配。
- **影响对象**：可用性/进度（报错）
- **影响详情**：多模态模型（Qwen3.8-27B NVFP4）开启 --mm-encoder-tp-mode data、TP=4、sm_120 时，多图请求触发引擎 crash；各维度单独使用均正常，仅该组合崩溃。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/52654

### INF-0716 `sglang_deepep_fp8_ue8m0_mixed_scale_first_request_assert` SGLang deepep MoE 权重 scale 未转 UE8M0 致崩溃

`故障` · `active` · `infra_off` · `partial` · `documented` · `var` · 发现 2026-08-18 · 旧号 7.43

- **原因对象**：量化 / sglang / deepep FP8 MoE 权重 scale
- **原因行为**：Blackwell deepep-fp8 路径未对 MoE 专家权重 scale 做 UE8M0 重量化：激活 scale 为 UE8M0 打包，权重 scale 仍为原生 fp32，而 DeepGEMM 要求纯指数格式。
- **影响对象**：可用性/进度（报错）
- **影响详情**：GB300/sm_103 上 deepep FP8 MoE（DeepSeek 系）：启动与 DeepGEMM warmup 均成功（warmup 用全 1 scale 恰好合法），首个真实请求的非 2 的幂 scale 触发 device assert，所有 TP rank 被杀。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/35388
- **关联**：var:INF-0439

### INF-0717 `sglang_compressed_tensors_lm_head_quant_dispatch_missing` SGLang CT 量化 lm_head 未分发，输出恒为 token 0

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-08-18 · 旧号 7.44

- **原因对象**：量化 / sglang / CT get_quant_method 层类型分发
- **原因行为**：get_quant_method 只分发 LinearBase/FusedMoE，lm_head 落回非量化 embedding 方法建普通 weight；检查点的 weight_packed 无人读，head 用未初始化内存。
- **影响对象**：模型权重；输出内容（静默错误）
- **影响详情**：加载带量化 lm_head 的 compressed-tensors 检查点时，服务正常启动、无任何错误，每个请求都返回重复的 token 0（“!!!!!!”）；与硬件无关，任何设备可复现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/35358；https://github.com/sgl-project/sglang/pull/35228

### INF-0719 `vllm_triton_blockfp8_ktile_scale_oob_pair` vLLM Triton 量化 GEMM K-tile 越界读与跨组错 scale

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-08-19 · 旧号 7.46

- **原因对象**：量化 / vllm / Triton 量化 GEMM K-tile 处理
- **原因行为**：(1) GPTQ/AWQ MoE Triton 内核权重 load 缺 k_mask，K 非整块时末次迭代越界读；(2) block-FP8 W8A8 GEMM 每 K tile 只取一个 scale，块长不整除 group_k 时选错。
- **影响对象**：数值结果（静默错误）
- **影响详情**：路径 1 因 scale 被 mask 为 0 数值恰好抵消，目前无症状但为潜伏越界读；路径 2 仅在非默认 autotune 配置（BLOCK_SIZE_K > group_k）下触发，跨两个量化组的 tile 用错 scale，数值静默出错；当前默认配置未触发。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/52576

### INF-0720 `vllm_blockfp8_tp_subblock_shard_load_fail` vLLM block-FP8 检查点在 TP 分片小于量化块时加载崩溃

`故障` · `active` · `infra_off` · `partial` · `documented` · `var` · 发现 2026-08-19 · 旧号 7.47

- **原因对象**：量化 / vllm / block-FP8 TP 分片子块 padding
- **原因行为**：block-FP8 线性层加载缺通用的 K、N 维子块 padding：rank 本地 K=64 不被 block_k=128 整除即拒绝；merged-column 本地 N 小于 block_n 时 scale 加载越界。
- **影响对象**：可用性/进度（报错）
- **影响详情**：Qwen3.6-35B-A3B-FP8 在 TP8（8×H100）下每 rank K 分片仅 64，加载期报 ValueError；仅 pad K 后又在 load_merged_column_weight 越界报错；显式选 Marlin 后端同样失败。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/52997
- **关联**：var:INF-1134

### INF-0721 `vllm_int4_per_token_head_kv_quant_kernel_roundtrip_wrong` vLLM int4 per-token-head KV 写入内核往返错误

`故障` · `active` · `infra_off` · `yes` · `documented` · `NEW` · 发现 2026-08-20 · 旧号 7.48

- **原因对象**：量化 / vllm / int4 KV 量化写入 Triton 内核
- **原因行为**：int4 per-token-head KV 写入内核的 nibble 打包/解包往返偏差远超容差；int8/fp8 分支正常，部分参数组合逐运行 pass/fail 翻转，提示位操作含未定义或竞态成分。
- **影响对象**：KV状态；输出内容（静默错误）
- **影响详情**：单测确定性复现：head_size=64、num_kv_heads=1、num_tokens=42 时 54.7% 元素错配；head_size=128 或 num_kv_heads=4 时 pass/fail 翻转。线上启用该模式则写入 cache 的值即错，输出静默退化。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/53086

### INF-0722 `vllm_draft_model_quant_config_packed_mapping_missing` vLLM draft 模型量化配置缺 packed 映射，加载期崩溃

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-08-20 · 旧号 7.49

- **原因对象**：量化 / vllm / get_draft_quant_config
- **原因行为**：draft 模型在 get_draft_quant_config 自建量化配置，不经 configure_quant_config，packed_modules_mapping 恒空，fused 投影无法映射回子模块而按未量化构建。
- **影响对象**：可用性/进度（报错）
- **影响详情**：量化 draft 模型（如 compressed-tensors FP8）开启推测解码时，加载期 weight_scale 无目标，报 AttributeError；非 fused 层量化正常，报错表面与量化无关；无需 GPU 即可复现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/53116

### INF-0723 `vllm_ue8m0_group_quant_backend_scale_divergence` vLLM FP8 组量化双后端 UE8M0 scale floor 不一致

`故障` · `active` · `infra_off` · `config` · `verified` · `NEW` · 发现 2026-08-22 · 旧号 7.50

- **原因对象**：量化 / vllm / FP8 组量化 Triton fallback
- **原因行为**：use_ue8m0=True 时，C++ op 把 scale floor 到 1e-10 再上取 2 的幂，Triton fallback 只 floor absmax、不 floor scale，两后端对近零组的 scale 不一致。
- **影响对象**：数值结果（静默错误）
- **影响详情**：全零组 C++ 得 2^-33、fallback 得 2^-42；absmax 低于约 2.61e-8 的组两后端给出不同 scale 与 fp8 码。连续输入走 C++、非连续走 Triton，同一算子随输入连续性 bitwise 分歧，无 crash、无 NaN。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/53339

### INF-0724 `vllm_turboquant_fp16_zero_point_overflow_poisoned_cache` TurboQuant KV 的 fp16 zero-point 溢出注入 ∞

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-08-22 · 旧号 7.51

- **原因对象**：量化 / vllm / triton_turboquant_store
- **原因行为**：triton_turboquant_store 不论输入 dtype 一律用 fp16 存 value scale 与 zero-point；bf16 输入含 |min|>65504 的 outlier 时 zero-point 溢出。
- **影响对象**：KV状态；输出内容（数值异常）
- **影响详情**：--kv-cache-dtype turboquant_* 且 bf16 KV 值含 attention-sink 级 outlier 时，store→decode 重建值为 ∞，被投毒的 cache line 进入 attention，输出质量不可预测退化；无 crash，probe 级可复现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/53334

### INF-0725 `sglang_compressed_tensors_kv_cache_scheme_scales_dropped` SGLang 静默丢弃 CT 检查点的 KV cache scale

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-08-22 · 旧号 7.52

- **原因对象**：量化 / sglang / CT 配置与 RadixAttention 分支；权重/检查点 / sglang / qwen3_5 weight mapper
- **原因行为**：三缺口叠加：from_config 不转发 kv_cache_scheme；get_quant_method 无 RadixAttention 分支，不建 k/v scale；weight mapper 只认 ModelOpt 式命名。
- **影响对象**：KV状态；输出内容（静默错误）
- **影响详情**：compressed-tensors 检查点声明 kv_cache_scheme 并用 --kv-cache-dtype fp8_e4m3 时，校准 k_scale/v_scale 被丢弃，KV 按无 scale 量化、精度系统性劣化；无 crash、无告警，仅加载日志一行 not found。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/pull/35455

### INF-0726 `vllm_cutlass_fp8_kernel_pre_sm89_ungated_autoselect` vLLM CUTLASS FP8 内核未查算力，pre-Ada 上被自动选中

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-08-22 · 旧号 7.53

- **原因对象**：量化 / vllm / CUTLASS FP8 kernel 能力门控
- **原因行为**：CUTLASS FP8 linear kernel 的 is_supported() 只查平台为 CUDA、不探测算力，而底层能力检查拒绝 SM89 以下；CUDA 候选列表中 CUTLASS 排在 Marlin 之前。
- **影响对象**：数值结果；可用性/进度（待定）
- **影响详情**：pre-SM89 GPU（A100、30 系及更早）运行 FP8 模型时 CUTLASS 赢得自动选择且无法回退，走不支持路径，结果为 crash 或错误计算（视 CUTLASS 内部行为）；移除 VLLM_TEST_FORCE_FP8_MARLIN 后该路径变为可达。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/53376

### INF-0727 `hf_finegrainedfp8_modules_to_not_convert_wrapper_prefix_mismatch` transformers FP8 排除表前缀失配，bf16 模块被转 fp8

`故障` · `active` · `infra_off` · `config` · `verified` · `NEW` · 发现 2026-08-26 · 旧号 7.54

- **原因对象**：量化 / transformers / FineGrainedFP8 排除表匹配
- **原因行为**：FineGrainedFP8 的 modules_to_not_convert 按多模态 wrapper 路径书写，text-only 加载时模块名缺 language_model 段，前缀匹配全部落空；已有键重映射不作用于排除表。
- **影响对象**：模型权重；输出内容（质量退化）
- **影响详情**：AutoModelForCausalLM 加载 Qwen3.8-Flash-Next-FP8 时，应保 bf16 的 PLE 投影、hyper-connection mixer、linear-attn 模块被无 scale cast 到 fp8；加载零报错、无警告，仅生成质量下降。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/huggingface/transformers/issues/48349

### INF-0728 `sglang_modelslim_fused_gdn_scheme_fallback_int8_as_bf16_garbage` SGLang ModelSlim 回退把 int8 权重当 bf16 致输出垃圾

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-08-26 · 旧号 7.55

- **原因对象**：量化 / sglang / ModelSlim fused 层回退路径
- **原因行为**：ModelSlim 不识别 fused GDN 线性层的 scheme，回退“不量化”时把 int8 权重原样拷进 bf16 参数（位型重解释）；vision 塔被强制 bf16 后，NPU 量化 matmul 收到 3-D 输入。
- **影响对象**：模型权重；输出内容；可用性/进度（静默错误）
- **影响详情**：Ascend NPU 上加载 ModelSlim W8A8 的 fused GDN 混合模型：48 层报 Unsupported scheme 警告后，语言路径输出全为垃圾且无报错；vision 路径的 3-D matmul 直接 crash。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/36423

### INF-0729 `vllm_xpu_autoround_int4_ark_woq_concurrent_garbage_no_optout` vLLM XPU AutoRound int4 并发≥2 时输出变多语言乱码

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-27 · 旧号 7.56

- **原因对象**：量化 / vllm / XPU ARK WOQ int4 后端
- **原因行为**：按 checkpoint 自动选中的 ARK WOQ int4 量化后端在并发请求路径上计算出错（具体根因待定），且没有配置项可退出该后端。
- **影响对象**：输出内容（静默错误）
- **影响详情**：Arc Pro B70 单卡服务 Qwen3.6-27B int4-AutoRound：温度 0 单请求逐字正确，≥2 个并发相同请求的响应全部退化为多语言 garbage，仍返回 HTTP 200，无报错、无日志；9B 模型部分损坏，严重度随规模放大。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/53211

### INF-0730 `vllm_dsv4_ue8m0_flashinfer_trtllm_moe_backend_distribution_distortion` vLLM DSV4 UE8M0 默认 TRTLLM MoE 后端致分布畸变

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-08-27 · 旧号 7.57

- **原因对象**：量化 / vllm / FLASHINFER_TRTLLM FP8 MoE
- **原因行为**：UE8M0 block-scale MoE 走 flashinfer trtllm_fp8_block_scale_moe 时与 deep_gemm 数值显著分歧（根因待定）；未指定 --moe-backend 时默认选中此后端。
- **影响对象**：输出分布；输出内容（静默错误）
- **影响详情**：4×GB200 上 DSV4-Pro-Base 英文 prompt 的 top-5 logprobs 被瑞典/芬兰/立陶宛语素占据，perplexity 虚高、base model 评测失效；无 crash、无 NaN；改用 deep_gemm 即恢复，DSv3/Kimi-K2 不复现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/53411

### INF-0731 `sglang_mxfp4_moe_runner_dispatch_else_swallow_nine_backends_fail_late` SGLang mxfp4 runner 分发 else 吞掉 9 个后端值

`故障` · `active` · `infra_off` · `other_stack` · `verified` · `NEW` · 发现 2026-08-27 · 旧号 7.58

- **原因对象**：量化 / sglang / create_moe_runner 分发链
- **原因行为**：create_moe_runner 链尾 else 仅 pass：除 flashinfer_mxfp4 外，另 9 个合法 backend 值（ascend、cutlass 等）落入后既不建 runner 也无 bypass 标志。
- **影响对象**：可用性/进度（报错）
- **影响详情**：mxfp4 检查点配这 9 个 --moe-runner-backend 值之一时，加载期零输出、健康检查通过，首个 MoE forward 才报 AttributeError: no attribute 'runner'，错误信息与根因（不支持的配置组合）无关。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/36569

### INF-0733 `vllm_nvfp4_moe_weight_only_uninitialized_input_scale_zeroes_all_experts` vLLM NVFP4 MoE 未初始化 input_scale 使专家输出归零

`故障` · `active` · `infra_off` · `config` · `verified` · `NEW` · 发现 2026-08-28 · 旧号 7.60

- **原因对象**：量化 / vllm / ModelOptNvFp4FusedMoE
- **原因行为**：接受无 input_scale 的 weight-only NVFP4 检查点，w13_input_scale 由 torch.empty 注册且从不填充，加载后不校验即折进反量化 alpha 与激活 scale（实测为 0.0）。
- **影响对象**：数值结果；输出内容（静默错误）
- **影响详情**：weight-only modelopt NVFP4 MoE 检查点配 flashinfer_cutlass backend 时，专家输出恒零，每个 prompt 返回单 token 重复，无错误、无 NaN、无警告；marlin backend 正确；跨硬件复现，易误归因为 GB10。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/54189

### INF-0734 `vllm_quark_fse_check_substring_overmatch_shared_expert_gate_silent_disable` vLLM Quark FSE 检查子串过宽误命中 gate，共享专家融合静默失效

`故障` · `active` · `infra_off` · `config` · `verified` · `NEW` · 发现 2026-08-28 · 旧号 7.61

- **原因对象**：量化 / vllm / Quark FSE 兼容性检查 helper
- **原因行为**：FSE 兼容性检查在 Quark exclude 列表里搜子串 shared_expert（#51695 重构丢了尾部句点），误命中按设计总被排除的 shared_expert_gate 而判为不兼容。
- **影响对象**：性能/资源（性能退化）
- **影响详情**：ROCm/AITER 上开启 VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS 跑 Quark MXFP4 Qwen MoE 检查点时，FSE 被拒用、走 unfused shared expert 路径；无失败、精度不变，仅每 MoE 层一条启动 warning。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/54206

### INF-0735 `sglang_gptq_marlin_moe_bf16_scale_dtype_and_tp_descact_w2_sharding_nan` SGLang GPTQ Marlin MoE scale dtype 与分片错误

`故障` · `active` · `infra_off` · `other_stack` · `verified` · `NEW` · 发现 2026-08-28 · 旧号 7.62

- **原因对象**：量化 / sglang / GPTQ Marlin MoE scales 加载
- **原因行为**：(1) w13/w2 scales 恒按 float16 分配，不跟随 params_dtype；(2) desc_act=True 且 TP>1 时每 rank 需完整 w2 scale 表，现逻辑却加载 TP 分片版本。
- **影响对象**：输出分布；数值结果；可用性/进度（数值异常）
- **影响详情**：BF16 模型下 Marlin MoE kernel 因 hidden_states 与 scale dtype 失配断言 crash；act-order × TP 下反量化读到错位 scale 产生 NaN 并传到 logits，采样器报 inf/nan 错误。RTX 4090×2 TP2 复现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/36822

### INF-0736 `vllm_compressed_tensors_config_collapses_gdn_linear_attention_token_distinctness` vLLM 存在 CT 量化配置即令未量化 GDN 层丢失 token 区分度

`故障` · `active` · `ambiguous` · `partial` · `documented` · `NEW` · 发现 2026-08-28 · 旧号 7.63

- **原因对象**：量化 / vllm / CT 配置下 GDN linear_attn 路径
- **原因行为**：只要存在 compressed-tensors 量化配置（即使 GDN Linear 均为 UnquantizedLinearMethod），GDN linear-attention 层的包装或路径选择即被改变；具体机制原文未定位。
- **影响对象**：隐状态；输出内容（静默错误）
- **影响详情**：hybrid GDN+MoE 模型换用 CT W4A16 配置后，Layer-0 linear_attn 输出 self-cosine 从 0.05 升至 0.78、范数正常，后续层坍塌；输出多语言重复 soup，无 crash、无 NaN；warmup 可过；fp8 配置正常。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/54252

### INF-0737 `vllm_modelopt_nvfp4_invalid_utf8_token_ids_ct_clean` vLLM ModelOpt NVFP4 检查点生成无效 UTF-8 token

`故障` · `active` · `infra_off` · `partial` · `verified` · `NEW` · 发现 2026-08-29 · 旧号 7.64

- **原因对象**：量化 / vllm / ModelOpt NVFP4 检查点格式语义
- **原因行为**：vLLM 对 ModelOpt NVFP4 检查点格式语义的处理有误（具体环节待定）；同精度、同 kernel、同 KV 布局的 compressed-tensors 版本无此问题。
- **影响对象**：输出内容；输出分布（静默错误）
- **影响详情**：SM120 上生成韩文等多字节文本时，ModelOpt 版输出的 token id 不构成合法 UTF-8（6 次运行出现 62–94 个 U+FFFD，CT 版为 0）；HTTP 200、吞吐正常、英文基本正常，temp=0 也命中，易误判为渲染或客户端问题。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/54150

### INF-0738 `vllm_quark_identity_packed_modules_mapping_disables_layer_quant_config` vLLM Quark 恒等 packed 映射使逐层量化配置回退全局

`故障` · `active` · `infra_off` · `partial` · `verified` · `NEW` · 发现 2026-08-31 · 旧号 7.65

- **原因对象**：量化 / vllm / Quark 逐层量化配置查找
- **原因行为**：Quark 两条配置查找路径把 shard_name == layer_name 当作“无 shard 信息”哨兵，恒等映射条目即使在 layer_quant_config 中有自身条目也回退 global_quant_config。
- **影响对象**：模型权重；可用性/进度（报错）
- **影响详情**：Quark 混合精度检查点 + 含恒等 packed_modules_mapping 的模型（plamo3/glm4_1v/molmo 等）：该层按全局 scheme 构建，无警告无日志，直到加载期 shape mismatch 报错；若两 scheme 张量形状相同则可能静默精度错配。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/54641

### INF-0739 `vllm_modelopt_nvfp4_ple_fp8_embedding_method_never_selected` vLLM NVFP4 带 FP8 PLE 表检查点因方法未选中加载失败

`故障` · `active` · `infra_off` · `partial` · `documented` · `var` · 发现 2026-09-01 · 旧号 7.66

- **原因对象**：量化 / vllm / PLE embedding 量化方法选择
- **原因行为**：量化方法分发按主配置 NVFP4 为 ngram_embedding 建 weight loader；为 FP8 PLE 表设计的 Qwen4ExpPLEFp8EmbeddingMethod 已存在，但选择条件不覆盖该组合。
- **影响对象**：可用性/进度（报错）
- **影响详情**：加载带 FP8 量化 PLE 表的 ModelOpt NVFP4 检查点（如 Qwen3.8-Flash-Next-NVFP4）时，FP8 布局的 weight_scale 无加载路径，读到 203/206 分片后报 ValueError；nightly 与 08-26 构建均复现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/54765
- **关联**：var:INF-1141

### INF-0742 `vllm_modelopt_nvfp4_moe_w13_mismatched_gscale_warn_then_use_anyway` NVFP4 MoE w13 仅用 gate 侧 scale 致 up 投影错缩放

`故障` · `active` · `infra_off` · `config` · `verified` · `NEW` · 发现 2026-09-04 · 旧号 7.69

- **原因对象**：量化 / vllm / NVFP4 MoE w13 scale 融合
- **原因行为**：融合 gate/up 为 w13 时无条件取 gate 侧 per-tensor global scale；allclose 检出 w1/w3 scale 不匹配后只 warning_once，仍用 gate scale 反量化 w13。
- **影响对象**：模型权重；数值结果（静默错误）
- **影响详情**：gate/up scale 不匹配的 checkpoint（如社区再量化的 GLM-5.3-Flash-ABLITERATED-NVFP4，69.1% 对不匹配、比值最大 10）上 up 投影静默错值，43 层 MoE 全受影响、整体质量塌方；官方 checkpoint 0% 不匹配，自测不可见。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/54974

### INF-0743 `vllm_nvfp4_qwen38_24t_a95b_ampere_serving_unreachable` 官方 NVFP4 Qwen3.8-2.4T 在 A100 上无法启动

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-09-04 · 旧号 7.70

- **原因对象**：量化 / vllm / linear 后端自动选择、NVFP4×SM80
- **原因行为**：多量化方法 checkpoint 逐 kernel 做能力探测：SM80 上自动选中只收 INT8 的 CutlassFP8ScaledMMLinearKernel；NVFP4 MoE×Ampere 组合也未声明支持、未验证。
- **影响对象**：可用性/进度（报错）
- **影响详情**：32×A100（TP16×PP2）nightly 栈服务 nvidia/Qwen3.8-2.4T-A95B-NVFP4 时 profile run crash 或 backend 拒绝，无法完成启动；无静默形态。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/55010

### INF-0744 `vllm_cutlass_fp8_ampere_autoselect_production_manifestation` CUTLASS FP8 在 A100 被自动选中致 NVFP4 模型启动崩溃

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `var` · 发现 2026-09-04 · 旧号 7.71

- **原因对象**：量化 / vllm / CutlassFP8ScaledMM 能力探测
- **原因行为**：CutlassFP8ScaledMMLinearKernel 的 is_supported 不查算力，在 pre-Ada GPU 上自动选择胜出，把 E4M3 操作数喂给只收 INT8 的 SM80 dispatcher。
- **影响对象**：可用性/进度（报错）
- **影响详情**：32×A100 加载官方 Qwen3.8-2.4T-A95B-NVFP4 时日志显示选中 CutlassFP8ScaledMMLinearKernel，随后 profile run 启动期 crash；含 FP8 linear 的模型在 pre-Ada GPU 上均可触发。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/55008
- **关联**：var:INF-0726

### INF-0745 `vllm_modelopt_mixed_precision_fp8_block_scales_mtp_experts_undispatched` ModelOpt 混合精度 FP8 块 scale 专家与 MTP 层加载失败

`故障` · `active` · `infra_off` · `config` · `verified` · `NEW` · 发现 2026-09-05 · 旧号 7.72

- **原因对象**：量化 / vllm / RoutedExperts 量化方法分派；模型定义 / vllm / Qwen4Exp MTP 层索引命名
- **原因行为**：RoutedExperts if-chain 缺 FP8_BLOCK_SCALES，返回 None 使专家层建成非量化；MTP 层名按 num_hidden_layers+idx 构建，与量化配置的 mtp.layers.0 错位。
- **影响对象**：可用性/进度（报错）
- **影响详情**：加载 nvidia/Qwen3.8-Flash-Next-NVFP4（含 MTP）即报 AttributeError（缺 w2_weight_scale_inv）；两处都修才能加载，补丁后 MTP3 接受率 59.1%。缺陷 (a) 若无参数缺失报错路径，未量化专家层可能静默以 BF16 服务。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/55496

### INF-0746 `vllm_manual_nvfp4_fusion_scale_buffer_unpadded_oob_write` 手工 SiLU+NVFP4 fusion scale 缓冲未 pad 致越界写

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-09-05 · 旧号 7.73

- **原因对象**：量化 / vllm / SiLU+NVFP4 融合 scale 分配
- **原因行为**：#51415 手工 fusion 按 [num_tokens, ceil(K/64)*4] 分配 scale，而 kernel 按 128×4 swizzle 以整 128 行 tile 写 scale，分配未 pad 到 tile 边界。
- **影响对象**：可用性/进度；数值结果（报错）
- **影响详情**：token 数非 128 倍数（小 batch 或末尾残 tile）时越界写：CI 在 silu_mul_cvt_fp16_to_fp4 报 WARP_ILLEGAL_ADDRESS，#51415 合入后连败 17 次；是否崩取决于 heap 布局，也可能静默损坏相邻缓冲。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/55456

### INF-0747 `vllm_cpu_wna16_int4_gemm_avx512_compile_guard_capability_mismatch` CPU WNA16 int4 GEMM 未查 AVX-512 致 AVX2 崩溃

`故障` · `active` · `infra_off` · `config` · `documented` · `var` · 发现 2026-09-06 · 旧号 7.74

- **原因对象**：量化 / vllm / CPUWNA16LinearKernel 能力探测
- **原因行为**：cpu_gemm_wna16 只在 AVX512F/riscv_v/s390x 下编译注册，而 CPUWNA16LinearKernel.can_implement() 只查 is_cpu、权重 dtype 与形状，不查 AVX-512。
- **影响对象**：可用性/进度（报错）
- **影响详情**：AVX2-only x86（如 i7-6700HQ）+ 官方 +cpu wheel 运行 AWQ/GPTQ int4 模型时，加载与预热成功，首次 forward 才报 AttributeError（_C 无 cpu_gemm_wna16），服务看似启动完成后才死。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/55613
- **关联**：var:INF-0726

### INF-0748 `vllm_cutlass_c3x_scaled_mm_packed_stride_reconstruction_ignores_leading_strides` cutlass_scaled_mm 丢弃 leading stride 静默错值

`故障` · `active` · `infra_off` · `yes` · `verified` · `NEW` · 发现 2026-09-06 · 旧号 7.75

- **原因对象**：量化 / vllm / C3x caller stride 构造
- **原因行为**：op 入口接受任意 leading stride，但 cutlass_gemm_caller.cuh 用 make_cute_packed_stride 按 shape 重建 packed stride，丢弃实际 stride。
- **影响对象**：数值结果（静默错误）
- **影响详情**：A、B 或 output 任一为 sliced/padded 视图（如 1024×1024 上取 512×512）时 w8a8 int8 GEMM 约 8.1% 元素错（max abs err 3.04），全 packed 为 0；H800 上 0.28.0 wheel 与 main 均复现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/55534

### INF-0749 `trtllm_nvfp4_gate_up_weight_scale2_max_reconcile_silent_accuracy_loss` TRT-LLM NVFP4 MoE 取 max 调和 scale 致精度损失

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `var` · 发现 2026-09-06 · 旧号 7.76

- **原因对象**：量化 / tensorrt-llm / MoE 加载器 scale_2 调和
- **原因行为**：MoE 权重加载器对同一专家 w1/w3 不同的 weight_scale_2 一律取 max() 调和并 warn；为单 alpha 后端设计的策略被推广到全部后端，含原生支持双标量的 trtllm-gen。
- **影响对象**：模型权重；数值结果（静默错误）
- **影响详情**：加载 gate/up weight_scale_2 不等的 NVFP4 MoE checkpoint 后，较小 scale 一侧量化步长被缩错，产生静默精度损失；无 crash，仅有 warning。两半相等时行为不变。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/NVIDIA/TensorRT-LLM/pull/18772
- **关联**：var:INF-0742

### INF-0750 `trtllm_moe_op_provider_resolved_before_per_layer_quant_config` TRT-LLM MoE 先于逐层量化配置选 provider 致崩溃

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `var` · 发现 2026-09-06 · 旧号 7.77

- **原因对象**：量化 / tensorrt-llm / FusedMoE op provider 选择
- **原因行为**：TRTLLMGenFusedMoE 在 __init__ 解析 op provider，而 ConfigurableMoE 到 create_weights() 前才安装最终 per-layer 量化配置，决策早于其输入定型。
- **影响对象**：可用性/进度（报错）
- **影响详情**：layerwise 量化 checkpoint 模型级声明 NVFP4、个别 MoE 层未量化时，该层在 __init__ 时看似量化而保留 native provider，但只有 FlashInfer 实现 BF16 MoE 内核，warmup 时 run_bf16_moe 失败。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/NVIDIA/TensorRT-LLM/pull/18770
- **关联**：var:INF-0557

### INF-0751 `vllm_nvfp4_gated_moe_end_padding_tp2_silent_corruption` NVFP4 gated-MoE TP2 end-padding 错位致输出腐蚀

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `var` · 发现 2026-09-07 · 旧号 7.78

- **原因对象**：量化 / vllm / NVFP4 MoE gate/up padding
- **原因行为**：TP2 后每投影 352 行不被 128 整除，把 fused gate+up 当连续行做 end-padding 而非两投影各自 pad，pad 行与 gate/up 错位使真实行按错 offset 解读；#48624 修复未覆盖此路径。
- **影响对象**：数值结果；输出内容（静默错误）
- **影响详情**：B200+TP2+FlashInfer TRT-LLM FP4 MoE 服务 Gemma-4-26B-A4B-NVFP4 时 GSM8K strict-invalid 100%、flexible 2.047%，91.8% 响应为打满上限的乱码，引擎无错；TP1 无需 padding，完全正确。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/55649
- **关联**：var:INF-0857

### INF-0752 `vllm_flashinfer_trtllm_nvfp4_kv_cache_qwen35_397b_invalid_output` NVFP4 KV cache 致 Qwen3.5-397B 输出塌方

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `var` · 发现 2026-09-07 · 旧号 7.79

- **原因对象**：量化 / vllm / TRT-LLM 后端 NVFP4 KV 解码路径
- **原因行为**：根因未定位：NVFP4（4-bit 浮点 + 双 scale）KV 量化在 FlashInfer TRT-LLM attention 后端的解码路径上对该模型族精度保持失效；并发档不敏感，指向确定性精度缺陷而非竞态。
- **影响对象**：KV状态；输出内容（质量退化）
- **影响详情**：B200 服务 Qwen3.5-397B-A17B-FP8 并用 NVFP4 KV 时，GSM8K flexible 仅 4.5–6.5%、invalid 93.2–94.9%（FP8 KV 为 96.664%），响应多为空或立即终止；并发 64 与 512 均坏，无错无警告。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/55673
- **关联**：var:INF-0837

### INF-0753 `vllm_glm53_nvfp4_dflash_sq80_worker_unknown_exit` GLM-5.3 NVFP4+DFlash 序列长≥80 时 worker 退出

`故障` · `active` · `infra_off` · `partial` · `documented` · `var` · 发现 2026-09-07 · 旧号 7.80

- **原因对象**：量化 / vllm / NVFP4 权重 × DFlash/GDN 路径
- **原因行为**：待定：原文未定位根因（日志在附件）；仅知 NVFP4 量化权重与 DFlash（GDN 线性注意力快速路径）的组合在序列长度 80 处存在兼容性边界。
- **影响对象**：可用性/进度（报错）
- **影响详情**：vLLM nightly 服务社区 GLM-5.3 NVFP4 量化（incoai glm-5.3-nvfp4+）并启用 DFlash 时，sq≥80 使 worker 以无法识别的 exit code 退出、服务终止；sq<80 正常。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/55625
- **关联**：var:INF-0749

### INF-0754 `flashinfer_trtllmgen_nvfp4_moe_situ_glu_gemm1_missing_output_scale_factors` NVFP4 MoE SiTU GEMM1 epilogue 漏写输出 scale

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `var` · 发现 2026-09-09 · 旧号 7.81

- **原因对象**：量化 / flashinfer / SiTU GEMM1 epilogue
- **原因行为**：trtllm-gen NVFP4 MoE 的 SiTU（siTuGlu）GEMM1 epilogue 在部分 tile/形状配置下不生成输出 block scale factor 的写出，下游反量化使用陈旧或未初始化的 scale。
- **影响对象**：数值结果（静默错误）
- **影响详情**：B200×8、消费 C++ 源与 trtllm-gen BMM artifact 的 flashinfer 构建中，受影响配置下 NVFP4 MoE 输出幅度系统性错误，无 crash；非全部配置触发。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/flashinfer-ai/flashinfer/issues/5053
- **关联**：var:INF-0742

### INF-0755 `flashinfer_sm107_nvfp4_mm_fp4_kernel_family_silently_disabled_import` flashinfer SM107 mm_fp4 内核因导入失败被静默禁用

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `var` · 发现 2026-09-09 · 旧号 7.82

- **原因对象**：量化 / flashinfer / SM107 mm_fp4 内核导入
- **原因行为**：sm107 内核文件从 nvidia-cutlass-dsl 不存在的 examples 命名空间导入父类，永远失败；两处裸 except ImportError: pass 吞掉异常，Sm107Kernel 恒为 None。
- **影响对象**：性能/资源；数值结果（性能退化）
- **影响详情**：所有测试环境中 SM107（Rubin）dense NVFP4 mm_fp4 都静默走非 SM107 分支，拿不到原生 tactics，造成性能损失并可能有精度差异；无警告、无日志、无错误，难以诊断。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/flashinfer-ai/flashinfer/issues/5065
- **关联**：var:INF-0750

### INF-0756 `flashinfer_vendored_trtllm_fp8_packed_cast_bf16_double_rounding_unpropagated_fix` flashinfer vendored fp8 转换双舍入修复未同步到副本

`故障` · `active` · `infra_off` · `partial` · `verified` · `NEW` · 发现 2026-09-11 · 旧号 7.83

- **原因对象**：量化 / flashinfer / vendored fp8x2 cast 双舍入
- **原因行为**：vendored 的 csrc/nv_internal/cudaTypeUtils.cuh 中 fp8x2_e4m3 转换仍先舍入到 bf16 再转 e4m3（两次舍入）；PR #4167 只修了 twin 副本，未传播。
- **影响对象**：数值结果（静默错误）
- **影响详情**：每个错值恰差一个可表示步且在元素容差内，只有聚合 mismatch 比例可见：A100 上 log-uniform 全域 2.34% 元素不同、真实激活 3.16% 输出不同。当前 fp8 实例化无 caller，重新接线或直接编译 csrc/nv_internal 即中招。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/flashinfer-ai/flashinfer/issues/5128

### INF-0757 `vllm_sm100_fp8_gemm_standard_blockwise_ignore_leading_strides` sm100 CUTLASS FP8 GEMM 忽略 leading stride

`故障` · `active` · `infra_off` · `yes` · `verified` · `var` · 发现 2026-09-11 · 旧号 7.84

- **原因对象**：量化 / vllm / sm100 CUTLASS FP8 GEMM 调用
- **原因行为**：sm100 CUTLASS FP8 GEMM 的 standard 与 blockwise 调用路径按逻辑 shape 推导步进，未用张量实际 leading stride；sm90 修复未覆盖 sm100，sm120 调用点仍在补丁外。
- **影响对象**：数值结果（静默错误）
- **影响详情**：带 padding 的 leading dimension 视图（如 (8,256)、stride (384,1) 的 blockwise a）被按 row stride 256 寻址而读错行，FP8 GEMM 输出静默错值，无 crash。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/56480
- **关联**：var:INF-0748

### INF-0758 `sglang_dsv41_fp8_wo_a_absorb_nonpow2_activation_scale_mantissa_truncated` SGLang DSV4.1 wo_a fp8 非 2 幂激活 scale 被截断

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-09-13 · 旧号 7.85

- **原因对象**：量化 / sglang / wo_a fp8_einsum 激活 scale
- **原因行为**：UE8M0 关闭分支给 deep_gemm.fp8_einsum 喂非 2 幂 fp32 per-token 激活 scale；内核要求 2 幂，但只对权重 scale 断言，对激活 scale 静默截断 mantissa。
- **影响对象**：数值结果；隐状态（静默错误）
- **影响详情**：DeepSeek-V4.1 wo_a absorb GEMM 输出约 25% 静默错值，MSA 输出整层污染，无 crash 无警告；SM121 GB10 上用真实 checkpoint 权重隔离复现，分支按 flag 而非 arch 选择，其它 arch 理论可达。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/39193

### INF-0759 `sglang_fp8_experts_f32_scale_checkpoint_cross_engine_accuracy_divergence` SGLang fp8 专家 F32 scale 精度比 vLLM 低 10pp

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-09-13 · 旧号 7.86

- **原因对象**：量化 / sglang / fp8 专家 F32 块 scale 路径
- **原因行为**：根因未定位；唯一区分维度是专家量化 dtype 与 scale dtype 组合（fp8 专家 + F32 块 scale），候选方向是 SGLang DeepGEMM/量化路径对该 scale 的处理与 vLLM 分歧。
- **影响对象**：输出内容（质量退化）
- **影响详情**：同机同权重同 harness 下，DeepSeek-V4-Flash fp8 专家 checkpoint 的 AIME25 pass@1：SGLang 0.5.17 为 84.58%、vLLM 0.28.0 为 94.79%，差距主体为错答；零警告、吞吐正常，官方 fp4 专家基模无差距。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/39311

### INF-0761 `vllm_compressed_tensors_mxfp4_w4a16_misclassified_w4a4_dispatch_sm100` MXFP4 W4A16 在 SM100+ 被误分派为 W4A4 内核

`故障` · `active` · `infra_off` · `config` · `verified` · `var` · 发现 2026-09-14 · 旧号 7.89

- **原因对象**：量化 / vllm / compressed-tensors 分派
- **原因行为**：quant-method selector 只按权重格式 _is_mxfp4 判别即构造 W4A4 Mxfp4 类，未检查 input_quant is None，丢弃激活量化元数据；NVFP4 分支已正确区分 use_a16。
- **影响对象**：数值结果；输出内容（质量退化）
- **影响详情**：SM100+（B200×8 验证）加载 MXFP4 W4A16 checkpoint（如 llmcompressor RTN 量化的 Qwen3-8B）时，激活按 W4A4 的 FP4 路径处理，输出静默退化，无 crash、不拒绝加载；同一 checkpoint 在不同 SM 代际结果不一致。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/56770
- **关联**：var:INF-0731

### INF-0763 `sglang_w8a8_block_fp8_matmul_triton_unsupported_k_tile_silently_wrong` block FP8 Triton matmul 缺 tile 整除守卫致错值

`故障` · `active` · `infra_off` · `partial` · `verified` · `var` · 发现 2026-09-15 · 旧号 7.91

- **原因对象**：量化 / sglang / block FP8 Triton tile 守卫
- **原因行为**：w8a8_block_fp8_matmul_triton 不校验 scale 块（32）与 BLOCK_SIZE_K（64）的整除关系，静默接受手工 BK64 配置，一个 K tile 跨两个 scale 块只应用一个尺度。
- **影响对象**：数值结果（静默错误）
- **影响详情**：block_size=[32,32] 输入配手工 BLOCK_SIZE_K=64 时，H200 确定性示例 512 个输出元素全为 64 而非 288（BK32 正确）；量级貌似正常，无 NaN、无 assert。默认由 tuner 选配置的服务路径不受影响。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/39626
- **关联**：var:INF-0731

### INF-0764 `sgl_deep_gemm_transform_sf_layout_non_owning_alias_scale_caching_allocator_reuse` sgl-deep-gemm scale 变换返回非 owning 别名致被覆写

`故障` · `active` · `infra_off` · `partial` · `verified` · `var` · 发现 2026-09-16 · 旧号 7.92

- **原因对象**：量化 / sgl-deep-gemm / transform_sf 布局变换
- **原因行为**：transform_sf_into_required_layout 可返回别名输入 scale 的视图，却不保留原分配 owner（无引用、record_stream 或延迟释放），输入释放后 allocator 可复用该页。
- **影响对象**：数值结果；模型权重（静默错误）
- **影响详情**：SM90（H20-3e）上直接调用该 API 变换权重 scale 并释放输入后，FP8 block scale 被无关分配内容覆写，后续 GEMM 输出错误但有限，无 crash；依赖分配时序，间歇复现。SGLang 包装已在 #32188 修复，直接调用库的下游不受保护。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/39684
- **关联**：var:INF-1152

### INF-0765 `quant_Nbit` 低位 RTN 权重 fake-quant 致模型自洽变笨

`故障` · `active` · `model_on` · `config` · `verified` · `NEW` · 发现 待定 · 旧号 7.88

- **原因对象**：量化 / 通用 / 投影权重 RTN fake-quant
- **原因行为**：对 q/k/v/o/gate/up/down 投影权重做对称 absmax RTN fake-quant（per_channel 或 group128，zero-point 为 0），位宽降到 4/3/2 bit 时舍入误差增大。
- **影响对象**：输出内容；隐状态（质量退化）
- **影响详情**：8-bit 几乎无可观测差异；4-bit 起答案质量下降、部分 greedy 答案翻转；2-bit 输出基本坍塌。全程无错误信号、无 NaN；误差随层深累积，末层 cos 坍塌快于 per-layer cos。
- **改进措施**：检测 待定；修复 待定
- **来源**：引用：本项目早期实测条目（原型阶段实测，量值不跨栈继承；

### INF-0767 `sglang_quickreduce_q8_low_amplitude_reciprocal_scale_cliff_block_attenuation` QuickReduce Q8 低幅值 scale 饱和致归约衰减

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `var` · 发现 2026-09-18 · 旧号 7.94

- **原因对象**：量化 / sglang / QuickReduce Q8 编解码
- **原因行为**：CodecQ8<half> 把 encoding_scale≈1/decoding_scale 存为 FP16，block_max<≈0.00195 时倒数饱和于 65504，编解码不再互逆，整块被衰减；two-shot 通信中可施加两次。
- **影响对象**：数值结果；输出内容（静默错误）
- **影响详情**：ROCm 启用 INT8 量化与 BF16→FP16 cast 时，低幅值激活块 all-reduce 结果系统性衰减（MI350X 长 prefill 层 4 相对 L2 误差 0.68，RCCL 为 0.0026），长上下文续写不连贯；四 rank 一致，无报错。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/40084
- **关联**：var:INF-0741

### INF-0768 `vllm_sm12x_fp8_deepgemm_float32_scale_route_unsupported_nan_or_runtimeerror` DeepGEMM fp8 路由忽略 scale 格式致 SM12x 输出 NaN

`故障` · `active` · `infra_off` · `config` · `verified` · `NEW` · 发现 2026-09-18 · 旧号 7.95

- **原因对象**：量化 / vllm / DeepGEMM fp8 linear 路由谓词
- **原因行为**：should_use_deepgemm_for_fp8_linear() 不看 scale 格式与设备族；E8M0 关闭时 scale 为 FLOAT32 仍路由到在 SM12x 上无 float32 scale 路径的 DeepGEMM。
- **影响对象**：数值结果（数值异常）
- **影响详情**：GB10（sm_121）上设置 VLLM_USE_DEEP_GEMM_E8M0=0 时，旧 DeepGEMM 下 fp8 block linear 输出 0.78%–38.7% 为 NaN，新版本改为 RuntimeError；默认值 1 时结果正确。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/57486

### INF-0769 `flashinfer_sm90_w4a16_mxfp4_fused_moe_tma_descriptor_update_race_wrong_expert_scales` MXFP4 fused MoE TMA 描述符竞态致用错专家 scale

`故障` · `active` · `infra_off` · `other_stack` · `verified` · `NEW` · 发现 2026-09-22 · 旧号 7.96

- **原因对象**：量化 / flashinfer / cutlass_fused_moe TMA 描述符
- **原因行为**：CUTLASS collective 在内核执行中原地覆写 TMA descriptor 的指针/形状字段，而前一 CTA 的 scale 加载仍在途，读到更新一半的描述符时从错误专家的 scale 区取数。
- **影响对象**：数值结果（不可复现）
- **影响详情**：flashinfer 0.6.10–0.6.15.post1 在 Hopper 上以 w4 group scaling 做大 M prefill 时，约 0.75% 调用中某专家单个 tile 行级误差 15–30%，相同调用结果不一致；decode（M≤256）正常，0.6.18 消除。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/flashinfer-ai/flashinfer/issues/5438

### INF-0771 `vllm_triton_fused_moe_per_channel_weight_scales_indexed_as_per_tensor_quark_static_int8` fused MoE 按 per-tensor 读取通道级权重 scale

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-09-24 · 旧号 7.98

- **原因对象**：量化 / vllm / fused MoE 权重 scale 索引布局
- **原因行为**：legacy Triton fused MoE 用同一布尔同时决定激活与权重 scale 布局，Quark 静态 INT8 的 per-tensor 激活+per-channel 权重组合下权重 scale 被按 per-tensor 读。
- **影响对象**：数值结果（静默错误）
- **影响详情**：Quark 静态 INT8 W8A8 MoE 走 legacy Triton 路径时，探针中把某专家全部 W13 通道 scale ×4 输出不变，per-channel 校准失效、输出带系统性量化误差；仅组件级观察，未做完整模型评测。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/58532

### INF-0773 `sglang_mimo_v2_fp8_label_mxfp4_experts_selects_fp8_runner_sm100` MiMo-V2 为 MXFP4 专家误选 FP8 MoE runner

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `var` · 发现 2026-09-28 · 旧号 7.100

- **原因对象**：量化 / sglang / MiMo-V2 MoE runner 自动选择
- **原因行为**：_mimo_v2_overrides 在 SM100、runner=auto 时只看 quant_method=fp8 恒选 flashinfer_trtllm，无视已检测到的 is_fp4_experts（专家实为 MXFP4）。
- **影响对象**：可用性/进度（报错）
- **影响详情**：4×B200 TP4 加载 MiMo-V2 混合 checkpoint（fp8 标签 + mxfp4 专家）时启动即报 RuntimeError（gemm1_weights 为 int8 而非 float8_e4m3fn）；显式指定 runner 不受影响；若下游 check 后移则可能变为静默。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/41569
- **关联**：var:INF-0771

### INF-0774 `sglang_nvfp4_kv_reuses_checkpoint_fp8_calibrated_scale_as_global_scale` nvfp4 KV 误用 fp8 校准 scale 作全局 scale

`故障` · `active` · `infra_off` · `config` · `verified` · `var` · 发现 2026-10-09 · 旧号 7.101

- **原因对象**：量化 / sglang / nvfp4 KV global scale 加载
- **原因行为**：maybe_remap_kv_scale_name() 无条件挂载 fp8 校准 k/v_scale，fp4 路径无一致性校验，fp8 scale（amax/448）被当作 NVFP4 global scale（amax/6）。
- **影响对象**：KV状态；输出内容（质量退化）
- **影响详情**：--kv-cache-dtype nvfp4 服务携带 fp8 校准 k/v_scale 的 checkpoint 时，短 prompt 正常，长上下文 needle 检索塌缩（1/4，删去该 scale 后 4/4），答案退化为问题回声；启动日志无任何 scale 提示。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/42369
- **关联**：var:vllm_compressed_tensors_kv_scale_dropped

### INF-0775 `vllm_marlin_int8_activation_negative_group_scales_read_as_uint16` Marlin int8 激活路径把负 group scale 读成大正数

`故障` · `active` · `infra_off` · `config` · `verified` · `var` · 发现 2026-10-09 · 旧号 7.102

- **原因对象**：量化 / vllm / Marlin int8 激活 scale 处理
- **原因行为**：marlin_act_int8_process_scales 用 s.max() 而非 abs().max() 归一化并存为 int16，内核按 uint16 重解释读取，负 scale 的符号位变成幅值。
- **影响对象**：数值结果；输出内容（静默错误）
- **影响详情**：VLLM_MARLIN_INPUT_DTYPE=int8 且 num_groups>1 时，含负 scale 的 checkpoint（如 AutoRound 对称 int4，约半数组为负）反量化权重误差达典型幅度的 170×；模型照常加载运行，输出质量塌缩而服务无告警。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/59403
- **关联**：var:INF-0735

### INF-0776 `vllm_exllama_rdna3_w4a16_group_size_not_multiple_32_silent_wrong_output` W4A16 内核遇 group_size 非 32 倍数静默错值

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `var` · 发现 2026-10-09 · 旧号 7.103

- **原因对象**：量化 / vllm / Exllama/RDNA3 W4A16 K 步进
- **原因行为**：Exllama GPTQ GEMM/reconstruct 与 RDNA3 W4A16 dense、fused-MoE 内核按 32 步进遍历 K、每 32 元素切换 scale，group_size 非 32 倍数时在块尾读错量化组。
- **影响对象**：数值结果；输出内容（静默错误）
- **影响详情**：group_size%32≠0（如 16）的 INT4 GPTQ/compressed-tensors 配置被接受但推理输出静默垃圾，CUDA sm_120 与 ROCm gfx1100 均复现；目前无已知公开 checkpoint 使用此类 group_size。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/60608
- **关联**：var:INF-1161

### INF-0777 `trtllm_calculate_nvfp4_global_scale_wrong_above_8192_nonmultiple_hidden` TRT-LLM NVFP4 全局 scale 算子宽隐藏维下算错

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-10-09 · 旧号 7.104

- **原因对象**：量化 / tensorrt-llm / nvfp4 global scale 计算内核
- **原因行为**：calculate_nvfp4_global_scale 把含 __syncthreads 的 getMaxAbs 放进 per-thread 循环，CTA 处理多 token 时线程发散使块归约失效；launcher 只查 n%256。
- **影响对象**：数值结果（静默错误）
- **影响详情**：hidden>8192 且非 8192 倍数时 per-token scale 出错：[512,12288] 中 309/512 个 token 错（如 28.0 应为 42.0），[256,28672] 中 101–114/256 错，量化误差增大；[512,8192] 等形状正确。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/NVIDIA/TensorRT-LLM/issues/19954

### INF-0778 `flashinfer_cute_dsl_mxfp8_e8m0_scale_byte_zero_zeros_nonzero_blocks` CuTe-DSL MXFP8 量化把 scale 字节 0 当零致整块归零

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-10-09 · 旧号 7.105

- **原因对象**：量化 / flashinfer / mxfp8 CuTe-DSL 逆 scale 函数
- **原因行为**：E8M0 scale 字节 0 表示 2^−127（非零），量化应使用有限逆 scale 2^127，但 CuTe-DSL 的 inverse-scale helper 对该编码显式返回零。
- **影响对象**：数值结果（静默错误）
- **影响详情**：mxfp8_quantize(backend="cute-dsl") 对 scale 字节为 0 的块（幅度极小的激活/权重）把非零且可精确表示的 BF16 值整块量化为零；GB200、flashinfer 0.6.18 最小复现，无报错。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/flashinfer-ai/flashinfer/issues/5996

### INF-0779 `sglang_compressed_tensors_w4a16_int4_serving_quality_degraded_vs_vllm` SGLang W4A16 int4 服务 PPL 显著劣于同权重 vLLM

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-10-09 · 旧号 7.106

- **原因对象**：量化 / sglang / W4A16 反量化/内核路径
- **原因行为**：未定位；同权重跨引擎对照把缺陷收敛到 SGLang 侧 compressed-tensors W4A16 反量化/内核路径，未逐层定位。
- **影响对象**：输出分布；输出内容（质量退化）
- **影响详情**：SGLang 0.5.19–0.5.21 服务 Qwen3.8-27B compressed-tensors W4A16（int4 对称 group128）时 wikitext PPL 9.98，同 checkpoint vLLM 为 6.05（+65%），输出仍流畅、无报错。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/42917

### INF-0781 `flashinfer_b12x_w4a16_mxfp4_e8m0_scale_bias_bf16_subnormal_weights` b12x W4A16 MXFP4 低 scale 字节致权重下溢

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-10-12 · 旧号 7.108

- **原因对象**：量化 / flashinfer / b12x W4A16 MXFP4 反量化补偿
- **原因行为**：FP4 unpack（v*2^-126）、E8M0 解码（2^(e-120)）与 bf16x2 MMA 使每个非零权重恒乘 2^-119，再于 epilogue 以 2^119 补偿；该补偿只在 121≤e≤247 窗口内正确。
- **影响对象**：数值结果；隐状态（静默错误）
- **影响详情**：scale 字节 <121 时 MMA 权重落入 BF16 subnormal，<114 时 FP4 0.5 被刷零；真实 MXFP4 checkpoint 字节多在 100–130，低 scale 行专家贡献静默消失；无 NaN/Inf，聚合 cos 仍达 0.99999；SM120 复现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/flashinfer-ai/flashinfer/issues/6316

### INF-0782 `flashinfer_mm_mxfp8_noncontiguous_out_ignores_strides_overwrites_neighbors` mm_mxfp8 无视非连续 out 的 stride 致垃圾与越界覆写

`故障` · `active` · `infra_off` · `partial` · `verified` · `NEW` · 发现 2026-10-12 · 旧号 7.109

- **原因对象**：量化 / flashinfer / mm_mxfp8 输出 stride 处理
- **原因行为**：cutlass/cute-dsl 后端内核按紧凑 row-major M×N 从 out.data_ptr() 写入，假设输出连续，却既不校验也不适配真实 stride。
- **影响对象**：数值结果（静默错误）
- **影响详情**：非连续 out（如列切片、列主序视图）配 cutlass/cute-dsl/auto（默认）后端时结果为垃圾（cos≈0.001），并越界覆写 out 之外的相邻内存；auto 偶发 ValueError，cudnn 后端正确。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/flashinfer-ai/flashinfer/issues/6210

### INF-0783 `flashinfer_trtllm_gen_context_nvfp4_kv_head_dim256_v_scale_missing_interleave` trtllm-gen prefill NVFP4 KV 用错 V scale

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-10-12 · 旧号 7.110

- **原因对象**：量化 / flashinfer / trtllm-gen 上下文内核 V scale
- **原因行为**：head_dim=256 时 V scale 的 stage offset 按 linear 应用、漏掉 ×4 token interleave，每 head 的 dims 128–255 读取 dims 32–159 的 scale。
- **影响对象**：数值结果；输出分布（质量退化）
- **影响详情**：SM100（B200）上使用 NVFP4 KV cache 且 head_dim=256（如 Qwen3.5）时，每次 prefill 注意力输出损坏，长上下文 MRCR 与 prompt NLL 显著劣化；KV 写入、decode 内核及 head_dim 64/128 正常。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/flashinfer-ai/flashinfer/issues/6310

### INF-0784 `vllm_dflash2_fp8_draft_ignored_layers_global_alias_missing_throughput_collapse` DFlash2 draft 漏补 ignored_layers 致投机吞吐崩塌

`故障` · `active` · `infra_off` · `partial` · `verified` · `NEW` · 发现 2026-10-12 · 旧号 7.111

- **原因对象**：量化 / vllm / DFlash2 draft fp8 排除层别名
- **原因行为**：_add_global_draft_layer_exclusions 只补 exclude_modules 的全局层号别名、不补 Fp8Config 的 ignored_layers，draft 卷积投影被建成 fp8 线性层。
- **影响对象**：模型权重；性能/资源（性能退化）
- **影响详情**：nightly 中加载 fp8 量化 DFlash2 draft 时有 loader warning；target 逐 token 校验保证输出正确，但 acceptance 从 4.3 降到 1.03、decode 吞吐 210→46 tok/s；9 月版本正常。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/61003

### INF-0796 `vllm_int8_per_token_head_kv_prefix_cache_off_first_token_corruption` vLLM int8 KV 关闭前缀缓存时首 token 损坏

`故障` · `active` · `infra_off` · `yes` · `verified` · `NEW` · 发现 2026-08-03 · 旧号 8.12

- **原因对象**：量化 / vllm / int8_per_token_head KV 缓存
- **原因行为**：int8_per_token_head KV 与关闭 prefix caching 的组合存在缺陷，具体出错代码原文未定位；已排除量化精度损失与内存压力。
- **影响对象**：输出内容（静默错误）
- **影响详情**：v0.26.0 V1、Gemma-4 31B hybrid、Triton backend、prefix caching 关闭时，响应开头几个 token 损坏（结构化输出首键出错），其余 2–3.5k token 正常；开启 prefix caching 则干净，KV 池利用率仅约 3%。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/50749

### INF-0804 `vllm_kimi_k3_fp8_kv_assertion_circular` vLLM Kimi-K3 fp8 KV 在 Hopper 上断言无法满足

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-08-07 · 旧号 8.20

- **原因对象**：量化 / vllm / fp8 prefill query 量化开关校验
- **原因行为**：use_prefill_query_quantization 仅在 Blackwell 生效、在非 Blackwell 设备上被静默忽略，而 Kimi-K3 fp8 KV 的 assertion 仍要求该 flag。
- **影响对象**：可用性/进度（报错）
- **影响详情**：H200（SM 9.0）上 Kimi-K3 + --kv-cache-dtype fp8 + FLASHMLA 启动时 assertion 失败；按提示设置该 flag 后同一 assertion 再次触发，Hopper 上无法满足，服务不可启动。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/51313

### INF-0838 `sglang_dsv4_flash_fp8_mi300x_length_smooth_degradation` SGLang DSV4-Flash FP8 在 MI300X 随长度平滑退化

`故障` · `active` · `ambiguous` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-26 · 旧号 8.54

- **原因对象**：量化 / sglang / block-FP8 KV/激活量化路径
- **原因行为**：逐一替换 DSA 组件仍未定位；概率性退化指向 FP8 量化路径（block-FP8 KV 或激活量化）误差随序列长度复合累积、逐步淹没信号，根因待定。
- **影响对象**：输出内容；输出分布（质量退化）
- **影响详情**：8×MI300X TP8 上 DeepSeek-V4-Flash FP8：<100 token 正确，约 300 token 起输出垃圾，NIAH 全长度 0/3；>2048 token 后同请求同 seed 结果不一致；无 crash、无 NaN、无错误，短上下文冒烟测试可通过。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/36390

### INF-0846 `vllm_flashinfer_mla_sparse_sm90_fp8_kv_storage_dtype_rejected_startup` SM90 sparse MLA 传错 fp8 KV dtype 致启动失败

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-09-03 · 旧号 8.62

- **原因对象**：量化 / vllm / SM90 sparse MLA builder
- **原因行为**：FlashInferMLASparseSM90Builder 把 fp8 KV 的存储 dtype（uint8）而非逻辑 view dtype（float8_e4m3fn）传给 plan()
- **影响对象**：可用性/进度；性能/资源（报错）
- **影响详情**：GLM-5.3-Flash 在 SM90 sparse MLA 后端开 --kv-cache-dtype fp8 时启动即抛 ValueError（kv_data_type torch.uint8 不支持）；放弃 fp8 KV 才能运行，KV 容量缩水约 1.78×
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/55221

### INF-0849 `sglang_minimax_m3_w4a16_triton_sparse_all_null_tokens_serves_success` SGLang 量化 embedding 未加载致输出恒为 token 0

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-09-05 · 旧号 8.65

- **原因对象**：量化 / sglang / get_quant_method 嵌入分支
- **原因行为**：compressed-tensors get_quant_method 漏处理 VocabParallelEmbedding，embedding 按未量化注册 bf16 weight，packed 权重未加载，表保持 torch.empty
- **影响对象**：模型权重；输出内容（静默错误）
- **影响详情**：DGX Spark（sm_121、TP2）上服务 MiniMax-M3 W4A16 时加载、graph 捕获与 API 均正常，但每个生成 token 都是 id 0（NUL 字节），无错误无 NaN；同权重 vLLM 正确，加载期仅有 params_dict 警告
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/38143

### INF-0852 `vllm_dsv4_sparse_indexer_missing_hadamard_rotation_quant_misalignment` DSV4 indexer 量化 kernel 漏 Hadamard 旋转

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-09-05 · 旧号 8.68

- **原因对象**：量化 / vllm / DSV4 indexer Q/K 量化内核
- **原因行为**：DSV4 indexer 的 Q fp8/mxfp4 与 compressor-insert 融合 kernel 未施加参考实现在 RoPE 后、量化前的 Hadamard 旋转；K 侧存了 rotate 参数但 forward 不读
- **影响对象**：KV状态（静默错误）
- **影响详情**：无报错；因旋转对 Q、K 同时施加，QK 点积数学上保持，检索质量可能基本正常，但 indexer 缓存内容与参考实现/SGLang 逐位不一致，跨栈 KV 互换或混用不同版本节点时语义错位
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/55459

### INF-0862 `flashinfer_gqa_decode_fp8_kv_sm75_smem_overflow_crash` FlashInfer sm75 fp8 KV GQA decode 共享内存溢出

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `var` · 发现 2026-09-08 · 旧号 8.78

- **原因对象**：量化 / flashinfer / fp8 KV decode smem 预算
- **原因行为**：fp8 KV inline-scale decode 内核的 smem_md 预算按单 KV head 组（MQA 形状）规划，未计入 GQA 组扩张，超出 Turing 64 KiB 共享内存上限
- **影响对象**：可用性/进度（报错）
- **影响详情**：FlashInfer 0.6.18 在 sm75（T600）上以 fp8 KV（e4m3/e5m2）跑 GQA decode，head_dim 64/128/256 × fp16/bf16 q 组合共享内存溢出、kernel launch 失败；非 GQA 同路径正常
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/flashinfer-ai/flashinfer/issues/5030
- **关联**：var:INF-0846

### INF-0876 `vllm_dsv41_flash_nightly_logprob_shift_tool_structured_text_fp8_ds_mla_kv_record_change` DSV4.1 KV 记录改 MXFP8 致结构化文本 logprob 位移

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `var` · 发现 2026-09-23 · 旧号 8.92

- **原因对象**：量化 / vllm / fp8_ds_mla KV 记录 MXFP8 化
- **原因行为**：#56893 把 V4.1 KV 记录改为全 512 维 MXFP8（含 RoPE 通道按 fp8 存，kv_bytes_per_token 584→528），改变了 RoPE 通道的量化路径；该归因来自社区源码核查，未做无 hook 复现
- **影响对象**：输出分布（静默错误）
- **影响详情**：nightly dc36fcce→cd10ed6f 后 teacher-forced logprobs 在短结构化文本上显著位移：agent 转录 top-1 翻转率 1.77%→12.66%，math/日语/德语 0%→9%；长 prose 基本不动，全语料 perplexity 不变，监控恒绿
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/58391
- **关联**：var:INF-0837

### INF-0879 `flashinfer_fa2_nvfp4_kv_prefill_wrong_head_dim64_sm120_sm121` FA2 NVFP4 KV prefill 在 head_dim=64 输出错值

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `var` · 发现 2026-10-09 · 旧号 8.95

- **原因对象**：量化 / flashinfer / FA2 NVFP4 KV prefill 内核
- **原因行为**：FA2 NVFP4 KV prefill 内核在 head_dim=64 路径存在缺陷，具体机制原文未给出（见 issue）
- **影响对象**：数值结果（静默错误）
- **影响详情**：SM120/SM121（GB10）当前 release 上 NVFP4 KV + head_dim=64 的 prefill 输出静默错误、无 crash；与 SM80 上未修复关闭的 #4748 同症状
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/flashinfer-ai/flashinfer/issues/5965
- **关联**：var:vllm_nvfp4_kv_cache_hnd_layout_probe_inverted_silent_corruption

### INF-1057 `vllm_mtp_spec_decode_turboquant_kv_cache_repetition_collapse_sm120` vLLM MTP 与 turboquant KV 组合致长上下文重复坍塌

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-08-15 · 旧号 11.34

- **原因对象**：量化 / vllm / TurboQuantMetadataBuilder
- **原因行为**：TurboQuantMetadataBuilder 声明矛盾：允许 FULL 图捕获 query_len=1+k 的 verify 批，却设 supports_spec_as_decode=False 使其走单 token 路由。
- **影响对象**：输出内容（静默错误）
- **影响详情**：sm_120 + hybrid GDN 模型 + MTP，KV dtype 为 turboquant_4bit_nc/3bit_nc 时，约 8k token 起的长 prompt 输出单 token 重复约 230 次；fp8 KV 正常、短 prompt 仍答对；k=1 时直接非法访存。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/52475；https://github.com/vllm-project/vllm/pull/53406

### INF-1077 `vllm_turboquant_k8v4_mtp_hybrid_gdn_prompt_gated_degenerate_output` TurboQuant k8v4 KV×MTP 下部分 prompt 静默退化

`故障` · `active` · `infra_off` · `partial` · `documented` · `var` · 发现 2026-08-27 · 旧号 11.54

- **原因对象**：量化 / vllm / TurboQuant k8v4 KV 缓存
- **原因行为**：待定：原文未定位根因；仅确认 turboquant_k8v4 KV 与 MTP 在 hybrid GDN 模型上组合触发，不依赖 #51812，与 TurboQuant KV × 推测解码退化（#40831/#40880）同族
- **影响对象**：输出内容（质量退化）
- **影响详情**：2×RTX 4090 TP=2、Qwen3.8-27B AWQ-INT4 hybrid + MTP：部分 prompt 坍塌为重复墙或 garbage token，其余正常；/health 200、无 error/warning，spec 接受率与吞吐正常，固定冒烟集易漏检
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/53180
- **关联**：var:INF-1057

### INF-1125 `hf_compressed_tensors_run_compressed_false_silent_random_init_moe` compressed-tensors 解压漏映射 MoE 专家致随机初始化

`故障` · `active` · `infra_off` · `yes` · `documented` · `var` · 发现 2026-08-01 · 旧号 12.9

- **原因对象**：量化 / transformers / run_compressed=False 解压器
- **原因行为**：run_compressed=False 路径的解压器不映射 packed 的 fused expert 张量（判为 UNEXPECTED/MISSING），这些权重被静默新初始化；属 5.10.1 版本回归。
- **影响对象**：模型权重；输出分布（静默错误）
- **影响详情**：transformers 5.10.1 加载 W4A16 的 Qwen3.6-35B-A3B 时 from_pretrained 成功、无报错，每个 MoE 层产生垃圾 logits，与未损坏 teacher 0% 一致；5.13.1 正常解压。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/huggingface/transformers/issues/47407
- **关联**：var:INF-1118

### INF-1130 `sglang_mxfp4_sharded_state_silent_expert_weight_drop` SGLang 导出 mxfp4 检查点静默丢弃 expert 权重

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-08-11 · 旧号 12.14

- **原因对象**：量化 / sglang / save_sharded_state 导出
- **原因行为**：save_sharded_state.py 以 sharded_state 格式导出 mxfp4 检查点时未序列化 expert 权重，量化权重格式的导出路径覆盖不全。
- **影响对象**：模型权重；输出内容（静默错误）
- **影响详情**：导出 gpt-oss-20b（TP=2）不报错、不警告，导出 safetensors 总大小显著小于原始，部分 expert 权重完全缺失；加载后缺失 expert 的输出为零或垃圾，行为不可预测。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/34448

### INF-1133 `vllm_quark_structured_quant_config_list_mapper_attributeerror` vLLM QuarkConfig 把结构化 list 当模块名映射致加载报错

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-08-15 · 旧号 12.17

- **原因对象**：量化 / vllm / Quark apply_vllm_mapper
- **原因行为**：QuarkConfig.apply_vllm_mapper() 把所有 list 值的量化配置项送进 WeightsMapper.apply_list()，后者假设元素都是字符串 module name，不兼容新版结构化 list。
- **影响对象**：可用性/进度（报错）
- **影响详情**：加载含结构化 list 配置项的新 Quark 量化 checkpoint（如 Qwen 3.8 quark int4）时抛 AttributeError: 'dict' object has no attribute 'endswith'，模型加载失败，无静默损坏。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/52454

### INF-1134 `vllm_compressed_tensors_fp8_parallel_lm_head_partition_metadata_uninitialized` vLLM W8A8-FP8 未初始化 lm_head 分区元数据致加载报错

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-08-15 · 旧号 12.18

- **原因对象**：量化 / vllm / W8A8Fp8 create_weights
- **原因行为**：W8A8Fp8.create_weights 未初始化 ParallelLMHead 加载所需的 output_size_per_partition 等分区属性；ParallelLMHead 不从 LinearBase 派生。
- **影响对象**：可用性/进度（报错）
- **影响详情**：加载 compressed-tensors W8A8-FP8 checkpoint（量化 lm_head，TP 加载路径）时抛 AttributeError：ParallelLMHead 无 output_size_per_partition 属性，fail-fast。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/52434

### INF-1140 `sglang_qwen35_mtp_weights_compressed_tensors_ignore_list_silent_drop` SGLang Qwen3.5 MTP 漏量化分支致推测解码失效

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-08-27 · 旧号 12.24

- **原因对象**：量化 / sglang / _mtp_quant_config 分支
- **原因行为**：_mtp_quant_config() 只有 quark 分支；compressed-tensors checkpoint 把 MTP linears 逐条列入 ignore，落入无 scale/无权重路径，draft 权重被静默丢弃。
- **影响对象**：模型权重；性能/资源（性能退化）
- **影响详情**：SGLang + Qwen3.5-family + compressed-tensors checkpoint 开 MTP 推测解码时，输出等同无推测，accept len 恒 1.000、零方差，吞吐塌 8 倍（30.7 vs 248 tok/s），无任何警告。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/35797

### INF-1141 `transformers_nvfp4_moe_decompress_missing_weight_global_scale` transformers NVFP4 MoE 反量化漏全局 scale

`故障` · `active` · `infra_off` · `partial` · `verified` · `NEW` · 发现 2026-08-28 · 旧号 12.25

- **原因对象**：量化 / transformers / update_weight_conversions
- **原因行为**：update_weight_conversions 收集 tensor 时只取 packed/scale/shape，漏掉 checkpoint 存储的 weight_global_scale，专家反量化时不带全局缩放。
- **影响对象**：模型权重；输出内容（静默错误）
- **影响详情**：加载按专家分块存储的 compressed-tensors NVFP4 MoE checkpoint 时无异常、missing_keys 为空，每个专家被错误常数缩放，下游推理基于错误权重进行；CPU 可复现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/huggingface/transformers/issues/48371

### INF-1151 `sglang_quantized_dflash2_draft_silent_zero_acceptance` SGLang 量化 DFlash2 draft 静默几乎全被拒绝

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `var` · 发现 2026-09-11 · 旧号 12.35

- **原因对象**：量化 / sglang / 量化 DFlash2 draft 权重路径
- **原因行为**：量化（compressed-tensors W4A16）DFlash2 draft 权重的某处处理（scale/zero-point 应用或反量化路径）与 verify 路径不兼容，draft 分布偏移；具体位点未定位，报告者未根因。
- **影响对象**：性能/资源（性能退化）
- **影响详情**：量化 draft checkpoint 正常加载服务、无错误无警告，acceptance 从约 3.71 塌到 1.03，decode 约 38 tok/s，比无 drafter（约 171）更慢；同权重 BF16 正常。auto-round 4-bit、Qwen3.8-27B、TP2 复现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/39087
- **关联**：var:INF-1147

### INF-1156 `sglang_minimax_h3_int8_convrot_quant_format_implies_qkv_layout_skips_reorder` SGLang int8 误判 QKV 布局致 MiniMax-H3 花屏

`故障` · `active` · `infra_off` · `other_stack` · `verified` · `var` · 发现 2026-09-18 · 旧号 12.40

- **原因对象**：量化 / sglang / KitchenInt8Config 布局判定
- **原因行为**：KitchenInt8Config 用“是否 int8 序列化”代理“QKV 是否 native 布局”，int8 checkpoint 一律跳过 MiniMax 官方 head-interleaved QKV 重排。
- **影响对象**：输出内容；模型权重（静默错误）
- **影响详情**：加载 MiniMax-H3-INT8-CONVROT 等 repack 时每个 attention block 从错误行读 Q/K/V，运行完成、无错误无 NaN，产出合法 MP4 但视频塌成平坦低对比场（std 50.78→2.80）、音频变噪声；单 GPU 未开 FSDP 推理即中招。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/40127
- **关联**：var:INF-0357

### INF-1161 `vllm_wna16_moe_partial_group_size_repeat_interleave_shape_mismatch_704_128` vLLM WNA16 MoE scale 组展开错位致加载崩溃

`故障` · `active` · `infra_off` · `config` · `documented` · `var` · 发现 2026-09-29 · 旧号 12.45

- **原因对象**：量化 / vllm / moe_wna16 scale 组展开
- **原因行为**：中间宽不整除 group size 时 create_weights() 把有效 group size 折半、权重按 repeat_interleave 展开，但 checkpoint 的 scale 组数未按同规则展开。
- **影响对象**：可用性/进度（报错）
- **影响详情**：Gemma-4-26B-A4B（中间宽 704、group 128）的 AutoRound W4A16 checkpoint 加载 w2 时 RuntimeError（11 vs 12），引擎无法启动；显式 --quantization 或 MoE backend override 均无法绕过。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/59312
- **关联**：var:INF-1155

### INF-1164 `vllm_mimo_v2_mtp_loader_drops_fp8_kv_scales_draft_falls_back_unit_scale` vLLM MiMoV2 MTP 加载丢弃 FP8 KV scale

`故障` · `active` · `infra_off` · `config` · `documented` · `var` · 发现 2026-10-09 · 旧号 12.48

- **原因对象**：量化 / vllm / MiMoV2MTP KV scale 加载
- **原因行为**：MiMoV2MTP.load_weights 缺 KV scale 名映射：融合名、ModelOpt 分投影名、.kv_scale 三种命名下 FP8 KV scale 均被跳过，保持 −1.0 后重置为 1.0。
- **影响对象**：KV状态；性能/资源（质量退化）
- **影响详情**：带 fp8 KV scale 的 MiMo-V2 MTP 每个 draft 层以 unit scale 量化/反量化 cache，输出有限、无报错，接受率/输出质量静默退化，唯一痕迹是易被误读为 checkpoint 问题的启动 warning；主模型 loader 正确。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/60333
- **关联**：var:vllm_compressed_tensors_kv_scale_dropped

### INF-1208 `vllm_mtp_predictor_quant_config_inheritance_pack_quantized_target_dense_bf16_draft_load_fail` vLLM MTP draft 继承 target 量化配置致密集分片加载崩溃

`故障` · `active` · `infra_off` · `config` · `documented` · `var` · 发现 2026-09-25 · 旧号 13.40

- **原因对象**：量化 / vllm / MTP predictor 量化配置继承
- **原因行为**：MTP 层沿用 target 的 quant_config 建为 pack-quantized；既有 workaround 仅覆盖 modelopt_fp4/GPTQ dynamic，不覆盖 MTP 留密集却未列入 ignore 的情形。
- **影响对象**：可用性/进度（报错）
- **影响详情**：vLLM v0.30 以 mtp 推测解码、target 为 compressed-tensors pack-quantized 且 MTP head 为密集 BF16 分片时，启动加载确定性崩溃（ValueError：无 fc.weight）；错误只报缺模块名，无诊断指引。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/58807
- **关联**：var:INF-1157

### INF-1267 `kivi_key_perchannel_outlier_granularity` KIVI：key cache 按 token 量化的粒度错配致准确率坍缩

`故障` · `active` · `infra_off` · `yes` · `documented` · `var` · 发现 待定 · 旧号 K6.2

- **原因对象**：量化 / 通用 / KV cache key 量化粒度
- **原因行为**：key cache 存在持续的大幅离群通道（value 没有），本应按 channel 量化；按 token/按 tensor 量化 key 时少数离群通道主导每个 token 的取值范围，压垮其余通道分辨率并破坏 QK^T。
- **影响对象**：KV状态；输出内容（质量退化）
- **影响详情**：低 bit KV + per-token key 粒度下推理任务准确率急剧坍缩（Llama-2-7B GSM8K 13.50→5.76）；改回 per-channel 后仅下降约 2%。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://arxiv.org/abs/2402.02750
- **关联**：var:INF-0675

### INF-1272 `trtllm_fp8_kv_reuse_scale_mismatch` TRT-LLM FP8 KV 复用时反量化 scale 不一致

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `var` · 发现 待定 · 旧号 K6.7

- **原因对象**：量化 / tensorrt-llm / FP8 KV 复用路径的反量化 scale
- **原因行为**：FP8 量化 KV 下启用 block reuse 时，reuse 路径读回 FP8 存储的 K/V，其 per-tensor/per-block 反量化 scale 无法与新计算的 KV 一致往返，且大的 KV 离群值超出 FP8 范围。
- **影响对象**：KV状态；输出内容（不可复现）
- **影响详情**：相同 query 的输出几乎完全错误且逐次运行都不同（非确定性）；FP16 下同一 reuse 路径输出正确。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/NVIDIA/TensorRT-LLM/issues/2699；https://github.com/NVIDIA/TensorRT-LLM/issues/4218
- **关联**：var:INF-0675

### INF-1293 `fp8_kv_fa3_longctx_accum_collapse` FP8 KV 内核累加不足致长上下文检索准确率塌方

`故障` · `active` · `ambiguous` · `config` · `verified` · `var` · 发现 待定 · 旧号 K9.2

- **原因对象**：量化 / vllm / FA3 FP8-KV 累加、DeepGEMM 缩放
- **原因行为**：FA3 的 FP8-KV 内核 FP32 累加不足；同时在 B200 上 DeepGEMM fp8_gemm_nt 的 E8M0 块缩放格式对非 MoE 的 FP8 线性层不一致。
- **影响对象**：数值结果；输出内容（质量退化）
- **影响详情**：短 QA 完全正常，128k 大海捞针准确率 91%→13%，无崩溃无 NaN；约几 k token 以下不可见，只在长上下文检索类评测上暴露（B200、FP8 KV）。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/37618；https://github.com/vllm-project/vllm/issues/23813
- **关联**：var:INF-0675

### INF-1294 `fp8_kv_e5m2_vl_default_scale_flip` fp8_e5m2 KV 默认 scale 损坏视觉语言模型的图像理解

`故障` · `active` · `ambiguous` · `config` · `verified` · `var` · 发现 待定 · 旧号 K9.3

- **原因对象**：量化 / vllm / fp8_e5m2 KV 未校准默认 scale
- **原因行为**：kv_cache_dtype="fp8_e5m2" 只有 2 位尾数，叠加未校准的默认 scale，把视觉 token 的 KV 分辨率彻底压垮，文本 token 的 KV 尚能承受。
- **影响对象**：KV状态；输出内容（静默错误）
- **影响详情**：纯文本请求正常，一旦输入含图像即给出事实错误的视觉描述（纯白图片答成红色），无警告无异常；见于 Qwen2/2.5-VL。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/41343
- **关联**：var:INF-0675

### INF-1295 `fp8_kv_mla_multiturn_garbage` MLA 模型 FP8 KV 量化误差跨轮复合致多轮对话乱码

`故障` · `active` · `ambiguous` · `config` · `documented` · `NEW` · 发现 待定 · 旧号 K9.4

- **原因对象**：量化 / vllm / MLA 潜变量 c_KV 的 fp8 量化
- **原因行为**：对 MLA 架构使用 --kv-cache-dtype fp8 时，低秩潜变量 c_KV 的 fp8 量化误差跨对话轮次不断复合，直至潜变量不可恢复。
- **影响对象**：KV状态；输出内容（质量退化）
- **影响详情**：首轮乃至前几轮质量正常，随轮数增加逐步退化为乱码，单轮回归测试无法复现；见于 GLM-4.7-Flash。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/38652

### INF-1334 `fp8_kv_missing_or_default_scale_silent_corruption` FP8 KV 的 scale 被过滤或取默认值致 KV 量化失真

`故障` · `active` · `ambiguous` · `config` · `documented` · `var` · 发现 待定 · 旧号 K10.23

- **原因对象**：量化 / vllm / FP8 KV scale 加载过滤与默认缩放
- **原因行为**：两条路径令 KV 在错误网格上量化：FP8 MoE 检查点的 k_scale/v_scale 在加载时被过滤、回落为 1.0；e5m2 路径使用默认单位缩放，小幅值 K/V 落入格式近零粗糙区。
- **影响对象**：KV状态；输出内容（质量退化）
- **影响详情**：启用 FP8 KV cache 后输出质量无声下滑，配置与日志一切正常、无告警；KV 幅值落在格式粗糙区时才显著，KV 被吸附到与取值范围无关的固定网格。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/25047；https://github.com/vllm-project/vllm/issues/41343
- **关联**：var:INF-0675

### INF-1335 `fp8_kv_frozen_calibration_miscalibration` vLLM FP8 KV 单批冻结校准低估范围致离群值截断

`故障` · `active` · `ambiguous` · `config` · `documented` · `var` · 发现 待定 · 旧号 K10.24

- **原因对象**：量化 / vllm / calculate-kv-scales 冻结校准
- **原因行为**：--calculate-kv-scales 从一次哑 pass 派生冻结的每层 KV 缩放；单批次校准低估解码时真实动态范围，之后超出范围的离群 token 被截断到饱和平台。
- **影响对象**：KV状态；输出内容（质量退化）
- **影响详情**：短输出正常，长序列后段开始质量下滑，是否发生强烈依赖 prompt 的数值分布；K/V 幅值在截断阈值处堆积成尖峰。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/37554
- **关联**：var:INF-0675

### INF-1336 `fp8_attn_query_scale_defaults_to_kscale` FP8 查询缺 q_scale 时默认取 k_scale 致 logit 偏置

`故障` · `active` · `ambiguous` · `config` · `documented` · `NEW` · 发现 待定 · 旧号 K10.25

- **原因对象**：量化 / vllm / FP8 查询 q_scale 默认值
- **原因行为**：查询也走 FP8 量化而未提供 q_scale 时，vLLM 直接令 q_scale := k_scale；Q 与 K 取值范围不同，每个 QK^T logit 都被错误缩放，属读取侧缩放错误。
- **影响对象**：数值结果；输出分布（静默错误）
- **影响详情**：仅 Q 也走 FP8 时出现：attention logit 系统性偏置，softmax 温度实质改变，输出偏硬或偏软且无告警；存储的 K/V 与参考一致。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/25047

### INF-1339 `lowbit_pertensor_key_quant_breaks_output` llama.cpp q4_0/q5_0 KV 量化粒度不足致 GLM 输出破坏

`故障` · `active` · `ambiguous` · `partial` · `documented` · `var` · 发现 待定 · 旧号 K10.28

- **原因对象**：量化 / llama.cpp / q4_0/q5_0 KV cache 量化
- **原因行为**：q4_0/q5_0 的 KV 量化按张量粒度，无法容纳键上按通道分布的离群值；该机制嫁接自 KIVI 论证，issue 未确凿指向此原因。
- **影响对象**：KV状态；输出内容（质量退化）
- **影响详情**：GLM 在 q8_0 KV 下正常，切到 q4_0/q5_0 后输出明显破坏，呈 bit 宽阈值效应；量化误差集中在键通道，值接近参考。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/ggml-org/llama.cpp/issues/19036
- **关联**：var:INF-1267

## 位置编码

### INF-0884 `yarn_attention_scaling_missing` YaRN 遗漏 attention_scaling 温度因子

`故障` · `active` · `model_on` · `config` · `verified` · `NEW` · 发现 待定 · 旧号 9.1

- **原因对象**：位置编码 / 通用 / YaRN attention_scaling
- **原因行为**：YaRN 只应用了 NTK-by-parts 频率重缩放，注意力温度因子 attention_scaling 被遗漏或置为 1.0，q、k 未乘 sqrt(1/t)，旋转角与 logits 温度补偿不再匹配
- **影响对象**：隐状态；输出内容（质量退化）
- **影响详情**：文本仍然连贯，但长上下文检索能力下降，表现为微妙的质量漂移；各层注意力分布一致地变尖或变平，幅度小且无突变点
- **改进措施**：检测 待定；修复 待定
- **来源**：https://arxiv.org/abs/2309.00071

### INF-0885 `rope_scaling_dropped_no_scaling` 服务侧加载丢弃 rope_scaling，退化为朴素 RoPE

`故障` · `active` · `model_on` · `config` · `verified` · `var` · 发现 待定 · 旧号 9.2

- **原因对象**：位置编码 / 通用 / rope_scaling 配置加载
- **原因行为**：服务侧加载路径把 rope_scaling 丢弃（置为 None），用 PI/NTK/YaRN/Llama3 类缩放训练过的长上下文权重按原始 base 频率的朴素 RoPE 运行
- **影响对象**：隐状态；输出分布（质量退化）
- **影响详情**：低于原始训练上下文长度时与健康模型不可区分，超过该长度 perplexity 陡峭断崖、迅速劣化；只在短序列上评估会完全漏检
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/unslothai/unsloth/issues/3083；https://github.com/vllm-project/vllm/issues/10293
- **关联**：var:INF-0912

### INF-0886 `rope_scaling_wrong_type` 服务时 rope_type 与训练不一致，频率带被错误重映射

`故障` · `active` · `model_on` · `config` · `verified` · `var` · 发现 待定 · 旧号 9.3

- **原因对象**：位置编码 / 通用 / rope_scaling.rope_type
- **原因行为**：服务时使用的 rope_type（linear/dynamic）与训练时（yarn/llama3）不同；transformers<4.43 不识别 llama3 类型而静默回落到其他类型，各旋转频率带按错误规则重映射
- **影响对象**：隐状态；输出内容（质量退化）
- **影响详情**：长上下文能力出现弱到中等的退化，误差只在部分旋转频率带上发生；对整层隐状态求全局范数的度量会因误差被平均而漏检
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/6689；https://github.com/OpenNMT/CTranslate2/issues/1743
- **关联**：var:INF-0912

### INF-0887 `rope_interleave_convention_mismatch` 跨 RoPE 约定移植权重未置换 q/k 维度

`故障` · `active` · `infra_off` · `config` · `verified` · `NEW` · 发现 待定 · 旧号 9.4

- **原因对象**：位置编码 / 通用 / q/k 权重跨 RoPE 约定转换
- **原因行为**：在 interleaved（Meta-LLaMA/GPT-J）与 half-split（HF/GPT-NeoX）两种 RoPE 约定间移植权重时，没有对 q/k 维度做相应置换，每次旋转都作用在错误的维度配对上
- **影响对象**：隐状态；输出内容（质量退化）
- **影响详情**：非常严重：很快输出乱码或不连贯文本，相对位置信息被彻底打乱；逐维误差幅度随维度索引非单调，跟随置换结构
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/zai-org/ChatGLM2-6B/issues/414

### INF-0888 `bf16_rope_precision_breakdown` BF16 计算 RoPE sin/cos 在长序列下精度崩溃

`故障` · `active` · `infra_off` · `config` · `verified` · `var` · 发现 待定 · 旧号 9.5

- **原因对象**：位置编码 / 通用 / RoPE cos/sin/inv_freq 精度
- **原因行为**：RoPE 的 cos/sin/inv_freq 以 BF16 计算/存储，位置 m 很大且落在某些频率带时精度不足，连续若干位置映射到同一数值，误差沿序列累积
- **影响对象**：隐状态；输出内容（质量退化）
- **影响详情**：受长度与 dtype 双重门控：短序列或 fp32 下完全正常，长序列 + bf16 下劣化，表现为重复和位置信息丢失；扰动随位置索引增大
- **改进措施**：检测 待定；修复 待定
- **来源**：https://arxiv.org/abs/2411.13476
- **关联**：var:INF-0765

### INF-0890 `vllm_mrotary_triton_neox_hardcode_gptj_wrong` MRotaryEmbedding 硬编码 NeoX 旋转，GPT-J 式模型出错

`故障` · `active` · `model_on` · `partial` · `documented` · `var` · 发现 2026-07-31 · 旧号 9.7

- **原因对象**：位置编码 / vllm / MRotaryEmbedding 内核
- **原因行为**：MRotaryEmbedding 的 Triton kernel 硬编码 NeoX-style 旋转（两半互换），没有按模型配置切换到 GPT-J style（相邻奇偶维度成对旋转）
- **影响对象**：数值结果；输出内容（静默错误）
- **影响详情**：GPT-J style 旋转模型（如 GLM-OCR）的 key 以错误方式旋转，attention 点积错位，输出仍连贯但静默错误，无 crash、无 NaN
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/49290
- **关联**：var:INF-0887

### INF-0891 `vllm_fused_qknorm_rope_divergent_warp_collective_partial_neox` fused_qk_norm_rope 部分旋转下 warp UB 损坏 Q/K

`故障` · `active` · `infra_off` · `yes` · `verified` · `NEW` · 发现 2026-08-03 · 旧号 9.8

- **原因对象**：位置编码 / vllm / fused_qk_norm_rope 内核
- **原因行为**：partial-NeoX 路径只有 rotary lanes 进入 RoPE 段，段内却以 FINAL_MASK 调用 __shfl_xor_sync 等 warp collective，mask 包含不执行的 lane，属未定义行为
- **影响对象**：数值结果；隐状态（静默错误）
- **影响详情**：packed 多 head/warp 内核中，每个 warp chunk 首个 head 之后全部 head 的 Q/K 被静默损坏；H200 上 token_heads_per_warp=8 时还可出现 NaN；无 crash，仅 rotary_dim<head_dim 的 NeoX 模型触发
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/50903

### INF-0893 `sglang_multimodal_draft_extend_mrope_oob_access` SGLang 多模态 draft-extend mRoPE 位置数不足致越界

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-08-09 · 旧号 9.10

- **原因对象**：位置编码 / sglang / _compute_mrope_positions
- **原因行为**：多模态 extend 分支的 fallback 调用 decode 形状的 _expand_mrope_from_input，每请求只产 1 个 position，而 draft-extend 需要 extend_seq_len 个。
- **影响对象**：可用性/进度（报错）
- **影响详情**：Qwen3-VL + EAGLE3 在首个 batch>1 的 draft-extend 崩溃：fused Triton mRoPE kernel CUDA illegal memory access，forward_native 报 shape mismatch；单请求正常，真实流量才崩。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/pull/34154

### INF-0894 `sglang_rope_flat_kernel_pow2_padding_column_overwrite` SGLang flat RoPE 内核缺列掩码，越界覆写相邻 head

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-08-17 · 旧号 9.11

- **原因对象**：位置编码 / sglang / flat RoPE Triton 内核列掩码
- **原因行为**：flat RoPE 内核按 next_pow2(rope_dim) 遍历列且 load/store 只带行掩码；rope_dim 非二次幂时 padding 列被旋转并 in-place 写入下一个 head，末行越过 tensor 末端。
- **影响对象**：数值结果；隐状态（静默错误）
- **影响详情**：rope_dim 非二次幂的模型配置（deepseek_v4 rope flat kernel）下无 crash，相邻 head 的 RoPE 输入被部分旋转两次，注意力模式静默错误；tensor 末端越界写是潜伏的内存破坏。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/35096

### INF-0895 `sglang_mrope_3d_positions_1d_fused_rope_flatten` SGLang 融合 RoPE 内核只读 mRoPE 首行致视觉 Q/K 错

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `var` · 发现 2026-08-18 · 旧号 9.12

- **原因对象**：位置编码 / sglang / qwen3_5 融合 rope_gate 内核选择
- **原因行为**：attn_output_gate=True 时 Qwen3.5 系无条件用只接受 1-D positions 的融合 QK-norm+RoPE kernel；(3,T) mRoPE 位置传入后只读 temporal 行，H/W 位置未施加。
- **影响对象**：数值结果；输出内容（质量退化）
- **影响详情**：Qwen3.6/3.8-27B 多模态图像/视频 prefill 时视觉 token 的 Q/K 静默错误，图像理解与 GUI 坐标定位（grounding）退化，文本流畅掩盖根因；v0.5.16/v0.5.17/main 均受影响，无 mrope_section 的文本版不受影响。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/35345
- **关联**：var:INF-0893

### INF-0896 `hf_qwen2vl_get_rope_index_bs1_shape_regression` transformers Qwen2-VL rope 索引形状回归

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-08-18 · 旧号 9.13

- **原因对象**：位置编码 / transformers / Qwen2-VL get_rope_index
- **原因行为**：transformers 5.15.0 中 Qwen2VLForConditionalGeneration 的 get_rope_index 在 mRoPE 位置索引计算中出现形状传播回归。
- **影响对象**：可用性/进度（报错）
- **影响详情**：官方 example 单图、batch_size=1 的最小负载即抛 shape mismatch 异常（fail-loud，非静默）；回退 transformers 版本可规避。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/huggingface/transformers/issues/48057

### INF-0897 `vllm_dynamic_ntk_negative_base_rope_nan` vLLM 动态 NTK 在服务长度低于训练长度时得复数 base 致 NaN

`故障` · `active` · `infra_off` · `config` · `verified` · `NEW` · 发现 2026-08-19 · 旧号 9.14

- **原因对象**：位置编码 / vllm / 动态 NTK RoPE 缩放 base 计算
- **原因行为**：动态 NTK 缩放项 inner=factor*served/trained-(factor-1) 未防守 served<trained 的负值，负底数分数幂得复数 base，cos/sin cache 转实 dtype 后为 NaN。
- **影响对象**：输出内容；数值结果（数值异常）
- **影响详情**：nomic-embed-text-v1 以 --max-model-len 512 服务（训练长度 2048）时 embedding 输出全 NaN，却以 HTTP 200 静默返回、无 crash，下游检索/聚类全部失效。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/52665

### INF-0900 `sglang_qwen35vl_fused_partial_interleaved_mrope_grounding_offset` SGLang 融合内核误处理 partial+interleaved MRoPE

`故障` · `active` · `infra_off` · `other_stack` · `verified` · `var` · 发现 2026-08-29 · 旧号 9.17

- **原因对象**：位置编码 / sglang / 融合 rope_gate 内核 MRoPE 处理
- **原因行为**：fused_qk_gemma_rmsnorm_rope_gate 内核对 partial_rotary_factor=0.25 与 mrope_interleaved=True 组合的内部处理有误；位置 id 与 rotary 参数均正确。
- **影响对象**：数值结果；输出内容（静默错误）
- **影响详情**：Qwen3.5-VL grounding bbox 尺寸正确但系统性偏移约 90px，中心精度 26.7%（HF/vLLM 80–82%），OCR/VQA 不受影响；首个 full_attention 层起发散，改走标准 MRotaryEmbedding 路径恢复到 81.7%。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/36762
- **关联**：var:INF-0899

### INF-0901 `hf_nested_rope_parameters_layer_types_misparse_pollution` transformers 短层模型嵌套 rope 参数误判为扁平

`故障` · `active` · `infra_off` · `config` · `verified` · `NEW` · 发现 2026-08-29 · 旧号 9.18

- **原因对象**：位置编码 / transformers / rope 参数嵌套判别
- **原因行为**：standardize_rope_params/validate_rope 以 key 是否为实际 layer_types 子集判别嵌套；短层模型无 full_attention 层时，嵌套 dict 被当作扁平全局 dict。
- **影响对象**：模型权重；指标/日志（静默错误）
- **影响详情**：Gemma3/Olmo3 两层等短 config 下顶层被注入 rope_type/rope_theta=None 并随 save_pretrained 持久化进 checkpoint，真实 per-layer yarn/linear 参数从未被校验，且产生假 warning；当前推理仍正确。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/huggingface/transformers/issues/48392

### INF-0903 `hf_transformers_rope_tf32_integer_position_truncation_b200` RoPE 角度 matmul 走 TF32 截断 2048 以上位置

`故障` · `active` · `infra_off` · `config` · `verified` · `var` · 发现 2026-08-31 · 旧号 9.20

- **原因对象**：位置编码 / transformers / RoPE forward 角度 matmul
- **原因行为**：RoPE 角度用 fp32 matmul（inv_freq @ position_ids），禁用 autocast 不阻止 TF32；精度为 high 时 TF32 的 10 位尾数把大于 2048 的整数位置截断（2049→2048）。
- **影响对象**：数值结果；输出内容（质量退化）
- **影响详情**：B200（sm_100）上 set_float32_matmul_precision("high")、batch>1 且上下文超过 2048 token 时 RoPE 角度大错，长上下文输出静默退化；A100 或 batch=1 不触发，Llama/Qwen/Mistral 共用此路径。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/huggingface/transformers/issues/48428
- **关联**：var:INF-0318

### INF-0904 `vllm_fourier_rope_sin_pad_one_corrupts_unrotated_channels` vLLM Fourier RoPE 的 sin padding 填 1 而非 0

`故障` · `active` · `infra_off` · `config` · `verified` · `NEW` · 发现 2026-09-04 · 旧号 9.21

- **原因对象**：位置编码 / vllm / FourierRoPE sin 表 padding
- **原因行为**：FourierRotaryEmbedding 对 sin 查找表的 padding 填 1 而非 0；未旋转通道本应施加恒等（sin=0、cos=1），却被施加幅度为 1 的虚假旋转分量。
- **影响对象**：数值结果；输出内容（静默错误）
- **影响详情**：使用 FourierRotaryEmbedding 的模型无 crash，padding 涉及的通道子集位置编码静默损坏，输出质量劣化、长上下文尤甚；短 prompt 上症状可能轻微到逃过冒烟测试。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/55204

### INF-0905 `sglang_tokenspeed_mla_nope_layer_rope_assumption_crash` SGLang tokenspeed MLA 对 NoPE 层假设 RoPE 存在

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-09-05 · 旧号 9.22

- **原因对象**：位置编码 / sglang / tokenspeed_mla 后端 RoPE 假设
- **原因行为**：解包 wrapper 后 Kimi Linear 的 NoPE MLA 层直落 TokenspeedMLABackend，其 prepare_prefill_qkv hook 未对 rotary_emb 为 None 设守卫。
- **影响对象**：可用性/进度（报错）
- **影响详情**：Kimi Linear + tokenspeed_mla 后端 + DCP + fp8_e4m3 下每个 TP rank 在 prefill 时 AttributeError 崩溃；B200 上三个 CI job 全部失败。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/pull/38158

### INF-0907 `vllm_yarn_alias_variants_double_scale_max_len_rope_cache_oob` vLLM YaRN 别名绕过精确串判定致 max_len 二次缩放

`故障` · `active` · `infra_off` · `config` · `verified` · `NEW` · 发现 2026-09-11 · 旧号 9.24

- **原因对象**：位置编码 / vllm / max_len 推导、vendor YaRN 类
- **原因行为**：_get_and_verify_max_len 只对精确串 "yarn" 重置派生长度，telechat3-yarn 等别名的已缩放长度被再乘 factor；vendor 类 attention_factor/mscale 与上游不一致。
- **影响对象**：数值结果；可用性/进度（质量退化）
- **影响详情**：TeleChat3-36B 最大长度算成 131,072（应 32,768）、sarvam-105b 算成 5,242,880，放行请求越过 cos/sin cache 多数 crash；未 crash 时 attention_factor 错位，长上下文质量无告警退化。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/56446

### INF-0908 `bug_perfect_disguise` position_ids 相邻对调 off-by-one，检测器近乎看不见

`故障` · `active` · `model_on` · `yes` · `verified` · `NEW` · 发现 待定 · 旧号 9.25

- **原因对象**：位置编码 / 通用 / RoPE position_ids 相邻对调
- **原因行为**：距末尾 offset 处两个相邻 position_ids 被对调的 off-by-one（原型以 rotary embedding forward pre-hook 注入，模拟服务栈构造 position_ids 的错误）。
- **影响对象**：输出内容；隐状态（静默错误）
- **影响详情**：输出流利、自信但错误，几乎每条 prompt 的 greedy 答案都翻转；中层隐状态仍在健康 per-channel 分布内，geometric/trajectory/spectral 与 cheap 族均近乎不可检测，用户侧看不出是基础设施 bug。
- **改进措施**：检测 待定；修复 待定
- **来源**：引用：本项目早期实测条目（原型阶段实测，量值不跨栈继承；

### INF-0909 `vllm_nomic_pooling_default_rope_despite_rotary_scaling_factor_2` vLLM Nomic 后端漏 NTK 缩放致长输入 embedding 偏离

`故障` · `active` · `infra_off` · `config` · `documented` · `var` · 发现 2026-10-09 · 旧号 9.31

- **原因对象**：位置编码 / vllm / Nomic 后端 rope 配置映射
- **原因行为**：vLLM native Nomic 后端生成有效 rope 配置时未把 checkpoint 的 rotary_scaling_factor=2 映射为 NTK scaling，生效为无缩放的 rope_type="default"。
- **影响对象**：输出内容（质量退化）
- **影响详情**：超过训练长度 2048 的输入返回有限但实质偏离参考实现的 embeddings（4096 token 最小 token cosine 0.9007，显式 Dynamic NTK factor 2 为 0.99999），检索质量静默退化；未修改基线即复现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/60165
- **关联**：var:INF-0901

### INF-0910 `sglang_get_rope_config_fabricates_rope_theta_10000_per_layer_type_configs` SGLang 解析 v5 逐层 rope 配置时伪造 rope_theta

`故障` · `active` · `infra_off` · `config` · `documented` · `var` · 发现 2026-10-09 · 旧号 9.32

- **原因对象**：位置编码 / sglang / get_rope_config 逐层类型解析
- **原因行为**：get_rope_config 解析 transformers v5 per-layer-type rope 配置时回退默认 rope_theta=10000，而非继承 checkpoint 中的值。
- **影响对象**：隐状态；数值结果；输出内容（质量退化）
- **影响详情**：使用 transformers v5 per-layer-type rope 配置的受影响 checkpoint，RoPE base 被静默改为 10000，无 crash，长上下文质量退化而输出仍流畅。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/43158
- **关联**：var:INF-0915

### INF-0911 `ttmetal_fused_qk_rope_cos_sin_shard_core_set_mismatch_cross_user_rotation` tt-metal 融合 QK rope cos/sin 与 Q/K 核集错位

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-10-12 · 旧号 9.33

- **原因对象**：位置编码 / tt-metal / RotarySetup cos/sin 分片核集
- **原因行为**：RotarySetup 按 2*batch 行填 cos/sin/trans_mat 核集（P300 上 11 宽），Q/K 落 8 宽行；validate 只查核数与 batch 维度，不校验核集一致。
- **影响对象**：数值结果；跨请求/跨样本数据（静默错误）
- **影响详情**：tt_transformers 在 P300 上用 rotary_embedding_llama_fused_qk 时，布局分歧处的用户每步被别的用户的 cos/sin 旋转或读到不含 cos 的 L1，位置编码静默错误，无 crash。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/59969

### INF-0912 `fault_rope` RoPE 故障注入原型：cos/sin 整体乘 factor 弱化位置信息

`故障` · `active` · `model_on` · `config` · `verified` · `NEW` · 发现 待定 · 旧号 9.26

- **原因对象**：位置编码 / 通用 / RoPE cos/sin 输出缩放
- **原因行为**：scale 模式用 rotary embedding forward hook 把输出 (cos, sin) 整体乘以 factor（默认 0.5），cos²+sin² 不再为 1，被旋转的 q/k 维度整体缩小；其余模式原型未接线。
- **影响对象**：隐状态；输出内容（质量退化）
- **影响详情**：位置信息被系统性弱化，长距离依赖与语序敏感任务先劣化；深度剖面从早期层就偏离，靠 trajectory 与 spectral 捕获，只看最后一层 geometric 距离和 cheap 族会漏检；模式名 scaled/scale 不匹配会静默落入 no-op。
- **改进措施**：检测 待定；修复 待定
- **来源**：引用：本项目早期实测条目（原型阶段实测，量值不跨栈继承；

### INF-0913 `ttmetal_hf_rope_decode_cos_sin_shard_grid_mismatches_q_grid_wrong_angles` tt-metal hf_rope 的 cos/sin 与 Q 分片网格错位

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-09-18 · 旧号 9.27

- **原因对象**：位置编码 / tt-metal / HfRotarySetup 分片 grid
- **原因行为**：（未确认）decode HfRotarySetup 在 11 宽 grid 上按 batch 行分片 cos/sin，decode Q 钉在 8×4 block；rotary_embedding_hf 读驻留 shard，跨核时角度错。
- **影响对象**：数值结果；输出内容（静默错误）
- **影响详情**：Blackhole 上 hf_rope decode batch 32 的 test_attention PCC 降到 0.767–0.897（门槛 0.99），batch 1 与 mllama_rope 均通过；Wormhole 不触发，decode 质量随 batch 劣化。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/56986

### INF-0914 `vllm_hunyuan_dynamic_rope_host_side_bool_sync_breaks_cudagraph_capture` vLLM 动态 RoPE 检查 host 同步致 cudagraph 捕获失败

`故障` · `active` · `infra_off` · `config` · `verified` · `NEW` · 发现 2026-09-21 · 旧号 9.28

- **原因对象**：位置编码 / vllm / dynamic_rope_update 钩子
- **原因行为**：Transformers 后端 dynamic_frequency_update 对张量 seq_len>max_seq_len_cached 求 bool，强制 D2H 同步，capture 内非法；被守卫的更新对本模型实为 no-op。
- **影响对象**：可用性/进度（报错）
- **影响详情**：vLLM ≥0.29 中 HunYuan Dense V1 等 dynamic-rope checkpoint 在所有 cudagraph 模式下 capture 失败，启动即死；≤0.28 原生实现正常，cudagraph_mode=NONE 可启动但 decode 性能受损。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/57890

### INF-0915 `vllm_hf_overrides_rope_scaling_dict_replaces_rope_parameters_wholesale_drops_rope_theta` vLLM hf-overrides 使 RoPE base 静默变为 10000

`故障` · `active` · `infra_off` · `config` · `documented` · `var` · 发现 2026-09-25 · 旧号 9.29

- **原因对象**：位置编码 / vllm / hf_overrides、get_rope 回退
- **原因行为**：--hf-overrides 的 rope_scaling 在 v5 中整体替换 rope_parameters、丢掉 rope_theta；直传 rope_parameters 的架构经 get_rope 回退默认 base 10000。
- **影响对象**：数值结果；输出内容（质量退化）
- **影响详情**：Llama-3.2-3B 重述自身 rope_scaling（应为 no-op）后 NLL 1.128→3.660、GSM8K 379→279，输出流畅、零日志；top300 模型中 64 个静默改变 RoPE base；config.json 路径无恙。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/58675
- **关联**：var:INF-0901

### INF-0916 `vllm_qwen3_omni_mrope_offset_double_counts_modality_start_token_all_multimodal_requests` vLLM Qwen3-Omni M-RoPE 位置误读 offset 多一格

`故障` · `active` · `infra_off` · `config` · `verified` · `NEW` · 发现 2026-09-27 · 旧号 9.30

- **原因对象**：位置编码 / vllm / Qwen3-Omni M-RoPE 位置装配
- **原因行为**：get_mrope_input_positions 把 PlaceholderRange.offset（首个 pad token）误当模态起始索引，每个媒体项多发一个 bos 槽位，M-RoPE 位置比 token 跨度长一格。
- **影响对象**：数值结果；输出内容（质量退化）
- **影响详情**：stock chat template 下无报错，但每个图/音/视频请求从首个 pad token 起 M-RoPE 位置全部偏移一位，grounding 与多模态理解静默退化、输出仍流畅；模板以媒体项结尾时抛 Position ids length mismatch。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/58902

### INF-1175 `bug_head_dim_kv_heads_misconfig` head_dim/KV 头数配置错误致 RoPE 误旋转

`故障` · `active` · `infra_off` · `yes` · `documented` · `var` · 发现 待定 · 旧号 13.7

- **原因对象**：位置编码 / 待定 / head_dim/num_kv_heads 配置
- **原因行为**：注意力的 head_dim 或 KV head 数被错误配置（实现 patch/错误配置），使按 head 施加的 RoPE 旋转错位。
- **影响对象**：隐状态；输出内容（质量退化）
- **影响详情**：原文未单独给出现象，按机制推断为语序/指代错误与质量崩塌；隐藏态 off-manifold，呈 per-head RoPE 误旋转签名，与其他 RoPE 故障高度共线、难以区分。
- **改进措施**：检测 待定；修复 待定
- **来源**：引用：PR #32495；
- **关联**：var:INF-0912

### INF-1182 `ttmetal_rotary_embedding_hf_sharded_data_format_not_reconfigured` tt-metal RoPE sharded 内核未重配数据格式

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-01 · 旧号 13.14

- **原因对象**：位置编码 / tt-metal / rotary_embedding_hf 分片内核
- **原因行为**：不重配数据格式：scalar CB 在 srcB 仍为 sin/cos 时 unpack，-1.0 rotate-half 因子解码错误；pack 前缺 pack_reconfig_data_format；head count 写死。
- **影响对象**：数值结果；隐状态（静默错误）
- **影响详情**：Tenstorrent 上特定 head count/config 组合下静默产出错误旋转嵌入，下游注意力基于错误位置编码计算；无 crash，不必然出现 NaN。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/51284

### INF-1249 `left_padding_cache_position_desync` 左填充 position_ids 未按 mask 推导，batch 与单条不一致

`故障` · `active` · `ambiguous` · `partial` · `documented` · `var` · 发现 待定 · 旧号 K4.2

- **原因对象**：位置编码 / transformers / 左填充 position_ids 推导
- **原因行为**：左填充时 position_ids/cache_position 未由 attention mask 推导而按下标朴素推断，计入 pad 前缀，真实 token 得到错误 RoPE 位置，pad 位置也被 attend
- **影响对象**：隐状态；输出内容（不可复现）
- **影响详情**：同一 prompt 在左填充 batch 与单条推理下生成结果不一致，程度随该行 pad 长度增大，batch=1 时消失；自 transformers 4.42 起仅 mask 有无即会移动缓存生成路径的 logits
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/huggingface/transformers/issues/29419；https://github.com/huggingface/transformers/issues/31943；https://github.com/huggingface/transformers/pull/31944；https://github.com/huggingface/transformers/issues/29149
- **关联**：var:INF-0788

### INF-1305 `chunked_prefill_position_ids_wrong_rope` chunked prefill 后续 chunk 位置编号错误致 RoPE 偏移

`故障` · `active` · `infra_off` · `partial` · `verified` · `NEW` · 发现 待定 · 旧号 K9.14

- **原因对象**：位置编码 / transformers / 分块预填充 position_ids 设置
- **原因行为**：chunked prefill 实现为第一个 chunk 之后的各 chunk 错误设置 position_ids，偏移由 chunk 边界派生、结构化且非恒定，使这些 token 被施加错误的 RoPE 旋转。
- **影响对象**：隐状态；输出内容（质量退化）
- **影响详情**：长 prompt 走 chunked prefill 时输出质量下降，短 prompt（单 chunk）正常；无报错。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/huggingface/transformers/issues/43082

### INF-1319 `deepspeed_fastgen_longrope_su_factor_dropped` FastGen rotary 内核忽略 Phi-3 LongRoPE 缩放因子

`故障` · `active` · `ambiguous` · `other_stack` · `documented` · `var` · 发现 待定 · 旧号 K10.8

- **原因对象**：位置编码 / deepspeed / FastGen 分块 KV rotary 内核
- **原因行为**：FastGen 的 ragged 分块 KV rotary 内核只携带 theta_base，忽视 Phi-3 LongRoPE 的 short/long 缩放因子，超过 original_max_position 后相位 OOD。
- **影响对象**：数值结果；隐状态；输出内容（质量退化）
- **影响详情**：短上下文正常；请求长度越过 LongRoPE 的 short/long 交叉点后，长上下文质量急剧劣化。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/huggingface/text-generation-inference/issues/2172
- **关联**：var:rope_scaling_dropped

### INF-1321 `lm_infinite_distance_ceiling_clamp_misconfig` LM-Infinite 距离上限配置错误致长程键 RoPE 损坏

`故障` · `active` · `ambiguous` · `other_stack` · `documented` · `NEW` · 发现 待定 · 旧号 K10.10

- **原因对象**：位置编码 / 通用 / LM-Infinite 距离上限设置
- **原因行为**：LM-Infinite 用 d' = min(d, L_pretrain) 夹紧相对距离，该上限设置错误：过高则相对距离仍 OOD、相位爆炸；过低则远处 token 全部坍缩到同一位置、彼此不可区分。
- **影响对象**：隐状态；输出内容（质量退化）
- **影响详情**：只影响窗口外的长程键：长上下文中远距离信息被噪声化或全挤在一个位置上无法定位，近距离行为正常，短 prompt 不触发。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://arxiv.org/abs/2308.16137

## 分词/模板

### INF-0917 `glitch_token_activation` glitch token 触发异常隐状态，输出怪异、回避或乱码

`故障` · `active` · `ambiguous` · `config` · `verified` · `NEW` · 发现 待定 · 旧号 10.1

- **原因对象**：分词/模板 / 通用 / glitch token 词表项
- **原因行为**：待定
- **影响对象**：输出内容；隐状态（质量退化）
- **影响详情**：输入含 glitch token 时输出怪异、回避、乱码或“不可言说”，伴随重复、类拒答与虚假幻觉；该位置隐状态在 layer 0/1 即为离群点，逐 token 范数与 geometric 距离可检测，整句聚合会削弱信号。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://aclanthology.org/2024.emnlp-main.649/；https://arxiv.org/abs/2405.05417；https://arxiv.org/abs/2404.09894

### INF-0918 `tokenizer_checkpoint_mismatch` 检查点配错 tokenizer，vocab/id 映射错位致垃圾输出

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 待定 · 旧号 10.2

- **原因对象**：分词/模板 / 通用 / tokenizer vocab/id 映射
- **原因行为**：检查点配了与训练时 vocab/id 映射不同的 tokenizer（如 OPT 的 bos_token 映射不一致、Llama tokenizer 用于 Mistral），embedding 行与 id 错位。
- **影响对象**：输入编码；输出内容（质量退化）
- **影响详情**：输出彻底垃圾或严重不连贯，每个 token 位置都被污染；整句 geometric 距离飙升，逐层范数/statistical 离群检测在所有位置都报警，几乎不会漏检，适合作检测器上界参照。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://arxiv.org/abs/2506.09713

### INF-0919 `fast_vs_slow_tokenizer_divergence` fast/slow tokenizer 切分不一致致 input_ids 分歧

`故障` · `active` · `ambiguous` · `config` · `documented` · `NEW` · 发现 待定 · 旧号 10.3

- **原因对象**：分词/模板 / transformers / fast/slow tokenizer 实现
- **原因行为**：Rust fast tokenizer 与 SentencePiece slow tokenizer 在归一化规则、unknown/special-token 处理或 OOV/词表边界处理上不一致，同一文本被切成不同 token 序列。
- **影响对象**：输入编码；隐状态（静默错误）
- **影响详情**：随 tokenizer/template 选择触发；从静默的细微漂移到特定输入上彻底错误的 token。ids 不同时受影响位置出现小幅几何漂移，需逐位置度量才可见；ids 一致时无隐藏态签名，只能对比两条 tokenize 路径的输出发现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://arxiv.org/abs/2506.09713

### INF-0920 `chat_template_faults` 对话模板故障指针：已由既有 fault_template 覆盖

`指针` · `active` · `待定` · `not_target` · `待定` · `existing` · 发现 待定 · 旧号 10.4

- **原因对象**：分词/模板 / 待定 / 对话模板（见 fault_template）
- **原因行为**：指针条目：源文档写明已被既有 fault_template 覆盖（no_template、wrong_system、dropped_bos、garbled），本条不重复描述机制，以 fault_template 实现为准。
- **影响对象**：输入编码（待定）
- **影响详情**：源文档未给出具体表现；签名形态可参照 fault_template（10.66）的 INPUT_LEVEL 深度剖面：扰动从输入层进入、沿层演化。
- **改进措施**：检测 待定；修复 待定
- **来源**：引用：源文档未给出（本条为指向既有实现的交叉引用，未附独立文献或 issue 编号）。

### INF-0921 `eos_template_absent_runaway` 结束 token 未被识别导致回答后无限生成

`故障` · `active` · `model_on` · `config` · `documented` · `NEW` · 发现 待定 · 旧号 10.5

- **原因对象**：分词/模板 / 通用 / 结束 token（EOS）识别配置
- **原因行为**：所用 tokenizer/模板未把模型实际输出的结束 token 识别为 EOS，生成循环不会在该 token 处停止。
- **影响对象**：输出内容（质量退化）
- **影响详情**：选用特定 tokenizer/模板时触发；先给出正确答案，随后永不停止、反复输出该结束 token。有用生成阶段隐藏态正常，隐藏态检测器基本漏检，需对输出 token id/长度做统计检查。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/turboderp-org/exllamav2/issues/803

### INF-0924 `transformers_chat_template_special_tokens_not_escaped` apply_chat_template 未转义用户消息中的特殊 token

`故障` · `active` · `model_on` · `config` · `documented` · `NEW` · 发现 2026-08-06 · 旧号 10.8

- **原因对象**：分词/模板 / transformers / apply_chat_template 转义
- **原因行为**：apply_chat_template 生成模板时未转义用户消息中出现的服务 token 字符串（如 <s>、[/INST]、</s>），它们被当作真实 special token 处理。
- **影响对象**：输入编码（静默错误）
- **影响详情**：用户消息含 special token 字符串时触发；无 crash、无 NaN，如 </s> 被当作序列结束符，模板截断或结构损坏；攻击者可借此在用户消息中嵌入 special token 操纵模板结构。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/huggingface/transformers/issues/47822

### INF-0926 `transformers_additional_special_tokens_silently_dropped` transformers 静默丢弃用户的附加 special tokens

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-08-07 · 旧号 10.10

- **原因对象**：分词/模板 / transformers / tokenizer 兼容 shim 参数转换
- **原因行为**：v5 兼容 shim 以 setdefault 写入 extra_special_tokens、先 pop 出 additional_special_tokens；前者已存在时 setdefault 不覆盖，pop 出的值被丢弃。
- **影响对象**：输入编码（静默错误）
- **影响详情**：tokenizer_config.json 同时含 extra_special_tokens 时，用户的 additional_special_tokens 静默丢失，tokenizer 少注册 special token，下游模板/chat 可能断裂；无 crash、无 NaN。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/huggingface/transformers/issues/47838

### INF-0929 `transformers_additional_special_tokens_setdefault_silent_drop` setdefault 遇空容器致 special token 静默丢失

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-08-09 · 旧号 10.13

- **原因对象**：分词/模板 / transformers / tokenizer 初始化 setdefault
- **原因行为**：setdefault 把 pop 出的 additional_special_tokens 写入 extra_special_tokens；后者为空容器时 key 已存在，不赋值，弹出的 token 被丢弃；3 个初始化路径均如此。
- **影响对象**：输入编码（静默错误）
- **影响详情**：tokenizer_config.json 含空容器 extra_special_tokens（{} 或 []）并声明 additional_special_tokens 时触发；无 crash、无错误，分词器缺少声明的 token，下游模型行为偏移。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/huggingface/transformers/pull/47855

### INF-0931 `vllm_deepseek_v4_tokenizer_vocab_size_overcount_crashes_guided_decoding` DSV4 tokenizer __len__ 双重计数致约束解码崩溃

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-08-10 · 旧号 10.15

- **原因对象**：分词/模板 / vllm / deepseek_v4/v32 __len__
- **原因行为**：__len__ 返回 vocab_size 加 added vocab 数，部分 added token id 已在 base vocab 内而被双重计数（transformers 5.x 下该写法已 stale），v32 同样。
- **影响对象**：可用性/进度（报错）
- **影响详情**：DeepSeek-V4/V3.2 tokenizer 配合 guided decoding（guidance backend）时触发；启动时 vocab_size 不匹配崩溃，如 V4-Flash-0731 算得 129283，真实为 129280。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/51727

### INF-0934 `hf_additional_special_tokens_setdefault_silent_discard` setdefault 求值顺序致 special tokens 静默丢失

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-08-11 · 旧号 10.18

- **原因对象**：分词/模板 / transformers / tokenizer v5 兼容 shim
- **原因行为**：两个 v5 兼容 shim 用 setdefault + pop 把 additional_special_tokens 转为 extra_special_tokens；参数先求值使 pop 总执行，后者已存在时弹出的列表被丢弃。
- **影响对象**：输入编码（静默错误）
- **影响详情**：tokenizer_config.json 同时含 additional_special_tokens 与 extra_special_tokens 时触发；无 crash，tokenizer 两者都收不到，缺失配置的 special tokens，模板/chat 格式可能错误而不报错。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/huggingface/transformers/issues/47838

### INF-0935 `sglang_simulate_acc_len_fill100_byte_fallback_detoken_o_n_squared` SGLang 模拟接受写死 id 100 且 detokenize 降级

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-08-13 · 旧号 10.19

- **原因对象**：分词/模板 / sglang / SIMULATE_ACC_LEN 模拟接受路径；分词/模板 / sglang / DetokenizerManager 提交判定
- **原因行为**：(A) 模拟接受路径 fill_(100) 覆盖全部 accepted id，该 id 在 byte-fallback 词表中为 U+FFFD；(B) detokenize 提交判定靠 endswith U+FFFD，offset 不前进。
- **影响对象**：性能/资源；输出内容（性能退化）
- **影响详情**：设置 SGLANG_SIMULATE_ACC_LEN 时整个输出变为 � 串、benchmark 失去意义；生产路径中任何含 byte-fallback token 的输出（如 DeepSeek-V4 词表）都使增量 detokenization 降级为 O(n²) 重新解码。无 crash。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/34740

### INF-0944 `vllm_tool_strict_flag_leaked_into_chat_template` vLLM 把 tools 的 strict 字段渲染进模型可见模板

`故障` · `active` · `infra_off` · `yes` · `documented` · `NEW` · 发现 2026-08-18 · 旧号 10.28

- **原因对象**：分词/模板 / vllm / tools 透传进 chat template
- **原因行为**：请求级 tools[].function.strict（服务端解码指令，仅用于决定是否挂 structural tags）被透传进 apply_chat_template，原样序列化进模型可见的 <tools> JSON 块。
- **影响对象**：输入编码；输出内容（静默错误）
- **影响详情**：chat completions 带 tools 且给出任一 strict 值时触发；24-tool agent 负载下 strict:false 使模型把畸形 tool call 泄进 content（0/5 成功、4/5 泄漏），tool_calls 为空，无 crash。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/52741

### INF-0959 `vllm_chat_template_content_null_renders_literal_none` vLLM content=null 被模板渲染为字面 None

`故障` · `active` · `infra_off` · `config` · `verified` · `NEW` · 发现 2026-08-30 · 旧号 10.43

- **原因对象**：分词/模板 / vllm / _postprocess_messages
- **原因行为**：_postprocess_messages 只规范化 tool_calls arguments、漏了 content，None 直达 apply_chat_template，Jinja {{ content }} 渲染成字面 "None"。
- **影响对象**：输入编码；输出内容（质量退化）
- **影响详情**：agent 客户端按 OpenAI 规范对 tool-call 轮发 content: null 时，GLM 系 per-turn think 模板每轮注入一个 None；单轮不可见，8–20 轮后输出退化为重复循环或耗尽 max_tokens、零 content token。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/54337

### INF-0966 `hf_luke_tokenizer_plaintext_path_drops_bos_eos` transformers Luke 分词器单文本路径丢 BOS/EOS

`故障` · `active` · `infra_off` · `yes` · `documented` · `NEW` · 发现 2026-09-05 · 旧号 10.50

- **原因对象**：分词/模板 / transformers / LukeTokenizer 后处理器
- **原因行为**：LukeTokenizer 的 TemplateProcessing post-processor 只在 pair 输入分支装配，单文本路径未安装 post-processor，不添加 <s>/</s> 特殊 token。
- **影响对象**：输入编码（静默错误）
- **影响详情**：plain-text（非 pair）输入时编码结果缺少 BOS/EOS，无报错无警告，Luke 类模型输出质量下降；text pair 路径正常，用 pair 形式的评测探测不到。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/huggingface/transformers/issues/48544

### INF-0967 `sglang_default_chat_template_kwargs_override_request_reasoning_effort` SGLang 服务端默认 reasoning_effort 覆盖请求值

`故障` · `active` · `infra_off` · `config` · `verified` · `NEW` · 发现 2026-09-05 · 旧号 10.51

- **原因对象**：分词/模板 / sglang / serving_chat 模板 kwargs 合并
- **原因行为**：_convert_to_internal_request 先把 reasoning_effort 从 kwargs pop 出，_process_messages 再 setdefault 合并服务端默认，键已缺失故默认值被插入并胜出。
- **影响对象**：输入编码；输出内容（静默错误）
- **影响详情**：配置 --default-chat-template-kwargs 含 reasoning_effort 时，请求显式指定的 reasoning_effort（kwargs 内或顶层）被覆盖为服务端默认（xhigh 按 medium 渲染），无拒绝无告警；思考深度与 token 消耗系统性错档。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/38104

### INF-0968 `vllm_include_stop_str_in_output_ignored_for_stop_token_ids` vLLM stop_token_ids 命中时停止符混入输出

`故障` · `active` · `infra_off` · `yes` · `documented` · `NEW` · 发现 2026-09-05 · 旧号 10.52

- **原因对象**：分词/模板 / vllm / 增量 detokenizer update()
- **原因行为**：增量 detokenizer update() 只在与 EOS 共用的 stop_terminated 下跳末 token，从不读 stop_token_ids；stop-string 路径各自处理了包含语义，token-id 路径漏了。
- **影响对象**：输出内容（静默错误）
- **影响详情**：include_stop_str_in_output=False 且命中 stop_token_ids 时，stop token 文本留在 output_text 与流式 delta 中；依赖输出不含停止符的下游解析（agent 协议、结构化拼接）被污染。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/55483

### INF-0969 `hf_barthez_tokenizer_hardcoded_unigram_bpe_vocab_typeerror` transformers Barthez 硬编码 Unigram 致加载报错

`故障` · `active` · `infra_off` · `yes` · `documented` · `NEW` · 发现 2026-09-06 · 旧号 10.53

- **原因对象**：分词/模板 / transformers / BarthezTokenizer.__init__
- **原因行为**：BarthezTokenizer 无条件构造 Unigram 且未声明 model 属性，而 moussaKam/barthez 实为 BPE；v5 原生格式转换把 dict vocab 直通 Unigram，merges 被丢。
- **影响对象**：可用性/进度（报错）
- **影响详情**：transformers v5 加载 moussaKam/barthez 时 fast/slow 路径均抛 TypeError（dict 非 Sequence），无 workaround，整模型不可用；v4.57.1 正常，测试只 pin 了 UNIGRAM 的 mbarthez。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/huggingface/transformers/issues/48567

### INF-0972 `trtllm_llm_args_tokenizer_alias_table_drift_bare_alias_crash` TRT-LLM 别名表副本漂移致自定义 tokenizer 加载失败

`故障` · `active` · `infra_off` · `other_stack` · `verified` · `var` · 发现 2026-09-06 · 旧号 10.56

- **原因对象**：分词/模板 / tensorrt-llm / llm_args tokenizer 别名副本
- **原因行为**：llm_args.py 手工维护第二份 TOKENIZER_ALIASES，canonical 表新增别名未镜像进副本；TorchLlmArgs 校验器对缺失别名按裸字符串 rsplit('.', 1) 解析。
- **影响对象**：可用性/进度（报错）
- **影响详情**：新注册的 --custom_tokenizer 别名经 TorchLlmArgs 校验路径时报 Failed to load custom tokenizer: not enough values to unpack，错误指向标识符格式而非缺失注册；同一别名在其它三个入口正常。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/NVIDIA/TensorRT-LLM/pull/18768
- **关联**：var:ttmetal_unary_chain_specialized_header_drops_chain_tail_op

### INF-0979 `sglang_b12x_branch_dsml_arguments_double_normalization_regression` SGLang b12x 分支回退修复致 DSML 工具参数多包一层

`故障` · `active` · `infra_off` · `other_stack` · `verified` · `var` · 发现 2026-09-13 · 旧号 10.63

- **原因对象**：分词/模板 / sglang / serving_chat/DSML 参数编码
- **原因行为**：预览分支丢失 #28035 修复：serving_chat 已把 arguments 归一化为 dict，encode_arguments_to_dsml 仍无条件 json.loads，异常后把 dict 再包一层 arguments。
- **影响对象**：输入编码；输出内容（静默错误）
- **影响详情**：b12x-vision 预览镜像（dev-v4f-2dgx 等三个 tag）+ DSV4 多轮 tool call 历史时，历史参数被编码为单个 arguments 参数而非展开的具名参数，模型看不到真实参数名/值，agentic 多轮质量退化且无日志。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/39277
- **关联**：var:INF-0549

### INF-0982 `fault_template` chat template 破坏：无模板/错 system/丢 BOS/打乱

`故障` · `active` · `model_on` · `not_target` · `documented` · `NEW` · 发现 待定 · 旧号 10.66

- **原因对象**：分词/模板 / 通用 / chat template 渲染
- **原因行为**：chat template 层破坏的四种模式：不套模板直接喂裸文本、错误的 system prompt、丢掉 BOS、模板被打乱。
- **影响对象**：输入编码（待定）
- **影响详情**：可用错误 chat_template 触发（属配置/数据）；原文未给出可观察后果。原型实现中四模式从未接线到分词路径，实际等价 no-op。
- **改进措施**：检测 待定；修复 待定
- **来源**：引用：本项目早期实测条目（原型阶段实测，量值不跨栈继承；

### INF-0994 `vllm_whisper_verbose_json_drops_words_after_last_segment_at_chunk_cuts` vLLM Whisper verbose_json 在切点处静默丢词

`故障` · `active` · `infra_off` · `config` · `verified` · `NEW` · 发现 2026-09-21 · 旧号 10.78

- **原因对象**：分词/模板 / vllm / Whisper 时间戳 segment 拼装
- **原因行为**：带时间戳解码路径在每个 chunk 末尾只输出到最后一个完整 segment 边界，边界后到音频切点的残段既不闭合为新 segment，也不并入文本。
- **影响对象**：输出内容（静默错误）
- **影响详情**：音频 >30s 且切点落在语音中时，verbose_json 的 segments 与 text 同时缺词，json/text 格式则保留（同 30s 窗 67 vs 59 词）；丢词量取决于切点与最近 segment 边界的距离。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/58029

### INF-0999 `vllm_cross_encoder_max_length_padding_token_type_ids_inherit_last_real_segment` vLLM cross-encoder 的 pad 位段 ID 填错

`故障` · `active` · `infra_off` · `config` · `verified` · `NEW` · 发现 2026-09-22 · 旧号 10.83

- **原因对象**：分词/模板 / vllm / token_type_ids 填充与压缩表示
- **原因行为**：host 侧用最后真实 token 的段 ID（token_type_ids[-1]）回填 pad 位而非 0；worker 侧压缩表示只存一个 0→1 边界，无法编码 0*1*0* 形状，重建后 pad 位仍为 1。
- **影响对象**：输入编码；输出分布（静默错误）
- **影响详情**：BERT 系 /score、/rerank 用 padding="max_length" 时 HTTP 200，但 pad token 被当作文档段送入模型，分数静默偏差、可翻转排序与阈值决策；短输入配长 max_length 时偏差更大。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/58138

### INF-1001 `transformers_qwen2_tokenizer_pretokenizer_regex_overwrite_from_constant_v5_regression` transformers v5 用常量覆盖 Qwen 发布的预分词正则

`故障` · `active` · `infra_off` · `config` · `verified` · `var` · 发现 2026-09-23 · 旧号 10.85

- **原因对象**：分词/模板 / transformers / Qwen2Tokenizer 预分词正则覆盖
- **原因行为**：Qwen2Tokenizer.__init__ 加载 tokenizer.json 后，用模块级常量 PRETOKENIZE_REGEX 无条件覆盖 pre_tokenizer，丢弃 Qwen3.6/3.8 发布的含 \p{M} 新正则。
- **影响对象**：输入编码；输出分布（静默错误）
- **影响详情**：transformers ≥v5 加载 Qwen3.6/3.8 系 tokenizer 时，含 combining marks 的文本（如阿拉伯语、天城文）token 序列与发布 tokenizer 不同（如 6 vs 8 token），英文不受影响；v4.57.6 正确。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/huggingface/transformers/issues/49066
- **关联**：var:INF-0972

### INF-1005 `vllm_chat_role_value_unvalidated_template_drops_whole_message_200` vLLM chat 接口不校验 role 取值，模板静默丢弃整条消息

`故障` · `active` · `infra_off` · `config` · `verified` · `NEW` · 发现 2026-09-28 · 旧号 10.89

- **原因对象**：分词/模板 / vllm / chat_utils 消息渲染与 role 校验
- **原因行为**：/v1/chat/completions 对 message role 只校验类型不校验取值，chat template 无匹配分支时静默跳过该消息，渲染器也不检查非空消息是否渲染出空段。
- **影响对象**：输入编码；输出内容（静默错误）
- **影响详情**：role 拼错（assistan）、大小写（USER）或自定义未覆盖时，HTTP 200、prompt_tokens 下降（如 39→23），该轮内容未进 prompt，模型仍自信作答；同请求在 /v1/messages 被正确 400。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/59090

### INF-1008 `vllm_derender_logprob_token_strings_stripped_sentencepiece_leading_space_dict_collision` vLLM derender logprob token 丢前导空格致键碰撞

`故障` · `active` · `infra_off` · `config` · `verified` · `NEW` · 发现 2026-09-29 · 旧号 10.92

- **原因对象**：分词/模板 / vllm / derender token 字符串转换
- **原因行为**：derender 用 convert_tokens_to_string 做单 token 转换，剥掉 Metaspace ▁ 前导空格，与引擎路径 convert_ids_list_to_tokens 的结果不一致。
- **影响对象**：输出内容；输出分布（静默错误）
- **影响详情**：SentencePiece tokenizer 下 derender 的 top_logprobs 里 ▁true 与 true 键碰撞、静默丢一个候选；text_offset 每遇前导空格 token 偏一位；chat derender 的 token/bytes 亦不符。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/59241

### INF-1018 `sglang_conversation_llama2_style_role_tags_from_index_parity_not_role` SGLang LLAMA2 风格模板按消息序号奇偶推导 role 标签

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-10-09 · 旧号 10.102

- **原因对象**：分词/模板 / sglang / conversation LLAMA2 风格模板
- **原因行为**：sglang.srt.parser.conversation 的 LLAMA2 风格模板按消息序号奇偶推导 role 标签，而不是使用消息自身的 role。
- **影响对象**：输入编码（静默错误）
- **影响详情**：role 序列非常规交替（多轮、纠错对话）时，渲染出的 prompt 中 user/assistant 标签静默错位或互换，无报错。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/43155

### INF-1036 `sglang_dsv4_flash_reasoning_effort_off_by_one` SGLang DSV4 reasoning_effort 映射错位一级

`故障` · `active` · `model_on` · `other_stack` · `verified` · `NEW` · 发现 2026-08-02 · 旧号 11.12

- **原因对象**：分词/模板 / sglang / DSV4 编码器副本 effort 前缀映射
- **原因行为**：SGLang 自维护的 DeepSeek 编码器副本把 reasoning_effort 到 prompt 前缀文本的映射错位一级：high 注入 low 的空文本，vendor 的 max 前缀不可达。
- **影响对象**：输入编码；输出内容（静默错误）
- **影响详情**：DeepSeek-V4-Flash-0731 请求 reasoning_effort=high 实际是 no-op（默认 effort），max 级别无法使用，推理强度静默低于用户预期；无 crash、无 NaN，v0.5.16 及 main 均受影响。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/33185

### INF-1058 `vllm_mtp_spec_decode_streaming_word_drop_byte_merge_boundary` vLLM MTP 流式输出在 byte-merge 边界间歇丢词

`故障` · `active` · `infra_off` · `yes` · `documented` · `NEW` · 发现 2026-08-15 · 旧号 11.35

- **原因对象**：分词/模板 / vllm / 流式增量 detokenization
- **原因行为**：MTP 推测解码一次提交多个 token 时，流式增量 detokenization 在 quote 邻接的 byte-merge token 边界丢失文本；原文未定位到具体代码。
- **影响对象**：输出内容（静默错误）
- **影响详情**：vLLM 0.27.1 + MTP（num_speculative_tokens=2）流式输出间歇丢词，呈约 15 分钟的时间簇；同请求非流式正确、关闭推测解码正确，raw /v1/completions 同样损坏，排除 parser 归因。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/52469

## 模型定义

### INF-0100 `ttmetal_tttv2_ci_migration_four_defects` TTTv2 迁入 tier-1 CI 暴露 models/common 四缺陷

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-21 · 旧号 1.101

- **原因对象**：模型定义 / tt-metal / attention decode 模块；解码/采样 / tt-metal / Sampling1DConfig 索引；观测/评测 / tt-metal / rmsnorm 测试容差
- **原因行为**：四个独立预存缺陷：attention decode PCC 回归、Sampling1DConfig.local_indices 值错、rmsnorm 容差不足、load_tensor 在冷 cache 路径 SIGBUS。
- **影响对象**：数值结果；测试判定（静默错误）
- **影响详情**：在强制 watcher、只读 /mnt/MLPerf、冷 tensor cache 组合下暴露（旧 T3K pipeline 从未施加）：attention PCC 回归与 local_indices 错值为静默数值错误，load_tensor SIGBUS；main 上字节级确定性复现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/54006

### INF-0112 `ttmetal_qwen3vl_vision_tower_aspect_ratio_pcc_degradation` Qwen3-VL vision tower PCC 随长宽比单调退化

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-27 · 旧号 1.113

- **原因对象**：模型定义 / tt-metal / Qwen3-VL vision tower
- **原因行为**：根因未定位：tt_dit 的 Qwen3-VL vision tower 疑有随输入长宽比变化的系统性几何/分块处理缺陷（失败按旋转成对出现，排除噪声）。
- **影响对象**：数值结果；隐状态（质量退化）
- **影响详情**：T3000 上 22 个用例 6 个 PCC 低于 99%：误差随长宽比单调放大（4:3→98.6%、16:9→96.9%、4:1→93.1%），同 canvas 旋转对得分几乎相同；另 2 个最大 canvas 用例触发 allocator TT_FATAL 崩溃。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/54498

### INF-0331 `funasr_fp16_dtype_token_degeneration_loop` FunASR fp16 推理下 ASR 转写退化为单 token 无限循环

`故障` · `active` · `ambiguous` · `partial` · `documented` · `var` · 发现 2026-09-01 · 旧号 4.15

- **原因对象**：模型定义 / vllm / FunASR encoder fp16 中间激活
- **原因行为**：fp16 窄动态范围下 ASR encoder（深层卷积/注意力混合）中间激活上溢或精度损失，使 decoder logits 坍缩到重复吸引子
- **影响对象**：输出内容；隐状态（质量退化）
- **影响详情**：vllm serve 的 /v1/audio/transcriptions 用 --dtype float16 时转写为整屏重复 token（zh.mp3 输出 locklocklock…），服务健康无 crash；--enforce-eager 下仍复现，切回 bf16/fp32 即恢复
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/54569
- **关联**：var:INF-0320

### INF-0332 `transformers_gptneox_eager_softmax_f32_hardcode_caps_fp64` GPT-NeoX softmax 硬编码 fp32 致 fp64 静默降精度

`故障` · `active` · `model_on` · `config` · `documented` · `var` · 发现 2026-09-02 · 旧号 4.16

- **原因对象**：模型定义 / transformers / GPT-NeoX eager softmax
- **原因行为**：eager attention 路径 softmax 硬编码 dtype=torch.float32：低精度下是升 cast，模型转 fp64 时却把每张 attention map 降回 fp32 再转回 fp64
- **影响对象**：数值结果；隐状态（静默错误）
- **影响详情**：无 crash、无 NaN；模型与输入全转 fp64 时前向内部只有 fp32 分辨率，残差 1e-9 扰动的响应被湮灭，扰动敏感性实验静默失败（Pythia-410M，transformers 5.14.1/PyTorch 2.13 CPU）
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/huggingface/transformers/issues/48459
- **关联**：var:INF-0318

### INF-0449 `sglang_dsv4_mhc_combine_nccl_pool_alias` SGLang mHC 回退占用 NCCL 对称池致输出损坏

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-08 · 旧号 6.56

- **原因对象**：模型定义 / sglang / mHC combine 非 TileLang 回退
- **原因行为**：非 TileLang mHC combine 回退在 use_symmetric_memory 内做 y-combine 归约，中间张量与 dtype cast 分配在 NCCL 对称池，与在飞 all-reduce buffer 别名。
- **影响对象**：输出内容；数值结果（静默错误）
- **影响详情**：DSV4 走非 TileLang mHC combine 且启用 symmetric memory 时触发，深度 5 的 verify shape（m=bs×6）恰好产生池占用碰撞；无 crash，committed token 间歇静默损坏（#33800 SM120 损坏的根因）。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/pull/34021

### INF-0471 `vllm_diffusiongemma_full_cudagraph_freezes_attention_mask` vLLM DiffusionGemma FULL graph 冻结 mask

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-08-12 · 旧号 6.78

- **原因对象**：模型定义 / vllm / DiffusionGemma causal 标志
- **原因行为**：_causal_buf 非 int32，metadata build 每次经 .to(int32) 分配新张量；FULL graph 绑定捕获时地址，后续对 _causal_buf 的更新到不了 graph 读的孤儿张量
- **影响对象**：隐状态；输出内容（静默错误）
- **影响详情**：默认 O2（decode_mode=FULL）下 vllm serve diffusiongemma 无参数即触发：经已捕获 FULL graph 的请求静默沿用 warmup 时的 causal/bidirectional 模式，输出质量不可预测地退化，E2E 影响难定量。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/51994

### INF-0624 `vllm_dsv41_dbo_microbatch_shared_sparse_index_buffer_overwrite` vLLM DSV4.1 DBO 微批共享稀疏索引 buffer 被覆写

`故障` · `active` · `infra_off` · `partial` · `verified` · `NEW` · 发现 2026-09-11 · 旧号 6.231

- **原因对象**：模型定义 / vllm / DSV4.1 稀疏索引模型级 buffer
- **原因行为**：top-k token 索引与候选块需跨层存活，被分配为模型级单份 buffer，未按 DBO 微批切分；微批 1 的 source 层可在微批 0 的 consumer 层读取前覆写同一 buffer。
- **影响对象**：数值结果；输出内容（静默错误）
- **影响详情**：DeepSeek-V4.1 开 DBO 微批时，微批 0 的 sparse attention 消费微批 1 的索引而选错 KV 块，输出静默损坏；GB200 四 EP rank 负控制均复现 top-k 覆写，#56400 报告的 NaN 疑与此路径有关。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/56476

### INF-0662 `vllm_dsv4_pro_long_prefill_threshold_small_chunks_confident_wrong_long_context` vLLM DSV4-Pro 小 prefill 阈值致超长上下文答错

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-10-09 · 旧号 6.269

- **原因对象**：模型定义 / vllm / DSV4 compressor 跨块状态
- **原因行为**：根因未定位；单 flag A/B 闭合到 --long-prefill-token-threshold=256 的小 prefill chunk，疑点为 DSV4 compressor/sparse-attention 跨块状态。
- **影响对象**：输出内容（静默错误）
- **影响详情**：DSV4-Pro（1M ctx、fp8 KV）设该 flag 后 >约 380k token prompt 检索自信答错（404k 起部分错、1,033k 全错），去掉 flag 至 1.03M 全对，无任何报错；换 top-k 后端与 indexer dtype 不改善。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/60905

### INF-0787 `causal_mask_future_leak` causal mask 缺失/反转/off-by-one 致未来信息泄漏

`故障` · `active` · `infra_off` · `yes` · `documented` · `NEW` · 发现 待定 · 旧号 8.3

- **原因对象**：模型定义 / 通用 / causal attention mask 构造
- **原因行为**：causal mask 生成错误：完全丢失使模型变双向，反转/转置成反因果，或 off-by-one 把 j<i 与 j<=i 写混。
- **影响对象**：输出分布；隐状态（静默错误）
- **影响详情**：logits 静默出错：off-by-one 只带来轻微漂移，mask 完全丢失则退化为双向注意力；SDPA 的 is_causal=False 可静默变成 no-op（transformers #36150）。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://arxiv.org/abs/2405.14862

### INF-0790 `sliding_window_misapplied` 滑动窗口注意力配置错误或后端未实现致长程召回退化

`故障` · `active` · `model_on` · `yes` · `documented` · `NEW` · 发现 待定 · 旧号 8.6

- **原因对象**：模型定义 / 通用 / sliding window 配置/后端支持
- **原因行为**：window_size 配置错误或为 None，或所选 attention backend 未实现 SWA，有效感受野按 O(sqrt(L)) 增长而非应有的 L*W。
- **影响对象**：输出内容（质量退化）
- **影响详情**：长上下文下长程召回平滑退化，无断崖、无报错、无数值离群，单看隐状态的检测难以发现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/huggingface/transformers/issues/32235；https://github.com/axolotl-ai-cloud/axolotl/issues/1213

### INF-0823 `vllm_transformers_backend_drops_learnable_attention_sink` vLLM Transformers 后端静默丢弃 attention sink

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-08-14 · 旧号 8.39

- **原因对象**：模型定义 / vllm / Transformers 后端 s_aux 处理
- **原因行为**：Transformers modeling backend 收到以 s_aux 传入的 per-head attention sink 后不做处理，未把 sink 折入 softmax 分母，等价于静默丢弃 sink token。
- **影响对象**：输出内容；隐状态（静默错误）
- **影响详情**：GPT-OSS、GraniteSWA/GraniteMoeSWA、DeepSeek V4、MiMo-V2-Flash 等带 learnable sink 的模型走 Transformers backend 时无 crash，输出静默错误，上下文越长漂移越大。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/52156

### INF-0829 `vllm_ds32_sparse_mla_short_prefill_dense_dispatch_capability_mismatch` vLLM DSV3.2 短 prefill 能力协商错配致输出重复 token

`故障` · `active` · `infra_off` · `yes` · `documented` · `NEW` · 发现 2026-08-16 · 旧号 8.43

- **原因对象**：模型定义 / vllm / DeepseekV32Attention；算子/内核 / vllm / sparse MLA 元数据 dense 派发
- **原因行为**：短 prefill 时 sparse-MLA 元数据按后端能力选 dense MHA、让 indexer 跳过 top-k，但 DeepseekV32Attention 只走 MQA，读到被清空的全 -1 top-k buffer。
- **影响对象**：输出内容；隐状态（静默错误）
- **影响详情**：DeepSeek-V3.2 架构（含 GLM-5.2-NVFP4 TP=4）在纯 prefill 序长 ≤ index_topk 且超过后端 decode 阈值时，无 crash，输出静默退化为重复 token。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/52512

### INF-0857 `vllm_glm53flash_kpool_topk_buffer_2048_index_width_overflow` kpool top-k 宽 2176 超 SM120 内核 2048

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `var` · 发现 2026-09-06 · 旧号 8.73

- **原因对象**：模型定义 / vllm / GLM-5.3 有效 index_topk
- **原因行为**：kpool indexer 把 top-k buffer 加池尾并 pad 到 128 倍数（2048→2176），而 SM120 sparse-MLA 内核按恰好 2048 宽实例化，模型侧未相应收缩 index_topk
- **影响对象**：可用性/进度；KV状态（报错）
- **影响详情**：SM120/GB10 上用 stock config 的 GLM-5.3-Flash，带 #53969 有效宽度检查时启动即被拒（要求 2048、实得 2176），需手改 config.json；无检查时内核静默收到超宽 page table，存在越界/错位风险
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/55563
- **关联**：var:INF-0840

### INF-0869 `sglang_torch_native_backend_noncausal_sliding_window_mask_hardcoded_causal` TorchNative 后端滑窗掩码硬编码因果，encoder 退化为全注意力

`故障` · `active` · `infra_off` · `other_stack` · `verified` · `var` · 发现 2026-09-14 · 旧号 8.86

- **原因对象**：模型定义 / sglang / TorchNative 后端滑窗掩码构造
- **原因行为**：_make_sliding_window_mask 把滑窗掩码硬编码为因果式，无非因果分支；forward_extend 只在 causal=True 时启用滑窗，causal=False 的 encoder 路径完全绕过滑窗
- **影响对象**：输出内容；数值结果；隐状态（静默错误）
- **影响详情**：causal=False 的双向滑窗 encoder（如 ModernBERT 族 ettin-encoder）在长序列下静默回退为全局注意力，embedding 与 HuggingFace 参照实质分歧、质量下降，无 crash、无警告
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/39300
- **关联**：var:INF-0835

### INF-1002 `transformers_gemma1_hidden_activation_guard_removed_exact_gelu_silently_since_v4_48` transformers≥4.48 Gemma 1 静默改用 erf GELU

`故障` · `active` · `infra_off` · `config` · `verified` · `var` · 发现 2026-09-23 · 旧号 10.86

- **原因对象**：模型定义 / transformers / GemmaConfig hidden_act 守卫
- **原因行为**：#35235 删去将 hidden_act="gelu" 改为 gelu_pytorch_tanh 的守卫，以 tanh 近似训练的 Gemma 1.0 checkpoint 按字面执行 exact erf GELU，无警告。
- **影响对象**：隐状态；输出分布（质量退化）
- **影响详情**：transformers ≥v4.48.0 加载 gemma-2b/7b/-it、codegemma-2b 时输出静默偏离训练分布，质量小幅但系统性退化，跨版本评测不一致；Gemma 1.1、Gemma 2 不受影响。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/huggingface/transformers/issues/49051
- **关联**：var:INF-1155

### INF-1078 `vllm_glm_moe_dsa_v2_runner_silent_torch_compile_fallback` vLLM GlmMoeDsa 缺 compile 装饰而静默跑 eager

`故障` · `active` · `infra_off` · `partial` · `verified` · `NEW` · 发现 2026-08-28 · 旧号 11.55

- **原因对象**：模型定义 / vllm / DSA 模型类缺 compile 支持装饰
- **原因行为**：DeepseekV32Model 缺 @support_torch_compile，编译成 no-op；补装饰后 fullgraph 拒绝 lru_cache 能力探测与 ContextVar.get，V1 回退撞 Inductor 断言
- **影响对象**：性能/资源（性能退化）
- **影响详情**：GLM-5.2 INT4 + --compilation-config mode=3：启动日志仅一行 "model does not support it" 警告后以 eager 运行，输出正确但延迟/吞吐偏离预期，警告易淹没在启动日志；8×A100 TP8 MTP k=3 实测
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/54197

### INF-1096 `qwen4exp_dflash_dspark_drafter_serving_five_blockers` Qwen4Exp 无法服务 DFlash/DSpark drafter

`故障` · `active` · `infra_off` · `config` · `verified` · `NEW` · 发现 2026-09-09 · 旧号 11.73

- **原因对象**：模型定义 / vllm / Qwen4Exp 层扩展删除捕获通路；运行时/执行 / vllm / V1/V2 model runner 输入准备；模型定义 / sglang / qwen4_exp decoder 层捕获通路
- **原因行为**：Qwen4Exp 删掉 layer_communicator，父类捕获收集点消失、aux hidden states 槽被占；V1 runner 不供 PLE inputs，V2 解引用 None 的 MTP getter
- **影响对象**：可用性/进度；性能/资源（报错）
- **影响详情**：vLLM V1 启动报 RuntimeError: PLE inputs were not prepared（无 drafter 亦然），V2 报 TypeError；SGLang 侧 aux_hidden_states 恒空、drafter 无法挂载；DFlash/DSpark 在该模型上不可用
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/56088；https://github.com/sgl-project/sglang/issues/38589

### INF-1127 `vllm_composite_vlm_tie_word_embeddings_wrong_config_level` vLLM 复合 VLM 读错 tie 标志层级致 lm_head 被丢弃

`故障` · `active` · `infra_off` · `yes` · `documented` · `var` · 发现 2026-08-04 · 旧号 12.11

- **原因对象**：模型定义 / vllm / Mistral3 复合 VLM wrapper
- **原因行为**：复合 VLM wrapper 从顶层 config 而非语言模型子 config 解析 tie_word_embeddings；顶层误为 true 时丢弃真实 lm_head.weight，改用 tied embedding 或随机初始化。
- **影响对象**：模型权重；输出内容（静默错误）
- **影响详情**：无 crash、无 NaN；Mistral3 等复合 VLM 生成“词表对得上但语义不连贯”的输出，表面像在说人话，实则语义错乱。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/51063
- **关联**：var:INF-1119

### INF-1169 `bug_gemma_logit_softcap_missing` Gemma2/3 的 tanh logit softcap 缺失

`故障` · `active` · `ambiguous` · `yes` · `verified` · `NEW` · 发现 待定 · 旧号 13.1

- **原因对象**：模型定义 / 通用 / 注意力与 lm_head softcap
- **原因行为**：Gemma2/3 应在 softmax 前施加 tanh soft-capping（attn cap 50、lm_head cap 30）；实现省略此步，或把 Gemma2 路由到不支持 softcap 的 SDPA/flash 路径。
- **影响对象**：输出分布；输出内容（静默错误）
- **影响详情**：不崩溃，logits 无界、分布细微偏移，模型自信地答错或陷入重复；flash-attention #1016 报 Gemma2 走 flash 时 100% 随机输出。信号集中在最终隐藏态、注意力 logit 量级与 per-layer attention-entropy。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/Dao-AILab/flash-attention/issues/1016

### INF-1170 `bug_prenorm_postnorm_swap` pre-norm 被改成 post-norm，破坏残差通路

`故障` · `active` · `infra_off` · `yes` · `documented` · `NEW` · 发现 待定 · 旧号 13.2

- **原因对象**：模型定义 / 通用 / decoder 层归一化位置
- **原因行为**：把归一化从子层之前（pre-norm）移到残差相加之后（post-norm），却仍使用按 pre-norm 训练的权重运行，破坏残差高速公路。
- **影响对象**：隐状态；输出内容（质量退化）
- **影响详情**：从第 1 层起激活分布严重扭曲、残差流范数不再随深度增长，到深层输出垃圾或 NaN，质量严重崩塌；per-layer 范数增长等 geometric/trajectory 度量可轻易捕获。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://arxiv.org/abs/1910.05895

### INF-1171 `bug_final_norm_missing` lm_head 前的最终归一化层被省略

`故障` · `active` · `model_on` · `yes` · `documented` · `NEW` · 发现 待定 · 旧号 13.3

- **原因对象**：模型定义 / 通用 / 最终 norm（model.norm）
- **原因行为**：去掉 lm_head 之前的最后一个 RMSNorm/LayerNorm，残差流未经归一化直接进入（通常 tied 的）lm_head 做点积。
- **影响对象**：输出分布；隐状态（质量退化）
- **影响详情**：logits 极大且彼此不一致、softmax 饱和，生成重复/退化，偶尔接近垃圾输出，分布呈低熵尖峰；最终隐藏态范数爆炸、kurtosis 尖峰，更早层不受影响，只看中间层的度量会全部漏检。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://arxiv.org/abs/2507.02559

### INF-1173 `bug_activation_fn_swap` 激活函数被替换为其他激活函数

`故障` · `active` · `ambiguous` · `yes` · `documented` · `NEW` · 发现 待定 · 旧号 13.5

- **原因对象**：模型定义 / 通用 / 激活函数实现
- **原因行为**：建模时使用了与原模型不同的激活函数，既可能是结构差异很大的替换（如 ReLU→SwiGLU），也可能是相近激活之间的互换。
- **影响对象**：隐状态；输出内容（质量退化）
- **影响详情**：结构差异大的替换后果灾难性，geometric 度量可直接检出 off-manifold；相近激活互换只留下细微的 on-manifold 统计偏移，需要分布级度量，单点 geometric 度量会漏检。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://arxiv.org/abs/2002.05202

### INF-1174 `bug_layer_skip_duplicate` 模型层被跳过或重复执行（skip/duplicate）

`故障` · `active` · `infra_off` · `yes` · `documented` · `NEW` · 发现 待定 · 旧号 13.6

- **原因对象**：模型定义 / 通用 / Transformer 层执行序列
- **原因行为**：实现 patch 或错误配置使某一层被整体跳过，或该层的变换被重复施加。
- **影响对象**：隐状态；输出内容（质量退化）
- **影响详情**：原文未单独给出用户可见现象，按机制推断为质量退化；逐层 delta-norm 出现零值（skip）或翻倍（duplicate），单点隐藏态快照可能仍落在健康范围内。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://arxiv.org/abs/2404.02258

### INF-1176 `zamba2_causality_violation` Zamba2 纯 PyTorch fallback 状态递推轴错误致因果性违反

`故障` · `active` · `infra_off` · `partial` · `verified` · `NEW` · 发现 2026-07-29 · 旧号 13.8

- **原因对象**：模型定义 / transformers / Zamba2 torch_forward 状态递推
- **原因行为**：未安装 mamba_ssm 时走的纯 PyTorch fallback（torch_forward）中，inter-chunk state recurrence 在错误的轴上 reduce。
- **影响对象**：隐状态；输出分布（质量退化）
- **影响详情**：自回归生成中，位置 P 的 token 会改变 P 之前位置的 logits 与隐状态，违反因果性；输出可能表现为“回溯修正”或质量退化。仅在未安装 mamba_ssm 的 fallback 路径触发。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/huggingface/transformers/issues/47475

### INF-1177 `phi3phimoe_logits_to_keep_none_4d_crash` Phi3/Phimoe logits_to_keep=None 致崩溃

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-07-29 · 旧号 13.9

- **原因对象**：模型定义 / transformers / Phi3/Phimoe 生成输入准备
- **原因行为**：prepare_inputs_for_generation 把 logits_to_keep=None 传入 forward，而未设为有效值，使 forward 输出 4 维 logits。
- **影响对象**：输出分布；可用性/进度（报错）
- **影响详情**：Phi3/Phimoe 生成时 torch.multinomial 因输入维度不合法（只接受 1 或 2 维）抛 RuntimeError 崩溃；无隐藏态签名，可由 logits shape 检查发现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/huggingface/transformers/issues/47530

### INF-1178 `jina_eurobert_backbone_misrouted_qwen3_path` vLLM 将 EuroBERT backbone 误路由到 Qwen3 实现

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-07-30 · 旧号 13.10

- **原因对象**：模型定义 / vllm / JinaV5 backbone 类型识别
- **原因行为**：模型注册/backbone 类型识别错误，把 JinaEmbeddingsV5 的 EuroBERT backbone 路由到 Qwen3 代码路径，用错误的模型实现加载和服务。
- **影响对象**：输出内容；隐状态；可用性/进度（静默错误）
- **影响详情**：部署 jina-embeddings-v5-text-nano 时，EuroBERT 模型按 Qwen3 的实现运行，架构不匹配导致输出错误或服务失败。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/50337

### INF-1190 `vllm_deepseek_vl2_incomplete_language_config_wrong_defaults` DeepSeek-VL2 缺失配置字段回退到错误默认值

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-08-10 · 旧号 13.22

- **原因对象**：模型定义 / vllm / DeepseekVLV2Config 默认值
- **原因行为**：vLLM 用内置 DeepseekVLV2TextConfig 而非 remote config 解析不完整的 language_config，省略字段落到通用默认值（vocab_size=32000、kv_lora_rank=None）。
- **影响对象**：可用性/进度（报错）
- **影响详情**：加载 deepseek-vl2/deepseek-vl2-small 时，DeepseekV2Attention.__init__ 因 None + 64 抛 TypeError；绕过后 embed_tokens 加载因 32000 与 102400 不符抛 AssertionError。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/51302

### INF-1191 `sglang_gemma4_unified_embedder_wrong_input_dim_audio` Gemma4 多模态 embedder 误把输出维度当输入维度

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-08-11 · 旧号 13.23

- **原因对象**：模型定义 / sglang / Gemma4 多模态 embedder 投影
- **原因行为**：Gemma4UnifiedMultimodalEmbedder 把 output_proj_dims（投影输出维度）当 nn.Linear 的 in_features；audio 的输入维度应为 audio_embed_dim（640）。
- **影响对象**：可用性/进度（报错）
- **影响详情**：加载同时含 vision 与 audio config 的 Gemma-4 Unified 检查点时，audio 的 embedding_projection.weight 尺寸不符，weight loading 抛 AssertionError；vision 因 3840 维巧合相等而侥幸通过。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/34392

### INF-1193 `vllm_rocm_deepseek_v4_preshuffled_weight_inlined_as_plain_linear` 删除 override 点致 ROCm preshuffled 权重被误乘

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-11 · 旧号 13.25

- **原因对象**：模型定义 / vllm / _fused_wqa_wkv_gemm 覆盖点
- **原因行为**：commit #51430 删除 _fused_wqa_wkv_gemm 并在调用点内联 fused_wqa_wkv，绕过 ROCm override，加载时已 preshuffle 的权重被普通 linear 相乘。
- **影响对象**：数值结果；输出内容（静默错误）
- **影响详情**：gfx950 上 amd/DeepSeek-V4-Flash-NVFP4 服务正常启动，但每个问题都回答不可解析的重复 token 流，GSM8K 为 0.0000（阈值 0.92）；CUDA 上完全不可见。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/51821

### INF-1195 `hf_gpt2_encoder_attention_mask_silently_discarded` GPT2 encoder_attention_mask 被覆盖为 None

`故障` · `active` · `infra_off` · `config` · `verified` · `NEW` · 发现 2026-08-13 · 旧号 13.27

- **原因对象**：模型定义 / transformers / GPT2Model encoder mask
- **原因行为**：GPT2Model 在 create_bidirectional_mask 前把 encoder_attention_mask 硬编码为 None，覆盖用户传入的 mask；DecisionTransformer 复制了同样代码。
- **影响对象**：隐状态；输出内容（静默错误）
- **影响详情**：encoder 输入含 padding token 时，cross-attention 以 attention_mask=None 计算，padding 的 corrupted values 被静默吸收，输出含 padding 污染；无 crash。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/huggingface/transformers/pull/47946

### INF-1196 `vllm_kimi_k3_megamoe_deepgemm_kwarg_name_mismatch_typeerror` vLLM Kimi-K3 MegaMoE 调 DeepGEMM 参数名错配致崩溃

`故障` · `active` · `infra_off` · `yes` · `documented` · `NEW` · 发现 2026-08-15 · 旧号 13.28

- **原因对象**：模型定义 / vllm / KimiMegaMoEExperts kwarg
- **原因行为**：MegaMoE experts 以 activation_beta 等 kwarg 调 fp8_fp4_mega_moe，而 pinned DeepGEMM 8b1392b 中参数名为 situ_beta/situ_linear_beta。
- **影响对象**：可用性/进度（报错）
- **影响详情**：Kimi-K3 MegaMoE 路径首次 forward 即抛 TypeError: unexpected keyword argument 'activation_beta'，fail-fast，无静默损坏。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/52445

### INF-1198 `vllm_paligemma_image_embed_residual_inverse_scale` vLLM PaliGemma 图像 embedding 残留反向缩放

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-08-18 · 旧号 13.30

- **原因对象**：模型定义 / vllm / PaliGemma 图像特征缩放
- **原因行为**：PR #36787 把 Gemma 的 sqrt(H) 归一化移入 text-only 的 embed_input_ids()，但 paligemma.py 仍对图像特征除以 sqrt(H)，上游原子变更只移植了文本半边。
- **影响对象**：输入编码；隐状态；输出内容（质量退化）
- **影响详情**：PaliGemma/ColPali 多模态推理时，图像特征以正确幅度的 1/sqrt(H)（H=2048 时约 1/45.25）进入 Gemma，图像近乎失效但输出流畅；无 crash、无 NaN。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/52667

### INF-1199 `vllm_nemotron_parse_lm_head_weight_tying_missing` vLLM NemotronParse 缺 lm_head 权重绑定致输出垃圾

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-08-20 · 旧号 13.31

- **原因对象**：模型定义 / vllm / NemotronParse lm_head 绑定
- **原因行为**：vLLM 的 NemotronParseForConditionalGeneration 实现不把 lm_head.weight 绑定（tie）到 decoder.embed_tokens.weight。
- **影响对象**：输出内容；模型权重（质量退化）
- **影响详情**：vLLM v0.26.0 与 0.27.1 上加载该检查点并服务均“正常”，但输出为 token 级全错的无意义垃圾文本；HF 参考实现正常。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/53019

### INF-1200 `vllm_dflash2_subclass_layer_override_shadowed_by_hardcode` vLLM DFlash2 draft 子类层覆盖被基类硬编码遮蔽致加载失败

`故障` · `active` · `infra_off` · `config` · `verified` · `NEW` · 发现 2026-08-23 · 旧号 13.32

- **原因对象**：模型定义 / vllm / DFlashQwen3Model 建层逻辑
- **原因行为**：基类 DFlashQwen3Model 构造器硬编码 DFlashQwen3DecoderLayer 建层，子类 decoder_layer_cls 覆盖不生效，DFlash2 draft 被建成无局部卷积的普通 DFlash 层；属回归。
- **影响对象**：可用性/进度（报错）
- **影响详情**：vLLM 选用 DFlash2 draft 做推测解码即触发：权重加载在 layers.0.attention_conv 上抛 ValueError（无此模块或参数），draft 不可用；错误信息指向权重名而非遮蔽根因，需读完继承链才能定位。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/53449

### INF-1202 `sglang_spark25_attn_gate_hardcoded_sigmoid_ignores_config` SGLang Spark2.5 attn 门控写死 sigmoid 忽略配置

`故障` · `active` · `infra_off` · `other_stack` · `verified` · `NEW` · 发现 2026-09-04 · 旧号 13.34

- **原因对象**：模型定义 / sglang / spark2_5.py attn 输出门控
- **原因行为**：Spark2.5 的 attention 输出门对 g_proj 输出无条件施加 torch.sigmoid，config 中的 gate_attn_act_mode 键存在但实现不消费；实现只覆盖默认配置形态。
- **影响对象**：隐状态；输出内容（静默错误）
- **影响详情**：加载 gate_attn_act_mode 非 sigmoid 的 Spark2.5 checkpoint 时，每 head 标量门控激活错误，attention 输出比例失真并随层数累积，模型输出静默偏差；默认配置模型无感，无 crash、无外部信号，CPU-only 可最小复现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/37639

### INF-1203 `bug_attn_scale` attention 缩放系数错位致注意力分布过尖或过平

`故障` · `active` · `infra_off` · `yes` · `verified` · `NEW` · 发现 待定 · 旧号 13.35

- **原因对象**：模型定义 / 通用 / attn.scaling 缩放系数
- **原因行为**：attention 缩放系数被乘以错误倍数（factor 0.5–2.0，单层或全层），对应现实中把 1/sqrt(head_dim) 写成 1/head_dim，或 head_dim 配错导致缩放错位。
- **影响对象**：隐状态；输出内容（质量退化）
- **影响详情**：factor>1 时注意力分布过尖、丢失上下文混合，factor<1 时过平、输出笼统；factor=2.0 明显劣化答案，factor=0.9 则可能几乎看不出来。计算确定、可重算、无 NaN，cheap 族全线漏检；单层呈 STEP、全层呈逐层累积。
- **改进措施**：检测 待定；修复 待定
- **来源**：引用：本项目早期实测条目（原型阶段实测，量值不跨栈继承；

### INF-1204 `bug_drop_residual` decoder block 丢失残差连接致表示偏移乃至崩溃

`故障` · `active` · `infra_off` · `yes` · `verified` · `NEW` · 发现 待定 · 旧号 13.36

- **原因对象**：模型定义 / 通用 / decoder block 残差连接
- **原因行为**：decoder block 的输出丢掉残差项 h_in，只留下子层变换 f(h_in)，抹掉残差高速路；对应重构 forward 时写漏一个 + 的 bug。
- **影响对象**：隐状态；输出内容（质量退化）
- **影响详情**：单层丢残差使该层之后的表示大幅偏移，全层丢残差时模型基本崩溃、输出无意义；残差流范数随深度增长的健康结构被破坏，residual_exactness_defect、layerwise_cosine_drift 等剧烈响应，单层版本需在注入层下游评估。
- **改进措施**：检测 待定；修复 待定
- **来源**：引用：本项目早期实测条目（原型阶段实测，量值不跨栈继承；

### INF-1205 `bug_rmsnorm_eps` RMSNorm eps 读错数量级致激活沿深度被压缩

`故障` · `active` · `infra_off` · `yes` · `documented` · `NEW` · 发现 待定 · 旧号 13.37

- **原因对象**：模型定义 / 通用 / RMSNorm variance_epsilon
- **原因行为**：RMSNorm 的 variance_epsilon 被设成异常大的值（健康值约 1e-6，最大取 1.0），对应从配置文件读错一个数量级。
- **影响对象**：隐状态；输出内容（静默错误）
- **影响详情**：rsqrt(var+eps) 被 eps 主导，受影响 norm 之后的隐状态整体缩小并沿深度累积；eps=1.0 时输出显著劣化，eps=1e-3 在多数层上则可能几乎无感。余弦类与对整体缩放不变的谱度量会漏检。
- **改进措施**：检测 待定；修复 待定
- **来源**：引用：本项目早期实测条目（原型阶段实测，量值不跨栈继承；

### INF-1206 `transformers_gemma4_audio_eager_bool_logical_not_on_additive_float_mask_uniform_attention` transformers Gemma4 音频塔 eager 掩码极性反转

`故障` · `active` · `infra_off` · `yes` · `verified` · `var` · 发现 2026-09-15 · 旧号 13.38

- **原因对象**：模型定义 / transformers / Gemma4AudioAttention 掩码
- **原因行为**：eager 路径按 bool 约定以 masked_fill(mask.logical_not()) 施加掩码，但拿到的是加性 float 掩码，logical_not 使极性整体反转；4D→5D 分块掩码转换也按 bool 书写。
- **影响对象**：隐状态；输出内容（静默错误）
- **影响详情**：Gemma4 音频塔使用 eager attention 时，允许位被填无效 logits、禁止位反而可 attend，全允许掩码下注意力退化为与内容无关的均匀平均，音频理解静默塌方；无 crash，sdpa 路径正确，换 attention 实现结果大变是唯一线索。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/huggingface/transformers/issues/48818
- **关联**：var:INF-0869

### INF-1207 `sglang_janus_pro_normalize_class_shadowed_by_vae_factory_normalize` SGLang Janus-Pro Normalize 同名遮蔽致图像归一化不可用

`故障` · `active` · `infra_off` · `other_stack` · `verified` · `NEW` · 发现 2026-09-16 · 旧号 13.39

- **原因对象**：模型定义 / sglang / Janus-Pro Normalize 同名遮蔽
- **原因行为**：多文件压平合并时同名未去重，后定义的 VAE factory 函数 Normalize 遮蔽同名 transform 类，CLIPVisionTower 的调用落到 factory；该类 forward 还误调 F.normalize。
- **影响对象**：可用性/进度；隐状态（报错）
- **影响详情**：SGLang Janus-Pro 的 CLIPVisionTower 以自定义 mean/std 初始化时报 TypeError（unexpected keyword 'mean'），图像归一化路径自 #3203 引入起不可用；绕过第一重遮蔽的部署会被第二缺陷错误归一化，形成静默面。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/39708

### INF-1209 `transformers_umt5_decoder_self_attention_is_causal_flag_never_reaches_attention_sdpa_regression` UMT5 decoder 漏传 is_causal 致自注意力非因果

`故障` · `active` · `infra_off` · `config` · `verified` · `var` · 发现 2026-09-27 · 旧号 13.41

- **原因对象**：模型定义 / transformers / UMT5LayerSelfAttention
- **原因行为**：UMT5LayerSelfAttention 未传 is_causal=config.is_decoder（T5/MT5 均传），SDPA 下该 flag 即掩码，decoder 自注意力失去因果性；#47014 迁移 SDPA 时漏改。
- **影响对象**：输出分布；训练信号（质量退化）
- **影响详情**：v5.15.0 起（v5.14.1 正确）UMT5 teacher-forced 前向中各位置可注意后续位置，位置 0 的 logits 随追加 token 改变；无报错，teacher forcing 评估与 logprob 训练目标被污染，用户仅见翻译质量下降。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/huggingface/transformers/issues/49134
- **关联**：var:INF-1195

### INF-1316 `gqa_head_group_repeat_vs_interleave_mismap` GQA KV 头扩展误用 torch.repeat 致查询头路由到错误 KV 组

`故障` · `active` · `infra_off` · `yes` · `documented` · `NEW` · 发现 待定 · 旧号 K10.5

- **原因对象**：模型定义 / 通用 / GQA KV 头扩展（torch.repeat）
- **原因行为**：GQA 的 KV 头扩展用 torch.repeat 平铺（g(h)=h mod H_KV），而非 repeat_kv/repeat_interleave（g(h)=⌊h/r⌋），形状一致无报错，每个查询头都被路由到错误 KV 组。
- **影响对象**：隐状态；输出内容（质量退化）
- **影响详情**：所有 GQA 层同时受影响，输出整体崩坏但无任何异常或报错；仅 MHA（H_KV=H_Q）或 MQA（H_KV=1）配置下恰好等价于正确映射。
- **改进措施**：检测 待定；修复 待定
- **来源**：引用：`gh`：SHI-Labs/NATTEN #92（"GQA without repeat()"）。

## 权重/检查点

### INF-0352 `ep_checkpoint_shard_mismatch` EP 重分片或跨框架转换致 expert 权重错位

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 待定 · 旧号 5.5

- **原因对象**：权重/检查点 / 通用 / EP 重分片/转换的 expert 排序
- **原因行为**：以不同 EP degree 加载、experts 数不被 ep_size 整除或 Megatron↔HF 排序约定不同，expert 落到错误 rank、映射到另一组权重；合并子项含 GQA head 复制、词表 padding 错。
- **影响对象**：模型权重；输出内容（静默错误）
- **影响详情**：错误的 expert 权重被施加，输出被破坏：expert i↔j 互换时两者都是合法权重、逐 token 看似正常而极隐蔽，shape 不符时则直接崩溃。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/deepspeedai/DeepSpeed/issues/5794；https://github.com/deepspeedai/DeepSpeed/issues/6891；https://github.com/sgl-project/sglang/issues/25117；https://github.com/NVIDIA/Megatron-LM/issues/1754

### INF-0382 `vllm_glm53flash_shared_transformer_qkv_key_missing_tp16_two_node` vLLM GLM-5.3-Flash 跨节点 TP16 权重键缺失

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-09-07 · 旧号 5.35

- **原因对象**：权重/检查点 / vllm / shared_transformer 权重映射表
- **原因行为**：shared_transformer（跨 TP rank 共享、不切分）模块的权重加载映射未覆盖 TP16 跨节点切分，映射表中缺 qkv_proj.weight 键；该键是否存在随 TP 切分系数翻转。
- **影响对象**：可用性/进度（报错）
- **影响详情**：官方 glm53-flash 镜像在 2 节点 16×H100 跨节点 TP 下，节点发现完成后于权重映射阶段抛 KeyError: 'layers.11.shared_transformer.self_attn.qkv_proj.weight'；单节点 8×H100 同镜像同权重正常。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/55725

### INF-0602 `sglang_inkling_unified_memory_checkpoint_logical_destination_untranslated` SGLang 统一内存 checkpoint 恢复缺逻辑→物理槽位翻译

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-09-06 · 旧号 6.209

- **原因对象**：权重/检查点 / sglang / 统一内存 checkpoint 恢复
- **原因行为**：checkpoint 恢复路径缺少逻辑目的地到物理槽位的翻译，把元数据中的逻辑索引直接当物理槽偏移写入。
- **影响对象**：模型权重；输出内容（静默错误）
- **影响详情**：走统一内存 checkpoint 恢复路径即触发：恢复后状态写入错误槽位，输出与修复后不同；无 crash，吞吐数字看似正常，PR 作者说明吞吐对照不能验证正确性。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/pull/38229

### INF-0843 `vllm_qwen35_kv_scale_checkpoint_route_missing_autoweights_loader` Qwen3.5 漏重映射 KV scale，fp8 KV 静默用 1.0

`故障` · `active` · `infra_off` · `config` · `verified` · `var` · 发现 2026-08-31 · 旧号 8.59

- **原因对象**：权重/检查点 / vllm / qwen3_5 load_weights 重映射
- **原因行为**：Qwen3.5 的 load_weights 未调 maybe_remap_kv_scale_name，加载器与 ModelOpt 也不重映射，checkpoint 的 k/v_scale 路由不到运行时参数
- **影响对象**：KV状态；模型权重（静默错误）
- **影响详情**：以 --kv-cache-dtype fp8 加载携带校准 KV scale 的 Qwen3.5 家族 checkpoint（如 NVFP4 版）时，无报错，仅加载期一条 WARNING 提示使用 scale 1.0；校准 KV 量化退化为 unit-scale，fp8 KV 精度系统性劣化
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/54623
- **关联**：var:INF-0725

### INF-0933 `hf_cached_files_erofs_silent_stale_return` 只读文件系统上 EROFS 漏检致 HF 缓存静默返回旧文件

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-08-11 · 旧号 10.17

- **原因对象**：权重/检查点 / transformers / cached_files 错误恢复分支
- **原因行为**：写失败 guard 只认 PermissionError（EACCES），只读挂载返回的 EROFS（errno 30）是普通 OSError，被漏检后落入 stale-cache 恢复分支。
- **影响对象**：模型权重；输入编码（静默错误）
- **影响详情**：HF Hub 缓存位于只读文件系统（mount -o ro，如 CI 预填缓存卷）时触发；无 crash，即使 Hub 有新版本也静默返回旧缓存文件，模型/分词器可能用旧版本且差异不可见。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/huggingface/transformers/pull/47852

### INF-1117 `wrong_dtype_cast_bf16_as_fp16` bf16 权重加载时被强转 fp16 溢出为 inf

`故障` · `active` · `infra_off` · `yes` · `verified` · `NEW` · 发现 待定 · 旧号 12.1

- **原因对象**：权重/检查点 / transformers / transformers 加载 dtype 转换
- **原因行为**：加载时经 torch_dtype 或 transformers v5 auto-dtype 把 bf16 权重强制转成 fp16，对全部权重做确定性逐元素转换，超出 fp16 最大值（约 65504）的元素溢出为 ±inf。
- **影响对象**：模型权重；输出内容（数值异常）
- **影响详情**：输出重复、与上下文无关的 token，最终为空输出或 NaN；命中饱和通道时激活直接出现 inf/nan，范数与谱全局偏移。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://arxiv.org/abs/2510.26788

### INF-1118 `from_pretrained_silent_random_init` transformers v5 from_pretrained 静默随机初始化

`故障` · `active` · `infra_off` · `yes` · `verified` · `NEW` · 发现 待定 · 旧号 12.2

- **原因对象**：权重/检查点 / transformers / v5 from_pretrained 键名转换
- **原因行为**：v5 动态权重加载下，自定义 PreTrainedModel 子类 from_pretrained 若无 converter 映射 legacy 键名，返回随机初始化权重，loading_info 干净、无 missing-key 警告。
- **影响对象**：模型权重；输出内容（质量退化）
- **影响详情**：线上服务的模型本质未训练，输出完全或近乎完全失效，加载时不报任何 error 或 warning；受影响模块从第一层起激活即远离健康流形。
- **改进措施**：检测 待定；修复 待定
- **来源**：引用：HF transformers #46620。

### INF-1119 `tied_embedding_random_lm_head` tie_word_embeddings 配置不符致 lm_head 随机初始化

`故障` · `active` · `infra_off` · `yes` · `verified` · `NEW` · 发现 待定 · 旧号 12.3

- **原因对象**：权重/检查点 / transformers / lm_head 加载与 tie 配置
- **原因行为**：config.json 的 tie_word_embeddings 与 checkpoint 实际保存方式不一致（声明不共享却无单独 lm_head.weight，或反之），加载器经 _init_weights 把缺失矩阵随机初始化。
- **影响对象**：模型权重；输出分布（质量退化）
- **影响详情**：输出完全是随机 token，top-1 近似在整个词表上均匀随机；final norm 之前隐藏态健康，异常只出现在 logits 层（熵接近 log(V)），隐藏态几何度量可能漏检。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/huggingface/transformers/issues/39812；https://github.com/sgl-project/sglang/issues/16722；https://github.com/gpustack/gpustack/issues/4268

### INF-1120 `lora_wrong_scaling_alpha` LoRA 缩放系数 alpha 错误致适配器过度或不足施加

`故障` · `active` · `model_on` · `yes` · `verified` · `NEW` · 发现 待定 · 旧号 12.4

- **原因对象**：权重/检查点 / 通用 / LoRA 缩放系数 alpha
- **原因行为**：适配器的缩放系数 alpha 取值错误（如 mlx-vlm 的 8 倍 alpha bug），低秩 delta 被统一标量放大或缩小。
- **影响对象**：模型权重；隐状态；输出内容（静默错误）
- **影响详情**：加载适配器即可复现：输出仍连贯，但风格错误、指令遵循校准失准；属 on-manifold 漂移，纯几何单点度量很可能漏检。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/Blaizzy/mlx-vlm/issues/845；https://github.com/Blaizzy/mlx-vlm/pull/846；https://github.com/huggingface/peft/issues/1836；https://github.com/vllm-project/vllm/issues/22077

### INF-1121 `merge_precision_loss` LoRA merge 与运行时 dtype 或运算顺序不一致致精度损失

`故障` · `active` · `model_on` · `yes` · `verified` · `NEW` · 发现 待定 · 旧号 12.5

- **原因对象**：权重/检查点 / peft / 适配器 merge dtype/运算顺序
- **原因行为**：把适配器 merge 进基权重（W += scaling*B@A）时使用与运行时不同的 dtype 或运算顺序，merged 权重与 unmerged 前向之间产生纯舍入级差异。
- **影响对象**：模型权重；输出内容（静默错误）
- **影响详情**：merged 与 unmerged 两条路径输出出现细微分歧：起初零星 token 不同，长生成中逐步累积放大；量级可能低于运行时噪声底，单靠隐藏态极难检测。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/huggingface/peft/issues/1836；https://github.com/huggingface/peft/issues/1035

### INF-1122 `stale_revision_serve` revision 解析错误或缓存过期致服务合法但非预期的模型

`故障` · `active` · `model_on` · `yes` · `documented` · `NEW` · 发现 待定 · 旧号 12.6

- **原因对象**：权重/检查点 / 待定 / revision 解析/本地缓存
- **原因行为**：revision='main' 解析到错误的 commit，或本地缓存过期，加载的是合法但不是预期的 checkpoint，全程无任何报错。
- **影响对象**：模型权重；输出内容（静默错误）
- **影响详情**：服务的模型自洽、输出看似正常，只有对照预期模型（跨版本基线比对）或校验 checkpoint 指纹才能发现偏离；单模型内部度量原则上漏检。
- **改进措施**：检测 待定；修复 待定
- **来源**：引用：HF Forum #175332、#139695。

### INF-1123 `safetensors_shard_mmap_corruption` 加载/文件/config 操作可复现硬报错或间歇垃圾输出

`故障` · `active` · `infra_off` · `yes` · `documented` · `var` · 发现 待定 · 旧号 12.7

- **原因对象**：权重/检查点 / 待定 / 加载/文件/config 操作
- **原因行为**：权重内容损坏；部分 FSDP 共享参数变体则把权重写入错误槽位。
- **影响对象**：模型权重；输出内容（不可复现）
- **影响详情**：或直接硬报错，或间歇性输出垃圾，呈 off-manifold 簇状跳变，单次快照易漏检；FSDP 写错槽位变体表现为持续性逐张量破坏。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://arxiv.org/abs/2304.11277
- **关联**：var:INF-0315

### INF-1124 `hf_silent_conversion_skip_model_type_mismatch_random_init` transformers 错误 Auto 类加载致转换跳过、权重随机初始化

`故障` · `active` · `infra_off` · `yes` · `documented` · `var` · 发现 2026-07-31 · 旧号 12.8

- **原因对象**：权重/检查点 / transformers / from_pretrained 检查点转换路径
- **原因行为**：转换过的 checkpoint 经错误 Auto 类加载（model_type 不匹配）时，框架静默跳过转换，未初始化的部分权重保持随机初始化。
- **影响对象**：模型权重；输出内容（静默错误）
- **影响详情**：加载“成功”，无 crash、无 NaN、无报错；模型流畅生成与真实权重无关的输出，单帧几何度量未必离群。复现于 Inkling-975B 多模态转换后 checkpoint、transformers 5.14.1。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/huggingface/transformers/issues/47405
- **关联**：var:INF-1118

### INF-1128 `vllm_lora_single_tensor_lora_a_not_normalized_packed_index_error` vLLM set_lora 未展开单 lora_a 致 packed 层报错

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-08-10 · 旧号 12.12

- **原因对象**：权重/检查点 / vllm / packed LoRA 层 set_lora
- **原因行为**：MergedColumnParallelLinearWithLoRA.set_lora 只 normalize lora_b；单个 2-D lora_a 未展开直接透传，lora_a[i] 取到 1-D 行而非第 i 个 slice 矩阵。
- **影响对象**：可用性/进度（报错）
- **影响详情**：LoRA serving 中 set_lora 收到单 tensor lora_a 且目标为 packed 层（fused QKV/gate_up）时：fully_sharded_loras=True 抛 IndexError，False 时拷贝期 shape mismatch。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/51423

### INF-1129 `ttmetal_unconditional_dequant_already_dequant_kimi_checkpoint` tt-metal 对已反量化 Kimi 检查点无条件反量化致加载报错

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-08-10 · 旧号 12.13

- **原因对象**：权重/检查点 / tt-metal / dequantize_state_dict 调用
- **原因行为**：加载路径对量化与已反量化（bf16）检查点统一调用 dequantize_state_dict，不检查 config 是否含 quant metadata；dequantizer 在处理张量前先读取 quant metadata。
- **影响对象**：可用性/进度（报错）
- **影响详情**：加载 Kimi-K2.6 已反量化 bf16 检查点（config.json 无 quantization_config）时，权重加载阶段抛 ValueError: Missing DeepSeek quantization_config.weight_block_size。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/pull/52743

### INF-1131 `sglang_update_weights_from_disk_silent_weight_drop` SGLang 热更新报成功却漏载 VAE buffer 与文本编码器

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-08-12 · 旧号 12.15

- **原因对象**：权重/检查点 / sglang / update_weights_from_disk
- **原因行为**：named_parameters() 排除 buffer，VAE BatchNorm buffer 从未加载；text encoder 走通用名称映射，不能去 model. 前缀或融合 q/k/v，实际未加载；API 无异常即算更新成功。
- **影响对象**：模型权重；输出内容（静默错误）
- **影响详情**：diffusion 模型调用 update_weights_from_disk 报告成功、无错误；VAE 沿用过期 latent 归一化统计，text encoder 仍为旧权重，输出质量不可预测地退化。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/pull/33588

### INF-1135 `vllm_routed_experts_lora_param_mapping_prefix_order` vLLM MoE 专家 LoRA 参数映射前缀拼接顺序颠倒

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-08-16 · 旧号 12.19

- **原因对象**：权重/检查点 / vllm / RoutedExperts LoRA 参数映射
- **原因行为**：build_expert_params_mapping 把 LoRA base-layer 前缀拼在 routed_experts_prefix 之后，与实际 LoRA 参数树的嵌套方向相反；#41184 重构时引入的回归。
- **影响对象**：模型权重（静默错误）
- **影响详情**：MoE 模型 + RoutedExperts + LoRA 时，依赖该 mapping 的 LoRA 加载/管理对专家层静默找不到参数或映射到错误对象，后续 LoRA 服务路径失效；具体表现取决于消费方。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/52552

### INF-1136 `hf_mps_async_bf16_load_fp32_dtype_segfault` transformers 在 MPS 上 bf16 转 fp32 异步加载段错误

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-17 · 旧号 12.20

- **原因对象**：权重/检查点 / transformers / from_pretrained 异步加载
- **原因行为**：异步加载路径中 dtype 转换与异步分片调度在 MPS 后端内存模型上交互不当，加载器未覆盖 dtype/分片/后端组合。
- **影响对象**：可用性/进度（报错）
- **影响详情**：macOS MPS 上加载 bf16 checkpoint 并指定 dtype=torch.float32、device_map="auto" 时进程级段错误，非静默。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/huggingface/transformers/issues/48029

### INF-1138 `vllm_kimi_k3_interleaved_weight_stream_double_finalize` vLLM Kimi-K3 交错权重流致 MegaMoE 重复 finalize

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-08-22 · 旧号 12.22

- **原因对象**：权重/检查点 / vllm / MegaMoE 权重 finalize
- **原因行为**：AutoWeightsLoader 假设同一顶层前缀相邻，交错流使 Kimi 语言模型 loader 被调用两次；每次结束都 finalize MegaMoE 权重并删除原始 weight 与 scale 参数，不可重入。
- **影响对象**：可用性/进度（报错）
- **影响详情**：仅在流式加载产出 language/vision/language 交错前缀时触发（传统全量加载不触发），后到组因缺失参数（如 w13_weight_scale）加载失败。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/53379

### INF-1139 `hf_v5_chunk_op_unequal_split_silent_reinit` transformers v5 Chunk 转换不校验均分致权重错切

`故障` · `active` · `infra_off` · `config` · `verified` · `NEW` · 发现 2026-08-26 · 旧号 12.23

- **原因对象**：权重/检查点 / transformers / Chunk.convert 转换 op
- **原因行为**：core_model_loading.py 的 Chunk.convert 用 torch.chunk 切分融合张量，guard 只校验块数与 target 数一致、不校验均分；尺寸不可整除时产出不等长块且块数吻合，guard 通过。
- **影响对象**：模型权重；可用性/进度（静默错误）
- **影响详情**：from_pretrained 转换不可整除的融合张量时，或在远离因果处 shape mismatch 崩溃、难以归因，或在 ignore_mismatched_sizes=True 下静默重初始化权重、无任何加载信号；CPU 可复现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/huggingface/transformers/issues/48323

### INF-1142 `ttmetal_attn_res_weight_load_local_index_no_global_offset` tt-metal 多 rank 按本地层号加载 AttnRes 权重致取到他层

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-08-30 · 旧号 12.26

- **原因对象**：权重/检查点 / tt-metal / load_attn_res_weights 层索引
- **原因行为**：load_attn_res_weights 取 rank 本地层计数且不加偏移，而 tensorbin cache 与 checkpoint 按全局层索引键控；完备性检查只验名字存在，shape 又相同，错层请求不被拦截。
- **影响对象**：模型权重；输出内容（静默错误）
- **影响详情**：多 rank 流水线（rank>1）下，持层 36..71 的 rank 请求 layers.0..35，用他人 query 正常跑完、输出看似合理，精度劣化且无错误警告；单 rank 不可见，同族验证习语使 PCC 校验误报。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/54884

### INF-1143 `ttmetal_dit_weight_cache_drops_non_parameter_derived_state` tt-metal tt_dit 权重缓存丢失非 Parameter 派生状态

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `var` · 发现 2026-09-01 · 旧号 12.27

- **原因对象**：权重/检查点 / tt-metal / tt_dit Module.save/load
- **原因行为**：Module.save/load 只序列化 named_parameters()，_prepare_torch_state 派生的宿主侧非 Parameter 值不进 weight cache、load 时也不重算，回退为构造器默认值。
- **影响对象**：模型权重；输出内容（静默错误）
- **影响详情**：使用 weight cache 时 cached 与非 cached 部署行为静默分歧，无错误、无 shape mismatch、无日志；实例：LTX-2.5 text encoder 的 prompt adherence 在每次 cached run 被毁。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/55014
- **关联**：var:INF-1131

### INF-1146 `vllm_lora_target_modules_filter_mismatch_silent_base_model_serving` vLLM LoRA 校验与应用过滤器不一致致静默服务基模型

`故障` · `active` · `infra_off` · `config` · `verified` · `var` · 发现 2026-09-03 · 旧号 12.30

- **原因对象**：权重/检查点 / vllm / LoRA 加载校验与模块匹配
- **原因行为**：Worker 侧 _load_adapter 按未过滤的 supported_lora_modules 校验，manager 侧建模块时却按 target_modules 过滤，set_lora 作用于零层。
- **影响对象**：输出内容（静默错误）
- **影响详情**：--lora-target-modules qkv_proj 下加载 MLP-only adapter：校验通过、激活成功、/v1/models 列出，但请求 HTTP 200 返回与基模型逐字相同的输出，无错误无警告；不设该参数则正常生效。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/55193
- **关联**：var:INF-1144

### INF-1148 `vllm_rslora_moe_pack_recompute_alpha_rank_drops_sqrt_r` vLLM pack_moe 重算 scaling 致 rsLoRA MoE 弱化

`故障` · `active` · `infra_off` · `yes` · `verified` · `NEW` · 发现 2026-09-06 · 旧号 12.32

- **原因对象**：权重/检查点 / vllm / pack_moe/pack_moe_stacked
- **原因行为**：pack_moe/pack_moe_stacked 从原始字段重算 scaling=lora_alpha/rank，无视 PEFTHelper 为 rsLoRA 存储的 lora_alpha/√r 因子。
- **影响对象**：模型权重；输出内容（静默错误）
- **影响详情**：rsLoRA adapter 作用于 MoE expert 模块时 delta 被削弱 √r 倍（r=16/alpha=32 应为 8 实为 2），无报错，输出看似健康但微调效果打折；dense 路径正常，CPU 可复现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/55545

### INF-1149 `sglang_glm53_transformers_roundtrip_oneway_rename_drops_97pct_weights` transformers 存盘名错位致 SGLang 静默丢 97% 权重

`故障` · `active` · `infra_off` · `other_stack` · `verified` · `NEW` · 发现 2026-09-09 · 旧号 12.33

- **原因对象**：权重/检查点 / transformers / conversion_mapping 单向重命名
- **原因行为**：transformers 读 GLM-5.3-Flash 时按 conversion_mapping 重命名张量，save_pretrained 不逆向改回；SGLang 加载器只认 released 名，未匹配张量无警告跳过。
- **影响对象**：模型权重；输出内容（静默错误）
- **影响详情**：经 save_pretrained 的 GLM-5.3-Flash checkpoint（微调/合并/dtype 重导出）在 SGLang 启动正常、/health 200，但生成 token 噪声；45 层实测丢 524/1609 张量（97% 文件），MoE/mHC/KDA 留在初始态。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/38618

### INF-1150 `sglang_qwen3_seqcls_head_bias_retained_loading_bias_free_checkpoint` SGLang Qwen3 分类头多出随机初始化 bias

`故障` · `active` · `infra_off` · `other_stack` · `verified` · `NEW` · 发现 2026-09-10 · 旧号 12.34

- **原因对象**：权重/检查点 / sglang / Qwen3 序列分类头 score 层
- **原因行为**：分类头用 nn.Linear 构建、默认 bias=True（transformers 参考为 bias=False）；load_weights 对缺失的 score.bias 无报错或置零守卫，随机初始值被保留。
- **影响对象**：输出分布；模型权重（静默错误）
- **影响详情**：加载只含 score.weight 的合法 checkpoint 时无任何报错，分类 logits 被逐标签的固定随机偏移污染（算 W@h+b_init 而非 W@h），rerank/分类阈值类任务系统性漂移；CPU 合成复现，880d6fa6 与 v0.5.10.post1 均复现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/38768

### INF-1152 `vllm_weight_cache_daemon_shared_ipc_refcount_slot_premature_free` vLLM 权重缓存 daemon 共享 IPC 引用计数槽致权重内存过早释放

`故障` · `active` · `infra_off` · `yes` · `verified` · `NEW` · 发现 2026-09-11 · 旧号 12.36

- **原因对象**：权重/检查点 / vllm / 权重缓存 daemon IPC 导出
- **原因行为**：daemon 对每个 get_state 响应只调一次 reduce_tensor，把同一组参数（含一个消费者的 refcount 槽）发给所有 client，引用计数与实际消费者数脱钩；启动时还留有无主初始引用。
- **影响对象**：模型权重；可用性/进度（静默错误）
- **影响详情**：多 engine 混合 copy/zero-copy client 时，copy-mode client 释放临时 IPC tensor 可使计数归零、底层 allocation 被回收，另一 engine 仍持映射并读悬垂权重：表现为 IMA 或静默权重损坏（取决于复用时机）。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/56472

### INF-1157 `sglang_glm5_next_mtp_skip_guard_raw_name_language_model_prefix_bypass_mtp_weights_into_main_model` SGLang GLM-5.3 MTP 跳过守卫被前缀绕过，MTP 权重进主模型

`故障` · `active` · `infra_off` · `other_stack` · `verified` · `NEW` · 发现 2026-09-19 · 旧号 12.41

- **原因对象**：权重/检查点 / sglang / load_weights NextN 层跳过守卫
- **原因行为**：load_weights 的 NextN 跳过守卫在原始名上检查 model.layers 前缀，checkpoint 名含 language_model. 中缀使守卫永不触发；名字规范化晚于该判定，MTP 层权重进入主模型 loader。
- **影响对象**：模型权重；可用性/进度（静默错误）
- **影响详情**：同格式 checkpoint（nvidia 官方 NVFP4）加载“成功”、服务正常，主模型多出 MTP 层权重、MTP 权重源错位；混合格式 checkpoint（layer-45 为 MXFP8）则在启动期 shape mismatch 报错。TP=2、2×DGX Spark 复现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/40285

### INF-1158 `sglang_lora_adapter_use_rslora_flag_ignored_alpha_over_r_served_as_scaling` SGLang 忽略 use_rslora，rsLoRA 缩放小 √r 倍

`故障` · `active` · `infra_off` · `other_stack` · `verified` · `var` · 发现 2026-09-23 · 旧号 12.42

- **原因对象**：权重/检查点 / sglang / LoRAAdapter scaling 计算
- **原因行为**：LoRAAdapter 初始化只读 lora_alpha 与 r、不读 use_rslora，恒以 lora_alpha/r 计算 scaling，而非 rsLoRA 的 lora_alpha/√r。
- **影响对象**：模型权重；输出分布（质量退化）
- **影响详情**：use_rslora=true 的 adapter 无错加载、服务正常，但更新被静默削弱 √r 倍（r=64/alpha=128 时 PEFT 为 16.0、SGLang 为 2.0），部署后表现接近基线；r 越大削弱越重。CPU 纯 Python 即可复现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/40835
- **关联**：var:INF-1148

### INF-1159 `sglang_gemma4_31b_attention_k_eq_v_lora_missing_v_proj_load_fails_threefold` SGLang Gemma-4-31B k_eq_v 架构 LoRA 加载三重失败

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-09-23 · 旧号 12.43

- **原因对象**：权重/检查点 / sglang / LoRA 加载器投影集与 buffer 维度
- **原因行为**：LoRA 加载器假设七投影齐备：get_hidden_dim 从顶层 Gemma4Config 读不存在的字段；per-layer buffer 按全局 head 数算维度；vision-tower 张量与语言塔 LoRA 名字碰撞。
- **影响对象**：可用性/进度；模型权重（报错）
- **影响详情**：Gemma-4-31B（10 个 full_attention 层无 v_proj）+ PEFT LoRA 启动即报错；绕过后混合 256/512 head_dim 层 buffer 错位，维度碰巧兼容的层可能静默错服务；k_eq_v 变体的 LoRA 路径整体不可用。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/40966

### INF-1162 `vllm_fast_start_weight_cache_key_hashes_only_safetensors_headers_ipc_cache_serves_wrong_checkpoint` vLLM 权重缓存键只哈希 header，服务错 checkpoint

`故障` · `active` · `infra_off` · `config` · `verified` · `NEW` · 发现 2026-10-09 · 旧号 12.46

- **原因对象**：权重/检查点 / vllm / WeightCacheKey 身份哈希
- **原因行为**：WeightCacheKey.checkpoint 只哈希 shard 文件名与 safetensors header（名/dtype/shape/offset），不含权重数据摘要，同布局的不同 checkpoint 得同一 key。
- **影响对象**：模型权重；输出内容（静默错误）
- **影响详情**：daemon 持 checkpoint A 时以 --load-format ipc_cache 启动 B（如基模与其微调、不同 step），引擎直接服务 A 的张量、从不读 B，API 仍显示 B，双方零警告；e2e 复现中 q_mod 引擎输出了 q 的结果。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/59647

### INF-1163 `vllm_peft_helper_ignores_rank_alpha_patterns_and_unimplemented_features_serves_wrong_adapter` vLLM 忽略 LoRA rank/alpha_pattern 与未实现特性

`故障` · `active` · `infra_off` · `config` · `verified` · `var` · 发现 2026-10-09 · 旧号 12.47

- **原因对象**：权重/检查点 / vllm / PEFTHelper adapter 配置解析
- **原因行为**：PEFTHelper 读取 rank_pattern/alpha_pattern 后丢弃，统一按 lora_alpha/r 缩放；from_dict 只留已知键，只拒 DoRA/modules_to_save，其余未实现特性被忽略。
- **影响对象**：模型权重；输出分布（质量退化）
- **影响详情**：此类 adapter 无错加载但以错误缩放服务：OLMoE-1B-7B（64 experts）held-out PPL 为 PEFT 10.06、vLLM 11.28（base 13.09），丢约 40% 增益；把缩放折叠进 lora_B 后恢复 10.06。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/59799
- **关联**：var:INF-1158

## 数据管线

### INF-0889 `datacollator_zero_based_position_ids` 扁平化 collator 的 0 起位置 id 使 RoBERTa 位置错位

`故障` · `active` · `model_on` · `config` · `documented` · `var` · 发现 2026-07-29 · 旧号 9.6

- **原因对象**：数据管线 / transformers / 扁平化 collator 位置 id
- **原因行为**：DataCollatorWithFlattening 生成 0 起始的 position_ids，而 RoBERTa 家族位置嵌入按 padding_idx 偏移（有效 id 从 2 开始），每个 token 的位置 id 都偏离正确值
- **影响对象**：输入编码；隐状态；输出内容（质量退化）
- **影响详情**：RoBERTa 家族模型仍能产生连贯输出，但准确率/质量静默下降，无 crash、无 NaN，用户难以察觉位置编码出错
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/huggingface/transformers/issues/47496
- **关联**：var:INF-0908

## 优化器

### INF-0176 `ttmetal_clip_grad_norm_undefined_p_decomposition_inf_zero` moreh_clip_grad_norm 对 inf/0 阶范数计算失效

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `var` · 发现 2026-09-06 · 旧号 1.177

- **原因对象**：优化器 / tt-metal / moreh_clip_grad_norm 阶数分解
- **原因行为**：把 norm_type 拆为整数/小数/符号三部分逐元素计算，该分解只对有限非零阶成立；host 侧仅检查 isnan，inf 与 0 穿透，step2 中 1/inf=0 使 x⁰ 全为 1。
- **影响对象**：梯度（静默错误）
- **影响详情**：norm_type∈{inf,−inf,0} 时 total_norm 恒为 1.0 或非有限：默认 error_if_nonfinite=False 下全体梯度被无差别按 max_norm 缩放或乘 NaN，梯度裁剪静默失效；为 True 时对有限梯度误报 norm 非有限。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/55587
- **关联**：var:ttmetal_argmax_comparator_nan_semantics_dtype_split

### INF-0193 `ttmetal_moreh_adam_bf16_hyperparameter_truncation_wrong_optimizer` moreh_adam/adamw 把超参截断到 bf16，实际运行的优化器不同

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-09-09 · 旧号 1.194

- **原因对象**：优化器 / tt-metal / moreh_adam/adamw 超参投递
- **原因行为**：reader 以位截断（非转换）把 lr、beta1、beta2、eps、weight_decay 投递为 bf16 标量；adamw 的 EMA 系数用截断 beta 在设备上算，bias correction 却用宿主精确 beta。
- **影响对象**：优化器状态；模型权重（静默错误）
- **影响详情**：beta2=0.999 实际为 0.99609375，二阶矩视界从 1000 步塌到 256 步；beta2=0.9999 与 0.999 截断后位相同；精确算术 replay 中参数轨迹偏离 torch.optim.AdamW 34–44%，PCC 因尺度不变而察觉不到。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/55935

## 损失计算

### INF-0199 `ttmetal_loss_preallocated_output_repurposed_as_unreduced_intermediate` ttnn.mse_loss/l1_loss 把预分配输出当作未归约中间结果

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-09-11 · 旧号 1.200

- **原因对象**：损失计算 / tt-metal / loss_function 预分配输出处理
- **原因行为**：SUM/MEAN 归约模式下把 optional_output_tensor 直接转发给 ttnn::subtract 作逐元素输出，sum/mean 另分配新 tensor 作返回值，未把归约结果写回预分配输出。
- **影响对象**：训练信号（静默错误）
- **影响详情**：预分配归约后标量形状时 subtract 形状失配直接 crash；预分配全输入形状时调用方 tensor 里是未归约的逐元素平方/绝对差，无警告，读该 tensor 的训练循环拿错误 loss 反传；两形态均确定复现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/56295

## MoE路由

### INF-0045 `ttmetal_moe_routing_remap_precedence_dead_check` MoeRoutingRemap 运算符优先级使校验失效致路由权重错误

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-07 · 旧号 1.45

- **原因对象**：MoE路由 / tt-metal / MoeRoutingRemap 校验与映射
- **原因行为**：validate_on_program_cache_miss 中 TT_FATAL 条件因 C++ 运算符优先级（== 比 ?: 紧）恒为非零 mesh 维度，校验永不触发；三元两臂也颠倒（cluster_axis==0 应映射到行）。
- **影响对象**：数值结果（静默错误）
- **影响详情**：多 device mesh 下错误配置照常运行而不报错，program factory 用截断的 non_zero_per_device 生成静默错误的 per-device 路由权重；无 crash、无 NaN。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/51860

### INF-0246 `ttmetal_moe_expert_token_remap_writer_group_boundary_off_by_one_drops_page_overwrites_next_core` moe_expert_token_remap 组计数双重推进致丢页与越核覆写

`故障` · `active` · `infra_off` · `yes` · `documented` · `NEW` · 发现 2026-09-23 · 旧号 1.247

- **原因对象**：MoE路由 / tt-metal / token_remap writer 组计数
- **原因行为**：writer 内核组计数器体内 reset 为 0 后又被 for 增量表达式推进为 1，首组之后每组只跨 reduction_size−1 页，组边界每次 flush 提前一页漂移。
- **影响对象**：数值结果（静默错误）
- **影响详情**：单 core 拥有多于一个归约组时，每组尾页的专家激活位不进入 reduced 输出，并覆写下一 core 的页；wormhole 128 页/rs=1 错 906/4096；现有测试每 core 单组故恒绿。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/57461

### INF-0324 `vllm_marlin_moe_within_expert_token_order_reduction_tree` vLLM Marlin MoE expert 内 token 序不定致归约树不同

`故障` · `active` · `ambiguous` · `yes` · `documented` · `var` · 发现 2026-08-16 · 旧号 4.8

- **原因对象**：MoE路由 / vllm / moe_align_block_size 内排序
- **原因行为**：moe_align_block_size() 只保证按 expert 分组、不保证 expert 内 token 顺序，Marlin MoE 多 threadblock split-K 因此可选用不同物理块与浮点归约树
- **影响对象**：数值结果（不可复现）
- **影响详情**：无 crash、无 NaN；同 expert 有多个 token 时，同一输入跨运行/跨调度得到小幅不同的 BF16 输出，取决于 token 被排进的块位置
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/52532
- **关联**：var:INF-0317

### INF-0349 `moe_routing_cascade` MoE router top-k 对微小扰动敏感致路由级联漂移

`故障` · `active` · `model_on` · `yes` · `verified` · `NEW` · 发现 待定 · 旧号 5.2

- **原因对象**：MoE路由 / 通用 / router top-k 专家选择
- **原因行为**：top-k 专家选择是不连续的 argmax，pre-router 隐状态上 sub-ULP 级变化即可把 router logit 推过 top-k 边界，改变 token 的专家分派，并沿约 58 个 MoE 层级联放大。
- **影响对象**：输出分布；输出内容（静默错误）
- **影响详情**：相对参考实现 logprob 漂移、偶发生成分叉，在 routing 真正翻转的 token 上放大为阶跃式差异；单机 MoE 即可触发，非 MoE 模型不适用。
- **改进措施**：检测 待定；修复 待定
- **来源**：引用：Fireworks AI blog (2026-03-10), Lesson #2。

### INF-0354 `sglang_eplb_rank_specific_logical_physical_map_corruption` SGLang EPLB 逐 rank 逻辑→物理映射不一致致精度静默下降

`故障` · `active` · `ambiguous` · `no` · `verified` · `NEW` · 发现 2026-08-01 · 旧号 5.7

- **原因对象**：MoE路由 / sglang / EPLB 逻辑→物理 expert 映射
- **原因行为**：无 a2a backend 时各 EP rank 对相同 token 跑 MoE 后求和，映射须各 rank 一致；但 _compute_logical_to_all_physical_map 把候选折叠为离本 rank 最近的副本。
- **影响对象**：隐状态；输出内容（质量退化）
- **影响详情**：默认 --moe-a2a-backend none 开 EPLB 且有冗余 expert 时（如 tp2/ep2、48 个冗余 expert），各 rank 求和了错误 expert 的输出，GSM8K 由约 0.63 跌至 0.42，无报错、无 NaN、无警告。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/pull/32962

### INF-0388 `vllm_async_eplb_rearrangement_step_counter_runs_negative_silent_progress_stop` vLLM async EPLB 事件握手丢失致专家重排静默停止

`故障` · `active` · `infra_off` · `config` · `verified` · `var` · 发现 2026-09-21 · 旧号 5.41

- **原因对象**：MoE路由 / vllm / async EPLB worker 事件握手
- **原因行为**：异步重排 worker 与主循环的 CpuGpuEvent 握手丢失一环：某次 transfer 的 consumed 事件从未置位，worker 永久停在 consumed_event.wait()；主循环重排计数器不复位、持续跑负。
- **影响对象**：性能/资源；指标/日志（性能退化）
- **影响详情**：EP16 跨节点、use_async=True、nixl 下，自 step 1000 起 EPLB 不再重排专家，无 warning/error；日志 steps until the next rearrangement 递减为 -1300、-2000 等负值，负载不均渐进恶化，吞吐/延迟缓退。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/57883
- **关联**：var:INF-0379

### INF-0389 `sglang_minimax_m3_ascend_fuseep_route_missing_redundant_allreduce_shared_expert_misshard` SGLang MiniMax-M3 漏判 FuseEP 致 MoE 输出重复归约

`故障` · `active` · `infra_off` · `config` · `documented` · `var` · 发现 2026-09-22 · 旧号 5.42

- **原因对象**：MoE路由 / sglang / MiniMaxM3MoE 分发与共享专家
- **原因行为**：MiniMaxM3MoE 漏判 is_ascend_fuseep()：FuseEP 已全局 gathered 的输出又做 TP all-reduce；shared expert 被 TP 切分而非复制；ep_size/top_k 未初始化。
- **影响对象**：数值结果；隐状态；输出内容（质量退化）
- **影响详情**：Ascend NPU 上 ascend_fuseep 后端且 tp>1 时，路由激活每层放大 tp_size 倍（TP8 即 ×8），叠加残缺的 shared expert 贡献；无 crash、无 NaN，服务看似正常但输出持续劣化，属性缺失还可能引发 KeyError。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/40735
- **关联**：var:INF-0373

### INF-0409 `sglang_deepep_uninitialized_rows_fp8_gemm_nan` SGLang DeepEP 下 FP8 linear 输入含未初始化行

`故障` · `active` · `infra_off` · `no` · `documented` · `NEW` · 发现 2026-07-31 · 旧号 6.16

- **原因对象**：MoE路由 / sglang / DeepEP 路径的 FP8 linear 输入行
- **原因行为**：使用 --moe-a2a-backend deepep 时，进入 block-FP8 linear 的部分行持有未初始化内存，各 GEMM 后端照常在这些行上计算。
- **影响对象**：数值结果；输出分布（静默错误）
- **影响详情**：Hopper 默认 deep_gemm 把垃圾饱和成有限值，静默产生错误结果、长期未被发现；换 flashinfer 后端则溢出为 NaN，开启 SGLANG_ENABLE_ASYNC_ASSERT 扫描 next_token_logits 时 scheduler 在启动期死亡。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/33106

### INF-0451 `sglang_qwen3moe_mori_double_all_reduce_garbage` SGLang Qwen3-MoE mori 后端双重 all-reduce

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-08 · 旧号 6.58

- **原因对象**：MoE路由 / sglang / qwen3_moe 的 a2a 后端门控
- **原因行为**：is_deepep() 精确匹配 enum，qwen3_moe 漏了其他模型已有的 is_mori() 门控，mori 落入 forward_normal，在已 combine 输出上再做 EP all-reduce。
- **影响对象**：数值结果；输出分布（静默错误）
- **影响详情**：ROCm gfx950 上 Qwen3-MoE 使用 mori a2a 后端时触发；无 crash，输出数值错误，8x MI355X 上 Qwen3-30B-A3B RL 复现中 SGLang rollout 与 Megatron 重算的 log probs 相差约 0.005。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/pull/34006

### INF-0462 `vllm_eplb_e8m0_scale_nccl_unsupported_dtype_init_fail` vLLM EPLB 对 E8M0 scale 做 NCCL 通信致启动失败

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-08-09 · 旧号 6.69

- **原因对象**：MoE路由 / vllm / EPLB set_eplb_state 注册
- **原因行为**：EPLB 按原 dtype 注册 MXFP4/MXFP8 expert 的 E8M0 scale，启动 profiling 时对其做 dummy all-gather，而 NCCL/Gloo/PyNCCL 都不支持该 dtype
- **影响对象**：可用性/进度（报错）
- **影响详情**：启用 EPLB 且 expert 用 E8M0 scale 的多卡部署，启动 profiling 时抛 TypeError（NCCL process group 不支持 Float8_e8m0fnu），engine 初始化失败。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/51568

### INF-0482 `sglang_moe_expert_id_namespace_runner_backend_mismatch_silent_oob` SGLang MoE 误判 runner 致 expert ID 越界写

`故障` · `active` · `infra_off` · `config` · `verified` · `NEW` · 发现 2026-08-13 · 旧号 6.89

- **原因对象**：MoE路由 / sglang / Dispatcher topk_ids 重映射
- **原因行为**：Dispatcher 按配置的 runner backend 决定是否把 topk_ids 映射到 rank 本地范围；ROCm 上 auto 且 AITER 可用时误判为保留全局 ID，实际 Triton runner 随之越界写
- **影响对象**：可用性/进度（报错）
- **影响详情**：ROCm 上 EP>1 的 compressed-tensors W4A16 MoE（如 Kimi-K2.5 EP8）warmup 时每个 rank 出现 GPU Memory access fault；--moe-runner-backend triton 可规避。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/pull/34701

### INF-0626 `vllm_dspark_cuda_graph_padded_rows_routed_by_dsv4_topk_ima` vLLM graph padding 行进入 dsv4_topk 路由致 IMA

`故障` · `active` · `infra_off` · `partial` · `verified` · `NEW` · 发现 2026-09-13 · 旧号 6.233

- **原因对象**：MoE路由 / vllm / dsv4_topk Triton 快速路由路径
- **原因行为**：CUDA graph 固定形状产生的 padding 行未把 is_padding 传进 MoE router，dsv4_topk 快速路径把 padding 行当真实 token 路由，产出的非法 expert id 被下游消费。
- **影响对象**：数值结果；可用性/进度（报错）
- **影响详情**：DeepSeek-V4.1-Flash + DSpark 推测解码 + FULL_DECODE_ONLY CUDA graph，max_num_seqs>256 高并发时出现异步 CUDA IMA，worker 与 EngineCore 死亡；8×H20 TP8/EP8 复现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/56389

### INF-0664 `vllm_cutlass_fp8_moe_padding_routes_minus1_wrong_expert_first_token_offset` vLLM CUTLASS MoE 把 −1 哨兵计入专家 offset

`故障` · `active` · `infra_off` · `config` · `verified` · `var` · 发现 2026-10-09 · 旧号 6.271

- **原因对象**：MoE路由 / vllm / moe_permute 专家 offset 计算
- **原因行为**：默认 VLLM_MOE_SKIP_PADDING=1 下 padding 行被写成 topk_ids=−1，CUTLASS MoE 分组未排除该哨兵，moe_permute 算错 expert_first_token_offset。
- **影响对象**：数值结果；输出内容（静默错误）
- **影响详情**：单请求小批服务时真实 token 被送入错误专家，MoE 输出静默错值；128 专家 top-4 fp8 模型首 token 即坍塌（'Yes' p=0.90→'\n' p=0.997）；批量请求、triton/marlin 后端、enforce-eager、关 CUDA graph 均正确。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/59686
- **关联**：var:vllm_dspark_graph_padded_rows_flow_into_dsv4_topk_routing_ima

### INF-0760 `flashinfer_moe_routing_enum_missing_sqrtsoftplus_wrong_expert_selection_risk` flashinfer fp4 MoE 路由枚举缺 sqrtsoftplus

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-09-13 · 旧号 7.87

- **原因对象**：MoE路由 / flashinfer / fp4 MoE RoutingMethodType
- **原因行为**：trtllm_fp4_block_scale_moe 以封闭的 RoutingMethodType 枚举硬编码路由评分，不含 DeepSeek-V4.1 的 sqrtsoftplus；接口不拒绝不支持的 scoring，按所传枚举执行。
- **影响对象**：输出分布；可用性/进度（报错）
- **影响详情**：SGLang 传 DeepSeekV3 时触发误导性 topkGroup<=4 assert；若用户改传其它 method 绕开，内核按错误评分函数静默为每个 token 选错专家，输出分布漂移且无报错。B200 4×TP4 环境报告。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/flashinfer-ai/flashinfer/issues/5190

### INF-0780 `vllm_grouped_topk_multigroup_kernel_scores_only_first_8_groups` vLLM grouped_topk 多组内核只评分前 8 组致 MoE 选错专家

`故障` · `active` · `infra_off` · `config` · `verified` · `NEW` · 发现 2026-10-09 · 旧号 7.107

- **原因对象**：MoE路由 / vllm / grouped_topk 多组路径内核
- **原因行为**：多组路径八 warp 内核每 warp 只评分一组，未随组数上限 32 扩展，第 9 组起从不被评分；smemGroupScores 仅 8 槽，warp0 按 laneIdx<numGroup 越界读。
- **影响对象**：数值结果；隐状态（静默错误）
- **影响详情**：n_group>8 时 grouped_topk 返回错误专家（144 专家 9 组：期望 [136..143] 实得 [120..127]），9/16/32 组全部失败、4/8 组正确；biased MoE router 默认把 ≤32 组送入该算子；暂无已知发布 checkpoint 触发。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/60752

### INF-1065 `sglang_eplb_dspark_draft_graph_capture_scatter_crash` SGLang EPLB 与 DSPARK 同启时 draft 图捕获崩溃

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-08-17 · 旧号 11.42

- **原因对象**：MoE路由 / sglang / EPLB on_select_experts
- **原因行为**：EPLB 的专家重映射与 draft 模型的 expert 选择在 CUDA graph capture 的静态化假设下不相容，on_select_experts 的 scatter 操作崩溃。
- **影响对象**：可用性/进度（报错）
- **影响详情**：同时启用 --enable-eplb 与 --speculative-algorithm DSPARK 时，draft 模型 CUDA graph capture 阶段崩溃，服务起不来；两特性单独使用均正常。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/34974

### INF-1104 `sglang_eagle_draft_num_token_non_padded_zero_topk_minus1_negative_index_wrap_moe_oob` SGLang EAGLE+EP 计数哨兵致 topk=-1 回绕越界写

`故障` · `active` · `infra_off` · `config` · `verified` · `NEW` · 发现 2026-09-18 · 旧号 11.81

- **原因对象**：MoE路由 / sglang / EP 专家映射与 padding 掩码
- **原因行为**：draft 批 num_tokens 被推为 0 而非 None，padded 掩码把全部 topk id 置 -1；dispatch 在 drop 检查前查 local_expert_mapping，-1 回绕到表尾真实专家致越界写。
- **影响对象**：输出内容（静默错误）
- **影响详情**：EAGLE + 专家并行 + overlap scheduler 高并发长生成时，CUDA graph 下部分在途请求 spec_accept_rate=0 并输出 !!!!! 垃圾至 token 上限，新请求正常；eager 下 cudaErrorIllegalAddress 崩溃。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/40156

### INF-1183 `vllm_rocm_kimi_k3_correction_bias_fp32_downcast_on_load` Kimi-K3 路由偏置在 ROCm 加载时被降为 BF16

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-08-02 · 旧号 13.15

- **原因对象**：MoE路由 / vllm / AMD 路由 correction bias
- **原因行为**：AMD 模型在环境 BF16 dtype 下构造 e_score_correction_bias 参数，标准 weight loader 把 checkpoint 中的 FP32 值拷入 BF16 存储，低位在加载时即永久丢失。
- **影响对象**：模型权重；隐状态（静默错误）
- **影响详情**：ROCm/AMD 上服务 Kimi-K3 时，grouped top-k 的 correction bias 精度永久受损，后续 AITER upcast 无法恢复，MoE 路由静默偏移；无 crash、无 NaN，与 NVIDIA 实现不一致。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/50761

### INF-1184 `hf_qwen35moe_data_dependent_dispatch_breaks_checkpoint_repro` Qwen3.5-MoE 数据依赖 expert 循环致重算图分叉

`故障` · `active` · `infra_off` · `yes` · `verified` · `NEW` · 发现 2026-08-02 · 旧号 13.16

- **原因对象**：MoE路由 / transformers / Qwen3_5MoeExperts 专家循环
- **原因行为**：Qwen3_5MoeExperts.forward 用 for 循环遍历 expert_hit，迭代次数依赖 router top-k；hidden_states 在前向与重算间稍有差异即令重算图分叉，违反 checkpoint 重算契约。
- **影响对象**：梯度；可用性/进度（数值异常）
- **影响详情**：LoRA 微调 Qwen3_5Moe 系模型并启用 gradient_checkpointing_enable() 时，抛 CheckpointError，或产生静默 NaN 梯度，取决于运行条件。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/huggingface/transformers/issues/47467

### INF-1185 `vllm_moe_skip_padding_router_inconsistency` 部分 MoE router 把 padding 行路由到真实 expert

`故障` · `active` · `infra_off` · `partial` · `verified` · `NEW` · 发现 2026-08-02 · 旧号 13.17

- **原因对象**：MoE路由 / vllm / BaseRouter 与各 router 实现
- **原因行为**：padding 仅在部分 fused top-k operator 中处理；DSV4 Triton、grouped top-k、custom 等 router 对 cudagraph/DP padding 行发射真实 expert ID。
- **影响对象**：数值结果；性能/资源（静默错误）
- **影响详情**：默认启用 VLLM_MOE_SKIP_PADDING 时，cudagraph/DP padding 行被当作真实 token 分派到 expert，执行本应跳过的工作，静默污染 expert 计算与 EPLB 映射；无 crash、无 NaN。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/50757

### INF-1201 `hf_switch_transformers_token_priority_cumsum_dim_and_router_tie` SwitchTransformers token 优先级 cumsum 疑错维

`故障` · `active` · `infra_off` · `config` · `speculative` · `NEW` · 发现 2026-08-25 · 旧号 13.33

- **原因对象**：MoE路由 / transformers / router token_priority 累加
- **原因行为**：token_priority 的 torch.cumsum 取 dim=-3，报告者按 expert_index 形状质疑应为 -2；另返回的 router_probs 与 router_logits 数值恒等，两点均待维护者裁定。
- **影响对象**：训练信号；数值结果（静默错误）
- **影响详情**：使用 SwitchTransformers 系 MoE 模型即暴露；若成立则无 crash、无 NaN，token 优先级静默错维，MoE 路由的训练/评估信号失真，下游若消费 router_probs 则分布偏差难以从外部察觉。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/huggingface/transformers/issues/48293

## 重计算/卸载

### INF-0430 `vllm_cpu_offload_mrv2_silent_noop` vLLM V2 runner 下 --cpu-offload-gb 静默不生效

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-08-07 · 旧号 6.37

- **原因对象**：重计算/卸载 / vllm / V2 model runner 权重卸载
- **原因行为**：V2 GPU model runner 激活时从不安装 offloader，--cpu-offload-gb（UVA 权重卸载）及 prefetch offload 选项被静默忽略，也不打印警告。
- **影响对象**：性能/资源；可用性/进度（性能退化）
- **影响详情**：DeepseekV2、Qwen2Moe 等默认走 V2 runner 的架构上设置卸载后，模型仍全量加载到 GPU；模型大于显存时报 OOM，错误信息建议降 --gpu-memory-utilization、增 TP 等无关 remedy，用户无从得知卸载未生效。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/51396

### INF-0472 `vllm_cpu_offload_gb_mrv2_silent_noop` vLLM --cpu-offload-gb 在 MRV2 上静默不生效

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-08-12 · 旧号 6.79

- **原因对象**：重计算/卸载 / vllm / MRV2 的 CPU offloader 注入
- **原因行为**：V2 GPU model runner 的安装路径不包含 CPU offloader 注入逻辑，--cpu-offload-gb（UVA weight offload）及 prefetch offload 选项被静默忽略，且无任何警告
- **影响对象**：性能/资源；可用性/进度（报错）
- **影响详情**：MRV2 下模型被完整加载到 GPU；模型大于显存时报 OOM，且提示降低 --gpu-memory-utilization 或增大 TP 等无关建议，用户无从得知 offload 未生效。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/51396

### INF-0587 `sglang_mapped_layer_courier_missing_record_stream_storage_reuse` SGLang 权重卸载缺 record_stream 致数据竞争

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-09-04 · 旧号 6.194

- **原因对象**：重计算/卸载 / sglang / mapped-weight courier 交接
- **原因行为**：courier 在私有 copy stream 上创建权重张量交给 compute 线程，_collect_mapped_layer() 只等 copy event，未对 compute stream 调用 record_stream。
- **影响对象**：模型权重；输出内容（静默错误）
- **影响详情**：diffusion layerwise-offload mapped-weight 路径中，最后的 Python 引用释放后 allocator 在 copy stream 完成即复用 storage，compute stream 可能仍在读：前向读到别的内容，输出偶发损坏/错值，随负载与时序漂移。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/37963

### INF-0616 `sglang_minimax_h3_fl2va_layerwise_offload_visually_corrupted_video` SGLang MiniMax-H3 layerwise offload 视频花屏

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `var` · 发现 2026-09-09 · 旧号 6.223

- **原因对象**：重计算/卸载 / sglang / layerwise offload（DiT 等层）
- **原因行为**：根因待定位；原文推测为 layerwise offload 下 DiT/text encoder 层在 host↔device 往返路径上的布局、同步或精度错误。
- **影响对象**：输出内容（静默错误）
- **影响详情**：MiniMax-H3 FL2VA 视频生成开启 layerwise offloading（约 24GB 单卡）时，API 返回成功、MP4 合法可播放，但解码出的视频帧严重花屏，只有人看帧才能发现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/38605
- **关联**：var:INF-0586

### INF-0620 `vllm_cpu_offload_gb_lost_across_enginecore_boundary_noopoffloader` vLLM cpu_offload_gb 跨进程丢失退化为 Noop 卸载

`故障` · `active` · `infra_off` · `config` · `documented` · `var` · 发现 2026-09-10 · 旧号 6.227

- **原因对象**：重计算/卸载 / vllm / cpu_offload_gb 跨进程传递
- **原因行为**：vllm_config 经 AsyncLLM spawn EngineCore 的 pickle 跨进程传递后，worker 侧收到 cpu_offload_gb=0；auto 后端把零预算静默映射为 NoopOffloader 而不报错。
- **影响对象**：性能/资源；可用性/进度（报错）
- **影响详情**：--cpu-offload-gb 被接受并回显在参数里，但权重从未卸载：模型大于显存时加载期 OOM；小于显存容差时以 Noop 状态正常服务、offload 整体失效，默认日志无痕迹（仅 DEBUG 级宣告）。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/56283
- **关联**：var:vllm_hidden_states_kv_budget_overestimate_startup_rejection

## RL训练回路

### INF-0385 `vllm_dense_dp_weight_transfer_ipc_payload_selection_dp_rank_reset_zero` vLLM DP 重配置将 rank 置 0 致权重 payload 选错

`故障` · `active` · `infra_off` · `partial` · `verified` · `var` · 发现 2026-09-15 · 旧号 5.38

- **原因对象**：RL训练回路 / vllm / 权重传输 payload 槽位选择
- **原因行为**：worker 按 data_parallel_rank 选本地 payload，但 reconfigure_for_independent_dp_rank() 将其重置为 0，非零 DP replica 均选中 DP0 payload。
- **影响对象**：可用性/进度；模型权重（报错）
- **影响详情**：DP≥2+TP 用 IPC 权重传输时，非零 replica 拿 GPU0/1 的 CUDA IPC 句柄到 GPU2/3 使用，UUID mismatch 使 /update_weights 返回 HTTP 500，服务未崩；若校验放宽则会跨 replica 静默错载权重。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/56949
- **关联**：var:INF-0369

## 观测/评测

### INF-0104 `ttmetal_comp_pcc_constant_fallback_fp32_tolerance_false_reject` comp_pcc 常量张量回退沿用 fp32 容差致 bf16 误判

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-25 · 旧号 1.105

- **原因对象**：观测/评测 / tt-metal / comp_pcc 常量张量 allclose 回退
- **原因行为**：PCC 对常量/单元素张量无定义，comp_pcc 回退到 torch.allclose 时沿用 fp32 级容差（rtol=1e-5、atol=1e-4），未按被比较 dtype 的实际精度（bf16 eps≈7.8e-3）放宽。
- **影响对象**：测试判定（静默错误）
- **影响详情**：常量或单元素 bf16/fp16 张量比较时，2 ULP 的正常 SFPU 误差即被判 PCC 0.0（如 tanh 输入 −0.75：torch −0.63671875 vs 设备 −0.6328125），正确内核被拒、CI 门禁误红。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/pull/54289

### INF-0111 `ttmetal_prefill_reference_bf16_rope_table_pcc_position_decay` prefill 验证参考用 bf16 RoPE 表致 PCC 随位置衰减

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `NEW` · 发现 2026-08-27 · 旧号 1.112

- **原因对象**：观测/评测 / tt-metal / deepseek_v3_d_p 参考 RoPE 表
- **原因行为**：prefill PCC 验证的 torch 参考从 bf16 实例化的参考模型读取 RoPE 表，inv_freq 成为 bf16 buffer（0.75 vs 真值 0.7498942），低位在后续 .float() 前已丢失。
- **影响对象**：测试判定（静默错误）
- **影响详情**：长序列 PCC 检查中 KVPE cache 的 PCC 随位置线性衰减（0.9976@1024 → 0.9462@5120），设备实现可能正确却被判失败、像设备 rope kernel bug；q/k 共享错表使 attention 输出 PCC 仍约 0.99，掩盖来源。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/54516

### INF-0134 `ttmetal_kv_cache_pcc_gate_parametrize_unreachable_never_ran` tt_transformers KV PCC 门恒为 False，从未执行

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `var` · 发现 2026-09-01 · 旧号 1.135

- **原因对象**：观测/评测 / tt-metal / test_model.py cache_pcc 门
- **原因行为**：cache_pcc = layers == 1 and not dummy_weights 在仅有的两个 parametrize 组合下恒为 False，KV PCC 门自 #21306 起从未执行，KV 对照与阈值 dict 均为死代码。
- **影响对象**：测试判定（静默错误）
- **影响详情**：CI 绿灯，但 tt-transformers 的 KV cache 正确性从未被任何门检验，KV 损坏可直接通过 CI；下标 bug 已由 #53816 修复，门的可达性仍未修。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/55064
- **关联**：var:INF-0104

### INF-0179 `ttmetal_is_real_is_imag_goldens_call_nonexistent_torch_api` is_real/is_imag golden 调用不存在 API 致验证失效

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `var` · 发现 2026-09-06 · 旧号 1.180

- **原因对象**：观测/评测 / tt-metal / is_real/imag golden、sweep
- **原因行为**：golden 函数调用不存在的 torch.is_imag/torch.is_real，比较前即抛 AttributeError；sweep 中 is_imag 的参考谓词与 ttnn 语义不符，且输入恒从非零区间采样使谓词恒为假。
- **影响对象**：测试判定（静默错误）
- **影响详情**：golden 对照从未跑通，sweep 以全 false 对全 false 恒通过，两 op 从未被真正验证；op 本身正确（修复后 0/1024 错），但其未来回归不会被 golden/sweep 发现。CI 测试中即触发。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/pull/55577
- **关联**：var:INF-0134

### INF-0231 `ttmetal_disagg_prefill_kv_gate_literal_layer_bound_deep_layers_never_scored` disagg prefill KV PCC 门层数缺省 61，深层从未评分

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `var` · 发现 2026-09-19 · 旧号 1.232

- **原因对象**：观测/评测 / tt-metal / prefill KV gate 层数缺省
- **原因行为**：KV gate 评分循环层数取环境变量 PREFILL_NUM_LAYERS，缺省为 Kimi-K2.7 深度的字面量 61 而非按模型配置解析；各模型 manifest 均未声明该变量，sc4 臂一律按 61 层定界。
- **影响对象**：测试判定（静默错误）
- **影响详情**：GLM-5.2（78 层）等深于 61 层且无 manifest pin 的模型，sc4 腿只评分 0..60 层、从不检查 61..77，深层 KV PCC 回归可全绿通过，缺失层也不报 missing。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/57088
- **关联**：var:INF-0104

### INF-0232 `ttmetal_selu_golden_drops_scale_alpha_kwargs_3416pct_off_pcc_passes` ttnn.selu golden 丢弃 scale/alpha，PCC 门恒过

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `var` · 发现 2026-09-20 · 旧号 1.233

- **原因对象**：观测/评测 / tt-metal / ttnn.selu golden
- **原因行为**：golden 直接调用无 scale/alpha 参数的 torch.nn.functional.selu，调用方传入的 scale/alpha 被 **kwargs 静默吸收，golden 恒计算标准常数 SELU。
- **影响对象**：测试判定（静默错误）
- **影响详情**：golden 对 scale=3.0,alpha=2.0 与 scale=0.5,alpha=0.1 返回逐位相同张量；设备输出与 golden 偏差最高 3416% 时 PCC>0.99 仍恒过，带非默认 scale/alpha 的 sweep 无法发现设备侧回归。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/57116
- **关联**：var:INF-0179

### INF-0233 `ttmetal_reduction_goldens_factory_drops_scalar_kwarg_300pct_pcc_1` 归约 golden 工厂吞掉 scalar 参数，六个 op 验证失效

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `var` · 发现 2026-09-20 · 旧号 1.234

- **原因对象**：观测/评测 / tt-metal / reduction golden 工厂
- **原因行为**：golden 生成工厂签名以 **_ 兜底，把 scalar= kwarg 吸收后从不转发给 torch 函数，golden 计算未缩放归约。
- **影响对象**：测试判定（静默错误）
- **影响详情**：ttnn.sum/mean/max/min/std/var 带 scalar= 调用时，设备缩放结果与 golden 偏差最高 300% 而 PCC 恒 1.0，六个 op 的 scalar 语义完全脱离验证覆盖。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/57115
- **关联**：var:INF-0179

### INF-0238 `ttmetal_grid_sample_golden_ignores_use_precomputed_grid_nan_3x_wide` grid_sample golden 丢弃模式 flag 返回 NaN

`故障` · `active` · `infra_off` · `hw_specific` · `verified` · `var` · 发现 2026-09-21 · 旧号 1.239

- **原因对象**：观测/评测 / tt-metal / grid_sample golden
- **原因行为**：golden 只实现标准 grid 模式，use_precomputed_grid 落入 **_ 被丢弃，无条件按 last_dim//2 解析 grid，把 int16 位型当归一化坐标使用。
- **影响对象**：测试判定（数值异常）
- **影响详情**：use_precomputed_grid=True 时 golden 无报错地返回 NaN 与无关有限值、形状 3× 过宽（(1,2,9,2) 应为 (1,2,3,2)），该文档化路径的参考验证从未真实存在，比较模式恒假阳性或恒炸。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/tenstorrent/tt-metal/issues/57149
- **关联**：var:INF-0232

### INF-0347 `flashinfer_rope_fp8_e5m2_reference_clamps_at_e4m3_max_448` flashinfer RoPE FP8 参考对 e5m2 误按 448 截断

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-10-12 · 旧号 4.31

- **原因对象**：观测/评测 / flashinfer / rope FP8 trace 参考 _q()
- **原因行为**：参考实现 _q() 的 clamp 常量硬编码为 e4m3 上限 ±448，未按 quantize_dtype 分派，请求 e5m2 时同样截断；rope 正确性测试从不传 quantize_dtype。
- **影响对象**：测试判定；数值结果（静默错误）
- **影响详情**：rope_quantize_fp8 用 float8_e5m2 时，参考侧把超过 448 的量级截到端点，e5m2 范围（±57344）被静默压缩 128×，与正确饱和的 CUDA 内核分歧，以 trace 为 golden 的校验失真。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/flashinfer-ai/flashinfer/issues/6173

### INF-0555 `vllm_liveness_monitor_spurious_sentinel_sigterm_healthy_engine` vLLM liveness monitor 误判 sentinel 杀死健康引擎

`故障` · `active` · `infra_off` · `待定` · `documented` · `NEW` · 发现 2026-08-29 · 旧号 6.162

- **原因对象**：观测/评测 / vllm / liveness monitor 判死逻辑
- **原因行为**：monitor_engine_liveness() 把进程 sentinel 就绪直接当作 EngineCore 死亡证据，不复核 exitcode 或心跳；进程实际存活时 sentinel 伪就绪即触发 shutdown()。
- **影响对象**：可用性/进度（报错）
- **影响详情**：V1 单机多进程、sm_110 类平台上间歇、时序敏感：健康引擎被 SIGTERM，甚至在请求成功完成后被杀，后续请求报 EngineDeadError，日志与真实崩溃无法区分；0.26.1rc1 dev1261 与 dev1177 均复现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/54184

### INF-0640 `vllm_skipped_prefix_lookup_requests_counted_full_external_cache_miss_metrics_dilution` vLLM 跳过前缀查找的请求被计为外部前缀缓存满额 miss

`故障` · `active` · `infra_off` · `config` · `verified` · `NEW` · 发现 2026-09-19 · 旧号 6.247

- **原因对象**：观测/评测 / vllm / 外部前缀缓存命中统计
- **原因行为**：调度器已对 skip_reading_prefix_cache 请求跳过 KV connector 命中，但外部缓存统计仍按 num_tokens 次查询、0 命中入账，与本地前缀缓存排除此类请求的规则相反。
- **影响对象**：指标/日志（静默错误）
- **影响详情**：含 prompt_logprobs 或 pooling 类请求的负载下，vllm:external_prefix_cache_queries/hits 与 External prefix cache hit rate 日志被系统性稀释（复现中降为 0），与并排本地命中率口径不一致；推理输出正确。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/57596

### INF-0732 `sglang_bench_one_batch_moe_backend_flag_silent_bf16_fallback_misbenchmark` SGLang bench_one_batch 忽略 MoE 后端参数

`故障` · `active` · `infra_off` · `other_stack` · `verified` · `NEW` · 发现 2026-08-27 · 旧号 7.59

- **原因对象**：观测/评测 / sglang / bench_one_batch 初始化流程
- **原因行为**：bench_one_batch 直接构造 ModelRunner，从不调用仅 scheduler 执行的 initialize_moe_config，runner backend 懒解析为 AUTO；日志仍打印请求的 backend。
- **影响对象**：指标/日志；性能/资源（静默错误）
- **影响详情**：用 bench_one_batch 测量化 MoE（如 Kimi-K3 mxfp4）时 --moe-runner-backend 被静默忽略，mxfp4 权重加载期物化为 BF16、专家走 triton fused_moe_kernel；基准数字实为 BF16 fallback。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/36395

### INF-0922 `trtllm_eval_thinking_budget_key_silently_ignored` trtllm-eval 默认 thinking_budget 键被忽略致分数减半

`故障` · `active` · `model_on` · `other_stack` · `documented` · `NEW` · 发现 2026-07-30 · 旧号 10.6

- **原因对象**：观测/评测 / tensorrt-llm / trtllm-eval aime 默认模板参数
- **原因行为**：aime25/aime26 子命令默认传 thinking_budget=32768，DeepSeek-V4 renderer 只读 thinking/enable_thinking，未知键被静默忽略，thinking 回退为 False。
- **影响对象**：输入编码；指标/日志（静默错误）
- **影响详情**：用 trtllm-eval aime25/aime26 评测 DeepSeek-V4 时触发；benchmark 以非 thinking 模式运行，报告的准确率约为真实 thinking 模式的一半，且无任何警告。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/NVIDIA/TensorRT-LLM/issues/17020

### INF-0923 `trtllm_eval_max_output_length_silently_discarded` trtllm-eval 生成长度参数被任务 yaml 静默覆盖

`故障` · `active` · `model_on` · `other_stack` · `verified` · `var` · 发现 2026-08-03 · 旧号 10.7

- **原因对象**：观测/评测 / tensorrt-llm / _get_sampling_params 映射
- **原因行为**：任务 yaml 定义 max_gen_toks 时，params_mapping 在生成前无条件覆盖 CLI 写入的 max_tokens；保护开关 preserve_caller_max_tokens 藏在文本子命令无法满足的两道门后。
- **影响对象**：输出内容；指标/日志（静默错误）
- **影响详情**：thinking 模型评测且任务 yaml 定义 max_gen_toks 时触发；输出被 yaml 上限截断，传更大的 --max_output_length 完全无效，无 crash、无警告，基准分数静默偏低。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/NVIDIA/TensorRT-LLM/issues/17022
- **关联**：var:INF-0922

## 配置/参数

### INF-0412 `vllm_attention_backend_envvar_silently_ignored` vLLM V1 静默忽略 VLLM_ATTENTION_BACKEND 环境变量

`故障` · `active` · `infra_off` · `config` · `verified` · `NEW` · 发现 2026-08-01 · 旧号 6.19

- **原因对象**：配置/参数 / vllm / VLLM_ATTENTION_BACKEND 读取
- **原因行为**：vLLM 0.26.0 V1 引擎源码中不存在 VLLM_ATTENTION_BACKEND 字面量，没有任何代码读取该环境变量，唯一有效途径是 attention_backend 关键字参数，设置该变量既不生效也不告警。
- **影响对象**：性能/资源（静默错误）
- **影响详情**：启动前设置该环境变量后引擎静默回退到默认后端选择，无警告或错误；按教程、博客旧指引操作的用户得到非预期的注意力后端（复现环境：单卡 Ampere sm_86、WSL2）。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/50292

### INF-0431 `vllm_yaml_falsey_silently_dropped` vLLM YAML 配置中的 bool false 被静默丢弃

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-08-07 · 旧号 6.38

- **原因对象**：配置/参数 / vllm / FlexibleArgumentParser
- **原因行为**：FlexibleArgumentParser 把 --config YAML 转为 argv 时只在值为真才追加 --key，bool False 被直接丢弃，而不是转成显式的关闭信号。
- **影响对象**：性能/资源（静默错误）
- **影响详情**：用 --config 加载 YAML 时，对默认为 None、后由其他逻辑决定的三态 bool flag，用户写的 false 被当作未设置，静默回退到真实默认值；无 crash、无警告（v0.25.1）。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/51401

### INF-0495 `vllm_lazydict_setitem_cache_invalidation_missing` vLLM LazyDict 覆写不失效缓存致先读后写返回旧值

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-08-16 · 旧号 6.102

- **原因对象**：配置/参数 / vllm / LazyDict.__setitem__
- **原因行为**：LazyDict 首次执行 factory 后缓存其值，但 __setitem__ 只替换 factory、不清除已缓存值，被覆写前已读过的 key 此后一直返回旧值，新 factory 永不执行。
- **影响对象**：待定（静默错误）
- **影响详情**：任何先读后写 LazyDict 的路径（配置/元数据对象）都会触发：无 crash、无报错，服务以陈旧配置运行且外部无法区分。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/pull/52541

### INF-0600 `sglang_legacy_sgl_env_rewrite_after_deprecation_registry_drop` SGLang 旧 SGL_ 变量重写晚于 deprecation 转发

`故障` · `active` · `infra_off` · `other_stack` · `verified` · `var` · 发现 2026-09-06 · 旧号 6.207

- **原因对象**：配置/参数 / sglang / _handle_deprecated_envs
- **原因行为**：_handle_deprecated_envs() 先做 deprecation registry 转发、后做 SGL_→SGLANG_ 重写；重写名本身已 deprecated 时错过 registry 轮次，链式转发永不发生。
- **影响对象**：性能/资源；指标/日志（静默错误）
- **影响详情**：设置 SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK 这类其重写名亦已 deprecated 的旧变量时，开关静默失效（TP 内存不平衡检查关不掉），无 crash，唯一线索是一条指向另一个 deprecated 名的告警。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/pull/38219
- **关联**：var:INF-0557

### INF-1126 `hf_missing_local_config_silent_default_init` transformers 本地目录缺 config.json 时静默回退默认配置

`故障` · `active` · `infra_off` · `yes` · `documented` · `var` · 发现 2026-08-02 · 旧号 12.10

- **原因对象**：配置/参数 / transformers / configuration_utils 配置加载
- **原因行为**：本地目录缺 config.json 时，缺失文件被映射为 None 再转成 {}，from_pretrained() 用 from_dict({}) 构建全默认配置，且无任何警告。
- **影响对象**：模型权重；输出内容（静默错误）
- **影响详情**：无 crash、无 NaN、无 warning；模型以类默认的 hidden size/层数/vocab size 等错误架构参数实例化，错误很晚才显现，后续加载/推理行为与真实模型不符。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/huggingface/transformers/pull/47712
- **关联**：var:INF-1118

## 构建/发布

### INF-0210 `triton_380_release_wheel_missing_thirteen_main_correctness_fixes` Triton 3.8.0 wheel 缺 main 上 13 个正确性修复

`故障` · `active` · `infra_off` · `partial` · `verified` · `var` · 发现 2026-09-13 · 旧号 1.211

- **原因对象**：构建/发布 / triton / release/3.8.x 分支
- **原因行为**：release/3.8.x 分支自 merge-base 后未回吸 main 上的正确性修复，PyPI 3.8.0 wheel 仍是含已知缺陷的旧代码。
- **影响对象**：数值结果；可用性/进度（静默错误）
- **影响详情**：3.8.0 wheel 用户在 sm_120 mxfp4 tl.dot_scaled、flatten range、tl.softmax dim≠−1 等公开 API 上静默拿到错值，另有 TMA 未对齐 stage 粘性设备故障与若干崩溃；用 main 测试逐条跑 wheel 即可暴露。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/triton-lang/triton/issues/11735
- **关联**：var:flashinfer_vendored_trtllm_fp8_cast_bf16_double_round_fix_not_propagated

### INF-0413 `vllm_xpu_gdn_attention_silent_memory_corruption_under_load` vLLM XPU 依赖锁定旧版内核致 GDN 注意力负载下静默内存损坏

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-08-01 · 旧号 6.20

- **原因对象**：构建/发布 / vllm / xpu.txt 锁定的 GDN 内核版本
- **原因行为**：GDN 注意力在负载下静默损坏内存的修复已合入 vllm-xpu-kernels，但 vLLM 的 requirements/xpu.txt 仍锁定早于该修复的版本，开箱即用的 XPU 安装仍带该缺陷。
- **影响对象**：输出内容；可用性/进度（静默错误）
- **影响详情**：Intel XPU（PyTorch 2.12.0+xpu、XPU runtime 20250）按 vLLM 默认依赖安装后，负载下走 GDN 注意力路径即触发，表现为渐进的输出退化或偶发崩溃。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/49924

### INF-0496 `vllm_precompiled_cuda_variant_failopen_fallback` vLLM 预编译 wheel 变体缺失时未校验 CUDA 兼容即回退

`故障` · `active` · `infra_off` · `config` · `documented` · `NEW` · 发现 2026-08-16 · 旧号 6.103

- **原因对象**：构建/发布 / vllm / precompiled wheel 变体回退
- **原因行为**：VLLM_USE_PRECOMPILED 选定的 CUDA wheel 变体元数据不可用时，回退到 root/default wheel 元数据，不经任何 CUDA 兼容性校验就把所选变体换成默认变体（fail-open）。
- **影响对象**：可用性/进度（报错）
- **影响详情**：如 PyTorch CUDA 12.9 选 cu129 但变体元数据缺失时，安装期无报错地装上 CUDA 变体不匹配的二进制，故障推迟到运行时以随机 crash 或内核错误浮现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/52544

### INF-0549 `sglang_uv_resolver_silent_old_version_regression_resurrection` uv<0.12 静默装旧版 sglang 复活 TP all-reduce 缺陷

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-08-27 · 旧号 6.156

- **原因对象**：构建/发布 / sglang / 旧版 down_proj 缺 all-reduce
- **原因行为**：uv<0.12 不遵循传递依赖的预发布版约束，而 sglang≥0.5.10 强制依赖仅有预发布版的 flash-attn-4，解析器静默回溯到 0.5.9，该版缺 TP>1 丢 all-reduce 的修复。
- **影响对象**：输出内容；隐状态（静默错误）
- **影响详情**：uv pip install sglang（uv<0.12）无错误无警告地装出 6 个月前的 0.5.9；TP>1 下输出流畅但错误（如数学回答错误），无 crash、无 NaN，版本号只出现在启动日志。uv pip compile 单行即可复现解析行为。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/35912

### INF-0740 `sglang_dsv4_flash_hybrid_quant_mxfp4_release_not_carried_assert` SGLang 发布版加载 DSV4-Flash 混合量化检查点断言失败

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `var` · 发现 2026-09-01 · 旧号 7.67

- **原因对象**：构建/发布 / sglang / flashinfer_mxfp4 MoE 加载路径
- **原因行为**：tag 版 flashinfer_mxfp4 路径未处理 routed experts 为 I8 + UE8M0 scale、而 config 只声明 fp8 块量化的布局，hidden-size 断言失败；修复在 main 但未进发布版。
- **影响对象**：可用性/进度（报错）
- **影响详情**：SGLang v0.5.17/v0.5.18 加载 DeepSeek-V4-Flash-0731 混合量化检查点并用 flashinfer_mxfp4 runner，启动即报 Hidden size mismatch 断言；同 checkpoint 同硬件在 main（2026-08-30）正常。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/37342
- **关联**：var:sglang_dsv4_ue8m0_default_moe_backend_distribution_distortion

## 模型自身

### INF-1211 `no_literal_match_collapse` 无字面重叠时长上下文检索 attention 塌缩（NoLiMa）

`故障` · `active` · `model_on` · `not_target` · `verified` · `var` · 发现 待定 · 旧号 14.1

- **原因对象**：模型自身 / 模型 / 长上下文非字面检索 attention
- **原因行为**：query 与 gold needle 无词面重叠时 QK 相似度矩阵退化为低秩，softmax 无法在 gold 位置锐化，gold 的 attention share 随上下文变长塌缩；属能力上限，实现均健康。
- **影响对象**：输出内容（质量退化）
- **影响详情**：在标准字面 NIAH 上表现完美的模型，在语义等价但非字面的检索任务上于 2K–32K 中等长度灾难性失败；隐藏态健康，激活范数离群检测与 statistical、cheap 族均漏检，需 attention 谱探针加长度扫描。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://arxiv.org/abs/2502.05167
- **关联**：var:INF-1218

### INF-1212 `lost_in_the_middle` Lost in the middle：长上下文中部信息检索失效

`故障` · `active` · `model_on` · `not_target` · `documented` · `var` · 发现 待定 · 旧号 14.2

- **原因对象**：模型自身 / 模型 / softmax 注意力质量摊薄
- **原因行为**：softmax 归一化下总注意力质量守恒，被越来越多的位置瓜分：按位置呈 U 形摊薄（首尾强、中部弱），全局 attention 熵随长度趋向 log N。
- **影响对象**：输出内容（质量退化）
- **影响详情**：长上下文检索呈 primacy/recency 偏置：放在上下文开头或结尾的信息能被取到，放在中部的取不到；attention 熵随长度单调增长。
- **改进措施**：检测 待定；修复 待定
- **来源**：引用：Liu et al. TACL 2024；
- **关联**：var:INF-1218

### INF-1213 `multi_hop_reasoning_cliff` 长上下文多跳/聚合推理先于单点检索崩溃（RULER）

`故障` · `active` · `model_on` · `not_target` · `verified` · `var` · 发现 待定 · 旧号 14.3

- **原因对象**：模型自身 / 模型 / 长上下文多步 attention 路由
- **原因行为**：聚合、变量追踪、多跳任务需要跨层、跨位置的多步 attention 路由，这类路由在长上下文下先于单 needle 检索崩溃；属模型能力上限。
- **影响对象**：输出内容（质量退化）
- **影响详情**：NIAH 在 32K 仍通过时有效上下文已塌缩到 4–8K：聚合/变量追踪任务出现陡峭准确率悬崖，模型自信给出错误计数或丢失中间状态；激活范数检测器与单 pass 统计量均不响应。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://arxiv.org/abs/2404.06654
- **关联**：var:INF-1218

### INF-1214 `padding_reasoning_anticorrelation` padding 拉长输入致推理下降而 perplexity 不变差

`故障` · `active` · `model_on` · `not_target` · `documented` · `var` · 发现 待定 · 旧号 14.4

- **原因对象**：模型自身 / 模型 / 长输入下的推理能力
- **原因行为**：仅用无信息 padding 把输入拉长（推理所需信息量不变）时模型推理准确率下降，而下一 token perplexity 与推理表现反相关；属模型能力边界。
- **影响对象**：输出内容；指标/日志（静默错误）
- **影响详情**：padding 拉长输入后 perplexity 看上去更好或至少不变差，推理任务却失败：基于 perplexity / 平均 log-prob 的健康检查绿灯而下游答案错误，且给出与真实退化反向的信号。
- **改进措施**：检测 待定；修复 待定
- **来源**：引用：Levy et al. ACL 2024。
- **关联**：var:INF-1218

### INF-1215 `retrieval_vs_reasoning_conflict` 上下文与参数化记忆冲突时模型忽略上下文作答

`故障` · `active` · `model_on` · `not_target` · `documented` · `var` · 发现 待定 · 旧号 14.5

- **原因对象**：模型自身 / 模型 / 上下文与参数化记忆的取舍
- **原因行为**：上下文提供的事实与模型参数化记忆冲突时，模型倾向忽略上下文、按参数化记忆作答；属知识冲突场景下的能力上限，不涉及权重或 kernel 损坏。
- **影响对象**：输出内容（静默错误）
- **影响详情**：RAG/上下文注入场景下，模型给出与所提供文档相矛盾、来自预训练记忆的答案（原文由机制推得，未单列行为）；与幻觉条目重叠，具体可检测度量原文未给出。
- **改进措施**：检测 待定；修复 待定
- **来源**：引用：EMNLP 2024 Knowledge Conflicts survey。
- **关联**：var:INF-1228

### INF-1217 `video_world_model_temporal_rope_horizon_addressing_failure` 视频世界模型超出训练 horizon 后 KV 视觉记忆寻址失效

`故障` · `active` · `model_on` · `not_target` · `documented` · `NEW` · 发现 2026-08-22 · 旧号 14.7

- **原因对象**：模型自身 / 模型 / 视频 world model KV 视觉记忆
- **原因行为**：rollout 超出训练 horizon 后 temporal RoPE 偏移超出训练所见范围，注意力无法取回相关视觉信息；在 RoPE 旋转后空间朴素压缩 cache 会把不同相位的 key 平均而损坏记忆；属表示能力限制。
- **影响对象**：输出内容；KV状态（质量退化）
- **影响详情**：交互式视频 world model 长 rollout 超出训练 horizon 后，视觉一致性与上下文依赖静默退化，模型“忘记”早期帧内容；无 crash，权重与 kernel 均健康。
- **改进措施**：检测 待定；修复 待定
- **来源**：http://arxiv.org/abs/2608.07408

### INF-1218 `long_context` 长上下文下模型能力衰退（提示集标签型样本）

`故障` · `active` · `model_on` · `not_target` · `documented` · `NEW` · 发现 待定 · 旧号 14.8

- **原因对象**：模型自身 / 模型 / 长上下文信息整合能力
- **原因行为**：连贯长上下文使模型在流形上走到能力边界，token 表示的有效维度塌缩（attention 稀释）；故障由提示集选择实现，与等长填充/重复上下文对照，非模型变更。
- **影响对象**：输出内容（质量退化）
- **影响详情**：长上下文下检索/多跳推理准确率下降，答案倾向复述而非整合，模型无任何异常信号；effective_rank 显著响应、trajectory 呈 GRADUAL 形状，nan_inf_flag 等 cheap 族与单 token geometric 距离漏检。
- **改进措施**：检测 待定；修复 待定
- **来源**：引用：本项目早期实测条目（原型阶段实测，量值不跨栈继承；

### INF-1219 `confabulation_high_entropy` 知识缺失时语义层高熵的虚构答案（confabulation）

`故障` · `active` · `model_on` · `not_target` · `verified` · `var` · 发现 待定 · 旧号 15.1

- **原因对象**：模型自身 / 模型 / 缺乏知识时的答案语义分布
- **原因行为**：模型对所问内容确实缺乏信息，逐 token 概率并不低，但答案语义层的分布平坦、高熵，采样输出对随机种子敏感、内容任意。
- **影响对象**：输出内容（静默错误）
- **影响详情**：给出流畅、看似合理但错误的答案，同一问题重采样得到互相矛盾的答案；单 token 置信度发现不了，需跨样本 semantic entropy、self-consistency 或 EigenScore，cheap 族仅有较钝信号。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://www.nature.com/articles/s41586-024-07421-0
- **关联**：var:INF-1228

### INF-1220 `overconfident_systematic_hallucination` 过度自信的系统性幻觉：答错且输出熵低

`故障` · `active` · `model_on` · `not_target` · `verified` · `var` · 发现 待定 · 旧号 15.2

- **原因对象**：模型自身 / 模型 / RLHF 置信校准与知识截止事实
- **原因行为**：合并两个子故障：RLHF 诱导的过度自信使模型给错误答案赋予高概率；按知识截止时的旧事实毫不犹豫作答。共同点是在某主题上一贯地错，输出熵低而答案错。
- **影响对象**：输出内容；输出分布（静默错误）
- **影响详情**：自信陈述谬误，重采样不改变答案；lexical / predictive entropy 都很低，朴素熵检测器直接漏检，statistical 与 cheap 族系统性漏检，需 truth-direction 线性探针或 EigenScore 聚类。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://arxiv.org/abs/2207.05221；https://arxiv.org/abs/2505.01997；https://arxiv.org/abs/2402.13213
- **关联**：var:INF-1228

### INF-1221 `over_refusal_exaggerated_safety` 安全对齐过度致良性请求被拒答

`故障` · `active` · `model_on` · `not_target` · `verified` · `NEW` · 发现 待定 · 旧号 15.3

- **原因对象**：模型自身 / 模型 / 安全对齐 refusal 电路
- **原因行为**：安全对齐过度激活，对表面上像不安全、实际良性的请求给出拒答；该电路干净、单一、可隔离。
- **影响对象**：输出内容（质量退化）
- **影响详情**：对良性请求返回拒答模板（XSTest、OR-Bench 类评测可测）；隐藏态在健康流形内，Mahalanobis 离群、激活范数与 cheap 族均无信号，中层线性探针有望区分。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://arxiv.org/abs/2308.01263；https://arxiv.org/abs/2405.20947

### INF-1222 `instruction_constraint_drift` 指令约束被忽略或在长生成中逐步漂移丢失

`故障` · `active` · `model_on` · `not_target` · `verified` · `NEW` · 发现 待定 · 旧号 15.4

- **原因对象**：模型自身 / 模型 / 指令约束的表示保持
- **原因行为**：模型未遵守显式指令约束（长度、措辞、格式、否定）：或一开始就忽略，或约束表示随层数/生成步数在长 CoT 中逐渐衰减；权重与 kernel 均健康。
- **影响对象**：输出内容（质量退化）
- **影响详情**：输出长度/格式错误、指定措辞缺失、否定式要求被无视、约束在生成中途被放弃；单层快照的 geometric 族看不到，需跨层轨迹探针追踪约束 token 影响力，cheap 族无信号。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://arxiv.org/abs/2512.14754；https://arxiv.org/abs/2510.15211；https://arxiv.org/abs/2505.11423

### INF-1223 `early_layer_knowledge_gap_dola` 事实性集中于后层，输出权重前移致事实错误增加（DoLa）

`故障` · `active` · `model_on` · `not_target` · `verified` · `NEW` · 发现 待定 · 旧号 15.5

- **原因对象**：模型自身 / 模型 / 早层/早退出层的输出分布
- **原因行为**：事实性知识集中在后层，早层/过早退出层承载的分布更弱、更不正确；early-exit、layer-mixing、推测解码等把有效输出权重前移的扰动会在不离开流形的前提下降低事实性。
- **影响对象**：输出内容；输出分布（质量退化）
- **影响详情**：有效输出分布被拉向早层时事实性错误增加；预测答错样本上早层到最终层的 KL 显著更高，可由逐层 logit-lens 一致性或早层→最终层 KL 测量，离流形几何检测无效。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://arxiv.org/abs/2309.03883

### INF-1224 `sycophancy` 用户表露立场后模型改口附和（sycophancy）

`故障` · `active` · `model_on` · `not_target` · `verified` · `NEW` · 发现 待定 · 旧号 15.6

- **原因对象**：模型自身 / 模型 / 偏好条件化电路
- **原因行为**：用户表露的立场激活模型的偏好条件化电路，模型输出迎合用户偏好/偏见的答案，而不是中立提问下会给的答案。
- **影响对象**：输出内容（静默错误）
- **影响详情**：同一事实问题，在用户先表明立场后模型改口附和该立场（原文由机制推得）；谄媚与中立回答在中层隐藏空间沿偏好方向可分，离流形几何量、statistical 与 cheap 族均无信号。
- **改进措施**：检测 待定；修复 待定
- **来源**：引用：Sharma et al. ICLR 2024。

### INF-1225 `intrinsic_extrinsic_faithfulness` 输出相对源材料不忠实：内在矛盾与外在虚构

`故障` · `active` · `model_on` · `not_target` · `documented` · `var` · 发现 待定 · 旧号 15.7

- **原因对象**：模型自身 / 模型 / 对 grounding/source 的忠实性
- **原因行为**：合并 copy/paraphrase/binding 三类子故障：输出相对给定 grounding/source 不忠实，intrinsic 与源材料矛盾，extrinsic 引入源中没有的内容；含复制错实体、属性绑到错误对象。
- **影响对象**：输出内容（静默错误）
- **影响详情**：摘要/RAG 输出与所给文档相矛盾，或凭空添加文档中不存在的细节；intrinsic 支可由 NLI/entailment 判据检出，extrinsic 支由跨样本 semantic entropy 检出，离流形几何检测在两支上均无效。
- **改进措施**：检测 待定；修复 待定
- **来源**：引用：Maynez ACL 2020；
- **关联**：var:INF-1228

### INF-1226 `detector_validated_internal_signals` 已验证的幻觉内部信号检测器基线（EigenScore、DoLa、线性探针）

`结论` · `active` · `model_on` · `not_target` · `verified` · `rationale` · 发现 待定 · 旧号 15.8

- **原因对象**：模型自身 / 模型 / 幻觉/长文本实体捏造
- **原因行为**：不适用；本行不产生行为效应，仅记录针对幻觉的已验证隐藏态／内部签名检测器。
- **影响对象**：输出内容（静默错误）
- **影响详情**：幻觉评测条件下出现；这类故障在内部信号留有痕迹，可由 EigenScore（隐藏态协方差谱分散度）、DoLa（早层与末层 logit 散度）、逐 token 线性探针读出
- **改进措施**：检测 待定；修复 待定
- **来源**：https://arxiv.org/abs/2309.03883

### INF-1228 `hallucination` 幻觉标签：模型计算全对但回答错误（难度匹配对照）

`故障` · `active` · `model_on` · `not_target` · `documented` · `NEW` · 发现 待定 · 旧号 15.10

- **原因对象**：模型自身 / 模型 / 幻觉（答错样本的前向）
- **原因行为**：模型计算本身无误，只是内容错误；故障完全由评分侧标签定义（incorrect 为故障、correct 为健康），用难度匹配提示靠采样温度让同一模型在对错间翻转
- **影响对象**：输出内容（静默错误）
- **影响详情**：生成流利、自信但错误的回答，无任何 infra 层面异常：重算一致、无 NaN、batch 不变
- **改进措施**：检测 待定；修复 待定
- **来源**：引用：本项目早期实测条目（原型阶段实测，量值不跨栈继承；

## 外部攻击

### INF-0294 `rowhammer_gddr6` rowhammer/HBM 备用行耗尽致相邻权重行多 bit 翻转

`故障` · `active` · `infra_off` · `no` · `verified` · `var` · 发现 待定 · 旧号 2.1

- **原因对象**：外部攻击 / 硬件 / GDDR6/HBM 显存相邻权重行
- **原因行为**：以极高频率反复锤击 GDDR6 某一行，经电磁耦合翻转相邻行中的 bit（GPUHammer 攻击）；HBM 备用行/row-remap 池耗尽（Xid 63，磨损老化）产生同形态的连续权重行损坏。
- **影响对象**：模型权重；输出内容（质量退化）
- **影响详情**：某个 projection 若干相邻权重行上多个 bit 被持久翻转，在所有 token 上恒定存在；A6000 上跨 4 个 bank 翻转多达 8 bit，被针对的 ML 权重准确率最高降 80%；轻则静默细微偏置，命中指数位时出现 NaN。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://www.usenix.org/conference/usenixsecurity25/presentation/lin-shaopeng；https://arxiv.org/abs/2507.08166；https://arxiv.org/abs/2503.11901
- **关联**：var:INF-0315

### INF-0297 `prisonbreak_bitflip_jailbreak` PrisonBreak：翻转少量权重 bit 解除对齐 LLM 的拒答

`故障` · `active` · `ambiguous` · `partial` · `verified` · `NEW` · 发现 待定 · 旧号 3.1

- **原因对象**：外部攻击 / 通用 / safety-critical 权重 bit
- **原因行为**：攻击者用基于梯度的 profiling 离线定位 safety-critical 权重，只翻转其中约 5–25 个 bit（Rowhammer 可行范围内）以解除拒答，所需 flip 数比先前攻击少至多 40x。
- **影响对象**：输出内容；模型权重（静默错误）
- **影响详情**：绝大多数输入上输出良性、与健康模型无异，通用能力基本保留，常规质量回归测不出；只有收到有害请求时模型不再拒答而是照做。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://arxiv.org/abs/2412.07192；https://arxiv.org/abs/2507.03236

### INF-0298 `bithydra_eos_suppression` BitHydra：翻转 EOS 相关权重位使生成永不终止

`故障` · `active` · `infra_off` · `yes` · `verified` · `NEW` · 发现 待定 · 旧号 3.2

- **原因对象**：外部攻击 / 通用 / EOS embedding/lm_head 行
- **原因行为**：攻击者用 ADMM 选出 bit 位置，在 EOS token 的 embedding 行及与之 tied 的 lm_head 行上翻转 3–30 个 bit，以最小化 P(<EOS>)。
- **影响对象**：输出内容；输出分布（静默错误）
- **影响详情**：少至 3 个 bit 即可让约 100% 的 prompt 一直生成到 max output length，语义常仍连贯、看似只是话多；中间层隐状态基本不变，异常只体现在最终层 EOS-logit 时间序列与生成长度上。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://arxiv.org/abs/2505.16670

### INF-0299 `tbt_targeted_bit_trojan` TBT/ProFlip：少量 bit 翻转植入休眠后门

`故障` · `active` · `ambiguous` · `yes` · `verified` · `NEW` · 发现 待定 · 旧号 3.3

- **原因对象**：外部攻击 / 通用 / 少量权重 bit（休眠后门）
- **原因行为**：攻击者翻转少量权重 bit 植入仅响应特定 trigger 的休眠后门（TBT）；ProFlip 渐进搜索并翻转 bit，使攻击落在 Rowhammer 可行范围内。
- **影响对象**：输出内容；模型权重（静默错误）
- **影响详情**：良性输入上准确率正常、指标无退化；只有携带 trigger 的输入被导向攻击者预设的 target completion。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_ProFlip_Targeted_Trojan_Attack_With_Progressive_Bit_Flips_ICCV_2021_paper.pdf

### INF-0300 `code_level_opcode_flip_frameflip` FrameFlip：翻转 GEMM 库分支指令致全局破坏

`故障` · `active` · `infra_off` · `no` · `verified` · `NEW` · 发现 待定 · 旧号 3.4

- **原因对象**：外部攻击 / 通用 / BLAS/GEMM 库条件分支指令
- **原因行为**：攻击者经 DRAM Rowhammer 打代码页，翻转编译后 BLAS/GEMM 库中一条条件分支指令的 1 bit，使分支判定反转，GEMM 每次调用都走错误路径。
- **影响对象**：数值结果；输出内容（静默错误）
- **影响详情**：确定性、与模型无关：使用该库的任何模型对所有输入都输出固定/错误结果；逐样本尖峰检测器可能不触发，纯 PyTorch forward-hook 层面无法注入。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://www.usenix.org/conference/usenixsecurity24/presentation/li-shaofeng

### INF-0301 `bias_param_bit_flip` bias 参数 bit 翻转使 feature map 整体平移致定向误分类

`故障` · `active` · `ambiguous` · `yes` · `verified` · `NEW` · 发现 待定 · 旧号 3.5

- **原因对象**：外部攻击 / 通用 / bias 参数 bit
- **原因行为**：攻击者翻转 bias 参数（而非权重矩阵）中的 bit，bias 偏移与输入无关地作用于所有位置，使整张 feature map 均匀平移。
- **影响对象**：模型权重；隐状态（静默错误）
- **影响详情**：少量甚至单个 bias bit 翻转即导致一致的误分类/输出偏移，单 bit 破坏力远高于随机权重 flip；激活 channel 在所有位置恒定偏移，逐位置尖峰检测会漏检。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://dl.acm.org/doi/10.5555/3199700.3199718

### INF-0302 `bfa_progressive_sign_exponent` BFA 渐进位搜索与 sign/exponent bit 翻转

`故障` · `active` · `infra_off` · `partial` · `verified` · `var` · 发现 待定 · 旧号 3.6

- **原因对象**：外部攻击 / 通用 / 权重 sign/exponent bit
- **原因行为**：攻击者用 Progressive Bit Search 按梯度逐步挑选破坏力最大的权重 bit；翻 sign bit 获得最大单次杠杆，翻 exponent bit 可由单个 bit 产生 inf/nan。
- **影响对象**：模型权重；隐状态（数值异常）
- **影响详情**：效应随所翻 bit 类别构成连续谱：exponent bit/naninf 端为数值爆炸与 NaN 传播，mantissa 低位端仅细微数值畸变，sign bit 端幅值不一定大但方向翻转。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://arxiv.org/abs/2502.07408
- **关联**：var:INF-0315

### INF-0303 `bfa_decision_level_cognitive_hijacking` CogBias 决策级劫持：bit 翻转在目标话题上操纵 LLM 立场

`故障` · `active` · `ambiguous` · `partial` · `documented` · `var` · 发现 2026-07-30 · 旧号 3.7

- **原因对象**：外部攻击 / 通用 / 目标话题相关的关键权重 bit
- **原因行为**：攻击者以可微情感评估器为优化信号、多目标 loss 联合约束，用 BitScout 定位关键 bit，在超稀疏预算下翻转少量权重 bit。
- **影响对象**：输出内容；模型权重（静默错误）
- **影响详情**：目标话题上立场稳定偏移（如推荐偏向、事实判断倾斜），非目标任务与整体输出分布几乎不受影响，无 crash、无 NaN；在 Llama-3.2-3B、Mistral-7B、Qwen2.5-14B 上验证。
- **改进措施**：检测 待定；修复 待定
- **来源**：http://arxiv.org/abs/2607.25227v1
- **关联**：var:INF-0302

### INF-0305 `tfl_targeted_bit_flip_attack_llm` TFL：梯度定位高敏感 bit 定向操纵 LLM 输出

`故障` · `active` · `infra_off` · `partial` · `documented` · `var` · 发现 2026-08-04 · 旧号 3.9

- **原因对象**：外部攻击 / 通用 / LLM 权重中的高敏感 bit
- **原因行为**：攻击者经梯度引导定位 LLM 权重中的高敏感 bit，在个位数 bit 的翻转预算下仅翻转其极性，不修改权重数值结构。
- **影响对象**：输出内容；模型权重（静默错误）
- **影响详情**：无 crash、无 NaN；特定输入上输出被定向操纵为攻击者指定响应，非目标输入功能与 perplexity 保持不变。
- **改进措施**：检测 待定；修复 待定
- **来源**：http://arxiv.org/abs/2602.17837v1
- **关联**：var:INF-0299

### INF-0306 `bfa_cognitive_bias_injection_cogbias` CogBias：bit 翻转向 LLM 注入认知偏见

`故障` · `active` · `ambiguous` · `partial` · `documented` · `NEW` · 发现 2026-08-12 · 旧号 3.10

- **原因对象**：外部攻击 / 通用 / 目标话题相关的关键权重 bit
- **原因行为**：攻击者用可微 sentiment evaluator 作优化信号、多目标 loss 约束，BitScout 定位关键 bit，在超稀疏预算下翻转少量权重 bit 注入认知偏见。
- **影响对象**：输出内容；模型权重（静默错误）
- **影响详情**：目标话题上认知立场系统性显著偏移，非目标任务功能正常；不触发安全拒绝、不改变输出结构，常规安全/质量回归测不出，需立场/情感监控才能发现。
- **改进措施**：检测 待定；修复 待定
- **来源**：http://arxiv.org/abs/2607.25227v1

### INF-0308 `rotated_robustness_training_free_bfa_defense` Rotated Robustness：权重正交旋转的免训练 BFA 防御

`方法` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-08-17 · 旧号 3.12

- **原因对象**：外部攻击 / 通用 / LLM 权重矩阵中的敏感位
- **原因行为**：BFA 或随机位翻转改写 LLM 权重位；位位置敏感度不均（存在敏感位热点），翻转少数敏感位即可造成灾难性单点失效
- **影响对象**：模型权重；输出内容（静默错误）
- **影响详情**：未防护时单个敏感位翻转可致灾难性单点失效；对权重施加正交旋转后扰动能量被均匀摊薄，权重损坏降级为均匀小扰动
- **改进措施**：检测 待定；修复 待定
- **来源**：https://arxiv.org/abs/2603.16382

### INF-0309 `targeted_bfa_llm_agent_decision_surface` 定向 BFA 翻转 LLM Agent 决策相关权重位

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-08-17 · 旧号 3.13

- **原因对象**：外部攻击 / 通用 / Agent 工具选择/终止决策相关权重位
- **原因行为**：攻击者搜索并翻转少量权重位，使 Agent 在工具调用选择、任务终止判断等离散决策点上翻转，而不攻击通用能力
- **影响对象**：模型权重；输出内容（静默错误）
- **影响详情**：无 crash、无 NaN，通用 benchmark 分数几乎不变；仅在 agent 工作流中出现错误工具调用或不终止，任务成功率被定向破坏，通用能力监控看不见，需任务级行为监控
- **改进措施**：检测 待定；修复 待定
- **来源**：https://arxiv.org/abs/2603.10042

### INF-0311 `gguf_single_bit_semantic_failure_bitsifter` .gguf 量化权重单比特翻转定向诱发语义级失败（BitSifter）

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-08-20 · 旧号 3.15

- **原因对象**：外部攻击 / 通用 / .gguf 量化权重的语义敏感位
- **原因行为**：攻击者经 Rowhammer/BFA 翻转 .gguf 量化权重中的单个语义敏感位；脆弱位集中于特定 tensor，可用敏感熵模型与 BitSifter 高效定位
- **影响对象**：输出内容；模型权重（静默错误）
- **影响详情**：无 crash、无 NaN，模型正常服务但被定向诱发事实错误、推理退化或有害内容；与普通幻觉/能力退化不可区分，诊断需权重位级比对
- **改进措施**：检测 待定；修复 待定
- **来源**：https://arxiv.org/abs/2510.00490

### INF-0312 `moe_bitflip_denial_of_wallet_groundhog_gbfa` Groundhog：MoE 路由权重位翻转致 token 用量暴涨

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-08-31 · 旧号 3.16

- **原因对象**：外部攻击 / 通用 / MoE routing 层终止相关权重位
- **原因行为**：攻击者定位 routing 层中与 EOS 等终止 token 激活关联的权重位并翻转，等效停用这些 expert，使终止信号无法正常汇聚
- **影响对象**：模型权重；输出内容；性能/资源（性能退化）
- **影响详情**：平均停用不到 4 个 expert 即令四个 MoE LLM 输出 token 用量平均膨胀 5912%，多数样本打到 max tokens，语义大体保留；无 crash、无 NaN，仅 token 计费/长度分布能暴露
- **改进措施**：检测 待定；修复 待定
- **来源**：https://arxiv.org/abs/2608.25276

### INF-0313 `rowhammer_hardware_aware_backdoor_robbin` ROBBIN：硬件感知的 Rowhammer 推理期后门注入

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-08-31 · 旧号 3.17

- **原因对象**：外部攻击 / 通用 / DRAM 中权重页及其物理映射
- **原因行为**：攻击者先刻画目标 DRAM 的可利用翻转位，再迭代安排权重的 DRAM 页映射，使 Rowhammer 翻转（含附带翻转）与算法后门对齐
- **影响对象**：模型权重；输出内容（静默错误）
- **影响详情**：无 crash；触发输入上后门行为稳定生效，正常输入精度保留，跨三种 DRAM 设备保持鲁棒（评估于 ResNet-20/VGG-16 + CIFAR-10）
- **改进措施**：检测 待定；修复 待定
- **来源**：https://arxiv.org/abs/2608.23774

### INF-0314 `jitterflip_host_jit_control_plane_bitflip_serving` JITterFlip：翻转宿主侧 JIT 服务控制平面的位翻转攻击

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-09-01 · 旧号 3.18

- **原因对象**：外部攻击 / 通用 / 宿主 JIT artifact 选择/编排决策
- **原因行为**：攻击者经 Rowhammer 翻转宿主 CPU 内存中 JIT 编译服务的控制平面决策状态，使请求走慢路径或被喂错 artifact，不接触权重也不改计算指令
- **影响对象**：性能/资源；输出内容（性能退化）
- **影响详情**：单一位翻转可产生两种形态：输出正确但延迟放大 2.03×–181.90×（sponge），或 gibberish 输出（PPL 比率 15.45×–2.48×10⁶×）；跨 4 个 LLM 可迁移、绕过既有 BFA 防御，sponge 仅延迟/能耗可见
- **改进措施**：检测 待定；修复 待定
- **来源**：https://arxiv.org/abs/2608.29745

### INF-0363 `p2p_distributed_inference_malicious_node_output_tampering` P2P 分布式推理中恶意节点篡改层输出致结果静默腐蚀

`故障` · `active` · `ambiguous` · `other_stack` · `documented` · `NEW` · 发现 2026-08-15 · 旧号 5.16

- **原因对象**：外部攻击 / 通用 / P2P 推理中恶意节点负责的层输出
- **原因行为**：P2P 分布式推理把 LLM 各层分摊到多方消费级硬件，任一参与方都可蓄意篡改自己负责层的输出。
- **影响对象**：输出内容；隐状态（静默错误）
- **影响详情**：被篡改层的输出向后传播为语义损坏的最终回答，调用方无任何错误信号；出现在不可信节点池上的 P2P/去中心化推理研究系统，中心化 vLLM 部署不直接触发。
- **改进措施**：检测 待定；修复 待定
- **来源**：http://arxiv.org/abs/2607.19490；https://arxiv.org/abs/2607.28884

### INF-0828 `multi_agent_kv_cache_hidden_state_integrity_channel` 多智能体间直传 KV-cache 隐藏状态可被篡改且文本审计不可见

`故障` · `active` · `ambiguous` · `other_stack` · `documented` · `NEW` · 发现 2026-08-15 · 旧号 8.42

- **原因对象**：外部攻击 / 通用 / 跨 agent 直传的 KV-cache 状态
- **原因行为**：多智能体协作中 agent 在发送短可见消息的同时把完整 KV-cache 隐藏状态传给另一模型，该隐藏状态可被上游篡改，接收模型直接基于被污染的 KV 推理，文本层审计覆盖不到该通道。
- **影响对象**：KV状态；输出内容（静默错误）
- **影响详情**：多 agent QA 设置下，被篡改的 KV 隐藏状态让最终模型系统性说谎或答错，而可见消息看似无害，对消息文本的一切审查都检不出；单体 serving 不直接触发。
- **改进措施**：检测 待定；修复 待定
- **来源**：http://arxiv.org/abs/2606.28958

## 待定

### INF-0336 `sglang_sm120_decode_head_pad_removal_reproducibly_changes_greedy_output_unexplained` SGLang SM120 去 q 头填充后 greedy 输出不明改变

`故障` · `active` · `model_on` · `hw_specific` · `speculative` · `NEW` · 发现 2026-09-13 · 旧号 4.20

- **原因对象**：待定 / sglang / decode q 头填充相关路径（未定位）
- **原因行为**：未定位。移除 decode 阶段 q 无条件 pad 到 64 头后，某处未知执行路径选择随之改变（候选：autotune 状态、kernel 变体选择、q buffer 分配/初始化）；注意力内核本身已被排除。
- **影响对象**：输出内容（静默错误）
- **影响详情**：4×GB10 TP4 上 DeepSeek-V4 greedy（temperature=0）输出在移除 pad 补丁前后可复现地分歧（未打补丁双 boot 4/4 一致，打补丁 1/4 一致），而内核隔离 A/B bit-identical、kernel inventory 0/299 差异。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/39235

### INF-0544 `sglang_dflash2_concurrent_cross_request_context_bleed` SGLang DFlash2 高并发下整段 prompt 跨请求串扰

`故障` · `active` · `infra_off` · `other_stack` · `speculative` · `NEW` · 发现 2026-08-26 · 旧号 6.151

- **原因对象**：待定 / sglang / 待定
- **原因行为**：待定：原文列出 batch 内 KV/radix cache 键碰撞、DFlash2 draft 池状态跨请求复用、调度器 batch 组装错位三种候选机制，信息不足以裁定根因。
- **影响对象**：跨请求/跨样本数据；输出内容（静默错误）
- **影响详情**：SGLang cookbook DFlash2 + Qwen3.8-27B、RTX Pro 6000 高并发下间歇出现：编码工作流中模型突然回答并发的 GPQA 题目，整段 prompt 归属错位而非逐 token 乱码；模型在思维链中察觉收到不属于本任务的问题但仍作答，无 crash。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/sgl-project/sglang/issues/36548

### INF-0608 `vllm_dsv4flash_deep_context_exact_retrieval_mod4_prompt_length_wrong_token` DeepSeek-V4-Flash 长度≡3 (mod 4) 时深上下文检索错

`故障` · `active` · `infra_off` · `config` · `verified` · `NEW` · 发现 2026-09-08 · 旧号 6.215

- **原因对象**：待定 / 待定 / 待定
- **原因行为**：待定（跨栈复现已排除权重与单栈归因，根因未定位）。
- **影响对象**：输出内容（静默错误）
- **影响详情**：reasoning 关闭时 ~57k prompt 中 ~28k 处的 needle 在 prompt 长度 ≡3 (mod 4) 时确定性检索错末位（8417→8411）；reasoning 开启、needle 靠前或后有 padding 则对；vLLM、SGLang 与两家托管商同错。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/55927

### INF-0628 `vllm_glm53_flash_accumulated_decode_reasoning_long_session_repeat_collapse` GLM-5.3-Flash 长会话累计 decode 深度后重复崩塌

`故障` · `active` · `infra_off` · `other_stack` · `documented` · `NEW` · 发现 2026-09-14 · 旧号 6.235

- **原因对象**：待定 / 待定 / 待定
- **原因行为**：根因未定位；实验收窄到跨请求累计 decode/reasoning token 深度这一变量（仅 prefill 预热不触发），原文候选含 KV/RoPE 慢漂移、采样状态、W4A16 激活漂移或模型固有行为。
- **影响对象**：输出内容（质量退化）
- **影响详情**：长 reasoning 多 agent 会话运行数分钟至数十分钟后突然塌为重复低熵输出（`= = = =`、`.0.0.0.`），卡在 reasoning 内不再发 tool call；顺序重放 76 个成功请求后第 77 个复现，单独冷重放健康，SGLang 同病。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/56868

### INF-0630 `vllm_glm53_nvfp4_gb10_concurrent_fragment_token_state_poisoning_fffd_cascade` vLLM GLM-5.3 nvfp4 并发下非 ASCII 输出被毒化

`故障` · `active` · `infra_off` · `partial` · `verified` · `NEW` · 发现 2026-09-15 · 旧号 6.237

- **原因对象**：待定 / vllm / 待定
- **原因行为**：根因未定位；并发 batching 使非 ASCII 高密度位置的采样分布被 byte-fragment token 系统性霸占，fragment 进入上下文后模型以超高置信级联放大；唯一可用后端无法 A/B。
- **影响对象**：输出分布；输出内容（静默错误）
- **影响详情**：2×GB10 TP2 并发 4 路 emoji/非 ASCII 密集请求时出现 U+FFFD 与错误 CJK 字形（8–80/500 字符），损坏位 top-3 全为 byte-fragment；顺序请求与 ASCII-only 生成不受影响，无 NaN、无日志。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/57087

### INF-0672 `vllm_sm120_apertus_70b_fp8_v0_25_1_output_degeneration_rollback_recovers` vLLM 0.25.1 SM120 上 Apertus-70B FP8 输出退化

`故障` · `active` · `infra_off` · `hw_specific` · `documented` · `NEW` · 发现 2026-10-12 · 旧号 6.279

- **原因对象**：待定 / vllm / 0.22.1→0.25.1 间回归（未定位）
- **原因行为**：vLLM 0.22.1→0.25.1 之间引入的缺陷，回滚二分证据完整，具体内核未定位（栈含 CutlassFP8ScaledMM、FlashInfer 注意力与采样、Inductor、CUDA graph）。
- **影响对象**：输出内容（质量退化）
- **影响详情**：SM120（RTX PRO 6000 Blackwell）+ Apertus-70B FP8-dynamic + vLLM 0.25.1 下，temp=0 非流式请求即输出不连贯重复文本直到 length 上限（HTTP 200）；同 checkpoint 同硬件 v0.22.1 完全正常。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/61100

### INF-0772 `vllm_glm_mxfp4_dep8_ep8_gb200_gsm8k_zero_nightly_0914_regression` vLLM GLM MXFP4 DP8+EP 服务 GSM8K 精度塌至 0

`故障` · `active` · `ambiguous` · `config` · `documented` · `NEW` · 发现 2026-09-25 · 旧号 7.99

- **原因对象**：待定 / vllm / 待定
- **原因行为**：原因未定位：nightly 2026-09-14 之后回归，症状与 MXFP4 量化错值、分层 KV、DP 聚合或 reasoning parser 分流均相容，尚未分离。
- **影响对象**：输出内容（质量退化）
- **影响详情**：8×GB200 DP8+EP 服务 GLM-5.3-MXFP4 时，GSM8K 1319 请求中 1178 个 content 为 null（答案在 reasoning_content），2638 响应槽中 2356 为空串，非空响应退化重复；GLM5.2-mxfp4-fp8 同样出现。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/58729

### INF-0961 `vllm_qwen38_flash_next_thai_orthography_corruption_llamacpp_clean` vLLM 上 Qwen3.8-Flash-Next 泰文正字法损坏

`故障` · `active` · `infra_off` · `partial` · `documented` · `NEW` · 发现 2026-09-01 · 旧号 10.45

- **原因对象**：待定 / vllm / 待定
- **原因行为**：原文未定位缺陷机制（tokenizer 往返、采样、内核数值均未排除）；已排除 persistent_topk 非确定性与 QSA block 选择，llama.cpp 同预算对照把范围压缩到 vLLM 侧。
- **影响对象**：输出内容（质量退化）
- **影响详情**：GB10 sm_121 TP2、FP8 下泰文输出出现声调标记错位、combining mark 双写、字符替换、sara am 丢失，约 12–18 处/1000 泰文字符，5 种 vLLM 配置均复现，损坏率随 prompt 长度变化；llama.cpp 同预算 130 样本零缺陷。
- **改进措施**：检测 待定；修复 待定
- **来源**：https://github.com/vllm-project/vllm/issues/54739
