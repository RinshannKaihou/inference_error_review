# LLM 推理故障每日扫描报告 — 2026-08-05

- **扫描日期**：2026-08-05（覆盖 2026-08-04 ~ 2026-08-05 的 7 天滚动窗口，主查 24h 增量）
- **数据源**：GitHub Search API（6 个目标仓库 + 4 组关键词，共 10 次查询）+ arXiv API（6 组关键词）
- **主文档基准**：扫描前 319 条（第 0 层 17 + 第 1 层 188 + 第 2 层 114）

---

## 一、GitHub Issues / PRs 发现

### 1.1 vLLM（vllm-project/vllm）

#### `[新签名]` #51125 — Non-gated MoE w13 shard count hardcoded to 2 → uninitialized memory → NaN contamination of expert weights
- **链接**：https://github.com/vllm-project/vllm/pull/51125
- **日期**：2026-08-05 open
- **简述**：非门控 MoE（`is_act_and_mul=False`，如 NemotronH 的 `relu2_no_mul`）只把 up projection 融进 `w13`，所以 `w13` 只持有 **单个** `intermediate_size_per_partition` shard 而非两个 gate/up shard。但 **14 种 MoE 量化方法把 shard 数硬编码为 2**：在线路径 `reload/meta.py` 用 `torch.empty_strided` 物化 → 第二 shard 槽是**未初始化内存**，随后流入 block/per-tensor amax、量化权重和 GEMM（`0 × NaN = NaN`）；per-block scale/bias 也被 over-allocate；zero-padding 步进用错误的半计算边界 → 未清零的 pad 行**污染真实行的共享 per-block 权重 scale**。
- **归类**：量化（第 7 类）。属「MoE w13 shard count 错误 → 未初始化内存 → 静默 NaN 污染」新子签名。**判定：新子签名，新增**。

#### `[已知类别·新实例]` #51187 — `VLLM_BATCH_INVARIANT=1` still returns non-identical logprobs across repeats of byte-identical concurrent workload (~44 co-resident sequences)
- **链接**：https://github.com/vllm-project/vllm/issues/51187
- **日期**：2026-08-05 open
- **简述**：开启 `VLLM_BATCH_INVARIANT=1` 后，对**同一未变服务进程**重复字节相同的并发负载（44–45 序程共驻）仍产生**不同 logprobs**（5 次重复产出 3 种不同输出）；唯一改变结果的旋钮是 `--max-num-seqs=8`（小批则干净）。差异是 later-vs-later（非首批 warmup），max logprob delta 4.69e-02，**token id 不变**只 logprob 动。即「batch-invariant 模式在 ~44 序程并发下静默失效」。
- **归类**：数值/精度（第 4 类），确定性。与 4.1 `fp_reduction_order_nondeterminism`（batch-invariance / split-K 非确定性）同族——4.1 已含「batch-invariant split-K matmul」「vLLM batched-kernel nondeterminism」。本条是**该家族在高并发序程下的新实例**，揭示 batch-invariant 模式并非充分保证。**判定：已知类别新实例，作为 4.1 的补充来源引用**（不新增独立条目，因机制已被 4.1 覆盖；但记录为该条目的并发失效新观测）。

#### `[新签名]` #51198 — `--enable-prefix-caching` is a silent 0%-hit no-op on Mamba-2/GDN hybrids (Qwen3_5MoeForConditionalGeneration)
- **链接**：https://github.com/vllm-project/vllm/issues/51198
- **日期**：2026-08-05 open
- **简述**：在 `Qwen/Qwen3.6-35B-A3B`（arch `Qwen3_5MoeForConditionalGeneration`，Mamba-2/GDN 混合）上，`--enable-prefix-caching` 对共享相同长前缀的请求给出 **0% 命中率**。原因是 recurrent conv+SSM 状态不可被 APC（attention-block prefix cache）追踪，所以该 flag 是一个**静默 no-op**——调用方期望共享前缀 prefill 复用却拿不到。无 warning，无 error。
- **归类**：服务栈/缓存（第 6 类）。属「prefix caching 对混合 SSM 架构静默 no-op」新子签名——功能静默失效（控制平面），非数据损坏，但造成"启用即应有效"的承诺被静默违背。**判定：新子签名，新增**。

#### `[新签名]` #51203 — MiniMax-M3 FP8 query allocation not stable across CUDA graph capture/replay → eager CUTLASS path receives `query_fp8=None` and asserts
- **链接**：https://github.com/vllm-project/vllm/pull/51203
- **日期**：2026-08-05 open
- **简述**：MiniMax-M3 在 eager CUDA-graph break 外分配 FP8 query buffer，但分配条件依赖在 eager attention 段内重新评估的请求 metadata。一个无 CUTLASS MSA plan 时捕获的 graph，后来在 replay 时可能已有合法 plan → eager CUTLASS 路径收到 `query_fp8=None` 并在 `assert query_fp8 is not None` 失败。修复：当配置支持 CUTLASS decode 时总是分配 query_fp8（capture/replay 间静态）。
- **归类**：服务栈/缓存（第 6 类）或量化（第 7 类）。属「CUDA graph capture/replay 间 FP8 query 分配条件依赖动态 metadata → replay 时 None」新子签名。与 K8.7 `cudagraph_padded_batch_stale_block_table`、6.x CUDA-graph replay 静态缓冲问题同域但对象是 **FP8 query 分配条件**而非 block-table/slot。**判定：新子签名，新增**（偏 crash，但揭示 CUDA-graph capture 与动态 backend 选择的不变量破裂）。

### 1.2 SGLang（sgl-project/sglang）

#### `[新签名]` #33726 — Qwen3-VL DeepStack inputs silently dropped during breakable CUDA graph (BCG) replay → silent correctness regression
- **链接**：https://github.com/sgl-project/sglang/pull/33726
- **日期**：2026-08-05 open
- **简述**：Qwen3-VL 通过 `input_deepstack_embeds` 把中间视觉特征传到选定 LM 层。但 Prefill breakable CUDA graph (BCG) replay **未把该请求专属张量复制进稳定的 graph input slot** → 正常 BCG 执行表现得像**显式 zero-DeepStack 运行**而非 eager 执行。请求仍成功完成（**静默正确性问题**）。A/B：修复前 `bcg_normal` vs `eager_normal` 仅前 7 token 相同、max abs logprob diff 1.154；修复后 15 token 相同、diff 0.071。修复前 `bcg_normal` ≡ `bcg_zero_deepstack`（diff 0.000）。
- **归类**：服务栈/缓存（第 6 类）。属「breakable CUDA graph replay 丢弃请求专属中间张量 → 静默正确性退化」新子签名。与 K8.7 / 6.x CUDA-graph 静态缓冲问题同族，但对象是**多模态中间特征（DeepStack embeds）**而非 block-table/mamba track。**判定：新子签名，新增**。

### 1.3 Tenstorrent tt-metal

#### `[新签名]` #52177 — FPU reduce path truncates source registers to 10-bit mantissa → greedy top-k tie-break index reduce returns garbage → wrong argmax
- **链接**：https://github.com/tenstorrent/tt-metal/pull/52177
- **日期**：2026-08-05 closed
- **简述**：`_adjust_values_for_tiebreak` 用 `ttnn.min(masked_idx_float32, dim=3)` 找并列最大值中的最低 global index。但 `ttnn.min` 对 FLOAT32 张量走 **FPU reduce 路径**（只有 int32 和 opt-in accurate fp32 mean 走 SFPU）。FPU 把源寄存器截断到 **10-bit mantissa**，所以 100k 量级的词表 index 回来被舍入到一个**不等于任何实际 index 的值** → 后续 `eq()` 匹配不到任何东西 → winner mask 全零 → argmax 退回 array position 0 而非最低全局 index 候选。修复：tie-break 的 index reduce 改用 int32。
- **归类**：硬件/SDC（第 1 类）或解码/采样（第 11 类）。属「FPU reduce mantissa 截断 → 索引比较静默失败 → 错误 argmax」新子签名——数值精度 bug 导致采样/top-k 选择错误。根因在硬件 reduce 路径的精度限制。**判定：新子签名，新增到第 1 类**（硬件数值正确性，因根因是 FPU mantissa 截断）。

#### `[新签名]` #52252 / #52256 — Blackhole destination-reuse synchronization race → nondeterministic FP32 Welford layernorm output (WATCHER-off timing-sensitive)
- **链接**：https://github.com/tenstorrent/tt-metal/issues/52252 · https://github.com/tenstorrent/tt-metal/pull/52256
- **日期**：2026-08-05 open（closes #46523）
- **简述**：Blackhole 的 destination reuse 用 dummy unpack 操作设置 source-valid 并清除 retained source banks，但**不等待对应的 destination-to-source move**。在背靠背工作负载下，dummy unpack 可能在 math 正把 DEST 移入该 bank 时清除 SrcA/SrcB → **时序敏感的、非确定性 FP32 Welford layernorm 输出**（WATCHER 禁用时出现，启用时消失——经典竞态特征）。修复：在 destination-reuse dummy unpack 上设 `Stall_Clr_Cntrl`。
- **归类**：硬件/SDC（第 1 类）。属「硬件同步竞态 → 非确定性 layernorm 输出」新子签名——时序敏感的静默数值错误。**判定：新子签名，新增**。

#### `[记录·非主条目]` #52172 — SFPU edge cases: unary sweep fed all ops `uniform(0.1, 1.1)`, masking negative-branch defects
- **链接**：https://github.com/tenstorrent/tt-metal/pull/52172
- **日期**：2026-08-05 open
- **简述**：SFPU 测试 sweep 之前给所有 op 喂正域 `uniform(0.1, 1.1)`，导致 31 个 op 的 `x<0` 分支从未被执行；`UnarySFPUGolden` 从未量化 Bfp8_b 输入；`exp` 在 ~8 之后系统性过冲 ~5.7%。这揭示**测试覆盖盲区**而非单一推理故障。
- **归类**：硬件/SDC（第 1 类）测试基础设施。**判定：不新增为主条目**（是测试覆盖改进，揭示潜在缺陷但非已发推理故障）。记录为风险监控。

### 1.4 NVIDIA TensorRT-LLM

- **#17306**（beam search + C++ KVCacheManagerV2）：feature/fix PR，添加 beam search 支持而非修复已发静默 bug。**不新增**。
- **#17305**（reasoning mode from rendered prompt）：parser 逻辑修复，非数值正确性。**不新增**。
- **#17321**（MiniMaxVLLayerNorm meta-init）：meta-init crash 修复，非静默正确性。**不新增**。

### 1.5 Triton / Transformers

- 本周期 triton-lang/triton correctness 查询返回 0 条；huggingface/transformers wrong inference 查询返回 0 条。**无新发现**。

---

## 二、arXiv 论文发现

| # | 标题 | arXiv ID | 日期 | 状态 |
|---|------|----------|------|------|
| A1 | Don't Regenerate, Debug: A Domain-Specific Agent for Repairing Near-Miss Hardware Operators | 2608.02712 | 2026-08-03 | **新**：修复 near-miss 硬件算子的 agent 方法论，非单一故障 |
| A2 | LLM-PRISM: Characterizing Silent Data Corruption from Permanent GPU Faults in LLM Training | 2604.10390 | 2026-04-12 | **新**：永久 GPU 故障在 LLM **训练**中产生的 SDC 特征刻画 |
| A3 | How a Bit Becomes a Story: Semantic Steering via Differentiable Fault Injection | 2512.14715 | 2025-12-09 | **新**：可微故障注入实现语义引导，BFA 家族新角度 |
| A4 | When Latent Agents Lie: KV-Cache Integrity in Multi-Agent LLM Collaboration | 2606.28958 | 2026-06-27 | **新**：多智能体协作中的 KV-cache 完整性攻击 |
| A5 | Decision-Level Hijacking (Bit-Flip) | 2607.25227 | 2026-07-28 | **已在主文档** 3.7 |
| A6 | Bit-Flip Vulnerability of Shared KV-Cache Blocks | 2604.17249 | 2026-04-19 | **已在主文档** K4.9 |
| A7 | TFL: Targeted Bit-Flip Attack on LLM | 2602.17837 | 2026-02-19 | **已在主文档** 3.9（2026-08-04 合并） |
| A8 | Rotated Robustness (defense) | 2603.16382 | 2026-03-17 | 防御论文，非故障。**不新增** |
| A9 | Continuous Discovery of Vulnerabilities in LLM Serving Systems with Fuzzing | 2605.11202 | 2026-05-11 | 方法论，非单一故障。**不新增** |
| A10 | Hollow-LLM Attack (ZK verification) | 2607.28884 | 2026-07-30 | ZK 验证层，偏离推理执行核心范围。**不新增**（已在 08-04 评估） |

### arXiv 查重 / 评估结果

- **A1 (2608.02712)**：agent 修复 near-miss 算子的方法论。非单一故障模式。**作为第 1 类补充来源引用，不单独新增**。
- **A2 (2604.10390 LLM-PRISM)**：刻画永久 GPU 故障在 LLM **训练**中的 SDC 特征。训练而非推理，但与第 1/2 类 SDC 强相关。主文档第 1 类已有永久故障/SDC 条目（1.x）。**作为第 1 类补充来源引用**，刻画了硬件永久故障 → SDC 的特征分类，对推理 SDC 检测有参考价值。鉴于主文档聚焦推理正确性，**不单独新增为主条目**，可作为 1.x 的补充来源。
- **A3 (2512.14715)**：可微故障注入实现语义引导——BFA 家族的"可微"变体。第 3 类已有 9 条 BFA 变体。**作为第 3 类补充来源引用，不单独新增**（机制被现有 BFA 家族覆盖）。
- **A4 (2606.28958)**：多智能体协作中 KV-cache 完整性攻击。与 K4（cross-request 缓存键污染）相关但聚焦多智能体协作场景的新攻击面。**可作为 K4 补充来源引用**，鉴于其偏离单服务推理核心，**不单独新增**。

---

## 三、分类汇总

### 确认新增的条目（7 条）

| 序号 | 来源 | 故障名 | 目标类别 | 类型 |
|------|------|--------|----------|------|
| 1 | vLLM #51125 | `vllm_nongated_moe_w13_shard_hardcode2_uninit_nan` | 第 7 类 | 新签名 |
| 2 | vLLM #51198 | `vllm_prefix_cache_silent_noop_mamba_gdn_hybrid` | 第 6 类 | 新签名 |
| 3 | vLLM #51203 | `vllm_minimax_m3_fp8_query_alloc_unstable_cudagraph_replay` | 第 6 类 | 新签名 |
| 4 | SGLang #33726 | `sglang_qwen3vl_deepstack_input_dropped_bcg_replay` | 第 6 类 | 新签名 |
| 5 | tt-metal #52177 | `ttmetal_fpu_reduce_10bit_mantissa_truncation_topk_tiebreak` | 第 1 类 | 新签名 |
| 6 | tt-metal #52252/#52256 | `ttmetal_blackhole_dest_reuse_sync_race_nondeterministic_welford` | 第 1 类 | 新签名 |
| 7 | vLLM #51187 | （作为 4.1 的补充来源引用，不新增独立条目） | 第 4 类 | 已知类别新实例 |

**实际新增独立条目：6 条**（#51187 作为 4.1 补充来源，不占新编号）。

### 跳过的条目

| 来源 | 原因 |
|------|------|
| vLLM #51187 | 机制被 4.1 覆盖，作为补充来源引用 |
| vLLM #51200 | `MiniMAXGemmaRMSNorm` 无条件调 FlashInfer CUDA kernel 破坏非 CUDA 平台——平台可用性 bug，非静默数值正确性 |
| vLLM #51206 | Sampler per-request seed 在部分 seeded CPU batch 被丢弃——已知解码/采样确定性家族，与 4.1 同域；机制较窄（CPU 路径 early-return），作为解码/采样已知类的窄实例，**不新增**（可作 11 类风险记录） |
| vLLM #51122 | `enable_prompt_embeds` + `enable_lora` 并发 illegal memory access——crash 非 silent |
| vLLM #51161 | KV offload chunked local attention assert——crash 修复 |
| vLLM #51164 | Step3.5ReasoningParser 潜在（latent，#50918 合并后才生效）——parser 字符串处理 |
| vLLM #51182 | DeepGEMM SM90 paged mqa prefix_sum out of range——build/CI |
| vLLM #51192 | torchvision CVE 依赖升级——安全依赖 |
| tt-metal #52172 | SFPU 测试覆盖改进，非已发推理故障 |
| TRT-LLM #17306/#17305/#17321 | feature/crash 修复，非静默正确性 |
| arxiv 2608.02712 | 方法论（agent 修复 near-miss 算子） |
| arxiv 2604.10390 (LLM-PRISM) | 训练 SDC，作为 1 类补充来源引用 |
| arxiv 2512.14715 | BFA 可微变体，被第 3 类覆盖 |
| arxiv 2606.28958 | 多智能体 KV 完整性，偏离核心 |
| arxiv 2607.25227 / 2604.17249 / 2602.17837 | 已在主文档 |
| arxiv 2603.16382 / 2605.11202 / 2607.28884 | 防御/方法论/ZK，非故障 |

---

## 四、API 调用记录

| # | 查询 | 结果数 | 状态 |
|---|------|--------|------|
| 1 | `repo:vllm-project/vllm label:bug created:>2026-08-04` | 49 | OK |
| 2 | `repo:sgl-project/sglang label:bug created:>2026-08-04` | 0 | OK |
| 3 | `repo:sgl-project/sglang correctness created:>2026-08-04` | 12 | OK |
| 4 | `repo:NVIDIA/TensorRT-LLM correctness created:>2026-08-04` | 16 | OK |
| 5 | `repo:tenstorrent/tt-metal correctness created:>2026-08-04` | 13 | OK |
| 6 | `repo:triton-lang/triton correctness created:>2026-08-04` | 0 | OK |
| 7 | `repo:huggingface/transformers wrong inference created:>2026-08-04` | 0 | OK |
| 8 | `"silent data corruption" OR "wrong output" inference LLM created:>2026-08-04` | 930（噪声高） | OK |
| 9 | `repo:vllm-project/vllm numerical OR non-deterministic OR NaN created:>2026-08-04` | 8 | OK |
| 10 | `repo:vllm-project/vllm correctness OR "wrong result" OR "silent" created:>2026-08-04` | 29 | OK |
| 11 | arXiv: silent data corruption AND LLM | 3 | OK |
| 12 | arXiv: bit flip AND LLM AND weight | 5 | OK |
| 13 | arXiv: LLM inference AND correctness | 5 | OK |
| 14 | arXiv: KV cache AND corruption | 5 | OK |
| 15 | arXiv: GPU soft error AND machine learning | 0 | OK |
| 16 | arXiv: determinism AND LLM inference | 5 | OK |

**速率限制**：无 HTTP 403。GitHub 查询间隔 ≥7s。arXiv 查询间隔 ≥3s。
