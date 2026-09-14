# LLM 推理故障每日扫描 — 2026-08-12

- **扫描日期**：2026-08-12（窗口主要覆盖 2026-08-01 ~ 2026-08-12，arxiv 滚动 60 天）
- **搜索来源**：GitHub Issues/PR Search API（10 次查询）+ arxiv API（8 次查询）
- **目标仓库**：vllm-project/vllm · sgl-project/sglang · NVIDIA/TensorRT-LLM · huggingface/transformers · triton-lang/triton · tenstorrent/tt-metal

---

## API 调用记录

| # | 查询 | 命中数 | 去重后新增候选 |
|---|------|--------|----------------|
| 1 | vllm label:bug created:>2026-08-05 | 256 | — |
| 2 | vllm silent/wrong/incorrect/corruption/NaN in:title（自 08-01） | 31 | 12 新 |
| 3 | vllm KV cache/quantization/bit flip in:title（自 08-01） | 126 | 16 新 |
| 4 | sglang silent/wrong/corruption/NaN/incorrect in:title（自 08-01） | 32 | 10 新 |
| 5 | TensorRT-LLM wrong/incorrect/corruption/silent/NaN in:title（自 07-12） | 13 | 4 新 |
| 6 | transformers wrong/incorrect/corruption/silent/NaN in:title（自 08-01） | 13 | 5 新（多为非故障类） |
| 7 | triton wrong/incorrect/numerical/NaN in:title（自 07-12） | 9 | 2 新 |
| 8 | tt-metal wrong/corruption/silent/NaN/numerical/incorrect in:title（自 08-01） | 32 | 10 新 |
| 9 | vllm FP8/NVFP4/INT4/mixed precision label:bug（自 08-01） | 114 | 18 新（多为非正确性） |
| 10 | 跨仓库 "silent data corruption"/rowhammer/"non-deterministic" in:title | 1037 | 全部噪声（非 LLM 栈） |

arxiv 查询：silent data corruption LLM · bit flip LLM weight · KV cache corruption · LLM inference correctness · GPU soft error ML · quantization error LLM · weight corruption LLM · fault injection LLM inference

> GitHub 未命中 rate limit（全部 10 次查询均成功）。

---

## 扫描结果（新发现 — 去重后）

### 🔴 新故障模式 / 新子签名

#### 1. tt-metal `TTSampling.topk` 多核路径 iota 索引宽度只有 logits 的一半→错误 token — `[新签名]`
- **来源**：tt-metal #52928（2026-08-12 open） https://github.com/tenstorrent/tt-metal/issues/52928
- **描述**：单芯片 `TTSampling` 在 top-k 路径（非 argmax fast path）上返回**错误的 token**。`decode_forward` 交给 `ttnn.topk` 的 `iota` `indices_tensor` 宽度仅为 logits 的一半（padded vocab/2），导致多核 top-k 选择从截断的 logits 行选出 token。Mistral-7B-Instruct 在 500 个 teacher-forced token 上 Top-1 从 96.4%（argmax 路径）跌到 83.2%（k=1 top-k 路径），部分返回的 token **超出词表范围**。CI 使用 `temperature=0` 走 argmax fast path 因此全绿，长期掩盖了此 bug。这是 tt-metal #34763（自 2025-12 开启）的根因。
- **分类**：解码/采样 → 第 11 类（post-forward 结构性盲点）

#### 2. tt-metal `sdpa_decode` fp32_dest_acc_en=True 静默错误 + 96 head 挂起 — `[新签名]`
- **来源**：tt-metal #52903（2026-08-12 open，修复 #50564） https://github.com/tenstorrent/tt-metal/issues/50564
- **描述**：`sdpa_decode` 在 `fp32_dest_acc_en=True` 时对**大多数 shape 静默产生错误结果**，并在 96 query heads 时挂起。影响所有启用 fp32 目标累加的 SDPA decode 路径。
- **分类**：硬件/内核原语 → 第 1 类

#### 3. tt-metal `ttnn.scatter` 单索引更新静默丢弃 — `[新签名]`
- **来源**：tt-metal #52946（2026-08-12 open） https://github.com/tenstorrent/tt-metal/issues/52946
- **描述**：`ttnn.scatter` 写入 rank-1 operand 且**仅 1 个索引**时编译运行成功但**静默丢弃写入**（返回 operand 原值）。`jnp.zeros(3).at[jnp.array([0]).set(7)` 返回 `[0,0,0]`。两个或更多索引正常工作。根因：`pre_scatter_transform_tensor` 对 shape `{1}` 早退，导致 index/source 保持 rank-1 而 input 已 4D 变换，scatter kernel 的维度计算读越界，`in_bounds()` 恒为 false 直接跳过 scatter 阶段。
- **分类**：硬件/SDC 内核原语 → 第 1 类

#### 4. tt-metal `ttnn.pow` bf16 路径 z>=128 时指数回绕为 NaN — `[新签名]`
- **来源**：tt-metal #52707（2026-08-11 closed/merged） https://github.com/tenstorrent/tt-metal/pull/52707
- **描述**：bf16/non-fp32-dest pow 路径只在低端 clamp `z = y*log2(x)`（`z < -127`），高端 `z >= 128` 时 biased-exponent 重建**回绕而非饱和**，导致有限的大结果返回 NaN。`ttnn.pow(2.0, 128.0)` → NaN（应 +inf）；`10.0**39.0` → NaN。本 PR 添加高端 guard。**与 #52675（已编目）互补**：#52675 是同一路径的**精度**问题（误差随指数线性增长），本条是**溢出→NaN**问题——同一 kernel 的两个独立缺陷。
- **分类**：硬件/SDC 内核原语 → 第 1 类（var: ttmetal_pow）

#### 5. tt-metal `tril`/`triu` mask 乘法使 masked inf→NaN — `[新签名]`
- **来源**：tt-metal #52038（open）/ 修复 #52150（2026-08-12 closed） https://github.com/tenstorrent/tt-metal/issues/52038
- **描述**：`ttnn.tril`/`ttnn.triu` 由 `multiply(input, mask)` 实现，mask 是 0/1 index tensor。IEEE-754 下 `0 * ±inf = NaN` 且 `0 * NaN = NaN`，所以 masked-out 区域的任何 inf/NaN 变成 **NaN** 而非 0（fp32 路径；bf16 因指数域窄侥幸存活）。32×32 fp32 `-inf` 张量在 `tril` 下 496/496 masked cell 变 NaN。修复改用 `where(index, input, 0.0f)`。
- **分类**：硬件/SDC 内核原语 → 第 1 类

#### 6. tt-metal `ttnn.div(a, inf)` fp32 返回 NaN 而非 0 — `[新签名]`
- **来源**：tt-metal #52114（2026-08-12 open） https://github.com/tenstorrent/tt-metal/issues/52114 · 早期版本 #52112
- **描述**：启用 fp32 DEST 累加时，`ttnn.div(a, inf)` 对任意有限 `a` 返回 **NaN**（IEEE-754 要求 0）。乘法得到正确商，但 residual 精化路径中毒了正确结果。对依赖 div-by-inf 的数值稳定化（如 attention 归一化）可静默传播 NaN。
- **分类**：硬件/SDC 内核原语 → 第 1 类

#### 7. tt-metal bf16 add/sub 在 Inf-Inf 返回 0 而非 NaN — `[新签名]`
- **来源**：tt-metal #52141（2026-08-12 open） https://github.com/tenstorrent/tt-metal/issues/52141
- **描述**：`ttnn` 的 bfloat16 add/subtract 在 `Inf - Inf` 时返回 **0.0**，而 float32 路径正确返回 NaN。operand dtype 选择 compute engine，bf16 走了不同的 kernel 路径。与 IEEE-754 不符——Inf-Inf 应为 NaN，返回 0.0 会掩盖本应被检测到的数值异常。
- **分类**：硬件/SDC 内核原语 → 第 1 类

#### 8. tt-metal sharded softmax `subblock_w > Dest` 静默错误 — `[新签名]`
- **来源**：tt-metal #52040（2026-08-12 open） https://github.com/tenstorrent/tt-metal/issues/52040
- **描述**：`ttnn` sharded softmax 接受 `subblock_w` 大于 Dest 容量时**静默返回错误结果**——无 bounds 检查。用户可传入合法但超出硬件限制的 `subblock_w`，softmax 在 partial reduction 阶段丢弃数据。
- **分类**：硬件/SDC 内核原语 → 第 1 类

#### 9. vLLM DiffusionGemma FULL CUDA graph 冻结 attention mask — `[新签名]`
- **来源**：vLLM #51994（2026-08-12 open PR） https://github.com/vllm-project/vllm/pull/51994
- **描述**：`DiffusionGemmaModelState._causal_buf`（每请求 causal/encoder vs bidirectional/denoise 切换标志）在 `CUDAGraphMode.FULL` 捕获后**停止更新**。`build()` 通过 `causal.to(torch.int32)` 对非 int32 causal 张量每次分配新张量，FULL-mode graph 绑定捕获时看到的地址，之后永远读那个孤儿张量。任何经已捕获 FULL graph 服务的请求**静默**使用捕获（warmup）时的 causal/bidirectional 模式。**默认 O2 优化级解析为 FULL_AND_PIECEWISE，decode_mode=FULL**，所以 `vllm serve google/diffusiongemma-26B-A4B-it` 无参数即触发。E2E 影响难定量（denoising 自身 run-to-run 方差相近），但机制已用 capture/replay 测试直接复现。
- **分类**：服务栈/CUDA graph → 第 6 类

#### 10. vLLM PP decode FULL CUDA graph 捕获 gloo recv_object→静默挂起 — `[新签名]`
- **来源**：vLLM #51887（2026-08-12 closed） https://github.com/vllm-project/vllm/issues/51887
- **描述**：`VLLM_USE_BREAKABLE_CUDAGRAPH=1` 自动启用时（无 `@support_torch_compile` 的模型），vLLM 在 `splitting_ops` 为空时捕获 **full-decode** CUDA graph。在 PP>1 多节点上，decode forward 含 PP stage 边界的 gloo `recv_object`（host 端 Python 跨节点对象传输）。被捕获的 graph 重放时无 host 参与，graph 内的 `recv_object` 永远无法被服务→每个 PP stage 永久阻塞。失败模式：捕获**成功**（无非法内存访问），然后**第一个 decode 请求静默挂起**（~300s 后 `RPC call to sample_tokens timed out`）。与 #46253（TP 互补）不同。
- **分类**：分布式/多 GPU + CUDA graph → 第 5 类

#### 11. SGLang FlashInfer TRTLLM NVFP4 MoE tile-192 产生 NaN — `[新签名]`
- **来源**：SGLang #34629（2026-08-12 open） https://github.com/sgl-project/sglang/issues/34629
- **描述**：`flashinfer_trtllm` 在线 NVFP4 MoE 在升级对齐的 FlashInfer trio 超过 `0.6.16rc4` 后**数值失效**。Blackwell上新暴露的 tile-192 TRTLLM_GEN BMM 路径产生非有限 MoE 输出，200 例 GSM8K 测试得分 `0.0`。tile-128 tactic 通过全部正确性工作负载，两个 shape-valid tile-192 tactic 在启动 autotuning 期间失败。这是**依赖/集成回归**（FlashInfer 侧），非 SGLang 特性回归。
- **分类**：量化/内核 → 第 7 类

#### 12. SGLang aiter backend 静默丢弃 attention sinks — `[新签名]`
- **来源**：SGLang #34463（2026-08-12 open） https://github.com/sgl-project/sglang/pull/34463
- **描述**：aiter attention backend 的两个调用点**静默丢弃 attention `sinks`**。Site 1：`extend_attention_fwd` 用 16 位置参数 + 1 keyword 调用，但 callee 签名增长后 `sliding_window_size`（位置 19）和 `sinks`（位置 20）都静默回退默认值。两处都不报错而是产生**错误 logits**。通过 AST 而非肉眼证明。
- **分类**：KV 缓存/注意力 → 第 8 类

#### 13. SGLang mamba radix cache `extra_buffer` 缓存错误 SSM state — `[新签名]`
- **来源**：SGLang #33450（2026-08-12 open） https://github.com/sgl-project/sglang/pull/33450 · 关联 #31833
- **描述**：`--mamba-radix-cache-strategy extra_buffer` 在 NemotronH 模型上，mamba radix cache 对多请求 prefill batch 中的**大多数请求缓存了错误 token 位置的中间 SSM state**。无 crash、无 assert——cache 持有错误位置的状态。`_init_track_ssm_indices` 对每个未对齐 tracked request 构建 index 时，`num_h_states`/`cumsum` 计算在未对齐 chunk 边界上错位。
- **分类**：KV 缓存/注意力（SSM/recurrent state） → 第 8 类

#### 14. tt-metal `ttnn.topk` single-core 路径静默忽略 indices_tensor — `[已知类别，新机制]`
- **来源**：tt-metal #51921（2026-08-12 open） https://github.com/tenstorrent/tt-metal/issues/51921
- **描述**：`ttnn.topk(..., indices_tensor=...)` 的 `indices_tensor` 参数在 single-core 路径被**完全忽略**——`topk_single_core_program_factory.cpp` 硬编码 `GENERATE_INDICES="1"`，只有 `TopKMultiCoreProgramFactory` 读取该张量。W<8192 等所有 single-core shape 的 caller 静默获得生成的 iota 索引，传入的 payload 被丢弃。无 shape/dtype 契约检查。与 #52928 同族（topk 索引正确性）。
- **分类**：硬件/SDC 内核原语 → 第 1 类

#### 15. vLLM Mistral-Small-3.1 FP8 Pixtral 图像输入返回 NaN — `[已知类别]`
- **来源**：vLLM #52034（2026-08-12 closed） https://github.com/vllm-project/vllm/issues/52034
- **描述**：FP8-dynamic Mistral-Small-3.1（Pixtral）检查点：text/chat/tool/structured 请求正常，但**任何含图像的请求返回 NaN**。引擎不 crash；API 序列化失败（`Out of range float values are not JSON compliant: nan`）。vision_tower/multi_modal_projector/lm_head 保持 bf16（ignore list），仅 LM Linear 层 FP8 量化。`--enforce-eager` 可工作，编译启用时失败。
- **分类**：量化/精度 → 第 7 类

### 🟡 已知类别的新案例（补充）

#### 16. SGLang `update_weights_from_disk` 静默丢弃权重 — `[已知类别]`
- **来源**：SGLang #33588（2026-08-12 open） https://github.com/sgl-project/sglang/pull/33588
- **描述**：`update_weights_from_disk` 报告成功但实际丢弃权重：(a) VAE BatchNorm buffers 未加载（`named_parameters()` 排除 buffers），留下过期 latent 归一化统计；(b) text encoders 走通用 name-mapping 路径，无法 strip `model.` 前缀或 fuse q/k/v，无权重加载；(c) API 只要无异常就计为已更新。本 PR 加 `named_buffers()`、专用 TextEncoder 路径、per-module loaded count 报告。
- **分类**：权重/检查点加载 → 第 12 类

#### 17. vLLM `--cpu-offload-gb` 在 MRV2 静默 no-op — `[已知类别]`
- **来源**：vLLM #51396（2026-08-12 open） https://github.com/vllm-project/vllm/issues/51396
- **描述**：`--cpu-offload-gb`（UVA weight offloading）及 prefetch offload 选项在 V2 GPU model runner 激活时**静默忽略**——offloader 从未安装，模型完全加载到 GPU，无警告。对需要 offload 的工作负载（模型大于 VRAM），可见失败是 OOM，且消息建议无关 remedies（`lower --gpu-memory-utilization` 等）。
- **分类**：服务栈/配置 → 第 6 类

#### 18. vLLM `--enable-prefix-caching` 在 Mamba-2/GDN 混合模型是静默 0%-hit no-op — `[已知类别]`
- **来源**：vLLM #51198（2026-08-12 open） https://github.com/vllm-project/vllm/issues/51198
- **描述**：`Qwen3.6-35B-A3B`（Mamba-2/GDN hybrid）上 `--enable-prefix-caching` 即使请求共享长相同前缀也**0% prefix-cache hit**。recurrent conv+SSM state 不可 prefix-cache（APC 跟踪 attention KV blocks），所以 flag 实际是静默 no-op——caller 期望共享前缀 prefill 复用却得不到。
- **分类**：服务栈/缓存 → 第 6 类

#### 19. vLLM DeepGEMM paged-MQA metadata OOB（V4-Flash dummy-batch 全零长度）— `[已知类别]`
- **来源**：vLLM #51972（2026-08-12 open PR） https://github.com/vllm-project/vllm/pull/51972 · 关联 issue #51970
- **描述**：DeepSeek-V4-Flash 在 FULL cudagraph 捕获时 CUDA illegal memory access。indexer metadata builder 将 dummy-batch seq_lens（1）除以 `compress_ratio`（V4-Flash=4）得全零长度，DeepGEMM SM90 paged-MQA metadata scheduler 在所有长度为零且 `batch_size` 为 32 的倍数时（capture size 总是）读越界一个元素。`compress_ratio=1` 的模型永不触发。PR clamp scheduler-metadata 输入最小为 1（semantics-preserving，logits kernel 从 true context_lens 重新推导）。
- **分类**：服务栈/缓存 → 第 6 类

#### 20. SGLang DSPARK 取错误 key `acc_linear_penalities` — `[已知类别]`
- **来源**：SGLang #33493（2026-08-12 open） https://github.com/sgl-project/sglang/issues/33493
- **描述**：DSPARK 从 `sampling_info` 取 key `acc_linear_penalities`（应为 `acc_linear_penalties`，typo）。若 sampling_info 无该 key 则 KeyError 或静默跳过惩罚应用。
- **分类**：解码/采样 → 第 11 类

#### 21. tt-metal `ttnn.sort` ROW_MAJOR DRAM 多核路径错误值 — `[已知类别]`
- **来源**：tt-metal #52532（2026-08-12 open） https://github.com/tenstorrent/tt-metal/issues/52532
- **描述**：`ttnn.sort` 在 ROW_MAJOR 输入的 DRAM 多核路径返回**错误值**。与已编目的 topk/scatter 选择算法缺陷同族。
- **分类**：硬件/SDC 内核原语 → 第 1 类

### 📄 arxiv 新发现（可作为新条目或补充来源）

#### A1. SCOUT：LLM 预训练 SDC 运行时定位框架 — `[新论文，工具/检测]`
- **来源**：arxiv 2608.11034（2026-08-11） https://arxiv.org/abs/2608.11034 · 开源 https://github.com/LMResiliency/lm-resiliency
- **描述**：基于 strict-majority replica consensus 的运行时故障定位框架。out-of-band CPU observer 在训练 hang 时仍响应；in-situ replay 在真实 model state/kernel/alloc/comm/thermal/memory 旁重现 SDC。Collective fingerprints 暴露 rank-local protocol divergence；clean replay coverage 认证 checkpoint 数值完整性，防止从 SDC-corrupted state 恢复。集成 PyTorch/TorchTitan/Megatron-Core/DeepSpeed。
- **分类**：硬件/SDC 检测工具 → 第 1 类（作为检测/工具补充来源，强相关）

#### A2. CogBias：Bit-Flip 攻击实现 LLM 决策级劫持 — `[新论文，BFA 变体]`
- **来源**：arxiv 2607.25227（2026-07-28） https://arxiv.org/abs/2607.25227
- **描述**：定义"决策级劫持"威胁——通过翻转少量权重 bit 隐蔽、低成本、持久地操纵 LLM 认知立场，间接影响下游决策者。提出 CogBias 框架：可微分 sentiment evaluator 将主观偏好转为优化信号，多目标 loss 联合约束，BitScout 定位关键 bit，在超稀疏 flip 预算下实现定向认知干预。Llama-3.2-3B/Mistral-7B/Qwen2.5-14B 上少量 bit 翻转即稳定诱导目标话题立场显著偏移，对非目标任务影响有限。**与已编目 BFA 家族同族但新签名：认知偏见注入而非功能性退化。**
- **分类**：比特翻转攻击 → 第 3 类（新子签名：cognitive_bias_injection）

#### A3. Don't Regenerate, Debug：修复 near-miss 硬件算子 — `[新论文，工具]`
- **来源**：arxiv 2608.02712（2026-08-03） https://arxiv.org/abs/2608.02712
- **描述**：针对 GPU/NPU kernel 生成的 debug agent。许多被丢弃的候选是"near-miss"算子——编译运行通过但数值验证失败。Debug Pass@1 66.7% vs Regenerate Pass@3 40.7%，token 消耗少 92.8%。**与本项目相关性**：揭示了"编译运行通过但数值错误"的 near-miss kernel 是硬件算子层的系统性故障来源，与本编目中大量 tt-metal/Triton "静默错误"条目同源。
- **分类**：作为第 1 类（硬件/SDC）的方法论补充来源引用

---

## 重点突出的新发现

1. **tt-metal `TTSampling.topk` iota 宽度 bug（#52928）** 是本周最严重的采样正确性缺陷——CI 因走 argmax fast path 长期掩盖，实际 top-k 路径返回超出词表的 token，是 #34763（2025-12 开启）的根因。
2. **tt-metal 数值正确性集群（#52903/#52946/#52707/#52038/#52114/#52141/#52040）** 本周集中爆发 7 条新的 tt-metal 内核数值正确性 bug，覆盖 SDPA/scatter/pow/tril/div/add/softmax——SFPU/数值路径的系统性缺陷。
3. **vLLM DiffusionGemma FULL CUDA graph 冻结 attention mask（#51994）** 是新的 CUDA-graph capture 正确性 bug：默认 O2 即触发，无参数 `vllm serve` 即落入 FULL decode graph。
4. **SGLang FlashInfer NVFP4 tile-192 NaN（#34629）** 是依赖升级引入的量化回归——GSM8K 直接 0.0 分，tile-128 正常。
5. **arxiv CogBias（2607.25227）** 是 BFA 家族的新子签名：不破坏功能性，而是注入认知偏见，对齐安全视角。

---

## 本次合并计划

将上述 **15 条新子签名 + 6 条已知类别新案例 + 1 条 arxiv 新子签名** = **共 22 条新条目** 合并进主文档，分布：
- 第 1 类（硬件/SDC）：+9（#52903, #52946, #52707, #52038, #52114, #52141, #52040, #51921, #52532）
- 第 3 类（BFA）：+1（arxiv CogBias）
- 第 5 类（分布式）：+1（vLLM #51887 PP decode graph hang）
- 第 6 类（服务栈）：+4（vLLM #51994, #51396, #51198, #51972）
- 第 7 类（量化）：+2（SGLang #34629, vLLM #52034）
- 第 8 类（KV/注意力）：+2（SGLang #34463, #33450）
- 第 11 类（解码/采样）：+2（tt-metal #52928, SGLang #33493）
- 第 12 类（权重/检查点）：+1（SGLang #33588）

arxiv SCOUT（2608.11034）与 Don't Regenerate Debug（2608.02712）作为补充来源引用，不占新编号。
