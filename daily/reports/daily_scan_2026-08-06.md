# LLM 推理故障每日扫描报告 — 2026-08-06

**扫描日期**：2026-08-06（UTC）
**搜索窗口**：GitHub `created:>2026-08-05`（部分关键词放宽至 `>2026-08-01`）+ arxiv 近期
**主参考文档基线**：325 条（合并前）

---

## 扫描范围与 API 调用记录

| # | 目标 | 查询 | 结果数 |
|---|------|------|--------|
| 1 | vLLM | `repo:vllm-project/vllm label:bug created:>2026-08-05` | 30 |
| 2 | SGLang | `repo:sgl-project/sglang label:bug created:>2026-08-05` | 0 |
| 3 | SGLang | `"wrong result" OR incorrect created:>2026-08-01` | 21 |
| 4 | TRT-LLM | `label:bug created:>2026-08-01` | 0 |
| 5 | TRT-LLM | `correctness in:title created:>2026-08-01` | 0 |
| 6 | triton | `label:bug created:>2026-08-01` | 3 |
| 7 | tt-metal | `correctness in:title created:>2026-08-01` | 1 |
| 8 | tt-metal | `"wrong result" OR NaN OR incorrect created:>2026-08-01` | 87 |
| 9 | transformers | `label:bug correctness in:title created:>2026-08-01` | 0 |
| 10 | transformers | `"wrong output" OR incorrect OR NaN created:>2026-08-01` | 2 |
| 11 | SGLang | `deterministic OR speculative OR NaN created:>2026-08-03` | 105 |
| 12 | transformers | `label:bug created:>2026-08-03` | 2 |
| A1 | arxiv | `silent data corruption LLM` | 5 |
| A2 | arxiv | `bit flip attack LLM` | 5 |
| A3 | arxiv | `LLM inference correctness` | 5 |
| A4 | arxiv | `KV cache corruption` | 5 |
| A5 | arxiv | `GPU soft error learning` | 3 |
| A6 | arxiv | `quantization LLM error` | 5 |

GitHub Search API 调用 12 次（限额 10/min，未触发 403）；arxiv 6 次。

---

## 去重结果

对照 FAULT_MASTER_REFERENCE.zh.md 现有 362 个 `#编号` 引用，本日发现的下列条目**均未出现在主文档中**（全部为新条目）。

---

## 发现条目（按故障领域分类）

### 硬件 / SDC（SFPU / 内核原语缺陷）

1. **[已知类别 var:1.x] tt-metal #52278 — `ttnn.polygamma` silently wrong for x < 0**
   - 来源：https://github.com/tenstorrent/tt-metal/issues/52278
   - 机制：Euler-Maclaurin tail 在 `z = x + 6` 固定位移处展开，对负输入 `z` 坍缩，op 返回静默错误有限值（含符号翻转），无验证、无文档范围。

2. **[已知类别 var:1.x] tt-metal #52202 — `ttnn.i1` / `ttnn.erfc` 在拟合域外返回常量**
   - 来源：https://github.com/tenstorrent/tt-metal/pull/52202
   - 机制：两个 op 把输入钉在近似有效范围边缘并求值，域外返回单一值（如 `i1(89)=i1(1000)=i1(3e38)=1.15477e+37`）；SFPU pinning 不留痕迹（无 flag/NaN/IEEE overflow）。65536 个 bf16 pattern 中 47.9% 返回常量。

3. **[已知类别 var:1.x] tt-metal #52271 — `ttnn.sort` packer 格式未重置导致静默错误数据**
   - 来源：https://github.com/tenstorrent/tt-metal/pull/52271
   - 机制：sort 在 uint32 index 阶段后 packer data-format 寄存器留在 32-bit 态，后续 value untilize 的 `pack_untilize_init` 未完全重置，packer 把 4-byte datum 写进 2-byte bf16 页面 → stride-2 byte 交织损坏，每隔一个 datum 是垃圾（inf/nan）。

4. **[已知类别 var:1.x] tt-metal #52302 — matmul K-tile padding 对 block-float dtype 未初始化**
   - 来源：https://github.com/tenstorrent/tt-metal/pull/52302
   - 机制：`pad_last_ktile()` 只处理 Float32/Float16_b，对 `bfloat8_b`/`bfloat4_b` 在 K 非 tile 对齐时填充未初始化 → 陈旧 L1 数据参与 matmul 归约。

5. **[已知类别 var:1.x] tt-metal #52326 — `pad` 把 float32 pad value 当成两个 bf16 半字打包**
   - 来源：https://github.com/tenstorrent/tt-metal/pull/52326
   - 机制：5 个 pad 程序工厂有 INT32/UINT32 与 UINT16 分支，无 FLOAT32 分支，float32 落入 bf16 分支 → 填充值被 16-bit 舍入。

6. **[已知类别 var:1.x] tt-metal #52227 — dispatcher paged write bank walk 步幅用原始 page size 而非对齐值**
   - 来源：https://github.com/tenstorrent/tt-metal/pull/52227（closes #52165 Blackhole、#52166 Wormhole）
   - 机制：#51812 用手写 bank walk 替换 `get_noc_addr`，步幅用原始 page size 而非 allocator 对齐值 → page size 非对齐时每行后首行偏短，写入位置错误。

7. **[已知类别 var:1.x] tt-metal #52371 — `selective_reduce_combine` 用错误的 dense token-map stride**
   - 来源：https://github.com/tenstorrent/tt-metal/issues/52371
   - 机制：MoE `selective_reduce_combine` 用 `total_tokens` 除得 stride，但每 expert 行含 `total_tokens + 1`（含 `-1` 终止符）→ host 传错 per-token stride；Watcher 下触发确定性 assert。

8. **[已知类别 var:1.x] tt-metal #52342 — sdpa_decode mask helper 在 get_read_ptr 而非 get_write_ptr 填充**
   - 来源：https://github.com/tenstorrent/tt-metal/issues/52342
   - 机制：3 个 mask 生成 helper 在 `reserve_back` 后用 `get_read_ptr()` 取地址，但 reserved slot 地址是 `get_write_ptr()`；仅缓冲区为空时两者相等。当前无可观测错误行为（每次执行仅调用一次），但属潜在 latent bug。

### 服务栈 / 缓存正确性

9. **[已知类别 var:6.x] vLLM #51318 — DSv4 C128A decode topk row stride 非捕获稳定 → 瞬态输出损坏**
   - 来源：https://github.com/vllm-project/vllm/pull/51318
   - 机制：`_build_c128a_metadata` 的 row stride 从 batch 的 `max_seq_len` 派生，但 cudagraph capture 用 `max_model_len`（8192 for 1M context），runtime 用 128–4096 → row 0 永远对齐正确，其余行 stride 不匹配。表现：并发长上下文负载下 DSv4-Flash 出现 mid-generation `<|begin_of_sentence|>` burst 和 multilingual token salad，每 batch 一个请求，从不是第一个。**静默输出损坏**。

10. **[已知类别 var:6.x] vLLM #51287 — `VLLM_BATCH_INVARIANT=1` 不禁用 async scheduling → 非确定性**
    - 来源：https://github.com/vllm-project/vllm/pull/51287
    - 机制：该 flag 固定 kernel 使结果不依赖 batch size/composition，但不管 scheduler 把请求放进哪个 batch；async scheduling 默认开启且在当前 step 退役前组合下一步 batch，batch 成员依赖 wall-clock timing → timing jitter 进入数值。无 guard、无警告、无文档；服务器静默服务不可复现结果。

11. **[已知类别 var:6.x] vLLM #51250 — Prefix caching 对 Mamba-2/GDN hybrid 静默无效（0% hit）**
    - 来源：https://github.com/vllm-project/vllm/issues/51250
    - 机制：Qwen3.6-35B-A3B（`Qwen3_5MoeForConditionalGeneration`，Mamba-2/GDN 混合）启用 prefix caching 后，对完全相同的长前缀请求 cache hit 为 0%——recurrent/SSM 状态不存入 APC（只跟踪 attention KV block），无 prefix-cache 条目创建。prefix caching 实为静默 no-op。

12. **[已知类别 var:6.x] vLLM #51228 — EncoderDecoderCacheManager slot release 非幂等 → 编码器预算静默停止执行**
    - 来源：https://github.com/vllm-project/vllm/pull/51228
    - 机制：encoder-decoder 模型（Whisper/Voxtral），`allocate()` 每请求一次，但 `_free_encoder_inputs` 每 step 调用 `free_encoder_input()` 且无引用守卫 → `num_free_slots` 每步重复计入，编码器缓存预算在首个 decode step 后静默失效。

13. **[已知类别 var:6.x] SGLang #33872 — DSV4 推测解码 draft token >4 时静默 KV 损坏**
    - 来源：https://github.com/sgl-project/sglang/pull/33872
    - 机制：DSV4 speculative verify batch 的 compress write-plan 用乐观 `seq_len = prefix_len + num_draft_tokens` 构建，但验证后序列回滚到 `prefix_len + accept_len`。`c_plan.cuh` 硬编码 `kMaxMTPDraftTokens = 4`，`speculative_num_draft_tokens > 4` 时 plan under-writes ring → `seq_len - 4` 到 `seq_len - num_draft_tokens` 之间的 token 永不持久化。**静默——无 illegal access、无 NaN、无 assert**，压缩状态从陈旧 slot 计算，表现为安静精度回归。

14. **[已知类别 var:6.x] SGLang #33877 — PD 分离 MTP 元数据 buffer 不对称 → Mooncake 传输失败**
    - 来源：https://github.com/sgl-project/sglang/pull/33877
    - 机制：GLM-5.2-FP8 PD 分离，Prefill 无 MTP + `tc_piecewise` CUDA Graph，Decode 有 MTP，两节点分配不同 `output_hidden_states` metadata buffer schema（Prefill 用 16-element FP32 padding，Decode 用模型 MTP hidden size/dtype）→ Mooncake `Failed to get segment descriptor`。

### KV 缓存 / 注意力机制故障

15. **[已知类别 var:8.x] SGLang #33603 — CPU 后端静默忽略 encoder 模型的双向 sliding window attention**
    - 来源：https://github.com/sgl-project/sglang/issues/33603
    - 机制：encoder 模型（如 ModernBERT）在 CPU（`--device cpu`）上，CPU attention 后端对长上下文静默回退到 full global attention 而非 sliding window → 与 HF Transformers 的双向上下文对等性破坏，cosine similarity 大幅下降（如 0.86）。

### 解码 / 采样故障

16. **[已知类别 var:11.x] vLLM #51230 — MiniMax-M3 thinking 启用时 structured output 静默忽略**
    - 来源：https://github.com/vllm-project/vllm/pull/51230
    - 机制：`thinking_mode: enabled` 时，scheduler 的 structured-output gate 永远观察不到 reasoning 结束——`MiniMaxM3ReasoningParser.is_reasoning_end_streaming` 在 `_reasoning_active_streaming` 置位时早返回 `False`，而该 flag 只被 frontend/detokenizer 的 `extract_reasoning_streaming` 清除（scheduler 从不调用）。→ `reasoning_ended` 不锁存，`should_fill_bitmask` 永远 `False`，grammar 不启用，请求正常完成 → **对客户端不可见**。

17. **[已知类别 var:11.x] vLLM #51226 — CPU 上当 batch 内另一请求无 seed 时 seed 被忽略**
    - 来源：https://github.com/vllm-project/vllm/issues/51226
    - 机制：CPU 后端，`SamplingParams(seed=...)` 在 batch 同时含无 seed 请求时不被尊重——seeded 请求输出每次不同；给第二个请求也加 seed 后第一个突然可复现。seeded 请求的确定性依赖同 batch 其他请求的采样参数。

18. **[已知类别 var:11.x] SGLang #33695 — deterministic sampling 拒绝 min-p + seed 组合**
    - 来源：https://github.com/sgl-project/sglang/issues/33695
    - 机制：`SamplingParams` 接受 `min_p` + seed 组合，但 `top_k_top_p_min_p_sampling_from_probs_torch()` 抛 `AssertionError: With sampling seed, multinomial_with_seed will provide wrong results`——min_p 过滤后 seeded multinomial 语义未定义/未实现。

19. **[已知类别 var:11.x] SGLang #33643 — 推测解码下采样惩罚只按 1 token/step 累积（应为 accept_len）**
    - 来源：https://github.com/sgl-project/sglang/pull/33643
    - 机制：decode step 提交 `accept_len` token，但 `cumulate_penalty_output_tokens` 只喂 `req.output_ids[-1]` 给 penalizer。`frequency_penalty` 有效强度变 ~`nominal/L`；`min_new_tokens` 按 decode step 而非 token 计数，EOS 被 mask ~`L`× 更久（错误硬约束）；`presence/repetition_penalty` 丢弃 token 延迟激活。

20. **[已知类别 var:11.x] SGLang #33356 — DSpark 大 decode CUDA-Graph capture 非确定性 illegal memory（TP8）**
    - 来源：https://github.com/sgl-project/sglang/issues/33356
    - 机制：SGLang v0.5.16 + DeepSeek-V4-Pro-DSSpark + TP8，大 decode CUDA-Graph capture 在 server 启动时以 CUDA illegal-memory 或 host SIGSEGV 失败；跨独立启动，首个失败 shape/rank/surfacing kernel/mode 均不同。H200/TP4 独立复现。非确定性——属 capture 时序/内存竞态。

### 模型图 / 架构 / 分词器 / 模板

21. **[已知类别 var:13.x] vLLM #51305 — DSL kernel 仅支持 fp16/bf16 输入，float32 dtype 时损坏**
    - 来源：https://github.com/vllm-project/vllm/issues/51305
    - 机制：DSL（CUTLASS DSL）kernel 只支持 fp16/bf16 输入，`torch.float32` dtype 时行为损坏（描述待补充细节）。

22. **[已知类别 var:10.x] transformers #47822 — 模板生成中 special tokens 未转义**
    - 来源：https://github.com/huggingface/transformers/issues/47822
    - 机制：`apply_chat_template` 生成模板时，用户消息中包含的服务 token（如 `<s>`）未被转义 → 可能被当作真实 special token 处理，造成注入/模板损坏。

23. **[已知类别 var:10.x] transformers #47752 — `generation_config` 值优先级错误**
    - 来源：https://github.com/huggingface/transformers/issues/47752 · 修复 PR #47754
    - 机制：`generation_config` 的值优先级不正确，导致生成参数（如 temperature/top_p）被错误覆盖。

### Triton 内核数值正确性

24. **[已知类别 var:1.x/4.x] triton #11153 — 编译后 kernel 静默返回错误结果（interpreter 正确）**
    - 来源：https://github.com/triton-lang/triton/issues/11153
    - 机制：同一 kernel 在 Triton interpreter 与编译 GPU 下结果不同——FMA kernel `y = A @ x` 对 `ones(4,4) @ ones(4)` 编译后返回 `[2,4,4,4]`（期望 `[4,4,4,4]`），interpreter 正确。语义等价的重写 kernel 产生正确结果。**静默数值错误，仅 cross-check 才能发现**。

25. **[已知类别 var:4.x] triton #11165 — softmax + floor + exp 与 PyTorch 不一致**
    - 来源：https://github.com/triton-lang/triton/issues/11165
    - 机制：32×32 float32 kernel 做 softmax→sum→floor→exp，对确定性 ramp 输入返回与 PyTorch 参考不同的结果；归一化后 floor 的舍入边界处理与参考实现不一致。

### arxiv 论文（强相关但多为训练/防御视角，非推理服务故障新模式）

- **arXiv:2608.03854 — "Quantization Effects on Biomedical LLM Reliability"**（2026-08-04）：量化对生物医学 LLM 可靠性的影响。属量化质量评估，非新故障机制。
- **arXiv:2607.25227 — "Decision-Level Hijacking via Bit-Flip Attacks"**（2026-07-28）：BFA 在决策层注入认知偏差。属 BFA 攻击新变体（第 3 类已有覆盖）。
- **arXiv:2608.02712 — "Don't Regenerate, Debug"**（2026-08-03）：修复 near-miss 硬件算子的领域专用 agent。方法论，非故障编目。
- 其余 arxiv 结果多为训练阶段 SDC/防御，与推理服务故障编目相关性较弱，不单独编目。

---

## 重点新发现（突出）

本日最值得关注的新案例：

1. **vLLM #51318（DSv4 C128A topk stride）** — 并发长上下文下**静默输出损坏**（multilingual token salad），cudagraph capture stride 与 runtime stride 不匹配。典型的"capture-time vs runtime metadata 分歧"签名，与 #50749（Gemma-4 prefix caching 相关损坏）同族但机制不同。
2. **SGLang #33872（DSV4 draft>4 静默 KV 损坏）** — **完全静默**（无 crash/NaN/assert），仅表现为精度回归。compress-state ring 的 under-write 是推测解码与压缩状态交互的新签名。
3. **vLLM #51287（batch-invariant 不禁用 async scheduling）** — 揭示 `VLLM_BATCH_INVARIANT` 在默认 async scheduling 下**不兑现其承诺**，是对现有 4.1 确定性条目的重要补充（4.1 已记 #51187 高并发非确定性，本条补另一面：scheduler 层的非确定性源）。
4. **tt-metal #52202（i1/erfc 域外返回常量）** — SFPU pinning 无痕迹，47.9% bf16 pattern 返回错误常量，是"SFPU 近似域外静默错误"签名的强实例。

无全新故障领域类别发现；全部为已有类别的新实例/新子签名。

---

## 将合并进主文档的条目（25 条，全部为新）

见主文档合并结果。
