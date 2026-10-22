# LLM 推理故障每日扫描报告 — 2026-08-08

**扫描日期**：2026-08-08（UTC）
**搜索窗口**：GitHub `created:>2026-08-01`（7 天滚动窗口）+ arxiv 近期
**主参考文档基线**：373 条（合并前）

---

## 扫描范围与 API 调用记录

| # | 目标 | 查询 | 结果数 |
|---|------|------|--------|
| 1 | vLLM | `label:bug created:>2026-08-01` | 283 |
| 2 | vLLM | `wrong OR incorrect OR corruption OR NaN OR silent OR determinism in:title created:>2026-08-01` | 19 |
| 3 | vLLM | `Bugfix silent OR stale OR wrong OR corrupt OR ignored in:title created:>2026-08-01` (p1+p2) | 223 |
| 4 | SGLang | `wrong OR incorrect OR corruption OR NaN OR silent OR determinism in:title created:>2026-08-01` | 23 |
| 5 | SGLang | `fix silent OR wrong OR corrupt OR ignored OR stale in:title created:>2026-08-01` (p1+p2) | 220 |
| 6 | tt-metal | `incorrect OR correctness OR silent OR wrong OR NaN OR corruption in:title created:>2026-08-01` | 26 |
| 7 | triton | `incorrect OR correctness OR wrong OR miscompile OR determinism in:title created:>2026-08-01` | 2 |
| 8 | transformers | `incorrect OR silent OR wrong OR NaN OR corruption in:title created:>2026-08-01` | 5 |
| 9 | TRT-LLM | `incorrect OR correctness OR wrong OR silent OR NaN in:title created:>2026-08-01` | 2 |
| A1 | arxiv | `silent data corruption LLM` | 3 |
| A2 | arxiv | `bit flip LLM` | 5 |
| A3 | arxiv | `KV cache corruption` | 5 |
| A4 | arxiv | `LLM inference correctness` | 5 |
| A5 | arxiv | `GPU soft error learning` | 3 |
| A6 | arxiv | `quantization LLM reliability` | 5 |
| A7 | arxiv | `LLM serving vulnerability` | 4 |

GitHub Search API 调用约 11 次（限额 10/min，每次间隔 ≥7s，未触发 403）；arxiv 7 次。

---

## 去重结果

对照 FAULT_MASTER_REFERENCE.zh.md 现有 ~422 个 `#编号` 引用，本日发现的下列条目经逐一比对**未出现在主文档中**（全部为新条目）。

---

## 发现条目（按故障领域分类）

### 硬件 / SDC（SFPU / 数值原语 / NaN 传播缺陷）

1. **[已知类别 var:1.x] tt-metal #52607 — `xielu` 对大负输入返回 Inf/NaN 及 ~1e38（指数下溢 clamp）**
   - 来源：https://github.com/tenstorrent/tt-metal/issues/52607（修复 #52608）
   - 机制：`xielu` 大负分支调用 `_sfpu_neg_exp_f32_`，该 helper 将缩放后的指数 clamp 在 `UNDERFLOW_THRESHOLD = -126.5`。对 x∈[-95,-80]，正确值约 26（小正数），但 SFPU 返回 Inf（bf16）/NaN（fp32）及 ~1e38 符号翻转。Wormhole B0 实测 60 输入点。

2. **[已知类别 var:1.x] tt-metal #52593 — `ttnn.pow`/`rpow` 静默吞掉 NaN 操作数（log2 快速路径无 is_nan 守卫）**
   - 来源：https://github.com/tenstorrent/tt-metal/issues/52593（修复 #52599）
   - 机制：log2 快速路径中 `sfpi::setexp(NaN, 127)` 将 NaN 重写为有限值（保留非零尾数、强制指数 127），NaN 被静默销毁；两个 power kernel 中零个 `sfpi::is_nan` 守卫。`pow(NaN, y)` / `pow(x, NaN)` 返回有限值而非 NaN，违反 IEEE-754。

3. **[已知类别 var:1.x] tt-metal #52178 — SFPU 饱和钳位在拟合域外静默返回常数（影响 erfc/digamma/softplus/i1，类 bug）**
   - 来源：https://github.com/tenstorrent/tt-metal/issues/52178
   - 机制：5 个 SFPU unary kernel（`erfc`、`digamma`、`softplus`、`i1` 及已修的）在拟合域外用*饱和*操作将参数/范围缩减索引钳入有效域。饱和操作不留下任何 trace（非 NaN、非 assert），返回域外的常数。共因在 SFPU 饱和语义层，非各 kernel 单独缺陷。

4. **[已知类别 var:1.x] tt-metal #52141 — bf16 add/sub 对 `Inf-Inf` 返回 0.0，fp32 返回 NaN（操作数 dtype 选引擎不一致）**
   - 来源：https://github.com/tenstorrent/tt-metal/issues/52141
   - 机制：`ttnn.subtract`/`add` 在 bf16 操作数下对 `Inf-Inf`、`Inf*0`、`Inf/Inf` 返回 0.0 或 +Inf，而 fp32 返回 NaN（IEEE 正确）。同一表达式、同一形状，仅 dtype 不同→结果不同。bf16 与 fp32 走不同 compute engine，NaN 生成路径不一致。Blackhole P300 实测。

5. **[已知类别 var:1.x] tt-metal #52112 — `ttnn.div(a, inf)` fp32 返回 NaN 而非 0（Markstein 残差精炼毒化正确结果）**
   - 来源：https://github.com/tenstorrent/tt-metal/issues/52112（#52114 同族）
   - 机制：fp32 DEST 累加下，`calculate_sfpu_binary_div` 先用 `reciprocal_iter` 得到正确的 0，但随后 Markstein 残差精炼 `result + (in0 - result*in1)*r` 对 `in1=Inf` 产出 `0 + (a - 0*Inf)*0 = NaN`。正确中间结果被残差步骤毁掉。IEEE 要求 `a/Inf = 0`。

6. **[已知类别 var:1.x] tt-metal #52036 — `xlogy` NaN 守卫是死代码（`in1 == nan` 恒假，`xlogy(x, NaN)` 返回 ~89）**
   - 来源：https://github.com/tenstorrent/tt-metal/issues/52036（修复 #52149）
   - 机制：`xlogy` 的 NaN 守卫用 `in1 == nan`，但 IEEE-754 中 NaN≠NaN，该比较恒假→守卫永不触发。`xlogy(x, NaN)` 返回 `x*log(x)≈89`（对 x=8）而非 NaN。应为 `sfpi::is_nan`。

### 服务栈 / 缓存正确性（vLLM）

7. **[已知类别 var:6.x] vLLM #51482 — prefix caching 关闭时 free_blocks 意外将 LIFO 翻转为 FIFO（块重用顺序反转）**
   - 来源：https://github.com/vllm-project/vllm/pull/51482
   - 机制：#48017 自称"纯 no-op"，实际条件 `block.block_hash is None and self.enable_caching` 在 prefix caching 禁用时把所有释放块走 `append_n`（FIFO）而非 `prepend_n`（LIFO）。分配从队头 pop→从"立即重用小块热集"变为"循环整个 block-id 空间"。TPU v7x 确定性触发 `RuntimeUnexpectedCoreHalt`。

8. **[已知类别 var:6.x] vLLM #51328 — `ExtractHiddenStatesProposer` 忽略 per-group slot_mapping，用错误 block_size 散射写入 → 70%+ 零值和 NaN/inf 隐藏态**
   - 来源：https://github.com/vllm-project/vllm/pull/51328
   - 机制：`propose()` 忽略传入的 `slot_mappings`（per-layer dict，用正确 group block_size 计算），改用 `common_attn_metadata.slot_mapping`（主组 block_size=1024）。CacheOnly 层 block_size=28 时 `basic_cache()` 用 28 除/取模→错误 block/offset→散射写入未分配块。产出零值和 NaN/inf 隐藏态，毒化下游训练。

9. **[已知类别 var:6.x] vLLM #51495 — LFM2 ShortConv 前缀不匹配 quant ignore list → conv 投影被静默按量化构建、权重加载失败**
   - 来源：https://github.com/vllm-project/vllm/pull/51495
   - 机制：`ShortConv` 存在属性 `.short_conv` 但构造时 `prefix=f"{prefix}.conv"`。quant ignore 经 `hf_to_vllm_mapper` 重写 `.conv.`→`.short_conv.`，而层报告前缀仍含 `.conv.`。`should_ignore_layer()` 返回 False→conv 投影按量化线性方法构建，但 checkpoint 只有无量化 `weight`→加载失败。

10. **[已知类别 var:7.x→6.x] vLLM #51477 — ue8m0 组量化融合 kernel 静默丢失 power-of-two scale 契约（发出 float32 group scale）**
    - 来源：https://github.com/vllm-project/vllm/pull/51477
    - 机制：norm+quant 和 act+quant 融合 pass 为 `is_e8m0=True` 注册变体，但重写目标 `_C.rms_norm_per_block_quant`/`silu_and_mul_per_block_quant` 无 ue8m0 参数，恒发 float32 group scale。匹配 `per_token_group_fp8_quant(..., scale_ue8m0=True)` 时，编译图静默丢失 DeepGEMM 路径依赖的 power-of-two scale 契约。SM120 实测 128/128 非 2 的幂 scale。

11. **[已知类别 var:11.x] vLLM #51508 — GDN/KDA 推测解码 stale zero-accept 步骤：Triton 越界崩溃，CPU 对齐模式静默 -1 索引损坏**
    - 来源：https://github.com/vllm-project/vllm/pull/51508
    - 机制：async-scheduling 丢弃的步骤把 `valid_sampled_token_count=0` 写入 `num_accepted_tokens`。所有消费者假设计数在 `1..1+num_spec_tokens`，用 `count-1` 索引→`-1`。Triton kernel（`fused_recurrent`/`causal_conv1d_update` 等）越界崩溃；但 CPU align 模式 `-1` 是合法 Python 索引→静默从 state 末尾读取，输出损坏无错误。

12. **[已知类别 var:8.x] vLLM #51483 — Kimi-K3 无状态首 chunk 被分类为 decode → 读取回收块残留的陈旧 KDA state**
    - 来源：https://github.com/vllm-project/vllm/pull/51483
    - 机制：非 spec 分支用 `split_decodes_and_prefills(decode_threshold=1)`，全 batch 单 token 行时 `num_prefills=0`，走纯 decode 路径无条件读 `conv_state[slot]`。但 Mamba 块重分配时不清零（`zero-on-allocate` 仅限 attention specs）→无状态首 chunk 静默继承前一拥有者的 KDA state。

13. **[已知类别 var:6.x] vLLM #51466 — DSD K-lookup 计入 prefill 请求导致 K 在 batch 边界反复 thrash（吞吐崩溃）**
    - 来源：https://github.com/vllm-project/vllm/pull/51466
    - 机制：动态推测解码 K-lookup 用 `len(num_scheduled_tokens)`（含 mid-prefill/chunked-prefill），prefill 混入 decode 暂时膨胀 batch size，K 在值间反复跳变（如 2→0→2）。每次翻转改变 verify-token 数、forward shape、CUDA graph dispatch。是 #49548 吞吐崩溃的诱因之一。

14. **[已知类别 var:6.x] vLLM #51468 — `truncate_computed_blocks` 对 hybrid full-attn+Mamba + KV connector 的断言杀死 prefill worker**
    - 来源：https://github.com/vllm-project/vllm/pull/51468
    - 机制：断言要求每个 KV cache group 至少持 `num_computed_tokens // block_size` 块。hybrid 模型 + KV connector 时，Mamba group 可合法持更少（其块是整个 recurrent state）。断言把良性差异变成 engine-fatal `AssertionError`。

15. **[已知类别 var:11.x] vLLM #51299 — `check_stop` 在 `min_tokens` 未达时提前返回，跳过 `max_model_len` 检查 → 请求永久挂起**
    - 来源：https://github.com/vllm-project/vllm/pull/51299
    - 机制：`check_stop()` 在 `num_output_tokens < min_tokens` 时提前返回（#26317 加），但也跳过下方 `max_model_len`/`max_tokens` 长度检查。若请求在到达 `min_tokens` 前已撞 `max_model_len`，调度器每步算 `num_new_tokens==0` 并跳过，请求永远留在 running queue 持有 KV 块。参数验证通过，易触发。

16. **[已知类别 var:8.x] vLLM #51295 — hybrid 稀疏注意力 Eagle drop 导致 prefix cache 意外全 miss**
    - 来源：https://github.com/vllm-project/vllm/pull/51295
    - 机制：Kimi K3 + Eagle，当 `prompt_length=24577`（非块对齐）时，Mamba 在两个边界位置（块对齐 + PMU 对齐）的 checkpoint 都坍缩到 24576，无 mamba checkpoint 能满足"额外缓存一块 full attention 以 drop"的要求→全 miss。

17. **[已知类别 var:6.x] vLLM #51489 — DSV4 FlashMLA sparse prefill 对 0-query-len chunk 崩溃（TMA 描述符 s_q=0）**
    - 来源：https://github.com/vllm-project/vllm/pull/51489
    - 机制：KVTransfer/SimpleCPUOffloadConnector 调度的 prefill 在 KV 加载期间含 `query_len==0` 请求。`get_prefill_chunk_plan` 可产出仅含 0-query-len 的 chunk→`query_start==query_end`→`_forward_prefill` 切出空张量传给 `flash_mla_sparse_fwd`，TMA 描述符 `s_q==0` abort。缺不变式：非保证每个 prefill 请求 ≥1 query token。

18. **[已知类别 var:10.x] vLLM #51400 — Kimi K3 turn-close 标记泄漏进流式 content（`<|close|>message<|sep|>` 未剥离）**
    - 来源：https://github.com/vllm-project/vllm/pull/51400
    - 机制：非流式 `_strip_response_content` 剥离 turn-close 标记，但流式 `_extract_response_content` 仅在 `tools open`/`response close` 停止 content。无 `response close` 的回合把原始 XTL 标记流给客户端。无 crash、无 NaN。

19. **[已知类别 var:6.x] vLLM #51306 — V2 CPU model runner 结构化输出假设 CUDA 可用 → engine 终止**
    - 来源：https://github.com/vllm-project/vllm/pull/51306
    - 机制：V2 CPU runner 继承 `StructuredOutputsWorker`，其结构化输出路径假设 CUDA→创建 CUDA stream、分配 pinned memory、调 Triton kernel。CPU-only 部署的结构化输出请求终止 engine。

### 服务栈 / 缓存正确性（SGLang）

20. **[已知类别 var:6.x] SGLang #34067 — 批量 KV free 别名：延迟 free 队列保留对可变 request-to-token 行的 tensor view，rematch 覆盖后 free 错误索引**
    - 来源：https://github.com/sgl-project/sglang/pull/34067
    - 机制：free group 保留由可变 `req_to_token` 行支撑的 tensor view。`cache_unfinished_req` 在 `free_group_end` 消费排队 view 前可覆写该行（radix-cache rematch）→释放替换索引而非原始 KV 索引。覆盖 token/paged/SWA/NPU/HiSparse/unified 分配器。

21. **[已知类别 var:6.x] SGLang #34076 — overlap 调度器在 forward stream kernel 仍读取时释放 overlap batch tensor（两处 use-after-free）**
    - 来源：https://github.com/sgl-project/sglang/pull/34076
    - 机制：缝隙1——`_forward_isolation` exit restore 丢弃 mid-forward 重绑定的 tensor（Spec-V2 worker 在 forward 内重绑 `input_ids`/`out_cache_loc`），forward-stream kernel 可能仍读。缝隙2——`mix_running_indices` 释放。两处均静默 in-range 损坏：缓存分配器把释放块给同流重分配，server 继续运行发错误 token。

22. **[已知类别 var:6.x] SGLang #34077 — relay staging 写未隔离 in-flight forward 的 tail relay 写 → 新请求首 resolve 静默读前一拥有者的 token**
    - 来源：https://github.com/sgl-project/sglang/pull/34077
    - 机制：overlap 调度下 FutureMap relay 从两流写同一 pool-indexed 行：forward stream 的 tail relay 写 vs schedule stream 的 batch-prep staging。无排序→若前一拥有者的最终 forward 仍在执行，stale tail write 可落在 staging write 之后→bootstrapped 请求首 resolve 静默读前一拥有者的 bonus token/topk/seq_lens。值在范围内，无越界。

23. **[已知类别 var:8.x] SGLang #33974 — 统一内存 + DSPARK：page hand-out 清零 + CuTe int32 slot-stride 回绕 → NaN**
    - 来源：https://github.com/sgl-project/sglang/pull/33974
    - 机制：统一内存池上接 DSPARK 推测解码的两个 NaN/损坏根因——(1) page hand-out 清零；(2) CuTe int32 slot-stride 回绕。Kimi-K3（TP8, 2x4 GB300）和 Kimi-Linear-48B（TP1 x8 B300）端到端验证。

24. **[已知类别 var:8.x] SGLang #33517 — 确定性 Triton extend + 统一内存 → 静默 NaN logits（out_cache_loc 未转译）**
    - 来源：https://github.com/sgl-project/sglang/pull/33517
    - 机制：`--enable-deterministic-inference` 路由 Triton extend 走 1-stage kernel，从 pool 读 prefix+extend 半的 KV。该 kernel 被传未转译的 `forward_batch.out_cache_loc`。统一内存池上 `out_cache_loc` 存*虚拟* id→读错位置→NaN logits。仅当 `SGLANG_ENABLE_ASYNC_ASSERT` 武装时 abort，否则静默返回有限 top-k logprobs。三标志任二组合无害。

25. **[已知类别 var:6.x] SGLang #34021 — DSV4 mHC combine 在 NCCL 对称池分配 transient → 与在飞 all-reduce buffer 别名 → token 损坏**
    - 来源：https://github.com/sgl-project/sglang/pull/34021（根因 #33800）
    - 机制：非 TileLang mHC combine fallback 在 `use_symmetric_memory` 上下文内运行 y-combine reduction，大中间和 dtype cast 在 NCCL 对称池分配。transient 与在飞 symmetric all-reduce buffer 间歇别名→损坏值进入 committed token。深度 5 非代码特殊，仅其 verify shape（m=bs×6）产生池占用碰撞模式。

26. **[已知类别 var:1.x] SGLang #34027 — SM120 page-split kernel int32 offset 溢出 → ~2GiB 外越界或静默 stray write**
    - 来源：https://github.com/sgl-project/sglang/pull/34027
    - 机制：`page_idx`/`sub` 源自 `tl.program_id(0)`（int32）。`page_idx * src_stride0` 为 int32 积，在 page_idx≈57,359 回绕（大 SM120 KV 池可达）。回绕的符号扩展 offset 落在预期 buffer 外 ~2GiB：未映射则非法内存访问 crash，映射则静默 stray write 进相邻 GPU 分配（权重/KV/scratch）。

27. **[已知类别 var:6.x] SGLang #34006 — Qwen3-MoE mori a2a 后端：`is_deepep()` 精确匹配漏 mori → 双重 all-reduce → garbage**
    - 来源：https://github.com/sgl-project/sglang/pull/34006
    - 机制：`MoeA2ABackend.is_deepep()` 精确 enum 匹配，mori 运行落入 `forward_normal`。`forward_normal` 在 mori 已 combine 的输出上再叠加 expert-parallel all-reduce。deepseek_v2/v4/glm4_moe 已门控 `is_mori()`，qwen3_moe 被漏。8x MI355X（gfx950）Qwen3-30B-A3B RL 复现。

28. **[已知类别 var:6.x] SGLang #33862 — HiCache `write_through` + 存储后端：host 镜像立即 pin 且永不释放 → 存储层静默饥饿死锁**
    - 来源：https://github.com/sgl-project/sglang/pull/33862
    - 机制：`write_through` 下每个插入节点立即 pin host 镜像；backup ack 只 `release_host()`（减计数，内存仍附 `node.host_value`）；唯一回收器 `evict_host` 拒绝 device-resident 节点。host(L2) 池小于 device 池时静默饥饿存储层。

29. **[已知类别 var:11.x] SGLang #33805 — DFLASH/DSPARK 单请求生成被提前杀死（SWA 驱逐不可达 → SWA 池耗尽）**
    - 来源：https://github.com/sgl-project/sglang/pull/33805
    - 机制：`spec_prepare_for_decode` 路由 dflash-family 到 `DFlashDraftInputV2.prepare_for_decode`，不调 `maybe_evict_swa`、不推进 `decode_batch_idx`（驱逐门要求 ≥1）。hybrid-SWA 模型 SWA KV 逐 token 保留→单请求约 `swa_full_tokens_ratio * max_total_num_tokens` 后耗尽 SWA 池被调度器 retract。DeepSeek-V4-Flash-0731 在 77,824 token 精确复现。

30. **[已知类别 var:8.x] SGLang #33810 — GDN chunked extend kernel 将 -1 padding sentinel 馈入指针算术（状态池前寻址）**
    - 来源：https://github.com/sgl-project/sglang/pull/33810
    - 机制：`_forward_metadata` 用状态索引 -1 毒化 padding 行。decode kernel 守卫 `if idx >= 0`，但 chunked gated-delta-net extend kernel 将 -1 直接馈入 initial-state load 和 final-state store 的指针算术→在状态池前寻址。`boundary_check` 不捕获（验证声明块形状，非分配）。breakable prefill CG + DP attention 暴露。

31. **[已知类别 var:8.x] SGLang #33872 — DSV4 推测 draft token >4 时 compress-state ring 静默欠写（从 stale slot 计算，静默精度退化）**
    - 来源：https://github.com/sgl-project/sglang/pull/33872
    - 机制：DSV4 spec verify 的 compress write-plan 用乐观 `seq_len = prefix_len + num_draft_tokens`，但验证后序列尾回滚到 `prefix_len + accept_len`。`c_plan.cuh` 硬编码 pad `kMaxMTPDraftTokens=4`。`num_draft_tokens>4` 时 plan 欠写 ring：`seq_len-4` 到 `seq_len-num_draft_tokens` 间 token 永不持久化。无越界/NaN/assert，仅静默精度退化。

32. **[已知类别 var:11.x] SGLang #34023 — DSpark compact/confidence 路径静默忽略 `--speculative-dspark-block-size`，锁定 checkpoint 原生 gamma**
    - 来源：https://github.com/sgl-project/sglang/pull/34023
    - 机制：compact/confidence verify 路径用 `self.gamma`（checkpoint 原生 `dspark_block_size`）view post-hc-combine 张量，而非运行时 draft window。其他消费者（static verify/planner/SPS）均遵 `--speculative-dspark-block-size`。compact 模式仅在 checkpoint 原生深度工作，其他 gamma 下 decode-graph capture 形状错误。DSV4-Flash 原生深度 5 恰是 SM120 上损坏的深度，两 bug 互掩。

33. **[已知类别 var:11.x] SGLang #33493 — DSPARK 从 `sampling_info` 取错误键 `acc_linear_penalities`（拼写错误 → penalty 静默 None）**
    - 来源：https://github.com/sgl-project/sglang/issues/33493
    - 机制：`dflash_utils.py` 用 `getattr(sampling_info, "acc_linear_penalities", None)`（多了 'i'），但 `sampling_info` 的属性是 `acc_additive_penalties`。getattr 默认返回 None→惩罚静默不应用。拼写错误导致的静默 no-op。

### 解码 / 采样故障

34. **[已知类别 var:11.x] vLLM #51304 — NaN-in-logits 检查在 MRV1 下每步阻塞 D2H，禁用 async scheduling**
    - 来源：https://github.com/vllm-project/vllm/pull/51304
    - 机制：`VLLM_COMPUTE_NANS_IN_LOGITS` 在 MRV1 的 `_bookkeeping_sync` 做 `logits.isnan().sum().cpu()`（主流阻塞 D2H）。不是 correctness bug 而是性能/可观测性缺陷——但影响 NaN 检测默认化计划。

---

## arxiv 论文（强相关但多为训练/防御/方法论视角）

- **arXiv:2604.17249 — "Bit-Flip Vulnerability of Shared KV-Cache Blocks in LLM Serving Systems"**（2026-04-19）：Rowhammer on GPU DRAM 对 vLLM Prefix Caching 的共享 KV-cache 块的位翻转攻击。**已编目**（3.x 类）。
- **arXiv:2607.25227 — "Decision-Level Hijacking: Injecting Cognitive Bias into LLMs via Bit-Flip Attacks"**（2026-07-28）：BFA 在决策层注入认知偏置。**已编目**（3.x 类）。
- **arXiv:2607.19623 — "From Bit-Position Sensitivity to Unequal Error Protection"**（2026-07-21）：16 工作负载的 per-bit-position 故障敏感性与不等错误保护。**已编目**（3.x 类补充）。
- **arXiv:2605.04563 — "RangeGuard: Efficient, Bounded Approximate Error Correction for Reliable DNNs"**（2026-05-06）：DRAM 密度增长下多比特错误增多，RangeGuard 提供有界近似纠错。BFA/可靠性视角，可作为第 3 类补充。
- **arXiv:2608.05863 — "Runtime Observability for Heterogeneous Attention Memory"**（2026-08-06）：latent cache/稀疏选择器/recurrent state 各以不同形式失效。方法论/工具，与 KV cache 监控相关，可作为附录方法论引用。
- **arXiv:2606.28958 — "When Latent Agents Lie: KV-Cache Integrity in Multi-Agent LLM Collaboration"**（2026-06-27）：多智能体系统中 agent 可发短消息同时传完整 KV-cache state，hidden state 可欺骗最终模型。KV cache 完整性新攻击面。
- **arXiv:2607.10855 — "Reliability Scaling Laws for Quantized Large Language Models"**（2026-07-12）：量化 LLM 的可靠性缩放律。量化可靠性分析，可作为量化类补充视角。
- **arXiv:2608.03854 — "Quantization Effects on Biomedical LLM Reliability"**（2026-08-04）：量化对生物医学 LLM 可靠性影响。质量评估（前次扫描已记）。

---

## 重点新发现（突出）

本日最值得关注的新案例：

1. **tt-metal #52178（SFPU 饱和钳位类 bug）** — **类签名**：5 个 SFPU unary kernel 在拟合域外经饱和操作静默返回常数。共因在 SFPU 饱和语义层而非各 kernel。与已编目的 1.43-1.47 同属 SFPU 数值原语缺陷族，但"饱和钳位无 trace"是新的系统性子模式。
2. **vLLM #51482（free_blocks LIFO→FIFO 翻转）** — **完全静默**：一个自称"纯 no-op"的 PR 意外翻转块重用顺序，TPU v7x 上确定性触发硬件 halt。属"缓存块池释放顺序"的新子签名。
3. **SGLang #34076/#34077（overlap 调度器 use-after-free / relay 写未隔离）** — 两处静默 in-range 损坏：forward-stream kernel 仍读时释放 tensor；relay staging 写与 in-flight forward tail 写竞争。属 overlap 调度器并发安全的新签名族。
4. **SGLang #34027（SM120 page-split int32 溢出→静默 stray write）** — page_idx int32 积在大 KV 池回绕，落 buffer 外 ~2GiB；映射则静默 stray write 进相邻分配。与已编目 int32 溢出同族但新实例。
5. **SGLang #33517（确定性 Triton extend 静默 NaN）** — 三标志组合（任二无害），out_cache_loc 未转译→静默 NaN logits。确定性推理路径的新签名。

无全新故障领域类别发现；全部为已有类别的新实例/新子签名。

---

## 将合并进主文档的条目（33 条，全部为新）

见主文档合并结果。
