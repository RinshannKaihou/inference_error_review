# LLM 推理故障每日扫描报告 — 2026-08-03

- **扫描日期**：2026-08-03（UTC）
- **滚动窗口**：GitHub Issues 过去 7 天（created:>2026-07-27，HF transformers 放宽至 2026-07-20）+ 标题/正文全文检索；arxiv 最近论文
- **主文档基线**：合并前 **296 条**（A1–A17 基础原语 17 + 第 1–15 类扩展目录 165 + K1.1–K10.31 KV 扩展 114）

---

## API 调用记录

### GitHub Search API（8 次查询，均成功，未触发 403 限流，查询间隔 7s）

| # | 查询 | total_count |
|---|---|---|
| Q1 | `vllm-project/vllm` label:bug created:>2026-07-27 | 258 |
| Q2 | `vllm-project/vllm` wrong/corruption/NaN/silent/incorrect/correctness in:title created:>2026-07-27 | 21 |
| Q3 | `sgl-project/sglang` 同上关键词 in:title created:>2026-07-27 | 15 |
| Q4 | `NVIDIA/TensorRT-LLM` 同上关键词 in:title created:>2026-07-27 | 4 |
| Q5 | `huggingface/transformers` wrong/NaN/corrupt/correctness/silent/incorrect in:title created:>2026-07-20 | 8 |
| Q6 | `triton-lang/triton` NaN/miscompile/numerical/incorrect in:title created:>2026-07-27 | 1 |
| Q7 | `tenstorrent/tt-metal` wrong/NaN/correctness/numerical/corrupt in:title created:>2026-07-27 | 12 |
| Q8 | `vllm-project/vllm` KV cache/quantization/nondeterministic in:title created:>2026-07-27 | 85 |

另用 Core API 抓取 10 个重点 issue 正文（#50881 #50749 #50901 #50889 #50903 #33397 #33414 #51873 #51903 #17022）。

### arxiv API（5 次查询，`curl -sL` 跟随 301 重定向）

| # | 关键词 | 返回 |
|---|---|---|
| A1 | "silent data corruption" AND LLM | 3（最新 2604.10390 LLM-PRISM 训练场景，昨日已记录不合并） |
| A2 | "bit flip" AND LLM AND weight | 5（2607.25227=3.7 已编目；2604.17249/2603.16382 已在主文档引用范围） |
| A3 | "KV cache" AND corruption AND inference | 4（2605.11202/2510.17098 昨日已记录；2605.08114 KV 量化统计质量，非故障机制） |
| A4 | "LLM inference" AND correctness | 6（2607.28884 ZK 验证、2607.25018 级联、2607.19490 P2P——均非具体故障机制） |
| A5 | "soft error" AND GPU AND learning | 3（2021/2024 旧文，非 LLM 推理） |

**arxiv 本轮无新增合并候选。**

---

## 扫描结果（逐条标注）

### 🆕 新故障模式 / 新签名（本轮重点）

- **vLLM #50881 + PR #50882** — MiniMax-M3 fp8_e5m2 KV cache on SM80（A100/A800）：bf16 路径 KV 写入走 fused CUDA op（graph-safe），fp8 路径走 `MiniMaxM3SparseAttention._insert_kv` 纯 Python advanced-index scatter。CUDA graph 捕获下该写入**损坏 index-K cache**，sparse indexer 选错 top-k 块 → attention 读错 KV。`--enforce-eager` 输出完全正常（实测 round-trip 误差 4.5% 排除精度问题），任何 CUDA-graph 模式生成纯垃圾。属「**非 graph-safe 的 Python KV 写入路径 → CUDA graph 捕获下索引缓存损坏**」新子签名。
- **vLLM #50749** — `int8_per_token_head` KV + **prefix caching 关闭** 时损坏**前几个生成 token**（Gemma-4 hybrid、Triton、KV 池利用率 ~3% 无压力）。完整 2×2 因子实验（n=48/arm）：prefix caching 开则干净。精度损失不可能依赖 cache 开关 → 正确性 bug。与已编目 #50702（池打满+驱逐+prefix on）机制不同。属「**int8 per-token-head KV × prefix-caching-off 交互 → 首 token 损坏（空闲池）**」新子签名。
- **SGLang #33397** — DeepSeek-V4-Flash-0731 on 2× H200（`--attention-backend dsv4` + DP attention）：**输出损坏随 batch size 放大**——轻微语言退化 → tool-call markup 错通道 → 完全 token soup。两套几乎无共享的配置均复现；同机同 checkpoint 同 fp8 KV 的 vLLM 0.26.0 无此问题。疑似 DSA（HiSparse）路径 batching 下 KV/page-index 损坏（与 #31482/#31901 同类但非 PD 路径）。未修复。属「**稀疏注意力页索引并发下渐进损坏**」新子签名。
- **vLLM #50903** — `fused_qk_norm_rope` partial-NeoX 路径（`rotary_dim < head_dim`）：仅 rotary lanes 进入 RoPE 段，但 `__shfl_xor_sync` 用**全 warp mask**（FINAL_MASK）调用 → 指向不执行 collective 的 lane = **UB**。packed kernel 中该无效 collective 在下一 head 的 RMSNorm reduction 之前 → warp chunk 内第一个之后的**每个 head Q/K 静默错误**；H200 上 `token_heads_per_warp=8` 还可产生 NaN。属「**partial rotary 下 divergent warp collective UB → 后续 head Q/K 损坏**」新子签名。
- **SGLang #33414** — auto-round GPTQ MoE：`AutoRoundConfig.apply_gptq_quant_layer` 委托 `MoeWNA16Config` 重跑 Marlin 资格检查，但委托 config **缺 `desc_act` 键** → 资格检查永远不合格 → 所有 GPTQ-packed auto-round MoE checkpoint 的 experts **静默全部落到 Triton wna16 fallback**；且 `use_marlin=False` 分支引用了只在 marlin 分支定义的变量（潜在 NameError）。属「**量化 kernel 分派资格检查缺键 → 静默回退通用 runner**」新子签名。
- **tt-metal #51873** — Quasar **硬件问题**：每个 Tensix 的高 16 个 tile counter（TC≥16）在 remapper 中用 5 位表示，而 overlay tensix 仅 4 位 → 编程/使用 TC A(≥16) 时 **TC A%16 被镜像损坏**。workaround：TC A 重映射到另一个 TC B(≥16)，完全并行 intra-tensix DFB 降为 8 个。属「**硬件 tile counter 位宽错配 → 镜像计数器损坏**」新子签名（硬件 erratum 级，Resnet on Quasar 调试中暴露）。
- **tt-metal #51903**（修复 #51215 findings 1&4）— retile compute kernel 进入 tilize 前**未调用 `pack_reconfig_data_format`**：当 `ttnn.tilize` 同时改变 tile 形状+dtype（BFLOAT16→BFLOAT8_B）时，每个 tile 以**输入格式**打包进 out_cb → 输出张量损坏；另有 padding reader 未对齐 4B 存储：行宽非 4 倍数时静默丢尾字节并覆盖真实输入。属「**tilize 缺 pack 格式重配置 → 错误 dtype 字节流输出**」新子签名。
- **vLLM #50901** — `BaseThinkingReasoningParser.extract_reasoning`：`<think>` 出现在流中段时，部分输出文本**既不进 content 也不进 reasoning**——静默丢失。影响所有未覆写该方法的继承 parser（deepseek_r1、poolside_v1）。属「**reasoning 分界解析器静默丢文本**」新子签名（与 11.11 流式 tool-call 丢文本同族不同机制）。
- **vLLM #50889** — Gemma4 tool parser（非流式）：并发 agentic 负载下间歇返回 `finish_reason="tool_calls"` 但 `tool_calls == []`（空数组），原文也不 surfaced——**模型的 tool move 被静默吞掉**（数千请求中 ~5 次/2 天）。属「**tool parser 并发下空 tool_calls + finish_reason 谎称 tool_calls**」新子签名。

### 🔄 已知类别的新案例

| # | 标题 | 来源 | 分类标注 |
|---|---|---|---|
| 1 | trtllm-eval `--max_output_length` 被 yaml `max_gen_toks` 无条件覆盖，CLI 静默无效 | TRT-LLM #17022 | [已知类别] → 第 10 类（var:10.6 `trtllm_eval_thinking_budget_key_silently_ignored` 族，eval harness 配置静默丢弃） |
| 2 | ttnn.fmod/remainder unary scalar：fix-up 循环反转 + guard 不可达 → 错号越界余数 | tt-metal #51441 | [已知类别] → 第 1 类（var:1.18 `ttmetal_fmod_out_of_range_wrong_sign` 族，unary 变体；昨日已分类但遗漏未合并，今日补入） |

### 跳过（已在主文档 / 重复签名 / 昨日已决）

| # | 标题 | 原因 |
|---|---|---|
| - | #50767 override-generation-config 静默忽略 penalty | 已在主文档（6.x，2026-08-02 合并） |
| - | #50671 / #50649 / #50427 / #50292 / #50201 / #50084 | 已在主文档 |
| - | #33268 / #33245 / #33206 / #33187 / #33186 / #33106 / #32968 / #32962 / #32922 / #32807 | 已在主文档 |
| - | #33263 SGLang abort fully-NaN logits | 与已编目 #33187/#33206 同一设计（实现 PR） |
| - | #17158/#17159 / #17020 / #47712 / #47496 / #47475 / #47467 | 已在主文档 |
| - | #50603 RDNA3 Triton paged-attention 非确定性 | 昨日已决：与 14.6 同族 |
| - | #50486 / #50477 / #32736 / #11118 / #51436 | 昨日已决：弱候选/备选未合并（#11118 仍为备选） |
| - | #51562 / #51557 / #51540/#51537 / #51503 / #51470 / #51661 / #51346 | 已在主文档（第 1 类） |

### 观察但不合并（非静默正确性 / 出范围）

| # | 标题 | 原因 |
|---|---|---|
| - | vLLM #50766 benchmark 静默 TP=1 | benchmark harness 配置问题，非推理服务正确性 |
| - | vLLM #50780 CUDA graph profiler 误计 → KV 池 −18~24% | 容量核算 bug，非输出正确性 |
| - | vLLM #50792/#50795 prompt_logprobs OverflowError（PD 分离） | crash 非静默 |
| - | vLLM #50877 DSpark spec decode MNNVL allreduce buffer insufficient | crash 非静默 |
| - | vLLM #50927 Gemma4 unified parser Rust 无界递归栈溢出 | crash/DoS 非静默 |
| - | vLLM #50283 / #50323 / #50376 / #50190 / #50110 | 防护性 feature / CI 稳定性，非故障案例 |
| - | SGLang #33353 diffusion ring parallelism 静默误算 | diffusion 出 LLM 推理范围，记录不合并 |
| - | tt-metal #51436 floor_div NaN guard 冗余 | 性能优化非正确性 |
| - | vLLM #50883/#50894/#50864/#50928/#50906/#50918 | KV offload/调度/kernel 修复，正文未见静默输出损坏证据，本轮观察不合并 |

---

## 本轮合并决策

**合并进主文档的新条目：11 条**（9 个新子签名 + 2 个已知类别变体）：

- **第 1 类（硬件/SDC）+3**：`ttmetal_tilize_pack_reconfig_missing_wrong_dtype_output`（#51903）、`ttmetal_quasar_upper_tile_counter_mirror_corruption`（#51873）、`ttmetal_fmod_remainder_unary_scalar_inverted_fixup`（#51441，var:1.18）
- **第 7 类（量化）+1**：`sglang_autoround_gptq_moe_marlin_branch_inverted_silent_triton_fallback`（#33414）
- **第 8 类（KV 缓存/注意力）+3**：`vllm_minimax_m3_fp8e5m2_kv_python_scatter_cuda_graph_corruption`（#50881+#50882）、`vllm_int8_per_token_head_kv_prefix_cache_off_first_token_corruption`（#50749）、`sglang_dsv4_flash_dsa_hisparse_progressive_corruption_under_batching`（#33397）
- **第 9 类（位置编码/RoPE）+1**：`vllm_fused_qknorm_rope_divergent_warp_collective_partial_neox`（#50903）
- **第 10 类（分词器/模板）+1**：`trtllm_eval_max_output_length_silently_discarded`（#17022，var:10.6）
- **第 11 类（解码/采样）+2**：`vllm_base_thinking_reasoning_parser_midstream_think_silent_text_drop`（#50901）、`vllm_gemma4_tool_parser_empty_tool_calls_concurrent_silent_loss`（#50889）

合并后主文档：**296 → 307 条**（+11）。
