# LLM 推理故障每日扫描报告 — 2026-08-01

- **扫描日期**：2026-08-01（UTC）
- **滚动窗口**：GitHub Issues 过去 7 天（created:>2026-07-25）+ 标题全文检索；arxiv 最近论文
- **主文档基线**：合并前 **261 条**（A1–A17 基础原语 17 + 第 1–15 类扩展目录 130 + K1.1–K10.31 KV 扩展 114）

---

## API 调用记录

### GitHub Search API（10 次查询，均成功，未触发 403 限流）

| # | 查询 | total_count |
|---|---|---|
| Q1 | `vllm-project/vllm` label:bug created:>2026-07-25 | 247 |
| Q2 | `vllm-project/vllm` wrong/corruption/NaN/silent/incorrect/correctness in:title | 765 |
| Q3 | `sgl-project/sglang` wrong/corruption/NaN/silent/incorrect/correctness in:title | 486 |
| Q4 | `NVIDIA/TensorRT-LLM` wrong/corruption/NaN/silent/incorrect/correctness in:title | 131 |
| Q5 | `huggingface/transformers` wrong/NaN/corrupt/correctness/silent/incorrect in:title created:>2026-07-15 | 12 |
| Q6 | `triton-lang/triton` wrong/numerical/NaN/corrupt/incorrect in:title | 207 |
| Q7 | `tenstorrent/tt-metal` wrong/NaN/correctness/numerical/corrupt in:title | 344 |
| Q8 | 全站 "silent data corruption" LLM | 1842（噪声高，多数无关 agent 框架） |
| Q9 | 全站 "bit flip" LLM weights | 87 |
| Q10 | 全站 "KV cache" corrupt OR incorrect inference | 37416（噪声高） |

> Q8/Q10 全站查询噪声高（多为 LLM agent 应用框架），有价值发现来自 Q1–Q7 的定向仓库搜索。

### arxiv API（6 次查询，`curl -sL` 跟随 301 重定向）

| # | 关键词 | 返回 |
|---|---|---|
| A1 | "silent data corruption" AND LLM | 3（最新 2026-04，训练场景为主，均已在主文档引用范围内） |
| A2 | "bit flip" AND LLM | 5（最新 2026-07-28，均为已编目论文） |
| A3 | "KV cache" AND corruption AND inference | 4（含 2510.17098 Cache-Side Vulnerabilities，新候选） |
| A4 | "LLM inference" AND correctness | 8（多为 cascade/系统方向，非具体正确性故障机制） |
| A5 | "GPU soft error" AND learning | 0 |
| A6 | quantization AND "wrong result" AND LLM | 0 |

---

## 扫描结果（逐条标注）

### 🆕 新故障模式 / 新签名

本轮无全新的顶层故障领域类别。但发现多个**已有类别中的新子签名**，且 #50671（GPU CC 下 UVA 零拷贝静默输入损坏）揭示了一个此前编目中**未覆盖的触发条件**（机密计算环境改变 CUDA 运行时语义 → 静默破坏 live-alias 契约）。

### 🔄 已知类别的新案例（去重后确认不在主文档）

| # | 标题 | 来源 | 分类标注 |
|---|---|---|---|
| 1 | fix(uva): fix silent input corruption and remote DoS under GPU CC — is_pinned() 在 CC 下误判 → 每个请求喂全零 token、HTTP 200 | vLLM #50671 | [已知类别·新子签名] 服务栈 → 第 6 类 |
| 2 | [Bug/Design] SGLANG_SANITIZE_NAN_LOGITS: 全 NaN logits 行 → softmax 均匀分布 → 纯随机 token 以 200 流式回客户端 | SGLang #33187 | [已知类别·新子签名] 解码/采样 → 第 11 类 |
| 3 | CompressedTensorsConfig(run_compressed=False) silently random-initializes unmapped fused-MoE expert weights on 5.10.x | HF transformers #47407 | [已知类别] 权重加载 → 第 12 类（var:from_pretrained_silent_random_init） |
| 4 | VLLM_ATTENTION_BACKEND env var is silently ignored — 字面量在源码中不存在，静默回退默认后端 | vLLM #50292 | [已知类别·新子签名] 服务栈 → 第 6 类（配置静默忽略族） |
| 5 | [Bug][XPU]: GDN attention silently corrupts memory under load — fix 已合并但 requirements/xpu.txt 锁定旧版本 | vLLM #49924 | [已知类别] 服务栈 → 第 6 类 |
| 6 | [AMD] Fix silent greedy fallback in speculative decoding verify on ROCm — _is_hip 短路使所有 HIP 请求强制 temperature=0 | SGLang #32922 | [已知类别·新子签名] 解码/采样 → 第 11 类 |
| 7 | [Bug]: SamplingParams accepts NaN for top_p, min_p and temperature — _validate 的正向范围检查不拒 NaN | TRT-LLM #17158 / #17159 | [已知类别·新子签名] 解码/采样 → 第 11 类 |
| 8 | [Bug] EAGLE speculative decoding defeats radix prefix reuse for multi-turn traffic — silent 97%→40-53% reuse collapse | SGLang #32459 | [已知类别·新子签名] KV缓存/注意力 → 第 8 类 |
| 9 | [Blackwell] Fix incorrectly pipelined RMW accumulator consumed by another MMA — sm_100 上 num_warps>=4 + num_stages>=2 静默误编译 | triton #10993 | [已知类别·新子签名] 硬件/SDC（内核累加器调度） → 第 1 类 |
| 10 | [ttnn/experimental/transformer] 5 defects: rotary_embedding_hf_sharded 数据格式未重配、hardcoded head count、wrong head stride | tt-metal #51284 | [已知类别·新子签名] 模型图/架构（位置编码内核数据格式） → 第 13 类 |
| 11 | Fix silently wrong EPLB output with --moe-a2a-backend none — per-rank 逻辑→物理映射破坏 rank-invariant dispatch，GSM8K 0.63→0.42 | SGLang #32962 | [已知类别·新子签名] 分布式/多GPU（EP 路由正确性） → 第 5 类 |
| 12 | [Bug] Vanilla attention uses incorrect KV indices for layer-specific cache layouts — 元数据在无 layer_idx 时一次性解析 | TRT-LLM #16801 | [已知类别·新子签名] KV缓存/注意力（层特定索引） → 第 8 类 |
| 13 | [ROCm][Bugfix] Kimi-K3 Fix KDA NaN on mixed batches + racy autotune config (num_stages=4 产生 ~1e38 垃圾值) | vLLM #50649 | [已知类别] 服务栈 → 第 6 类 |
| 14 | [CPU] MatMulNBits accuracy_level=4 (int8 activation quant) selects wrong argmax token on massive-activation LLMs | onnxruntime #29849 | [已知类别·新子签名] 量化（int8 激活量化在巨大激活通道上选错 argmax） → 第 7 类 |
| 15 | fix: top_logprobs entries silently collapse on decode-text collision — byte-fallback 分词器下不同 token id 解码同字符串 | SGLang #32387 | [已知类别] 解码/采样 → 第 11 类 |
| 16 | [None][fix] Emit KV cache removed events for silently detached reuse-tree blocks — 外部镜像永久泄漏 hash | TRT-LLM #16335 | [已知类别] KV缓存/注意力 → 第 8 类 |
| 17 | [Kernel] Harden top_k_per_row against NaN — NaN-dominated 行把直方图阈值 bin 落在 NaN bin，插入排序全 false → 输出未初始化 smem 作 index | vLLM #50201 | [已知类别] 服务栈 → 第 6 类（var:vllm_dsv4_sm12x_nan_mqa_topk_illegal_access） |
| 18 | ttnn.fmod: returns values outside [0,|s|) and wrong sign for |x|>~1e7 — 非精度不足，而是数学不可能的答案 | tt-metal #51661 | [已知类别] 硬件/SDC（数值正确性） → 第 1 类 |
| 19 | tt_metal/hw/firmware active_erisc.cc:115 — wrong-units: .data 字节大小传给 word-count 拷贝（4x overcopy，越界写 .bss/stack） | tt-metal #51238 | [已知类别] 硬件/SDC（固件拷贝单位错误） → 第 1 类 |
| 20 | [ttnn/embedding] 4 high-severity defects: index-CB overflow, dead traversal branch, wrong face offset, floor-vs-ceil tiles | tt-metal #51233 | 候选，**本轮列为备选未合并**（静态审计，embedding 路径，代表性弱于 #51284） |

### 跳过（已在主文档 / 重复签名）

| # | 标题 | 原因 |
|---|---|---|
| - | #50603 RDNA3 first-call non-determinism + long-seq corruption | 已与 14.6 `rdna_mamba_paged_attention_fallback_collapse` 同族（2026-07-31 扫描） |
| - | #50427 FlexAttention paged K/V offsets overflow int32 | 已在主文档 |
| - | #50084 NVFP4 KV block scale swizzle mismatch | 已在主文档（7.9） |
| - | #50067 EngineCore dies silently during warmup on GB10 | 已在主文档（6.15） |
| - | #33106 SGLang deepep uninitialized rows | 已在主文档（6.16） |
| - | #32968 SGLang Kimi-K3 [PAD] storm NaN | 已在主文档 |
| - | #32807 SGLang Gemma-3 RMSNorm high-rank skip | 已在主文档（13.13） |
| - | #17020 trtllm-eval aime silently disable thinking | 已在主文档（10.6） |
| - | #47496 DataCollatorWithFlattening 0-based position_ids | 已在主文档（9.6） |
| - | #47475 Zamba2 causality violation | 已在主文档（13.8） |
| - | #47405 Silent conversion skip on model_type mismatch | 已在主文档（12.8） |
| - | #51470 ttnn.maximum/minimum/clamp silently drop NaN | 已在主文档（1.14） |
| - | #51503 DRAM ND-sharded buffer corrupts duplicate x-coords | 已在主文档（1.15） |
| - | #51557 bf16 loses all 254 NaN encodings | 已在主文档（1.10 同族） |
| - | #51540 / #51537 SFPU isclose Inf/NaN bitpattern miscompare | 已在主文档（1.8 同族） |
| - | #10821 / #10927 triton dual-dot single accumulator / gather-load | 已在主文档（1.16 / 同族） |

### arxiv 论文（去重后）

| # | 标题 | arxiv | 分类标注 |
|---|---|---|---|
| P1 | Can Transformer Memory Be Corrupted? Investigating Cache-Side Vulnerabilities in LLMs | 2510.17098（2025-10-20） | [已知类别] KV 缓存侧信道攻击 → 补充来源引用，不单独编目（与 2.1/2.2 rowhammer + KV 同族） |
| P2 | LLM-PRISM: Characterizing SDC from Permanent GPU Faults in LLM Training | 2604.10390（2026-04-12） | [训练场景] 非推理正确性，**记入报告不合并** |
| P3 | Exploring SDC as a Reliability Challenge in LLM Training | 2604.00726（2026-04-01） | [训练场景] **记入报告不合并** |
| P4 | Decision-Level Hijacking via Bit-Flip Attacks | 2607.25227 | 已在主文档（3.7） |
| P5 | Bit-Flip Vulnerability of Shared KV-Cache Blocks | 2604.17249 | 已在主文档（2 处引用） |
| P6 | From Bit-Position Sensitivity to Unequal Error Protection | 2607.19623 | 已在主文档（3.8） |

---

## 本轮合并决策

**合并进主文档的新条目：19 条**（均为已知类别的新实例，含多个新子签名）：

- **第 1 类（硬件/SDC）+3**：`triton_blackwell_rmw_accumulator_mma_miscompile`（#10993）、`ttmetal_fmod_out_of_range_wrong_sign`（#51661）、`ttmetal_erisc_byte_vs_word_count_overcopy`（#51238）
- **第 5 类（分布式/多GPU）+1**：`sglang_eplb_rank_specific_logical_physical_map_corruption`（#32962）
- **第 6 类（服务栈）+5**：`vllm_uva_gpu_cc_silent_zero_token_input`（#50671）、`vllm_attention_backend_envvar_silently_ignored`（#50292）、`vllm_xpu_gdn_attention_silent_memory_corruption_under_load`（#49924）、`vllm_rocm_kimi_k3_kda_mixed_batch_nan_autotune_race`（#50649）、`vllm_topk_per_row_nan_uninit_smem_index`（#50201，var:vllm_dsv4_sm12x_nan_mqa_topk_illegal_access）
- **第 7 类（量化）+1**：`onnxruntime_matmulnbits_accuracy_level4_int8_act_massive_activation_wrong_argmax`（#29849）
- **第 8 类（KV缓存/注意力）+3**：`sglang_eagle_spec_decode_defeats_radix_prefix_reuse`（#32459）、`trtllm_vanilla_attention_layer_specific_kv_index_misresolution`（#16801）、`trtllm_reuse_tree_block_silent_detach_no_removed_event`（#16335）
- **第 11 类（解码/采样）+4**：`sglang_nan_logits_uniform_random_sample_200`（#33187）、`sglang_rocm_spec_decode_verify_silent_greedy_forced_temp0`（#32922）、`trtllm_samplingparams_nan_top_p_min_p_temperature`（#17158）、`sglang_top_logprobs_decode_text_collision_collapse`（#32387）
- **第 12 类（权重加载）+1**：`hf_compressed_tensors_run_compressed_false_silent_random_init_moe`（#47407，var:from_pretrained_silent_random_init）
- **第 13 类（模型图/架构）+1**：`ttmetal_rotary_embedding_hf_sharded_data_format_not_reconfigured`（#51284）

**备选未合并**：#51233（embedding 4 缺陷，静态审计，代表性弱）。

合并后主文档：**261 → 280 条**。
