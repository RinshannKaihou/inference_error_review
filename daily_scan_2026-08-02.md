# LLM 推理故障每日扫描报告 — 2026-08-02

- **扫描日期**：2026-08-02（UTC）
- **滚动窗口**：GitHub Issues 过去 7 天（created:>2026-07-26）+ 标题/正文全文检索；arxiv 最近论文
- **主文档基线**：合并前 **280 条**（A1–A17 基础原语 17 + 第 1–15 类扩展目录 149 + K1.1–K10.31 KV 扩展 114）

---

## API 调用记录

### GitHub Search API（10 次查询，均成功，未触发 403 限流）

| # | 查询 | total_count |
|---|---|---|
| Q1 | `vllm-project/vllm` label:bug created:>2026-07-26 | 257 |
| Q2 | `vllm-project/vllm` wrong/corruption/NaN/silent/incorrect/correctness in:title created:>2026-07-26 | 22 |
| Q3 | `sgl-project/sglang` wrong/corruption/NaN/silent/incorrect/correctness in:title created:>2026-07-26 | 13 |
| Q4 | `NVIDIA/TensorRT-LLM` wrong/corruption/NaN/silent/incorrect/correctness in:title created:>2026-07-26 | 4 |
| Q5 | `huggingface/transformers` wrong/NaN/corrupt/correctness/silent/incorrect in:title created:>2026-07-19 | 8 |
| Q6 | `triton-lang/triton` NaN/miscompile/overflow/numerical/label:bug created:>2026-07-26 | 1+4+1 (3 次子查询) |
| Q7 | `tenstorrent/tt-metal` wrong/NaN/correctness/numerical/corrupt in:title created:>2026-07-26 | 22 |
| Q8 | `sgl-project/sglang` cache_salt/cross-run/DeepSeek-V4/fused wq_a in:title | 471 (含 DeepSeek-V4 集中爆发) |
| Q9 | `triton-lang/triton` label:bug created:>2026-07-26 | 4 |
| Q10 | `vllm-project/vllm` override-generation-config / FlashInfer sampler / Gemma3 intermediate_tensors | 3+ 定向 |

### arxiv API（7 次查询，`curl -sL` 跟随 301 重定向）

| # | 关键词 | 返回 |
|---|---|---|
| A1 | "silent data corruption" AND LLM | 3（最新 2026-04，训练场景，已在主文档引用范围） |
| A2 | "bit flip" AND LLM AND weight | 5（最新 2607.25227 = 3.7 已编目） |
| A3 | "KV cache" AND corruption AND inference | 4（含 2510.17098 Cache-Side，已引用；**新候选 2605.11202**） |
| A4 | "LLM inference" AND correctness | 6（多系统/级联方向，非具体机制） |
| A5 | "soft error" AND GPU AND learning | 3（旧，2021/2024，非 LLM 推理） |
| A6 | quantization AND "wrong result" AND LLM | 0 |
| A7 | "KV cache" AND attack AND serving | 5（含 2606.21842 Agent-Assisted Side-Channel RAG KV，**新候选**） |

---

## 扫描结果（逐条标注）

### 🆕 新故障模式 / 新签名

本轮无全新的顶层故障领域类别。但发现多个**已有类别中的新子签名**：

- **#33245**（SGLang）— 量化 DeepSeek-V4 的 fused `wq_a`+`wkv` 融合投影加载路径**只映射未量化的 `.weight` 名**，packed 张量名（`qweight`/`qzeros`/`scales`）无处可去被**静默丢弃**，服务正常启动、捕获 CUDA graph、返回 HTTP 200——输出是流畅的垃圾。属「融合投影加载路径只认未量化名 → packed 权重静默丢弃」新子签名。
- **#33268**（SGLang）— HiCache 存储键（storage key）**遗漏 `kv_cache_dtype`** 字段。两次相同模型、不同 `--kv-cache-dtype`（如 bf16→fp8_e4m3）的运行产生**相同 key**，第二次运行静默复用错误 dtype 字节格式的缓存。属「持久缓存键未含 dtype → 跨运行格式碰撞」新子签名。
- **#50761**（vLLM ROCm）— Kimi-K3 的 `e_score_correction_bias` 在 checkpoint 中是 FP32，AMD 模型在**权重加载前**就把该参数建在环境 BF16 dtype 下，标准 loader 把 FP32 拷进 BF16 存储；后续 AITER 运行时 upcast **无法恢复被丢弃的低位**。属「FP32 关键参数在加载时被静默降精度，post-load upcast 无法恢复」新子签名。
- **#11148**（Triton）— 同一 kernel 在 `BLOCK_SIZE=512/1024` 时崩溃（非法内存访问），在 32/64/128/256/2048 时**完全正确**；移除 5 个注入块之一、或把 `tl.trans` 换成 `tl.reshape`、或移除死跳转块中的 `return` 均消除崩溃。属「特定 tile 形状触发的 codegen/调度静默误编译」新子签名。
- **#51346**（tt-metal）— Quasar 架构上 `sfpi::lut2`/`lut2_sign` 把分段查表表的 6 个字写入 WH/BH 的通用 LReg，但 Quasar 的 `SFPLUTFP32` 表源在**常量 LReg（9–14）**而非通用 LReg——表源寄存器类错配 → 静默错误查表值。属「架构代际寄存器类迁移未跟随 → LUT 表源错配」新子签名。
- **#51562**（tt-metal）— `ttnn.sort` 的 TopK SFPU LLK 默认用 **bf16** 比较 tile 值；bf16 仅 8 位尾数，整数 >256 全部舍入到同一 bf16 值成为「相等」，比较器把相等的元素留在 merge-sort 网络碰巧放置的顺序 → 返回的 indices **非确定且可错**。属「低精度比较器把不同整数折叠为相等 → 排序索引非确定」新子签名。

### 🔄 已知类别的新案例（去重后确认不在主文档）

| # | 标题 | 来源 | 分类标注 |
|---|---|---|---|
| 1 | [Bug] HiCache storage keys omit kv_cache_dtype — silent cross-run cache collisions | SGLang #33268 | [已知类别·新子签名] 服务栈 → 第 6 类（持久缓存键缺 dtype 维度） |
| 2 | [Bug] Quantized DeepSeek-V4: fused wq_a+wkv path silently drops every layer's packed weights (server runs, output is garbage) | SGLang #33245 | [已知类别·新子签名] 量化/权重加载 → 第 7 类（var:quantized_kernel_bias_omission 族，packed 权重静默丢弃） |
| 3 | [Bug] Multi-node TP rank-divergence deadlock: one rank wedges in NCCL proxy append (logits all-gather), peer idles at request broadcast — DeepSeek-V4 + DSpark on 2× DGX Spark (GB10) | SGLang #33289 | [已知类别·新子签名] 分布式/多GPU → 第 5 类（per-rank batch 分歧 → 不匹配 collective → NCCL 死锁） |
| 4 | [Bugfix][Kimi-K3] Preserve MoE correction bias in FP32 — AMD 模型在加载前把 FP32 bias 参数建在 BF16，post-load upcast 无法恢复低位 | vLLM #50761 | [已知类别·新子签名] 模型图/架构 → 第 13 类（关键参数加载时静默降精度） |
| 5 | [PD][NixlPush][Bugfix] Notify D of NIXL writes that failed to post so it stops waiting — decode 节点数 notification 永远等不到，请求挂到 lease 超时 | vLLM #50754 | [已知类别·新子签名] 服务栈/KV分离式 → 第 6 类（var:pd_disagg，notification 计数失配） |
| 6 | [Bugfix] Fix DeepSeek-V4 non-EP TBO for attention TP > 1 — full-TP variable-length collectives 把 DP-local 副本当独立分片处理 → gather/combine 布局损坏 | SGLang #33250 | [已知类别·新子签名] 分布式/多GPU → 第 5 类（TP×DP attention 副本误分片） |
| 7 | [Bug]: override-generation-config silently ignores presence_penalty / frequency_penalty (missing from available_params whitelist) | vLLM #50767 | [已知类别] 服务栈 → 第 6 类（var:配置静默忽略族，同 6.19） |
| 8 | [Bug] MiMo tool-call streaming: text after the last tool call is silently dropped; split bot-token flushes markup into content | SGLang #33186 | [已知类别·新子签名] 解码/采样 → 第 11 类（流式输出构造层静默丢文本） |
| 9 | [Bug] DeepSeek-V4-Flash-0731: reasoning_effort mapped one level off — `high` is a no-op and vendor `max` is unreachable | SGLang #33185 | [已知类别] 解码/采样 → 第 11 类（配置映射 off-by-one） |
| 10 | Qwen3.5-MoE (Qwen3_5MoeExperts) data-dependent expert dispatch breaks torch.utils.checkpoint reproducibility — CheckpointError / silent NaN gradients | HF transformers #47467 | [已知类别·新子签名] 模型图/架构 → 第 13 类（数据依赖控制流破坏 checkpoint 重算契约 → 静默 NaN 梯度） |
| 11 | config: raise on missing local config file instead of silently falling back to defaults — 缺 config.json 的本地目录返回全默认初始化配置无警告 | HF transformers #47712 | [已知类别] 权重加载 → 第 12 类（var:from_pretrained_silent_random_init 族，配置层静默回退） |
| 12 | [Bugfix] Fix DeepSeek-V4 non-EP TBO for attention TP > 1 (duplicate of #33250, same fix lineage) | SGLang #33217 | 跳过（与 #33250 同族） |
| 13 | BLOCK_SIZE-dependent illegal memory access: crashes at 512/1024, passes at other sizes | triton #11148 | [已知类别·新子签名] 硬件/SDC → 第 1 类（tile 形状触发 codegen 误编译） |
| 14 | Autotuner accepts invalid prune_configs_by["top_k"] values — 0/negative/NaN/inf 各产生不同静默错误行为 | triton #11072 | [已知类别·新子签名] 硬件/SDC → 第 1 类（调优器参数校验缺失 → 静默错误 config 选择） |
| 15 | [AMD] Delegate NaN-propagating min/max to LLVM — AMD 后端为 arith.minimumf/maximumf 注册两个竞争模式 | triton #11118 | 候选弱（语义/性能方向），**本轮列为备选未合并** |
| 16 | `sfpi::lut2`/`sfpi::lut2_sign` use the WH/BH LUT table registers on Quasar, giving silently wrong results | tt-metal #51346 | [已知类别·新子签名] 硬件/SDC → 第 1 类（架构代际寄存器类迁移未跟随） |
| 17 | Fix ttnn.sort wrong indices for bf16 — bf16 8-bit mantissa 把整数 >256 折叠为相等，排序索引非确定 | tt-metal #51562 | [已知类别·新子签名] 硬件/SDC → 第 1 类（低精度比较器折叠整数） |
| 18 | ttnn.fmod / ttnn.remainder (unary scalar): inverted fix-up loop + unreachable guard return wrong-signed out-of-range remainders | tt-metal #51441 | [已知类别] 硬件/SDC → 第 1 类（var:1.18 fmod 同族，unary 标量路径变体） |
| 19 | [ttnn/ccl] reduce_to_root: wrong intermediate tensor spec and a 2-page CB overflow | tt-metal #51235 | [已知类别] 硬件/SDC → 第 1 类（静态审计，CCL CB 溢出） |
| 20 | [ttnn/data_movement] 4 compute-kernel API misuses causing silent wrong data | tt-metal #51215 | 候选弱（静态审计 4 发现），**本轮列为备选未合并** |
| 21 | [ttnn/data_movement] 11 wrong-index / wrong-shape defects causing silent wrong data | tt-metal #51214 | 候选弱（静态审计 11 发现，批量），**本轮列为备选未合并** |
| 22 | [Bugfix] Skip MoE padding across all router paths — VLLM_MOE_SKIP_PADDING 仅在部分 fused top-k 实现，DSV4 Triton/grouped/custom router 可对 padding 行发射真实 expert ID | vLLM #50757 | [已知类别·新子签名] 模型图/架构 → 第 13 类（MoE padding 契约跨 router 实现不一致） |
| 23 | [Bugfix][Frontend] Constrain Anthropic cache_salt to non-empty | vLLM #50764 | 候选弱（前端输入校验），**本轮列为备选未合并** |
| 24 | [Bugfix] Preflight FlashInfer sampler JIT compilability instead of crashing engine warmup | vLLM #50747 | 候选弱（warmup 崩溃，非静默正确性），**本轮列为备选未合并** |

### 跳过（已在主文档 / 重复签名）

| # | 标题 | 原因 |
|---|---|---|
| - | #50671 UVA silent input corruption under GPU CC | 已在主文档（6.18） |
| - | #50649 ROCm Kimi-K3 KDA NaN mixed batch + autotune race | 已在主文档（6.21） |
| - | #50603 RDNA3 first-call non-determinism + long-seq corruption | 已与 14.6 同族 |
| - | #50486 Warn when --block-size silently discarded | 已与 6.19 配置静默忽略族相关，弱 |
| - | #50427 FlexAttention paged K/V offsets overflow int32 | 已在主文档（K3.10） |
| - | #50477 gemma4 parser silently ignores forced tool_choice | 前端解析，弱 |
| - | #50292 VLLM_ATTENTION_BACKEND silently ignored | 已在主文档（6.19） |
| - | #50201 Harden top_k_per_row against NaN | 已在主文档（6.22） |
| - | #50183 NaN handling in rejection sampler tl.argmax | 与 11.x NaN 采样族相关，已覆盖 |
| - | #50084 NVFP4 KV block scale swizzle mismatch | 已在主文档（7.9） |
| - | #50067 EngineCore dies silently during warmup GB10 | 已在主文档（6.15） |
| - | #49924 GDN attention silently corrupts memory under load XPU | 已在主文档（6.20） |
| - | #33187 SGLANG_SANITIZE_NAN_LOGITS | 已在主文档（11.7） |
| - | #33106 SGLang deepep uninitialized rows | 已在主文档（6.16） |
| - | #32968 SGLang kimi-k3 [PAD] storm NaN | 已在主文档 |
| - | #32962 SGLang EPLB rank map corruption | 已在主文档（5.7） |
| - | #32922 SGLang ROCm spec decode verify silent greedy | 已在主文档（11.8） |
| - | #32807 SGLang Gemma-3 RMSNorm high-rank skip | 已在主文档（13.13） |
| - | #32736 compressed-tensors mixed-precision silently loads unquantized | 与 12.9 同族（静默随机初始化），弱变体 |
| - | #32459 SGLang EAGLE spec decode defeats radix reuse | 已在主文档（8.8） |
| - | #17158/#17159 TRT-LLM SamplingParams NaN | 已在主文档（11.9） |
| - | #17020 trtllm-eval aime silently disable thinking | 已在主文档（10.6） |
| - | #47496 DataCollatorWithFlattening 0-based position_ids | 已在主文档（9.6） |
| - | #47475 Zamba2 causality violation | 已在主文档（13.8） |
| - | #47405 Silent conversion skip model_type mismatch | 已在主文档（12.8） |
| - | #47407 CompressedTensors run_compressed=False silent random init | 已在主文档（12.9） |
| - | #11111 triton fp8 e5m2 transposed store | 已在主文档（1.12） |
| - | #11065 inline_asm_elementwise replicated elements | 已在主文档（1.17 同族） |
| - | #51661 ttnn.fmod out of range wrong sign | 已在主文档（1.18） |
| - | #51557 bf16 loses NaN encodings | 已在主文档（1.10 同族） |
| - | #51540/#51537 SFPU isclose Inf/NaN bitpattern | 已在主文档（1.8 同族） |
| - | #51503 DRAM ND-sharded duplicate x-coords | 已在主文档（1.15） |
| - | #51470 ttnn.maximum/minimum/clamp drop NaN | 已在主文档（1.14） |
| - | #51284 ttnn transformer 5 defects rotary_embedding | 已在主文档（13.14） |
| - | #51238 active_erisc byte vs word count overcopy | 已在主文档（1.19） |
| - | #51233 ttnn/embedding 4 defects | 备选未合并（2026-08-01），仍弱 |

### arxiv 论文（去重后）

| # | 标题 | arxiv | 分类标注 |
|---|---|---|---|
| P1 | Continuous Discovery of Vulnerabilities in LLM Serving Systems with Fuzzing | 2605.11202（2026-05-11） | [已知类别] 服务栈模糊测试框架，方法论而非具体故障机制 → **记入报告不合并**（可作为服务栈类的方法论补充引用） |
| P2 | Agent-Assisted Side-Channel Attacks on Non-Prefix KV Cache in RAG | 2606.21842（2026-06-20） | [已知类别] KV 缓存侧信道 → 与 K4.x cross-request / 2.1 rowhammer 同族，**记入报告不合并**（攻击场景，非推理正确性 bug） |
| P3 | Integrity of peer-to-peer distributed LLM inference under malicious nodes | 2607.19490（2026-07-21） | [非推理正确性] P2P 完整性，**记入报告不合并** |
| P4 | LLM-PRISM: Characterizing SDC from Permanent GPU Faults in LLM Training | 2604.10390（2026-04-12） | [训练场景] **记入报告不合并** |
| P5 | Decision-Level Hijacking via Bit-Flip Attacks | 2607.25227 | 已在主文档（3.7） |

---

## 本轮合并决策

**合并进主文档的新条目：15 条**（均为已知类别的新实例，含 6 个新子签名）：

- **第 1 类（硬件/SDC）+4**：`triton_blocksize_dependent_ima_codegen`（#11148）、`triton_autotuner_invalid_topk_silent_config_selection`（#11072）、`ttmetal_quasar_lut2_wrong_lreg_class`（#51346）、`ttmetal_sort_bf16_integer_collapse_nondeterministic_index`（#51562）
- **第 5 类（分布式/多GPU）+2**：`sglang_multinode_tp_rank_divergence_nccl_deadlock`（#33289）、`sglang_dsv4_nonep_tbo_attn_tp_replica_misshard`（#33250）
- **第 6 类（服务栈）+3**：`sglang_hicache_storage_key_missing_kv_dtype`（#33268）、`vllm_pd_nixlpush_failed_write_notification_undercount`（#50754）、`vllm_override_generation_config_silent_param_drop`（#50767，var:配置静默忽略族）
- **第 7 类（量化）+1**：`sglang_dsv4_fused_wqa_wkv_packed_weight_silent_drop`（#33245，var:quantized_kernel_bias_omission 族）
- **第 11 类（解码/采样）+2**：`sglang_mimo_toolcall_streaming_silent_text_drop`（#33186）、`sglang_dsv4_flash_reasoning_effort_off_by_one`（#33185）
- **第 12 类（权重加载）+1**：`hf_missing_local_config_silent_default_init`（#47712，var:from_pretrained_silent_random_init 族）
- **第 13 类（模型图/架构）+2**：`vllm_rocm_kimi_k3_correction_bias_fp32_downcast_on_load`（#50761）、`hf_qwen35moe_data_dependent_dispatch_breaks_checkpoint_repro`（#47467）。另将 `vllm_moe_skip_padding_router_inconsistency`（#50757）作为第 13 类的补充（MoE padding 契约）。

**备选未合并**：#11118（NaN min/max LLVM 委托，语义方向）、#51215/#51214（静态审计批量数据移动缺陷）、#50764（cache_salt 非空校验）、#50747（FlashInfer JIT warmup 崩溃）、#33217（与 #33250 同族）、#32736（与 12.9 同族变体）。

合并后主文档：**280 → 296 条**（+16；含 #50757 作为第 13 类补充条目，共 16 条而非 15，因 #50757 与 #47467/#50761 同归第 13 类）。
