# LLM 推理故障每日扫描报告 — 2026-08-04

- **扫描日期**：2026-08-04（覆盖 2026-07-28 ~ 2026-08-04 的 7 天滚动窗口）
- **数据源**：GitHub Search API（6 个目标仓库 + 4 组跨仓库关键词，共 10 次查询）+ arXiv API（5 组关键词）
- **主文档基准**：扫描前 307 条（第 0 层 17 + 第 1 层 176 + 第 2 层 114）

---

## 一、GitHub Issues / PRs 发现

### 1.1 vLLM（vllm-project/vllm）

#### `[已知类别·新实例]` #51049 — `fused_qk_norm_rope` multi-head-per-warp kernel corrupts partial-NeoX QK-Norm models on SM90
- **链接**：https://github.com/vllm-project/vllm/issues/51049
- **日期**：2026-08-04 open
- **简述**：vLLM 0.23.0/main，H200 SM90。`fused_qk_norm_rope` Inductor fusion kernel（`fusedQKNormRopeKernelNTokenHeads`）对 **partial-NeoX**（`rotary_dim < head_dim`）模型（如 `inclusionAI/Ling-mini-2.0`，`head_dim=128, rotary_dim=64`）的除每 warp chunk 第一个 head 之外的全部 head 产生错误 Q/K 值；`token_heads_per_warp=8` 时产生 NaN。端到端生成完全崩溃。
- **归类**：位置编码/RoPE 领域。**但**：主文档 9.8 `vllm_fused_qknorm_rope_divergent_warp_collective_partial_neox`（2026-08-03 扫描，PR #50903）已编目**完全同一故障签名**（partial-NeoX + divergent warp collective UB，同一 kernel `fusedQKNormRopeKernelNTokenHeads`，同 SM90 现象）。#51049 是该 bug 的 **issue 形态**（用户报告），与 #50903 PR 同一根因。**判定：重复（同一根因的 issue/PR 对），不新增**。
- **去重**：根因 = 9.8。issue 编号 51049 在主文档中未出现，但机制与 #50903 完全相同 → 合并为 9.8 的补充来源。

#### `[新签名]` #51063 — Composite VLM wrapper (Mistral3ForConditionalGeneration) resolves `tie_word_embeddings` from wrong (top-level) config, silently discarding real `lm_head.weight`
- **链接**：https://github.com/vllm-project/vllm/issues/51063
- **日期**：2026-08-04 open
- **简述**：vLLM 的复合 VLM wrapper `Mistral3ForConditionalGeneration` 从**顶层 config** 解析 `tie_word_embeddings`，而非从语言模型子 config。当顶层错误地为 `true` 时，真实的 `lm_head.weight` 被**静默丢弃**，模型用随机/tied embedding 当 lm_head → 产生"词表对得上但语义胡言乱语"的输出（coherent-vocabulary-but-incoherent output）。
- **归类**：权重/适配器/检查点加载故障（第 12 类）。机制：config 层级解析错误 → 真实权重静默丢弃。与 12.3 `tied_embedding_random_lm_head`（随机 lm_head）同族但根因不同（不是 tied 逻辑本身错，而是**复合模型 config 解析层级错误**导致 tied 标志被误取）。**判定：新子签名，新增**。

#### `[新签名]` #51039 — Kimi-K3: all requests degenerate to repeated token after long-context prefill (NaN logits; packed KDA prefill suspected)
- **链接**：https://github.com/vllm-project/vllm/issues/51039
- **日期**：2026-08-04 open
- **简述**：vLLM 0.26.0 serving Kimi-K3，长上下文（~240K，远在 1M spec 内）预填充后，**所有后续请求**退化为每步返回单一重复退化 token（pad token id=0），永不出 EOS，每个请求跑到 max_tokens。已证明 logits 为 NaN（argmax over NaN → index 0）。**感染性**：损坏在请求间传播且永久（2→8 并发增长，持续 ~19 小时直到进程重启）。根因怀疑：packed KDA prefill 将 decode 与 prefill 打包进单次 `_flashkda_prefill` kernel launch，NaN 中毒的运行请求通过共享 kernel launch + state cache writeback 感染新请求。
- **归类**：服务栈/缓存正确性（第 6 类）或 KV缓存/注意力（第 8 类）。这是**跨请求感染 + NaN 传播**的新签名：packed prefill kernel 的跨序列 KV 状态污染。**判定：新子签名，新增到第 6 类**（服务栈正确性，因为根因在 vLLM 的 KDA 调度路径而非 KV cache 数据结构本身）。

#### `[已知类别·新实例]` #51009 — DSpark speculative decoding acceptance collapses after position 0 on DeepSeek-V4-Flash-0731
- **链接**：https://github.com/vllm-project/vllm/issues/51009
- **日期**：2026-08-04 open
- **简述**：vLLM 0.26.1rc1 nightly，DeepSeek-V4-Flash-0731，2× DGX Spark（GB10 SM121a），DSPark 推测解码：position 0 接受率正常（85.7%），但 position 1+ 链式崩溃（conditional ~25%），平均接受长度仅 1.46。目标模型单独输出正确。怀疑 markov head 在 position 1+ 的 `_sample_sequential` / `map_draft_to_target` 映射问题，或 MTP 权重命名不匹配。
- **归类**：解码/采样（第 11 类），推测解码正确性。与 K8.12 `specdec_vllm_short_conv_lfm2_fix`（推测解码 + 短上下文）同域但不同模型/栈。**判定：新子签名（推测解码接受率 position-0 后链式崩溃），新增**。

#### `[已知类别·新实例]` #51038 — Fix MXFP4 conversion for FlashInfer CUTLASS
- **链接**：https://github.com/vllm-project/vllm/pull/51038
- **日期**：2026-08-04 open
- **简述**：MXFP4 backend selector 选择 FlashInfer CUTLASS 时，`convert_weight_to_mxfp4_moe_kernel_format` 缺对应分支 → 模型加载后 ValueError。修复：重排 fused gate/up 权重从 [w1;w3] 到 FlashInfer CUTLASS 期望的 [w3;w1] layout。
- **归类**：量化（第 7 类），权重 layout 转换 bug。属量化 kernel 的权重 layout 不匹配 → 加载失败（crash，非静默）。**判定：与 7.17 `sglang_dsv4_fused_wqa_wkv_packed_weight_silent_drop`（packed weight 静默丢失）同域但不同机制（这个是 crash 非 silent）。新子签名，新增**。

#### `[已知类别·新实例]` #51042 — Propagate decode SWA index width for DSpark + FLASHINFER_MLA_SPARSE_DSV4
- **链接**：https://github.com/vllm-project/vllm/pull/51042
- **日期**：2026-08-04 open
- **简述**：DSpark non-causal draft batch 用 `noncausal_index_width`（比 `window_size` 宽）分配 `decode_swa_indices`，但 FlashInfer DSV4 sparse MLA 仍传 `window_size` → reshape 维度不匹配 crash（`shape [7,128] invalid for input of size 1792`）。
- **归类**：量化/注意力 kernel 接口（第 8 类 KV/注意力）。属 SWA 索引宽度在 draft 与 FlashInfer 之间不一致。**判定：偏 crash bug（非静默正确性），但涉及 MLA sparse 注意力索引。新实例，新增**。

#### `[已知类别·新实例]` #50987 — Fix fused MTP RMSNorm dtype mismatch (DeepSeek V4)
- **链接**：https://github.com/vllm-project/vllm/pull/50987
- **日期**：2026-08-04 closed
- **简述**：fused MTP RMSNorm 的两个运行时分支给 `x` 推断出不同 dtype：masked `tl.where` 路径提升到 FP32，hnorm 路径保留输入 dtype → Triton 要求合并的 local variable 类型一致 → kernel 编译失败。
- **归类**：模型图/架构（第 13 类）或量化 kernel。属 Triton fused kernel 的 dtype 一致性 bug。**判定：编译期 crash（非静默），但属 fused kernel dtype 正确性。新实例，新增**。

### 1.2 SGLang（sgl-project/sglang）

#### `[新签名]` #33579 — DSPARK/DFLASH spec v2 at page_size=1: req_to_token row too narrow — silent KV slot leak at context boundary
- **链接**：https://github.com/sgl-project/sglang/issues/33579
- **日期**：2026-08-04 open
- **简述**：DFLASH-family（含 DSPARK）spec v2 在每步为每请求预留 `committed + 2*num_draft_tokens` KV slot（2x 为 overlap 双缓冲），但 `req_to_token` 行仅在 `page_size>1` 时才按该预留量分配。`page_size=1` 时行宽 = `context_len + 4 + num_draft_tokens`，小于预留水位。DSPARK gamma=16（`num_draft_tokens=17`）：行余量=21，每步预留超出量=34。当请求接近上下文边界（`committed > context_len - 13`）时：(1) 行写入**溢出到邻居请求的行**；(2) release slice 在行尾静默 clamp，溢出的尾部 slot 永不释放 → **永久泄漏**（观察到每个边界命中请求泄漏 11-27 slot），邻居行头部被覆盖（二阶损坏/double-free 风险）。症状延迟且混乱：长 eval 在上下文限截断时正常完成，随后空闲自检 kill 服务器报 `token_to_kv_` 错误。
- **归类**：KV缓存/注意力（第 8 类）或服务栈（第 6 类）。属 **page_size=1 下 KV slot 预留水位 > 行宽 → 跨请求行溢出 + 静默泄漏**的新签名。**判定：新子签名，新增到第 6 类**（SGLang 服务栈正确性，因为根因在调度器的 slot 预留逻辑）。

#### `[已知类别·新实例]` #33522 — Fast Engine Recovery: Weight Cache Daemon (roadmap)
- **链接**：https://github.com/sgl-project/sglang/issues/33522
- **日期**：2026-08-04 open
- **简述**：Roadmap issue。Phase 1 已落地 per-rank daemon 持有 post-quantized weights via CUDA IPC。提到了 parity harness 需求（disk-load vs daemon-load state_dict diff）——隐含的正确性风险是 **weight cache daemon 加载的权重与 disk 加载的不一致**。但本身不是已报告的正确性 bug，是 roadmap。
- **归类**：权重/检查点加载（第 12 类）。**判定：roadmap 非已发 bug，不新增为主条目，但可作为风险监控记录**。

### 1.3 Tenstorrent tt-metal

#### `[新签名]` #51979 — softsign returns 0 instead of ±1 for large magnitudes (SFPU reciprocal underflow)
- **链接**：https://github.com/tenstorrent/tt-metal/pull/51979
- **日期**：2026-08-03 open
- **简述**：`calculate_softsign()` 计算 `x/(1+|x|)` 时用 `v * reciprocal(1+|v|)`。当 `1/(1+|v|)` 低于最小正常数时被 flush-to-zero，乘积坍缩为 0——而正确答案在该区域是 ±1。Wormhole 从 ~1.06e38 起丢失，Blackhole 早一个 binade（~8.51e37）。现有测试用标准正态分布采样，永远达不到这些量级 → 测试盲区。
- **归类**：硬件/SDC（第 1 类），SFPU 数值正确性。与 1.11 `sfpu_exp_overflow_guard_missing`（SFPU exp 溢出缺少 guard）同族。**判定：新子签名（SFPU reciprocal 中间值下溢 → activation 静默错误），新增**。

#### `[新签名]` #52107 — overflow-safe logaddexp/logaddexp2 via native SFPU kernel
- **链接**：https://github.com/tenstorrent/tt-metal/pull/52107
- **日期**：2026-08-04 open
- **简述**：`ttnn.logaddexp`/`logaddexp2` 对大输入静默返回 `inf`，因为 naive 分解 `log(exp(a)+exp(b))` 在 |a|或|b|>~88.7 时 `exp` 溢出。修复用数值稳定恒等式 `max(a,b)+log(1+exp(-|a-b|))`。
- **归类**：硬件/SDC（第 1 类），SFPU 数值正确性。**判定：新子签名（naive exp decomposition 溢出 → 静默 inf），新增**。

#### `[已知类别·新实例]` #52024 — Fix SFPU RNG correlation and improve FP32 uniform random quality
- **链接**：https://github.com/tenstorrent/tt-metal/pull/52024
- **日期**：2026-08-03 open（closes #52014, #50487）
- **简述**：`ttnn.rand`/`ttnn.uniform` 此前在硬件 PRNG lane 与相邻 per-core seed 间暴露**相关性**，可产生 seed 依赖的样本均值偏移，远超预期 IID 采样误差。修复混合 xorshift32 lane salt + ARX 置换 + Weyl 乘子扩散。
- **归类**：硬件/SDC（第 1 类），RNG 质量。**判定：新子签名（SFPU RNG lane/seed 相关性 → 非均匀采样），新增**。

### 1.4 NVIDIA TensorRT-LLM

#### `[新签名]` #17264 — Fix one-model MTP KV cache accounting
- **链接**：https://github.com/NVIDIA/TensorRT-LLM/pull/17264
- **日期**：2026-08-04 open
- **简述**：one-model MTP（multi-token prediction）推测解码的第一个 overlapped generation step 会**高估其 cached length** 并访问**无效 FlashInfer page** → CUDA illegal memory access。KV scheduler 也缺 Python 侧 draft-token count → 容量规划错误。
- **归类**：KV缓存/注意力（第 8 类）或解码/采样（第 11 类）。属推测解码 MTP 的 KV cache 计数错误 → 非法内存访问。**判定：新子签名（one-model MTP 首 step KV cache 高估 → 非法页访问），新增**。

### 1.5 Triton / Transformers

- **triton #11091**（2026-07-29）：cubin line-info 对 source mtime 敏感 → 可复现构建问题。属构建工具链，非推理正确性。**不新增**。
- **transformers #47645**：清理 `_no_split_modules` 条目。非正确性 bug。**不新增**。

---

## 二、arXiv 论文发现

| # | 标题 | arXiv ID | 日期 | 状态 |
|---|------|----------|------|------|
| A1 | Decision-Level Hijacking: Injecting Cognitive Bias into LLMs via Bit-Flip Attacks | 2607.25227 | 2026-07-28 | **已在主文档** 3.7 `bfa_decision_level_cognitive_hijacking`（2026-07-30 扫描合并） |
| A2 | Bit-Flip Vulnerability of Shared KV-Cache Blocks in LLM Serving Systems | 2604.17249 | 2026-04-19 | **查重**：主文档 K4.9 `prefix_cache_shared_block_bitflip_silent_divergence` 覆盖共享 KV cache 块的 bitflip。需确认是否同一论文 |
| A3 | Rotated Robustness: Training-Free Defense against Bit-Flip Attacks on LLMs | 2603.16382 | 2026-03-17 | 防御论文，非新故障模式。**不新增** |
| A4 | TFL: Targeted Bit-Flip Attack on Large Language Model | 2602.17837 | 2026-02-19 | **查重**：主文档 3.x BFA 家族已覆盖靶向 bitflip。需确认 |
| A5 | Continuous Discovery of Vulnerabilities in LLM Serving Systems with Fuzzing | 2605.11202 | 2026-05-11 | **新**：系统化 fuzzing 发现 LLM serving 漏洞的方法论。可作为第 6 类的补充来源 |
| A6 | Hollow-LLM Attack: Computationally Trivial Weights in Zero-Knowledge Verification of LLM Inference | 2607.28884 | 2026-07-30 | **新签名**：ZK 验证中用计算平凡的权重欺骗验证者——推理完整性攻击的新角度 |

### arXiv 查重结果

- **A1 (2607.25227)**：已在主文档 3.7。**跳过**。
- **A2 (2604.17249)**：检查主文档 K4.9 `prefix_cache_shared_block_bitflip_silent_divergence`。该条来源标注为 arXiv:2604.17249v2 → **已在主文档**。**跳过**。
- **A4 (2602.17837 TFL)**：检查主文档第 3 类 BFA 家族。搜索 `TFL`/`2602.17837` → 未命中。但第 3 类已有 8 条 BFA 变体（含靶向 trojan、EOS 抑制、cognitive hijacking）。TFL 是又一靶向 bitflip 变体。**判定：已知类别新实例，可新增为 3.9**。
- **A5 (2605.11202)**：fuzzing 方法论论文，非单一故障。**作为第 6 类补充来源引用，不单独新增**。
- **A6 (2607.28884)**：ZK 推理验证攻击，属完整性/可信推理新域。主文档无覆盖。**判定：新签名，但属"验证层"而非"推理执行层"故障——编目为第 13 类（模型图/架构）的新条目或附录风险记录**。鉴于其偏离本项目"推理执行正确性"核心范围，**作为附录 C 盲点补充而非主条目**。

---

## 三、分类汇总

### 确认新增的条目（9 条）

| 序号 | 来源 | 故障名 | 目标类别 | 类型 |
|------|------|--------|----------|------|
| 1 | vLLM #51063 | `vllm_composite_vlm_tie_word_embeddings_wrong_config_level` | 第 12 类 | 新签名 |
| 2 | vLLM #51039 | `vllm_kimi_k3_packed_kda_prefill_nan_cross_request_infection` | 第 6 类 | 新签名 |
| 3 | vLLM #51009 | `vllm_dspark_acceptance_collapse_after_pos0_dsv4_flash` | 第 11 类 | 新签名 |
| 4 | vLLM #51038 | `vllm_mxfp4_flashinfer_cutlass_weight_layout_w1w3_swap_crash` | 第 7 类 | 新签名 |
| 5 | vLLM #51042 | `vllm_dspark_swa_index_width_flashinfer_mla_sparse_crash` | 第 8 类 | 新签名 |
| 6 | vLLM #50987 | `vllm_fused_mtp_rmsnorm_branch_dtype_mismatch_compile_fail` | 第 13 类 | 新签名 |
| 7 | SGLang #33579 | `sglang_dspark_pagesize1_req_to_token_row_overflow_leak` | 第 6 类 | 新签名 |
| 8 | tt-metal #51979 | `ttmetal_sfpu_softsign_reciprocal_underflow_zero` | 第 1 类 | 新签名 |
| 9 | tt-metal #52107 | `ttmetal_logaddexp_naive_exp_decomposition_overflow_inf` | 第 1 类 | 新签名 |
| 10 | tt-metal #52024 | `ttmetal_sfpu_rng_lane_seed_correlation_nonuniform` | 第 1 类 | 新签名 |
| 11 | TRT-LLM #17264 | `trtllm_onemodel_mtp_kv_cache_overcount_illegal_page_access` | 第 8 类 | 新签名 |
| 12 | arXiv 2602.17837 | `tfl_targeted_bit_flip_attack_llm` | 第 3 类 | 已知类别新实例 |

**实际新增：12 条**（修正上表序号）

### 跳过的条目

| 来源 | 原因 |
|------|------|
| vLLM #51049 | 根因 = 主文档 9.8（#50903），同一故障的 issue 形态 |
| arXiv 2607.25227 | 已在主文档 3.7 |
| arXiv 2604.17249 | 已在主文档 K4.9 |
| arXiv 2603.16382 | 防御论文，非故障 |
| arXiv 2605.11202 | 方法论，非单一故障 |
| arXiv 2607.28884 | ZK 验证层，偏离推理执行核心范围 |
| SGLang #33522 | Roadmap，非已发 bug |
| triton #11091 | 构建工具链，非推理正确性 |
| transformers #47645 | 清理 PR，非 bug |

---

## 四、API 调用记录

| # | 查询 | 结果数 | 状态 |
|---|------|--------|------|
| 1 | `repo:vllm-project/vllm label:bug created:>2026-07-28` | 15 | OK |
| 2 | `repo:sgl-project/sglang label:bug created:>2026-07-28` | 4 | OK |
| 3 | `repo:huggingface/transformers wrong inference created:>2026-07-28` | 1 | OK |
| 4 | `repo:NVIDIA/TensorRT-LLM correctness created:>2026-07-28` | 15 | OK |
| 5 | `repo:triton-lang/triton correctness created:>2026-07-28` | 3 | OK |
| 6 | `repo:tenstorrent/tt-metal correctness created:>2026-07-28` | 15 | OK |
| 7 | `silent data corruption OR NaN inference created:>2026-07-28` | 15（噪声高） | OK |
| 8 | `KV cache corruption created:>2026-07-28` | 15 | OK |
| 9 | `quantization wrong result OR bit flip created:>2026-07-28` | 15 | OK |
| 10 | `repo:vllm-project/vllm numerical OR non-deterministic created:>2026-07-28` | 15 | OK |
| 11 | arXiv: silent data corruption AND LLM | 3 | OK |
| 12 | arXiv: bit flip AND LLM AND weight | 5 | OK |
| 13 | arXiv: LLM inference AND correctness | 5 | OK |
| 14 | arXiv: KV cache AND corruption AND inference | 5 | OK |

**速率限制**：无 HTTP 403。GitHub 查询间隔 ≥7s。
