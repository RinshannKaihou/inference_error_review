# LLM 推理故障每日扫描报告 — 2026-07-31

- **扫描日期**：2026-07-31（UTC）
- **滚动窗口**：GitHub Issues 过去 7 天（created:>2026-07-24）+ 标题全文检索；arxiv 最近论文
- **主文档基线**：合并前 **250 条**（A1–A17 基础原语 17 + 第 1–15 类扩展目录 119 + K1.1–K10.31 KV 扩展 114）

---

## API 调用记录

### GitHub Search API（10 次查询，均成功，未触发 403 限流）

| # | 查询 | total_count |
|---|---|---|
| Q1 | `vllm-project/vllm` label:bug created:>2026-07-24 | 56 |
| Q2 | `vllm-project/vllm` wrong/corruption/NaN in:title | 279 |
| Q3 | `sgl-project/sglang` correctness/wrong/corrupt/NaN in:title | 108 |
| Q4 | `NVIDIA/TensorRT-LLM` wrong/corruption/NaN/correctness in:title | 26 |
| Q5 | `huggingface/transformers` wrong/NaN/corrupt/correctness in:title created:>2026-07-15 | 5 |
| Q6 | `triton-lang/triton` wrong/numerical/NaN/corrupt/incorrect in:title | 125 |
| Q7 | `tenstorrent/tt-metal` wrong/NaN/correctness/numerical/corrupt in:title | 229 |
| Q8 | 全站 "silent data corruption" LLM | 472 |
| Q9 | 全站 "bit flip" LLM weights | 55 |
| Q10 | 全站 "KV cache" corrupt/incorrect | 93875 |

### arxiv API（6 次查询，`curl -sL` 跟随 301 重定向）

| # | 关键词 | 返回 |
|---|---|---|
| A1 | "silent data corruption" AND LLM | 3（最新 2026-04） |
| A2 | "bit flip" AND LLM | 5（最新 2026-07-28） |
| A3 | "KV cache" AND corruption AND inference | 4 |
| A4 | "LLM inference" AND correctness | 8 |
| A5 | "GPU soft error" AND learning | 0 |
| A6 | quantization AND "wrong result" AND LLM | 0 |

---

## 扫描结果（逐条标注）

### 🆕 新故障模式 / 新签名

无完全全新的故障领域类别。本轮发现均为**已知类别的新实例**，但其中若干揭示了具体的新子签名（见"已知类别新案例"中的 `[新子签名]` 标注）。

### 🔄 已知类别的新案例（去重后确认不在主文档）

| # | 标题 | 来源 | 分类标注 |
|---|---|---|---|
| 1 | gfx1100 (RDNA3): first-call non-determinism + long-seq corruption from Triton paged-attention fallback (ROCm 7.14) | vLLM #50603 | [已知类别] 服务栈/注意力 — **已与 14.6 `rdna_mamba_paged_attention_fallback_collapse` 同族**，跳过（重复签名） |
| 2 | AxionML/Gemma-4-12B-NVFP4 fails with shape mismatch in flashinfer_scaled_fp4_mm on RTX 5060 Ti (sm_120) | vLLM #50577 | [已知类别] 量化 → 第 7 类 |
| 3 | MiniMax-M3-NVFP4 illegal memory access | vLLM #50557 | [已知类别] 量化 → 第 7 类 |
| 4 | FlexAttention paged K/V offsets overflow int32 — crash *or* silent wrong output | vLLM #50427 | [已知类别] **已在主文档**，跳过（重复） |
| 5 | DeepSeek-V4 on SM12x: NaN MQA logits drive top_k_per_row_prefill to emit uninitialized smem as indices → illegal memory access | vLLM #49896 | [已知类别] 服务栈 → 第 6 类 |
| 6 | MRotaryEmbedding Triton kernel hardcodes Neox-style rotation, producing wrong results for GPT-J style models (GLM-OCR) | vLLM #49290 | [已知类别·新子签名] RoPE 旋转约定 → 第 9 类 |
| 7 | Uninitialized rows reach the block-FP8 GEMMs with --moe-a2a-backend deepep (masked by DeepGEMM, NaN with flashinfer) | SGLang #33106 | [已知类别] 服务栈 → 第 6 类 |
| 8 | [Gemma-3] RMSNorm: higher-rank q_norm/k_norm skip the fused CUDA kernel, and mixed-dtype weights return NaNs | SGLang #32807 | [已知类别·新子签名] 模型图/架构 → 第 13 类 |
| 9 | Silent conversion skip on model_type mismatch: loading a converted checkpoint via the wrong Auto class yields randomly-initialized weights that generate fluently | HF transformers #47405 | [已知类别] 权重加载 → 第 12 类（var:from_pretrained_silent_random_init） |
| 10 | ttnn.maximum / ttnn.minimum / ttnn.clamp / ttnn.hardtanh silently drop NaN | tt-metal #51470 | [已知类别·新子签名] 硬件/SDC（NaN 传播） → 第 1 类 |
| 11 | DRAM ND-sharded buffer corrupts data when shard grid has duplicate x-coords | tt-metal #51503 | [已知类别] 硬件/SDC（数据损坏） → 第 1 类 |
| 12 | Two `tl.dot` chained through one accumulator in a K-loop compute wrong results on sm_103 (B300) | triton #10821 | [已知类别] 硬件/SDC（内核数值正确性） → 第 1 类 |
| 13 | Tensor-descriptor gather/load × tl.dot produces incorrect results on Blackwell when K is not a multiple of BLOCK_SIZE_K | triton #10927 | [已知类别] 硬件/SDC — 候选，**本轮列为备选未合并**（与 #10821 同族，仅合并代表性一条） |
| 14 | `sfpi::lut2`/`sfpi::lut2_sign` use the WH/BH LUT table registers on Quasar, giving silently wrong results | tt-metal #51346 | [已知类别] 硬件/SDC — 候选，**本轮列为备选未合并**（架构特定，代表性弱于 #51470/#51503） |

### arxiv 论文（去重后）

| # | 标题 | arxiv | 分类标注 |
|---|---|---|---|
| P1 | From Bit-Position Sensitivity to Unequal Error Protection for DNN Inference Memory | 2607.19623（2026-07-21） | [已知类别] 比特翻转/SDC → 第 3 类（BFA 家族补充来源） |
| P2 | Decision-Level Hijacking: Injecting Cognitive Bias into LLMs via Bit-Flip Attacks | 2607.25227（2026-07-28） | **已在主文档**（3.7 `bfa_decision_level_cognitive_hijacking`），跳过 |
| P3 | Bit-Flip Vulnerability of Shared KV-Cache Blocks in LLM Serving Systems | 2604.17249 | **已在主文档**（2 处引用），跳过 |
| P4 | Continuous Discovery of Vulnerabilities in LLM Serving Systems with Fuzzing | 2605.11202 | [方法论参考] 非具体故障机制，**记入报告不合并** |
| P5 | Integrity of peer-to-peer distributed LLM inference under malicious nodes | 2607.19490 | [安全方向] 非静默正确性故障，**记入报告不合并** |

---

## 本轮合并决策

**合并进主文档的新条目：11 条**（均为已知类别的新实例，含 3 个新子签名）：

- 第 1 类（硬件/SDC）+3：`ttmetal_sfpswap_nan_drop`、`ttmetal_dram_sharded_dup_xcoord_corruption`、`triton_dual_dot_single_accumulator_sm103`
- 第 3 类（BFA）+1：`bit_position_sensitivity_unequal_error_protection`（arxiv 补充来源）
- 第 6 类（服务栈）+2：`sglang_deepep_uninitialized_rows_fp8_gemm_nan`、`vllm_dsv4_sm12x_nan_mqa_topk_illegal_access`
- 第 7 类（量化）+2：`vllm_gemma4_nvfp4_fp4mm_shape_mismatch_sm120`、`vllm_minimax_m3_nvfp4_illegal_mem_access`
- 第 9 类（RoPE）+1：`vllm_mrotary_triton_neox_hardcode_gptj_wrong`
- 第 12 类（权重加载）+1：`hf_silent_conversion_skip_model_type_mismatch_random_init`
- 第 13 类（模型图/架构）+1：`sglang_gemma3_rmsnorm_highrank_skip_fused_mixed_dtype_nan`

**跳过（重复/已在主文档）**：#50427、arxiv 2607.25227（=3.7）、arxiv 2604.17249（已引用 2 处）、#50603（与 14.6 同族）。

合并后主文档：**250 → 261 条**。
