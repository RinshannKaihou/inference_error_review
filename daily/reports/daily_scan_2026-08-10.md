# LLM 推理故障每日扫描报告 — 2026-08-10

**扫描日期**：2026-08-10（UTC）
**搜索窗口**：GitHub `created:>2026-08-03`（过去 7 天滚动窗口）+ arxiv 近期
**主参考文档基线**：423 条（合并前）

---

## 扫描范围与 API 调用记录

| # | 目标 | 查询 | 结果数 |
|---|------|------|--------|
| 1 | vLLM | `correctness created:>2026-08-03` | 97 |
| 2 | vLLM | `wrong output created:>2026-08-03 label:bug` | 19 |
| 3 | vLLM | `NaN inference created:>2026-08-03 label:bug` | 2 |
| 4 | SGLang | `correctness created:>2026-08-03 label:bug` | 1 |
| 5 | SGLang | `non-deterministic created:>2026-08-03` | 25 |
| 6 | transformers | `RoPE OR tokenizer OR rotary created:>2026-08-03 label:bug` | 2 |
| 7 | TRT-LLM | `correctness OR wrong created:>2026-08-03` | 105 |
| 8 | triton | `correctness OR NaN OR numerical created:>2026-08-03` | 4 |
| 9 | tt-metal | `correctness OR numerical created:>2026-08-03` | 133 |
| 10 | tt-metal | `quantization OR SDC OR precision created:>2026-08-03` | 44 |
| A1 | arxiv | `bit flip LLM inference` | 5 |
| A2 | arxiv | `inference correctness` | 3（rate-limited，部分超时） |
| A3 | arxiv | `silent data corruption GPU` | rate-limited（503/超时） |

> **arxiv 限速说明**：export.arxiv.org 本次扫描期间持续返回 `Rate exceeded`（HTTP 503），多次重试后仅成功 2 次查询。未成功的查询标记为 rate-limited。

---

## 扫描发现汇总

### 新故障模式（18 条新合并进主文档）

#### 第 1 类 · 硬件 / SDC（+2）

**1. Triton #11217 — `[新签名]` AMD bfloat16 multiply rounds toward zero instead of to nearest even**
- **链接**：https://github.com/triton-lang/triton/issues/11217
- **来源**：triton-lang/triton
- **描述**：AMD 后端的 `bfloat16` multiply 被降低为 `v_dot2_bf16_bf16`（第二 lane 置零），该指令在 f32 中累加但不执行 bf16 舍入步骤，导致低位 mantissa bits 被截断。结果与截断（truncation）100% 匹配，与 round-to-nearest-even 仅 52.58% 匹配。系统性舍入模式差异（非噪声），会累积，导致 AMD 结果与 torch/CUDA 后端分叉。在 RoPE kernel 中以 bf16 频率计算时，~41% 输出超出 rtol=1e-3。

**2. tt-metal #52735 — `[已知类别]` silu ignores caller's APPROXIMATION_MODE**
- **链接**：https://github.com/tenstorrent/tt-metal/pull/52735
- **来源**：tenstorrent/tt-metal
- **描述**：`calculate_silu` 是 llk_sfpu/ 中唯一不声明 `APPROXIMATION_MODE` 的激活函数。调用者设置 `math_approx_mode=true` 时仍获得精确 sigmoid 而非近似路径。flag 被静默忽略。归类为硬件内核配置 flag 未生效。

#### 第 6 类 · 服务栈 / 缓存正确性（+3）

**3. vLLM #51637 — `[新签名]` MooncakeStoreConnector Request-ID ABA corrupts async store job accounting and leaks KV blocks**
- **链接**：https://github.com/vllm-project/vllm/issues/51637
- **来源**：vllm-project/vllm
- **描述**：分离式 KV 传输中，Request-ID 在 preemption 后被复用（ABA 问题），导致 MooncakeStoreConnector 的异步存储作业记账被破坏并泄漏 KV blocks。复现于 vLLM 0.25/0.26 + Mooncake 0.3.12 + Qwen3-8B。

**4. SGLang #33470 — `[新签名]` inference_mode mismatch can break lazy buffers**
- **链接**：https://github.com/sgl-project/sglang/issues/33470
- **来源**：sgl-project/sglang
- **描述**：warmup/autotune forward 在 `torch.inference_mode()` 下运行，real serving forward 不在。lazy-allocated buffers 在 warmup 下首次分配成为 inference tensors，后续 forward path 的 Python-side inplace update 失败：`RuntimeError: Inplace update to inference tensor outside InferenceMode`。

**5. SGLang #34297 — `[新签名]` DSpark CUDA Graph correctness and robustness roadmap**
- **链接**：https://github.com/sgl-project/sglang/issues/34297
- **来源**：sgl-project/sglang
- **描述**：DSpark CUDA-Graph 执行正确性的系统性路线图，将故障分解为 5 个独立可测试层：distributed control-plane consistency → verifier state/write-window correctness → kernel-produced plan/index correctness → capture-initialization ordering → replay-time ownership/stream/lifetime contracts。引用 #31195（Bug 1）、#32183（rewrite）、#32467（Bug 2）、#33795。

#### 第 7 类 · 量化（+3）

**6. vLLM #51380 — `[新签名]` ROCm AITER FP8 MLA prefill produces run-dependent wrong answers on short prompts**
- **链接**：https://github.com/vllm-project/vllm/pull/51380
- **来源**：vllm-project/vllm
- **描述**：ROCm 上 AITER FP8 MLA prefill 对所有非 chunked-context 请求（含短 prompt）都选 FP8 prefill。短 prompt（603-698 tokens, GSM8K-100 5-shot）下 FP8 prefill 改变 borderline logits，产生**运行依赖的错误答案**——重跑同一个失败 prompt 可能返回正确答案。修复：短 prompt（<1024 tokens）路由到 BF16 FlashAttention。

**7. TRT-LLM #17476 — `[新签名]` Static FP8 Cosmos3 checkpoint re-quantization on fused projections discards calibration**
- **链接**：https://github.com/NVIDIA/TensorRT-LLM/pull/17476
- **来源**：NVIDIA/TensorRT-LLM
- **描述**：静态 FP8 校准的 Cosmos3 Nano/Super 检查点在加载时不忠实：topology 将 Q/K/V 融合为单个 `Linear`、gate/up 融合为另一个。per-tensor FP8 Linear 只有一个 weight scale，loader 保留 `max(scales)` 并对组内其他成员重新量化：`qWi' = FP8-round((si * qWi) / max(s))`。影响 Nano 252 个 fused shards 中的 141 个、Super 448 个中的 252 个（~56%），受影响 shards 的附加误差为数值底限的 **7-18 倍**。

**8. SGLang #34179 — `[新签名]` W4A4 MegaMoE FP4 activation deterministic nearby-integer substitution + silent empty completions**
- **链接**：https://github.com/sgl-project/sglang/issues/34179
- **来源**：sgl-project/sglang
- **描述**：DeepSeek-V4-Flash-0731 在 8x B200 + MegaMoE 上，W4A4（FP4 activation packing）vs W4A8（默认 FP8 activation）A/B 测试。三个有界质量观察：(1) 长 prefill 空 completion 不对称（10 vs 2，HTTP 200 + 立即 EOS + 无输出）；(2) **确定性 nearby-integer 替换**——"12+23=?" W4A4 答 "36"（p=0.9788，"35" 被踢出 top-8），W4A8 答 "35"（p=1.0000）；(3) ASCII soft fail（1.6% 非 ASCII）。TTFT 本身无差异。

#### 第 8 类 · KV 缓存 / 注意力（+3）

**9. SGLang #34190 — `[新签名]` FlashInfer no-split CUDA-graph plan grids launch uninitialized scheduler entries**
- **链接**：https://github.com/sgl-project/sglang/pull/34190
- **来源**：sgl-project/sglang
- **描述**：确定性 CUDA-graph decode 中 FlashInfer FA2 tensor-core path 禁用 split-KV。planner 将 grid-x 填充到 split-work occupancy bound，但因 split-KV 禁用，padded tail 无 validity mask → kernel 可能 launch scheduler entries 从未初始化的 block。zeroed tail 仅产生冗余工作；reused nonzero tail data 产生条件性 stale-index 风险。上游：flashinfer#4002 / flashinfer#4430。

**10. TRT-LLM #17466 — `[新签名]` Window vector layer indexing inconsistency (model-global vs pipeline-stage local)**
- **链接**：https://github.com/NVIDIA/TensorRT-LLM/pull/17466
- **来源**：NVIDIA/TensorRT-LLM
- **描述**：`KvCacheConfig.max_attention_window` 是 model-global repeating pattern，但 KV-cache pools 和 attention metadata 是 pipeline-stage local。之前 V1 部分分片、V2 保持全局，消费者需猜测用哪个 layer index。修复将 configured pattern 一次性解析为精确 per-local-layer `max_attention_window_vec`。

**11. vLLM #51682 — `[新签名]` AMD packed KDA decode kernel assumes unit-stride state_indices**
- **链接**：https://github.com/vllm-project/vllm/pull/51682
- **来源**：vllm-project/vllm
- **描述**：AMD 版 `fused_recurrent_kda_packed_decode` 将 `state_indices` 视为单位步幅，拒绝任何非连续 tensor。NVIDIA 版已接受 `stride_state_indices`。推测解码开启时 `block_table_tensor[:, 0]` 产生 strided view → AMD kernel warmup 失败（`ValueError: must be contiguous and one-dimensional`）。无推测解码时不暴露。

#### 第 10 类 · 分词器 / 对话模板（+2）

**12. vLLM #51727 — `[新签名]` DeepSeek V4/3.2 tokenizer vocab size overcount crashes guided decoding**
- **链接**：https://github.com/vllm-project/vllm/pull/51727
- **来源**：vllm-project/vllm
- **描述**：`DeepseekV4Tokenizer.__len__` 返回 `vocab_size + len(get_added_vocab())`，但 1283 个 added tokens 中 3 个 id < 128000（已在 base vocab 中）→ 真实 count 为 129280 而非 129283。override 在 transformers 5.x TokenizersBackend 下既冗余又错误（`len()` 已包含 added tokens）。导致 vocab_size mismatch 崩溃 guided decoding。`deepseek_v32.py` 有相同 stale pattern。

**13. TRT-LLM #17464 — `[新签名]` Harmony silently discards malformed tool-call messages (empty tool_calls, HTTP 200, no logs)**
- **链接**：https://github.com/NVIDIA/TensorRT-LLM/pull/17464
- **来源**：NVIDIA/TensorRT-LLM
- **描述**：`HarmonyAdapter._strip_incomplete_messages` 当无 `<|message|>` token 跟随时，删除最后一个 `<|start|>` 之后的所有内容。格式错误的 tool-call 消息被静默丢弃 → 返回空 `tool_calls` 数组 + 空 `content`，HTTP 200，`finish_reason: "stop"`，日志无任何记录。调用者无法区分此情况与模型拒绝调用工具。此 PR 修复"静默"而非上游 corruption。

#### 第 11 类 · 解码 / 采样（+1）

**14. vLLM #51641 — `[新签名]` Streaming reasoning content misrouted when stop parameter set (stop buffer desynchronizes delta_text from delta_token_ids)**
- **链接**：https://github.com/vllm-project/vllm/issues/51641
- **来源**：vllm-project/vllm
- **描述**：流式请求含非空 `stop` list + `include_stop_str_in_output=false` 时，V1 detokenizer 暂存最多 `max(len(stop)) - 1` 字符到 stop buffer，但对应的 `delta_token_ids` **不暂存**。Qwen3 reasoning parser 用 `delta_token_ids` 检测 `</think>`、用 `delta_text` 分割——两者不同步 → reasoning tail 和 `</think>` 标记泄漏到 `content`。在 MTP/推测解码/异步输出下 token batch 跨 `</think>` 边界时触发。

#### 第 12 类 · 权重 / 检查点加载（+2）

**15. vLLM #51423 — `[新签名]` LoRA single-tensor lora_a not normalized in merged set_lora → IndexError on packed layers**
- **链接**：https://github.com/vllm-project/vllm/pull/51423
- **来源**：vllm-project/vllm
- **描述**：`MergedColumnParallelLinearWithLoRA.set_lora()` 只 normalize `lora_b`，单 2-D `lora_a`（shape `[rank, input_size]`）直接 pass through → `lora_a[i]` 返回 1-D row 而非第 i slice 的矩阵 → `IndexError` on fused QKV（3 slices）/gate_up（2 slices）。影响 `fully_sharded_loras=True/False` 两种路径。

**16. tt-metal #52743/#52630 — `[新签名]` Unconditional dequantize on already-dequantized Kimi checkpoint → crash on missing quant metadata**
- **链接**：https://github.com/tenstorrent/tt-metal/pull/52743
- **来源**：tenstorrent/tt-metal
- **描述**：Kimi-K2.6 检查点同时以量化和非量化版本发布，加载路径统一调用 `dequantize_state_dict`。已反量化的检查点（bf16）`config.json` 无 `quantization_config`，但 dequantizer 读取 quant metadata 后才触碰 tensor → `ValueError: Missing DeepSeek quantization_config.weight_block_size`。修复：`passthrough_state_dict` 检查 config 是否含 quant metadata，若无则原样 materialize。

#### 第 13 类 · 模型图 / 架构配置（+2）

**17. vLLM #51493 — `[新签名]` Gemma 4 MoE falls back to V1 model runner → MTP drafter wrong input dimension**
- **链接**：https://github.com/vllm-project/vllm/pull/51493
- **来源**：vllm-project/vllm
- **描述**：Gemma 4 dense 变体通过 `not is_moe` 分支获得 V2 model runner，但 MoE 变体（26B A4B）回退 V1。V1 path 给 Gemma4 MTP drafter 输入错误维度：checkpoint `pre_projection.weight` = `[1024, 5632]` = `2 * backbone_hidden(2816)`，V1 喂 `backbone_hidden + draft_hidden = 2816 + 1024 = 3840` → `reduction dim mismatch` → engine init 失败。

**18. vLLM #51302 — `[新签名]` deepseek-vl2 incomplete language_config relies on original DeepseekV2Config defaults**
- **链接**：https://github.com/vllm-project/vllm/pull/51302
- **来源**：vllm-project/vllm
- **描述**：DeepSeek-VL2 检查点 `language_config` 不完整（如 `deepseek-vl2-small` 无 `vocab_size` 和 `kv_lora_rank`），依赖原始 DeepSeek-VL2 `DeepseekV2Config` 默认值（`vocab_size=102400, kv_lora_rank=512`）。vLLM 用内置 transformers `DeepseekV2Config` 解析，其通用默认值不同（`vocab_size=32000, kv_lora_rank=None`）→ `NoneType + int` → `TypeError` + `AssertionError`。

---

### arxiv 发现

| 论文 | 状态 |
|------|------|
| Decision-Level Hijacking (2607.25227) | **已在主文档中**（2 次引用） |
| From Bit-Position Sensitivity to UEP (2607.19623) | **已在主文档中**（2 次引用，条目 1.x 附近） |
| Teaching LLMs Bit Manipulation Puzzles (2606.23672) | 不相关（推理挑战赛方案，非故障模式） |
| RangeGuard: Bounded Approximate EC for DNNs (2605.04563) | **防御/缓解论文**——引入 Range Identifier（RID）做有界近似纠错，非新故障模式。bit flip 经 attention/residual/norm 层放大并传播的机制已被 BFA 家族（第 3 类）覆盖。作为补充来源引用，不单独编目 |
| Bit-Flip Vulnerability of Shared KV-Cache (2604.17249) | **已在主文档中**（2 次引用） |

---

### 已跳过的重复条目（已在主文档中）

| 条目 | 已有编号 |
|------|----------|
| vLLM #51049 (fused_qk_norm_rope partial-NeoX corruption) | 条目 9.8（来源 PR #50903） |
| tt-metal #52675 (ttnn.pow bf16 error) | 条目 1.54 |
| vLLM #51620 / #51684 (DFlash fused KV quant bypass) | 条目 6.60（来源 #51581） |
| SGLang #34154 (mRoPE draft-extend) | 已有条目 9.10 |
| SGLang #34153 (chunked prefill abort race) | 已有条目 6.67 |
| SGLang #34077 / #34076 (relay staging / overlap batch) | 已有条目 |
| transformers #47838 / #47822 | 已有条目 10.10/10.13 等 |

---

## 重点突出

### 最值得关注的新发现

1. **Triton #11217（AMD bf16 multiply 截断）**：这是一个**影响所有 AMD GPU 上 Triton kernel 中 bf16 乘法**的系统性舍入模式 bug。bf16 multiply 100% 匹配截断而非 RNE，导致 RoPE 等 kernel 在 AMD 上与 CUDA/torch 后端结果分叉。影响面极广。

2. **TRT-LLM #17476（静态 FP8 重新量化）**：ModelOpt 校准的 FP8 检查点在加载时因 topology 融合而被**二次量化**，56% 的 fused shards 误差为数值底限的 7-18 倍。检查点本身正确，是 fusion topology 丢弃了校准。静默数值退化。

3. **SGLang #34179（W4A4 确定性 nearby-integer 替换）**：FP4 activation packing 导致 "12+23=?" 的答案从 "35" 确定性地变为 "36"（p=0.9788），"35" 完全被踢出 top-8。这是 FP4 激活量化的**有界正确性信号**。

4. **vLLM #51380（ROCm FP8 MLA 运行依赖错误答案）**：FP8 prefill 在短 prompt 上改变 borderline logits，产生**重跑可变**的错误答案——同一 prompt 重跑可能返回正确答案。这类非确定性正确性 bug 极难调试。

---

## 分类标注汇总

| 类别 | 新增条数 | 新编号 |
|------|----------|--------|
| 第 1 类 · 硬件/SDC | +2 | 1.56–1.57 |
| 第 6 类 · 服务栈/缓存 | +3 | 6.70–6.72 |
| 第 7 类 · 量化 | +3 | 7.25–7.27 |
| 第 8 类 · KV缓存/注意力 | +3 | 8.27–8.29 |
| 第 10 类 · 分词器/模板 | +2 | 10.15–10.16 |
| 第 11 类 · 解码/采样 | +1 | 11.25 |
| 第 12 类 · 权重/检查点 | +2 | 12.12–12.13 |
| 第 13 类 · 模型图/架构 | +2 | 13.21–13.22 |
| **合计** | **+18** | **423 → 441** |
