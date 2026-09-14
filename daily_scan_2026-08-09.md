# LLM 推理故障每日扫描报告 — 2026-08-09

**扫描日期**：2026-08-09（UTC）
**搜索窗口**：GitHub `created:>2026-08-08`（过去 1 天滚动窗口）+ arxiv 近期
**主参考文档基线**：406 条（合并前）

---

## 扫描范围与 API 调用记录

| # | 目标 | 查询 | 结果数 |
|---|------|------|--------|
| 1 | vLLM | `label:bug created:>2026-08-08` | 19 |
| 2 | vLLM | `wrong OR incorrect OR corruption OR NaN OR silent OR determinism in:title created:>2026-08-08` | 4 |
| 3 | vLLM | `silent OR stale OR wrong OR corrupt OR ignored in:title created:>2026-08-08` | 3 |
| 4 | vLLM | `bugfix in:title created:>2026-08-08` | 14 |
| 5 | SGLang | `wrong OR incorrect OR corruption OR NaN OR silent OR determinism in:title created:>2026-08-08` | 2 |
| 6 | SGLang | `fix silent OR wrong OR corrupt OR ignored OR stale OR bugfix in:title created:>2026-08-08` | 14 |
| 7 | tt-metal | `incorrect OR correctness OR silent OR wrong OR NaN OR corruption in:title created:>2026-08-08` | 2 |
| 8 | triton | `correctness OR wrong OR fix in:title created:>2026-08-08` | 2 |
| 9 | transformers | `incorrect OR silent OR wrong OR NaN OR corruption in:title created:>2026-08-08` | 3 |
| 10 | TRT-LLM | `incorrect OR correctness OR wrong OR silent OR NaN in:title created:>2026-08-08` | 0 |
| A1 | arxiv | `silent data corruption LLM` | 3 |
| A2 | arxiv | `bit flip LLM` | 5 |
| A3 | arxiv | `KV cache corruption` | 5 |
| A4 | arxiv | `LLM inference correctness` | 5 |
| A5 | arxiv | `GPU soft error learning` | 0 |
| A6 | arxiv | `quantization LLM reliability` | 5 |
| A7 | arxiv | `LLM serving vulnerability` | 5 |

GitHub Search API 调用约 12 次（限额 10/min search，已分开调用并间隔 ≥7s，未触发 403）；arxiv 9 次（单独配额）。

---

## 去重结果

对照 FAULT_MASTER_REFERENCE.zh.md 现有 406 个条目，本日发现的条目经逐一比对 issue/PR 编号：

- **#34189**（SGLang DSV4 spec draft > 4 compress ring）：是 #33872（已编目为 8.25）的 supersede 修复，**不重复添加**，仅作为 8.25 的补充来源。
- **#51564**（vLLM Reject heterogeneous vocab）：是 #51561 的修复 PR，**不单独添加**，与 #51561 合并描述。
- 其余 16 条 issue/PR 编号均**未出现在主文档中**（全部为新条目）。

---

## 发现条目（按故障领域分类）

### 硬件 / SDC（SFPU 数值原语 / cache 损坏）

1. **[已知类别 var:1.x] tt-metal #52675 — `ttnn.pow` bf16 路径相对误差随指数线性增长（~0.2%·|y|，y=100 时 18%）；溢出返回 NaN 而非 inf**
   - 来源：https://github.com/tenstorrent/tt-metal/issues/52675
   - 机制：`ttnn.pow(tensor, float_scalar)` / `ttnn.pow(tensor, tensor)` / `rpow` 在 bf16/BFLOAT8_B/BFLOAT4_B 下走 `_sfpu_unary_power_21f_`/`_sfpu_binary_power_21f_`，用单一 3 阶 minimax 多项式算 log2 mantissa，多项式绝对误差（max ≈2.8e-3）被指数 y 放大→结果相对误差 ≈2.0e-3·|y| 无上界。`ttnn.pow(0.996094, 100.0)` 返回 0.8008（真值 0.6761，18% 偏差）。此外结果 ≥2^128 时 biased exponent 回绕到 255+非零尾数→NaN（应为 inf）。fp32 路径已在 #49625 重建为 NR reciprocal+atanh log，但 bf16（多数用户默认）仍用原算法。int32 exponent 重载用精确重复平方，故 `pow(x,100)` 准确但 `pow(x,100.0)` 不准确。

2. **[已知类别 var:1.x] tt-metal #52671 — `ttnn.update_cache` 在 batch>32 且非 32 的倍数时静默用 tile padding 覆盖真实用户行（非确定性）**
   - 来源：https://github.com/tenstorrent/tt-metal/issues/52671
   - 机制：`ttnn.update_cache` 当缓存 batch `Bcache > 32` 且非 32 的倍数时，将输入 tensor 的 tile padding 写到真实用户行上。受影响用户行返回零。无任何形状拒绝或警告。对若干尺寸结果还**非确定性**：同输入同调用每次 dispatch 返回不同数量的损坏行，偶尔返回完全正确答案→测试极易漏检。probe 实测：Bcache=33 时正确行 2–33 变化、损坏 0–31 行；Bcache=65 同样非确定。根因在 `update_cache_multi_core_program_factory.cpp` 的 tile padding 循环边界。

### 服务栈 / 缓存正确性（vLLM）

3. **[已知类别 var:6.x] vLLM #51581 — DFlash fused-KV projection 对 sliced qkv_proj 权重调 `F.linear`，绕过量化方法，静默忽略 `weight_scale` → 数值错误的 draft（降 spec acceptance）**
   - 来源：https://github.com/vllm-project/vllm/issues/51581
   - 机制：`Qwen3DFlash` 构建 fused KV GEMM 时切 `qkv_proj.weight` 切片并用裸 `F.linear`，完全绕过模块量化方法。当 `qkv_proj` 是量化 Linear 时，`.weight` 是 packed/quantized tensor。dtype 碰巧对齐时不 raise 而是**静默忽略 weight_scale**→数值错误 draft（spec decode 不报错，仅表现为 acceptance 下降）。对任何 weight-only 方案（fp8/compressed-tensors/AWQ/GPTQ）适用。不是 #42588（ROCm activations→fp32，已修）的重复。修复 #51582。

4. **[已知类别 var:6.x] vLLM #51580 — ROCm v0.26.0 release：AITER fused-MoE 路径在 gfx942 上从第二步 decode 起产 NaN logits（argmax 恒 token 0）**
   - 来源：https://github.com/vllm-project/vllm/issues/51580
   - 机制：`VLLM_ROCM_USE_AITER=1` 服务 Qwen3.5-397B-A17B-FP8（TP8, 8x MI325X gfx942）：第一个 token 正确（有限 logprob），此后每步 decode 输出 token id 0（argmax over NaN logits）。隔离测试排除 KV dtype/attention（`VLLM_ROCM_USE_AITER_MHA=0` 仍 NaN），仅 `VLLM_ROCM_USE_AITER_MOE=0` 或 `AITER=0` 干净。缺陷在 AITER fused-MoE 路径。已在 main 修复，release 镜像仍存在。logprobs 请求返回 400 `Out of range float values are not JSON compliant: nan`。

5. **[已知类别 var:6.x] vLLM #51525 — 显式 `VLLM_CUTLASS` NVFP4 MoE 在 SM_120 静默产空补全（content:null, finish_reason:length）**
   - 来源：https://github.com/vllm-project/vllm/issues/51525
   - 机制：SM_120（RTX 5090 Laptop）服务 NVFP4 MoE checkpoint + `--moe-backend cutlass`（resolve 到 `VLLM_CUTLASS`）：服务消费整个 `max_tokens` 预算返回 `content: null` + `finish_reason: "length"`。无异常、无警告、HTTP 200。切换 `--moe-backend marlin` 正确。`--moe-backend auto` 不选此后端（选 FLASHINFER_CUTLASS）→需显式 opt-in 才触达，但一旦进入失败不可见（短 prompt smoke pass、吞吐正常）。native-Windows 构建。

6. **[已知类别 var:6.x] vLLM #51561 — `use_heterogeneous_vocab`(TLI) + `use_local_argmax_reduction` 两者同时启用时 local-argmax 提前返回，静默跳过 intersection mask 与 draft→target 映射**
   - 来源：https://github.com/vllm-project/vllm/issues/51561（修复 #51564）
   - 机制：两个独立 opt-in flag 无互斥校验。同启时 `_greedy_sample` 的 local-argmax 分支提前返回，跳过异构词表的 intersection mask 与 draft→target ID 映射。`get_top_tokens` 返回 draft-vocab 索引，其可选 `draft_id_to_target_id` remap 是 EAGLE/MTP offset 表，两类 drafter（Qwen3/Llama）均未定义→无翻译；输入侧仍 target→draft 映射→两方向不一致。结果：降 spec acceptance 或向 rejection sampling 暴露未映射 token id。修复 #51564 在 SpeculativeConfig 配置时拒绝不支持的组合。

### 服务栈 / 缓存正确性（SGLang）

7. **[已知类别 var:6.x] SGLang #34184 — prefill CUDA graph 下 stale track rows 把当前窗口 scatter 到前一请求的 conv checkpoint（整个生成错误）**
   - 来源：https://github.com/sgl-project/sglang/pull/34184
   - 机制：hybrid-SWA 模型 prefix-cache hit 可 decode 出它从未生产过的 conv state。replayed prefill graph 读 `_capture_req_slots` 行，但 `mamba_track_mask`/`mamba_track_indices` 保留前一 replay 的 live batch 之后的残留→当前窗口的 track scatter 落到前一请求的 checkpoint。之后 restore 该 prefix 的请求**整个生成错误**。需 prefill graph + chunked prefill + 并发 + prefix hit。track buffer 的 clear 被守卫在 `bs != raw_bs`（`fill_from`），但 prefill graph 因 pad tokens 而非 requests 使 `padded_bs == raw_bs`→clear 永不触发（死代码）。与 #34043（spec 路径 scatter 特化）邻近但独立。

8. **[已知类别 var:1.x→6.x] SGLang #34167 — DSA top-k v2 在 CUDA 13.1+ 静默丢弃非 primary rank 输出（tmp_out 留零→phase 3.5 拷零到正确 DSMEM 地址）**
   - 来源：https://github.com/sgl-project/sglang/pull/34167（#33835 根因）
   - 机制：DeepSeek-V4 DSA decode 的 fused small-batch cluster kernel `TopKCluster::forward` phase-3 scatter 目的上提：`cur_out = is_primary ? problem.out : smem->tmp_out`。`problem.out` 可为 `shared::cluster`(DSMEM) alias，`smem->tmp_out` 为 `shared::cta`。合并成单指针变量后 **cicc 13.1+ 在 sm_90a 上 mis-lower block-local 分支**，静默丢弃每个非 primary rank 的 staged output→`tmp_out` 留零→phase 3.5 忠实拷零到正确 DSMEM 地址。slot-diff 显示 512 槽中 129 正确、一个 383 槽 bad run（7 个 peer 的贡献）。**触发器是 CUDA toolkit 版本而非 GPU arch**：同源同硬件 12.9/13.0 正确、13.1/13.2/13.3 错误。

9. **[已知类别 var:1.x→6.x] SGLang #34142 — CP round-robin shard 单行时 row pitch 被 `cp_size` 倍膨胀 → DSV4 prefill GPU memory fault**
   - 来源：https://github.com/sgl-project/sglang/pull/34142
   - 机制：`dsa_cp_round_robin_split_data` 末行 `input_.view(-1, cp_size, ...)[:, cp_rank].contiguous()`。当 prefill chunk 恰 `cp_size` token 时每 rank shard 仅 1 行→size-1 外维 `is_contiguous()` 恒 True→`.contiguous()` 为 no-op→shard 保留 `cp_size` 倍膨胀的 row pitch（`cp_size*row_numel` 而非 `row_numel`）。下游 kernel 用 `stride(0)` 作 pitch 将 shard 视为 `cp_size` 倍大。DSV4（hc_mult=4, hidden=7168, cp=8）pitch 229376 vs 28672；aiter `mhc_pre` 据此 size buffer descriptor 越界寻址→memory fault。

10. **[已知类别 var:6.x] SGLang #34153 — final chunked-prefill abort 提交竞态：取消请求可暴露 1 token 后 finish:abort**
    - 来源：https://github.com/sgl-project/sglang/pull/34153（#34149 / #34112）
    - 机制：abort 可在 final chunked-prefill `EXTEND` 启动后、其延迟结果提交前到达。`abort_request()` 置 `req.to_finish=FINISH_ABORT()`，但 prefill 结果处理器先前 append 了 sampled token 与 token 派生元数据（在 `update_finish_state()` 消费 pending finish 之前）。CUDA/HIP overlap loop 中，调度下一 batch 发生在处理延迟 final-prefill 结果之前→无 admission guard 时 pending-finish prefill 请求可进入乐观 decode、增 `decode_batch_idx`/KV committed watermark、分配额外 decode slot 后结果才丢弃。结果：取消请求暴露 1 token 然后 abort。

11. **[已知类别 var:6.x] SGLang #34177 — Cache-DiT 在 breakable CUDA graph 开启时静默保持禁用（1.00x speedup，无错误无警告）**
    - 来源：https://github.com/sgl-project/sglang/pull/34177
    - 机制：`DenoisingStage._maybe_enable_cache_dit` 在 `--enable-breakable-cuda-graph` 时提前返回（合理：Cache-DiT 的 step-skipping 控制流不能烘焙进 captured graph）。但**静默**：同启 BCG + `SGLANG_CACHE_DIT_ENABLED=1` 不报错不警告，唯一症状是 1.00x speedup。与 #33827（Cache-DiT 成功 mount 但 cache 从不 hit）同症状但不同诊断。MiniMax-H3 还可为 `quality="high"` batch 动态 mount→per-request 也可静默失效。

### 量化

12. **[已知类别 var:7.x] vLLM #51545 — NVFP4 MoE CuTeDSL 路径：non-gated 中间尺寸非 128 行 tile 对齐时静默输出错误**
    - 来源：https://github.com/vllm-project/vllm/pull/51545
    - 机制：FlashInfer CuTeDSL NVFP4 MoE 路径在 rank-local intermediate size 非 CuTeDSL 128 行 tile 对齐时输出错误。intermediate size 来自模型架构与并行布局（非运行时 batching 参数）。修复将 non-gated NVFP4 MoE 权重与 scale pad 到 128 行中间 tile 后再 swizzle scale，保留原始未 pad intermediate size（零填充使输出不变）。属「kernel tile 对齐契约静默违反」量化子签名。

### KV 缓存 / 注意力

13. **[已知类别 var:8.x] SGLang #34159 — 确定性推理 all-reduce 在 tp>1 时仍非确定（NCCL channel count 随消息大小变→prefill 与 decode logprob 不一致）**
    - 来源：https://github.com/sgl-project/sglang/pull/34159
    - 机制：`--enable-deterministic-inference` pin NCCL 算法为 `allreduce:tree` 并禁用 custom/symmetric-memory all-reduce，使给定消息大小的 reduction 可复现，但**不**对消息大小不变：NCCL 按消息大小选 channel count，每 channel 承载不同形状 tree→同一 element 的 reduce 顺序随 batch 中共享 token 数而变→确定性模式下 prefill 与 decode 对一 token 的 logprob 不一致。pin channel count 闭合该缺口：Inkling-Small-NVFP4 tp=4 prefill-vs-decode 检查从 0.151→**逐位一致**（16384 logprob 0 差异）。注：仅消 all-reduce 源；非 aten-override kernel 仍留分歧（Qwen3-Next tp=4: 6.44e-4→5.06e-4）。

### 位置编码 / RoPE

14. **[已知类别 var:9.x] SGLang #34154 — 多模态 draft-extend 每个 extended token 只产 1 个 mRoPE position → fused Triton mRoPE 越界（native 则 shape mismatch）**
    - 来源：https://github.com/sgl-project/sglang/pull/34154（#34152）
    - 机制：Qwen3-VL target + EAGLE3 draft 在首个携带非平凡 batch 的 draft-extend 崩溃。`ForwardBatch._compute_mrope_positions` 为 forward pass 有 `sum(extend_lens)=batch_size*num_draft_tokens` token 的 draft-extend 产 `batch_size` 个 mRoPE position→rotary embedding 索引越过 position tensor 末端：fused Triton mRoPE kernel 越界 CUDA illegal memory access，`forward_native` shape mismatch。链路：`mm_input.mrope_positions` 仅跨原始 prompt；draft-extend 时 `prepare_for_extend_to_fill_draft_kvcache` 设 `prefix_lens=seq_lens`（已越过 prompt）→prompt 切片为空→fallback 运行；fallback 调 decode-shaped `_expand_mrope_from_input` 返回单 position，但 extend 需 `extend_seq_len` 个。单请求正常，真实流量才崩。

### 分词器 / 对话模板

15. **[已知类别 var:10.x] transformers #47855 — `additional_special_tokens` 在 `extra_special_tokens` 为空容器时被 `setdefault` 静默丢弃（token 静默丢失）**
    - 来源：https://github.com/huggingface/transformers/pull/47855（#47838）
    - 机制：`tokenization_utils_base.py` 中 `setdefault("extra_special_tokens", init_kwargs.pop("additional_special_tokens"))` 当 `tokenizer_config.json` 含空容器 `extra_special_tokens: {}`/`[]` 时，`setdefault` 因 key 已存在返回现有空 dict，不设置被 pop 的 `additional_special_tokens`→静默 token 丢失。3 个初始化路径均有此问题。修复改为显式赋值逻辑（`extra_special_tokens` 非空才 fallback 赋值）。

16. **[已知类别 var:10.x] SGLang #34178 — DSV4 reasoning + tool call 流式解析器 6 个 bug（静默丢弃 tool call / 特殊 token 泄漏进 API 输出）**
    - 来源：https://github.com/sgl-project/sglang/pull/34178
    - 机制：DSV4 reasoning + tool call 解析管线 6 个流式 bug：`DeepSeekV4Detector` 缺 `tool_start_token`→DSML 块滞留 `reasoning_content`、tool parser 从不处理；base class chunk 边界无条件 `self._buffer=""` 销毁部分 token（应 holdback think/tool 前缀）；其余涉及边界切分与 MTP/DSpark 多 token batching。后果：静默丢弃 tool call 或特殊 token 泄漏进 API 输出。79 个新/更新测试覆盖 chunk-size 不变性、边界切分、多 token 批处理。

### 服务栈（EPLB / 初始化）

17. **[已知类别 var:6.x] vLLM #51568 — EPLB 对 E8M0 expert scale 用数值 dtype 注册，NCCL all-gather dummy 时 `TypeError: Float8_e8m0fnu not supported`（engine init 失败）**
    - 来源：https://github.com/vllm-project/vllm/pull/51568（#51567）
    - 机制：MXFP4/MXFP8 Marlin expert 的 weight scale 为 `torch.float8_e8m0fnu`。EPLB `rearrange_expert_weights_inplace()` 启动 profiling 做 dummy device-group all-gather 预留通信 buffer，但 PyTorch NCCL 不支持 E8M0→`TypeError`。修复在 `set_eplb_state` 注册边界将 E8M0 暴露为存储保留的 `uint8` view（`view` 非 `to`，保持 shape/stride/offset/byte/指针/storage，EPLB 移动原始 scale 字节，kernel 继续读原 E8M0）。

---

## arxiv 论文（强相关，多数已编目或为补充视角）

- **arXiv:2608.02712 — "Don't Regenerate, Debug: A Domain-Specific Agent for Repairing Near-Miss Hardware Operators"**（2026-08-03）：用领域专用 agent 修复 near-miss 硬件算子（即数值正确性有缺陷但能编译运行的 kernel）。与第 1 类 SFPU/kernel 数值缺陷族直接相关，可作为**方法论补充引用**（kernel 正确性自动修复）。
- **arXiv:2607.28884 — "Hollow-LLM Attack: Computationally Trivial Weights in Zero-Knowledge Verification of LLM Inference"**（2026-07-30）：ZK LLM 推理验证中，provider 可用"计算平凡"权重（高度量化/稀疏）欺骗验证者。**权重完整性/推理保真度**新攻击面，可作为第 3 类或第 12 类补充视角。
- **arXiv:2608.05660 — "Reasoning Errors Have a Region and a Direction in the Residual-Stream Trajectory of LLMs"**（2026-08-06）：在层间残差流位移中定位推理错误的方向与区域。**内部信号监控方法论**，可作附录监控可行性引用（trajectory 度量族）。
- 其余（Bit-Flip/KV-Cache/Quantization 系列）均**已编目**（3.x / 7.x 类），无新发现。

---

## 重点新发现（突出）

本日最值得关注的新案例：

1. **SGLang #34167（DSA top-k 在 CUDA 13.1+ 静默丢弃非 primary rank 输出）** — **类签名**：编译器版本触发的静默数值损坏。cicc 13.1+ 在 sm_90a 上 mis-lower `shared::cluster`/`shared::cta` 合并指针的 block-local 分支→7 个 peer 的贡献静默归零，phase 3.5 忠实拷零到正确地址。同源同硬件仅 toolkit 版本不同→正确/错误。是 #33835 非法内存访问的根因。**编译器版本作为静默损坏触发器的新签名**。
2. **tt-metal #52675（ttnn.pow bf16 相对误差随指数无界增长）** — 静默、有限、错误值跨整个文档化输入域；bf16 是多数用户默认 dtype。与 1.49-1.53 SFPU 数值原语缺陷同族但本条是"多项式近似误差被指数放大"的系统性子模式。
3. **tt-metal #52671（update_cache batch>32 且非 32 倍数静默损坏 + 非确定性）** — KV cache 写入 tile padding 覆盖真实用户行，部分尺寸非确定性（同输入每次不同损坏行数，偶尔完全正确）→测试极易漏检。**tile-padding 交互非确定性**新子签名。
4. **SGLang #34184（prefill graph stale track rows 污染 conv checkpoint，整个生成错误）** — 完全静默，受影响请求**整个生成**错误。track buffer clear 是死代码（`bs==raw_bs` 永真）。
5. **SGLang #34159（确定性推理 tp>1 all-reduce 仍非确定）** — 确定性模式下 prefill/decode logprob 不一致（0.151 KL），对"确定性推理"承诺的根本性违反。

无全新故障领域类别发现；全部为已有类别的新实例/新子签名。

---

## 将合并进主文档的条目（16 条，全部为新；另 1 条补充来源）

见主文档合并结果。
