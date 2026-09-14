# LLM 推理故障每日扫描报告 — 2026-08-13

**扫描日期**：2026-08-13
**搜索窗口**：created > 2026-08-05（GitHub），arxiv 按提交日期降序
**搜索目标**：6 个 GitHub 仓库（vllm, sglang, transformers, TensorRT-LLM, triton, tt-metal）+ arxiv

---

## 📊 扫描摘要

- GitHub API 调用：10 次（Issues Search × 10）
- arxiv API 调用：7 次
- 候选条目：22 条
- 去重后新增条目：19 条（3 条已存在于主文档）

---

## 🔴 新故障模式 / 新签名

#### 1. transformers #47942: NemotronH 多 GPU device_map 静默错误输出 — `[新签名]`
- **来源**：transformers #47942（2026-08-13 open）https://github.com/huggingface/transformers/issues/47942
- **描述**：Nemotron-3.5-Lightning-30B（hybrid Mamba-2 + latent-MoE）在 `device_map="auto"` 多 GPU 分发下产生**近随机输出且无任何错误/警告**。三个独立缺陷复合：
  1. `use_cache=True`（默认）在 device boundary 上损坏 hybrid scan cache → loss ≈10（131k vocab 近随机），`use_cache=False` 给出正常 loss 2.6–3.0
  2. 即使 `use_cache=False`，正确性取决于 device map 在何处放 split boundary（2 GPU loss 2.66✅，3 GPU loss 10.23❌，4 GPU loss 9.51❌）——某些 boundary 切断了必须保持 device-local 的 Mamba/latent state
  3. 安装 mamba-ssm + causal-conv1d 后，fused Triton kernel 在 dispatch hooks 下 crash（device-context mismatch）
- **分类**：分布式/多 GPU + 模型图/架构 → 第 5 类（新子签名：hybrid_scan_cache_device_boundary_corruption）

#### 2. transformers #47932: 推测解码 candidate generator 返回未采样的 q 分布 — `[新签名]`
- **来源**：transformers #47932（2026-08-12 open）https://github.com/huggingface/transformers/issues/47932
- **描述**：`_speculative_sampling` 将 candidate generator 返回的 `candidate_logits` 当作 `q`（drafter proposal distribution）。但三个 block drafter 返回的 logits **不是其 token 实际采样的分布**：`DFlashTokenCandidateGenerator` 从 processor-applied logits 采样 token，但返回 raw logits（丢弃 processor 应用结果）。`do_sample=True` 时 acceptance test 针对错误的 q 运行，**破坏推测解码的无损性保证**。静默改变输出分布，不 crash。
- **分类**：解码/采样（推测解码正确性） → 第 11 类（新子签名：spec_decode_q_distribution_mismatch）

#### 3. vllm #52109: DeepSeek-V4-Flash ROCm gfx942 长 prompt 检索静默损坏 — `[已知类别，新签名]`
- **来源**：vllm #52109（2026-08-13 open）https://github.com/vllm-project/vllm/issues/52109
- **描述**：DeepSeek-V4-Flash 在 8×MI325X（gfx942）上，needle-in-haystack 检索在短 prompt（≤3,611 tokens）完美（3/3），在 ≥5,294 tokens 完全崩溃（0/3）。**无 crash、无 error，健康服务器静默返回错误结果**。GSM8K 短 prompt 正确性不受影响。AITER sparse indexer 的问题——gfx942 版本的 #40018（gfx950 prompt_len > 20K garbage），阈值更低（~4-5k）。`max_model_len`、chunked-prefill boundary、split-KV heuristic 均排除。
- **分类**：服务栈/缓存（稀疏注意力长上下文） → 第 6 类（新子签名：gfx942_sparse_sw_retrieval_collapse）

#### 4. vllm #51667: DiskBackend KV offload cross-batch buffer race 静默损坏 GPU KV cache — `[已知类别，新签名]`
- **来源**：vllm #51667（2026-08-10 open PR）https://github.com/vllm-project/vllm/pull/51667 · 基于 #49644
- **描述**：DiskBackend（KV offloading）的 `_do_load` double-buffer pipeline 中，pending H2D events 在每次调用中追踪在 local list，与 `_do_store` 不同——**返回前从不 drain**。后续 `_do_load` 调用的 preadv 可在上一调用的 trailing H2D copy 仍在读取时覆盖 buffer slot，**静默损坏 reload 时的 GPU KV cache**。PR 在 `_do_load` 返回前 drain trailing events，镜像 `_do_store` 模式。
- **分类**：服务栈/缓存（KV offload） → 第 6 类（新子签名：disk_offload_double_buffer_race）

---

## 🟡 已知类别的新案例

#### 5. vllm #52225: SM120 持续多小时负载下 recurring Xid 13 warp errors — `[已知类别]`
- **来源**：vllm #52225（2026-08-13 open）https://github.com/vllm-project/vllm/issues/52225
- **描述**：Nemotron-3.5-Lightning-30B-A3B-NVFP4（marlin MoE, hybrid Mamba）在 RTX PRO 6000 Blackwell（SM120）持续多小时负载下出现 **chip-wide recurring Xid 13 errors**（misaligned address / illegal instruction / out-of-range register warp errors）。不是单点故障——chip 范围 warp 错误，表明硬件级 SDC 或 kernel 不稳定性在持续负载下累积。
- **分类**：硬件/SDC → 第 1 类

#### 6. vllm #52207: GDN non-spec state_indices strided view 非 contiguous — `[已知类别]`
- **来源**：vllm #52207（2026-08-13 open）https://github.com/vllm-project/vllm/pull/52207
- **描述**：GDNAttentionMetadataBuilder 的 non-spec decode 分支设 `non_spec_state_indices_tensor = block_table_tensor[:, 0]`——strided column view（stride=block_width，非 1）。`.contiguous()` 对 ≤1 element tensor 是 no-op，导致 recurrent decode kernel 收到 stride≠1 的 tensor。kernel 报错 `state_indices must be contiguous and one-dimensional`。需要 `clone(memory_format=torch.contiguous_format)` 保证 stride(0)==1。
- **分类**：服务栈/缓存（recurrent state kernel） → 第 6 类

#### 7. vllm #52205: Kimi-K3 KDA recurrent kernel FULL cudagraph 下 GPU fault — `[已知类别]`
- **来源**：vllm #52205（2026-08-13 open）https://github.com/vllm-project/vllm/pull/52205
- **描述**：Kimi-K3 KDA op 装饰 `@eager_break_during_capture`，但 `eager_break_during_capture` 的 FULL runtime-mode fast-path 会 inline 被装饰的调用到 graph 中。当 FULL uniform-decode batch 运行 KDA（如 FULL_AND_PIECEWISE 下的 spec-decode verify 路径），recurrent kernel（`fused_recurrent_kda`/`chunk_kda_with_fused_gate`/`causal_conv1d_update`）被捕获到 FULL graph，GPU fault during capture/replay：`Memory access fault ... Reason: Unknown`。target-only cudagraph（decode, no spec）正常。
- **分类**：服务栈/缓存（cudagraph capture） → 第 6 类

#### 8. vllm #51798: Kimi-K3-NVFP4 8×B300 reasoning channel 退化输出 — `[已知类别]`
- **来源**：vllm #51798（2026-08-11 open）https://github.com/vllm-project/vllm/issues/51798
- **描述**：RedHatAI/Kimi-K3-NVFP4 在 8×B300, TP8, v0.27.0 上产生**退化、不连贯的 reasoning channel 输出**。生产部署已回滚。nvfp4 量化 + 特定版本组合导致推理通道质量崩溃。
- **分类**：量化/精度 → 第 7 类

#### 9. vllm #51771: EAGLE/MTP block drop + prefix caching 对 ≥3 attention group 混合模型未测试 — `[已知类别]`
- **来源**：vllm #51771（2026-08-11 open）https://github.com/vllm-project/vllm/issues/51771
- **描述**：DeepSeek-V4-Flash + DSpark 构建 4 attention groups（1 full MLA + 3 sliding-window family），`use_eagle()` 返回 True，但 EAGLE block-drop 的 general convergence loop 对此 shape 未经 CI 测试（所有测试仅覆盖 ≤2 groups）。结果：prefix-cache 完全失效（0.0% cache loss）。SWA attn 不满足 downward-closed property，cache hit at token j 不保证 cache hit at i<j。
- **分类**：服务栈/缓存（prefix caching） → 第 6 类

#### 10. vllm #51766: Mamba running CoW 在 external hits 后误分类 — `[已知类别]`
- **来源**：vllm #51766（2026-08-11 open）https://github.com/vllm-project/vllm/pull/51766
- **描述**：`allocate_external_computed_blocks()` 可填充 request 的 Mamba block table，但首次 continuation 留在同一 Mamba block 时，`allocate_new_blocks()` 提前返回不将 request 加入 `_allocated_block_reqs`。后续 partial-tail CoW 被误分类为 first-prefill CoW——Core 替换 live block 并返回 replacement，worker 视 request 为 running 并 append returned block IDs。**Core 和 worker 对 live Mamba state column 的理解不一致**。
- **分类**：服务栈/缓存（block management） → 第 6 类

#### 11. transformers #47946: GPT2 encoder_attention_mask 被静默丢弃 — `[已知类别]`
- **来源**：transformers #47946（2026-08-13 open）https://github.com/huggingface/transformers/pull/47946
- **描述**：`GPT2Model` 第 591 行 `encoder_attention_mask = None` 在传递给 `create_bidirectional_mask` 前覆盖用户定义的 mask。padding token 不被忽略，corrupted padding values 影响 cross-attention 计算。由 masking refactor #42848 引入。DecisionTransformer 同样受影响（复制 GPT2 代码）。
- **分类**：模型图/架构 → 第 13 类

#### 12. sglang #34765: FP8 MLA KV scale propagation 不一致 — `[已知类别]`
- **来源**：sglang #34765（2026-08-13 open）https://github.com/sgl-project/sglang/pull/34765
- **描述**：校准 FP8 MLA KV cache 必须用相同 scale convention 写和读。某些 MLA 和 DSA 路径写 cache 时用 unit scale，BMM1 仅恢复 K scale，BMM2 省略 V scale。MLA 存储一个 compressed latent row 同时作为 BMM1 的 K 和 BMM2 的 V，K 和 V descales 必须一致——不一致导致非 unit 校准在 attention 操作中 inconsistent。
- **分类**：量化/精度 → 第 7 类

#### 13. sglang #34760: Mamba state donation misalignment under DCP — `[已知类别]`
- **来源**：sglang #34760（2026-08-13 open）https://github.com/sgl-project/sglang/pull/34760
- **描述**：Hybrid linear-attention 模型（Kimi-K3, 69 KDA + 24 MLA）以 `--dcp-size > 1` + Mamba radix cache 服务时，从 cached prefix resume 产生的 logits **确定性地偏离** fresh prefill（per-token |dlogprob| up to 6.9, 157/1081 位置 |dlp|>2），从 resume boundary 开始衰减 ~700 tokens。根因：DCP 加宽 tree page 但 Mamba state-tracking grid 未加宽。radix key floor 到加宽 page 但 donated recurrent state 仍编码 `mamba_last_track_seqlen` tokens，state 可领先 node 达 384 tokens。
- **分类**：服务栈/缓存（Mamba state） → 第 6 类

#### 14. sglang #34740: SGLANG_SIMULATE_ACC_LEN 静默降级 detokenization 为 O(n²) + byte-fallback 死锁 — `[已知类别，新签名]`
- **来源**：sglang #34740（2026-08-13 open）https://github.com/sgl-project/sglang/issues/34740
- **描述**：两个独立缺陷复合：(A) `generate_simulated_accept_index()` 用硬编码 `predict.fill_(100)` 覆盖所有 accepted token——在 byte-fallback vocab 中 id 100 解码为 U+FFFD（DeepSeek-V4-Pro ids 97–111 均为 `�`），整个输出变为 `�` 串；(B) `DetokenizerManager` 的 commit gate 用 `endswith("\ufffd")` 字符级启发式，byte-fallback token 使 incremental detokenization offset 永不前进，降级为 O(n²)。
- **分类**：分词器/模板 → 第 10 类

#### 15. sglang #34742: DSpark draft logits 当 target lm_head 量化时 shape/dtype 错误 — `[已知类别]`
- **来源**：sglang #34742（2026-08-13 open）https://github.com/sgl-project/sglang/pull/34742
- **描述**：DSpark draft 借用 target 的 `lm_head` 并通过 dense matmul on `.weight` 投影。当 target 检查点量化 lm_head（如 ModelOpt NVFP4 export），weight 被打包为 `uint8 [vocab, hidden/2]`，matmul 获得错误 K，`hidden.to(weight.dtype)` cast 将 bf16 activations 重解释为 uint8。CUDA graph capture 时 `mat1 and mat2 shapes cannot be multiplied (51x5120 and 2560x248320)`。eager 路径同样失败。
- **分类**：量化/精度（推测解码） → 第 7 类

#### 16. sglang #34701: MoE expert-ID namespace 不匹配导致 EP>1 silent OOB write — `[已知类别，新签名]`
- **来源**：sglang #34701（2026-08-13 open）https://github.com/sgl-project/sglang/pull/34701
- **描述**：`StandardDispatcher` 从 `get_moe_runner_backend()` 决定是否 remap topk_ids 到 rank-local expert range——仅 AITER runner 保持 global expert IDs。但配置的 backend 不一定是该层实际拿到的 runner backend。compressed-tensors W4A16 总是构建 Triton runner，但 ROCm 上配置 `auto` + AITER 可用时 dispatcher 误判 "AITER runner, keep global IDs"，实际运行 Triton kernels（sized `num_experts/ep_size`）。Kimi-K2.5 EP8 时 expert 383 索引 48-entry buffer——**silent out-of-bounds write**。warmup 时每个 rank GPU fault。
- **分类**：模型图/架构（MoE 路由） → 第 13 类

#### 17. trt-llm #17658: KVCM V2 page tables 在 Confidential Compute 下 GPU fault — `[已知类别]`
- **来源**：trt-llm #17658（2026-08-13 open）https://github.com/NVIDIA/TensorRT-LLM/pull/17658 · nvbugs/6248648
- **描述**：KVCM V2 将四个 host tensor 传给直接从 GPU 解引用它们的 CUDA kernel。Confidential Compute 无法将 host allocation 映射到 GPU 地址空间，kernel fault 而非执行 H2D transfer。需在 host mapping 不可用时 gather active base-page rows 到 device staging。
- **分类**：服务栈/缓存（KV cache page table） → 第 6 类

#### 18. trt-llm #17656: MiniMax-M3 Eagle draft sub-page pool bound 错误 — `[已知类别]`
- **来源**：trt-llm #17656（2026-08-13 open）https://github.com/NVIDIA/TensorRT-LLM/pull/17656
- **描述**：`MiniMaxM3DraftSubpageView` 将 `blocks_in_primary_pool` 委托给 wrapped target `MiniMaxM3KVCacheManagerV2`——该 bound 以 target manager 的 128-token page 单位表示，相对不同 pool base。FlashInfer 构建 draft paged-KV tensor/TMA descriptor 时 under-report flat page count。draft block-table entries 指向 descriptor 声明范围外的物理 slot，**产生错误 draft KV reads**。Target logits 和最终 token 正确——表现为 cold-start acceptance collapse（1.8–2.0 → 3.3+）而非输出正确性失败。
- **分类**：服务栈/缓存（speculative decode draft KV） → 第 6 类

#### 19. trt-llm #17664: AutoTuner profiling cache 跨设备加载 — `[已知类别]`
- **来源**：trt-llm #17664（2026-08-13 open）https://github.com/NVIDIA/TensorRT-LLM/pull/17664
- **描述**：`AutoTunerProfilingCache` 写入 `lib_version`/`device_name`/`device_capability` 但从不读回。cache file 从另一 GPU 加载时不报错，`get_cache_key` 不含 device——不同设备的 entry 可 hit 并返回从未在此设备测量的 tactic，日志无任何提示。可能导致次优或错误的 kernel tactic 被静默选择。
- **分类**：服务栈/配置 → 第 6 类

#### 20. trt-llm #17635: 大型混合注意力 FP8 serving 的 MoE/MTP fixes — `[已知类别]`
- **来源**：trt-llm #17635（2026-08-13 open）https://github.com/NVIDIA/TensorRT-LLM/pull/17635 · TRTLLM-15304
- **描述**：(1) FP8 MoE activation quantization clamp block amax 到 1e-10——零 block 产生 finite scale 和零 FP8 output；(2) MoE workspace sizing 改用 activation-specific token capacity；(3) cooperative routing 仅应用于支持的 large expert tier；(4) Qwen3NextMTP.forward 临时应用 draft token counts 并在所有 exit path 恢复原始 metadata。
- **分类**：量化/精度 + 服务栈 → 第 7 类

#### 21. tt-metal #53073: MiniMax-H3 audio T-sharding conv_pre 返回未初始化内存 — `[已知类别，新签名]`
- **来源**：tt-metal #53073（2026-08-13 open）https://github.com/tenstorrent/tt-metal/pull/53073
- **描述**：MiniMax-H3 audio decode T-sharded 路径两个 correctness fix：(1) `conv_pre`（2048→1024, k=7）在 sharded path 返回**未初始化内存**（absmax 2.6e+11 vs reference 1.391，即 −204 dB）——这是每个 T-sharded decode 返回错误音频的原因；(2) `tap_matmul` 在 T-sharding 下需要 full split_mode。修复：conv_pre 在 full sequence 上 unsharded 运行，T 立即在其后 partition。
- **分类**：硬件/SDC 内核原语 → 第 1 类

---

## 📄 arxiv 新发现

#### A1. Runtime Observability for Heterogeneous Attention Memory — `[新论文，工具/检测]`
- **来源**：arxiv 2608.05863（2026-08-06）https://arxiv.org/abs/2608.05863 · 开源 https://github.com/metask-ai/witprobe-attention-memory
- **描述**：现代模型不再保持纯 KV cache——latent cache、learned sparse selector 和 recurrent state 各以不同形式承载模型记忆，且在压缩下各以不同方式失败。提出 runtime observability contract 覆盖所有四类 memory（plain KV, latent, sparse, recurrent），三类 operator，实例化于六种模型配置五种架构族。应用于 served DeepSeek-V4 stack with packed compressed-KV prototype，**将一次 silent corruption 定位到精确的 structural boundary**——在 eviction-free、identity-isolated regime 中精确，所有观察到的 failure 在 eviction 或 slot-reuse regime。fused always-on probe 在 CUDA graph 内的 serving noise floor 以下观测声明的一层子集。
- **分类**：KV 缓存/注意力检测工具 → 第 8 类（作为检测/工具补充来源，强相关）

#### A2. From Bit-Position Sensitivity to Unequal Error Protection for DNN Inference Memory — `[新论文，BFA/SDC 变体]`
- **来源**：arxiv 2607.19623（2026-07-21）https://arxiv.org/abs/2607.19623
- **描述**：跨 16 个 workload（transformer + attention-free CNN）和三种浮点格式刻画 per-bit-position fault sensitivity。核心发现：**sharp bit-sensitivity transition**——翻转任何最低有效 fraction bit 到 data-type 特定阈值 Xsafe 以下，在确定性单 bit stress test 下 degrade task metrics <1%。敏感度在 upper fraction bits 上升，在 **exponent-mantissa boundary 处剧增**（单 bit flip 造成 catastrophic collapse）。推导 per-data-type Xsafe floors（FP16: 6, BF16: 4, FP32: 15），workload-aware tiers 可将 ECC savings 提至 37.5–62.5 而无需 retrain。Text-conditioned diffusion model 决定保守 floor；vision encoder、NLU model 和 resilient LLM 容忍更宽 bypass region。
- **分类**：比特翻转攻击 / 硬件 SDC → 第 3 类（新子签名：uep_bit_position_sensitivity_map）

---

## 📝 API 调用记录

| # | 查询 | 结果 |
|---|------|------|
| 1 | vllm label:bug created:>2026-08-05 | 15 items |
| 2 | vllm wrong output inference created:>2026-08-05 | 7 items |
| 3 | vllm NaN inference created:>2026-08-05 | 3 items |
| 4 | vllm KV cache corruption created:>2026-08-05 | 12 items |
| 5 | sglang correctness OR wrong created:>2026-08-05 | 12 items |
| 6 | sglang silent OR corruption OR garbage created:>2026-08-05 | 12 items |
| 7 | TensorRT-LLM correctness OR wrong OR NaN created:>2026-08-05 | 12 items |
| 8 | triton wrong result OR numerical OR incorrect created:>2026-08-05 | 9 items |
| 9 | tt-metal wrong result OR incorrect OR corruption created:>2026-08-05 | 12 items |
| 10 | transformers wrong output OR NaN OR correctness created:>2026-08-05 | 10 items |
| A1-A7 | arxiv 7 组关键词查询 | 35 results |

### 去重检查

| issue/PR | 状态 |
|----------|------|
| #52225, #52207, #52205, #52109, #51798, #51667, #51766, #51771 | NEW |
| #34765, #34760, #34740, #34701, #34742 | NEW |
| #47942, #47932, #47946 | NEW |
| #17635, #17656, #17658, #17664 | NEW |
| #53073 | NEW |
| #52150, #52107, #47886 | 已存在于主文档（跳过） |
| arxiv 2608.05863, 2607.19623 | NEW |
