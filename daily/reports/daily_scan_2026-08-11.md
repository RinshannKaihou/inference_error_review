# LLM 推理故障每日扫描 — 2026-08-11

- **扫描日期**：2026-08-11（窗口 2026-08-04 ~ 2026-08-10）
- **搜索来源**：GitHub Issues/PR Search API（10 次查询）+ arxiv API（8 次查询）
- **目标仓库**：vllm-project/vllm · sgl-project/sglang · NVIDIA/TensorRT-LLM · huggingface/transformers · triton-lang/triton · tenstorrent/tt-metal

---

## API 调用记录

| # | 查询 | 命中数 |
|---|------|--------|
| 1 | vllm label:bug created:>2026-08-04 | 30 |
| 2 | vllm silent/wrong/incorrect/corruption/NaN in:title | 20 |
| 3 | vllm KV cache/quantization/bit flip in:title | 30 |
| 4 | sglang silent/wrong/corruption/NaN in:title | 16 |
| 5 | TensorRT-LLM wrong/incorrect/corruption/silent/NaN in:title | 3 |
| 6 | transformers wrong/incorrect/corruption/silent/NaN in:title | 8 |
| 7 | triton wrong/incorrect/numerical in:title | 1 |
| 8 | tt-metal wrong/corruption/silent/NaN/numerical in:title | 20 |
| 9 | vllm FP8/NVFP4/INT4/mixed precision label:bug | 20 |
| 10 | cross-repo "silent data corruption"/rowhammer/"non-deterministic" | 20（噪声） |

arxiv 查询：silent data corruption LLM · bit flip LLM · KV cache corruption · LLM inference correctness · GPU soft error ML · quantization error LLM · weight corruption LLM · fault injection NN GPU

---

## 扫描结果（新发现 — 去重后）

### 🔴 新故障模式 / 新子签名

#### 1. vLLM `persistent_topk` 静默丢弃 top-k 候选 — `[新签名]`
- **来源**：vLLM #51782（2026-08-11 open） https://github.com/vllm-project/vllm/issues/51782
- **描述**：`torch.ops._C.persistent_topk` 在 logits 值域窄、多元素落入同一粗直方图 bin 时静默返回错误结果——**丢弃真实 top-k 候选并替换为更小的值**。失败是数据相关的（非纯 shape），k=2048 + 长行最易触发。输出无 -1 slot、无重复索引，只能通过与参考对比才能发现。
- **分类**：数值正确性 / 选择算法缺陷 → 第 4 类（数值/精度）

#### 2. SGLang mxfp4 + sharded_state 静默丢弃 expert 权重 — `[新签名]`
- **来源**：SGLang #34448（2026-08-11 open） https://github.com/sgl-project/sglang/issues/34448
- **描述**：`save_sharded_state.py` 导出 mxfp4 检查点时**静默丢弃 expert 权重**。`openai/gpt-oss-20b` 导出后的 safetensors 总大小显著小于原始检查点。
- **分类**：权重/检查点加载 → 第 12 类

#### 3. SGLang B300 sm_103 两个 SM10x-gated kernel 静默失败 — `[新签名]`
- **来源**：SGLang #34340（2026-08-11 open） https://github.com/sgl-project/sglang/issues/34340
- **描述**：两个 SM10x 专属 kernel 在 B300（sm_103）上失败：(a) cutedsl TGV BF16 GEMM 抛 Xid 13 CGA "CTA Not Present"；(b) trtllm-gen MoE finalize **静默挂起**。根因是 `is_sm100_supported()` 是家族检查（major==10），sm_103 走了 SM100 专属路径。
- **分类**：硬件/内核原语 → 第 1 类

#### 4. SGLang MiniMax-H3 FSDP-inference 静默损坏视频/音频输出 — `[新签名]`
- **来源**：SGLang #34227（2026-08-10 open） https://github.com/sgl-project/sglang/issues/34227
- **描述**：`--use-fsdp-inference true` 服务 MiniMax-H3 时**静默产出损坏输出**（视频帧被压成暗色低对比图像、音频输出为噪声），**服务器日志无任何错误或警告**。移除 flag 产出正确视频+音频。
- **分类**：分布式/多 GPU → 第 5 类

#### 5. SGLang LFM2-VL 多模态缓存 hit 返回错误长度嵌入 — `[已知类别]`
- **来源**：SGLang #33751（2026-08-05 PR） https://github.com/sgl-project/sglang/pull/33751
- **描述**：per-image cache hit 可返回**错误长度嵌入**——在 H100 C=32 压力下，混合前缀 batch 间歇性获得 7,126 个缓存 encoder token（应为 7,135），流入致命的 insufficient-embedding 分支导致 scheduler 崩溃。
- **分类**：KV 缓存 / 多模态 → 第 8 类

#### 6. SGLang Gemma4UnifiedMultimodalEmbedder 使用错误输入维度 — `[已知类别]`
- **来源**：SGLang #34392（2026-08-11 open） https://github.com/sgl-project/sglang/issues/34392
- **描述**：`Gemma4UnifiedMultimodalEmbedder` 用 `output_proj_dims` 作为 `nn.Linear` 的 in_features（文档说是输出维度）。vision embedder 因巧合匹配（3840=3840）侥幸通过，但 audio embedder 用 output_proj_dims（应为 audio_embed_dim=640）→ Linear in_features 错误 → 检查点 weight 加载失败。
- **分类**：模型图/架构 → 第 13 类

#### 7. tt-metal `ttnn.group_norm` 非 tile 对齐 H*W 静默错误 — `[已知类别]`
- **来源**：tt-metal #52685（2026-08-10 open） https://github.com/tenstorrent/tt-metal/issues/52685
- **描述**：`ttnn.group_norm` 在输入 tile padding 非零时对非 tile 对齐 H*W **静默错误**——假设 padding 行为零，但 `exp`/`softmax`/`slice`/`reshape` 等 producer 留下非零 padding。所有现有测试用 `from_torch` 零 padding 构建输入，因此整个测试套件通过。`fill_implicit_tile_padding` 在 BLOCK_SHARDED tensor 上**损坏数据**。
- **分类**：硬件/SDC 内核原语 → 第 1 类

#### 8. tt-metal `ttnn.pow` 对 x<0 静默错误（polygamma 同族） — `[已知类别]`
- **来源**：tt-metal #52278（2026-08-06 open） https://github.com/tenstorrent/tt-metal/issues/52278
- **描述**：`ttnn.polygamma` 在固定偏移 z=x+6 处应用 Euler-Maclaurin 尾校正，但展开式仅对大 z 有效。x 趋负时 z 缩小，展开式离开定义域——返回静默错误有限值（含符号翻转），无验证、无文档化范围。
- **分类**：硬件/SDC 内核原语 → 第 1 类

#### 9. tt-metal SFPU clamp 在域外返回静默常量 — `[已知类别]`
- **来源**：tt-metal #52178（2026-08-05 open） https://github.com/tenstorrent/tt-metal/issues/52178
- **描述**：5 个 SFPU unary kernel（erfc/digamma/softplus/i1）在函数非定义域区域返回**常量**而非正确值。共享根因：SFPU 饱和不留痕迹。
- **分类**：硬件/SDC 内核原语 → 第 1 类

#### 10. tt-metal MoE `selective_reduce_combine` 错误 dense token-map stride — `[已知类别]`
- **来源**：tt-metal #52371（2026-08-06 open） https://github.com/tenstorrent/tt-metal/issues/52371
- **描述**：`selective_reduce_combine` 用 `total_tokens` 除 expert row 推导 stride，但每行有 `total_tokens + 1` 个条目（含 -1 terminator）。reader/writer kernel 也用 `global_num_tokens + 1` 索引。结果：host 传给 kernel 的 per-token-index stride 错误。
- **分类**：硬件/SDC 内核原语 → 第 1 类

#### 11. transformers ASR pipeline 静默破坏立体声音频 — `[新签名]`
- **来源**：transformers #47886（2026-08-10 open） https://github.com/huggingface/transformers/issues/47886
- **描述**：`AutomaticSpeechRecognitionPipeline` 处理 `soundfile` 返回的 channels-last 立体声波形时，`mean(axis=0)` **沿时间轴平均**（而非沿通道轴），静默丢弃几乎所有音频。3 秒语音被转写为 `' you'`。
- **分类**：模型图/架构 → 第 13 类

#### 12. transformers WhisperFeatureExtractor 单个 NaN 样本→整矩阵 NaN — `[已知类别]`
- **来源**：transformers #47885（2026-08-10 open） https://github.com/huggingface/transformers/issues/47885 · 修复 PR #47894
- **描述**：波形中单个非有限样本使 `WhisperFeatureExtractor` 返回**全 NaN** 特征矩阵，ASR pipeline 从中产生**自信转写**，无错误无警告。根因：`np.max`/`torch.max` 对含 NaN 数组返回 NaN，`np.maximum(x, NaN)` 传播到每个 cell。
- **分类**：数值/精度 → 第 4 类

#### 13. vLLM Gemma-4 NVFP4 fp8 KV cache 在 Blackwell 静默回退 FA2 后崩溃 — `[已知类别]`
- **来源**：vLLM #51828（2026-08-11 open） https://github.com/vllm-project/vllm/issues/51828 · 修复 PR #51849
- **描述**：Gemma-4 31B NVFP4 + fp8 KV cache 在 Blackwell 上：后端选择在无 local-attention 信息时解析 FA4，运行时带 sliding_window 降级到 FA2，FA2 不支持 fp8 KV → warmup 崩溃。选择和运行时看到不同的有效 FA 版本。
- **分类**：量化/后端选择 → 第 7 类

#### 14. vLLM ROCm DSpark 推测解码 KV cache 布局重叠→静默 NaN — `[新签名]`
- **来源**：vLLM #51837（2026-08-11 PR） https://github.com/vllm-project/vllm/pull/51837
- **描述**：MI300 上 DSpark 推测解码 + Qwen3.6-NVFP4 产出垃圾（GSM8K 0.20 vs 0.95）。ROCm KV cache 布局 `(2, num_blocks, block_size, num_kv_heads, head_size)` 使 K/V 分居分配两端，block b 实际覆盖 page b/2 的半页 + page num_blocks/2+b/2 的半页。Attention block 和 Mamba state 页面解析到**重叠字节**互相覆盖。损坏**静默**——NaN 输出在有效上下文范围内。
- **分类**：KV 缓存/注意力 → 第 8 类

#### 15. vLLM 混合 KV 布局 fine-grained prefix-cache hit 导致 assertion — `[已知类别]`
- **来源**：vLLM #51843（2026-08-11 PR） https://github.com/vllm-project/vllm/pull/51843
- **描述**：含 Mamba align 组的混合模型启用 fine-grained prefix-cache hit，但 DSpark drafter 的 sliding-window KV cache manager 只支持 block-aligned lookup → assertion 崩溃。
- **分类**：服务栈/缓存 → 第 6 类

#### 16. vLLM KV block zeroing 只处理 full-attention 组→FP8 SWA 页面残留数据 — `[已知类别]`
- **来源**：vLLM #51749（2026-08-11 PR） https://github.com/vllm-project/vllm/pull/51749
- **描述**：`needs_kv_cache_zeroing` 对混合 Mamba 和混合精度 cache 启用，但 scheduler 只报告 full/MLA block 类型、worker zeroer 只处理 full-attention 组。回收的 FP8 sliding-window 页面**残留旧数据**。Kimi-K3 DSpark drafter SWA GQA 上观察到 acceptance length 从 4.8 在后续 GSM8K batch 上**塌缩**。
- **分类**：KV 缓存/注意力 → 第 8 类

#### 17. vLLM sliding window 从 KV cache group 而非 layer 读取→全局层丢失旧 token — `[已知类别]`
- **来源**：vLLM #51756（2026-08-11 PR） https://github.com/vllm-project/vllm/pull/51756
- **描述**：Gemma-3 `--disable-hybrid-kv-cache-manager` 将 sliding-window 层提升为 full-attention 存储，合并组仍记录 window。FlashAttention 从组 spec 读 window 并应用到组内每层，**全局层静默丢失 window 以外的所有旧 token**。
- **分类**：KV 缓存/注意力 → 第 8 类

#### 18. vLLM EAGLE volatile-tail 边界塌缩→零外部 token 恢复 — `[已知类别]`
- **来源**：vLLM #51761（2026-08-11 open） https://github.com/vllm-project/vllm/issues/51761
- **描述**：混合 EAGLE/MTP + 小 sliding-window 组下，CPU KV offload 在 EAGLE volatile-tail pop 将共享 replay 边界推到 full-attention 对齐边界以下时**恢复零外部 token**。C4 组需 chunk 7534/7535 但普通 per-segment store 只保留 7550/7551 的 prompt-end tail。
- **分类**：KV 缓存/注意力 → 第 8 类

#### 19. vLLM DeepSeek-V4 fused qnorm/rope/kv-insert 大 batch 崩溃 — `[已知类别]`
- **来源**：vLLM #51743（2026-08-11 open） https://github.com/vllm-project/vllm/issues/51743
- **描述**：`--max-num-batched-tokens` ≥ 24576 在持续负载几分钟后致命崩溃 EngineCore。**危险**：引擎初始化干净、通过 readiness、单请求正确——只有在真实流量组装大 batch 时才崩溃。堆栈在 `fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert`。分配对 memory profiler 不可见。
- **分类**：服务栈/缓存 → 第 6 类

#### 20. vLLM MultiConnector 重复 recving completion → scheduler 崩溃 — `[已知类别]`
- **来源**：vLLM #51786（2026-08-11 open） https://github.com/vllm-project/vllm/issues/51786
- **描述**：`MultiConnector` + 多 consumer connector（NixlConnector + LMCacheMPConnector）PD 部署下，`get_finished_recving_reqs()` 在两个不同 scheduler step 返回同一 `req_id`。Step N 从 Connector A 收到 completion→WAITING_FOR_KV→RUNNING。Step N+k 从 Connector B 收到同一 req_id→assertion 崩溃。
- **分类**：服务栈/缓存 → 第 6 类

#### 21. vLLM silent -inf logprobs（logprob_token_ids 与 logprobs=k 混批） — `[已知类别]`
- **来源**：vLLM #51789（2026-08-11 PR） https://github.com/vllm-project/vllm/pull/51789
- **描述**：`logprob_token_ids` 请求与 `logprobs=k` 请求同 batch 时，V1 Sampler.forward 用 specific-token gather 替换整个 batch 的 top-k 结果，其他请求得到 `-inf` 行（API: 一个 `<pad>: -9999.0` 条目）。文本不受影响→**静默失败**。两个并发请求即可触发。
- **分类**：解码/采样 → 第 11 类

#### 22. vLLM prompt_logprobs=0 truthiness 绕过 guard → engine crash — `[已知类别]`
- **来源**：vLLM #51846（2026-08-11 PR） https://github.com/vllm-project/vllm/pull/51846
- **描述**：`prompt_logprobs=0` 是有效值（"返回 prompt token 自身 logprob"），但多处用 truthiness 而非 `is not None` 检查→0 静默滑过。绕过 `--kv-sharing-fast-prefill` 不兼容 guard 和 `stream=True` 拒绝，下游 `is not None` 检查命中 bare `assert`→**杀死整个 EngineCore 进程**（所有在途和未来 client）。
- **分类**：解码/采样 → 第 11 类

#### 23. vLLM Mooncake store token_len 与 block_hashes 范围不匹配 — `[已知类别]`
- **来源**：vLLM #51820（2026-08-11 PR） https://github.com/vllm-project/vllm/pull/51820
- **描述**：推测解码（MTP）+ async scheduling 下，scheduler 在 `block_hashes` 扩展前乐观推进 token count→assertion 崩溃。clamp 修复崩溃但留下副作用：`_record_saved` 记录乐观 token count 而非实际保存的→worker resume offset 越过 clamped 范围，跳过的 token **永不重试**，外部 Mooncake store 留下小间隙。
- **分类**：服务栈/缓存 → 第 6 类

#### 24. vLLM DeepSeek-V4-Flash-0731 B300 1M KV 并发 3.5x 退化 — `[已知类别]`
- **来源**：vLLM #51799（2026-08-11 open） https://github.com/vllm-project/vllm/issues/51799
- **描述**：B300 上 DeepSeek-V4-Flash-0731 的 1M 上下文最大并发从 preview 的 27.08x 降到 0731+DSpark 的 7.73x（**3.5x 退化**）。远超正常 DSpark 开销。KV-per-token 增加类问题。
- **分类**：服务栈/缓存 → 第 6 类

#### 25. vLLM ROCm DeepSeek-V4 input GEMM preshuffled weight 被当普通 linear→全垃圾 — `[新签名]`
- **来源**：vLLM #51821（2026-08-11 PR） https://github.com/vllm-project/vllm/pull/51821
- **描述**：gfx950 上 `amd/DeepSeek-V4-Flash-NVFP4` GSM8K 准确率 0.0000（阈值 0.92）。根因：commit #51430 删除 `DeepseekV4Attention._fused_wqa_wkv_gemm` 并内联调用。该方法是 ROCm attention layer 的 override point——ROCm 上 `fused_wqa_wkv` 权重在加载时 **preshuffled**，只能由 AITER preshuffled-B GEMM 正确乘。内联后每个 attention layer 的 input projection 对 shuffled weight 跑普通 linear→**从第一层起产出垃圾**。CUDA 上不可见，ROCm override 读起来仍像连接的。
- **分类**：模型图/架构 → 第 13 类

#### 26. vLLM CuTe FA4 forward 首次调用 NameError（sm_120） — `[已知类别]`
- **来源**：vLLM #51776（2026-08-11 open） https://github.com/vllm-project/vllm/issues/51776
- **描述**：`vllm_flash_attn/cute/flash_fwd.py` 两个独立 Python 级 bug（`mDynamicCausal` NameError + 缺少 `self.is_split_kv`）使 SM80/SM120 FA4 在**首次调用**即失败。SM80 用 C++ FA2/FA3 kernel 所以 CuTe DSL 路径实质 dead，但 sm_120 上被选中。
- **分类**：模型图/架构 → 第 13 类

#### 27. vLLM cudaHostRegister 混合 pinned/unpinned TP ranks — `[已知类别]`
- **来源**：vLLM #51762（2026-08-11 open） https://github.com/vllm-project/vllm/issues/51762
- **描述**：CPU KV offload 各 rank 独立调用 `cudaHostRegister`，失败时 warn-and-continue。重复启动产生混合状态：一个 rank pinned、另一个 unpinned。同步传输性能被 unpinned rank 限制，pinned rank 仍付 bookkeeping 成本。
- **分类**：分布式/多 GPU → 第 5 类

#### 28. TRT-LLM interior control-token rejection 静默分裂 tool-call prelude — `[新签名]`
- **来源**：TRT-LLM #17437（2026-08-08 open） https://github.com/NVIDIA/TensorRT-LLM/issues/17437
- **描述**：单模型推测解码中，acceptance 路径只返回 `accepted_tokens` 和 `num_accepted_tokens`，从不检查 accepted token 流是否结构完整。当 draft 提议 tool-call 控制前导（CH MSG TOK END）且 target 不青睐某个内部 control token 时，accepted prefix 不完整：打开的控制块永不闭合，caller/parser 报 `finish_reason=stop` + 空 tool_calls + 空 content。模型明显意图是函数调用，调用却被**静默丢弃**。非推测解码下相同 well-formed draft 产出完整调用——差异只在 draft/accept 边界。
- **分类**：解码/采样 → 第 11 类

#### 29. transformers cached_files 只读文件系统静默返回旧缓存 — `[已知类别]`
- **来源**：transformers #47852（2026-08-09 PR） https://github.com/huggingface/transformers/pull/47852 · 根因 issue #47838
- **描述**：HF Hub cache 在只读文件系统上时，`hf_hub_download` 尝试写 snapshot pointer 失败（EROFS errno 30）。Python 只把 EACCES（13）提升为 PermissionError，EROFS（30）是普通 OSError，现有 `isinstance(e, PermissionError)` guard 漏检→stale-cache recovery 块**静默返回旧缓存文件**，即使 Hub 有更新版本。
- **分类**：分词器/加载 → 第 10 类

#### 30. transformers additional_special_tokens 迁移静默丢弃 — `[已知类别]`
- **来源**：transformers #47838（2026-08-07 open） https://github.com/huggingface/transformers/issues/47838 · 修复 PR #47855
- **描述**：`tokenizer_config.json` 同时含 `additional_special_tokens` 和 `extra_special_tokens` 时，`setdefault("extra_special_tokens", init_kwargs.pop("additional_special_tokens"))` 中 `.pop()` 总执行但 `setdefault` 不覆盖已存在的 key → popped list 被丢弃，tokenizer 最终**两者都收不到**。
- **分类**：分词器/模板 → 第 10 类

---

### arxiv 论文发现

| arxiv ID | 标题 | 相关性 | 状态 |
|----------|------|--------|------|
| 2608.05863 | Runtime Observability for Heterogeneous Attention Memory | **强相关**：形式化 KV/latent cache 运行时可观测性契约，在 DeepSeek-V4 上定位**静默损坏**到精确结构边界 | 新 |
| 2608.07408 | Addressable Memory for Video World Models | 中：长 horizon 视觉世界模型中 RoPE 偏移超出训练范围→模型无法可靠寻址存储内容；naive 压缩在 RoPE 旋转空间**损坏记忆** | 新 |
| 2607.25227 | Decision-Level Hijacking via Bit-Flip Attacks | 已编目（3.x BFA 家族） | DUP |
| 2608.02712 | Don't Regenerate, Debug: Near-Miss Hardware Operators | 中：LLM 生成内核的 near-miss operator（编译运行但数值验证失败）修复 agent | 新 |
| 2607.28884 | Hollow-LLM Attack: ZK Verification Effort Gap | 弱相关（验证而非推理正确性） | 新 |
| 2608.08910 | Tied Trit-Planes: PTQTP→9-Level Quantizer | 中：PTQTP 量化约束 + weight-reconstruction error 与参考保真度的**解离** | 新 |
| 2608.04074 | NOVA-KV: KV Cache VQ with Attention-Preserving Transforms | 中：2-bit KV cache 向量量化，attention 感知失真 | 新 |
| 2608.03854 | Quantization Effects on Biomedical LLM Reliability | 中：prompt template + scoring 规则对校准的影响**主导**量化效应 | 新 |
| 2606.28958 | When Latent Agents Lie: KV-Cache Integrity in Multi-Agent | 中：多 agent 协作中 KV-cache 隐藏状态可被篡改 | 新 |
| 2601.19912 | LLM Vulnerability to GPU Soft Errors (Instruction-Level FI) | **强相关**：首个 LLM 推理指令级故障注入研究 | 新 |

---

## 重点突出

### 最具价值的新发现

1. **vLLM ROCm DSpark KV cache 布局重叠（#51837）** — 全新签名：ROCm KV 布局 `(2, num_blocks, ...)` 导致 K/V 分居两端，block 与 page 的映射在混合 Mamba+attention 模型上**重叠**，静默产出 NaN。这是**跨厂商布局契约不一致**的系统性问题。

2. **vLLM ROCm DeepSeek-V4 input GEMM preshuffled weight（#51821）** — 全新签名：删除 override point 方法后，ROCm preshuffled 权重被当普通 linear 乘→从第一层起全垃圾。CUDA 上完全不可见，ROCm override 读起来仍像连接的。

3. **TRT-LLM spec-decode tool-call prelude 静默分裂（#17437）** — 全新签名：推测解码 acceptance 路径不检查 token 流结构完整性→tool-call 控制块打开后永不闭合→空 tool_calls + HTTP 200。

4. **vLLM persistent_topk 静默丢弃候选（#51782）** — 全新签名：选择算法在数据相关条件下静默返回错误结果，无异常标记。

5. **arxiv 2608.05863 Runtime Observability for Heterogeneous Attention Memory** — 形式化了 KV/latent cache 的运行时可观测性，并在真实 DeepSeek-V4 栈上定位静默损坏到精确结构边界——直接关联本项目的监控路线。
