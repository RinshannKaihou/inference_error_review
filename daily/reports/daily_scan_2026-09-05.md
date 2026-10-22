# 每日扫描报告 — 2026-09-05

- 扫描日期：2026-09-05（UTC）
- 扫描窗口：GitHub `created:>2026-09-04`；arxiv 最近提交排序
- 目标仓库：vllm-project/vllm、sgl-project/sglang、huggingface/transformers、NVIDIA/TensorRT-LLM、triton-lang/triton、tenstorrent/tt-metal
- 主文档基线：912 条（2026-09-04 合并后）

## 结果总览

- 共执行 10 组 GitHub Search API 查询（全部一次成功，无限流 403、无 Validation Failed）+ 5 组 arxiv API 查询（HTTPS）
- 去重核实后：**25 条新条目合并进主文档**（20 个新子签名 `[新类别/新签名]` + 5 个已知类别变体 `[已知类别]`），912 → **937**
- 另有 5 条修复 PR 作为已编目条目的补充来源引用（不占新编号）；1 条 PR（tt-metal #55567 div_bw 死守卫）核实后判定无已发布正确性影响，未编目

## 新合并条目（25 条）

### 第 1 类 · 硬件/SDC（+4）

| 编号 | 标题 | 来源 | 分类 | 简述 |
|---|---|---|---|---|
| 1.169 | [FuseNestedLoops] flatten=True hoists the inner bound load above a zero-trip outer loop and dereferences a pointer the source never reads | triton #11601 | `[已知类别]` var:1.127 | 同一 FuseNestedLoops pass 的第二形态：内层 bound 的标量 load 被提升到零次迭代的外层循环之上 → 源码从不触碰的指针被解引用 → illegal memory access（flatten=False 干净） |
| 1.170 | Matmul with fused activations: Fix Selu param order and missing Softplus param | tt-metal PR #55550（fixes #55252） | `[已知类别]` var:1.139 | 融合激活 epilogue 参数装配错：selu alpha/scale 顺序颠倒 + softplus 第三参（beta 倒数）未传默认 0；CI 容差 rtol=15.0×K（K=512 即 768,000%）接受几乎任何错误——测试分布放行结构 bug |
| 1.171 | fix(ttnn): handle zero inputs in prod_bw gradient calculation | tt-metal PR #55564（resolves #54551） | `[已知类别]` var:1.156 | prod_bw 用 `reciprocal(input)×(prod×grad)`：输入含精确 0 时 `+inf × 0.0 → NaN`，正确梯度应为有限值（零感知掩码 + 分支逻辑修复） |
| 1.172 | [deepseek_prefill] unified_routed_expert_ffn: act sender publishes its own INCL_SRC loopback slot on a SENT check, not a landed one | tt-metal #55551（修复 PR #55552） | `[新类别/新签名]` | NoC 多播发送方以 `NIU_MST_NONPOSTED_WR_REQ_SENT`（请求离开 NIU）而非数据落回本核 L1 为准发布 slot——发送方自己的 INCL_SRC 回环拷贝未落地即被 compute 读取；`writes_flushed` 语义误用 + 注释里的 rationale 与代码矛盾 |

### 第 6 类 · 服务栈/缓存正确性（+7）

| 编号 | 标题 | 来源 | 分类 | 简述 |
|---|---|---|---|---|
| 6.197 | [Security] Drop NixlPush WRITEs after the producer request ends | vLLM PR #55470 | `[新类别/新签名]` | P→D 握手等待期的完成回调捕获原始逻辑 block id，生产者请求已结束（lease 到期/推送完成）后仍提交 WRITE——源 block 可能已属其他请求：仍在运行的 decode 请求从**错误的 KV** 生成而 API 返回 200 |
| 6.198 | [Bug] HiCache host-memory sizing guard charges co-located ranks twice and rejects pools that fit | SGLang #38156（修复 PR #38157） | `[新类别/新签名]` | host 池 sizing 守卫把 `available` 按并存 rank 数均分，但各 TP rank 到达守卫的时刻不同且互不同步——后到者把先到者已 pin 的池重复计扣，物理上放得下的池被 `ValueError` 拒绝（tp8/1772GiB 实测间歇启动失败） |
| 6.199 | [Bugfix][Core] Retire Mamba states across null gaps | vLLM PR #55450 | `[新类别/新签名]` | Mamba align-mode prefill 在 state block 间留 null gap，退休扫描遇第一个 gap 即停——更老的 state 永不释放，最终耗尽池（内存泄漏式池枯竭；替代 #55435 中的对应修复） |
| 6.200 | [Fix] Evict only the mamba slots a prefill batch will actually allocate | SGLang PR #38151 | `[新类别/新签名]` | `alloc_req_slots` 按 `num_reqs×3` 的头room常量预留并驱逐 mamba checkpoint，而非实际将取数（prefix 命中者取 2、chunk 续传者取 0）——紧池下盈余驱逐在少数 slot 间循环，checkpoint 缓存被系统性洗掉 |
| 6.201 | [Bug] Decode running batch pinned at `max_running_requests - 1` under Chunked Prefill | SGLang #38118 | `[新类别/新签名]` | DP attention 下 running requests 恒定钉在 max−1（每 rank 一个 slot 永久闲置）：GLM-5-FP8、DP=8、max=128 时 8 个 rank 全部 `#running-req: 15` 而队列 ≥1——准入计数 off-by-one |
| 6.202 | [Bug]: Responses history trimming hangs on over-capacity input without a completed turn | TensorRT-LLM #18759 | `[新类别/新签名]` | `ConversationHistoryStore` 默认容量下存 65 条连续 user 消息（无 completed turn）进入非让出循环——纯组件级复现（无需 GPU），Harmony 无 final 消息的历史同型挂起 |
| 6.203 | [Bugfix] Fall back to T1 when ARC cannot reclaim enough entries from T2 | vLLM PR #55461 | `[新类别/新签名]` | CPU KV offload 的 ARC 驱逐在 T1 低于自适应目标时只扫 T2，T2 无合格候选即返回 None——batch 中途或 T2 被 pin/保护时 store 原子性失败，明明有足够可驱逐条目仍写入失败 |

### 第 7 类 · 量化（+2）

| 编号 | 标题 | 来源 | 分类 | 简述 |
|---|---|---|---|---|
| 7.72 | [Bug]: ModelOpt MIXED_PRECISION cannot load FP8_BLOCK_SCALES MTP experts (nvidia/Qwen3.8-Flash-Next-NVFP4 + MTP) | vLLM #55496（修复 PR #55498） | `[新类别/新签名]` | 两个独立缺陷叠加：(a) `RoutedExperts` 分支的硬编码 if-chain 不支持 `FP8_BLOCK_SCALES` → 层被建成非量化 → 加载期 `AttributeError: no parameter 'w2_weight_scale_inv'`；(b) MTP 层索引域错配（checkpoint 命名 `mtp.layers.0` vs 模型构建 `mtp.layers.48`）→ quant config 查找返回 None——与 #53790（NemotronHMTP mapper）同类：quant 配置的 checkpoint 命名空间到不了 draft |
| 7.73 | [Bugfix] Pad manual NVFP4 fusion scale buffers to full swizzled tiles | vLLM PR #55456 | `[新类别/新签名]` | #51415 的手工 SiLU+NVFP4 fusion 按 `[num_tokens, ceil(K/64)*4]` 分配 scale 存储，而 CUDA 内核按 128×4 swizzle 布局整 tile 写偏移——小 batch/末尾残 tile 越界写；CI（build 87376）在 `silu_mul_cvt_fp16_to_fp4` 记录 `CUDBG_EXCEPTION_WARP_ILLEGAL_ADDRESS`，open alert 17 连败 |

### 第 8 类 · KV 缓存/注意力（+4）

| 编号 | 标题 | 来源 | 分类 | 简述 |
|---|---|---|---|---|
| 8.65 | [Bug] MiniMax-M3 W4A16 (compressed-tensors) on 2x DGX Spark: serves but every token is id 0 — all-NUL output on the Triton MiniMaxSparse path | SGLang #38143 | `[新类别/新签名]` | 加载、捕获 CUDA graph、正常服务——但每个生成 token 都是 id 0（NUL 字节串），temp 0 全 prompt 复现，无错误无 NaN 断言无警告；路径为 sm_121 上的 Triton MiniMaxSparse 注意力 × compressed-tensors W4A16；同权重 vLLM 正确 |
| 8.66 | [Bugfix][Attention] Use packed MLA sizes for hybrid block alignment | vLLM PR #55449 | `[已知类别]` var:8.56 | hybrid block 对齐用语义 head size 定 MLA 页大小，漏掉 FP8/NVFP4 的 packed 存储位宽——注意力块与 Mamba padding 的物理页尺寸错配（kpool/对齐错配族在 MLA 侧的实例；替代 #55435 对应修复） |
| 8.67 | [None][fix] Reset the FlashInfer split-KV scratch before every attention pass | TensorRT-LLM PR #18755 | `[新类别/新签名]` | 共享 FlashInfer workspace（`torch.empty` 未清零）中，merge 覆盖到本 pass 注意力内核未写的 tile 时读回旧 plan 的 partials——旧 log-sum-exp 可为 −inf，merge 出非有限注意力行；CUDA graph 下 batch 组合变化时已观测到非有限输出 |
| 8.68 | [Bugfix][DSv4] Add missing Hadamard rotation to sparse indexer quant kernels | vLLM PR #55459 | `[新类别/新签名]` | DSV4 fused sparse-indexer Q/K 量化内核跳过参考实现 RoPE 后、量化前的 Hadamard 旋转：K 侧 compressor 存了 rotate 参数但 forward 从不读（显式遗漏），Q 侧 fp8/mxfp4 fused 内核同样未旋转——索引器缓存逐位偏离参考实现（QK 点积数学上保持，数值分布漂移） |

### 第 9 类 · 位置编码/RoPE（+1）

| 编号 | 标题 | 来源 | 分类 | 简述 |
|---|---|---|---|---|
| 9.22 | [Attention] Fix tokenspeed_mla FP8 prefill for NoPE MLA layers (Kimi Linear) | SGLang PR #38158（根因诊断 #38152，引入于 #35770） | `[新类别/新签名]` | #35770 让 `resolve_attn_backend()` 解包 `HybridLinearAttnBackend` 后，Kimi Linear 的 NoPE MLA 层（`rotary_emb is None`）从走 wrapper 的 BF16 回退路径变为直落 `TokenspeedMLABackend`——其 hook 假设 RoPE 存在，`rotary_emb.cos_sin_cache` 对 NoneType 取属性，DCP+fp8_e4m3 下每 TP rank prefill crash |

### 第 10 类 · 分词器/模板/结构化输出（+4）

| 编号 | 标题 | 来源 | 分类 | 简述 |
|---|---|---|---|---|
| 10.49 | [Bugfix] Qwen3 parser: a malformed <parameter> element no longer leaks into streamed arguments | vLLM PR #55497（fixes #55495） | `[已知类别]` var:10.42 | 参数名正则接受除 `>` 外任意字符：模型写出的畸形元素（缺名字/缺 `>`/多余 `=`）把后续全部内容（含闭合标签）当 key 流式透出——前缀无法扩展、`_flush_arg_converter` 静默返回 None，客户端持有未终结 JSON；Responses API 流（Codex CLI）此后该会话每个请求 400 |
| 10.50 | LukeTokenizer drops <s>/</s> on the plain-text path (no post-processor installed) | transformers #48544（修复 PR #48545） | `[新类别/新签名]` | plain-text 路径未安装 post-processor：BOS/EOS 特殊 token 在该路径被静默丢弃——同库 text+text pair 路径正常，静默输入损坏 |
| 10.51 | [Bug] `--default-chat-template-kwargs` with `reasoning_effort` silently overrides the per-request `reasoning_effort` | SGLang #38104 | `[新类别/新签名]` | 服务端默认值经 `setdefault` 回灌到已被 pop 出 `chat_template_kwargs` 的键上——请求级 `reasoning_effort`（及顶层等价字段）被静默丢弃，与所有其他 chat_template_kwargs 键的 request-first 优先级相反；文档仅建议"别同时配"，无拒绝无告警 |
| 10.52 | [Bugfix][Engine] Honor include_stop_str_in_output for stop token ids | vLLM PR #55483 | `[新类别/新签名]` | `include_stop_str_in_output=False` 只对 stop **strings** 截断：增量 detokenizer 从不检查 `sampling_params.stop_token_ids`——命中的 stop token 文本留在 `output_text` 与流式 delta 中（stop-string 三条路径不受影响，纯 token-id 路径语义缺口） |

### 第 11 类 · 解码/采样（+3）

| 编号 | 标题 | 来源 | 分类 | 简述 |
|---|---|---|---|---|
| 11.64 | [Bugfix] Index mamba spec-decode block tables by request slot, not batch row | vLLM PR #55506 | `[新类别/新签名]` | PP≥2 + MTP + prefix caching（Qwen3.8-Flash-Next hybrid）：spec context 一次性捕获的 block table 指针被按**每步重收集的 batch 序**索引，PP 下非末 rank 的 postprocess 用的是陈旧 batch 映射——14–33% 请求退化为常量 token 循环（token 1023 = all-NaN logits 行的确定性采样输出），acceptance 50%→26%；阈值恰为 2→3 并发 slot；与 #54173 的 IMA 同根（sm_121 响、sm_80 静默） |
| 11.65 | [Bugfix] Fix V1 GDN state recovery across mixed speculative batches | vLLM PR #55504 | `[新类别/新签名]` | spec/非 spec 混合 batch 的接受状态恢复分区 bug：当全部非 spec 请求的接受计数均为 1 时，合并实现丢弃非 spec 计数——conv caller 回退到 spec 分区的计数，offset 与条目数双错（在 #40738 及其 follow-up 基础上修复残余分区缺陷） |
| 11.66 | [Bug] NGRAM cached requests miss newly inserted longer contexts | SGLang #38129 | `[新类别/新签名]` | NGRAM 缓存请求复用 ID 时，插入更长上下文后（即使 `synchronize()` 后）仍返回陈旧 draft（BFS/PROB 均复现 `[30,70,80,90]` vs 应为 `[30,40,50,60]`）——CPU 引擎无 GPU 最小复现；draft 被 verify 拒绝故输出仍正确，代价是接受率/吞吐 |

## 补充来源引用（不占新编号，5 条）

| 修复 PR | 并入条目 | 说明 |
|---|---|---|
| vLLM PR #55507 | 6.131（#53142） | mamba align state index 以 mamba 组 block size 播种（修复 PR） |
| vLLM PR #55467 | 11.63（#55357） | PLE short-conv 内核忽略 `state_indices` 步幅——prefill 行 r≥1 读到 request 0 的 spec checkpoint 块（根因定位） |
| vLLM PR #55475 | 6.181（#54808/#49712） | structural-tag/engine 流式 parser 关闭 `supports_required_and_named`（修复 PR） |
| vLLM PR #55482 | 8.63（伴随 #55046） | encoder-only 后端选择忽略 KV cache dtype（修复 PR；另见 #55150/#55159 互补） |
| SGLang PR #38144 | 8.55（#36807） | fast_topk threshold-bin 溢出 exact fallback（修复 PR；实测 16K/34K 溢出计数） |

## 编目勘误（本次扫描期间发现并修正）

- **13.27 → 13.34**（`sglang_spark25_attn_gate_hardcoded_sigmoid_ignores_config`）：2026-09-04 合并时误用已被 `hf_gpt2_encoder_attention_mask_silently_discarded`（2026-08-13）占用的编号 13.27，本次合并前例行查重时发现并重编为 13.34；昨日汇总行已附勘误注。条目内容与计数不受影响（第 13 类仍为 34 条）。

## 跳过条目（核实后不编目）

- tt-metal PR #55567（div_bw 死守卫）：核实后发现已发布代码的语义与 PyTorch autograd 一致（守卫被求值后丢弃、无错误效果；保留反而会在 −0.0 引入符号 bug）——主要为死计算清理，无正确性影响
- vLLM PR #55442（GLM MTP disposable head 延迟分配）：内存峰值优化，非正确性缺陷
- vLLM PR #55443（MoE gate_proj 一致性检查）：启动期误拒（fail-loud false positive），影响面小
- vLLM PR #55471（NIXL pull peer 元数据恢复）：恢复路径加固，正确性显形不明确
- TRT-LLM PR #18750（VisualGen CuTeDSL FMHA 量化回归修复）：内部 JIRA 驱动，公开信息仅自动生成 notes，证据不足
- TRT-LLM PR #18754（KV guard page 诊断特性）/ #18747（flashinfer 版本 pin）：非故障案例
- vLLM PR #55478（multi_modal_uuids 多租户安全文档）、#38122（bearer token bytes 比较）：安全/加固类，非推理正确性

## arxiv 结果（无新编目论文）

- `silent data corruption LLM`：TrainSDC（已为 §3.1 补充来源）、SCOUT（已编目 2608.11034）、LLM-PRISM（已编目）、其余已编目
- `bit flip LLM inference`：JITterFlip（已编目 3.18）、Groundhog GBFA（已编目 3.16）、ROBBIN（已编目 3.17）、Decision-Level Hijacking（已编目）、Vector Symbolic Policy（不相关）
- `KV cache corruption`：When Latent Agents Lie（已编目）、Moment-KV（方法论文不收录）、Nexus（方法论文不收录）、Addressable Memory for Video World Models / Runtime Observability（系统方法论文，无新故障签名）
- `LLM inference correctness`：Compression-Aware Abstention（已编目 15.9 相关）、JITterFlip（已编目）、其余不相关
- `GPU soft error inference`：Instruction-Level FI（已编目 1.98）

## API 调用记录

| # | 查询 | 结果数 | 状态 |
|---|---|---|---|
| Q1 | vllm `wrong/incorrect/corruption created:>2026-09-04` | 24 | ✅ |
| Q2 | sglang `bug/wrong/corrupt/NaN created:>2026-09-04` | 20 | ✅ |
| Q3 | transformers `wrong/incorrect/corrupt/NaN created:>2026-09-04` | 2 | ✅ |
| Q4 | TensorRT-LLM `bug/wrong/corrupt/NaN created:>2026-09-04` | 13 | ✅ |
| Q5 | triton `wrong/incorrect/miscompile/corrupt created:>2026-09-04` | 1 | ✅ |
| Q6 | tt-metal `wrong/incorrect/corrupt/NaN/mismatch created:>2026-09-04` | 10 | ✅ |
| Q7 | vllm `quant/FP8/NVFP4/KV created:>2026-09-04 label:bug` | 17 | ✅ |
| Q8 | transformers `tokenizer/rope/config created:>2026-09-04 label:bug` | 1 | ✅ |
| Q9 | vllm `label:bug created:>2026-09-04`（前 20） | 46 | ✅ |
| Q10 | sglang `correctness/accuracy/garbage/silent created:>2026-09-04` | 54 | ✅ |
| A1-A5 | arxiv 5 组关键词（HTTPS） | — | ✅ 无新编目论文 |

条目核实直接使用 search API 附带正文；另对 vLLM #55496/#55507/#55504、tt-metal #55567/#55564 调用 core REST（`--http1.1`，全部成功）。
