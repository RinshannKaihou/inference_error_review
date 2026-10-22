# 每日扫描报告 — 2026-09-16

- **扫描窗口**：GitHub `created:>=2026-09-14`（与 09-15 扫描重叠一日，已逐条去重）；arxiv 按 submittedDate 降序取最新 5 条/组
- **扫描范围**：vLLM / SGLang / transformers / TensorRT-LLM / triton / tt-metal / flashinfer 共 7 个仓库 + arxiv 5 组关键词
- **结果**：新发现并合并 **13 条**（3 个新子签名 NEW + 10 个已知类别变体）；1100 → 1113

---

## 一、新合并条目（13 条）

### 🆕 新故障模式（NEW 子签名，3 条）

1. **[新类别/新签名]** **vLLM #57059 — ROCm custom all-reduce start barrier 全松弛 → XGMI 撕裂读**（→ 1.222）
   `barrier_at_start` 的 flag store/load 均为 `__ATOMIC_RELAXED`（无 release/acquire），`barrier_at_start_release` 存在但 HEAD 无内核调用。多 GPU XGMI P2P 下消费者在对端写入仍在途时读 staged buffer——torn bf16 值播种 NaN/巨大激活，间歇性单层损坏 + 长上下文垃圾 token 墙，静默、负载/时序依赖。SASS 级验证（gfx908/4×MI100）。
   https://github.com/vllm-project/vllm/issues/57059

2. **[新类别/新签名]** **flashinfer #5260 — ragged prefill plan() 接受非连续 indptr 静默错值**（→ 8.88）
   `BatchPrefillWithRaggedKVCacheWrapper.plan()` 对 stride-2 view 类 indptr 按 data_ptr 线性读、忽略 stride——varlen 边界错位、attention 输出部分错误；连续拷贝正确。与 vLLM #57130（5.39）构成"隐式 contiguity 假设"跨栈同机制对。
   https://github.com/flashinfer-ai/flashinfer/issues/5260

3. **[新类别/新签名]** **SGLang #39708 — Janus-Pro `Normalize` 类被 VAE factory 同名遮蔽**（→ 13.39）
   模块压平后 L1792 的 `def Normalize(in_channels, norm_type)` 遮蔽 L1095 的 transform class——`CLIPVisionTower` 的 `Normalize(mean=,std=)` 落到 factory 签名 TypeError，图像归一化路径自 #3203 引入起不可用；第二重缺陷：class forward 还调 `F.normalize` 语义错误。
   https://github.com/sgl-project/sglang/issues/39708

### 🔄 已知类别的新案例（var:，10 条）

4. **[已知类别·硬件/SDC]** **tt-metal #56738 — binary_ng 融合激活混合 dtype 位域旋转**（→ 1.221，var:1.170 融合激活参数装配错族）
   两操作数 dtype 不一致（fp32×bf16 等）时融合激活路径 unpack 以错误 DataFormat 解包——abs(−1000) 返回位旋转值；1024 元素中 1008–1022 错，同 dtype 0 错。等价两步调用正确。
   https://github.com/tenstorrent/tt-metal/issues/56738

5. **[已知类别·硬件/SDC]** **flashinfer #5252 — trtllm-gen sm103a context FMHA 指令级时序扰动静默错值**（→ 1.223，var:6.210 时序扰动暴露内核缺陷族）
   B300 预编译 cubin 对 `LDTM.x2 tmem` 回读（softmax 行 max 统计）存在未声明消费者依赖——任意指令延迟（含零延迟钩子）使 ~99% 输出错（diff 1e5–3.3e6）；内核位级确定（200/200）故标准 CI 不可见。
   https://github.com/flashinfer-ai/flashinfer/issues/5252

6. **[已知类别·分布式]** **vLLM #57130 — sparse NCCL 权重补丁非连续 view stride 被忽略**（→ 5.39，var:5.38 权重分发正确性族）
   `PyNcclCommunicator.broadcast` 从 data_ptr 按 numel 发底层存储相邻元素——stride-2 的 indices/values 错位错值静默到达推理侧（期望 [10,-1,20,-1] 实得 [10,11,-1,-1]）。
   https://github.com/vllm-project/vllm/issues/57130

7. **[已知类别·服务栈/缓存]** **SGLang #39830 — hierarchical cache host-tier 命中 20/20 答非所问**（→ 6.240，var:6.193 host 层恢复损坏族）
   hybrid GDN 模型 prefix 驱逐后从 host tier 恢复命中时输出"另一个 pass 的声音"——流畅但非对该 prompt 的回答；cached_tokens 全额报告、无 crash 无警告；cold 与 device 命中全对。device 路径同族（#37836/#39342/#31833）已报，host 路径首例。
   https://github.com/sgl-project/sglang/issues/39830

8. **[已知类别·量化]** **SGLang #39684 — sgl-deep-gemm transform_sf 非 owning alias 被 allocator 复用**（→ 7.92，var:12.36 分配生命周期族）
   SM90 `transform_sf_into_required_layout` 返回别名输入的 scale 且不保 owner——删除输入后 caching allocator 复用该页，FP8 权重 scale 变成无关内存内容；#32188 只修了 SGLang 包装，直接调用库 API 的下游（权重 scale 路径）不受保护。
   https://github.com/sgl-project/sglang/issues/39684

9. **[已知类别·KV缓存/注意力]** **vLLM #57156 — DSV4.1 SM120 CUDA graph dummy capture 毒化 null KV block**（→ 8.89，var:6.237 占位路径毒化共享槽族）
   图捕获期 dummy batch 把 NaN 写进 block 0；内核"无效索引 gather slot 0 + softmax mask 丢弃"路径的 NoPE/RoPE 数据 load 只做钳位不免疫 NaN——null block 成全局毒源，所有 graph 模式（FULL/PIECEWISE/FULL_DECODE_ONLY）从首层起 NaN，enforce-eager 正确。
   https://github.com/vllm-project/vllm/issues/57156

10. **[已知类别·解码/采样]** **vLLM #57136 — V2 encoder 多预算 CUDA graphs 共享池 varlen 输出未零初始化**（→ 11.80，var:11.51 图缓冲复用族）
    默认九档 token 预算共享 graph pool 时大图（1196 视觉 token）NaN embeddings、错误回答（双 H20 节点）；单档捕获/独立 pool/零初始化 varlen 输出均正确——六条件诊断矩阵定位"脏缓冲读"。
    https://github.com/vllm-project/vllm/issues/57136

11. **[已知类别·分词器/模板]** **SGLang #39836 — Qwen3CoderDetector 重复 `<parameter=` 标签正则截断覆写**（→ 10.69，var:10.42 parser 假设族）
    非贪婪正则的 `(?=<parameter=)` lookahead 把值内重复字面量当新参数起点——findall 两匹配、后者覆盖前者，数组参数截断；生产 4/3678（0.11%）全在数组值参数，流式路径同病。
    https://github.com/sgl-project/sglang/issues/39836

12. **[已知类别·权重/加载]** **SGLang #39782 — Anthropic 流式多 tool 参数并入单个 tool_use 块**（→ 12.38，var:10.55 流式解析族）
    `stream=true` 时三个 tool call 的 `input_json_delta` 全标 index=0 灌入同一块，聚合非法 JSON `{"a"}{"b"}{"c"}`；四模式对照仅 Anthropic×streaming 0/3 全错。
    https://github.com/sgl-project/sglang/issues/39782

13. **[已知类别·权重/加载]** **SGLang #39831 — GLM-5.3-Flash pin transformers 5.12.1 缺 glm5_next，AutoProcessor 静默降级 tokenizer**（→ 12.39，var:6.156 依赖静默降级族）
    5.12.1 无 `glm5_next` 模块，`AutoProcessor` 静默返回 `TokenizersBackend`——每图缩为 3 个占位 token、模型从幻觉作答，`has_image_understanding: true` 照常上报；文本路径正常。
    https://github.com/sgl-project/sglang/issues/39831

---

## 二、查重后跳过的已编目条目（本窗口再现）

- vLLM #57038（truncate_prompt_tokens 多模态扩展前截断）= 09-15 已编目 10.67
- vLLM #56949（dense DP IPC payload 选错）= 09-15 已编目 5.38
- vLLM #57019（HiSparse host KV decode 噪声）= 09-15 已编目 6.239（含 #56889 同根因）
- vLLM #57016（batch-invariant fused-MoE 无效）= 09-15 已编目 4.22
- vLLM #57014（hadacore 尾 warp cp.async wait）= 09-15 已编目 1.218
- vLLM #56981（Qwen3-ASR token 数分歧）= 09-15 已编目 10.68
- vLLM #57017（DSV2-Lite fp8 TRITON_MLA NaN）= 09-15 已编目 7.90
- vLLM #56868（GLM-5.3-Flash 长 decode 退化）= 09-15 已编目 6.235
- SGLang #39626（block-fp8 BK64 静默错值）= 09-15 已编目 7.91
- SGLang #39624 / #39515 / #39634 / #39499 / #39521 / #39429 = 09-15 已判定（fail-loud 已修/性能/tracking/RFC/feature）
- tt-metal #56659 / #56626 / #56577 / #56550 / #56532 / #56522 / #56521 / #56520 / #56518 / #56517 / #56702 / #56688 / #56673 = 09-14/09-15 已编目（1.219/1.220/8.87/1.214/留观/1.217/测试基建/1.217/留观/1.215/语境补充/bring-up/自撤）
- transformers #48818（Gemma4 audio eager 掩码反转）= 09-15 已编目 13.38
- transformers #48807（serge 集成 triage）= 09-15 已判定不收录
- TRT-LLM #19172（prompt embedding H2D pinned pool）= 09-15 已编目 12.37
- flashinfer #5238 / #5233 / #5221 = 09-15 已判定（fail-loud/留观/feature）
- triton #11795（MLIR segfault）= 09-15 已判定留观

## 三、判定不收录（本日新见）

- vLLM #57223（assistant_tokens_mask 多模态展开失配）— 训练侧 mask 消费者（loss scoring），推理服务面影响有限，暂留观
- vLLM #57125（Qwen3.8-Flash-Next-NVFP4 无法启动）— fail-loud 启动失败
- vLLM #57099（GPT-OSS Harmony 解析异常）— fail-loud 有明确异常
- vLLM #57219（多模态 uuid 无长度上界 → EngineCore pickle+SHA-256）— CWE-400 资源治理/安全面，非正确性
- vLLM #57159（OffloadingConnector KV 事件 advertised vs retrievable 粒度）— 外部消费者契约设计议题，留观
- vLLM #57138（qwen_sparse_attention layer_type 校验）— fail-loud
- vLLM #57230/#57228/#57187/#57177/#57175/#57149/#57144/#57111/#57106（性能优化/RFC/feature）
- SGLang #39740/#39521/#39499（RFC/roadmap）、#39634（PP×spec tracking）、#39429（perf feature）、#39692（indexer cache offload 缺失——"可能造成精度下降"无实测证据，留观）
- transformers #48842（context parallelism trainer.predict 重复行）— 训练侧
- flashinfer #5270/#5268/#5261（测试基建/CI）、#5258（cudnn LSE 形状不匹配 fail-loud 拒绝）
- tt-metal #56688（bring-up）、#56673（报告者自撤）

## 四、arxiv 结果（6 组查询全部成功）

| 关键词 | 结果 |
|---|---|
| `silent data corruption LLM` | TrainSDC 2608.30769 / SCOUT 2608.11034 / LLM-PRISM 2604.10390 / Exploring SDC 2604.00726 / Understanding SDC 2502.12340 — 均已编目（§3.1 补充来源） |
| `bit flip LLM inference` | JITterFlip 2608.29745（3.18）/ UEP 2607.19623（已引）/ RangeGuard 2605.04563（2.3）/ Targeted BFA on Agents 2603.10042（3.13）/ BitFlipScope 2512.22174（3.14）— 均已编目 |
| `KV cache corruption` | Addressable Memory 2608.07408（14.7）/ Runtime Observability 2608.05863（已引）/ Nexus 2608.20397（方法论文不收录）/ When Latent Agents Lie 2606.28958（已编目）/ Moment-KV 2605.29873（方法论文不收录） |
| `LLM inference correctness` | **ECHO 2609.17241（2026-09-15 新）**、**FlexEE 2609.17008（2026-09-15 新）** — 分别为推测解码早层编排与早退+offload 优化方法论文，非故障案例，不收录；Maverick 2609.10264（09-15 已判方法论文）/ Sample-Guided Top-K 2609.08450（已判方法论文）/ MetaKV 2609.07966（已判方法论文） |
| `GPU soft error machine learning` | 0 命中 |

## 五、API 调用记录

- GitHub Search API：10 组查询（vLLM bug label / vLLM title 关键词 / SGLang / transformers / TRT-LLM / triton / tt-metal / flashinfer / 全站 SDC / vLLM quant+KV+NaN），间隔 8s，**全部一次成功、无限流 403**
- GitHub core REST：18 次详情调用（`--http1.1`，间隔 3s）全部成功
- arxiv API：6 组查询（`curl -sL` 跟随重定向）全部成功，无 429

## 六、合并与导出

- 主文档：FAULT_MASTER_REFERENCE.zh.md 1100 → **1113** 条（滚动汇总/目录/轴分布/来源表均已按实际解析数更新）
- 分布增量：第 1 类 +3、第 5 类 +1、第 6 类 +1、第 7 类 +1、第 8 类 +2、第 10 类 +1、第 11 类 +1、第 12 类 +2、第 13 类 +1
- 导出：`python3 export_faults.py` → Parsed **1113** entries（A* 前缀条目自 09-14 重构后已不存在，Parsed = 总条数）；fault_catalog.xlsx（1113 rows × 2 sheets）+ fault_catalog.csv（1113 rows）
