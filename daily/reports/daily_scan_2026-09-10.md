# 每日扫描报告 — 2026-09-10

- 扫描日期：2026-09-10（滚动窗口 `created:>2026-09-09`，覆盖 2026-09-09T00:00Z 之后创建的 issue）
- 目标仓库：vllm-project/vllm、sgl-project/sglang、NVIDIA/TensorRT-LLM、triton-lang/triton、tenstorrent/tt-metal、huggingface/transformers、flashinfer-ai/flashinfer
- 主文档基线：1025 条（截至 2026-09-09 合并后）
- 去重方法：对每条候选在主文档全文检索 issue 编号（`#NNNNN`）。注意：`#38840` 命中 1 处为 vLLM **#38840**（NIXL 层数断言，6.130 来源）与 SGLang #38840 撞号——本日 SGLang #38840（enc-dec 边界页 double-free）实际未编目但归为 hang/crash 类不收录；`#55107` 命中 1 处为 1.98 条目内对 dest→src bank wait 修复 PR 的引用（正是本日 #56120 所 bisect 的同一 commit，作为上下文而非独立编目）。

---

## 一、扫描结果总览

- GitHub Search API：10 组查询全部一次成功（vLLM×4、SGLang×1、TRT-LLM×1、triton×1、tt-metal×1、transformers×1、flashinfer×1）；无限流 403、无 Validation Failed
- core REST 详情：16 次全部成功（`--http1.1`）
- arxiv：5 组关键词查询，命中论文均为此前已编目（TrainSDC/SCOUT/LLM-PRISM/JITterFlip/GBFA/ROBBIN/2605.11202/2510.22876/MetaKV/Signed Rescue/Sample-Guided Top-K/2608.29934 等）；**BLINK（arXiv:2609.06781）为 BN 统计完整性检测/缓解方法论文且评估对象为 CNN（ResNet/MobileNet），非 LLM 域，不收录**；无新编目论文
- **独立候选 30 条 → 合并 14 条**（9 个新子签名 + 5 个已知类别变体；16 条不收录见「三」）

## 二、已合并条目（14 条）

### A. 新子签名（NEW，9 条）

| # | 标题 | 来源 | 归类 |
|---|---|---|---|
| 1 | Persistent KV offload 在 RoPE scaling 改变后复用旧配置写入的 KV 块：B shared（factor 2 + A 的 factor-1 目录）与 B isolated（factor 2 + 空目录）在**首个生成 token** 即分歧（`</think>` vs `L`），top-20 logprob 最大差 4.06；自 cache 重启对照排除 factor-2 缓存自用问题——纯跨配置缓存键缺失 | vLLM #56311 | 第 9 类 → 9.23 |
| 2 | SWA branching 把**较晚**的 Mamba checkpoint 接到**较早**的前缀键上：SWA 提议 branch96、Mamba 网格只有 checkpoint64/128 可复用、prefill 到 192 存 state192，插入取 `min(96,192)` → state192 挂到 key96，后续请求命中 96 从错误状态恢复——warm-cache logprobs 三组输入全部分歧、greedy 输出一组改变（H200 真实 Inkling checkpoint + Rust TreeCore 复现） | SGLang #38815 | 第 6 类 → 6.225 |
| 3 | Qwen3ForSequenceClassification 用带 bias 的 nn.Linear 建分类头，加载只含 `score.weight` 无 `score.bias` 的 checkpoint 时**随机初始化的 bias 原样保留**、无任何报错——计算变为 `W@h + b_init`，静默改变分类 logits（CPU AST 提取复现，v0.5.10.post1 与 main 均中招） | SGLang #38768 | 第 12 类 → 12.34 |
| 4 | flashinfer Split-KV decode 每个 chunk 的 partial 输出经 `tmp_v`（按 `sizeof(float)` 分配却以 `DTypeO*` 读回）在 merge **之前**就被舍入到 bf16——分配的上半区从未写入，精度白白放弃；split-KV 开启时 1/3–1/2 的 decode 输出相对 fp32 参考偏差 ≥1 个 bf16 步长（关闭 split 降至 0.2%）；prefill 调度器同型（`sizeof_dtype_o` 传入后被丢弃） | flashinfer #5107 | 第 8 类 → 8.80 |
| 5 | ttnn.slice 的 SliceRmShardedProgramFactory 按 bounding box 行扫描把 runtime args 发给 core，而 sharded buffer 的实际放置按 `corerange_to_cores` 逐矩形遍历——网格不从 (0,0) 起、含空洞、或由多个矩形组成时 args 发错 core → 静默错误输出；唯一能拦截的 `TT_ASSERT` 在 release 构建被编译掉 | tt-metal #56162 | 第 1 类 → 1.196 |
| 6 | ttnn.softmax 对 ±inf/NaN 行返回 uniform 1/N 或全 0 而非 NaN（与自家 golden 相反）：all−inf 行 → uniform 0.0156（SFPU exp sanitize 把 NaN 输入钳到 −88.5 → 各 lane 同微小值 → 归一化成均匀分布）；含 +inf/NaN 行 → 全 0。**全−inf 行正是 additive mask 后全 padding 的 attention 行**——"貌似合法的概率分布"比 NaN 更危险（ttsim 验证） | tt-metal #56056 | 第 1 类 → 1.197 |
| 7 | tt-metal #55107 的 unpack 半边（"pipeline the dummy publications"，BH only）放宽了一个等待：MiniMax-H3 ring-attention SDPA 在 4x32 BH Galaxy quad 上**每个 t2va 请求都输出噪声**（视频 1.7–3.3 bits/px/frame vs 干净 0.34、音频钉在 0 dBFS），仅 revert 该 commit 即恢复 43/43 干净——修复 WH 竞态的 PR 自己在 BH 上引入静默错值，且 `llk_unpack_A_sdpa.h` 无 LLK 单测覆盖 | tt-metal #56120 | 第 1 类 → 1.198 |
| 8 | GPT-OSS-120B vLLM structured output 在 Galaxy 上三周从未过闸（HT 90.6–96.9%、LL 15.6–37.5%）：**首个 decode token 起垃圾**（`cosmos cosmos cosmos` 词循环），prefill 采样的 `<｜channel|>` 正确、其后 token 损坏；受控对照锁定 host-logits decode 路径（`trace_ids_decode[False]` + 全 padded-vocab logits 回读——structured 请求强制 host 采样）为 H1 嫌疑；部分 LL 流数百 token 后恢复连贯（SW 层 128 token 遗忘分叉，指向 H3 EP 行分歧） | tt-metal #56100 | 第 6 类 → 6.226 |
| 9 | flashinfer SM120 sparse-MLA **prefill**（DSv4 dual-cache）query >64 token 即输出损坏（65/69/122/160 token prompt 全垃圾，≤62 正确）：阈值正是 `_DECODE_MAX_TOKENS=64` 路由边界；调用方把 `>= topk_length[i]` 的 indices/extra_indices lane 掩成 −1 即修复（对应 #5075 "lanes beyond topk_len are not bounded"）；EP1+marlin 下非确定，指向读未初始化/陈旧内存 | flashinfer #5095 | 第 8 类 → 8.82 |

### B. 已知类别新实例（var:，5 条）

| # | 标题 | 来源 | 归类 | var 族 |
|---|---|---|---|---|
| 10 | Triton gfx942 block-pingpong 双 cluster 调度 LDS 竞态：`s_barrier` 发在 `ds_read` waitcnt 之前，另一 warp group 的 stage j+1 `ds_write` 落进单级 LDS 缓冲 → 128×128 tile 内 intra-tile 行 32–47 的 4×32 patch 消费**下一个 k 步**的 A tile（A[r,k+64]，100% 错元素均为精确 +BLOCK_K 偏移）；200/200 复现，关 pingpong 0/200；inductor max-autotune MI300 CI 验证 500/500 失败 | triton #11696 | 第 1 类 → 1.199 | var:1.9（存储/等待链同步缺失族） |
| 11 | ttnn sdpa_decode `k_chunk_size=32` 且 `cur_pos>=32`（第二个 chunk 进界）即返回垃圾；chunk 64/128 任意 chunk 数正确——而内核**默认路径自中招**：`get_chunk_size()` 取 KV 长度最大 2 的幂因子，任何 32·odd 长度（96/160/224…）默认选 32；FIBO-vlm（Qwen3-VL）96 位缓存解码垃圾 token、128 位正确 | tt-metal #56171 | 第 8 类 → 8.81 | var:1.63（sdpa_decode 配置敏感错值） |
| 12 | vLLM `--cpu-offload-gb` 被接受、回显在 non-default args 里，worker 侧却选出 NoopOffloader：`cpu_offload_gb=15.1` 跨 AsyncLLM→EngineCore pickle 边界后变为 0 → 19.33GB 权重全上 15.47GB 卡 → OOM；NoopOffloader 选择只在 DEBUG 级宣告，默认日志唯一露面处是 traceback——"静默失效的显式配置" | vLLM #56283 | 第 6 类 → 6.227 | var:6.136（配置传递/预算丢失族） |
| 13 | vLLM tool parser 非 streaming `extract_tool_calls` 在 tool-call 起始 token 处截断后**永不恢复**：DeepSeek-V3/hermes parser 丢 post-tool-call 文本（`It is 30 degrees.`），streaming 路径却作为 content delta 发出——同一 generation，`/v1/messages` streaming 显示尾文本、非 streaming 丢失 | vLLM #56263 | 第 10 类 → 10.60 | var:10.42（parser 通道终止符假设） |
| 14 | vLLM Beam Search（online/offline）构造最终解码文本时不理会 `skip_special_tokens`：Voxtral Realtime 转录暴露数百个 `[STREAMING_PAD]`/`[STREAMING_WORD]` 内部控制 token；常规采样正确过滤——token ID 本身合法，缺陷限于最终 detokenization 步骤 | vLLM #56209 | 第 10 类 → 10.61 | var:10.52（stop/special-token 过滤失效） |

## 三、不收录说明（16 条）

1. **vLLM #56347** `[Bug][ROCm] DeepSeek-V4.1 FULL_DECODE_ONLY segfault on 3rd decode token` — fail-loud segfault（全部 worker 死亡 + EngineDeadError），根因分析已定位 async scheduling × graph replay 的 runner 级重叠问题；无数值静默损坏，与 6.177（inductor×spec livelock）同域但为显性 crash。留观。
2. **SGLang #38849** `GLM 5.3 segfault on MI300 after ~1M tokens` — fail-loud scheduler segfault，报告未定位根因（日志只有 aiter fused_moe kernel 选择行）。留观。
3. **SGLang #38793** `H20 8card can't launch Qwen3.8-Flash-Next-FP8` — 启动期 CUDA graph capture crash（fail-fast）。与已编目 7.80（GLM-5.3-NVFP4+ DFlash worker 退出）同型，但无数值损坏细节，不单独编目。
4. **SGLang #38788** `Scripted-runtime rid reuse loses race with rid_to_state release` — 测试基础设施竞态（3 个 manual test 红），fail-loud 60s recv timeout。
5. **vLLM #56280** `Nemotron Image Embeddings int(None)` — fail-loud TypeError，无静默损坏。
6. **vLLM #56180** `custom_all_reduce.cuh:164 invalid argument` — CI 测试 CUDA error，fail-loud。
7. **vLLM #56281** `DSV4-Flash + DP>1 + EP + DSpark profile_run AssertionError` — 启动失败（DPMetadata.make 断言），fail-fast，与已编目 5.33（DSPARK draft 池 OOM）同族不重复。
8. **SGLang #38871** `VisionFlash4Attention drops MiMo-style window attention and sinks` — FA4 wrapper 丢弃 `window_size`/`s_aux` kwargs 的语义缺失（#37983 Triton 路径的姊妹）；报告为缺陷描述+修复方向，无定量损坏证据、无已发布受影响模型确认。与 6.225 同属"分支/语义错配"但证据等级不足，留观。
9. **SGLang #38854** `Qwen3.5 hybrid GPTQ in_proj_ba built as quantized` — 最终 fail-loud（`gptq_marlin_repack` size_n=96 断言 + SIGQUIT 重启循环），但前奏含 96×"weight not found in params_dict"（bf16 权重静默丢弃后才 crash）；vLLM 同 checkpoint 正常。crash 类暂不收录，"混精度 checkpoint 的 per-layer quant 归属"机制留观（与 7.77 同族）。
10. **flashinfer #5091** — cute-dsl kernel cache 测试断言，测试基础设施。
11. **flashinfer #5079 / #5077** — Feature request（per-token FC2 act scaling、MegaMoE workspace staging）。
12. **tt-metal #56096 / #56090 / #56086** — 纯性能回归（+50ms prefill、JIT 编译时间 +20–27%、CLIP encoder −2%），无数值正确性影响。
13. **tt-metal #56046 / #56038 / #56030 / #56029 / #56028** — GLM-5.2 逐层权重构建超时（性能）、warm_cache 零写入仍过 PCC（测试盲点分析，非产品故障）、无 padded prefill 测试（覆盖缺口）、bound indexer at token end（未完成工作项）、CI reviewer 失败。
14. **transformers #48693** `[ImageGPT] cross-attention crashes, ignores encoder_attention_mask, doubles cache` — 三合一报告：crash 部分 fail-loud；mask 忽略部分与已编目 **13.27**（GPT2 encoder_attention_mask 硬编码丢弃，ImageGPT 手拷自 GPT2）同根因；cache 翻倍为新观察但同一 PR 修复。作为 13.27 的补充证据留观，不占新编号。
15. **transformers #48690 / #48687** — 训练侧（DDP world size 乘 router aux loss、EP 下 router 梯度错误），loss 正确仅梯度错——非推理路径。
16. **SGLang #38840** `enc-dec shared boundary page double-freed (page_size>1)` — 会计损坏累积后 invariant checker SIGQUIT crash（fail-loud，默认 strict mem check 开启）；机制（边界页跨区域双释放）与 6.213（enc-dec cache manager 每步双计预算膨胀）近邻。crash 显性、且首请求后即炸，暂不收录，留观。

## 四、重点新发现摘要

1. **跨 RoPE 配置的持久 KV 复用（#56311）**：offload 目录的缓存键不含 positional 配置——factor 1 写的 KV 被 factor 2 进程读回，首个生成 token 即分歧。这是 6.180（prompt_cache_key 静默丢弃）在 **RoPE 维度**的对应物，且叠加"文件系统层持久化"使污染跨进程生命周期存活。六组对照实验设计严密（含 self-cache 重启对照）。
2. **分支缓存把 state192 挂到 key96（#38815）**：KV 可切片、checkpoint 不可切片——`min(所有组件请求长度)` 的插入语义在 SWA（页对齐）与 Mamba（网格对齐）目标分歧时必然错配。与 6.144（Mamba align 物化 × 查找上限错配）同族但根因在**多组件分支目标合并**这一新层。
3. **Split-KV 精度在 merge 前被舍入（#5107）**：`tmp_v` 按 fp32 分配、以 `DTypeO*` 读回——"分配了精度却不用"的纯浪费型缺陷，1/3–1/2 decode 输出偏差 ≥1 bf16 步长。与 8.67（split-KV scratch 陈旧 partials）同 locus 不同机制（那条是陈旧数据、这条是过早舍入）。
4. **softmax 把全−inf 行变成均匀分布（#56056）**：SFPU exp sanitize（NaN→−88.5 钳位）把数学上未定义的输入变成"貌似合法的概率分布"——全 padding attention 行正是这个输入。比返回 NaN 更危险：下游无从察觉。
5. **修复 PR 自身引入静默错值（#56120）**：#55107 修 WH 竞态的 BH 半边（pipelined dummy publications）在 Galaxy quad 上让 MiniMax-H3 t2va 100% 输出噪声——"修复未跨 arch 验证"（1.122 Quasar softplus 先例）的 LLK 规模重演，且无单测覆盖。
6. **默认路径即中招的 chunk 边界（#56171）**：`get_chunk_size()` 取最大 2 的幂因子 → 任何 32·odd 长度的 KV 默认选 32 → 第二 chunk 进界即垃圾。"内核自选配置落在坏档位"与 1.191（传 config 降 fidelity）构成一对镜像：一个是用户传参中招、一个是默认启发式中招。
7. **host 采样路径独享的损坏（#56100）**：structured 请求强制 host-logits decode（全 padded-vocab 回读）——同一 build 的 device 采样 demo 完全干净。三周 gate 从未过、脚本回显"✅ passed"是 echo 文本而非结果：**基准测试的假阳性掩盖链**。
8. **prefix>64 才坏的 sparse-MLA prefill（#5095）**：阈值恰为 decode→prefill 内核路由边界；调用方掩 -1 可修（#5075 同根因）。"短提示正常"使此类损坏极易漏过冒烟测试——与 6.215（DSV4 mod-4 长度门控）同款"长度条件化损坏"。

## 五、API 调用记录

| # | 类型 | 查询/目标 | 结果 |
|---|---|---|---|
| 1 | search | vllm correctness in:title | 0 命中 |
| 2 | search | vllm quantization wrong | 0 命中 |
| 3 | search | vllm corrupt OR NaN OR silently | 2 命中（2 有效） |
| 4 | search | sglang wrong OR corrupt OR NaN OR silent | 11 命中（8 有效） |
| 5 | search | TensorRT-LLM 全部新 issue | 0 命中 |
| 6 | search | triton wrong OR NaN OR miscompile | 1 命中（1 有效） |
| 7 | search | tt-metal numerical OR PCC OR NaN OR wrong | 13 命中（7 有效） |
| 8 | search | transformers wrong result OR NaN OR corrupt | 3 命中（1 有效） |
| 9 | search | flashinfer 全部新 issue | 5 命中（3 有效） |
| 10 | search | vllm label:bug 全部新 issue | 6 命中（4 有效） |
| 11–26 | core REST | 16 个 issue 详情（含 #56100/#56263/#56283/#56347 全文） | 全部 200 |
| 27–31 | arxiv | SDC+LLM / bitflip / KV cache corruption / LLM correctness / GPU soft error | 全部已编目或不收录（BLINK 2609.06781 为 CNN 域方法论文） |
