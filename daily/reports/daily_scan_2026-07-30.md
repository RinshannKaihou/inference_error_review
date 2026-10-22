# LLM 推理故障每日扫描 — 2026-07-30

> **扫描日期**：2026-07-30（UTC）
> **扫描窗口**：GitHub issues/PRs `created:>=2026-07-23`（7 天滚动）；arxiv 按提交日期降序取最近 6 篇
> **搜索范围**：6 个 GitHub 目标仓库（vllm / sglang / transformers / TensorRT-LLM / triton / tt-metal）+ 跨仓库全局搜索 + arxiv 5 组关键词
> **对照基准**：[`FAULT_MASTER_REFERENCE.zh.md`](FAULT_MASTER_REFERENCE.zh.md)（235 条已编目故障，三层来源）
> **API 状态**：GitHub Search API 9/9 成功（remaining 60/60，无 rate-limit）；arxiv 5/5 成功（HTTPS，`-L` 跟随 301）

---

## 滚动汇总

| 维度 | 数值 |
|---|---|
| GitHub 命中条目（去重后相关） | **14** |
| arxiv 命中条目（强相关） | **4** |
| 标注为 `[已知类别]` 的新实例 | 13 |
| 标注为 `[新类别/新签名]` 的候选 | **3** |
| 今日重点关注（🔴 高优先级） | 4 |

> **趋势观察**：本日窗口的故障集中在四个领域——(1) **硬件 SFPU 内核数值原语缺陷的持续暴露**：tt-metal 同一静态审计又发现 bf16 NaN 编码丢失（#51557）和 Quasar `exp` 溢出保护缺失（#51543），与 07-29 的 isclose 缺陷属同一审计家族但机制各异；(2) **长上下文 + 推测解码（MTP/DSPARK）下的静默错误**：Kimi-K3 服务栈暴露 [PAD] 风暴 + NaN 污染 logits（sglang #32968）；(3) **FlexAttention paged KV int32 溢出 → 崩溃或静默错误输出**（vllm #50427），这是一个干净的"整数溢出→静默错误"签名；(4) **arxiv 论文对 prefix-cache 共享块的 bit-flip 攻击做了首次系统刻画**（2604.17249），与编目 K§4 的跨请求缓存键污染互补，明确了"共享 KV 块无完整性校验"的威胁面。

---

## 🔴 重点关注：新故障签名 / 高价值案例

### 1. tt-metal #51557 — bfloat16 在非计算/非转换 op 上丢失全部 254 种 NaN 编码，统一变 Inf `[新签名]`

- **链接**：https://github.com/tenstorrent/tt-metal/issues/51557
- **来源**：tenstorrent/tt-metal（open，2026-07-30）
- **现象**：`bfloat16` 的 NaN 不经过任何 device op 后无法存活——即使是**不执行算术也不做格式转换**的 op（`ttnn.to_layout`、`ttnn.neg`、`ttnn.abs`、`ttnn.typecast(x, bfloat16)`）都把 NaN 变成同符号的 **Infinity**。host round-trip 和 `ttnn.clone` 能保留，说明 NaN 到达设备时是完整的，在返回途中被破坏。报告者构建了一个 build，**全部 254 个 bf16 NaN 编码** 在这些 op 上**比特精确地存活**，且全部 65024 个常规编码不变、无可测性能影响——证明这不是必然的硬件限制。本条 subsumes #31406（open since 2025-10-29）。
- **为什么是新签名**：编目第 4 类"数值/精度"和第 1.8 条 SFPU isclose 覆盖的是**特定算子**的数值错误。本条是**格式本身的 NaN 传播语义缺陷**——dtype 通过看似无害的数据搬运 op 静默改变语义（NaN→Inf），属于"**dtype NaN-canonicalization 静默错误**"新子签名。对 LLM 推理的影响：任何依赖 NaN 作为控制信号（梯度裁剪、稳定性检查、稀疏激活门控的 NaN 屏蔽）的路径在 Tenstorrent 上都会失效。
- **对照编目**：当前 235 条无"bf16 NaN 编码在搬运 op 上静默变 Inf"条目。

### 2. tt-metal #51543 — Quasar `exp` fp32 丢弃溢出/下溢保护，`exp(-100)` 返回 -4.3e33（负值） `[新签名]`

- **链接**：https://github.com/tenstorrent/tt-metal/issues/51543 · 修复 PR #51544
- **来源**：tenstorrent/tt-metal（open，2026-07-29，#51220 的 sub-issue item 3）
- **现象**：`_sfpu_exp_fp32_accurate_` 用 `i = round(a/ln2)` 算 `exp(a) = 2^i · exp(f)`，并把 `2^i` 直接写入 IEEE-754 指数字段 `as<vFloat>((i+127) << 23)`。这仅在 `i+127 ∈ [1,254]` 时合法。越界时 `i+127` 溢出到符号位并被 32 位位移截断为 9 位，结果是静默垃圾——且**经常为负值**：`exp(-88.5)` 返回 `-inf`，`exp(-100)` 返回 `-4.3e33`。Blackhole（移植来源）默认两端都守卫（overflow→inf，underflow→0），Quasar 把两个守卫都丢了。
- **为什么是新签名**：与 07-29 的 1.8/1.9 同属"硬件数学库静默错误值"，但本条是**超越函数的指数域溢出保护缺失**而非比较/同步语义——属于"**SFPU 超越函数范围保护缺失**"子签名。关键危害：exp 是 softmax、GELU、GLU/SiLU 门控的核心，这些在 LLM 每层都有；负值 exp 结果会让门控/softmax 产生反向符号的激活，可能静默翻转专家路由或注意力分布。
- **对照编目**：1.8 的正文提到"同一静态审计还发现了 Quasar exp 缺失范围保护"，但未正式编目；本日将其作为独立条目收录。

### 3. vllm #50427 — FlexAttention paged K/V offsets int32 溢出：崩溃 **或** 静默错误输出 `[新签名候选]`

- **链接**：https://github.com/vllm-project/vllm/issues/50427
- **来源**：vllm-project/vllm（open，2026-07-30，H800 ×8，PyTorch 2.11）
- **现象**：FlexAttention 后端把 paged K/V 的 block offset 打包成 int32。当 `num_gpu_blocks ≥ 2**31 / (block_size · 2 · num_kv_heads · head_size)` 时 offset 回绕。取决于回绕后值落入的地址是否映射，结果是**崩溃**或**静默读取错误 block 的 KV**（attention 用到损坏/错位的缓存）。
- **为什么是新签名候选**：编目 K§3（块/槽分配）和 K§8（控制流）覆盖的是**调度/分配逻辑**错误。本条是**算术整数溢出在 KV offset 编码上**——一个干净的"整型回绕 → 静默错误 KV 寻址"签名，与 RoPE int32 回绕、GQA shape int32 溢出同属更大类。属于"**KV 地址算术整数溢出**"子签名，在长上下文 + 大 KV 池的高显存卡（H800/H200/B200）上随上下文规模趋近触发阈值。
- **对照编目**：K§3 无"offset 整数溢出"条目；K§10.3 是 varlen OOB 写入但根因是布局而非算术溢出。

---

## 🔄 已知类别的新案例（GitHub）

### 服务栈 / KV cache / 调度正确性（第 6 类、K§3/K§5）

| # | 链接 | 标题（简） | 归类 | 说明 |
|---|---|---|---|---|
| 1 | [vllm #50425](https://github.com/vllm-project/vllm/issues/50425) | P/D Mooncake 分离式：decode 节点上的客户端断连不释放 prefill KV cache，孤儿传输累积耗尽 P 节点 KV | `K§5 cross-node-transfer` | 分离式 P/D 的跨节点清理 bug，prefill KV 在 D 取消后仍保留到超时。无 crash，资源泄漏致后续 stall。 |
| 2 | [vllm #50337](https://github.com/vllm-project/vllm/issues/50337) | JinaEmbeddingsV5：EuroBERT backbone 被错误路由到 Qwen3 代码路径 | `第13类 模型图/架构` | 模型注册/路由错误，backbone 类型误判。 |
| 3 | [sglang #32893](https://github.com/sgl-project/sglang/issues/32893) | GLM-5.2 W4AFP8 + EAGLE + TP8：draft prefill → DSA MHA indexer → 无法生成 MTP seed | `K§8 control-flow（推测解码）` + `第7类 量化` | 量化（W4AFP8）+ 推测解码（EAGLE/MTP）交互导致 draft 无法产出 seed。scheduler 异常。 |

### 量化 / 注意力数值（第 7 类、K§9）

| # | 链接 | 标题（简） | 归类 | 说明 |
|---|---|---|---|---|
| 4 | [sglang #32942](https://github.com/sgl-project/sglang/issues/32942) | 统一 deterministic extend attention 对 Grok/xAI 用**与常规路径不同的温度缩放**（reciprocal vs log2） | `第7类 量化` + `第9类 位置编码`（注意力温度缩放） | 不同 attention 后端的 xAI temperature scaling 公式不一致——启用 deterministic extend 路径会改变 Grok 的 attention logits。静默输出差异。 |
| 5 | [trt-llm #17052](https://github.com/NVIDIA/TensorRT-LLM/issues/17052) | Gemma NVFP4 在 RTX PRO 5000 Blackwell（SM120）上无法服务 | `第7类 量化（NVFP4）` | NVFP4 量化在 SM120 消费级 Blackwell 上的服务故障。 |
| 6 | [trt-llm PR #17085](https://github.com/NVIDIA/TensorRT-LLM/pull/17085) | 把 kimiK3AttnRes 固定到 sm_100 家族——`__CUDA_ARCH__ >= 1000` 误纳入 sm_120，`tcgen05`/TMEM PTX 在 sm_120f 上 ptxas 报错 | `第13类 模型图/架构` + 硬件内核 | 自定义 attention kernel 的架构守卫错误：datacenter Blackwell 专用 PTX 被误编译到消费级 Blackwell。 |

### 推测解码 / 长上下文（K§8、第 14 类）

| # | 链接 | 标题（简） | 归类 | 说明 |
|---|---|---|---|---|
| 7 | [sglang #32968](https://github.com/sgl-project/sglang/issues/32968) | Kimi-K3 + DSPARK：长上下文（~90k–237k token）下 [PAD]（id 163839）风暴 + NaN 污染 logits 致 CUDA assert 崩溃；最坏单流 17256 个 [PAD] | `K§8 control-flow` + `数值/NaN` + `第10类 分词器`（PAD 未过滤） | 推测解码（DSPARK）+ 长上下文下 NaN 进入 logits → multinomial assert 或 [PAD] 持续输出。根因：released kimi-k3 image 缺 #32477 修复，PAD 未被 stop/过滤。 |
| 8 | [sglang #32907](https://github.com/sgl-project/sglang/issues/32907) | cutedsl_bf16_gemm 2-CTA TGV kernel：缺尾部 cluster barrier → CUDBG_EXCEPTION_CLUSTER_BLOCK_NOT_PRESENT（SM103/PDL/CUDA-graph，Kimi-K3 shared-expert down proj） | `第1类 硬件/SDC`（内核同步语义） | 2-CTA cluster GEMM kernel 缺退出前 cluster barrier，block 0 退出后 block 1 仍对其 SMEM 发 barrier/TMA。与 1.9 kernel_sync_semantic 同族。 |
| 9 | [trt-llm #17027](https://github.com/NVIDIA/TensorRT-LLM/issues/17027) | GDN FlashInfer decode：`initial_state_indices.int()` 对已是 int32 的张量是 no-op，留下未对齐 index 切片 → Misaligned Tensor (32 字节对齐) 硬错误崩溃 | `第13类 模型图/架构`（kernel dispatch） | GDN（gated delta net）decode 的 index 对齐检查缺失，无 fallback 直接崩溃。 |
| 10 | [trt-llm #17020](https://github.com/NVIDIA/TensorRT-LLM/issues/17020) | `trtllm-eval aime25/aime26` 静默禁用 DeepSeek-V4 的 thinking（`thinking_budget` 键被忽略），准确率减半且无警告 | `第10类 分词器/模板`（对话模板参数） | 评测工具的 chat_template_kwargs 用了模型不认识的键，thinking 静默回退 False。静默错误基准数字。 |

### 内核数值 / 编译器（第 1 类、K§）

| # | 链接 | 标题（简） | 归类 | 说明 |
|---|---|---|---|---|
| 11 | [triton #11111](https://github.com/triton-lang/triton/issues/11111) | float8 e5m2 转置 store lowering 错误：min/max 带 `PropagateNan.ALL` 时转置输出损坏（Inf/NaN 字节），连续输出正确；改写为 `tl.where(a>b|a!=a,a,b)` 消失 | `第1类 硬件/SDC`（内核 codegen） | Triton store lowering/codegen bug，只在 `min.NaN.f32/max.NaN.f32` PTX intrinsic 路径出现，`setp+selp` 形式正确。损坏随 XBLOCK/num_warps 变化 → 寄存器分配/store 向量化问题。 |
| 12 | [vllm #50067](https://github.com/vllm-project/vllm/issues/50067) | EngineCore 在 GB10/sm_121 warmup 期间静默死亡（无 traceback、无 CUDA 错误、无 OOM），`CUDA_LAUNCH_BLOCKING=1` 可避开 | `第6类 服务栈`（静默崩溃） | 消费级 Blackwell（GB10）上 EngineCore 静默死亡，无任何错误信号。属"静默进程死亡"服务栈可靠性问题。 |

### 性能 / 架构集成（第 14 类）

| # | 链接 | 标题（简） | 归类 | 说明 |
|---|---|---|---|---|
| 13 | [trt-llm #16976](https://github.com/NVIDIA/TensorRT-LLM/issues/16976) | 混合 GDN（Qwen3.5-class）高并发下 mixed prefill+decode 的 decode 部分被 eager 执行（~4× attention 成本，~42ms/chunk 开销） | `第14类 长上下文` + 架构集成 | 混合 GDN 架构在并发下 decode 无法保持 graph 化，性能退化（非正确性）。 |

---

## 📄 arxiv 论文发现

### 强相关（LLM 推理故障/可靠性）

| # | 链接 | 标题 | 日期 | 分类 | 说明 |
|---|---|---|---|---|---|
| A1 | [arxiv 2607.25227](http://arxiv.org/abs/2607.25227v1) | **Decision-Level Hijacking: Injecting Cognitive Bias into LLMs via Bit-Flip Attacks** | 2026-07-28 | `[已知类别·bitflip]` 新攻击面 | 提出 CogBias 框架：通过可微情感评估器把主观偏好转成优化信号，用 BitScout 定位关键 bit，在超稀疏翻转预算下实现**定向认知立场操纵**（decision-level hijacking）。在 Llama-3.2-3B/Mistral-7B/Qwen2.5-14B 上，少量 bit 翻转即可稳定诱导目标话题的立场偏移，对非目标任务影响有限。**BFA 从"功能破坏"扩展到"价值对齐操纵"的新攻击目标**。 |
| A2 | [arxiv 2604.17249](http://arxiv.org/abs/2604.17249v2) | **Bit-Flip Vulnerability of Shared KV-Cache Blocks in LLM Serving Systems** | 2026-04-19 | `K§4 cross-request` 新威胁面 | 首次系统刻画 **vLLM Prefix Caching 共享 KV 块的 bit-flip 攻击面**。共享块是单一物理副本、无完整性校验。三大特性：(1) 静默发散——16 个 bf16 bit 位中 13 个产生**连贯但被篡改**的输出，无 clean baseline 不可区分；(2) 选择性传播——仅共享目标前缀的请求受影响；(3) 持久累积——无时间衰减，累积损害随后续请求线性增长。提出 checksum 反制在调度时检测任意单 bit 损害。**明确"共享 KV 块无完整性校验"威胁面**，与编目 K§4 互补。 |
| A3 | [arxiv 2607.19490](http://arxiv.org/abs/2607.19490v1) | **Integrity of peer-to-peer distributed LLM inference under malicious nodes** | 2026-07-21 | `[已知类别·分布式]` 检测方法 | P2P 分布式推理（层跨多节点）下任意节点可篡改其层输出。提出**金丝雀输入（canary）混合检测法**：选一组 secret canary 输入（已知正确激活），混入常规流量；恶意节点无法区分 canary 与真实查询，篡改时 canary 偏差暴露。408 配置下检测器达 AUROC 1.0。**为分布式推理完整性检测提供方法**，对齐编目第 5 类。 |
| A4 | [arxiv 2607.13205](http://arxiv.org/abs/2607.13205v1) | **Adaptive Filtering of the KV Cache: Diagnosing and Correcting Structural-Role Bias in LLM Inference** | 2026-07-14 | `K§6 eviction-compression` 方法论 | 诊断并纠正 KV cache 的**结构角色偏差**（structural-role bias）——某些 token 因位置/角色被系统性地过度保留或驱逐，引入自适应过滤。为 KV 驱逐策略的正确性（而非仅效率）提供新视角。 |

---

## 🆕 新类别 / 新签名汇总（建议编目收录）

1. **tt-metal #51557**：bf16 NaN 编码在非计算搬运 op 上静默变 Inf（NaN→Inf 的 dtype 语义破坏）。建议在 1.8/1.9 同族下新增"**dtype NaN-canonicalization 静默错误**"子签名。
2. **tt-metal #51543**：SFPU `exp` 超越函数指数域溢出保护缺失，`exp(-100)` 返回负值。建议新增"**SFPU 超越函数范围保护缺失**"子签名。
3. **vllm #50427**：FlexAttention paged KV offset int32 溢出→崩溃或静默错误 KV 寻址。建议在 K§3 增设"**KV 地址算术整数溢出**"条目。

---

## API 调用记录

| # | 查询 | 结果 |
|---|---|---|
| Q1 | vllm `label:bug` created:>=2026-07-23 | 15 命中（取全部） |
| Q2 | vllm `silent/wrong/corruption/NaN/correctness/determinism` in:title (7d) | 15 命中（取全部） |
| Q3b | sglang is:issue created:>=2026-07-23 | 20 命中（取全部） |
| Q4 | TensorRT-LLM is:issue created:>=2026-07-23 | 18 命中（取全部） |
| Q5 | transformers is:issue (RoPE/tokenizer/quant/NaN/wrong/silent) in:title (7d) | 3 命中 |
| Q6 | triton is:issue created:>=2026-07-23 | 9 命中 |
| Q7 | tt-metal correctness/silent/NaN/inf/precision (7d) | 20 命中 |
| Q8 | 全局 "silent data corruption"/"bit flip"/"KV cache corruption"/"weight corruption" (7d) | 20 命中（多为噪声） |
| Q9 | 全局 "quantization error"/"FP8 wrong"/"NVFP4"/"non-deterministic"/"numerical instability" (7d) | 20 命中（多为噪声） |
| Detail | 13 次单 issue GET（vllm/sglang/trt-llm/tt-metal/triton） | 全成功 |
| arxiv | 5 组关键词（SDC/bitflip/KV/correctness/GPU soft error） | 5/5 成功（HTTPS `-L`） |

> **rate limit**：全程 remaining 60/60，未触发限速。

---

*报告生成于 2026-07-30 UTC，由 LLM 推理可靠性每日扫描 cron 作业自动生成。*
