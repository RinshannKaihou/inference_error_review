# LLM 推理故障每日扫描 — 2026-07-28

> **扫描日期**：2026-07-28（UTC）
> **扫描窗口**：GitHub issues/PRs `created:>2026-07-25`（部分全局查询 `>2026-07-20`）；arxiv 按提交日期降序取最近 5 篇
> **搜索范围**：6 个 GitHub 目标仓库（vllm / sglang / transformers / TensorRT-LLM / triton / sgl-project/sglang）× 8 组关键词 + arxiv 6 组关键词
> **对照基准**：[`FAULT_MASTER_REFERENCE.zh.md`](FAULT_MASTER_REFERENCE.zh.md)（211 条已编目故障，三层来源）
> **API 状态**：GitHub Search API 8/8 成功（无 rate-limit）；arxiv 6/6 成功（HTTPS）

---

## 滚动汇总

| 维度 | 数值 |
|---|---|
| GitHub 命中条目（去重后相关） | **17** |
| arxiv 命中条目（强相关） | **4** |
| 标注为 `[已知类别]` 的新实例 | 14 |
| 标注为 `[新类别/新签名]` 的候选 | 4 |
| 今日重点关注（🔴 高优先级） | 5 |

> 说明：本日为该项目**首次**执行每日扫描，因此窗口略宽（≈3 天），后续每日窗口收敛到 24h。大量命中来自 vLLM/SGLang 对 **Kimi-K3** 和 **Gemma-4** 新架构的集成热潮——新架构集成期是静默正确性 bug 的高发窗口。

---

## 🔴 重点关注：新故障签名 / 高价值案例

### 1. SGLang #32666 — Frozen-KV MTP 在并发下输出损坏，**触发器是 draft depth 而非 batch size** `[新签名]`

- **链接**：https://github.com/sgl-project/sglang/issues/32666
- **来源**：sgl-project/sglang（open，2026-07-28）
- **现象**：单卡 B200 服务 `Gemma-4-31B-IT-NVFP4`，开启 Frozen-KV MTP（`--speculative-algorithm NEXTN`）。并发 ≥2 时所有无 grammar 约束的请求返回"退化 token 泥浆"并跑到 `max_tokens`；串行干净。
- **为什么是新签名**：报告者明确**证伪**了"verify batch 太大"的常规诊断——`depth 3 @ concurrency 6` 干净，而 `depth 5 @ concurrency 4` 损坏，两者 `concurrency × draft_tokens` 乘积相同（24）却结果相反。**触发器是单请求 draft 链深度**，不是批量大小。`depth 4` 时服务器甚至无法启动（verify CUDA graph 捕获时 illegal memory access）。
- **对照编目**：最近似 `K§8 control-flow（推测解码 / beam / CUDA-graph）`，但编目中无"**draft depth 门控**"这一诊断签名。建议作为 control-flow 类的新子签名收录。
- **影响**：生产级静默输出损坏，8/8 请求受影响。

### 2. SGLang #32652 — PD 分离式部署 Decode 控制路径失败对健康检查**不可见** `[新类别候选]`

- **链接**：https://github.com/sgl-project/sglang/issues/32652
- **来源**：sgl-project/sglang（open，2026-07-28，含 GHSA-98f6-8g99-45hr 私密漏洞披露）
- **现象**：Prefill/Decode 分离部署（Mooncake transfer backend）下，Decode 控制路径可失败，而 Decode 进程仍在运行、Decode 与 Router 的 `/health` 仍返回 **HTTP 200**，后续端到端推理请求一直 stall 到超时。
- **为什么是新类别候选**：编目第 6 类"服务栈/缓存正确性"覆盖了缓存键污染、调度错误等，但**"健康检查绿而控制路径死"的静默活性失败（silent liveness gap）**作为一个**可观测性层面的故障类**在 211 条中没有显式条目。报告者已于 2026-07-21 提交私密漏洞报告并公开时间戳。
- **影响**：依赖健康检查做故障转移的 router/orchestrator 会持续把流量打到死节点。

### 3. SGLang #32555 — Nemotron 混合架构 CUDA-graph 下 track-save 读取**stale tail** → 静默 logprob 损坏 `[已知类别·新机制]`

- **链接**：https://github.com/sgl-project/sglang/pull/32555
- **来源**：sgl-project/sglang（open PR，2026-07-27）
- **现象**：Mamba2 混合架构（NemotronH / Nemotron-3-Super-120B）+ CUDA graph 开启时，prefix 命中 decode-seeded mamba 条目返回**损坏的 logits**。KL ≈ 1.55（mean logprob diff ≈ 0.61）；关 CUDA graph 后 KL ≈ 0.002。无 crash、无 assert——**纯静默损坏**。
- **根因**：decode CUDA graph 下 `_replay_metadata` 每次 replay 在 `[:bs]` 前部重填静态 track 目标缓冲 `mamba_track_indices_buf`，却把**整个 max_bs 缓冲**交给了 `ForwardMetadata`，导致尾部 stale 数据被读。
- **对照编目**：`K§7 arch-specific-cache` + `K§8 control-flow（CUDA-graph）`。Mamba/radix cache × CUDA graph replay 的交互是**新架构集成期**的典型新实例。

### 4. SGLang #32564 — 传输在途时释放 KV 页 → 下一请求被覆盖 `[已知类别·跨节点]`

- **链接**：https://github.com/sgl-project/sglang/pull/32564
- **来源**：sgl-project/sglang（open PR，2026-07-27）
- **现象**：prefill→decode 传输期间发生 abort/timeout 会把请求的 KV 页归还分配器，**但不取消传输**。下一个请求分配到同一批页后，被仍在途的旧写入覆盖——下一请求持有**上一请求的 KV**，全程无任何错误抛出。
- **对照编目**：精确命中 `K§5 cross-node-transfer（分离式 / 卸载 / RDMA）`。报告者注明窗口窄但生产中因频繁 abort 慢请求而实际命中，**唯一症状是"给用户的答案略错"**。

### 5. vLLM #50084 — NVFP4 KV 无条件按 SM100 swizzle 写 V block scales，**静默损坏** SM120 FlashInfer XQA 路由 `[已知类别·量化×KV]`

- **链接**：https://github.com/vllm-project/vllm/issues/50084（修复 PR #50085）
- **来源**：vllm-project/vllm（open，2026-07-28，标签 quantization）
- **现象**：`reshape_and_cache_nvfp4_kernel` 写 K block scales 用线性布局，写 V block scales 用 SM100 `trtllm-gen` MHA 所需的 4×4 swizzle——**对所有架构无条件执行**。在 SM120（RTX 5090）上 NVFP4 KV 路由（FA2 prefill + FlashInfer XQA decode）通过**普通 NHD tensor strides** 消费 scale 张量，XQA CUDA 代码**无 V-scale de-swizzle**。
- **根因**：swizzled producer 接到 linear consumer；所有张量 shape 正确故无报错。
- **对照编目**：`§7 量化（PTQ/低位/FP8/微缩放）` × `§8 KV 缓存`。架构条件化的 layout 不匹配是 NVFP4 这一代新精度格式的典型新实例。

---

## 📋 全部命中条目（按仓库分组）

### vLLM（vllm-project/vllm）

| # | 标题 | 日期 | 分类 |
|---|---|---|---|
| [#50084](https://github.com/vllm-project/vllm/issues/50084) | NVFP4 KV 无条件按 SM100 swizzle 写 V block scales，静默损坏 SM120 XQA | 07-28 | `[已知类别]` 量化×KV（🔴 见上） |
| [#50147](https://github.com/vllm-project/vllm/issues/50147) | Kimi-K3 (TP=8, prefix caching / mamba 'align') 并发下 recurring illegal-memory-access，MTBF 与并发反相关 | 07-28 | `[已知类别]` 混合架构×分布式§5/§6 |
| [#50158](https://github.com/vllm-project/vllm/issues/50158) | EAGLE `embed_tokens` 跨 TP rank 判定不一致：rank0 保留 drafter embedding、rank1 复用 target 的；acceptance 崩到 ~0.45/draft | 07-28 | `[已知类别]` 分布式/TP§5 × 推测解码 |
| [#50159](https://github.com/vllm-project/vllm/issues/50159) | Model Runner V2 + Gemma-4：profiler 多报 ~0.89 GiB "available KV" → 饱和负载下 CUDA OOM | 07-28 | `[已知类别]` KV 容量配置§8 |
| [#50150](https://github.com/vllm-project/vllm/issues/50150) | eager `reshape_and_cache_flash` 路径 host-memory 泄漏 | 07-28 | `[已知类别]` KV cache 内核§8（资源泄漏） |
| [#50136](https://github.com/vllm-project/vllm/issues/50136) | `should_custom_ar()` 的 size 阈值使 all-reduce kernel 选择**依赖 batch 组成** → 非确定 FP 归约顺序 | 07-28 | `[已知类别]` 确定性§4 / 分布式§5 |
| [#50137](https://github.com/vllm-project/vllm/pull/50137) | 不要在 `RoutedExperts.load_weights` 转置 fused MoE 量化 scales：fused per-channel scale `[E,2*I,1]` 被误转置 → 错轴 → shape `[0,1]` | 07-28 | `[已知类别]` 量化加载§7 × MoE |
| [#50085](https://github.com/vllm-project/vllm/pull/50085) | [NVFP4] 在 SM120/SM121 KV cache 写 linear V block scales（#50084 修复） | 07-28 | `[已知类别]` 量化×KV |

### SGLang（sgl-project/sglang）

| # | 标题 | 日期 | 分类 |
|---|---|---|---|
| [#32666](https://github.com/sgl-project/sglang/issues/32666) | Frozen-KV MTP 并发下输出损坏，**draft depth 门控**（🔴 见上） | 07-28 | `[新签名]` control-flow§K§8 |
| [#32652](https://github.com/sgl-project/sglang/issues/32652) | PD 分离式 Decode 控制路径失败对健康检查不可见（🔴 见上） | 07-28 | `[新类别候选]` 服务栈可观测性§6 |
| [#32555](https://github.com/sgl-project/sglang/pull/32555) | Nemotron decode track-save 读 CUDA-graph track buffer stale tail → 静默 logprob 损坏（🔴 见上） | 07-27 | `[已知类别·新机制]` arch-specific§K§7×CUDA-graph |
| [#32564](https://github.com/sgl-project/sglang/pull/32564) | Mooncake 传输在途时释放 KV 页 → 下一请求被覆盖（🔴 见上） | 07-27 | `[已知类别]` cross-node-transfer§K§5 |

### TensorRT-LLM（NVIDIA/TensorRT-LLM）

| # | 标题 | 日期 | 分类 |
|---|---|---|---|
| [#16944](https://github.com/NVIDIA/TensorRT-LLM/pull/16944) | MRoPE delta 读槽在混合 batch 中未保持 dense：attention kernel 按 generation batch index 索引 `mrope_position_deltas`，后续请求读到**别人的 delta**且尾部 OOB；**静默损坏，无 shape check** | 07-28 | `[已知类别]` RoPE§9 × 混合 batch |
| [#16936](https://github.com/NVIDIA/TensorRT-LLM/pull/16936) | Qwen3.5 weight-load：`mark_consumed(name)` 过度消费整个子树 → 后代模块未初始化（**garbage output**）；host RSS 不释放致 OOM | 07-28 | `[已知类别]` 权重加载§12 |
| [#16933](https://github.com/NVIDIA/TensorRT-LLM/pull/16933) | 支持 large FP8 quantization grids：规避 CUDA 65535 维 grid 限制（nvbugs/6424956） | 07-28 | `[已知类别]` 量化§7（大 token 数边界） |

### Triton（triton-lang/triton）

| # | 标题 | 日期 | 分类 |
|---|---|---|---|
| [#11071](https://github.com/triton-lang/triton/issues/11071) | `tl.topk` 接受非法 k 值（0、-1、非 2 的幂、大于维度）→ 编译成**看似合理实则错误**的结果而非报错 | 07-28 | `[新签名]` 内核编译器正确性——静默错误输出 |

> 注：triton #11071 是编译器层面的**输入契约缺失**类问题——无效输入被静默降级为"plausible but incorrect"输出。这属于一个值得关注的**内核正确性盲点**：上层 LLM kernel 若误传 k 值，会得到错误 attention/topk 结果而无任何异常信号。

### HuggingFace transformers

- 本次窗口内 `wrong output / incorrect / NaN inference` 查询返回 **0 条**（`created:>2026-07-25`）。transformers 仓库近 3 天无匹配的正确性 issue。

---

## arxiv 论文（强相关）

arxiv 短语查询对 "silent data corruption AND LLM" 等返回的多为不相关结果（arxiv 全文检索的短语匹配较弱），但量化/精度相关查询命中了以下近期工作：

| 标题 | 日期 | 链接 | 分类 |
|---|---|---|---|
| ACRL: Adaptive Control of Training-Inference Discrepancy for Stable RL | 2026-07-27 | [arxiv 2607.24062](http://arxiv.org/abs/2607.24062v1) | `[已知类别]` 训练-推理数值差异§4 |
| C-PTQ: Fisher-weighted Channel-wise Sensitivity for PTQ of MLLMs | 2026-07-23 | [arxiv 2607.21076](http://arxiv.org/abs/2607.21076v1) | `[已知类别]` 量化§7（通道敏感度缓解精度损失） |
| MXSens: Sensitivity-Aware Mixed-Precision Quantization for Efficient LLM Inference | 2026-07-20 | [arxiv 2607.17733](http://arxiv.org/abs/2607.17733v1) | `[已知类别]` 量化§7（混合精度，outlier 处理） |
| SelectInfer: Selective Neuron Loading for On-Device LLMs | 2026-07-20 | [arxiv 2607.18081](http://arxiv.org/abs/2607.18081v1) | `[已知类别]` 权重/检查点加载§12 |

> 本日 arxiv 无全新的故障**模式**论文；命中均为已知类别（量化精度缓解、训练-推理差异）的工程改进。

---

## 趋势观察

1. **新架构集成期 = 静默损坏高发期**：本日 17 条相关命中中，绝大多数围绕 **Kimi-K3**（混合 Mamba+attention MoE）、**Gemma-4**（hybrid sliding/global attention）、**Nemotron**（Mamba2 混合）、**NVFP4/SM120**（Blackwell 新精度）的集成。混合架构 + CUDA graph + 推测解码三者叠加时，track buffer / KV layout / draft depth 的契约极易被破坏且**无 shape check 兜底**。
2. **"静默"是本日关键词**：#32666、#32555、#32564、#16944、#50084、#11071 全部明确出现 "silent"/"no error"/"no shape check"/"plausible but incorrect" 表述——损坏不 crash、不报错、只产生略错或完全退化的输出。
3. **可观测性缺口**：#32652（健康检查绿而控制路径死）提示**服务栈活性检测**本身可能成为故障掩盖层，值得作为新故障子类跟踪。
4. **编译器契约缺失**：triton #11071 揭示上层 kernel 作者若误用 `tl.topk` 的 k 值，会得到静默错误结果——这是 LLM attention/topk kernel 的一个上游供应链风险点。

---

## 附录：搜索查询清单（可复现）

**GitHub Search API（8 次，均成功）**
1. `repo:vllm-project/vllm wrong OR incorrect OR corruption created:>2026-07-25`
2. `repo:vllm-project/vllm NaN OR inf OR silent created:>2026-07-25`
3. `repo:vllm-project/vllm KV cache OR quantization OR quant created:>2026-07-25`
4. `repo:sgl-project/sglang wrong OR incorrect OR corruption OR NaN created:>2026-07-25`
5. `repo:huggingface/transformers wrong output OR incorrect OR NaN inference created:>2026-07-25` → 0 命中
6. `repo:NVIDIA/TensorRT-LLM correctness OR wrong OR NaN OR corruption created:>2026-07-25`
7. `repo:triton-lang/triton numerical OR wrong OR incorrect OR precision created:>2026-07-25`
8. `"silent data corruption" OR "bit flip" OR "weight corruption" created:>2026-07-20`（全局，噪声大）

**arxiv API（6 次，HTTPS，均 HTTP 200）**
1. `all:"silent data corruption" AND all:LLM`
2. `all:"bit flip" AND all:LLM`
3. `all:"KV cache" AND all:corruption`
4. `all:"LLM inference" AND all:correctness`
5. `all:GPU AND all:"soft error" AND all:learning`
6. `all:quantization AND all:LLM AND all:accuracy`

---

## 🔄 晚间增量扫描（2026-07-28 ~23:01 UTC）

> 本节为同日**第二次扫描**（首次见上文 17:46 UTC），窗口 `created:>2026-07-27`，覆盖首次扫描后约 5 小时内新增的 issue。GitHub 8/8 查询成功，arxiv 7/7 成功（HTTPS）。已去重首次扫描条目。**共发现 12 条新的相关条目**，其中 3 条为静默正确性新签名/新机制。

### 🆕 增量 · 新发现（首次扫描后新增）

#### 1. transformers #47597 — 外部构建 `past_key_values` 时位置跟踪**静默损坏**（无异常） `[已知类别·教科书级静默签名]`

- **链接**：https://github.com/huggingface/transformers/pull/47597
- **来源**：huggingface/transformers（closed PR，2026-07-28 04:45 UTC）
- **现象**：`generate()` 接受用户自建 `past_key_values`，但当 `attention_mask` 长度 ≠ `input_ids` 长度时（恰好是 prefill/decode 分离做压缩后重建 cache 的典型场景），`_prefill()` **静默错误处理位置**。**输出错误且无异常、无 crash**——报告者明确指出"the kind of bug that passes CI and makes it into production"。实测 Qwen3-VL-8B INT8 量化在 MMBench-Video 上因该 bug 掉 **10.7pp**（0.303→0.196）。
- **对照编目**：精确命中 `§8 KV 缓存` + `§9 RoPE/位置编码`，且是 `K§6 eviction-compression（驱逐/压缩后重建 cache）` 的上游库级实例。这是本项目编目中"prefill/decode 分离 + cache 重建 → 位置错位"的**最干净的工业级实证**。

#### 2. triton #11081 — AMD 后端 `emitFence` 的 acquire/release 栅栏**被交换** `[新签名候选]`

- **链接**：https://github.com/triton-lang/triton/issues/11081
- **来源**：triton-lang/triton（open PR，2026-07-28 20:15 UTC）
- **现象**：`getOrderingFlags()` 返回 `{emitAcquireFence, emitReleaseFence}`，但调用方按 `{emitReleaseFence, emitAcquireFence}` 绑定——acquire 路径发了 pre-atomic release 栅栏，release 路径反之。即**内存序被成对反转**，可导致 buffer atomic 操作的竞态 / 非确定结果。
- **为什么值得关注**：这是**编译器后端的内存序正确性**问题，与首次扫描的 #11071（`tl.topk` 静默错误）同属"编译器契约缺失/反转"家族。LLM kernel 在 AMD/ROCm 上做 buffer atomic（如 attention 的 reductions、all-reduce 路径）若依赖正确栅栏序，会得到**静默非确定或损坏**输出，无任何异常。211 条编目无"编译器后端栅栏内存序"显式条目——建议作为新子签名跟踪。

#### 3. SGLang #32669 — DeepSeek-V4 non-EP TBO 在 attention-TP>1 时 gather/combine 元数据**静默错配** `[已知类别·分布式新实例]`

- **链接**：https://github.com/sgl-project/sglang/issues/32669
- **来源**：sgl-project/sglang（open bug，2026-07-28 16:43 UTC）
- **现象**：DeepSeek-V4 non-EP 两批 overlap 在 `--enable-dp-attention` 且 `attn_tp_size>1`（如 TP16/DP4 attn-TP4，或 TP16/DP2 attn-TP8）时，`_forward_layers_tbo` 跨整个 TP 组 gather 每个 child 的 padded 长度与 input IDs，使每个 DP shard 被每个 attn-TP rank 各表示一次；`dp_reduce_scatterv_async` 用这些**重复的尺寸**做全 TP reduce-scatter——**Hidden rows 与 input IDs 不再与 DP shard 一一对应**。
- **对照编目**：`§5 分布式/多GPU` + `§6 服务栈`。**现有 CI 用 TP8/DP8（attn_tp_size==1）未覆盖**，故静默漏网——典型的"配置空间盲点"实例。

### 🔄 增量 · 已知类别的新案例

| 仓库 | # | 标题（时间） | 分类 |
|---|---|---|---|
| vllm | [#50067](https://github.com/vllm-project/vllm/issues/50067) | **EngineCore 在 GB10/sm_121 warmup 时静默死亡**——无 traceback、无 CUDA 错、无 OOM、无 segfault；进程进入 `R` 状态几秒后死，父进程收 SIGTERM。`CUDA_LAUNCH_BLOCKING=1` 可规避（07-28 02:14） | `[已知类别]` 异步/启动正确性§6（静默死亡，可观测性盲点） |
| vllm | [#50183](https://github.com/vllm-project/vllm/issues/50183) | 推测解码 rejection sampler：`tl.argmax` 在全 NaN 块返回越界 block index → 非法内存访问；用 `tl.where` 映射 NaN→-inf 并 clamp 修复（07-28 21:06） | `[已知类别]` NaN 处理§4 × 推测解码§K§8 |
| vllm | [#50181](https://github.com/vllm-project/vllm/issues/50181) | Kimi-K3 fp8 KV：`use_prefill_query_quantization` 标志被 MLA prefill backend 选择**静默丢弃**（07-28 20:38） | `[已知类别]` 量化§7 × KV§8（配置静默丢弃） |
| vllm | [#50072](https://github.com/vllm-project/vllm/issues/50072) | GB10/sm_121 warmup 显式加 sync 点以**暴露异步 CUDA 错误**（07-28 03:08） | `[已知类别]` 异步错误可观测性§6 |
| sglang | [#32658](https://github.com/sgl-project/sglang/pull/32658) | KDA chunk recurrence：radix `extra_buffer` 从 BF16 的 `h` 拷贝到 FP32 累加器 pool 时**静默截断精度**，checkpoint 恢复时已降精度（07-28 13:39） | `[已知类别]` 数值精度§4 × radix§K§6（静默舍入） |
| sglang | [#32606](https://github.com/sgl-project/sglang/pull/32606) | 混合 Mamba 模型 + 推测解码时，mamba tracking 被 `spec_algorithm` 关闭后**仍 replay prefill CUDA graph** → 精度回归（07-28 03:52） | `[已知类别]` CUDA-graph§K§8 × arch§K§7（精度回归） |
| sglang | [#32670](https://github.com/sgl-project/sglang/pull/32670) | gemma3：fuse 高秩 RMSNorm 并**guard 混合 dtype 权重**（07-28 17:17） | `[已知类别]` 权重/dtype§12 × 归一化精度 |
| trt-llm | [#16939](https://github.com/NVIDIA/TensorRT-LLM/issues/16939) | `trtllm-serve` 启动期持有 **bound 但未 listen 的 socket** → TCP 探测得 RST，与"进程已死"字节级相同，健康探针无法区分"加载中"与"崩溃"（07-28 07:52） | `[已知类别]` 服务栈可观测性§6（与 #32652 静默活性 gap 同族） |
| trt-llm | [#16925](https://github.com/NVIDIA/TensorRT-LLM/issues/16925) | DSA：在 MTP draft loop 内**重建 token→request 映射**（07-28 02:40） | `[已知类别]` 推测解码§K§8 |
| trt-llm | [#16948](https://github.com/NVIDIA/TensorRT-LLM/issues/16948) | 注册 NIXL 内存时**跳过空 aux buffer**（07-28 10:42） | `[已知类别]` cross-node/卸载§K§5 |
| trt-llm | [#16965](https://github.com/NVIDIA/TensorRT-LLM/issues/16965) | DSpark 推测 drafter + 结构化输出的正确性修复（07-28 18:11） | `[已知类别]` 推测解码§K§8 × 采样§11 |

### 📄 增量 · arxiv 新发现

| 标题 | 日期 | 链接 | 分类 |
|---|---|---|---|
| From Bit-Position Sensitivity to Unequal Error Protection for DNN Inference Memory | 2026-07-21 | [arxiv 2607.19623](http://arxiv.org/abs/2607.19623v1) | `[已知类别]` bit flip§3（按比特位置敏感度的非均等错误保护，推断内存防护） |

> 补充（非 24h 窗口但与今日主题强相关，供编目参考）：
> - *Demystifying Numerical Instability in LLM Inference*（2606.21023, 2026-06-19）— 数值不稳定性与可复现推断，§4。
> - *The Illusion of Equivalence: Systematic FP16 Divergence in KV-Cached Autoregressive Inference*（2604.15409, 2026-04-16）— KV-cache × FP16 系统性发散，§4×§8。
> - *When Latent Agents Lie: KV-Cache Integrity in Multi-Agent LLM Collaboration*（2606.28958, 2026-06-27）— KV-cache 完整性 / 跨请求污染，§K§4。

### 📈 晚间趋势补充

5. **`generate()` + 用户自建 KV cache 的静默位置 bug 跨库存在**：transformers #47597 证明上层 API 在 prefill/decode 分离 + cache 重建时静默错位——这正好印证本日早间发现的 SGLang/TRT-LLM 一系列"cache 重建/压缩后位置/精度静默损坏"主题，**从框架库到服务栈形成完整链路**的同一类静默故障。
6. **编译器后端成为静默故障新上游源**：triton #11081（AMD 栅栏反转）+ 早间 #11071（topk 静默错误）共指"kernel 编译器后端正确性缺口"——LLM 推理 kernel 作者即使逻辑正确，也可能因底层内存序/输入契约被静默误导。值得作为新族别跟踪。

### 增量附录：查询清单（晚 23:01 UTC）

**GitHub（8 次，均成功）**
1. `repo:vllm-project/vllm wrong OR incorrect OR corruption OR silent OR NaN created:>2026-07-27`
2. `repo:sgl-project/sglang wrong OR incorrect OR corruption OR silent OR NaN OR correctness created:>2026-07-27`
3. `repo:NVIDIA/TensorRT-LLM correctness OR wrong OR NaN OR corruption OR silent OR incorrect created:>2026-07-27`
4. `repo:triton-lang/triton numerical OR wrong OR incorrect OR precision OR NaN OR corruption created:>2026-07-27`
5. `repo:huggingface/transformers wrong OR incorrect OR NaN OR corruption OR silent created:>2026-07-27`
6-8. 细节查询：vllm #50067/#50183/#50181、sglang #32669/#32658/#32606、triton #11081、trt-llm #16939

**arxiv（7 次，HTTPS，均 HTTP 200）**：LLM+inference+correctness、bit flip+LLM、GPU+soft error+learning、KV cache+corruption、quantization+LLM+accuracy、silent data corruption+LLM、numerical instability+LLM

---

*报告由每日 cron 扫描自动生成。对照基准为 `FAULT_MASTER_REFERENCE.zh.md`（211 条）。本日为同日第二次（晚间增量）扫描。*
