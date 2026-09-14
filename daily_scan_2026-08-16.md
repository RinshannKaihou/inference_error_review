# 每日扫描报告 — 2026-08-16

- 扫描日期：2026-08-16（滚动窗口 created:>=2026-08-09，7 天）
- 数据源：GitHub Search API（vLLM / SGLang / TensorRT-LLM / Triton / tt-metal / transformers）+ arxiv API
- 结论：**发现 24 条新条目**（全部确认与主文档现有 issue 编号 / arxiv ID / 故障名不重复），另有 1 条补充来源引用（vLLM PR #52543 → 12.18）。合并后主文档 548 → **572 条**。

---

## 一、重点新发现（新类别 / 新签名）

### 1. vLLM #52531 — Kimi-K3 多机 TP CUDA graph 捕获三种模式静默损坏输出 ⭐
- 链接：https://github.com/vllm-project/vllm/issues/52531
- TP16 双节点 B200（EFA、无 NVSwitch/MNNVL）服务 Kimi-K3：任何形式的 CUDA graph 捕获都产生错误输出或 crash，`enforce_eager` 100% 正确。三种模式：FULL_AND_PIECEWISE → 流利但语义随机的多语文本 + 第 3 次请求 host SIGSEGV（cuGraphLaunch）；PIECEWISE → 确定性塌缩为单 token 重复（`@ @ @ ...`）；BREAKABLE=0 → illegal memory access。FlashInfer fused all-reduce 在该拓扑被禁用（#50386 的 latent-MoE 修复不适用），collectives 走 NCCL over EFA。batch=1 首请求即复现。
- 分类：`[新类别/新签名]`（多机无 NVSwitch 拓扑 × CUDA graph × 非 FlashInfer all-reduce 路径）→ 合并为 **6.99**

### 2. vLLM PR #52553 — FlashInfer 共享 GEMM workspace `resize_()` 搬迁地址 → CUDA graph replay use-after-free ⭐
- 链接：https://github.com/vllm-project/vllm/pull/52553
- FlashInfer `gemm_base.py` 的共享 32 MiB workspace 在执行计划需要更多空间时用 `tensor.resize_()` 扩容——`resize_()` 释放旧 storage 并搬迁 buffer，已捕获的 CUDA graph 仍解引用旧地址，caching allocator 立即将其复用给无关张量 → 每次 replay 都是 use-after-free。仪表化证据：cuDNN FP8 plan 比 32 MiB 多要 256 字节、发生在 piecewise 捕获中途；8 张已捕获 graph 持有 stale pointer，服务数分钟后死亡，GPU coredump 显示 `cudnn_generated_fort_native_*` fault 在「两个 int32 拼出来的指针」上。根因修复（上游 flashinfer#4553 + vLLM 侧捕获前预扩容）。这一条同时解释了 SM120 NVFP4 持续负载 crash（#52540，及 #34948 部分报告）。
- 分类：`[新类别/新签名]`（运行时 buffer 搬迁 × graph 捕获指针冻结）→ 合并为 **6.100**（#52540 作为症状来源并入）

### 3. SGLang PR #35049 — decode 侧 abort 不取消在飞 KV 传输 → 页面被复用后遭迟到写入覆写 ⭐
- 链接：https://github.com/sgl-project/sglang/pull/35049
- 长 ITL abort / `/abort_request` 时，prefill→decode KV transfer 仍在飞行，但请求的 KV pages 与 req-pool slot 被**立即释放且不取消传输**。下一个请求分到同一批 page 后，被仍在飞行的旧写入覆写——**下一个请求静默持有被污染的 KV**。教科书级服务栈 SDC 模式（释放-复用竞态 × 异步传输）。
- 分类：`[新类别/新签名]`（abort 路径 × 异步 KV 传输生命周期）→ 合并为 **6.104**

### 4. vLLM PR #52516 — Mooncake 异构 TP 复制 GQA head 传输缺口：KV-ready 但 75% 字节缺失 ⭐
- 链接：https://github.com/vllm-project/vllm/pull/52516
- `producer_cache_replicated=True` 被规划器当成「每个 rank 持有整块」。但 `num_kv_heads < producer_tp_size` 时每个唯一 KV head 被复制、每个 rank 只拥有一个 head 大小的切片。TP8 prefill → TP1 decode、4 个 KV head：decode 期望 262,144 字节整块，旧规划器只从 rank 0 传 offset 0（65,536 字节），**其余 196,608 字节未传输，而 decode 仍按 Mooncake 的成功回执标记 KV-ready** → 基于不完整 KV 的静默错误输出。官方 Qwen3-235B-A22B-FP8 + vLLM v0.27.1 复现（非定制模型问题）。
- 分类：`[新类别/新签名]`（异构 TP 规划 × head 复制语义）→ 合并为 **5.17**

### 5. vLLM PR #52512 — DeepSeek V3.2 sparse MLA 短 prefill 误派发 dense MHA，但 wrapper 只实现 MQA → 输出退化为重复 token ⭐
- 链接：https://github.com/vllm-project/vllm/pull/52512
- 通用 sparse-MLA 元数据路径在 prefill 序长 ≤ `index_topk` 时可选 dense MHA 并允许 sparse indexer 跳过 top-k 打分；但 `DeepseekV32Attention` 只调 `forward_mqa`、不执行 dense-MHA 后端 → MQA 消费被清空的 all-`-1` top-k buffer，**GLM-5.2 输出退化为重复 token**（静默）。
- 分类：`[新类别/新签名]`（能力协商只查后端不查模型层）→ 合并为 **8.43**

### 6. 其余新签名
- **vLLM PR #52530 / #52520**：hybrid Mamba `--mamba-cache-mode align`，近 KV 池上限的请求被接纳、chunk-prefill 到 98.9%、最后一块分不到被无限重排——**零输出 token、整个 prefill 反复丢弃、无 preemption 记账**（livelock）。→ **6.101**（关联 PR #52527：稀疏保留 checkpoint 缺失时 prefix cache 把「找到又丢弃」报成干净 miss，观测盲区）
- **vLLM PR #52541**：`LazyDict` 首读缓存值，`__setitem__` 只换 factory 不失效缓存 → 先读后写的 key 永远返回旧值（配置对象静默陈旧）。→ **6.102**
- **vLLM #52544 / PR #52545**：`VLLM_USE_PRECOMPILED` 选定 CUDA variant 元数据缺失时**无兼容性校验地回退** root/default wheel（fail-open 安装路径）。→ **6.103**
- **vLLM #52551**：Qwen3.6-27B（dense Gated-DeltaNet）V1 引擎两种模式永久硬 wedge（2 张大图：scheduler 准入卡死，调大 `--max-num-batched-tokens` 修复；~180k 长多轮 prefill 并发：不修复）——生成冻结而 `/health` 仍 200，仅重启可恢复。→ **6.106**
- **TRT-LLM PR #17769 / #17763**：HTTP cluster-storage 点读已拒绝过期条目，但 `get_prefix()` 不查 TTL → 过期到后台清扫之间的窗口内**陈旧 key 仍可见**。→ **6.105**
- **SGLang PR #35020**：dense FP8 Marlin 的 bias 被 permute 进 Marlin tile 布局，但 wrapper 在 GEMM 返回后按**原始逻辑通道序**加 bias → Qwen2 QKV projection 等投影结果错误。→ **7.38**
- **vLLM PR #52550**：sparse-attention indexer K-cache dtype 由两个不相干旋钮控制（bool `use_fp4_indexer_cache` vs enum `indexer_kv_dtype`），两个都传时**静默按 bool 解析**、enum 被忽略。→ **8.44**
- **vLLM PR #52529**：batched chat completions 的 `echo=true` 不检查 `conversation[-1]` 角色（默认 `add_generation_prompt=true` 下最后一条是 **user prompt**）→ 把用户 prompt 拼在助手回答前面返回。→ **10.21**
- **vLLM PR #52552**：`RoutedExperts.build_expert_params_mapping` 的 LoRA base-layer 前缀拼接顺序反了（`experts.routed_experts.base_layer.w13_`，实际参数树是 `experts.base_layer.routed_...`，#41184 重构引入）→ MoE 专家层 LoRA 参数映射断裂。→ **12.19**
- **tt-metal #53285**：三个入口用 `TT_ASSERT` 做输入校验，而 `TT_ASSERT` 在 Release 构建被编译掉 → `polyval` 空 vector 越界读 `coeffs[0]`、`complex_tensor` shape 校验完全消失（fail-open 校验）。→ **1.78**
- **tt-metal #53279**：`ttnn.lerp` 用无分支公式，`lerp(a,b,1.0)` 不返回 b（33.8% 采样对不满足端点语义；torch.lerp 分支保端点）。→ **1.79**
- **tt-metal #53278**：`ttnn.lcm` 接受 int32 全域，内核却假设 ~15-bit 操作数（`gcd_body<15>` + u16 最终乘法），无任何强制。→ **1.80**
- **Triton PR #11319**：kernel cache 条目是文件组（IR/cubin/metadata+manifest），单文件 `os.replace` 原子但**组发布不原子**——多进程/多主机共享 cache dir 并发编译同一 key 时读者可装载**混组**（A 的 cubin + B 的 metadata）。→ **1.81**
- **Triton PR #11309（fixes #10987）**：`LayoutPropagation::resolveConflicts` 的 fallback 取 `*info.encodings.begin()`（任意第一个候选）——可选出 lane base 全零的 **lane-degenerate layout**，warp 每个 lane 持有同一元素（miscompile）。→ **1.82**
- **Triton PR #11289（revert #10906 + 正确修复）**：RDNA 上同一 workgroup 的 wave 可跑在不同 CU（不同 L0 cache），scalar `tl.atomic_*` 携带内存序时的 barrier 地址空间处理不当 → miscompile；#10906 的修法在 RDNA 上过度同步损失性能。→ **1.83**
- **arxiv 2606.19526 SPINE**：GDB 驱动的故障注入 profiler，把累积权重位翻转直接注到 edge CPU 目标二进制上，产出 per-layer 故障谱（免重训/免改码）；指导选择性加固。→ **3.11**（BFA/SDC 方法论条目）

## 二、已知类别的新实例

- **vLLM PR #52532（fixes #52525）**：`moe_align_block_size()` 只保证按 expert 分组、不保证 expert 内 token 顺序 → Marlin MoE 多 threadblock split-K 路径选择不同物理块位置、走不同浮点归约树 → BF16 小差异（归约序非确定性）。`[已知类别]`，Cov `var:fp_reduction_order_nondeterminism` → 合并为 **4.8**
- **vLLM PR #52543**：`ParallelLMHead` 是唯一不从 `LinearBase` 派生的可量化投影，缺几何属性 → compressed-tensors 量化 `lm_head` 加载即 `AttributeError`。与 12.18（#52434/#52451）同一接口缺口 → **作为 12.18 补充来源引用，不占新编号**
- **TRT-LLM PR #17774（fixes #17296）**：`DeepSeekR1Parser.finish()` 把恰为 `<think>`/`</think>` 的残留 buffer 当标记丢弃——但只有「正在等待的那个 tag」才是分隔符 → EOS 时推理内容整块丢失。`[已知类别]`（reasoning parser 丢弃，同 10.20 族）→ **10.22**
- **TRT-LLM PR #17748（fixes #17740）**：Qwen3 的 `eot_token`（`\n</tool_call>`）以 `tool_call_separator`（`\n`）开头 → 工具调用结束后保留的 eot 被误当新 tool call 起点，parser 滞留 JSON 分支、**后续助手内容被 buffer 吞掉不下发**。`[已知类别]`（流式解析器状态机）→ **10.23**
- **arxiv**：SCOUT（2608.11034）、LLM-PRISM（2604.10390）、Decision-Level Hijacking（2607.25227）、UEP（2607.19623）、When Latent Agents Lie（2606.28958）、Runtime Observability（2608.05863）——均已在编目（§3.1 / 3.7 / 3.8 / 8.42 等），无新论文需增录。

## 三、已扫描但未编目（低价值 / fail-fast / 非正确性）

- vLLM #52535（TP 整除错误信息改进）、#52537（bench 数据集功能）、#52538（ROCm graph capture 流绑定，性能/正确性修复但无用户侧故障报告）、#52522（batch-invariant spec decode 功能 PR）
- TRT-LLM #17781（SkipSoftmax CuTeDSL 功能）、#17778（MoE Router Replay RFC）、#17767（空结果误判 HTTP 400，fail-visible）、#17749（disagg 取消保护默认关闭，liveness 类，记录为相关）、#17760/#17758/#17756/#17752（CI/工程类）
- tt-metal #53303/#53304（conv L1 估算 TT_FATAL，fail-closed crash）、#53281（Blackhole ethernet core 未复位 → 下一负载 init 超时，设备状态泄漏/liveness）、#53264/#53266（测试硬编码 8-bank，非运行时缺陷）、#53288（perf 门禁）、#53302（测试拓扑）
- Triton #11314（指针 bitcast axis-info 修正，防御性）、#11234（while 条件 i1 规范化，IR 有效性）、#11281/#11279/#11266/#11254（性能/工具链）
- transformers #47917（TimmWrapper SDPA 派发 ValueError，fail-fast）、#47914（bnb+PEFT warmup AttributeError，fail-fast crash）、#47925/#47893（import 错误，fail-fast）、#47879（Gemma4 视频处理，已关闭）
- 全局 `silent data corruption` 搜索（642 命中）无 GPU/LLM 相关新内容

## 四、API 调用记录

| # | 调用 | 结果 |
|---|---|---|
| 1 | GitHub search `repo:vllm-project/vllm label:bug created:>=2026-08-09` | ✅ 12 条 |
| 2 | GitHub search `repo:sgl-project/sglang correctness OR "wrong result" OR NaN` | ❌ 首次查询串 OR/引号组合返回空（total_count null），改写后重试 ✅ 117 条命中，浏览前 10 |
| 3 | GitHub search `repo:NVIDIA/TensorRT-LLM correctness OR "wrong output" OR NaN` | ✅ 12 条 |
| 4 | GitHub search `repo:triton-lang/triton miscompile OR "wrong result" OR numerical OR NaN` | ✅ 12 条 |
| 5 | GitHub search `repo:tenstorrent/tt-metal numerical OR PCC OR correctness` | ✅ 12 条 |
| 6 | GitHub search `repo:vllm-project/vllm "KV cache" OR quantization OR "bit flip"` | ✅ 12 条 |
| 7 | GitHub search `repo:huggingface/transformers label:bug` | ✅ 5 条 |
| 8 | GitHub search `"silent data corruption" GPU OR LLM` | ✅ 642 命中，无相关 |
| 9-16 | issue/PR 详情 API（#52531/#52512/#52551/#52540/#52553/#52516/#52527/#52530） | ✅ 各 1 次（core API，10 次/小时独立限额，未触顶） |
| 17-20 | arxiv API ×4（SDC+LLM / bit flip / KV cache corruption / LLM correctness） | ✅（注意 301 重定向需 `curl -sL`） |
| - | 事故记录：首轮 8 次 search 调用消耗于 jq 过滤器语法错误（`if` 缺 `end`），等待限流窗口重置（reset=1786921414）后用 jq 程序文件重跑成功；无 HTTP 403 拦截 |
