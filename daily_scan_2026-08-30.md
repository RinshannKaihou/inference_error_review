# LLM 推理故障每日扫描报告 — 2026-08-30

- **扫描日期**：2026-08-30（滚动窗口 created:>=2026-08-29，与昨日报告重叠部分已按主文档 issue 编号查重剔除）
- **搜索源**：GitHub Issues Search API（vllm / sglang / transformers / TensorRT-LLM / triton / tt-metal）+ arxiv API
- **基线**：FAULT_MASTER_REFERENCE.zh.md 合并前 789 条（第 0 层 17 + 第 1 层 658 + 第 2 层 114）
- **结果**：新发现并合并 **16 条**（15 个新子签名 + 1 个已知类别变体）；昨日已编目条目（#54359/#54317/#36943/#36941/#36938/#36935/#54813/#54798/#54833）今日再现均查重跳过。arxiv 命中均为已编目论文（SCOUT、LLM-PRISM、BitFlipScope、Rotated Robustness 等）；两篇未编目论文（Nexus 2608.20397、Moment-KV 2605.29873）为优化/压缩方法论文、非故障案例，不编目。

---

## 一、新编目条目（16 条，已合并进主文档）

### 第 1 类 · 硬件 / SDC（数值内核原语）（+4）

1. **tt-metal #54882** — ttnn.tanh_bw: fp32 path uses a BF16-scoped asymptotic tail, dropping the (1+e^-2x)^-2 factor — up to 65,346 fp32 ULP — https://github.com/tenstorrent/tt-metal/issues/54882
   `[新类别/新签名]`（→ 1.125）fp32 路径复用 BF16 精度目标的渐近尾式 `4·exp(−2|x|)`，丢掉分母 `(1+exp(−2|x|))⁻²`：|x|≥3 偏大最多 65,346 fp32 ULP（0.5%），|x|=3.0 处**阶跃不连续**；BF16 路径 1 ULP → 穷举 BF16 校验全绿。近似式的 dtype 限定符未随路径传播。
2. **tt-metal #54855** — ttnn.digamma FLOAT32 is ~4,500 ULP off for x > 102 — https://github.com/tenstorrent/tt-metal/issues/54855
   `[新类别/新签名]`（→ 1.126）x>102 走 Bernoulli 渐近式且复用按低精度目标实现的共享 log 体：全域中位 4,543 / 最大 9,080 ULP；独立 60 位计算验证 golden bit-exact（测的是 kernel）。与 1.125 同日同构造模式（BF16 目标近似被 fp32 继承）；与 1.33/1.50 的 digamma 条目机制不同（域内欠精 vs 域外饱和常数）。
3. **triton #11519** — [FuseNestedLoops] matchPositiveTripCount accepts an llvm.assume without checking it dominates the loop, and miscompiles — https://github.com/triton-lang/triton/issues/11519
   `[新类别/新签名]`（→ 1.127）优化 pass 把函数内**任意位置**的 llvm.assume 当"内层循环至少执行一次"的全局事实，跳过零迭代版本分裂：硬件实测产出错误返回值 + 源程序从不执行的越界 store（Blackwell + L4、Triton 3.6/3.7.1；flatten=False 同参数全对）。证明与控制流脱钩的编译器 miscompile。
4. **vLLM #54308** — GDN / Qwen3-Next decode degenerates to a single repeated token on non-Blackwell GPUs with an fp32 SSM cache — https://github.com/vllm-project/vllm/issues/54308
   `[新类别/新签名]`（→ 1.128）`fused_sigmoid_gating_delta_rule_update` 读出 fp32 累加但输出张量按激活 dtype（fp16）分配——fp32 SSM cache 的**未归一化累加器**长上下文后超 fp16 的 65504 → inf → RMSNormGated 的 inf/inf=NaN → decode 永久重复单 token（`!!!!`）。Blackwell（fi/cutedsl 全 fp32 读出）不触发，仅 Triton 路径；修复 PR #54146 一行。

### 第 6 类 · 服务栈 / 缓存正确性（+7）

5. **vLLM #54458** — Hybrid mamba/attention page alignment inflates attention blocks to 7808 tokens — per-request KV footprint of ~23 giant pages collapses concurrency and prefix caching — https://github.com/vllm-project/vllm/issues/54458
   `[新类别/新签名]`（→ 6.168）attention 页字节对齐 ≥ mamba 页字节把 block size 抬到 7,808 token；~23 个 KV group 各需每请求一页 → **固定 ~23 巨页足迹（118k–150k token 当量）与 prompt 长度无关**：并发天花板 2-3 请求、35k 缓存 prompt 3 次交错 admission 后命中 23,808→0；收紧 max-model-len 无效。
6. **SGLang #37160** — LMCache integration ignores per-request cache_salt: external storage keys are token-only, breaking tenant isolation — https://github.com/sgl-project/sglang/issues/37160
   `[已知类别]`（→ 6.169，var:lmcache_cache_salt_isolation_missing）LOOKUP/STORE 都不转发作用域内现成的 cache_salt（StoreMetadata 无 salt 字段）——不同租户同 prompt 共享外部缓存：实测第二账户 1,198 ms vs 3,340 ms 重算基线 = 跨租户计时/计算侧信道。LMCache wire key 本有 salt 字段（vLLM 集成会填），仅 SGLang 路径丢弃。K9.10 家族第三实例。
7. **TensorRT-LLM #18407** — KVCacheV2Scheduler retains active PEFT adapters after KV suspension — https://github.com/NVIDIA/TensorRT-LLM/issues/18407
   `[新类别/新签名]`（→ 6.170）KV V2 挂起只释放 KV 页、不调 `mark_request_done(pause=True)`：被挂起请求的 adapter 保持 device-active，另一 adapter 的 `ensure_batch` 无法从满 cache 驱逐——**为缓解压力做的挂起反而制造新的资源死锁面**；恢复路径同样缺 `add_request_peft`。mocked 状态机 harness 断言双失败/补丁双通过。
8. **vLLM #54360** — Speculative decoding (mtp and dflash) silently disables prefix-cache hits for hybrid GDN models on nightly; worked on v0.24.0 — https://github.com/vllm-project/vllm/issues/54360
   `[新类别/新签名]`（→ 6.171）启用任一推测解码使 `prefix_cache_hits_total` 静默归 **0**（align-mode 启动日志正常但永不命中；18,219-token 二次发送 9.46s→0.64s/命中 17,248 vs spec 下命中 0）；v0.24.0 同机同模型正确共存 = 跨版本回归，无 crash 无告警，仅 metrics 可见。
9. **vLLM #54392** — PD-admitted Mamba request is spec-padded before prefill completes, then align split truncates the 8-token window to 5 — https://github.com/vllm-project/vllm/issues/54392
   `[新类别/新签名]`（→ 6.172）1+K=8 合成 padding 窗被 #51113 的 Mamba 边界裁剪到 5 个物理 token，但 `pad_spec_decode` 仍 true、仍发布 7 个推测占位——请求级契约（1 target + 7 drafts = 8 vs 物理 5）违背而**全局 batch shape 全程合法**；Ascend DSpark proposer 按错误窗口解释 → 引擎终止。合成 padding 未原子化穿过调度器限制器链。
10. **vLLM #54383** — First boot of a new `(max-num-batched-tokens, max-num-seqs)` shape is granted a ~5% smaller KV cache than subsequent identical boots — https://github.com/vllm-project/vllm/issues/54383
    `[新类别/新签名]`（→ 6.173）首启的 compile+CUDA-graph capture 分配先于容量探测且部分驻留 → 首启池小 ~5%（82,688 vs 88,096 / 82,256 vs 86,528 token 双组实测），差异随 compile cache 持久化固化：**部署容量取决于启动历史**；严格 drain 排除残留显存。容量探测的非确定性。
11. **SGLang #37111** — Qwen3.8-Flash-Next QSA + NEXTN decode graph silently corrupts output on GB10 TP2 — https://github.com/sgl-project/sglang/issues/37111
    `[新类别/新签名]`（→ 6.174）decode CUDA graph 开启 vs eager 受控 A/B：graph 路径 1,024-token essay 退化为重复标点、~25K-token 四 marker 检索失败——HTTP 200 + 吞吐更高，健康检查与吞吐基准原理上不可见；eager 对照两 gate 全过。与 #37052（loud asserts）为独立报告。

### 第 8 类 · KV 缓存 / 注意力（+1）

12. **vLLM #54434** — hy_v4 (Hy4-preview): degenerate output with FLASHINFER_MLA_SPARSE_SM120 — the only available sparse backend on SM120 — https://github.com/vllm-project/vllm/issues/54434
    `[新类别/新签名]`（→ 8.58）SM120 唯一 sparse-MLA 后端上任意 prompt（含 5-token）temp=0 退化（`" of of of…"`）；系统性排除（MoE 后端 ×3 byte-identical、MXFP8 kernel ×2、模型侧核对、5-token 排除 scoring）后嫌疑集中 fp8_ds_mla KV SM120 数值与 gated DSA/IndexCache 处理（hy_v4 是该后端**首个 gated-MLA 消费者**）；后端拒绝 bf16 KV 使 fp8 数值无法 dtype A/B 排除。根因开放。

### 第 9 类 · 位置编码 / RoPE（+1）

13. **SGLang #37097** — Pretokenized image requests can crash GLM MRoPE with a stale retokenized mask — https://github.com/sgl-project/sglang/issues/37097
    `[新类别/新签名]`（→ 9.19）`input_ids` 用客户端精确 ID（#26555 恢复）、`attention_mask` 用 HF processor 对 `decode(ids)` **重分词**结果——decode→re-tokenize 对非规范序列不保长（两个换行 token 合并为一），MRoPE 以 N+1 索引 N 长 mask 抛 IndexError；两次各自合理的改动之间缺失所有权更新，长多轮 RL 最易命中。

### 第 10 类 · 分词器 / 对话模板（+1）

14. **vLLM #54337** — Assistant `content=null` + `tool_calls` renders literal "None" into chat template context — degenerate outputs in agent workloads — https://github.com/vllm-project/vllm/issues/54337
    `[新类别/新签名]`（→ 10.43）OpenAI agent 规范的 `content: null` 直达 `apply_chat_template`，Jinja `{{ content }}` 字符串化为 "None" 注入上下文；每 tool-call 轮一个 None，8-20 轮后推理坍缩（8-gram 重复率 1.000、零 content token）；`content=""` 即干净。协议层规范化修复（已 closed）。

### 第 11 类 · 解码 / 采样（+1）

15. **vLLM #54437** — structured output can sample unconstrained tokens when the draft hand-off goes stale (spec decode + async scheduling + PP) — https://github.com/vllm-project/vllm/issues/54437
    `[新类别/新签名]`（→ 11.57）`DraftTokensHandler` 单槽无身份、每步覆写——drafts 丢失后 `-1` 占位存活进 grammar_bitmask，真实采样行被写入 `_full_mask`（**全词表允许**）；worker 从 GPU 副本验证真实 drafts，acceptance 直走进无约束行：**mask 失败方向是 open**。token 碰巧合法 = 静默带毒继续；否则报错在下游 FSM（根因不可见）。main 上潜伏，PP-capable drafter 落地即活。

### 第 12 类 · 权重 / 适配器 / 检查点加载（+1）

16. **tt-metal #54884** — AttnRes weight loading ignores first_layer_idx: a pipeline rank silently loads another rank's queries — https://github.com/tenstorrent/tt-metal/issues/54884
    `[新类别/新签名]`（→ 12.26）rank 取 LOCAL 层数无偏移，而 cache/checkpoint 按**全局层索引**键控——请求的名字全存在（cache 含全 93 层）→ 完备性检查通过、shape 全同 → rank 拿**别人的 query** 正常起跑；单 rank `first_layer_idx=0` 使既有测试按构造全盲。同族：`mla_layer_ids[:num]` 习语把 rank-1 layer-15 对 layer-3 golden 评分 PCC 0.006（验证面误报）；伴生 #54892 使 hybrid 模型无法经 runner 做 PCC 校验。

## 二、重点新发现摘要

今日最值得注意的五条：

1. **11.57（#54437）grammar bitmask 的 fail-open**——结构化输出约束的失败方向是"全放开"而非"全禁止"：丢失的 draft 让 `-1` 占位在真实采样行写成全词表允许，worker 侧验证的数据源与 mask 数据源分歧。token 碰巧合法时静默投毒、否则报错出现在下游 FSM——两个形态都不指向根因。校验面的失败方向（fail-open vs fail-closed）值得作为通用评审维度。
2. **1.125/1.126（#54882/#54855）近似精度的 dtype 限定符未传播**——同日两条同构造：为 BF16 精度目标写的渐近尾式/共享 log 体被 fp32 路径继承，而 BF16 穷举校验全绿。"另一 dtype 校验通过"不是 fp32 正确性证据；步跃不连续（|x|=3）是可编程检测的签名。
3. **1.127（#11519）编译器 pass 的证明与控制流脱钩**——nondominating llvm.assume 被当全局事实，跳过安全版本分裂：硬件实测的错误值 + 越契约 store。对一切依赖 Triton 的推理栈是任意 kernel 的上游正确性风险，且 flatten 开关的 eager/compiled 对照是有效隔离手段。
4. **12.26（#54884）分片索引命名空间混淆 + 检查对错命名空间通过**——"每个请求的名字都存在"使完备性检查结构性失明；单 rank 部署 local=global 使测试套件按构造不可见；伴生的验证面缺陷（错误 golden 评分 PCC 0.006 误报、hybrid 无法验证）说明该混淆同时污染故障面与验证面。
5. **6.173（#54383）容量探测依赖启动历史**——首启 shape 永久小 ~5% 池直到 compile cache 失效：同配置部署的容量不可复现，是压测/容量规划噪声的隐性来源（无任何错误信号）。

## 三、已编目条目重复出现（未新增）

| 条目 | 对应主文档条目 |
|---|---|
| vLLM #54359 ROCm kpool 自我覆写（08-29） | 8.56（昨日编目） |
| vLLM #54317 GLM-5.3-Flash 三内核 IMA（08-29） | 6.159（昨日编目） |
| SGLang #36943 symm-mem NCCL 死锁（08-29） | 5.25（昨日编目） |
| SGLang #36941 GB10 长 prefill 静默 worker 死亡（08-29） | 6.167（昨日编目） |
| SGLang #36938 prefill logprob 游标错位（08-29） | 11.56（昨日编目） |
| SGLang #36935 branch reuse LRU 坍塌（08-29） | 6.166（昨日编目） |
| tt-metal #54813 KDA t_inv（08-29） | 1.118（昨日编目） |
| tt-metal #54798 MatmulMultiCoreReuse 丢块（08-29） | 1.119（昨日编目） |
| tt-metal #54833 SDPA 零填充 mask 列（08-29） | 8.57（昨日编目） |

## 四、跳过（未编目）及理由

- **vLLM #54415**（KV-offload shared-mmap 多节点 init hang，活性/功能缺口）、**#54413/#54414**（KV-offloading connector 拒绝 hybrid 小块 / recent-window 排除——Feature 请求）、**#54426/#54443/#54363**（RFC：QSA fp8 KV、在线 RL 权重版本、filesystem offload 完整性）、**#54340**（assert 崩溃优化讨论）、**#54354**（DP 组共享卡预算 feature）、**#54387/#54389**（Doc/Feature）。
- **vLLM #54455/#54425**（sampler warmup 覆盖缺口——首请求 JIT/延迟类，无正确性损坏证据）、**#54453/#54438/#54369**（Performance 类）、**#54459**（MSVC 构建移植）、**#54377**（Turing fp16 初始化 crash，loud）、**#54350/#54349**（XPU AWQ 选择器 crash，loud fail）、**#54385**（GB10 dual-rank Xid 31 warmup crash，loud 硬件故障类）、**#54451/#54376**（ROCm DFlash/FSE 不可用 = 功能缺口/回退，无静默损坏证据；#54376 记此备查——若证实静默禁用可类比 7.61）。
- **SGLang #37052**（GB10 invalid-probability asserts，loud，昨日已评估）、**#37059**（audio hang，活性，昨日已评估）、**#37089**（Marlin 线程配置 + FA-CuTe capture 启动 crash，loud bring-up）、**#37105**（DSA backend blockers after NameError，loud bring-up）、**#37022**（mooncake session 不存活，infra）、**#37031**（MLX stub 无 loader crash，loud）、**#37128**（Spec V2 OTel span 缺失，观测缺口记备查）、**#37115**（closed 修复）、**#37066**（slot budget +1 margin，调参 feature）、**#37150**（调参 feature）。
- **transformers #48419**（HfArgumentParser 静默损坏 tuple/set CLI 字段——工具链配置解析层、非推理正确性；边界案例记此备查）。
- **triton #11512**（NFS 共享 cache 并发 crash，昨日已评估跳过）。
- **tt-metal #54892**（build_kv_chunk_table 拒绝 hybrid = 验证路径缺口，fail-loud assert；作为 12.26 伴生证据在条目内引用）、**#54843**（同族 local/global 混淆，在 12.26 引用）、**#54820/#54828/#54827/#54826/#54825**（perf 清理类）、**#54869**（LayerAck FIFO 永不排空，runner hang）、**#54870**（吞吐测量虚高 3×）、**#54835/#54842/#54839/#54831/#54880/#54809/#54804**（昨日已评估或 CI 追踪类）。

## 五、arxiv 搜索结果（无新编目）

关键词：`silent data corruption LLM` / `bit flip LLM inference` / `KV cache corruption` / `GPU soft error inference`。命中（SCOUT 2608.11034、LLM-PRISM 2604.10390、Exploring SDC 2604.00726、Understanding SDC 2502.12340、Bit-Position Sensitivity 2607.19623、RangeGuard 2605.04563、Rotated Robustness 2603.16382、Targeted BFA on Agents 2603.10042、BitFlipScope 2512.22174、Addressable Memory 2608.07408、Runtime Observability 2608.05863、When Latent Agents Lie 2606.28958、指令级 GPU 软错误 FI 2601.19912）均已在先前扫描编目或引用。两篇未出现过的：**Nexus（2608.20397，depth-adaptive KV splicing + 工具路由系统）与 Moment-KV（2605.29873，decode 期 KV 压缩方法）均为方法/系统论文而非故障案例研究**，按编目标准不收录。

## 六、API 调用记录

| # | 查询 | 命中 |
|---|---|---|
| 1 | vllm all issues created:>=2026-08-29 | 39 |
| 2 | vllm + wrong/incorrect/corruption/NaN/silent | 15 |
| 3 | sglang all issues created:>=2026-08-29 | 17 |
| 4 | transformers + wrong/incorrect/silent/corruption/NaN | 1 |
| 5 | TensorRT-LLM all issues | 1 |
| 6 | triton all issues | 2 |
| 7 | tt-metal + correctness | 5 |
| 8 | tt-metal + PCC/NaN/mismatch | 22 |
| 9-12 | arxiv API ×4 组关键词 | 见第五节 |

（另有 19 次 issue 详情 REST 抓取用于机制核实；GitHub Search 未触发限流。）
