# LLM 推理故障每日扫描报告 — 2026-08-18

- **扫描日期**：2026-08-18（UTC）
- **扫描窗口**：GitHub `created:>=2026-08-17`（重点）+ `2026-08-14..17`（补扫）；arxiv 最近提交
- **对照基线**：FAULT_MASTER_REFERENCE.zh.md @ 609 条（截至 2026-08-17）
- **结论**：**发现并合并 30 条新条目**（23 个新子签名 + 7 个已知类别变体；含 1 篇 arxiv 方法论条目 GRIEF），全部确认与主文档现有 issue 编号 / arxiv ID / 故障名不重复。另有约 10 组补充来源引用并入条目正文不占新编号（flashinfer #3125/#3238、SGLang #29453/#29522/#30512/#30645、vLLM #44297/#52258/#52471、PR #36787、#34941 作为 #34947 dup）。合并后主文档 609 → **639 条**。

> 环境备注：与昨日相同，本机 HTTP 代理不转发流量，全部 API 调用走 `curl --noproxy '*'` 直连成功。GitHub Search API 未认证限额 10 次/分钟，本次使用 11 次（含 1 次 tt-metal label 语法失败重试），无 403。

---

## 一、重点新发现（[新类别/新签名]）

### A. 服务栈 / KV 缓存（最重）

1. **[新类别/新签名]** [SGLang #34947] **DSA sparse-MLA prefill：单次 extend >65535 token 时 gridDim.z 溢出 → 0 个 attention kernel 被发射，输出张量保持未初始化 sentinel 原样返回，无任何异常** — onset 精确到 1 token（65535 正常 / 65536 全静默）；profiler 证明 0/num_layers 个 attention kernel 运行；`bench_one_batch` 构造性命中并报出**虚假地快**的长上下文 prefill 数字。前作修复链全部未落地（flashinfer #3125 closed 无 fix、sglang #29453 closed 无代码修复、#29522 guard PR closed 未合并、#3238 结构性修复 sglang 未采纳）。→ 新子签名「**kernel 启动参数溢出的静默 zero-launch**」：不是算错而是**根本没算**。归入第 8 类（8.46）。
   https://github.com/sgl-project/sglang/issues/34947

2. **[新类别/新签名]** [SGLang #34959] **DSPARK 推测解码静默篡改 prompt 原样 identifier（kimi-k3→kimi-kw），损坏率随 verify 窗口宽度单调上升（block 6/temp 0 达 1.03%，关推测 0/4561）** — 温度 0 也发生；同配置仅关推测解码零损坏。与昨日 35150（GDN 状态漂移）互补：本条是宽窗口 verify 的 token 级替换且无关状态累积。→ 新子签名「**推测解码 verify 路径非无损（grounded token 替换）**」。归入第 11 类（11.46）。
   https://github.com/sgl-project/sglang/issues/34959

3. **[新类别/新签名]** [vLLM #52693] **streaming input 续写轮采样到 stop token（零输出轮）→ 请求以 RUNNING 状态重入队，scheduler raise 杀死 EngineCore** — 普通多轮 chat 流量即可触发（temperature>0），一个用户的采样偶然性炸掉整引擎与全部并发请求；5 实例观测 9 次引擎死亡。→ 新子签名「**采样结果 × 调度器状态机非法转移**」。归入第 6 类（6.119）。
   https://github.com/vllm-project/vllm/issues/52693

4. **[新类别/新签名]** [SGLang #35358] **compressed-tensors 量化 lm_head：`get_quant_method` 不分发 `ParallelLMHead` → checkpoint 的 weight_packed 无人读、新建参数未初始化 → 每请求输出全量 token 0（`!!!!!!...`），服务正常启动零报错** — vLLM 源码有该分支，SGLang adaptation 掉了。→ 新子签名「**量化方法分发矩阵缺层 → 权重静默未初始化**」。归入第 7 类（7.44）。
   https://github.com/sgl-project/sglang/issues/35358

5. **[新类别/新签名]** [vLLM #52741] **OpenAI `tools[].function.strict`（服务端解码指令）被原样渲染进 model-visible `<tools>` 块，实际改变 tool-call 行为** — 24-tool 负载下 `strict:false` 使 0/5 成功调用、4/5 把畸形 tool call 泄进 content；同 token 量中性填充不触发（语义效应非长度噪声）。→ 新子签名「**服务端控制字段 × prompt 渲染边界穿透**」。归入第 10 类（10.28）。
   https://github.com/vllm-project/vllm/issues/52741

6. **[新类别/新签名]** [vLLM #52667] **PaliGemma 图像 embedding 残留反向缩放：Gemma sqrt(H) 归一化重构只移植了文本半边，图像特征以 1/sqrt(H)（ColPali 下 1/45.25）幅度进模型** — transformers 在 0e7cb4e3 原子改两半，vLLM 拆半移植。→ 新子签名「**上游原子变更被拆半移植 × 变体模型补偿逻辑残留**」。归入第 13 类（13.30）。
   https://github.com/vllm-project/vllm/issues/52667

7. **[新类别/新签名]** [TRT-LLM #17911] **sm_120 e4m3 kernel 枚举漏 head_size 256 → fused FP8-KV context FMHA 静默缺失，回退路径要 ~626GB workspace——表现为"prefix caching 不工作"且零诊断** — 构建期 kernel 生成矩阵的形状空洞；补上 256 后 prefix 复用 probe 4096/4181 blocks。→ 新子签名「**按硬件枚举生成的 kernel 矩阵形状空洞 × 静默回退**」。归入第 8 类（8.47）。
   https://github.com/NVIDIA/TensorRT-LLM/issues/17911

### B. 硬件 / 内核

8. **[新类别/新签名]** [tt-metal #53537] **program cache 命中路径从不重验 CB 区域 vs 实时 L1 占用 → 两次运行间 L1 布局变化使 CB 与 resident tensor 静默重叠，L1 数据被覆写** — 校验只在冷路径（miss）执行，热路径只 patch 地址。→ 新子签名「**缓存复用跳过首次分配时的布局校验**」。归入第 1 类（1.90）。
   https://github.com/tenstorrent/tt-metal/issues/53537

9. **[新类别/新签名]** [tt-metal #53538] **`get_max_l1_space()` 只查 bank 0 分配器，看不见 HYBRID per-core 分配 → CB 预算高估；验证器却逐核遍历——同一约束两个查询实现不一致** — untilize/transpose/matmul 等全走该 helper，与 1.90 构成上下游双缺口。归入第 1 类（1.91）。
   https://github.com/tenstorrent/tt-metal/issues/53538

10. **[新类别/新签名]** [tt-metal #53566] **Quasar 背靠背 sub-device 测试：第二个测试必触发 Watcher L1 corruption——`noc_init()` 缺 `overlay_cmd_buff_init`，per-core NOC 状态跨程序复用未重置**。→ 新子签名「**硬件上下文跨程序复用未重初始化**」。归入第 1 类（1.92）。
    https://github.com/tenstorrent/tt-metal/issues/53566

### C. 其他新签名

11. **[新类别/新签名]** [vLLM #52756] **MTP drafter 超限 skip 路径返回全零 token ID 而非零条 draft → scheduler 当真 draft 去 verify（0% 接受），且 batch 级 max_seq_len 门控让一条超限序列停掉全批推测** — 静默性能回退 + 指标误导。归入第 11 类（11.47）。
12. **[新类别/新签名]** [SGLang #35270] **radix `evictable_size()` 按节点引用计数不去重共享页 → 默认开启的 idle 不变量检查比错单位杀 scheduler（"空闲即死"，全部请求完成后死）** — 探针证明无泄漏无 UAF，纯计数单位错配。归入第 6 类（6.115）。
13. **[新类别/新签名]** [vLLM #52808] **PD Multi Tier：producer 已持有块时反而不供给 → consumer 白等 30s 超时后本地重算；prefix cache 越好失败越多（效果与设计意图倒置）**。归入第 6 类（6.112）。
14. **[新类别/新签名]** [SGLang #35345] **(3,T) mRoPE 位置传入只接受 1-D 的 fused QK RMSNorm+RoPE kernel → 只读 temporal 行，H/W 位置从未施加——视觉 token Q/K 静默错误（Qwen3.6/3.8-27B 多模态全版本）**。归入第 9 类（9.12，Cov: var:9.10）。
15. **[新类别/新签名]** [arxiv 2605.11202] **GRIEF：LLM 服务引擎灰盒模糊测试（定时多请求 trace 为一等输入），vLLM+SGLang 发现 15 漏洞、10 确认、2 CVE——KV 隔离失效/跨请求污染/noisy-neighbor DoS** — 服务层共享状态故障的可发现性方法论。归入第 6 类（6.121，方法论条目）。

---

## 二、已知类别的其他新案例（详细合并清单）

| # | Issue | 简述 | 类别 | 判定 |
|---|---|---|---|---|
| SGLang #35257 | DSA top-k v2 tie bin 溢出保留 atomicAdd 先到的 2048 个 → exact tie-break 在任意子集上跑，静默返回非真 top-k（1M ctx 实测 42 行中 2–3 行错） | 4 | [已知类别] var:4.6 persistent_topk（4.10） |
| vLLM #52778 | CustomAllreduce 准入跨节点 TP=2 组（MNNVL GB200）→ cudaIpcOpenMemHandle abort；IPC handle 仅节点内有效 | 5 | [已知类别] 新实例（5.20） |
| vLLM #52735 | OffloadingConnector + MTP/EAGLE 推测解码：只存永不回供（XPU + hybrid GDN 报告） | 6 | [已知类别] OffloadingConnector 家族新变体（6.113） |
| vLLM #52802 | vllm-bnb-plugin 与 in-tree bitsandbytes 重复注册 apply_bnb_4bit → 启动崩 | 6 | [已知类别] 新实例（6.114） |
| SGLang #35229 | /v1/messages Anthropic 适配层丢弃 PD bootstrap 字段 → decode worker 拒绝 | 6 | [已知类别] 新实例（6.116） |
| TRT-LLM #17926 | Nemotron-3-Ultra 1M max_seq_len 且 avg_seq_len 未设 → KvCacheManagerV2 ~256k prompt segfault（设 9000 则 800k 通过） | 6 | [已知类别] 新实例（6.117） |
| TRT-LLM #17908 | BlockRadixTree.clear() 遇 childless RootBlock 死循环 | 6 | [已知类别] 新实例（6.118） |
| vLLM #52792 | Kimi-K3 GDN + ngram 推测：spec 路径 strided state_indices 被 kernel 拒 + 混合 spec/non-spec 并发批 GPU fault（ROCm/aiter） | 6 | [已知类别] var:6.83（6.120） |
| SGLang #35324 | DSPARK + DSV4-Flash 生产 ~每周一次 sampling NaN device assert 杀全部 TP rank | 11 | [已知类别] var:K8.14（11.49） |
| SGLang #35388 | deepep FP8 MoE GB300：激活 scale UE8M0、权重 scale 原生 fp32——warmup 用全 1 scale 恰好合法，**只炸首个真实请求** | 7 | [已知类别] var:6.46 UE8M0 族（7.43） |
| SGLang #35354 | DSpark base-logits 绕过 quant_method.apply 直接 matmul NVFP4 打包 lm_head（[248320,2560]）→ 形状崩 | 7 | [已知类别] 新实例（7.45） |
| HF transformers #48057 | Qwen2-VL get_rope_index 在 batch_size=1 单图即 shape mismatch（transformers 5.15.0 回归） | 9 | [已知类别] 新实例（9.13） |
| TRT-LLM #17916 | DeepSeekR1Parser 类忽略 per-request enable_thinking=false → 整回复进 reasoning_content、content 空（子类 NemotronV3 已修，父类未回移） | 10 | [已知类别] var:10.26（10.29） |
| TRT-LLM #17917 | tool parser 匹配 marker 却提取 0 调用时返回 200 无诊断 | 10 | [已知类别] 10.22/10.23 族第三面（10.30） |
| vLLM #52767 | MTP + structural tag：仍向已终止 grammar matcher 喂 token（#44297 残留；当前负载仅告警，tool call 结果正确） | 11 | [已知类别] var:11.37 族（11.48） |

## 三、跳过项（已编目 / 非正确性 / 报告记录不合并）

- **已编目（编号核对后跳过）**：vLLM #52627/#52617/#52620/#52599/#52594/#52540/#52531/#52475 等（8/15–8/17 已合并）；SGLang #35150/#35129/#35096/#35083/#35144/#35122/#35201；TRT-LLM #17740；HF #48029/#48030；tt-metal #53435/#53432/#53283/#53282/#53389/#53379；Triton #11328/#11326；arxiv 2608.11034/2604.10390/2607.25227/2607.19623/2607.28884/2608.05863/2607.19490。
- **报告记录但不合并**（crash 带清晰错误 / 纯性能 / 功能缺口 / 诊断 QoL，正确性价值低或已由同族条目代表）：
  - vLLM #52835（mm item 超 mm_processor_cache_gb 启动 ValueError）、#52834（--mm-processor-kwargs 无法按模态 scope、flat size 泄入 image 预算）、#52768（Gemma4 v0.27.1 启动 AmbiguousGlobalPerLayerAttributeError，已 closed）、#52761（DSV4-Flash-0731 不支持 PP，显式报错）、#52728（--cudagraph-metrics 无统计）、#52773（Hybrid+KV offloading 丢弃 divergent prefix 命中——性能类，6.111 已代表该族）
  - SGLang #35385（WSL2 误选 CUDA IPC transport crash）、#35348（ROCm torch.compile launcher 缺参）、#35291（MLA fp8 KV 每层反量化全池——性能）、#35295（流式 ASR 词修订半词后缀——delta diff 算法）、#35252（MoE tuner 写运行时不读的 config 文件——静默 no-op 性能）、#35241（PrefillDelayer DP Attention 反馈环 prefill 停摆）、#35189（NIXL/UCX segfault 未修——旧编号复核）
  - TRT-LLM #17912/#17909/#17910/#17913/#17914/#17915（serve 内部生命周期 crash 族：sentinel/EngineDeadError 复利/SimplePool 孤儿/CudaStream 泄漏/SO_REUSEADDR）、#17882（MAX_UTILIZATION resume RequestError）、#17917 已合并（10.30）
  - Triton #11352（tl.cat 误导性报错）、#11344（desc.scatter sm_12x ptxas 晚崩而非编译期报错）、#11320（fp8 matmul 半速率）、#11321/#11322/#11327（AMD 编译器 crash）
  - tt-metal #53512（Batch-1 Galaxy main 上推理坏，信息不足）、#53545（GLM 5.2 perf 回归）、#53528/#53532（ND hang）、#53559（DE firmware 读不存在寄存器 hang）
  - arxiv：Belayer（RL 训练容错，非推理）、Conformal Cascade（精度保证方法）、Don't Regenerate Debug（算子修复 agent）——均非故障模式编目对象

## 四、API 调用记录

| # | 调用 | 结果 |
|---|---|---|
| 1 | GitHub Search: vllm issues created≥2026-08-17 | 67 条 |
| 2 | GitHub Search: sglang issues created≥2026-08-16 | 41 条 |
| 3 | GitHub Search: TensorRT-LLM issues created≥2026-08-16 | 17 条 |
| 4 | GitHub Search: transformers label:bug created≥2026-08-16 | 3 条 |
| 5 | GitHub Search: triton issues created≥2026-08-14 | 8 条 |
| 6 | GitHub Search: tt-metal label:numerics created≥2026-08-14（label 语法失败，空响应） | 0 条 |
| 7 | GitHub Search: vllm corruption/incorrect/silent/wrong created≥2026-08-14 | 57 条 |
| 8 | GitHub Search: sglang correctness/wrong/NaN/corruption created≥2026-08-14 | 17 条 |
| 9 | GitHub Search: cross-repo "silent data corruption" created≥2026-08-11 | 639 条（噪声高，仅 tt-metal #53584 相关度低） |
| 10 | GitHub Search: cross-repo "KV cache" corruption OR "bit flip" created≥2026-08-11 | 5601 条（噪声高，捞得 vLLM #52838 修复 PR 线索） |
| 11 | GitHub Search: tt-metal issues created≥2026-08-15（#6 重试，无 label） | 214 条 |
| — | issues 详情（core API，不计 search 限额）：vllm 52833/52808/52802/52792/52767/52756/52741/52735/52778/52667/52693/52834；sglang 35388/35358/35345/35257/35270/35252/35229/35324/34959/34947/35354/35295/35242；trtllm 17926/17916/17911/17908；tt-metal 53566/53538/53537；hf 48057 | OK |
| — | arxiv API ×4（silent data corruption LLM / bit flip LLM inference / KV cache corruption / LLM inference correctness） | OK，1 篇新编目（GRIEF 2605.11202） |

> 跨仓库关键词搜索（#9/#10）信噪比极低（匹配大量无关仓库），明日可考虑加 `llm OR vllm OR inference` 限定或改为按目标仓库逐一定向搜索。
