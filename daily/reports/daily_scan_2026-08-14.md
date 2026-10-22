# LLM 推理故障每日扫描 — 2026-08-14

- 扫描日期：2026-08-14（UTC）
- 滚动窗口：GitHub `created:>2026-08-12`；arxiv 最近提交
- 主文档基线：FAULT_MASTER_REFERENCE.zh.md，共 516 条（合并前）
- 搜索目标：vllm-project/vllm · sgl-project/sglang · NVIDIA/TensorRT-LLM · triton-lang/triton · tenstorrent/tt-metal · huggingface/transformers + arxiv

---

## 一、GitHub Issues / PR 发现

### 将合并进主文档的新条目（去重后确认不在编目中）

1. **[Bug] Legacy-compat reciprocal Newton seed 1.442695 is 9.6x less accurate than ~4/3** — tt-metal #53106（open，"Bad Outputs"，LLK/SFPU）
   https://github.com/tenstorrent/tt-metal/issues/53106
   `_reciprocal_compat_` 沿用 Grayskull 硬连线常量 1.442695（ln2 倒数）作 Newton 迭代种子，而源码注释自己写着理想值是 ~1.33。Blackhole P300 全域扫描 max rel err 1.4752e-03（9.4 bits，0.38 bf16 ULP），9.6x 劣于理想种子。历史兼容数值常量成为系统性精度损耗源。
   → `[已知类别]`（硬件/SDC 类）· 新子签名：**legacy-compat 数值常量精度妥协**。归入第 1 类 → 1.75。

2. **NemotronH MTP speculator: 0% draft acceptance on Intel XPU — V1 execution path bypasses standard sampling chain** — vLLM #52107（closed）
   https://github.com/vllm-project/vllm/issues/52107
   V1 speculator 执行路径完全绕过标准采样/验证链（`_process_engine_step`、`post_step`、`sample`、`RejectionSampler.__call__`）：draft 提出（spec 计数器递增）但从未验证 → 0% draft acceptance，静默失去全部推测解码收益。Intel Arc Pro B70（Xe2）。
   → `[已知类别]`（分布式/多后端 + 解码/采样交界）· 新子签名：**执行路径绕过标准采样链**。归入第 5 类（平台特定执行路径）→ 5.14。

3. **[Bug][MooncakeStoreConnector]: Preemption frees GPU blocks while async store jobs are still reading them, storing KV that does not belong to the key** — vLLM #52360（open）+ 修复 PR #52372
   https://github.com/vllm-project/vllm/issues/52360
   Preemption 释放 GPU block 时异步 store 任务仍在读取，Mooncake store ledger 按 key 存入**不属于该 key 的 KV 数据** → 分离式 KV 传输层的跨请求缓存键污染。修复：store job 引用 GPU block + ledger 按 save_seq 键控。
   → `[已知类别]`（服务栈/缓存）· 新子签名：**异步 store 生命周期与 preemption 竞态 → KV 键-值错配**。归入第 6 类 → 6.93。**监控相关性高**（silent，键值错配型 SDC）。

4. **[TRTLLM-15404][fix] VisualGen: refuse static quant recipes against unquantized checkpoints (silent weight corruption)** — TRT-LLM PR #17699（open）
   https://github.com/NVIDIA/TensorRT-LLM/pull/17699
   对未量化 checkpoint 应用静态 FP8/FP8_BLOCK_SCALES/NVFP4 量化 recipe 时缺少 scale tensor → **静默权重损坏**，无报错。修复加 fail-fast 校验（ValueError 指明缺失模块）。
   → `[已知类别]`（服务栈/量化加载）· 新子签名：**静态量化 recipe × 未量化 checkpoint 的静默权重损坏**。归入第 6 类 → 6.94。

5. **Fix swa tombstone blocking prefix match** — SGLang PR #34870（open，root fix of #34823 crash）
   https://github.com/sgl-project/sglang/pull/34870
   hybrid SWA + EAGLE 推测解码 + 统一 radix cache 下，out-of-window 释放留下的 SWA tombstone 使非 HiCache 部署的 rematch 停在 root（返回 0 indices）→ `cache_protected_len` 永不前进 → pool 记账漂移，最终每 TP rank "pool memory leak detected" crash。
   → `[已知类别]`（服务栈/缓存）· 新子签名：**tombstone 拒绝前缀匹配 → 保护长度停滞 → pool 记账崩溃**。归入第 6 类 → 6.95。

6. **[Bug]: NVFP4 MoE monolithic backend produces semantically corrupt output for non-gated ReLU² MoE (Nemotron-3-Nano) on SM100** — vLLM #52308（open）+ 修复 PR #52316（closed：对 non-gated MoE 禁用 monolithic 后端）
   https://github.com/vllm-project/vllm/issues/52308
   B200（SM100）默认自动选择 `FLASHINFER_TRTLLM` monolithic NVFP4 MoE 后端（trtllm_fp4_block_scale_moe），对 non-gated ReLU² MoE 输出**语法流利但语义损坏**，greedy 100% 确定性复现；同 checkpoint 同 vLLM 在 SM120 正确，`--moe-backend=flashinfer_cutlass` 在同一 B200 上完全修复。A/B/A 实验隔离到后端本身。
   → `[已知类别]`（量化）· 新子签名：**monolithic FP4 MoE 内核对 non-gated 激活的静默语义损坏（架构代差）**。归入第 7 类 → 7.35。

7. **[Bug][ROCm/gfx942]: GLM-5.2-FP8 — first request after GPU idle emits garbage; piecewise CUDA graph cold replay corrupts the request's own prefill** — vLLM #52150（open）
   https://github.com/vllm-project/vllm/issues/52150
   空闲 ≥20–50s 后的第一个请求从首 token 起输出 garbage（token-0 噪声），且**该请求永不自愈**（max_tokens=300 全程噪声）——其自身 prefill 写入的 KV/indexer cache 已被污染，后续 decode 持续消费中毒上下文；0.5s 后的下一个请求完全正常。workaround：`cudagraph_mode=FULL_DECODE_ONLY`。8×MI325X，MoE AITER blockscale-FP8。
   → `[已知类别]`（KV 缓存/注意力 + CUDA-graph 交互）· 新子签名：**piecewise CUDA graph 冷重放损坏请求自身 prefill（idle 触发、单请求持久中毒）**。归入第 8 类 → 8.38。**监控相关性极高**（idle→首请求、自愈失败模式新颖）。

8. **[Bugfix] Apply attention sinks in the Transformers backend** — vLLM PR #52156（open）
   https://github.com/vllm-project/vllm/pull/52156
   带 learnable attention sink 的模型（GPT-OSS、GraniteSWA、GraniteMoeSWA、DeepSeek V4、MiMo-V2-Flash）把 per-head sink 以 `s_aux` 传给 attention 接口，但 vLLM Transformers backend 未把 sink 折入 softmax 分母 → **sink token 被静默丢弃、输出错误**（#48270 因此加了专用类绕过）。修复在 backend 层统一处理，使所有 sink 模型正确。
   → `[已知类别]`（KV 缓存/注意力机制）· 新子签名：**attention sink 参数在通用 backend 被静默丢弃**。归入第 8 类 → 8.39。

9. **[Bug]: Client stop strings match inside the reasoning segment for think-in-prompt models, truncating CoT and yielding content=None** — vLLM #52393（open）
   https://github.com/vllm-project/vllm/issues/52393
   think-in-prompt 模板下生成从 `<think>` 段内开始，V1 detokenizer 对**整个输出流**求值客户端 `stop` 串——CoT 自然复述的 stop 串（如 lm-eval 的 `"Question:"`）在段内命中：`</think>` 永不出现、reasoning parser 无法闭合、返回 `content: None` + `finish_reason: stop`。GSM8K n=50 实测 8–15/50 被斩首（分数 0.66–0.84），抑制后 1/50（0.98）。
   → `[已知类别]`（解码/采样）· 新子签名：**stop 串作用域跨越推理段边界 → CoT 截断**。post-forward 结构性盲点。归入第 11 类 → 11.32。

10. **[NPU] Fix NEXTN/MTP conv-window layout corrupting verify output** — SGLang PR #34794（open）
    https://github.com/sgl-project/sglang/pull/34794
    Ascend 910B3 + NEXTN/MTP 推测解码 + w8a8 量化 checkpoint：`causal_conv1d` 恒从 `conv_states` **FRONT** 读 base conv window，而 verify 路径按 END 布局写入（`num_accepted_tokens=1` → offset=0）→ GDN（linear-attention/mamba）verify 输出被损坏，**部分 prompt 持久性答错**（如 `2+2` 输出 garbage），MTP 关闭即正确。
    → `[已知类别]`（解码/采样）· 新子签名：**conv-window FRONT/END 布局错配损坏推测 verify**。归入第 11 类 → 11.33。

11. **[None][fix] Refresh group greedy state per iteration** — TRT-LLM PR #17710（open，fixes Hecate 复现）
    https://github.com/NVIDIA/TensorRT-LLM/pull/17710
    `group_all_greedy_sample` 决策在迭代间不复位：上一迭代的同步 group greedy 决策泄漏到当前 batch 的 local greedy/non-greedy flag 推导，后续 sampling-buffer 人口继续复用过期决策。DSpark rejection sampling + attention DP + LM-head TP 触发。与 #17671（non-greedy CUDA-graph capture override 生命周期）为姊妹缺陷。
    → `[已知类别]`（权重/状态生命周期 + 采样）· 新子签名：**跨迭代 group 采样决策状态泄漏**。归入第 12 类 → 12.16。

### 检查后跳过的条目（已编目或不属正确性故障）

- vLLM #52109（DeepSeek-V4-Flash silent retrieval corruption）— 已编目（8.x，2026-08-13 合并）。
- vLLM #52339（FlashMLA sparse prefill phase1.cuh crash @161K）— kernel crash（非法内存访问），非静默正确性损坏；属崩溃类，暂不编目。
- vLLM #52330（DP startup CPU 线程超订 15x 慢加载）— 性能问题，非正确性。
- vLLM #52262（Intel XPU spec decode 报错）、#52280（LaTeX 转义格式）、#52306（多模态分块，修复 PR 已在途，机制为既有 chunked-prefill 类）、#52371（SGLang→vLLM 测试钉子）、#52164（DSA 性能路径）— 崩溃/格式/性能类或已知类别弱实例。
- SGLang #34815（PP8 TTFT 30s 地板）— 性能，非正确性。
- SGLang #34847（HiSparse copy_cache grid sizing，prefetch replay 路径）、#34824（torch.compile 符号计数修复，编译期失败非静默错值）、#34765 / #34760 — 已编目（2026-08-13）。
- TRT-LLM #17714（NcclEP 算法硬编码，通信功能缺陷非数值正确性）、#17724（Qwen3.8 VLM generic 路径，功能完整性）。
- triton-lang 11305/11306/11309 — 编译器优化重构，无用户可见数值错误报告。
- tt-metal #53153（GPT-OSS RingJointSDPA 统一 chunked sliding 路径）— 修复型重构，issue 正文未给出可引用的静默错值证据，降级为观察项。

## 二、arxiv 发现

| 日期 | 标题 | 链接 | 处置 |
|---|---|---|---|
| 2026-08-11 | SCOUT: Symmetric Consensus Outlier Detection for Failure Localization in LLM Pre-Training | arxiv 2608.11034 | 作为 §3.1 SDC 条目的**补充来源**引用（训练侧重，与推理监控弱相关） |
| 2026-08-06 | Runtime Observability for Heterogeneous Attention Memory | arxiv 2608.05863 | 已编目（8.37） |
| 2026-06-27 | When Latent Agents Lie: KV-Cache Integrity in Multi-Agent LLM Collaboration | arxiv 2606.28958 | 观察：多智能体 KV 完整性（安全向），暂不编目 |
| 2026-07-30 | Hollow-LLM Attack（零知识验证中的权重替换） | arxiv 2607.28884 | 观察：ZK 验证向，暂不编目 |
| 2026-07-27 | Integrity of peer-to-peer distributed LLM inference under malicious nodes | arxiv 2607.19490 | 观察：P2P 完整性，暂不编目 |

其余（Moment-KV、Addressable Memory 等）为压缩/架构方法论文，非故障案例。

## 三、重点新发现摘要

1. **vLLM #52150（GLM-5.2-FP8 ROCm CUDA graph 冷重放）** — 最值得关注的静默故障：idle ≥20–50s → 首请求从自身 prefill 起被污染且永不自愈，温度 0、短 prompt 即可复现。对生产监控是"空闲后首请求"这一时间结构信号的理想用例。
2. **vLLM #52360（Mooncake 异步 store 竞态）** — 分离式 KV 传输层的键-值错配：store ledger 存入不属于该 key 的 KV，是 K§5 locus 在服务栈侧的新变体。
3. **vLLM #52308（NVFP4 non-gated ReLU² MoE）** — 量化内核"流利但语义损坏"+ 硬件代差（SM100 坏 / SM120 好）+ 后端 A/B/A 可完全隔离，是量化类条目的教科书案例。
4. **tt-metal #53106（Newton 种子 legacy-compat）** — 硬件内核数值正确性的新形态：非崩溃、非 NaN，而是全域系统性 9.6x 精度损耗，源头是历史兼容常量。

## 四、API 调用记录

- GitHub Search API（未认证，8 次，间隔 ≥8s）：vllm wrong output / vllm KV+quant / vllm NaN / sglang correctness / hf transformers inference wrong（0 命中）/ trtllm incorrect / triton numerical / tt-metal pcc。无 403。
- GitHub Issues/PRs API（核心 API，独立限额）：vllm #52360/#52150/#52306/#52393/#52107/#52308/#52156；sglang #34870/#34794/#34824；TRT-LLM PR #17699/#17710；tt-metal #53106/PR #53153。
- arxiv API（`curl -sL`，4 次查询）：SDC+LLM / bit flip+LLM inference（0 新命中）/ KV cache+corruption / LLM inference+correctness。

## 五、合并计划（本次执行）

新增 11 条：第 1 类 +1（1.75）、第 5 类 +1（5.14）、第 6 类 +3（6.93–6.95）、第 7 类 +1（7.35）、第 8 类 +2（8.38–8.39）、第 11 类 +2（11.32–11.33）、第 12 类 +1（12.16）；合计 516 → **527**。SCOUT（2608.11034）作为 §3.1 补充来源引用（不占新编号）。
