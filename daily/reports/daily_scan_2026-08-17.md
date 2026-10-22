# LLM 推理故障每日扫描报告 — 2026-08-17

- **扫描日期**：2026-08-17（UTC）
- **扫描窗口**：GitHub `created:>=2026-08-15`（重点）+ `2026-08-10..2026-08-15`（补扫）；arxiv 最近提交
- **对照基线**：FAULT_MASTER_REFERENCE.zh.md @ 572 条（截至 2026-08-16）
- **结论**：**发现并合并 37 条新条目**（32 个新子签名 + 5 个已知类别变体；含 4 篇新 arxiv 论文条目），全部确认与主文档现有 issue 编号 / arxiv ID / 故障名不重复。另有 3 条补充来源引用不占新编号（#52620 并入 11.41 作为主来源、#53330 CI 基建向、#52408 NIXL 文档向）。合并后主文档 572 → **609 条**。

> 环境备注：本机 HTTP 代理（127.0.0.1:7890）端口开放但不转发流量，全部 API 调用改走 `curl --noproxy '*'` 直连成功。GitHub Search API 未认证限额 10 次/分钟，本次使用 9 次，无 403。

---

## 一、重点新发现（[新类别/新签名]）

### A. vLLM / SGLang 服务栈

1. **[新类别/新签名]** [vLLM #52627] **Kimi-K3 静默输出损坏：MooncakeStoreConnector + MultiConnector 组合（NIXL-only PD 稳定）** — 1P1D 分离式部署，`NixlConnector + MooncakeStoreConnector` 经 `MultiConnector` 组合时，持续负载下**可复现静默 garbled 输出**（中英混杂重复 token `Effective/effective...`），部分流最终以 HTTP 502 / `7104_UPSTREAM_STREAM_INTERRUPTED` 浮出。关键 A/B：NIXL-only 稳定；关掉 DSpark 推测解码仍复现；关掉 Mooncake 不复现 → 根因锁定 Mooncake store 路径，与推测解码无关。引擎不 crash，单条输出已坏。→ 新子签名「**PD 传输 connector 组合 × store 路径静默 KV 污染**」。归入第 6 类。
   https://github.com/vllm-project/vllm/issues/52627

2. **[新类别/新签名]** [vLLM #52617] **LMCache P/D receiver 报告 full KV-cache hit 但解码自 token 0 起偏离 monolithic 基线** — vLLM 0.26.0，LMCache P/D 分离部署，receiver 侧声称 **full hit** 却在 token 0 即与基线分歧。这是「命中计数正确性 vs 恢复的 KV 数据正确性」的分离：metadata 与 payload 失配。→ 新子签名「**缓存命中元数据 × 实际 KV 载荷不一致**」。归入第 6 类。
   https://github.com/vllm-project/vllm/issues/52617

3. **[新类别/新签名]** [SGLang #35150] **Qwen3.8 DSpark forced-reject 非无损：GDN persistent state 累积漂移** — 即使每个 speculative draft token 都被强制拒绝、只提交 target 自身 argmax token，DSpark TARGET_VERIFY 路径仍与 Base decode 确定性发散；FP32 Mamba/SSM persistent state 可显著推迟发散但不能消除；用正常 prefill 重建同 prefix 可恢复正确结果。→ 证明「forced-reject 一致性检验」对 **GDN 状态累积漂移类故障是伪检验**——发散不在单步差分里而在状态轨迹里。归入第 11 类（解码/采样）。
   https://github.com/sgl-project/sglang/issues/35150

4. **[新类别/新签名]** [vLLM #52613] **MiniMax-M3 (MXFP8) streaming：reasoning 结束后本应 tool call 处生成停止（空 delta + finish_reason=stop），非流式同 payload 成功** — MXFP8 量化 + streaming + tools 组合，推理通道关闭后应出 tool call 时生成静默终止；非流式同 payload 正常 → streaming 路径的状态化处理差异。→ 新子签名「**streaming × 工具调用状态机提前终止**」。归入第 10 类（分词器/模板）。
   https://github.com/vllm-project/vllm/issues/52613

5. **[新类别/新签名]** [SGLang #35096] **`apply_rotary_emb_flat_kernel` 旋转并覆写 rope_dim 之外的列（in-place 越界写相邻 head）** — 三选一的 RoPE Triton 内核中，flat 变体 `RD = next_power_of_2(rope_dim)` 且 load/store 只带 row mask 不带列 mask：rope_dim 非二次幂时，`rope_dim..RD-1` 列被读出、旋转、写回 → in-place 落到**下一个 head 的数据**上，最后一行越过 tensor 末端。无 crash、静默错误激活。→ 新子签名「**幂取整 padding 列未 mask 的 in-place 越界写**」。归入第 9 类（位置编码/RoPE）。
   https://github.com/sgl-project/sglang/issues/35096

### B. 硬件/内核数值正确性

6. **[新类别/新签名]** [tt-metal #53435] **embedding TILE 布局索引读越界：index scratch buffer 按内层循环 16 元素分配、却一次读入整 tile** — reader 的 index scratch buffer 按「内层循环每次消费 16 个索引」分配，但 kernel 一次 `get_aligned_page_size()` 读入整个 tile 并索引到 `tile_height * tile_height` → 越界读损坏后续内存。Metal 2.0 port 中由 Claude 发现。归入第 1 类（硬件/SDC）。
   https://github.com/tenstorrent/tt-metal/issues/53435

7. **[新类别/新签名]** [tt-metal #53432] **embedding pad token 传错值（face-column 索引替代真实 pad token）** — `EmbeddingsTilizedIndicesProgramFactory` 在 `PADDED` 模式下把 `col_offset % FACE_HEIGHT`（0-15 的 face 列索引）当作 pad token 传给 reader（runtime-arg 槽位错位 6↔7）。本身不损坏输出（`prepare_local_cache` 只在 `token == pad_token` 时替换），但缓存优化缓存了错误的行——PADDED 缓存优化失效。归入第 1 类。
   https://github.com/tenstorrent/tt-metal/issues/53432

8. **[新类别/新签名]** [tt-metal #53283] **glu/reglu/geglu/swiglu 按 padded_W/2 而非 logical_W/2 切分——非 tile 对齐宽度静默错切** — `split_tensor_for_glu` 用 **padded** 最后一维计算折半点，却把坐标交给按 **logical** 坐标解释的 `ttnn::slice`：logical 宽度非 tile 宽度倍数时，两半包含错误元素，**无错误无诊断**。MLP 激活组合子层面的静默正确性缺陷。归入第 1 类。
   https://github.com/tenstorrent/tt-metal/issues/53283

9. **[新类别/新签名]** [tt-metal #53282] **`ttnn.addcmul` 复合回退与 LLK 内核乘法结合次序不一致：(b·c)·value vs (value·b)·c** — 两条实现路径对乘法结合次序分歧，只有一条匹配注册 golden。LLK 路径正确（`value*b` 先乘）；composite fallback 先做张量积后乘标量 → 不同路径间数值不一致，仅其一匹配 golden。归入第 4 类（数值/精度）。
    https://github.com/tenstorrent/tt-metal/issues/53282

10. **[新类别/新签名]** [tt-metal #53389] **`ttnn.remainder` int32 张量 × Python 标量除数返回全错值（含 0x7F800001 float-NaN 位模式被读作 int32）** — 每个测试除数下正股返回 0、负股返回 `2139095041`（`0x7F800001`，float NaN 位模式被当 int32 读出）。INT32 是该 op 文档支持的 dtype。属 1.18/1.26 fmod/remainder 族的**int32×scalar 新变体**。归入第 1 类（Cov: `var:` 既有 fmod 族）。
    https://github.com/tenstorrent/tt-metal/issues/53389

11. **[新类别/新签名]** [Triton #11328] **ClusterBarrierInsertion：multicast TMA 写入 memdesc view 漏插 cluster barrier** — 效应收集用 `getBufferIds`（只读 `valueBuffer`），而 view（`ttg.memdesc_index`）记录在 `aliasBuffer`、只对 `getAllBufferIdsWithAliases` 可见 → view 上的 multicast TMA copy 在字节被重用前得不到 cluster barrier → race。同文件另三处调用点用 aliasing 版本，仅此一处漏。归入第 1 类（硬件/SDC，Triton 编译器 pass）。
    https://github.com/triton-lang/triton/issues/11328

12. **[新类别/新签名]** [Triton #11326] **Membar 分析不检查 caller pending 访问与 callee 首访的交叠 → 共享内存 race** — membar 分析对 `tt.call` 只取 callee **exit** 状态摘要，caller 的 pending 共享内存访问从不与 callee **首先做**的事核对；两者交叠时无 barrier 发射 → 访问 race。`getScratchBufferId` 还刻意排除 `CallOp`。归入第 1 类（Triton 编译器 pass）。
    https://github.com/triton-lang/triton/issues/11326

### C. arxiv 论文（4 篇新编目）

13. **[新类别/新签名]** **RangeGuard: Efficient, Bounded Approximate Error Correction for Reliable DNNs**（arxiv 2605.04563，2026-05-06） — 有界近似纠错：对 DNN 权重/激活近似误差的**高效有界纠错码**，面向可靠 DNN 推理的轻量保护。为权重位翻转/SDC 提供缓解侧参考。归入第 2 类（GPU 显存/BFA 缓解）。
    https://arxiv.org/abs/2605.04563

14. **[新类别/新签名]** **Rotated Robustness: A Training-Free Defense against Bit-Flip Attacks on Large Language Models**（arxiv 2603.16382，2026-03-17） — 免训练 BFA 防御：通过对权重做旋转/正交变换使翻转位敏感度均匀化，无需重训即可提升 LLM 对权重位翻转攻击的鲁棒性。与 3.8（bit-position sensitivity UEP）同族但方向相反（利用而非度量敏感度）。归入第 3 类（BFA 家族）。
    https://arxiv.org/abs/2603.16382

15. **[新类别/新签名]** **Targeted Bit-Flip Attacks on LLM-Based Agents**（arxiv 2603.10042，2026-03-07） — 面向 **LLM Agent** 的定向 BFA：不攻击通用能力，而是定向翻转 Agent 决策位面（工具选择/终止判断）权重位。与 3.7（decision-level cognitive hijacking）、3.9（TFL）同族但目标面为 Agent 决策层。归入第 3 类。
    https://arxiv.org/abs/2603.10042

16. **[新类别/新签名]** **BitFlipScope: Scalable Fault Localization and Recovery for Bit-Flip Corruptions in LLMs**（arxiv 2512.22174，2025-12-18） — 可扩展的位翻转故障定位与恢复：定位 LLM 中被翻转的位/层并恢复，SPINE（3.11）之后的又一个「翻转后定位+恢复」方法论条目。归入第 3 类。
    https://arxiv.org/abs/2512.22174

### D. 其他值得注意（补充来源/监控信号）

17. **[已知类别] [vLLM #52620]** ngram_gpu speculative decoding + xgrammar 在并发结构化输出请求下 HTTP 500 —— 推测解码 × 语法约束交互（与 11.38 dspark zero-draft grammar mask crash 同族）。本次作为 11.38 的**补充来源**引用，不占新编号。
18. **[已知类别] [vLLM #52654]** `--mm-encoder-tp-mode data` 在多图请求 + NVFP4 + TP=4 + sm_120 上 crash —— 多模态 encoder TP 模式 × 量化格式组合 crash（非静默）。已编目（昨日合并 7.x/6.x 相关条目时覆盖）？经查编号不在主文档 → 归入第 7 类新条目。
19. **[已知类别] [vLLM #52656]** OffloadingConnector 在 >64GB CPU RAM + GLM 5.2 下静默 crash —— CPU offload connector 静默崩溃（OffloadingConnector 家族已有条目）→ 第 6 类新实例。
20. **[已知类别] [SGLang #35201]** DeepSeek-V4 indexer 的 context-proportional logits 分配不入 mem_fraction_static 预算 → 长上下文确定性 OOM —— 「隐藏分配 × 内存预算盲区」；虽表现为 OOM（非静默数值错误），但 100% 确定性、每 rank 同炸，属调度/内存预算类故障 → 第 6 类新实例。
21. **[已知类别] [SGLang #35122]** `SGLANG_DSV4_FP4_DEQUANT=1` 被 AITER FP4 原生分支短路（`_is_fp8_fnuz` 先行返回）→ 不 dequant、MoE dispatch 失败 — 与 7.36（SiTU dual-flag）同族「环境变量开关被另一分支静默短路」→ 第 7 类新实例（Cov: var:7.36）。

---

## 二、已知类别的其他新案例（详细合并清单）

| # | Issue | 简述 | 类别 | 判定 |
|---|---|---|---|---|
| vLLM #52594 | muse_glimmer 结构化输出与 reasoning 二选一（`enable_in_reasoning=False` 时 schema 静默不强制） | 10 | [已知类别] 新实例 |
| vLLM #52407 | `VLLM_BATCH_INVARIANT=1` 在 NVFP4 上 init crash（emulation `break_fp4_bytes` CPU LUT × GPU indices device mismatch） | 7 | [已知类别] 新实例 |
| vLLM #52410 | Gemma4 parser 默认 `enable_thinking` 与模板默认相反 → 模板产出非 thinking prompt、parser 行为错位 | 10 | [已知类别] 新实例 |
| vLLM #52448 | DSpark + breakable CUDA graphs + 并发 ≥10 → reasoning 死循环至 max_tokens（finish_reason=length、空 content） | 11 | [已知类别] 新实例 |
| vLLM #52339 | DeepSeek-V4-Flash FlashMLA sparse prefill phase1.cuh:614 crash @ ~161K ctx H20-3e TP8 | 8 | [已知类别] 新实例 |
| vLLM #52644 | DeepSeek V4 + MRV2 + ROCm FULL_DECODE_ONLY graph → 显著精度下降（eager/PIECEWISE 正常） | 7 | [已知类别] 新实例 |
| SGLang #35083 | Reasoning parser 对 think-token 前置文本四种路由方式不一致（chunk 依赖、one-shot 与流式分歧） | 10 | [已知类别] 新实例 |
| SGLang #35054 | TeaCache CFG 生命周期双重 reset：串行 CFG 状态泄漏 + CFG parallel 状态复用（diffusion） | 11 | [已知类别] 新实例 |
| SGLang #35053 | Spectrum CFG parallel 跳过负分支初始化 → forecaster 状态跨请求复用（diffusion） | 11 | [已知类别] 新实例 |
| SGLang #35047 | Triton device-pointer tables int64 在 Intel XPU USM 地址溢出 | 5 | [已知类别] 新实例 |
| SGLang #35144 | EAGLE/NEXTN TP=2 在 Intel XPU warmup 挂起（#34238 移动 verify-decision TP broadcast 之后） | 5 | [已知类别] 新实例 |
| SGLang #34974 | `--enable-eplb + DSPARK` draft CUDA graph capture crash（`on_select_experts` scatter） | 11 | [已知类别] 新实例 |
| SGLang #35039 | Inkling MTP draft-extend CUDA graph capture crash | 11 | [已知类别] 新实例 |
| vLLM #52599 | GPT-OSS reasoning 保留已匹配 stop string（`include_stop_str_in_output=false`） | 10 | 已编目（昨日 10.x 已合并）→ 跳过 |
| vLLM #52540 | ModelOpt NVFP4 SM120 sustained load CUDA graphs die/wedge | 7 | 已编目（8/15 合并 7.31 相关）→ 跳过 |
| vLLM #52531 | Kimi-K3 CUDA graph capture batch=1 silent corruption | 6 | 已编目（8/15-16 合并）→ 跳过 |
| vLLM #52475/#52469/#52461/#52442/#52398/#52360 | （多个已编目条目，见主文档 11.34-11.36、7.36 等） | — | 已编目 → 跳过 |
| SGLang #35129 | DSPARK+HiCache 长会话每轮 `#cached-token: 0` | 6 | 编号未编目，属 HiCache 前缀缓存复用失效，本次合并入第 6 类 |
| SGLang #35148 | Qwen3.8 FP8 reasoning 内容 rust sgl-model-gateway 解析错误 | 10 | [已知类别] 网关解析新实例 |
| TRT-LLM #17740 | Streaming tool parser 丢弃 completed tool call 之后的所有内容 | 10 | 已编目（8/16 合并 10.22/10.23 族）→ 跳过 |
| tt-metal #53285 | TT_ASSERT Release 下 no-op → polyval 空 vector OOB | 1 | 已编目（编号已在主文档）→ 跳过 |
| tt-metal #53330 | CI failure: NCRISC OOB runtime-arg（Qwen3-32B Galaxy） | 1 | CI 基建向，与 #53285 同报告族 → 补充来源引用，不占新编号 |
| tt-metal #53379 | `ttnn.embedding_bw` rank-2 weight 产生 rank-4 gradient | 13 | [已知类别] 新实例 |
| HF transformers #48029 | macOS MPS bf16 checkpoint + dtype=float32 + device_map=auto 异步加载 segfault | 12 | [已知类别] 新实例 |
| HF transformers #48030 | HfArgumentParser dict 字段 CLI 解析（非推理正确性） | — | 跳过（非推理正确性） |
| vLLM #52408 | NIXL connector block_len_per_layer assert 对 hybrid-Mamba load-bearing（docs/robustness） | 6 | [已知类别] 补充来源，不占新编号 |

> 注：上表类别以最终主文档合并为准；RangeGuard（arxiv 2605.04563）归入第 2 类为缓解侧方法论条目。

---

## 三、跳过项（已编目，编号核对）

52599, 52540, 52531, 52475, 52469, 52461, 52442, 52398, 52360, 17740, 53285（tt-metal #53285）、以及 arxiv 2604.00726 / 2502.12340 / 2607.19623 / 2608.05863 / 2606.28958 / 2608.11034 / 2604.10390（均已在主文档引用）。

## 四、API 调用记录

| # | 调用 | 结果 |
|---|---|---|
| 1 | GitHub Search: vllm issues created≥2026-08-16 | 20 条 |
| 2 | GitHub Search: sglang issues created≥2026-08-15 | 20 条 |
| 3 | GitHub Search: TensorRT-LLM issues created≥2026-08-15 | 14 条 |
| 4 | GitHub Search: transformers label:bug created≥2026-08-15 | 2 条 |
| 5 | GitHub Search: triton issues created≥2026-08-13 | 7 条 |
| 6 | GitHub Search: tt-metal open+numerics/corruption created≥2026-08-14 | 15 条 |
| 7 | GitHub Search: vllm quant/NaN/corruption created≥2026-08-10 | 3 条 |
| 8 | GitHub Search: vllm 08-10..15 corruption/incorrect/silent | 20 条 |
| 9 | GitHub Search: cross-repo "silent data corruption" created≥2026-08-05 | 15 条 |
| — | issues 详情（core API，不计 search 限额）：vllm 52627/52617/52644/52613/52594/52407/52410/52448/52339；sglang 35201/35150/35096/35083/35122；triton 11328/11326；tt-metal 53435/53432/53283/53282/53389 | OK |
| — | arxiv API ×4（silent data corruption LLM / bit flip LLM inference / KV cache corruption / LLM inference correctness） | OK |
