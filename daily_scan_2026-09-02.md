# LLM 推理故障每日扫描报告 — 2026-09-02

- **扫描日期**：2026-09-02（滚动窗口 created:>=2026-09-01，与昨日报告重叠部分已按主文档 issue 编号查重剔除）
- **搜索源**：GitHub Issues Search API（vllm / sglang / transformers / TensorRT-LLM / triton / tt-metal）+ arxiv API
- **基线**：FAULT_MASTER_REFERENCE.zh.md 合并前 844 条（第 0 层 17 + 第 1 层 713 + 第 2 层 114）
- **结果**：新发现并合并 **12 条**（8 个新子签名 + 4 个已知类别变体）；当日 arxiv 命中均为已编目论文（SCOUT、LLM-PRISM、TrainSDC、JITterFlip、Compression-Aware Abstention、Bit-Position Sensitivity/UEP、RangeGuard、Rotated Robustness、Flip-Agent、When Latent Agents Lie、Addressable Memory）或既往已判定不收录的方法论文（Moment-KV、Nexus、Runtime Observability），无新编目。
- **API 事件记录**：GitHub Search API 首批 8 组中 5 组因查询含 **>5 个 AND/OR 算子**被 `Validation Failed` 拒绝（非限流），拆分为两组 ≤4 算子的查询重发后全部成功（共 16 次 search 调用）；`api.github.com` 直连在 HTTP/2 下连续超时（3 次 exit=28），改用 `--http1.1` 后 15 次 issue detail 调用全部成功；arxiv 5 组查询全部成功；无 403 限流事件。

---

## 一、新编目条目（12 条，已合并进主文档）

### 第 1 类 · 硬件 / SDC（数值内核原语）（+9）

1. **tt-metal #55130** — ttnn.bias_gelu silently computes the approximate GELU while ttnn.gelu defaults to exact — https://github.com/tenstorrent/tt-metal/issues/55130
   `[新类别/新签名]`（→ 1.137）`bias_gelu` 与 `gelu(add, fast_and_approximate_mode=True)` **逐位相同**，对 torch float64 golden 最大误差 2.34e-02（`gelu(add)` 默认为 6.8e-07，差 ~34,000×）；x=−3 处返回恰好 0（真值 −0.00398）；经 `bind_binary_operation` 绑定故**不存在** `fast_and_approximate_mode` 参数、docstring 亦不提及近似——调用方无法请求精确形式。WH n150 + BH p150b 双机验证；报告当日即 closed(completed)。
2. **tt-metal #55265** — matmul(activation="exp"/"log"/"log2"/"log10"/"log1p") silently selects the approximate kernel while ttnn.exp defaults to accurate — and no string spelling reaches the accurate one — https://github.com/tenstorrent/tt-metal/issues/55265
   `[新类别/新签名]`（→ 1.138）unary 表中 EXP/LOG/LOG1P/LOG2/LOG10 五项硬编码 `approximate=true`，与表内其余全部算子"裸名=精确、`*_approx`=快速"的约定相反且**无任何 `*_approx` 兄弟拼写**；LLK 模板默认与 nanobind 默认均为 false（文档明示"Defaults to False"）。近似路径是 Schraudolph 3-分数位 bit-trick——百分比量级相对误差；经 `ttnn.matmul/linear` 与 `dit_rms_norm_unary_fused`（docstring 声称等价——对 exp 不成立）可达；log 近似路径还跳过归一化使 −0.0/subnormal 不返回 −inf。nightly 测试把字符串 case 重写为无参对象，**该表从未在设备上被测过**。
3. **tt-metal #55266** — matmul/linear with bias + SOFTPLUS returns exactly 0 below the threshold: the fused-bias epilogue drops activation_param2 (beta_reciprocal) — https://github.com/tenstorrent/tt-metal/issues/55266
   `[新类别/新签名]`（→ 1.139）融合 bias epilogue 不转发 `activation_param2`（beta_reciprocal），阈值以下 softplus 输出**恰为 0**；host 侧算对了参数、每个工厂都按名推送——只有 bias epilogue 丢掉。nightly 测试因输入分布（K=4096/8192 randn → std≈64/90 vs threshold=20，~40% 元素线性直通且幅值数百）PCC 被大幅元素主导：建模 PCC=0.994931，以 0.0009 之差**通过** 0.994 阈值——测试分布盲区放行内核 bug 的实证。
4. **tt-metal #55273** — Quasar: gtz/gez/ltz/lez return 1 for NaN — the comparison-to-zero predicate has no inf term, which Wormhole and Blackhole both carry — https://github.com/tenstorrent/tt-metal/issues/55273
   `[已知类别]`（→ 1.140，var:1.122）Quasar 比零谓词只看符号位 + "幅值≠0"，无 inf 项拒绝 NaN：`gtz(+NaN)`/`gez(+NaN)` 返 1.0，`ltz(−NaN)`/`lez(−NaN)` 对称返 1.0；WH/BH 版本载入 `BF1616_INF` 并加 `|v|<=inf` 链显式拒绝 NaN。issue 自述"与 #54857 同形"——**修复未跨 arch 传播**（1.122）在谓词内核的新实例。
5. **tt-metal #55269** — [LLK][BH] p_unpacr_nop defines three Wormhole-packed constants that silently mis-encode on Blackhole — https://github.com/tenstorrent/tt-metal/issues/55269
   `[新类别/新签名]`（→ 1.141）BH 的 `UNPACR_NOP` 是 9 独立操作数编码，WH 是单打包立即数；BH 头文件却携带三个 WH 打包值常量，`Unpack_Pop` 仅 2 位——溢出到邻位后每个常量编码出与其名字相反的控制（RESET_ALL_BANKS→clear-to-ONE；STALL_RESET_WR_RDY→clear both banks；SET_DVALID→**DVALID 永不发布**，显形为 hang）；`TTI_UNPACR_NOP` 从不调用 VALID 检查故无诊断。**潜伏非现役**（main/分支零引用），但 dest-reuse 工作正并排发布 WH+BH（#54683/#55107），移植替换即触发。另附 p_zeroacc 32B 模式编码未决问题。
6. **tt-metal #55276** — Quasar: two DFBs chained through a Tensix stage silently return wrong data when a DM endpoint outnumbers the Tensix side — https://github.com/tenstorrent/tt-metal/issues/55276
   `[新类别/新签名]`（→ 1.142）Metal 2.0 `DM→DFB→Tensix→DFB→DM` 管线在**任一 DFB 的 DM 线程数多于 Tensix 线程数**时静默交付错数据（违反 `R<=C and W<=C` 即坏）：45.83%–70.83% word 错、无 hang 无错误无降速；最大配置（6 DM + 4 Neo）bit-exact——非"多线程坏了"而是**线程数关系**触发。无 TTNN 参与的 gtest 身份 oracle 与 TTNN-vs-torch golden 双仪器在合法 (R,C,W) 全空间 31/31 一致确认。
7. **tt-metal #55227** — avg_pool2d global fast path: padded ROW_MAJOR input is silently wrong for specific shapes — https://github.com/tenstorrent/tt-metal/issues/55227
   `[新类别/新签名]`（→ 1.143）#54410 的守卫只校验 pad **数量**并推断 pad **位置**（假设尾部后缀），但 `pad_to_tile` 对 NHWC 的最后两维填充——pad 是**交错**的（每 H 行一份 W_pad）；pool2d 全局快路径把 (N,H,W) fold 成单归约轴后守卫的 roundup 支恒过（`H*ceil32(W)==ceil32(H*W)` 即通过），dense RM reader 按**逻辑行数**读页、跨过交错 pad：`[1,2,17,64]` 期望均值 1.0 实得 0.559（19/34），最差 44%；H≤64、W≤128 共 **328 个 (H,W) 组合**命中；两个不同输入产生 byte-identical 的 canonical tensor spec，reduce_op.cpp 层无法修复。
8. **tt-metal #55272** — conv1d/conv2d silently ignore conv_config.activation on the 1D-depthwise path — https://github.com/tenstorrent/tt-metal/issues/55272
   `[新类别/新签名]`（→ 1.144）`is_1d_depthwise_conv` 守卫排除 `has_bias`（专用内核不能融合 bias）却**漏掉 activation**——带 activation 无 bias 的 depthwise conv 落到零 activation 处理的 `compute_depthwise_conv1d` 内核（grep SFPU activation 符号 = 0）；host 仍计算 `pack_relu`、构建并附上 SFPU activation defines，但该内核的 compile-time args 到 index 12 截止——defines 全部死码；conv 目录无任何 `unary_chain`/`apply_activation` 兜底（不同于 ttnn.matmul 的回退设计）。无报错、activation 不在任何地方被应用。
9. **tt-metal #55180** — [TT-NOP Injector][BH] hang race on tiny_tiles_multi_op — https://github.com/tenstorrent/tt-metal/issues/55180
   `[已知类别]`（→ 1.145，var:1.99）#53394 TT-NOP 注入器时序扰动在 BH `tiny_tiles_multi_op` 上 `unpack:2 STALLWAIT`（`_llk_unpack_A_`）延迟 6 拍即 hang；手工在该 STALLWAIT 前插 6 个 `UNPACR_NOP` 同样挂起——1.99（NOP 注入器 LLK unpack→math handoff 竞态族）在 Blackhole 的新实例（既有族成员：WH eltwise_binary #53823、BH mul_reduce_scalar #53412、WH generalized_moe_gate #53418）。

### 第 4 类 · 数值 / 精度（+1）

10. **transformers #48459** — GPT-NeoX: hardcoded float32 softmax silently caps float64 inference at float32 resolution — https://github.com/huggingface/transformers/issues/48459
    `[已知类别]`（→ 4.16，var:4.2）eager attention 路径 `softmax(..., dtype=torch.float32)` 硬编码：fp16/bf16 下是合理升精度，fp32 下是 no-op，但模型显式转 **float64** 后每张 attention map 被**静默降回 fp32** 再转回 fp64——对 1e-9 量级中游 residual 扰动的响应被湮灭（Pythia-410M-deduped + forward hook 复现）；修复为 `dtype=query.dtype` 或 `promote_types(query.dtype, float32)`。「声明的计算精度被静默截断」在 dtype 上界方向的显形（4.2 tf32 截断的镜像：那条是乘法精度、本条是 softmax 精度且方向向下）。

### 第 5 类 · 分布式 / 多 GPU（+1）

11. **vLLM #55000** — [Bug]: DP8 P2P tier startup hangs while creating secondary NIXL/UCX agents — https://github.com/vllm-project/vllm/issues/55000
    `[已知类别]`（→ 5.29，var:5.28）fresh pod（8×H200、DP8+EP、GLM-5.2-FP8、MultiConnector=NixlConnector+OffloadingConnector/TieringOffloadingSpec 含 p2p 二级 tier）：8 个 primary NIXL agent 全部初始化成功，某个**二级** `NixlTransport` agent 停在 `_NixlAgent(...)` 内部（"Backend UCX was instantiated" 日志之前）；DP supervisor 等全部 rank → serving port 永不打开、pod 永不 Ready、8 GPU 占用闲置；无异常、无重启、无超时；发生在任何推理请求之前。nixl-cu13 1.3.2 / CUDA 13.0.2 / NCCL 2.30.7，nightly 同路径。与 5.28（FlashInfer MoE DP/EP 启动 wedge）同为 DP 拓扑启动期 silent wedge，本条在 UCX agent 初始化层。

### 第 6 类 · 服务栈 / 缓存正确性（+1）

12. **SGLang #37606** — Prefill breakable CUDA graph reuses weak-ref'd break inputs across buckets → wrong greedy output / IMA; e2e validation of #37448 — https://github.com/sgl-project/sglang/issues/37606
    `[新类别/新签名]`（→ 6.183）prefill breakable CUDA graph（BCG）把 graph-break inputs 存为**弱引用**（`captured_args = tuple(_weak_ref_if_tensor(a)…)`），注释假设共享 mempool 的 `use_count` 钉住存储——它只钉住 **pool 对象**而非 blocks；Python 引用释放后分配器把 blocks 重新雕刻给**后捕获的 bucket**（捕获序从大到小，later capture 恒存在），replay 早 bucket 即读到**另一 bucket 拥有的内存**：按布局或为静默错 greedy 输出、或为 CUDA IMA。默认路径全员命中（`default_prefill_backend()` 在 CUDA 返回 breakable，统一注意力 graph break 把每层 q/k/v/output 送过 eager 边界）。#30584 只修了同类 eager-output bridge（`captured_output`）；修复 PR **#37448**（恢复强引用 + 回归测试 `test_break_inputs_survive_later_capture_into_shared_pool`）已被本 issue 以 ~80 行 model-free 复现器端到端验证；弱引用优化引入于 #22218，风险当时已在 review thread 指出。

### 补充来源（不占新编号）

- **tt-metal PR #54410** — reduce_op.cpp 的 padded ROW_MAJOR 守卫（resnet50 个案的修复）——作为 1.143 的前置修复与"守卫推断位置"缺陷载体引用。
- **SGLang PR #37448** — break inputs 强引用修复——作为 6.183 的修复来源引用。
- **tt-metal #54683 / #55107 / PR #54731** — dest-reuse WH+BH 并排发布与 shared helper 计划——作为 1.141 潜伏风险的现实性证据引用。

---

## 二、查重后跳过/仅记录不编目的条目

- **vLLM #54843 / #54808 / #54785**（窗口内命中）— 已编目（→ 11.58 / 6.181 / 1.134，2026-09-01 合并），跳过。
- **SGLang #37475**（窗口内命中）— 已编目（→ 11.59），跳过。
- **tt-metal #55064 / #55032 / #55075 / #55014**（窗口内命中）— 已编目（→ 1.135 / 1.132 / 1.133 / 12.27），跳过。
- **triton #11519**（7 天窗口 r6 命中，created 2026-08-30）— 已编目（→ 1.127，2026-08-30 合并），跳过。
- **tt-metal #55144** — Hang at `(Kimi-K2.6-1T) chunked prefill padded accuracy 15k@5k traced`（CI 机器 hang，当日 closed）——仅 CI run 链接与机器名、无机制细节（无栈、无复现路径），信息不足以编目；若重开并补机制再编。
- **TensorRT-LLM / triton 当日窗口** — correctness 关键词命中 0 条（r3=0、r6 仅已编目的 #11519）。
- **transformers 窗口其余命中** — q4 仅 #48459 一条 correctness 相关（已编目为 4.16）。
- **arxiv 全部命中** — 已编目论文（见报告头）或不收录的方法论文（Moment-KV、Nexus 深度自适应 KV 拼接、Runtime Observability 异构注意力内存观测性——昨日已判定为观测性方法论文非故障刻画），无新增。

---

## 三、重点新发现摘要

1. **SGLang #37606（6.183）** — 当日最重要：**默认 prefill 路径全员命中**的 CUDA graph 跨 bucket 内存混叠——弱引用"优化"假设 mempool 引用计数钉住 block 存储（只钉 pool），早 bucket replay 读到晚 bucket 的内存，静默错 greedy 输出或 IMA 二选一。#22218 引入时 review thread 已指出风险、#30584 只修了输出侧——输入侧弱引用留存至今。model-free 复现器 + 修复 PR 端到端验证俱全。
2. **tt-metal #55276（1.142）** — 数据流线程**关系**触发（`R<=C and W<=C`）的静默错数据：非"多线程坏了"（最大线程配置 bit-exact），而是 DM 端点多于 Tensix 端点这一拓扑关系；双独立仪器在合法配置全空间 31/31 确认。
3. **tt-metal 近似-by-default 三连（1.137/1.138/1.144）** — 同日三例"API 契约静默违反"族：bias_gelu 无出口的近似 GELU（34,000× 误差）、字符串激活强制近似内核且无精确拼写（含被 docstring 背书的等价性破绽）、conv depthwise 静默丢 activation——调用方在无警告、无参数、无文档的情况下拿到不同的数学。
4. **tt-metal #55266（1.139）的测试分布盲区** — nightly PCC 门以 0.0009 之差放行"阈值下全零"的内核 bug：大 K randn 输入使 PCC 被线性直通的大元素主导。**阈值 + 分布共同构成假阴性**，对一切依赖 PCC 门的量化/内核验证有方法论警示（与 1.105 假阳性方向互补）。
5. **transformers #48459（4.16）** — fp64 推理被 eager 路径硬编码 fp32 softmax 静默封顶——精度截断族在"声明的 dtype 高于路径硬编码"方向的显形，对 fp64 参考实现/数值验证工作流是结构性陷阱。

---

## 四、API 调用记录

| # | 查询 | 命中 | 状态 |
|---|---|---|---|
| 1 | vllm correctness/wrong/corruption/silent/NaN/garbage in:title (created:>=2026-09-01) | 3 | OK |
| 2 | vllm quant/fp8/KV cache/speculative/bit flip/determinism/SDC/ECC in:title | — | Validation Failed（>5 AND/OR） |
| 3 | sglang correctness/wrong/garbage/NaN/corruption/silent/corrupted in:title | — | Validation Failed（>5 AND/OR） |
| 4 | transformers wrong output/NaN/rope/tokenizer/incorrect/corruption in:title+inference | 1 | OK |
| 5 | TensorRT-LLM correctness/wrong/NaN/corruption/output/crash/hang in:title | — | Validation Failed（>5 AND/OR） |
| 6 | triton wrong/incorrect/NaN/miscompile/corruption/silent in:title（7 天窗口） | 0 | OK |
| 7 | tt-metal numerical/pcc/wrong/NaN/corruption/incorrect/hang/silent/bitexact in:title | — | Validation Failed（>5 AND/OR） |
| 8 | TensorRT-LLM quantization/wrong result/incorrect result/bit flip/weight corruption/nondeterministic/numerical | — | Validation Failed（>5 AND/OR） |
| 9 | sglang correctness/wrong/garbage/NaN in:title | 2 | OK（拆分重发） |
| 10 | sglang corruption/silent/corrupted/incorrect in:title | 1 | OK（拆分重发） |
| 11 | TensorRT-LLM correctness/wrong/NaN/corruption in:title | 0 | OK（拆分重发） |
| 12 | tt-metal numerical/pcc/wrong/NaN in:title | 7 | OK（拆分重发） |
| 13 | tt-metal corruption/incorrect/hang/silent in:title | 11 | OK（拆分重发） |
| 14 | triton wrong/incorrect/NaN/miscompile in:title（created:>=2026-08-26） | 1 | OK（拆分重发；已编目 #11519） |
| 15 | TensorRT-LLM quantization/nondeterministic/numerical/wrong result in:title | 0 | OK（拆分重发） |
| 16 | vllm hang/deadlock/wedge/nondeterministic in:title | 2 | OK（拆分重发） |
| A1–A5 | arxiv: SDC+LLM / bit flip+LLM inference / KV cache+corruption / LLM inference+correctness / GPU soft error+learning | 5/5/5/5/0 | OK |

core REST issue 详情调用 15 次全部成功（HTTP/2 直连超时 3 次后改 `--http1.1`）；无 rate-limit 403 事件。
