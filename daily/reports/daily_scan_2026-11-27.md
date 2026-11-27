# 每日扫描日报 2026-11-27（推理库）

## 0. Git 前置检查与共享采集批次

- 工作目录 `/workspace/inference_error_review`；分支 `main`，上游 `origin/main`；`git status --porcelain -uall` 为空；`git fetch origin main` 成功；HEAD 与 `origin/main` 一致（`fa36fa55aee9c5ee5563fb8ebbe9da9254689080`，前一条为 `daily audit 2026-11-27: inference`）。通过，开始扫描。
- 批次：`/root/.hermes/profiles/third/cache/fault_scan_sources/batch-20261127T230012748398-nwckcsww`，`generated_at=2026-11-27T23:00:12Z`，`registry_hash=2f6422d0…8d94d`，22 个注册来源全部选中，`complete=true`，`failures=[]`，`cached=false`。
- 窗口：`2026-11-26T22:00:12Z → 2026-11-27T23:00:12Z`（约 25 小时；消费水位 `2026-11-26T23:00:12Z`，含 1 小时重叠）。
- 候选索引 `inference.jsonl` 49,110 条；`inference_delta.jsonl` **5,069 条**（issue 2,168 + PR 2,901）。其中窗口内**新建 issue 365 条**，其余 1,803 条为旧 issue 在本窗口被更新。

### 逐仓覆盖（窗口内 issue+PR）

| 平台 | 仓库 | fetched | issue/PR | 分页 | complete |
|---|---|---|---:|---|---|
| gitcode | Ascend/pytorch | 749 | 546/203 | issues 6/6、pulls 3/3 | 是 |
| gitcode | Ascend/op-plugin | 95 | 60/35 | 1/1、1/1 | 是 |
| gitcode | Ascend/torchair | 91 | 7/84 | 1/1、1/1 | 是 |
| gitcode | cann/ops-transformer | 938 | 453/485 | 5/5、5/5 | 是 |
| gitcode | cann/ops-nn | 1,532 | 759/773 | 8/8、8/8 | 是 |
| gitcode | cann/ops-math | 250 | 106/144 | 2/2、2/2 | 是 |
| gitcode | cann/ge | 121 | 38/83 | 1/1、1/1 | 是 |
| gitcode | cann/runtime | 131 | 24/107 | 1/1、2/2 | 是 |
| gitcode | cann/hccl | 162 | 31/131 | 1/1、2/2 | 是 |
| gitcode | cann/hcomm | 202 | 30/172 | 1/1、2/2 | 是 |
| gitcode | mindspore/mindspore | 14 | 3/11 | 1/1、1/1 | 是 |
| gitcode | mindspore/mindformers | 18 | 0/18 | 1/1、1/1 | 是 |
| gitcode | Ascend/msprobe | 15 | 7/8 | 1/1、1/1 | 是 |
| gitcode | Ascend/mssanitizer | 20 | 7/13 | 1/1、1/1 | 是 |
| gitcode | Ascend/MindSpeed 三仓 | 23/23/0 | 训练域 | 1/1 | 是 |
| gitcode | Ascend/MindIE-LLM | 3 | 0/3 | 1/1、1/1 | 是 |
| gitcode | Ascend/msmodelslim | 33 | 8/25 | 1/1、1/1 | 是 |
| github | vllm-project/vllm-ascend | 578 | issues 6 页 | 6/6 | 是 |
| github | sgl-project/sgl-kernel-npu | 26 | issues 1 页 | 1/1 | 是 |
| github | triton-lang/triton-ascend | 90 | issues 1 页 | 1/1 | 是 |

本轮 **22/22 源 complete**，无分页缺口、无 API 失败；与上一轮相比无补齐项。

## 1. NPU 来源判定

- 低成本筛选：2,168 条 issue 中 1,169 条标题命中精度/静默/错值/NaN/溢出/不一致/越界/量化/KV 等关键词；其中窗口内新建 issue 有 253 条命中。按“新建优先 + 机制可定位 + 任务影响”缩到约 40 条标题级复核，实际读批次正文 20 条（含源码行号、SELFPROOF、canary 对照）。2,901 条 PR 只做标题级分类。
- 本轮新增 **13 条**，均 `status: pending_audit`、`confidence: documented`（报告者自述/自带复现/源码定位；本轮未本地复现，不写 `verified`）。

| 编号 | 来源 | 机制摘要 | 关系 |
|---|---|---|---|
| INF-2618 | [ops-math#5340](https://gitcode.com/cann/ops-math/issues/5340) | `split_v` UB-split(103) 的 `DataCopyGatherVf` 用 uint16 lane 算行×pitch+列偏移，Ascend960 单块 B8 >65536 元素时索引回绕，B8/B16 输出整行错值，无报错。 | — |
| INF-2619 | [vllm-ascend#23545](https://github.com/vllm-project/vllm-ascend/issues/23545) | KimiLinear 默认 mamba cache mode='align' + prefix caching 时，跨 prompt 复用 GDN 状态：3 个共享长前缀的并发请求互相返回对方内容，8K 请求退化 128 个 `!`；关闭 APC 后检索 0.0/33.3→100.0。 | — |
| INF-2620 | [vllm#70199](https://github.com/vllm-project/vllm/issues/70199) | FA3 causal sliding-window prefill 在 head dim 128/256、长 query chunk 下偏差 0.118–0.188（FA2 0.009–0.010）；vLLM 自测 4 个 hd256 SWA 用例失败，最大 0.336。 | — |
| INF-2621 | [vllm#70287](https://github.com/vllm-project/vllm/issues/70287) | `VLLM_USE_NCCL_SYMM_MEM=1` 时 `all_reduce_symmetric_with_copy` 闭包 TP 组通信器，非 TP 组（DCP/ETP）≥4 rank 的 all-reduce 静默返回 TP 组之和。 | — |
| INF-2622 | [vllm#70379](https://github.com/vllm-project/vllm/issues/70379) | `grouped_topk` 不校验 scores 连续性、不用 stride(0)：`base[:, :16]` 与连续副本数值相同，路由权重却差 0.0792。 | — |
| INF-2623 | [Ascend/pytorch#6509](https://gitcode.com/Ascend/pytorch/issues/6509) | NPU Inductor persistent masked softmax 行长 229、归约块对齐 232，r_mask 未参与读写/归约，尾部 3 lane 读相邻行；子图最大误差 0.9386，重复运行波动。 | — |
| INF-2624 | [vllm#70420](https://github.com/vllm-project/vllm/issues/70420) | TRITON_MLA 按 q.dtype 分配 decode LSE（bf16），DCP combine 按 `exp(lse_r-lse)` 加权，LSE 舍入 0.03 → 输出误差最高约 2.5%。 | — |
| INF-2625 | [vllm#70419](https://github.com/vllm-project/vllm/issues/70419) | `int4_per_token_head` KV + TRITON_ATTN 时，int4 分支不接 `chunk_lookback`/`rswa_*`，chunked local attention 退化成滑窗、R-SWA 退化成全 causal。 | — |
| INF-2626 | [sglang#53615](https://github.com/sgl-project/sglang/issues/53615) | `--enable-unified-memory` 下状态池是非连续视图，`FusedRecurrentUpdateFunction` 的 `@input_guard` 把参数 `.contiguous()`，更新落在临时副本；Jet-Nemotron 第二步起从零状态解码。 | — |
| INF-2627 | [transformers#51072](https://github.com/huggingface/transformers/issues/51072) | GptOss/AfMoE/GraniteMoeHybrid 的 `mask_kwargs` 漏传 `position_ids`，packed 批不生成块对角掩码，文档间互相可见。 | — |
| INF-2628 | [flashinfer#9838](https://github.com/flashinfer-ai/flashinfer/issues/9838) | 六个 CUDA JIT RMSNorm kernel 用 uint32 算 `bx*stride`，行起点 ≥2^32 元素时回绕；3 行 bf16、行距 2^31 时第 3 行等于第 1 行。 | var INF-2540 |
| INF-2629 | [sglang#53616](https://github.com/sgl-project/sglang/issues/53616) | lightning_attention prefill 的块间衰减用 `exp(-s*r)`，但 KV[i] 是块前状态、query 实为 r+1 步；`linear_backend='minimax'` 下 256 token 后全错。 | var INF-1391 |
| INF-2630 | [ops-transformer#8951](https://gitcode.com/cann/ops-transformer/issues/8951) | KvRmsNormRopeCache FullLoad 在 FP32 先平方累加，BF16 分量 2^80 时平方 2^160 溢出，rstd=0，RMS 输出与写入 CKV 缓存的值被清零。 | var INF-2608 |

### 补充证据与归并（不另占号）

- INF-2521（自动生成 aclnn 出参 stride 静默半写）新增 [ops-math#5310](https://gitcode.com/cann/ops-math/issues/5310)：用 canary 把根因定位到 `l0op::ViewCopy` 按 dst `storageDims` 元素体积界定拷贝范围，而不是按 view shape×strides；同一调用把 storageDims 改成 1D `[maxOffset+1]` 后写满。另新增 [ops-transformer#8894](https://gitcode.com/cann/ops-transformer/issues/8894)：GroupedMatMulAlltoAllvV2 出参 y 非连续时 rc=0 但 `matched_ratio=0.125/0.008`，连续 y 2000/2000 PASS。
- INF-2603（MC2 按连续内存读非连续 x2）新增 [ops-transformer#8896](https://gitcode.com/cann/ops-transformer/issues/8896)：950PR 双 rank 自包含 C++ 复现 AllGatherMatmul，非转置非连续 x2 与出参视图均 rc=0、`out=[23552,0]`；A2 分支已有连续性校验，950 转发 V2 主路径缺失。
- 同机制判定：INF-2629 与 INF-1391 是同一 off-by-one 衰减在两套独立实现（vLLM `lightning_attn.py` vs SGLang `lightning_attn.py`）；INF-2630 与 INF-2608 同属 FP32 平方溢出后 rstd=0（不同算子与载体）；INF-2628 与 INF-2540 同属 32 位偏移算术溢出（阈值 2^32 vs 2^31）。工具回显目标标题均与预期一致。

### 留观 / 不收（NPU）

- [ops-nn#10406](https://gitcode.com/cann/ops-nn/issues/10406)：`aclnnBucketize` 在 int16 boundaries 恰好 65536 条时 `static_cast<uint16_t>(65536)==0`，整片输出 0；机制干净但属特征分桶预处理、非 LLM 推理主路径，留观。
- [ops-nn#10359](https://gitcode.com/cann/ops-nn/issues/10359) / [#10368](https://gitcode.com/cann/ops-nn/issues/10368) / [#10361](https://gitcode.com/cann/ops-nn/issues/10361) / [#10370](https://gitcode.com/cann/ops-nn/issues/10370)：`aclnnElu` 负半轴 `exp(z)-1` 相消（极端 alpha/scale 1e6 下相对误差 100%）、`+inf` 属性与 fp64 标量经 `std::to_string` 截断；参数面过于极端或偏契约/文档，留观。
- [ops-math#5313](https://gitcode.com/cann/ops-math/issues/5313) / [#5329](https://gitcode.com/cann/ops-math/issues/5329) / [#5321](https://gitcode.com/cann/ops-math/issues/5321) / [#5312](https://gitcode.com/cann/ops-math/issues/5312)、[ops-nn#10366](https://gitcode.com/cann/ops-nn/issues/10366) / [#10372](https://gitcode.com/cann/ops-nn/issues/10372)：`IsEqualToOne` 缺 isnan 与整型 alpha >2^24 按 fp32 截断；属同一标量校验族，留观并待与既有 Add 系条目比对后再决定归并。
- [ops-transformer#8928](https://gitcode.com/cann/ops-transformer/issues/8928)：A3 qwen3.5 的 ChunkGatedDeltaRule 在 host tiling 直接读 device tensor，报错退出（fail-loud），不占静默编号。
- 纯文档/错误码文案/CI 脚本竞态/`memcpy_s` 返回值未检查类（如 [ops-transformer#8893](https://gitcode.com/cann/ops-transformer/issues/8893)、[#8880](https://gitcode.com/cann/ops-transformer/issues/8880)、[#8932](https://gitcode.com/cann/ops-transformer/issues/8932)、[cann/ge#936](https://gitcode.com/cann/ge/issues/936)、[cann/runtime#1529](https://gitcode.com/cann/runtime/issues/1529)）不占编号。
- 纯训练反向/优化器类（`HuberLossGrad`、`EluGradV2`、`ApplyFusedEmaAdam`、`Foreach*` 造数脚本等）归训练域，不在本库新建 `INF-*`。

## 2. GitHub 常规 7 仓

- 认证：`gh_api.py` 从 profile `.env` 读取 `GITHUB_TOKEN`，`/rate_limit` 与全部请求 `authenticated: true`（core 4988、search 30/30 起步）；Search 间隔 ≥2.2 秒。
- **API 账本**：7 仓 × 3 关键词组 = **21/21 HTTP 200**，`created:>=2026-11-20`（`cutoff=$(date -u -d '7 days ago' +%F)`），`per_page=100`、`--max-pages 2`；21 次均单页、`has_more=false`、`complete=true`，无分页缺口。

| 仓库 | correctness | silent | NaN/精度 | 合计 |
|---|---:|---:|---:|---:|
| vllm-project/vllm | 72 | 64 | 45 | 181 |
| sgl-project/sglang | 27 | 35 | 11 | 73 |
| huggingface/transformers | 13 | 28 | 6 | 47 |
| NVIDIA/TensorRT-LLM | 6 | 1 | 2 | 9 |
| triton-lang/triton | 7 | 15 | 12 | 34 |
| tenstorrent/tt-metal | 48 | 23 | 19 | 90 |
| flashinfer-ai/flashinfer | 20 | 12 | 6 | 38 |

去重后 **337 个 issue**。本窗口新建条目均来自正文深读（共 10 条：vllm#70199/#70287/#70379/#70419/#70420、sglang#53615/#53616、transformers#51072、flashinfer#9838，另读 transformers#51195 留观）。

### 留观 / 不收（GitHub）

- [transformers#51195](https://github.com/huggingface/transformers/issues/51195)：Inkling 2 Mini 4-bit 下 `Inkling2LoopMessageGate` 把激活 cast 到 packed uint8 权重 dtype，门控错值；机制干净但模型/后端面窄，本轮未读数表，留观。
- [vllm#70066](https://github.com/vllm-project/vllm/issues/70066)（Mamba prefix caching 在 preemption 后按错误 position 缓存状态）、[vllm#70425](https://github.com/vllm-project/vllm/issues/70425)（Mamba prefill checkpoint 被导出到重定位的投机块）、[vllm#70408](https://github.com/vllm-project/vllm/issues/70408)（KV connector 载入的块被当作本引擎计算过做前缀缓存）：与 INF-2619 同属 Mamba/前缀缓存状态归属族，但机制与触发不同，保留下一轮补查，不与本轮条目合并。
- [vllm#70375](https://github.com/vllm-project/vllm/issues/70375)/[#70376](https://github.com/vllm-project/vllm/issues/70376)/[#70377](https://github.com/vllm-project/vllm/issues/70377)/[#70378](https://github.com/vllm-project/vllm/issues/70378)/[#70380](https://github.com/vllm-project/vllm/issues/70380)/[#70381](https://github.com/vllm-project/vllm/issues/70381)/[#70382](https://github.com/vllm-project/vllm/issues/70382)/[#70383](https://github.com/vllm-project/vllm/issues/70383)、[sglang#53582](https://github.com/sgl-project/sglang/issues/53582)/[#53583](https://github.com/sgl-project/sglang/issues/53583)/[#53584](https://github.com/sgl-project/sglang/issues/53584)、[TensorRT-LLM#22992](https://github.com/NVIDIA/TensorRT-LLM/issues/22992)：同一“自定义算子忽略 stride/非连续视图”族，覆盖 reshape_and_cache、topk_softmax、rms_norm、shuffle_rows、awq_dequantize、convert_fp8、moe_sum、ggml_mul_mat_a8 等；与 INF-2596（vLLM 激活/门控族）同机制，需逐条确认归并或 var，本轮不批量占号。
- [vllm#70415](https://github.com/vllm-project/vllm/issues/70415)（`_bias_kernel` 把允许 token 置 -inf）、[#70417](https://github.com/vllm-project/vllm/issues/70417)（MTP draft-input shift kernel warp 竞争）、[#70418](https://github.com/vllm-project/vllm/issues/70418)（Mistral Large 4 `_round_bf16` 把 NaN 变 -0.0）：标题级高信号，正文未读完，保留补查。
- 纯性能、RFC/Feature/Tracking、fail-loud（SIGFPE/SIGABRT/IMA/启动失败）、训练域候选按标题级分类不收；例如 triton#12923/#12945/#12954、tt-metal 大量 tt-train/LLK perf 条目。

## 3. arXiv

- 查询 1 `"silent data corruption" AND "large language model"`：HTTP 200，6 条有效 entry；最近 2610.33475（架构综述）、2610.18557v3（SDC 检测的 Cauchy-Schwarz 界），其余为已收录的 TrainSDC/Anatomy/2502.12340。
- 查询 2 `"bit flip" AND inference`：HTTP 200，25 条有效 entry；近 7 日 [2611.34684 BridgeHammer](http://arxiv.org/abs/2611.34684v1)（CPU-iGPU 共享内存 Rowhammer 后门）与 [2611.34630](http://arxiv.org/abs/2611.34630v1)（bit-slice 加速器）均非自然静默故障；[2610.30798](http://arxiv.org/abs/2610.30798v1) 为 Rowhammer 时延攻击，均在既往日报留观。
- 查询 3 `"KV cache" AND corruption`：HTTP 200，20 条有效 entry；新出现 [2611.31892](http://arxiv.org/abs/2611.31892v1)（persistent KV 与 GPU 执行协同设计，性能类）、[2609.17983](http://arxiv.org/abs/2609.17983v1)（文档编辑后 stale KV 的预算修复，方法类）、[2611.01174](http://arxiv.org/abs/2611.01174v1)（两层服务的升级策略），均非新故障机制；KVGuard 已在 INF-2467。
- 本轮无新增自然静默故障论文。

## 4. 回访与 RAG 返工单

- **旧来源回访**
  - [ops-nn#6232](https://gitcode.com/cann/ops-nn/issues/6232)（INF-1534）：仍 open，正文 2026-11-27 更新，补入零宽 dense_shape 的修复、canary 与 UT/ST 记录，关联 PR #11647；正文自述 UT/ST 未在本地执行，PR 合并状态未核实，结论维持。
  - [ops-nn#6256](https://gitcode.com/cann/ops-nn/issues/6256)（INF-1749）：仍 open，正文 2026-11-27 更新，给出非后缀多轴归约的分组修复与 `[0,2]` 回归用例，AArch64/CANN 9.2.0-beta.2 下 AICPU UT 30/30 通过，关联 PR #11271；未上设备端到端，PR 状态未核实。
  - [vllm-ascend#21902](https://github.com/vllm-project/vllm-ascend/issues/21902)（INF-2564）：GitHub API 仍 open，`updated_at=2026-11-16T11:59:43Z`、1 条评论，无新证据。
  - [flashinfer#6860](https://github.com/flashinfer-ai/flashinfer/issues/6860)（INF-2154）：仍 open，`updated_at=2026-10-20T12:19:50Z`、0 条评论，结论不变。
  - 近 7 日条目所引来源中，本窗口只在批次里检出上述两条 GitCode issue 有正文更新；其余（vllm-ascend#22798/#22199/#23400、vllm#69663、sglang#51531、flashinfer#9887、tt-metal#67315、triton#12861、ops-transformer#8413）未出现在 delta，窗口内无更新。
- **RAG 返工单**：`list-todos --scope inference` 原有 7 条，全部核对原 issue 后处理并核销：
  - 因果逆序：INF-2158 → `["数值结果","输出分布"]`；INF-2600 → `["KV状态","输出内容"]`；INF-2615 → `["输入编码","输出分布","输出内容"]`。
  - 弱检索：INF-2611、INF-2617、INF-2587、INF-2603 在 `effect.detail` 补入用户侧表象词（次正规清零/索引取样乱数据/scale=0 非对齐长度/转置非连续权重进 MC2），均在 150 字内且不超出原文证据；`done-todo` 全部返回 `ok: true`。

## 5. 未覆盖范围与缺口

- NPU 采集：22/22 源 complete，无分页缺口。评估：2,168 条 issue → 1,169 条关键词命中 → 约 40 条标题级 → 20 条正文深读；2,901 条 PR 仅标题分类。未逐条评估的候选保留在批次 `inference_delta.jsonl`，不能据此宣称“无新发现”。
- GitHub：21 组 Search 均 200 且单页完成；337 个唯一 issue 只深读 10 条正文，其余按标题级分类，未逐条拉评论线程。`vllm._custom_ops` / `sgl_kernel` 的 stride 族、Mamba 状态族仍有大量标题级候选待下一轮归并。
- arXiv：3 组短语查询各取 6/25/20 条，未覆盖全库；新论文只读摘要。
- 本轮 13 条均为报告者自述 + 批次正文/源码定位，未在本地复现，`confidence: documented`；NPU 条目的机型/版本按报告者实测写入，未外推到 910B/A5。

## 6. 写入与校验

- 新增 13 条：**INF-2618–INF-2630**，均 `pending_audit`，交 2026-11-28 02:00 UTC 审计。`effect.objects` 按物理因果直接度排序，数组内无重复。
- 关系核对：INF-2628→INF-2540（is_inf 偏移溢出）、INF-2629→INF-1391（vLLM lightning_attn 衰减 off-by-one）、INF-2630→INF-2608（RegBase RMSNorm 溢出归零），工具回显标题均一致。
- 补充证据：INF-2521 + ops-math#5310 + ops-transformer#8894（附 ViewCopy/storageDims 根因说明）；INF-2603 + ops-transformer#8896。
- 回访修改：INF-1534、INF-1749、INF-2564、INF-2154（各追加 notes）。
- `blindspots.md` 新增 IB.1 一条：本窗口 ops-math#5322–#5328、ops-nn#10343/#10383/#10386/#10388/#10391、ops-transformer#8897/#8898/#8904/#8908/#8924 等内核 UT 忽略 `compare_data.py` 失败退出码、写输出不比对 golden、golden 与自身比较、整型坐标用相对容差漏检 off-by-one，属测试基础设施假绿。
- 待运行 `build` / `validate` 后回填本条。
- `catalog_tool.py build` 成功生成 `views/`；`validate` 返回 `ok: true`：entries **2630**、patterns **210**、errors 0、warnings 54、pending_audit 15。编目总数 2,630。
