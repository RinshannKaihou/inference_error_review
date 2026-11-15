# 每日扫描日报 2026-11-15（推理库）

## 0. Git 前置检查

- 工作目录 `/workspace/inference_error_review`；分支 `main`，上游 `origin/main`；`git status --porcelain -uall` 为空；`git fetch origin main` 成功；HEAD 与 `origin/main` 一致（`1bd093c643482f616cf9a637b5b0b434af519144`），ahead/behind 0/0。通过，开始扫描。
- 本库无脏文件、无暂存变更、无本地待推提交，远程未领先。

## 1. NPU 新增来源（共享批次）

- 批次：`/root/.hermes/profiles/third/cache/fault_scan_sources/batch-20261115T230012800543-rkiyg29b`，`generated_at=2026-11-15T23:00:12Z`，`cached: false`，`complete: true`；`failures` 为空；22/22 源 `complete`；registry_hash `2f6422d0…8d94d`。
- 窗口 `2026-11-14T22:00:12Z → 2026-11-15T23:00:12Z`（约 25 小时）。inference 候选索引 **34,012 条**（19 个推理适用仓库，含历史滚动缓存）；`created_at` 落在窗口内的新候选 **169 条**（GitCode 140 + GitHub 29），窗口内被更新触发的旧项 **660 条**。
- 判定方法：169 条新候选标题全部过目；对 24 条强关键词/静默数值候选读批次缓存正文与源码锚点（含 2 条经 GitCode API 复核）；其余按标题+摘要判定。未额外为 GitCode/AtomGit 逐个拉评论（除 #6625 撤回核查）。

### 逐仓覆盖（issue/PR 总数与页数，均 complete）

| 平台 | 仓库 | fetched | 分页明细 | complete |
|---|---|---|---|---|
| gitcode | Ascend/pytorch | 43 | issues 6(1p) pulls 37(1p) | 是 |
| gitcode | Ascend/op-plugin | 3 | issues 0 pulls 3(1p) | 是 |
| gitcode | Ascend/torchair | 67 | issues 0 pulls 67(1p) | 是 |
| gitcode | cann/ops-transformer | 144 | issues 61(1p) pulls 83(1p) | 是 |
| gitcode | cann/ops-nn | 57 | issues 22(1p) pulls 35(1p) | 是 |
| gitcode | cann/ops-math | 73 | issues 29(1p) pulls 44(1p) | 是 |
| gitcode | cann/ge | 2 | issues 2(1p) pulls 0 | 是 |
| gitcode | cann/runtime | 9 | issues 2(1p) pulls 7(1p) | 是 |
| gitcode | cann/hccl | 36 | issues 2(1p) pulls 34(1p) | 是 |
| gitcode | cann/hcomm | 50 | issues 9(1p) pulls 41(1p) | 是 |
| gitcode | mindspore/mindspore | 6 | issues 0 pulls 6(1p) | 是 |
| gitcode | mindspore/mindformers | 0 | 无更新 | 是 |
| gitcode | Ascend/msprobe | 2 | issues 2(1p) pulls 0 | 是 |
| gitcode | Ascend/mssanitizer | 2 | issues 0 pulls 2(1p) | 是 |
| gitcode | Ascend/MindSpeed | 3 | issues 1 pulls 2 | 是（训练域，不进本视图） |
| gitcode | Ascend/MindSpeed-LLM | 2 | issues 0 pulls 2 | 是（训练域） |
| gitcode | Ascend/MindSpeed-RL | 1 | issues 0 pulls 1 | 是（训练域） |
| gitcode | Ascend/MindIE-LLM | 0 | 无更新 | 是 |
| gitcode | Ascend/msmodelslim | 3 | issues 2 pulls 1 | 是 |
| github | vllm-project/vllm-ascend | 141 | issues 2 页 | 是 |
| github | sgl-project/sgl-kernel-npu | 6 | issues 1 页 | 是 |
| github | triton-lang/triton-ascend | 14 | issues 1 页 | 是 |

合计 fetched 664。训练域三个 MindSpeed 仓不在 inference scope，不进入本视图。

### 新增条目（6 条，pending_audit）

| 编号 | 来源 | 机制摘要 |
|---|---|---|
| INF-2560 | [ops-transformer#7589](https://gitcode.com/cann/ops-transformer/issues/7589) | `aclnnDequantRopeQuantKvcache` 在 `Nkv==1` 时走 rope/quant 特化分支，整条 `scaleK/scaleV/offset` 量化通路被跳过，`kCacheRef/vCacheRef` 写 `clamp(round(kOut))`：scale=4 时 10.0 写成 10（应 2），scale=0.5 时仍 10（应 20）；Nkv=2/4 同配置正确。5 卡 ×3 次 15/15 一致，两段 rc=0。950PR + CANN 9.3.0。 |
| INF-2561 | [ops-transformer#7581](https://gitcode.com/cann/ops-transformer/issues/7581) | `aclnnMixedQuantSparseFlashMlaMetadata` 在全部入参合法时，metadata 是否写入取决于出参缓冲区预填值：0x00/0x5A 正常写 1360 个 int32，0xA5/0xFF 整份未写，两段 rc=0；T1=batch×maxSeqlenQ 为 1024 倍数且 cmpTopk>0、cmpMaskMode==0 触发。空调度仍无条件 return true（master 亦然），安装包与 master 分支差异已在条目 remarks 注明。 |
| INF-2562 | [ops-transformer#7604](https://gitcode.com/cann/ops-transformer/issues/7604) | `aclnnMoeTokenUnpermuteWithEp` 在 `rangeOptional[0]!=0` 时按 `(全局行号 − start)` 取址：range=[8,16] 输出 0,0,0,0,1000,5000,9000,13000（应 17000…），max|diff|=16000；6000 条用例中 range[0]>0 的 1782 条有 1594 条 FAIL（89.5%），rc 全 0。issue closed，无修复 PR，closed 不证明已修复。 |
| INF-2563 | [ops-transformer#7582](https://gitcode.com/cann/ops-transformer/issues/7582) | `aclnnMoeTokenPermuteV2` 只校验 `permuteTokensOut` dtype 不校验 shape：声明行数小于 M 时只写声明行、其余保留调用方内存且两段 rc=0（canary 残留 32/64、48/64）；列数/维度错则首段 0、次段 561103（不在文档错误码集合）。 |
| INF-2564 | [vllm-ascend#21902](https://github.com/vllm-project/vllm-ascend/issues/21902) | 310P 压缩 SplitFuse 在 context length 2049 违反因果性：只改因果截断后 +4..+19 的 16 个 V，目标 query 行 0→0.5674（应 0），2048 时仍 0；block128/head_dim128 变体 23/384 行同样读到未来 key。两卡一致，无报错。 |
| INF-2565 | [ops-math#4777](https://gitcode.com/cann/ops-math/issues/4777) / [#4778](https://gitcode.com/cann/ops-math/issues/4778) | real_div/real 的 compare_data 把 int32 输出按 float32 解析、空目录直接 PASSED；diag_part 驱动把 fp32/int32 硬编码成 float16、比对空列表判 PASSED。var→INF-1761（compare_data 脚本族，回显标题「compare_data.py 用 zip 比对致多余输出漏检」）。修复 PR #7468/#7469 open。 |

### 补充已有条目 / 来源勘误

- **INF-2283**（aclnnInplaceAtan fp32 大值饱和）：补 [#4764](https://gitcode.com/cann/ops-math/issues/4764)（非 inplace 的 `aclnnAtan` 同样饱和到 1.5707354546：|x|≥16384 起恒定、1e5 偏差 427 ULP、1e8/±inf 511 ULP；同机 Tanh/Log/Sqrt/Erf 正确；官方 202 个用例 range_values 全在 [-5,5]）。机制判断不变，confidence 仍 documented（未定位到具体近似分段）。
- **INF-2167 撤回（勘误）**：[#6625](https://gitcode.com/cann/ops-transformer/issues/6625) 标题已改为「[已撤回] 复核结论有误（探针未下发该参数）」，报告者评论说明 `suVal` 未与 `su=1` 联动、`sequsedQOptional` 实际从未下发；修正探针后 sequsedQ 1024→0 时首核 M 区间 43..85→0..0、启用 AIC 24→1，越界 2048 返回 507018。原「sequsedQ 整条数据通路丢弃」不成立，条目已 `status: retired` 并写入 notes。
- **INF-2173**：[#6622](https://gitcode.com/cann/ops-transformer/issues/6622) 状态 closed（updated 2026-11-10T18:24:45Z），无修复 PR/发布版本可核；结论不变，已 note。

### NPU 留观 / 不收（本轮判定）

- **计算/校验类留观（触发输入超出文档契约或影响未实测）**：ops-transformer **#7600**（FfnWorkerBatching 出参 y 声明偏小未拒，保护带 4096B 被改写，两段 rc=0；OOB 写但无静默错值证据）、**#7605**（ApplyRotaryPosEmbV2 kD<qD 返回 0，key 尾部 padding 被改写；文档要求 D 相等）、**#7609**（NormRopeConcat seqRope 上界按 max 放行，越界读 NaN 扩散；文档上界为 min）、**#7614**（GroupedMatmulAdd 非法 groupList 四类全部 rc=0，输出 NaN/越界值）、**#7619**（QuantSparseFlashMlaMetadata 出参非连续时两段 rc=0 且整块未写；文档标 ×，应拒未拒）、**#7596/#7597**（MoeUpdateExpert eplbTable count=0 写非法专家号 0；F=2 读越界写 0）、**#7583**（MoeTokenPermute INT8 indices≥10240 写 0/-1，issue closed，输入属文档明确不支持范围）、**#7602**（MoeTokenPermuteWithEpGrad rangeOptional 空时 tokenGradOut 全零，训练反向）、**#7592**（NormRopeConcatBackward ropeType=2 梯度错，训练反向）、**#7606**（MoeTokenPermuteGrad 非连续 sortedIndices 被拒 561103，训练反向）、**#7613**（SparseFlashMlaGradMetadata 非法标量延后到 sync 才报）。
- **metadata/索引类留观**：**#7622**（DenseLightningIndexerKLLossGradMetadata 只写 index 0..4，index5/6..63 残留；属训练 KL loss grad metadata）、**#7623/#7624/#7625/#7628**（DenseLightningIndexerSoftmaxLseV2Metadata batch/dtype/seqUsedQ 校验缺口，未证任务影响）、**#7585**（GenericBlockSparseAttentionMetadata maxQSeqLen 约束）。
- **错误码/文档类不收**：ops-transformer **#7588/#7591/#7593/#7594/#7595/#7599/#7601/#7603/#7607/#7608/#7610/#7611/#7612/#7616/#7621/#7626/#7627**（错误码与文档表不一致、空指针返回 561103/161002、整数除零 host 崩溃、维测文案 V1/V2 等；fail-loud 或纯契约）、**#7617/#7618**（benchmark logging 占位符 TypeError）、**#7629/#7630**（新增训练侧融合算子需求）、**#7589 已收 INF-2560**。
- **ops-math**：#4762/#4763（Qr/Randperm、Linspace/LogSpace/Range 空指针返回 561103，文档 161001——错误码域）；#4765（matrix_diag UT 语法/越界，PR #7455 对应，测试脚本域，未证算子本体错）；#4772（build_genop.sh 吞退出码）、#4780–#4785（build 参数/CMake 缺失静默成功，构建工具域，不占静默故障号）；#4777/#4778 已收 INF-2565。
- **ops-nn**：#8846（GeGlu 非连续三类缺陷）**沿用 2026-11-14 判定留观**（机制明确、未实测 LLM 主路径）；#8847（CTCLossV2 咨询）、#8848–#8850（build preflight/reconfigure）、#8851（英文文档示例 CHECK_RET）不收。
- **hccl/hcomm**：#970/#971（PR 遗留检视意见跟踪，已合入改动的 review 项，未给新故障证据）、#1343/#1345（同型检视意见）、#1344（部分 CCU 原语返回码不上传调用栈，日志域、无任务影响实测）、#6668（~CollComm 释放 DPUTAG use-after-free 的修复 PR，open；故障表现为 SIGSEGV，fail-loud）、#6666/#6667/#6669/#6670（RDMA/D2N/RQ count 修复，无静默影响证据）、#4083（拓扑门禁放行，特性）——均留观/不收。
- **Ascend/pytorch**：#6106（fractional_max_pool2d 错误 kernel，与 #6112 独立，根因未定位）沿用 11-14 留观；#6115（triton_experimental HOP 兜底 RFC）、#6116–#6118（TypedStorage/BlockMask API 一致性）、#49691–#49698（assert_async/TypedStorage 测试与 WIP PR）、#49693（文档）不收或留观。
- **triton-ascend**：#3200（inductor+triton 入图报错，fail-loud）、#3197–#3199（ssbuffer 重构/回退/TCB 复用保护，性能与内部重构）留观/不收。
- **vllm-ascend**：#21904（torch_npu profiler 长上下文 trace 静默截断，观测工具域，closed；不转述为模型计算损坏）、#21912（layerwise prefetch 后端初始化前校验）、#21913（chunk_local_cumsum BLOCK_T clamp，crash）、#21915（v0.30.0 padded GDN ssm_state 高级索引 OOM，fail-loud）、#21918（KVTransfer finished_sending drain，可用性）、#21919（纯 GQA K/V 连续化，性能）、#21901/#21907/#21910（性能/特性）、文档/CI PR 不收。**#21902 已收 INF-2564**。

## 2. GitHub 常规 7 仓 + arXiv

- 认证状态：环境无 `GITHUB_TOKEN`（token=no），全部请求未认证；search 间隔 7 秒。
- **API 账本**：7 仓 × 3 关键词组（correctness / "silent+wrong+output" / NaN）= **21/21 HTTP 200**，`created:>=2026-11-08`（`cutoff=$(date -u -d '7 days ago' +%F)`），`per_page=100`，各 total 均 <100，无分页缺口。计数（correctness/silent/NaN）：vllm 20/12/5，sglang 19/5/11，transformers 3/3/2，TRT-LLM 0/0/1，triton 0/2/1，tt-metal 5/7/5，flashinfer 2/3/5。去重后 **94 个 issue**。
- **新增条目（2 条）**：
  - **INF-2566** [vllm#67772](https://github.com/vllm-project/vllm/issues/67772)：ALiBi 不是后端选择条件，`--kv-cache-dtype fp8` 在 SM89 把 BLOOM 路由到 FlashInfer，`FlashInferImpl` 存 `alibi_slopes` 却从不使用；prompt NLL 0.207→3.648，强制 TRITON_ATTN 为 0.218。SM100 默认命中、SM90 启动报错；#66950 只把静默错误变启动错误。
  - **INF-2567** [vllm#67802](https://github.com/vllm-project/vllm/issues/67802)：V2 model runner 的 CUDA graph padding 行不清理 `input_ids/positions`，动态 per-tensor FP8 对整块 `x.view(-1,K)` 求 amax 时把上一请求残留 token 计入；同一 prompt+seed 的 greedy logits 随请求历史变化、有效 token 量化精度被抬高，无报错。
- **新候选留观**：[vllm#67720](https://github.com/vllm-project/vllm/issues/67720)（`--data-parallel-start-rank` 被静默替换为 0，三条路径；配置域，未证输出错值）、[flashinfer#9497](https://github.com/flashinfer-ai/flashinfer/issues/9497)（GDN prefill metadata int64→int32 无范围检查，≥2^31 才触发）、[vllm#66806](https://github.com/vllm-project/vllm/issues/66806)（沿用 11-10 留观，graph/replay 输出差异）、[tt-metal#65627/#65634/#65680](https://github.com/tenstorrent/tt-metal/issues/65627)（悬赏聚合单，含多项缺陷但未拆分验证）、[tt-metal#65495](https://github.com/tenstorrent/tt-metal/issues/65495)（SFPI 编译 ICE，fail-loud）。
- **与既往判定比对（维持）**：vllm #66610=INF-2487、#66801=INF-2500、#66886=INF-2498、#67065=INF-2513、#67313=INF-2533、#66437=INF-2486、#66573 留观、#66435 fail-loud；sglang #50249=INF-2551、#49185/#49198=INF-2464、#49657 留观、#50377/#50499/#50521 留观；transformers #50599=INF-2520、#50603/#50669/#50484 留观、#50545/#50612/#50622 训练域；flashinfer #9333=INF-2514、#9343=INF-2519、#9279=INF-2531、#9396=INF-2530、#9316=INF-2501、#9372=INF-2531、#9486 留观；tt-metal #64834=INF-2480、#64826=INF-2482、#64827=INF-2483、#64840=INF-2484、#64845=INF-2485、#64851=INF-2505、#64933 留观；triton #12740=INF-2547、#12741=INF-2548、#12757 留观。
- **arXiv**：2 组短语查询（`"silent data corruption" AND LLM`、`"bit flip" AND inference`，sortBy=submittedDate）HTTP 200 有效，返回 8 + 20 篇；ID 均已在既往日报/编目出现（2610.33475、2610.18557=INF-1381、2609.19743→INF-0006 证据链、2609.06781、2608.29745=INF-0314、2608.18404=INF-0313、2607.19623→INF-0073/0304 等），**无新论文**。

## 3. 回访

- **INF-2283**：#4219 仍 open（updated 2026-10-30T08:33:12Z），本轮补 #4764 扩展值域证据。
- **INF-2167**：#6625 已撤回（closed、updated 2026-11-02T11:33:33+08:00），按报告者评论撤回条目，状态 retired，见「来源勘误」。
- **INF-2173**：#6622 closed（updated 2026-11-10T18:24:45Z），无修复 PR，结论不变。
- **INF-2555**：#7578 仍 open（updated 2026-11-14T21:56:29Z），无新证据。
- **INF-2550**：#67401/#67411 均 open（GitHub core API 2026-11-15 核对，评论 1/2 条），与 11-13/11-14 相同，不改判。
- 本轮未深读 ops-transformer #7566（MhcPreSinkhorn sumOut 偏置，11-14 留观）与 #7579（InplaceAttentionWorkerScheduler 32 位乘法溢出，11-14 留观），状态未变，沿用原判定。

## 4. 未覆盖范围与缺口

- NPU：22 源 issue/PR 列表均 `complete`、无 API 失败；但 169 条新候选仅深读 24 条正文，其余按标题+摘要判定，存在漏判可能。窗口内 660 条更新旧项只做标题级复核，未逐条重读。
- 新增 NPU 坏例多为 950PR/CANN 9.3.0 实测；910B/A2 的等价触发面未逐一验证，条目里已写实测机型，不外推。
- GitHub：21 组 search 均 200 且 total<100，无分页缺口；core API 仅用于 67772/67802/67720/9497 正文与 67401/67411 状态，未逐条拉评论线程。
- arXiv：仅 2 组短语查询，未覆盖全库。
- 未对 msprobe/mssanitizer 工具误报做新实验，相关结论仍为「未评估/无实测」。

## 5. 写入与校验

- 新条目 8 条：INF-2560～INF-2567，均 `status: pending_audit`，由 2026-11-16 02:00 UTC 审计复核。
- 修改条目：INF-2283（evidence + notes）、INF-2167（status retired + notes）、INF-2173（notes）。
- `catalog_tool.py build` 生成 `views/` 成功；`validate` 返回 `ok: true`，entries 2567、patterns 204、errors 0、warnings 70、pending_audit 8。
