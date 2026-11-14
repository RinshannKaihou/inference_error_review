# 每日扫描日报 2026-11-14（推理库）

## 0. Git 前置检查

- 工作目录 `/workspace/inference_error_review`；分支 `main`，上游 `origin/main`；`git status --porcelain -uall` 为空；`git fetch origin main` 成功；HEAD 与 `origin/main` 一致（`4b4d514191db81efa7d678b3467fd248ef37a70a`）。通过，开始扫描。
- 本库无脏文件、无暂存变更、无本地待推提交，远程未领先。

## 1. NPU 新增来源（共享批次）

- 批次：`/root/.hermes/profiles/third/cache/fault_scan_sources/batch-20261114T230012852853-v9wytci1`，`generated_at=2026-11-14T23:00:12Z`，`cached: false`，`complete: true`。
- 窗口 `2026-11-13T22:00:12Z → 2026-11-14T23:00:12Z`（约 25 小时）。22/22 源全部 `complete`，`failures` 为空；registry_hash `2f6422d00a3d93e14cefe3f2ae3eb409684dc34184969178ccdff1f581d8d94d`。
- inference 候选索引 **33,843 条**（gitcode 27,443 + github 6,400，19 个推理适用仓库；含历史滚动缓存的更新项）。`created_at` 落在窗口内的新候选 **362 条**（issue 119 + PR 243），为本轮判定主体；其余为窗口内被评论/更新触发的旧项。
- 判定方法：362 条新项标题全部过目；53 条强关键词标题命中（精度/静默/错值/溢出/NaN/越界/未初始化/竞态/量化/KV/wrong 等）；逐条读约 25 条批次缓存正文；另对 11-07～11-13 创建、尚未入编目/日报的 240 条强关键词旧候选做了标题级复核。未额外为 GitCode/AtomGit 拉 API。

### 逐仓覆盖（issues/pulls 总数与页数，均 complete）

| 平台 | 仓库 | fetched | 分页明细 |
|---|---|---|---|
| gitcode | Ascend/pytorch | 136 | issues 30（1 页）pulls 106（2 页） |
| gitcode | Ascend/op-plugin | 9 | issues 1 pulls 8 |
| gitcode | Ascend/torchair | 77 | issues 2 pulls 75 |
| gitcode | cann/ops-transformer | 191 | issues 49 pulls 142（2 页） |
| gitcode | cann/ops-nn | 214 | issues 102（2 页）pulls 112（2 页） |
| gitcode | cann/ops-math | 87 | issues 19 pulls 68 |
| gitcode | cann/ge | 39 | issues 4 pulls 35 |
| gitcode | cann/runtime | 24 | issues 7 pulls 17 |
| gitcode | cann/hccl | 59 | issues 8 pulls 51 |
| gitcode | cann/hcomm | 143 | issues 36 pulls 107（2 页） |
| gitcode | mindspore/mindspore | 14 | issues 2 pulls 12 |
| gitcode | mindspore/mindformers | 0 | 无更新 |
| gitcode | Ascend/msprobe | 4 | issues 2 pulls 2 |
| gitcode | Ascend/mssanitizer | 0 | 无更新 |
| gitcode | Ascend/msmodelslim | 11 | issues 5 pulls 6 |
| gitcode | Ascend/MindIE-LLM | 0 | 无更新（推理域 0 条） |
| github | vllm-project/vllm-ascend | 123 | issues 2 页 |
| github | sgl-project/sgl-kernel-npu | 9 | issues 1 页 |
| github | triton-lang/triton-ascend | 42 | issues 1 页 |

训练域 MindSpeed / MindSpeed-LLM / MindSpeed-RL 不在 inference scope，不进入本视图。

### 新增条目（8 条，pending_audit）

| 编号 | 来源 | 机制摘要 |
|---|---|---|
| INF-2552 | [vllm-ascend#21864](https://github.com/vllm-project/vllm-ascend/pull/21864) | 310P 独立 GDN 算子在 ragged MTP 下按当前 query 长度压平 `[B,W]` state 索引表，丢历史槽；accepted_prev>current_len 时读 `row_start+accepted_prev-1` 落到别的请求 state 并写回。修复 PR #21864（open，原 #18730 重提）保留完整 state 表、按行 stride 读写。 |
| INF-2553 | [Ascend/pytorch#6112](https://gitcode.com/Ascend/pytorch/issues/6112) | `triton_experimental` 的 `_apply_linearize` 按文本替换归约维槽位，误改 `npu_triton_helpers.lt(idx, 0)` 的常量 0 为 1/2；合法下标 0 被当负下标加整个维度后越界读。950DT 上 fractional_max_pool2d values maxdiff 16/32、indices 不一致，关 linearize 后 0。修复 PR [#49690](https://gitcode.com/Ascend/pytorch/merge_requests/49690)（open）。 |
| INF-2554 | [vllm-ascend#21891](https://github.com/vllm-project/vllm-ascend/pull/21891) | Mooncake Push 在 AscendStore 逐层 prefill offload 场景下，共享层 KV 的 WRITE 可能晚于模型推进到下一层；源页先被轮转复用覆盖，WRITE 读到的 KV 已不是原层内容，decode 端静默用错 KV。修复 PR #21891（open，依赖 #21256）在 layer hook 提交 shared-layer 读并在复用前 fence WRITE futures。 |
| INF-2555 | [ops-transformer#7578](https://gitcode.com/cann/ops-transformer/issues/7578) | `aclnnSparseFlashMlaMetadata` 的 AICPU `Compute` 对 sequsedQ 越界/负值返回 `KERNEL_STATUS_PARAM_INVALID`，但未映射为 aclnn 错误码：两段仍 0/0，metadata 2048 个 int32 全部保留调用方哨兵；下游按未初始化调度元数据跑 sparse MLA。 |
| INF-2556 | [ops-transformer#7565](https://gitcode.com/cann/ops-transformer/issues/7565) | `aclnnMhcPreSinkhorn` 的 `m_split` 内核 stage2 只对偶数 slice 做 CopyOut；列归一结果留在 `combFragLocal` 后写入 stage3 workspace，`normOut` 奇数 slice 从未写出。needBackward=true 且 B·S 非空时 681/681 用例奇数 slice 为 0.0 或 1e9 残留，偶数 slice 与 golden 一致。var→INF-1699（回显标题“MhcPreSinkhornBackward 输出 grad_x 全程未写”）。 |
| INF-2557 | [ops-math#4752](https://gitcode.com/cann/ops-math/issues/4752) | `SliceWithAxes` 的 `GetParams` 在 `GetConstIntToShape` 返回 false 时跳过切片参数，保留零 offsets 与完整输入 shape；kernel 只读 tiling 的 `fullOffsets/outShape`，运行期张量里的 offsets/size 从未被消费，非常量切片退化为整块拷贝。 |
| INF-2558 | [sglang#50443](https://github.com/sgl-project/sglang/issues/50443) | SGLang NPU 上 hybrid GDN/Mamba 模型（Qwen3.8-27B，A2 32GB）的 mamba state cache 占掉显存后 `max_total_num_tokens=45184`，128K 请求的 `max_new_tokens` 被调度器静默夹到约 1：benchmark 报 8/8 成功、总生成 8 token、TPOT 0.00，只有对比预期生成数才能发现。 |
| INF-2559 | [Ascend/pytorch#6114](https://gitcode.com/Ascend/pytorch/issues/6114) | 同一进程先用另一 shape 编译同一算子、再编目标 shape，后编 kernel 持续错值（fractional_max_pool2d 2×2 maxdiff=7，连调 3 次复现）；两次 kernel hash、launcher cache hash、autotune 配置与 run 实参全同，仅 XBLOCK 8/16 不同，指向运行期按 kernel 名复用 launcher/常量表状态。 |

### 补充已有条目（同机制新证据）

- **INF-2390**（eps 声明 double、实现按 float 截断）：补 [#8805](https://gitcode.com/cann/ops-nn/issues/8805)（aclnnLayerNorm，V3/V4 三处 `static_cast<float>(eps)`；eps=1e-50 与 0.0 输出位级一致、var=0 行 NaN/rstd=inf；eps=2e-45 时 rstd 偏差 19.5%）与 [#8828](https://gitcode.com/cann/ops-nn/issues/8828)（aclnnAddRmsNormCast，eps=1e-45 时 rstd 偏差 15.5%，≤1e-46 下溢为 0 致 inf）。两单均 A2/910B3 + CANN 9.3.0 实测、两段 0/0。
- **INF-2521**（出参 stride 静默半写）：补 [#8814](https://gitcode.com/cann/ops-nn/issues/8814)（aclnnBatchNormBackward 非连续 gradInput 仅 4/8 视图位写对）、[#8819](https://gitcode.com/cann/ops-nn/issues/8819)（FastBatchNormBackward 同型）、[#8750](https://gitcode.com/cann/ops-nn/issues/8750)（AddLayerNorm 950PR 非连续 yOut 8/16 超差）、[#8800](https://gitcode.com/cann/ops-nn/issues/8800)（BatchNorm 非连续 output 漏写）与 [#8689](https://gitcode.com/cann/ops-nn/issues/8689)（SituGluGrad 出参非连续一半未写，维护者确认与 #8663 同根因，修复 PR [#14455](https://gitcode.com/cann/ops-nn/merge_requests/14455) 改手写接口按 strides 回写）。

### NPU 留观 / 不收（本轮判定）

- **同族归并、不另占号**：ops-nn #8815/#8816/#8817/#8820/#8821/#8822（BatchNorm/FastBatchNorm 反向 eps 截断、nullptr 静默兜底、资料校验缺口）、#8806（mean/rstd 空 Tensor 561103 误拒）、#8825/#8826/#8827/#8829/#8830/#8831/#8834/#8840/#8845（RmsNormGrad/AddRmsNormCast/Softmax/LogSoftmax/GeGlu 输入间隔全拒或出参 keepdim/长度问题）与已收的非连续/eps 族同源，按同机制证据处理；#8832（SoftmaxBackward 空 tensor 跳过全部校验、out 未写）与 #8846（GeGlu 非连续三类缺陷）机制明确但未实测 LLM 主路径影响，列为留观。
- **留观（机制明确、影响未实测/触发窄）**：ops-nn **#8809**（LayerNormWithImplMode 在 V4 分流静默丢 implMode，mode2 与 mode0 位级全等；无任务级质量实测）、#8810 + PR #14442（960 beta FMA 相消残差经量化放大，但无 960 实机、仅 950 条件编译模拟）、#8811 + PR #14446（KthValue/Median/NanMedian/Unique 负零恢复失效，-0.0→+0.0 位模式错，对 LLM 任务影响低；归并偏移回绕为预防性）、#8846（GeGlu 输入合法伪转置静默错值 rel=5.8e2，出参部分写入并入 INF-2521）。
- **ops-transformer**：#7566（MhcPreSinkhorn sumOut slice0/1 在 +hcEps 之前写出，差恒 -1e-4；数值偏置、与 #7565 同文件不同缺陷）留观；#7579（InplaceAttentionWorkerScheduler 32 位乘法溢出绕过 buf 校验，仅证明校验绕过、未证明越界写）留观；#7567/#7570/#7571/#7572/#7573/#7574（文档/契约/三方分歧/校准类，非计算故障）；#7578 已收 INF-2555。
- **hccl #968**：DpuBatchSendRecvSoleMesh 接收端缺 drain，多轮后 wqe 溢出 `ibv_post_send failed ret:5`——fail-loud，不占静默故障编号。**hcomm #1336/#1337/#1338/#1339/#1342**：建链错误信息被 socket ERROR 洪泛掩盖、GetStatus 超时异常致 CHANNEL_CONNECT_ERROR_INFO 不打印等，属观测/日志盲区，留观。
- **cann/ge #5953/#5954/#5955**：OM2 常量 offset/size 负数与整数回绕、重名 GraphDef、RebuildOwnership 环检测——模型文件鲁棒性/安全加固，未证明推理静默错值，留观。**runtime #1376**：PyACL `malloc_with_cfg` 属性整数截断，触发面窄，留观。**msmodelslim #700**：QAT 分布式加载协议重构，需求类，不收。
- **Ascend/pytorch**：#6106（triton_experimental fractional_max_pool2d 错误 kernel，values maxdiff 2.31、indices 不一致，根因未定位，与 #6112 独立）留观；#6107/#6108（op-plugin dtype 感知回退缺失，报 161002）、#6109/#6110/#6111/#6113（构建/文档/meta_dropout out 重载编译失败）不收或留观。
- **vllm-ascend**：#21894（A3 上相同 greedy 请求偶发 JSON 双大括号；下游诊断发现首层 output-projection AllReduce 在 per-rank 输入相同时 reduced 输出不同，model-free HCCL replay 选 AIV + 默认 BF16 时输出变化、HCCL_DETERMINISTIC=true 后一致；报告者明确未确认 HCCL 缺陷与 JSON 故障的因果，留观）、#21869（routed experts capture 分片切片错，观测工具域）、#21883（spec draft 校验污染进程级并行配置，导致 target 退回普通 MC2，性能类）、#21893（MXFP fail-fast）、#21872/#21880 等文档修复，均不收。
- **triton-ascend #3171**：DynamicCV 反向缺 tail fill 与 L0C 复用同步缺失，GDN/KDA backward NaN、507015——训练域（反向），移交训练扫描，不在推理库占号。
- **ops-math #4755**（5162A Math 精度问题，描述过泛）、#4756（AcoshGrad 反向公式与前向语义不一致，训练梯度）、#7433（cholesky 非法主元降级 NaN，fail-loud）、#14442/#14446 对应条目已在上面留观，不重复。

## 2. GitHub 常规 7 仓 + arXiv

- 认证状态：环境无 `GITHUB_TOKEN`（`token=no`），全部请求未认证；search 间隔 7 秒。每组实际请求均带 `created:>=2026-11-07`（`cutoff=$(date -u -d '7 days ago' +%F)`），`per_page=100` 单页。
- **API 账本**：7 仓 × 3 关键词组 = **21/21 HTTP 200**，无分页未覆盖（total 均 <100）。计数（correctness / silent wrong output / NaN）：vllm 16/15/8，sglang 19/5/19，transformers 3/4/3，TRT-LLM 0/0/2，triton 0/2/1，tt-metal 5/9/6，flashinfer 2/4/9。去重后 114 个 issue，其中已编目或已在前日报判定 34 个，本轮新出现 80 个。
- **与既往判定比对（维持，无新证据不改判）**：vllm #66211=INF-2457、#67065=INF-2513、#66886=INF-2498、#67313=INF-2533、#66437=INF-2486、#66573 留观、#66435 fail-loud、#66290/#66782/#66794/#66942/#67019 RFC/Tracking；sglang #48964=INF-2447、#49657 留观（AMD gfx12 HIP all-reduce）、#49185/#49198=INF-2464、#49881=INF-2518、#50037=INF-2532、#50484 留观（KimiLinear padding 推进卷积/循环状态）；transformers #50599=INF-2520、#50669 留观（并发 chat template 丢 assistant mask）；flashinfer #9333=INF-2514、#9343=INF-2519、#9279=INF-2531、#9396=INF-2530、#9316=INF-2501；tt-metal #64834=INF-2480、#64811=INF-2481、#64826=INF-2482、#64827=INF-2483、#64840=INF-2484、#64845=INF-2485、#64851=INF-2505、#64834/#64840 族维持；TRT-LLM、triton 0 新命中。
- **新增条目（GitHub 常规来源 1 条）**：INF-2558 [sglang#50443](https://github.com/sgl-project/sglang/issues/50443)（NPU hybrid 模型 KV 池不足时静默只出 1 token）。另 INF-2550 的 #67401/#67411 今日复核仍 open（1/2 条评论），无新证据，维持 11-13 判定。
- **本轮 GitHub 留观/未深读**：#50499 / #9486（flashinfer sampling.softmax 在 GB10 冷启动 6/17 次返回 NaN 行）、#50521（FP16 Stream-K GEMM 精确零结果出 NaN）、#50532（FA3/FA4 cascade attention 重复计 learned sinks）、#50377（qwen3_coder/granite tool parser 对 NaN/Infinity 参数产出非 JSON）、#50759（MuseGlimmerAssistant no-cache forward 用错 sliding-window offset）、#50603（finegrained_fp8 在 CPU 静默错误反量化）、#12757（Blackwell `tl.dot(transposed)*scale` 累加错值，与 INF-0016 同族，按变体候选待下轮核）；其余 RFC/Feature/Perf/Tracking 不占编号。
- **arXiv**：2 组短语查询（"silent data corruption"+LLM、bit flip+inference，sortBy=submittedDate）HTTP 200 有效；返回 8 + 20 篇，ID 均已在既往日报/编目中出现（2608.30769、2608.11034、2604.10390、2604.00726、2502.12340、2608.23774=INF-0314、2608.18404=INF-0313、2607.18342、2605.08615 等），**无新论文**。

## 3. 回访（近 7 日条目 + 轮巡旧源）

- **INF-2390**：来源 #8096/#8694/#8731 仍 open（批次 updated 2026-11-14T06:35–06:41Z），本轮补 #8805/#8828，norm 族 eps 截断已扩到 10 个接口。
- **INF-2521**：#8663 仍 open（updated 2026-11-14T06:35Z）；补 #8750/#8800/#8814/#8819/#8689 五条同族证据，#8663 主单与维护者确认的 #8689→PR #14455 均已记入 notes。
- **INF-2163**：来源 #6658 状态 closed（updated 2026-11-11T04:24:43Z）；关闭原因、修复 PR 与发布版本未核实，closed 不证明已修复，条目结论不变，已 note。
- **INF-2550**：#67401/#67411 均 open（GitHub core API 2026-11-14 核对，分别 1/2 条评论），无新证据，不改判。
- 另核对近期来源：ops-nn #8390（INF-2461）不在本批窗口、ops-transformer #5140（INF-1699）不在本批窗口，均未取得新状态；未据此改判。

## 4. 未覆盖范围与缺口

- NPU：19 个推理仓库 issue/PR 列表均 `complete`，无 API 失败；但 **362 条新候选只深读约 25 条**，其余按标题判定/留观。未深读的 11-07～11-13 旧候选 240 条中，仅对标题做过复核；其中 ops-nn #8389（EmbeddingBag 空 indices 非空 output 未初始化）、#8415（softsign_grad 越域静默错值）、#8429（MSDA grad 缺 spatialShape/numKeys 校验）、ops-transformer #7455/#7472（NormRopeConcat/SparseFlashMlaGrad 出参非连续静默错值）、#7529（FusedFloydAttentionGrad 不校验 N，dqOut 后一半未写）尚待下轮深读，本轮未收录。
- GitHub：21 组 search 均 200 且 total<100，无分页缺口；core API 仅用于个别 issue 状态（67401/67411、50443 等），评论线程未逐条拉取。unauthenticated search 额度未耗尽。
- arXiv：仅 2 组短语查询，未覆盖全库；返回条目均已有判定。
- 未对 msprobe/mssanitizer 的既有工具误报做新实验，相关结论仍为“未评估/无实测”。

## 5. 写入与校验

- 新条目 8 条：INF-2552～INF-2559，均 `status: pending_audit`，由 2026-11-15 02:00 UTC 审计复核。
- 补充条目：INF-2390、INF-2521（evidence + notes）、INF-2163（notes）。
- `catalog_tool.py build` 生成 `views/` 成功；`validate` 返回 `ok: true`，entries 2559、patterns 203、errors 0、warnings 69、pending_audit 8。
