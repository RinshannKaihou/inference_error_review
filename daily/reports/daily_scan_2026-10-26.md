# 推理故障每日扫描 2026-10-26（UTC）

- 窗口：GitHub/arXiv `created:>=2026-10-19`（7 日）；NPU 共享批次 `batch-20261026T230043147986-d99c34of`（采集 since 2026-10-25T22:00:43Z / until 2026-10-26T23:00:43Z，22 仓全覆盖、0 failures，candidates 17614）。
- 编目：本轮新增 **19 条**（INF-2188 ~ INF-2206，pending_audit），总数 **2206**；validate `"ok": true`（n_errors=0）。
- Git：前置检查通过（main 干净、HEAD=origin/main=4fe2c44）；本轮末提交并推送。

## 一、原 GitHub 7 仓 + arXiv 覆盖账本

- GitHub 未认证 Search API（约 7-8s 间隔，GITHUB_TOKEN 未配置），16 组查询全部 `created:>=2026-10-19` 且全部 200 有效（中途限额重置一次，rem 从 9 自然回补，无 403 等待）：vllm×4（6/3/1/2）、sglang×2（1/0）、transformers×2（0/0）、TensorRT-LLM×1（0）、triton×2（0/11）、tt-metal×2+2（0/0/16/2）、flashinfer×1（0）。候选正文全部由 search item 自带 body 覆盖，未消耗 core API。
- 核对：vllm#63514=INF-2165、#63593=INF-2181、#62945=INF-2117、#62776/#63505 维持 10-19~10-25 判定；#63666/#63650/#63640/#62815 RFC/feature 不占号；triton#12426 维持 10-24（interpreter 调试域）；#12397=INF-1766、#12348=INF-1462、#12347/#12349/#12351/#12365/#12366/#12369/#12370/#12395 维持 10-19~10-25 判定；#12394=INF-2149、#12368=INF-1464；vllm#63250 维持 10-23/10-24 留观（机制未定位）；#62703 维持 10-23（fail-loud 4xx 域）。
- tt-metal 命中多为已判定项：#62176=INF-2180、#61919=INF-收录（10-21）、#62027/#62068/#62074/#62103 维持 10-22 判定；#61701=INF-1784；#61772（SDPA LoFi hang）挂起域维持；#61616/#61620/#61354/#61802 CI/构建域；#61674（closed reshape 96% 元素差——分支 commit 一次性测试失败，已 closed 无独立 issue 复现）留观。
- **新增收录（GitHub 侧无；本轮 GitHub 无新条目——所有新条目来自 NPU 来源，含 2 条 GitHub 平台 NPU 仓，见下节）**。
- arXiv：本轮未发出查询——GitHub 全部候选由 search body 覆盖后，时间预算优先分配给 NPU 候选深读（见未覆盖范围）。

### 新增判定摘要（GitHub 域）

- 无新条目、无改判。tt-metal#62561（BH TT-NOP injector mismatch race on topk_xl_split）与 #61488（WH 同型 topk_const_leak）属 INF-0098（1.99 NOP 注入器 LLK 竞态族）新成员：NOP 注入工具测试基建域，#61488 已有 10-23 判定（不占号）维持，#62561 按同族同域处理不占号。

## 二、NPU 来源扫描（batch-20261026T230043147986-d99c34of）

批次摘要：22/22 仓 complete、0 failures、17614 inference 候选（窗口内 created/updated 2778）。GitCode 系 issues+pulls 分页全取、GitHub 系 3 仓全页校验。批次缓存含逐条 body，本轮判定未额外消耗平台 API。

### 筛选过程

窗口 2778 行中标题强信号命中 250 条（ops-nn 100、ops-transformer 52、ops-math 46、hcomm 13、其余分散）；对照近 14 天日报完整 URL 查重后余 235 条候选；深读正文约 26 条（op-plugin#642、ops-transformer#6734/#6738/#6739/#6743/#6766/#6670/#6713/#6729/#6666/#5625 域、ops-nn#7334/#7347/#7359/#7363/#7372/#7378/#7387/#7388/#7389/#7392/#7414/#7321、ops-math#3999/#4006/#4087/#3974、msprobe#655、vllm-ascend#18209/#19720 等）。

### 新增条目（19 条，pending_audit）

1. **INF-2190** [NEW] [op-plugin#642](https://gitcode.com/Ascend/op-plugin/issues/642) — SDPA bool 掩码全遮 query 行：NPU 输出均匀 softmax（v.mean，allclose 5.96e-08）而非 CPU 的 0 行；加性 -inf 路径两端正一致；仅 bool 路径受影响。950PR/CANN 9.1.0。cross → TRN-0445（训练库同 issue 同机制已收录）。
2. **INF-2191** [NEW] [ops-transformer#6738](https://gitcode.com/cann/ops-transformer/issues/6738) — FIA V5 BSND/BSH + D=512 + 32<S<64 只写前 32 行输出（金丝雀 3.0/-7.0 保留、nz=16384/32256）；BNSD 同内存正确；max_abs=5.24（N=2 时 514）。950PR/CANN 9.2.0-beta.2。
3. **INF-2192** [NEW] [ops-transformer#6739](https://gitcode.com/cann/ops-transformer/issues/6739) — FIA V5 scaleValue=0 输出恒为 mean(V)/2（最大相对误差 0.5018，precision 线 5000 用例 6 条系统性 FAIL）；资料未声明 0 非法。
4. **INF-2193** [NEW] [ops-transformer#6734](https://gitcode.com/cann/ops-transformer/issues/6734) — FIA V5 TND 零长度 batch 静默越界写 GM（memcheck out of bounds 每核一条共数十条、数值正确）；零长度 batch 为必要条件（对照 0 告警）。
5. **INF-2194** [NEW] [ops-nn#7392](https://gitcode.com/cann/ops-nn/issues/7392) — MatmulEmuSplitWeight 混合布局权重静默算错：transposeW 只从 wHigh 推导、wLow 复用（op_api:249/:272-279、kernel:102），view 形状校验全过，最大相对误差 4.26（对照 4.5e-8）；资料声明支持非连续 Tensor。
6. **INF-2195** [NEW] [ops-nn#7335](https://gitcode.com/cann/ops-nn/issues/7335) — FlatQuant 中间值 |x''|>65504 时 quantScale=inf、out 整块 0（bf16 输入同样按 fp16 截断）；资料未声明上限；三组 C++ 复现。
7. **INF-2196** [NEW] [ops-math#3999](https://gitcode.com/cann/ops-math/issues/3999) — BitwiseOrScalar out=INT8 经 fp16 中转截断（>2048 整数按 8 对齐舍入，0|INT32_MAX→0 应 -1）；实测值与 fp16 低 8 位逐条一致；var → INF-2122（ttnn.where 整型经 float32 舍入，同「整型逻辑运算经浮点中转」跨平台变体）。
8. **INF-2197** [NEW] [ops-transformer#6766](https://gitcode.com/cann/ops-transformer/issues/6766) + 修复 PR [!14632](https://gitcode.com/cann/ops-transformer/merge_requests/14632)（open）— BSA 超大 blockSparseMask 展平偏移 uint32 回绕致精度失败（kvseqlen=10M、blockShape=128 实测；PR 提升 uint64 证实机制）。
9. **INF-2198** [NEW] [pytorch#5625](https://gitcode.com/Ascend/pytorch/issues/5625) + PR !48387 — NPU Inductor 八类缺陷（10-25 曾留观，本轮范围扩展重评）：第 1 类 persistent reduction mask 污染（GPT2/BERT 派生 768→1024 tile 无效 lane 非零）与第 2 类多轴 bias 梯度压平为推理库范围真实数值面；var → INF-2159、cross → TRN-0398（训练库同源 bias 梯度视角）。
10. **INF-2199** [NEW] [vllm-ascend#19720](https://github.com/vllm-project/vllm-ascend/issues/19720) — FIA V3 持续 decode 负载 MTE 48bit 地址故障（6 次崩溃同 kernel 签名、故障卡轮换、排除矩阵系统性排除 host 侧；与 #19611/#19608/#18828 同签名族留观并案）。
11. **INF-2200** [NEW] [vllm-ascend#18209](https://github.com/vllm-project/vllm-ascend/issues/18209) — 细粒度 TP rank_grid reshape(pp,dp,tp) 与实际 (ext_dp,dp,pp,pcp,tp) 布局不符，pp>1 时组内混流水线阶段、首个跨 DP 交换静默挂起；静态分析发现（无硬件复现）。
12. **INF-2201** [NEW] [ops-math#4087](https://gitcode.com/cann/ops-math/issues/4087) — acos_grad tiling 四处先运算后 uint32 窄化，totalLength≈2^41-511 时 blockNum=1 只算前 512 元素返回成功；asin_grad 全链 int64 无此问题。
13. **INF-2202** [NEW] [ops-nn#7378](https://gitcode.com/cann/ops-nn/issues/7378) — LayerNormQuant 未校验 gamma 第一维/zeroPoints 长度，非法形状静默接受只读第 0 行（资料 161002 明列应拒绝）。
14. **INF-2203** [NEW] [ops-nn#7334](https://gitcode.com/cann/ops-nn/issues/7334) + [ops-nn#7347](https://gitcode.com/cann/ops-nn/issues/7347) — GroupNorm/GroupNormSilu fp32 窄方差（均值≫标准差）rstdOut=NaN/1/sqrt(eps)、out 大块 NaN（两算子同机制合并收录；良态对照 6e-8 正常）。
15. **INF-2204** [NEW] [msprobe#655](https://gitcode.com/Ascend/msprobe/issues/655) + 修复 MR [!1125](https://gitcode.com/Ascend/msprobe/merge_requests/1125)（open）— 比对工具 handle_inf_nan 只查位置不查符号，+inf vs -inf 被同置 0 后判一致；观测/评测域收录（监测缺陷，非模型计算损坏）。
16. **INF-2188** [NEW] [ops-nn#7387](https://gitcode.com/cann/ops-nn/issues/7387) + #7388 + #7389 — FIA V5 入参校验族：掩码 dtype 与 x 不一致未拦截输出错值、innerPrecisionMode 非 0 被静默忽略、scaleValue=NaN 被当 1.0（三连报合并，主体为掩码 dtype 路径）。
17. **INF-2189** [NEW] [ops-nn#7414](https://gitcode.com/cann/ops-nn/issues/7414) — RotateQuant 空张量 (4,0) 早退返回成功但非空 scaleOut(4) 全部未写（金丝雀 1e30 保留）；var → INF-2187（GatherPaKvCache 空输入静默 no-op，同「空输入早退成功+非空输出未写」机制）。
18. **INF-2205** [NEW] [ops-math#4006](https://gitcode.com/cann/ops-math/issues/4006) — InplaceRemainderTensorScalar fp32 |x|≳2^24 取模结果整片塌 0（2^25~2^28 mod 3 全 0）、other 极小返回 ±inf 违反 |out|<|other|；int32/fp64 正确（A2 走 ops_legacy 内置实现）。
19. **INF-2206** [NEW] [ops-math#3974](https://gitcode.com/cann/ops-math/issues/3974)（closed）+ 修复 MR [!6660](https://gitcode.com/cann/ops-math/merge_requests/6660)（merged）— ConcatToConcatD 融合 pass 超 512 输入拆树中间层节点保留 ES 默认 desc（FLOAT/空 shape）真机静默算错；修复已合入、发布未核实。

### 拒绝 / 留观（NPU 主要项）

- **校验缺失/错误码契约大族**（ops-nn#7400/#7402/#7340/#7321/#7363/#7348、ops-math#3930/#3938/#3962 等）— 与 10-22~10-25 已判定「非法入参静默接受」大族同型，按族留观维持；其中 #7321（GroupNormSwish C=0 输出 inf/0）与 #7363（空 Tensor 出参契约）已并入 INF-2203 remarks，#7348（nullptr gamma/beta 按 1/0 代入）同算子族不同机制留观。
- **ops-transformer#6729**（GMMFR 无 shared_input 精度问题，Ascend950）— issue 正文仅两行（执行单算子用例、精度失败），无机制细节；修复 PR !14518 同日出现但描述未读；与 10-24 #6607（FinalizeRouting）相邻；留观下轮按 PR 描述评估。
- **vllm-ascend#19173**（kimi-k3 PD 重复高精度图 prefill 卡死）维持 10-19 挂起域判定；**#19320**（单机 A2 hcclGetRootInfo error 19）启动 fail-loud 域不占号；**#19611/#19608/#18828** 与 INF-2199 同签名族已在条目 remarks 并案。
- **ops-transformer#6670**（FfnWorkerBatching 二级指针未填静默空结果）与 **#6713**（maxQSeqLen 不一致两段 0 后 sync 507018）— 校验缺失族内证据完整但配置错误域为主（调用方传错参数），留观；#6666（NsaCompressGrad）维持 10-24 转训练库判定（反传算子）；#6743（FA dtype 错误码 161001 vs 161002）错误码契约域不占号。
- **torchair#804**（HCCL aiv aclgraph inplace 单算子执行两次精度问题，c09-20）— 正文仅两行无复现细节，本轮批次无新评论；留观维持。
- **msmodelslim/mssanitizer**：无新计算故障证据；msprobe#1103/#1124（工具功能域）不占号。

### 回访（状态变化）

- **INF-2116**：issue vllm-ascend#19357 已 **closed**；修复 PR #19401 仍 open（merge/发布未核实）。update 已记录。
- **INF-2162**：scatter 修复 MR [!14536](https://gitcode.com/cann/ops-transformer/merge_requests/14536) **已 merged**（10-26 批次快照；gather 侧 !14535 10-24 已合入）；发布版本未核实。update 已记录。
- 近 7 日条目来源快照：INF-2169~2176/2184/2185/2187 全部仍 open（u10-26 无实质状态变化）；INF-2177（mindspore#29702）不在本批 scope 内（mindspore 训推共有，训练批次覆盖）。
- 轮巡（≥3 个更早 open 条目）：INF-1470（op-plugin#602 仍 open，u10-19 无更新）、INF-1469（#605 仍 open）、INF-1471（#598 仍 open）；INF-2178（vllm-ascend#19281 open u10-24，修复 PR #19282 仍 open）。

## 三、API 账本与未覆盖范围

| 组 | 请求 | 结果 |
|---|---|---|
| GitHub search ×16 | 7 仓 correctness/silent/KV/quant/NaN/mismatch 等，全部 `created:>=2026-10-19` | 全部 200 有效，0 次限流等待（中途 rem 自然回补） |
| arXiv | 本轮未查询 | 见未覆盖范围 |
| NPU | 前置脚本共享批次 | 22/22 仓 complete，0 failures，未额外调用平台 API |

- **arXiv 本轮未覆盖**：GitHub 候选全部由 search body 覆盖后，时间预算分配给 NPU 候选深读；arXiv SDC/bit-flip/KV corruption 三组常规查询未发出，补查窗口顺延至下轮（连续 3 日无新机制论文的基线上风险低，但按规范如实记录）。
- 仅按索引/标题判定的候选：校验缺失族约 40 条（ops-nn/ops-math 契约类）、ops-transformer PR 流（!14518/#6729 修复描述未读）、hccl/hcomm/runtime 白盒修复 PR 族、vllm-ascend PR 流（#19744+ BugFix PR 正文未逐条读）——标题与正文严重不符的候选可能漏判。
- GitHub 评论级深查未做（无 GITHUB_TOKEN）：本轮新收录 19 条正文自含机制证据（复现脚本/源码锚点/对照实验），评论级补充留待每日审计。
- 累计等待限流 0 秒。

## 四、编目与视图

- `catalog_tool.py new`×19 全部 ok（两轮 component/behavior 长度修正后通过）；update×2（INF-2116 回访 closed、INF-2162 回访 merged）；`build` 重新生成 views/；`validate` ok（n_errors=0，warnings=49 为既有模式待定项）；stats：2206 条（active 2187 + pending_audit 19）。
