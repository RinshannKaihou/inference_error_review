# 推理故障每日扫描 2026-10-25（UTC）

- 窗口：GitHub/arXiv `created:>=2026-10-18`（7 日）；NPU 共享批次 `batch-20261025T230043149435-wqc7cvfm`（采集 since 2026-10-24T22:00:43Z / until 2026-10-25T23:00:43Z，22 仓全覆盖、0 failures，candidates 16687）。
- 编目：本轮新增 **19 条**（INF-2169 ~ INF-2187，pending_audit），总数 **2187**；validate `"ok": true`（n_errors=0）。
- Git：前置检查通过（main 干净、HEAD=origin/main=8d2b202）；本轮末提交并推送。

## 一、原 GitHub 7 仓 + arXiv 覆盖账本

- GitHub 未认证 Search API（约 7s 间隔，GITHUB_TOKEN 未配置），12 组查询全部 `created:>=2026-10-18` 且全部 200 有效：vllm×4（1/27/29/8）、sglang×2（5/6）、transformers×1（3）、TensorRT-LLM×1（0）、triton×1（13）、tt-metal×2（47/5）、flashinfer×1（1）。本轮未触发 core API 配额等待（候选正文全部由 search item 自带 body 覆盖）。
- 核对：tt-metal 47 条中绝大多数为 10-18~10-24 已判定项（62052=INF-2156、62124=INF-2119、62134=INF-2124、61550=INF-1773、61268=INF-1443、61402=INF-1467、61490=INF-1468、61625/61543/61535=INF-1778、61705=INF-1779、61666/61701=INF-1784、61540=INF-2115 等）；vllm#63514=INF-2165、#63480=INF-2168、#63168=INF-2118、#63170=INF-2121、#63309=INF-2147、#62626=INF-1762、#62697=INF-1764、#62811=INF-2114、#62945=INF-2117、#62304=INF-1442、#62283=INF-1459；sglang#45367=INF-1788、#45755=INF-2141、#44997=INF-1448、#44998=INF-1461、#44999=INF-1454；transformers#49873=INF-2134、#49888=INF-2158、#49810=INF-2157；triton#12368=INF-1464、#12348=INF-1462、#12397=INF-1766、#12347/#12349/#12351/#12365/#12366/#12369/#12370/#12395 维持 10-21~10-24 判定；flashinfer#6907=INF-2108。
- arXiv：3 组查询（SDC、bit-flip、KV corruption）全部 200 有效；近 7 日无新推理故障机制论文（最新相关仍为 10-22 Rowhammer latency attack、10-13 Cauchy-Schwarz SDC detection——均已既往覆盖，非新故障编目对象）。

### 新增条目（7 条，pending_audit）

1. **INF-2179** [NEW] [sglang#46259](https://github.com/sgl-project/sglang/issues/46259)（c10-24）— FlashInfer prefill 分裂不变性破坏：同 prompt 温度 0，radix 命中 1344 token 或 chunked 1344+61 均使首贪心 token 翻转（chat→query_robot_status，3/3）；根因为 ragged 整段 vs ragged+paged+merge_state 双数值分解差 1-2 bf16 ulp、FP8 放大 0.35 nat；隔离矩阵排除 GDN/conv/GEMM/checkpoint，无模型权重最小复现（T=1405 P=1344 D=256）。首个分叉在全注意力层而非 GDN 层。
2. **INF-2181** [NEW] [vllm#63593](https://github.com/vllm-project/vllm/issues/63593)（c10-25）— Qwen3.8-27B（GDN hybrid）MTP forced-reject 与 target-only 发散：margin 0.6-5.9 nats（底噪 ≤0.5）、victim r006 确定、首分叉随 batch 平移（pos154/181）；K>=5、n>=8、L>=192 触发；跨 2 次独立 boot 3/3 复现。#39273 明确排除 MTP，本例证明 MTP 拒绝路径有独立状态缺陷；var → INF-1062（SGLang DSpark GDN 状态漂移，同族「forced-reject 对 GDN 状态累积漂移是伪检验」）。
3. **INF-2186** [NEW] [sglang#46446](https://github.com/sgl-project/sglang/issues/46446)（c10-25）— mixed-chunk 调度改变贪心输出：Qwen3.5-9B 确定性推理（deterministic+seed42+温度0）两请求场景仅 mixed-chunk 开关不同，长请求第 82 token 翻转（41228 vs 8129）；main 3/3、+PR #45545 3/3；与 #46259 独立（不依赖前缀缓存命中）。根因未定位。
4. **INF-2183** [NEW] [tt-metal#62259](https://github.com/tenstorrent/tt-metal/issues/62259)（c10-23）— FPU REDUCE_ROW fp32 Dest 列16残留 face-1 部分和：row-major Float32 W>256 的 ttnn.sum 错数个百分点；bf16 fp32 中间链 ttnn.mean N·C>1 同错（既有 xfail）；fp32_dest_acc_en=False 干净；非 #61588/#62253 回归。
5. **INF-2180** [NEW] [tt-metal#62176](https://github.com/tenstorrent/tt-metal/issues/62176)（c10-22）— SFPU round 负 decimals 用 x*fl(10^-k) 代替精确除法，邻 tie 整步错值（50.000004/-2→0 应 100）；每 1.6e7 正 fp32 约 3.3 万错值@d=-2；既有测试输入为 bf16 值不触发。证据来自 ttsim BH（与 kernel 模型逐位一致），硅上未实测。
6. **INF-2178** [NEW] [vllm-ascend#19281](https://github.com/vllm-project/vllm-ascend/issues/19281)（c10-20，10-24 更新）+ 修复 PR [#19282](https://github.com/vllm-project/vllm-ascend/pull/19282)（open）— Mooncake V2 producer hold 过期后 KV 仍被接受：D 端完成消息只校验 request ID 不校验块持有；同 ID 复用旧完成消息可误放新 hold。离线控制路径复现（非 NPU 数据损坏重放）；var → INF-1350（vLLM NIXL lease 同构，6.282）。
7. **INF-2182** [NEW] [triton-ascend#2645](https://github.com/triton-lang/triton-ascend/issues/2645)（c10-20）— packed vs separate SwiGLU 反向梯度非 bitwise（前向 3.6.0 已修）：fp32 (32,16) 29% 元素差 2.4e-7、fp16 3-D 1/11520 差 3.8e-6；hfusion 重排特征。收录面为 NPU 上 bitwise 回归门控不可用、掩盖真缺陷检测；数值本身舍入量级。

### 重复判定（沿用近期判定，不新建）

- vllm#63505/#62752/#63000/#63343/#62243/#63336/#63602/#63121/#63331/#62483/#62252/#62420/#62351/#62466/#62554/#62762/#62776/#62350 等维持 10-19~10-24 判定（挂起/fail-loud/前端解析/性能域为主）；#62606=已收录维持；triton#12426（interpreter 负 axis——调试模式域维持 10-24 不占号）；sglang#45846（CI 分叉留观维持）、#45186/#45159/#46097 维持；flashinfer#6907 已收录。
- tt-metal 新命中 #62159（SFPU 读 ZEROACC 后未写行——RTL 仿真证明当前所有测试结果不变，ttsim 差异被掩盖为 UB 隐患，非现行计算错误，不占号留观）；#62379（BH RDCFG 下一条指令消费——RTL 仿真实测总读到新值，时序假设非已知失败，留观）；#62255/#62275（Quasar untilize 挂起族维持留观）；#62289（Quasar FPU 舍入与 bf16 精确加测试期望不符——测试期望域，非静默故障，留观）；#62049/#62057/#61852（conv 折叠/perf model 域）。
- transformers#49760（apply_chat_template 丢 processor_kwargs——模板渲染域候选中等，留观下轮）；#49961（CI triage 不占号）。

## 二、NPU 来源扫描（batch-20261025T230043149435-wqc7cvfm）

批次摘要：22/22 仓 complete、0 failures、16687 inference 候选。采集窗口 2026-10-24T22:00Z→2026-10-25T23:00Z。GitCode 系 issues+pulls 分页全取、GitHub 系 3 仓全页校验。批次缓存含逐条 body，本轮判定未额外消耗平台 API。

### 筛选过程

标题强信号命中 633 条（7 日窗口 5021 条中）；对照近 14 天日报完整 URL 查重后，深读正文约 22 条（op-plugin#640/#641/#599、ops-math#3920/#3925/#3963/#3968/#3972/#3924、ops-nn#7303/#7296、ops-transformer#6622/#6607/#6636/#6642/#6709/#6710/#6716、pytorch#5625、mindspore#29702、vllm-ascend#19281）。

### 新增条目（12 条，pending_audit）

1. **INF-2169** [NEW] [op-plugin#640](https://gitcode.com/Ascend/op-plugin/issues/640) + 修复 PR [!6099](https://gitcode.com/Ascend/op-plugin/merge_requests/6099)（open）— torch_npu `_foreach_add_` 对非连续视图（转置）静默不生效：无报错、原存储不变（maxabs 3.38），连续对照正常；三芯（910/910B3/950PR）逐位同错；fused 优化器/AMP 底层路径权重静默冻结。修复 PR 收紧 foreach 快路径门控。
2. **INF-2170** [NEW] [op-plugin#641](https://gitcode.com/Ascend/op-plugin/issues/641) — 复数特殊值传播与 CPU 不一致：reciprocal 按元素数（<3）分路径使 1/2/3/4 元素 cpu=finite→npu=nan 翻转；Inf/NaN 四则 nan↔inf 翻转、exp/log/sqrt/pow inf→nan；910+910B3 双机逐位一致，torch_npu 2.7.1。
3. **INF-2173** [NEW] [ops-transformer#6622](https://gitcode.com/cann/ops-transformer/issues/6622) — MoeTokenPermuteWithEp 环绕区间（start>end）尾部 end 行从未写（哨兵全残留，76 组缺口恒=end，start<end 不触发）；kernel 过滤条件用 rank 值域截断；cross → TRN-0393（训练库同条目，反向关联由审计补）。
4. **INF-2174** [NEW] [ops-transformer#6607](https://gitcode.com/cann/ops-transformer/issues/6607) — MoeTokenUnpermuteWithRoutingMap paddedMode=false 时 outIndex/permuteTokenId 从不写出（canary 0x5a5a5a5a 全残留，两段 rc=0）；paddedMode=true 对照正确；cross → TRN-0391。
5. **INF-2175** [NEW] [ops-math#3920](https://gitcode.com/cann/ops-math/issues/3920) — SimThreadExponential 每次调用尾部越界写 32~992B（fp32 n=256 写 992B；+4096B 留白消失证明紧贴尾部）；随机数/dropout 路径每步调用，显存池下静默污染相邻张量，延迟以 507035 暴露；12 接口阴性对照成立。
6. **INF-2171** [NEW] [ops-transformer#6636](https://gitcode.com/cann/ops-transformer/issues/6636) — PoolKeyIndexer poolTailK>0 时 sparseIndicesOut 产出越界索引（合法 0..31 出现 32/33；poolTailK=0 为 -1 标记）；下游按索引寻址读不存在的 KV 块。
7. **INF-2172** [NEW] [ops-transformer#6716](https://gitcode.com/cann/ops-transformer/issues/6716) — MoeInitRoutingV4 静态量化 round(x*scale+offset) 在 |x-trunc(x)|∈[0.49,0.5) 边界多进 1 LSB（9.49774→10）；125 用例 162 处、全部 .49x 区间；950PR+CANN 9.2.0-beta.2。
8. **INF-2176** [NEW] [ops-math#3963](https://gitcode.com/cann/ops-math/issues/3963) — aclnnAll out=UINT8 输出不是布尔真值而是 self 数值 mod-256 截断（[100,200,300,400]→[100,200,44,144]）；BOOL 对照正确；多 self dtype 均错。
9. **INF-2177** [NEW] [mindspore#29702](https://gitcode.com/mindspore/mindspore/issues/29702) — Ascend VideoDecoder 稀疏批量抽帧确定性码块损坏（[1810,2212,2615] 第 2615 帧 max 差 255/MAE 55.2，删任一前序索引恢复）；代码定位分组无 EOS/drain；EOS 诊断 2 次试验全恢复；多模态视频输入特征被污染。MindSpore 2.10.0+910B2C。
10. **INF-2184** [NEW] [ops-transformer#6642](https://gitcode.com/cann/ops-transformer/issues/6642) — RecurrentKda 资料声明 A2 支持但合法参数（BSND）100% 崩 507035（aivec UB 越界 core38/39）；16 组合凡 host 放行必崩、TND 被 161002 拒绝等于不可用；fail-loud 为主，按「host 校验缺失使合法入参直达设备+资料契约背离+无可用参数组合」收录。
11. **INF-2185** [NEW] [ops-nn#7303](https://gitcode.com/cann/ops-nn/issues/7303) — QuantMatmulV5 最小形状 1x1x1 kernel 静默越界读 392B（10/10 memcheck 命中，返回 0 输出正确，对照 kernel 干净）；950PR。
12. **INF-2187** [NEW] [ops-transformer#6676](https://gitcode.com/cann/ops-transformer/issues/6676) — GatherPaKvCache keyCache 首维=0 早退返回成功且不写任何输出（静默 no-op）；var → INF-2162（同算子族，10-24 建条）。

### 拒绝 / 留观（NPU 主要项）

- **ops-math 校验缺失族增量**（#3924 Bernoulli offset%4/prob 值域、#3925 IsInf 尾块 32B 整块越界读写、#3930 IsInf shape 不校验、#3935 SearchSorteds sorter 越界 507018、#3945 Multinomial 值域、#3954/#3955 InplaceRandom/Uniform from-to 范围回绕、#3968 TopkV2 rank-0、#3972 Arange out 长度静默截断、ops-nn#7296 DeepNorm 尾维形状）— 与 10-22/10-23 已判定「非法入参静默接受」大族同型（两段 rc=0 或延迟 sync 报错），按族留观防编号膨胀维持。其中 #3925（numel=1 越界写 31B、同形 IsNegInf 无此问题）与 #3972（out=16 推导 17 末值丢失、无越界）证据完整，若后续出现端到端影响再收录。
- **ops-transformer 校验缺失族增量**（#6709 activeNumOptional 不生效/忽略、#6710 MoeInitRoutingV2Grad expandedRowIdx 越界读脏数据或 507035、#6715 MoeInitRoutingV4 错误码 561103）— 同族维持留观；#6710 反传算子训练库视角为主。
- **Ascend/pytorch#5625**（Inductor 八类修复汇总：归约 mask 污染/多轴 layout/符号 stride 等，随 PR !48387）— 前 1/2 类为真实数值面（GPT2/BERT 派生归约无效 lane 污染），但 issue 为 PR 配套汇总报告、逐类触发条件与独立复现待拆，与已收录 INF-2159（同仓 Inductor 域）相邻；留观下轮按类评估。
- **triton-ascend#2646**（row_weighting tl.sum 大 shape fp32 累加序差 2-3e-5，CUDA 双路 0 差）— 累加顺序差异在 fp32 舍入界内（报告者自述非 correctness bug），与 INF-2182 相邻但无 bitwise 测试破坏申诉主体；留观并条评估。
- **vllm-ascend#19259**（Gemma4 KV 容量 6x 缩减，closed）— 引入 commit fd81546 定位完整，但属容量/资源域（1/6 预算缩放），无静默数值错误证据，修复 #19264；维持资源域不占号。#19726（310P determine_available_memory 记他进程内存——容量误判域留观）。
- **op-plugin#599/#604/#603** 维持 10-19/10-20 留观（本轮批次无新更新）；#589/#592 崩核族维持。
- **msmodelslim/msprobe/mssanitizer**：量化工具与观测域无新计算故障证据（#635 预检 feature、msprobe#653 维持观测域）。

### 回访（状态变化）

- **INF-2162**：修复 MR [!14535](https://gitcode.com/cann/ops-transformer/merge_requests/14535)（gather 越界读置零）**已 closed/合入 10-24T09:31Z**，发布版本未核实；!14536（scatter 跳过越界 slot）仍 open。update 已记录。
- **INF-2138**（ops-nn#7225 swiglu_mode 静默 mode0）：批次快照确认 issue closed（10-24T01:12Z）、MR !12598 已 merged（10-24 日报已记，本轮补记快照确认）。
- 其余 INF-2159/2160/2161/2163/2164/2167 在批次中状态无变化（open）。
- 轮巡（≥3 个更早 open 条目）：INF-1470（op-plugin#602 仍 open，u10-19）、INF-1469（#605 仍 open）、INF-1471（#598 仍 open）、INF-2116（vllm-ascend#19357 与修复 PR #19401 均仍 open，PR 10-24 有更新内容未读）。

## 三、API 账本与未覆盖范围

| 组 | 请求 | 结果 |
|---|---|---|
| GitHub search ×12 | vllm/sglang/transformers/trtllm/triton/tt-metal/flashinfer，全部 `created:>=2026-10-18` | 全部 200 有效，0 次限流等待 |
| arXiv ×3 | SDC/bit-flip/KV corruption | 200 有效（首轮 http 301 重定向，改 https 后成功），近 7 日无新机制 |
| NPU | 前置脚本共享批次 | 22/22 仓 complete，0 failures，未额外调用平台 API |

- 仅按索引/标题判定的候选：ops-math 校验缺失族约 10 条、ops-transformer #6709/#6710/#6715、Ascend/pytorch#5625 八类中未深读的 3-8 类、vllm-ascend PR 流（#19744/#19754/#19755 等修复 PR 正文未逐条读）、hccl/hcomm/runtime 白盒修复 PR 族——标题与正文严重不符的候选可能漏判。
- GitHub 评论级深查未做（无 GITHUB_TOKEN）：本轮新收录 19 条正文自含机制证据（复现/源码定位/对照实验），评论级补充留待每日审计。
- 累计等待限流 0 秒。

## 四、编目与视图

- `catalog_tool.py new`×19 全部 ok（数轮字段长度修正后通过）；update×4（INF-2169 撤回错误 cross 关联并勘误注记、INF-2162/INF-2138 回访）；`build` 重新生成 views/；`validate` ok（n_errors=0，warnings=46 为既有模式待定项）；stats：2187 条（pending_audit 19）。
