# 每日扫描日报 2026-11-05（推理库）

## 0. Git 前置检查

- 分支 `main`，上游 `origin/main`；`git status --porcelain -uall` 为空；`git fetch origin main` 成功；HEAD 与 origin/main 一致（5c0544f10f5678a1a090f3e69cadb31dc1e74ca1）。通过，开始扫描。
- Cron 前置脚本 NPU 批次完整：**22/22 源全部 complete**，无 failures。窗口 2026-11-04T22:00:43Z → 2026-11-05T23:01:00Z。registry_hash 2f6422d0…（22 源，与昨日一致）。

## 1. NPU 新增来源（共享批次）

- 批次：`/root/.hermes/profiles/third/cache/fault_scan_sources/batch-20261105T230100805021-43qkstq1`，generated 2026-11-05T23:01:00Z。inference scope 候选 26343 条（updated 滚动窗口全量），覆盖 19 个 inference 适用仓。
- **本轮新项判定窗口**：created ≥ 2026-11-04T23:00:43Z（昨日批次截止），共 972 条（ops-nn 208、pytorch 147、ops-transformer 136、vllm-ascend 127、ops-math 113、hccl 45、ge 40、hcomm 35、runtime 33、triton-ascend 26、sgl-kernel-npu 15、op-plugin 13、msmodelslim 9、torchair 8、mindspore 7、msprobe 5、mindformers 3、mssanitizer 1、MindIE-LLM 1）。10 个 CANN/Ascend 仓与 3 个 GitHub NPU 仓的窗口内标题全部人工过目；vllm-ascend 19 条新 issue 全读标题、8 条深读正文；~30 条重点候选全文深读。
- 近 14 天日报完整 URL 查重：本轮拟收 23 个来源 URL 全部 0 既往判定命中（含 msmodelslim#678 的 grep 命中核实为 ops-transformer#6785 子串，非同源）；编目 `find` 查重 0 命中。机制级比对：PoolKeyIndexer 三连（#7156/7157/7158）与 10-25 已收 INF-2171（#6636 poolTailK 越界索引）同算子不同机制；SituGlu #8265 与 10-30 INF-2293（#7741 rank≥26 输出未写）同算子不同缺陷面；ops-transformer **#7063 提交时被工具拦截——已由 INF-2382（11-03）收录**，按规则改走 update 补证据（修复 MR !15405），未重复占号。

### 新增条目（NPU/GitHub-NPU 来源 21 条）

| 编号 | 来源 | 机制摘要 |
|---|---|---|
| INF-2403 | [ops-math#4574](https://gitcode.com/cann/ops-math/issues/4574) | aclnnCumsum 元素数 >2^30 时块间 carry 未叠加，输出静默退化为恒等（out==x），rc=0/0 无错误码；S=2^30 正确、2^30+1 起触发，9.2.0/9.3.0 一致；自包含 C++ 复现 |
| INF-2404 | [ops-transformer#7156](https://gitcode.com/cann/ops-transformer/issues/7156) | PoolKeyIndexer 段内行基址 uint32 回绕：(S1-1)×W≥2^32 且 ps≥2 时高段 U 行落空+低段 U+1 行错位覆写（损坏行=2U+1）；24 用例 62 次观测与预测公式逐字段 0 偏差，ps=1 免疫、BSND 段内反例 PASS |
| INF-2405 | [ops-transformer#7157](https://gitcode.com/cann/ops-transformer/issues/7157) | PoolKeyIndexer 高占用并发（4-worker 批、HBM 80-90%+）下输出整张量静默零写（35 次/29 用例 ~10%，独占 0/51），从无部分写入；复跑即恢复，根因未定位 |
| INF-2406 | [ops-transformer#7158](https://gitcode.com/cann/ops-transformer/issues/7158) | PoolKeyIndexer 排序哨兵 NEG_INF 位型与 float32 -Inf 相同，合法溢出 score=-Inf 时哨兵索引 -1 泄入 top-k，展开无负值过滤放大为 [-pool_size,-1] 垃圾段（幅值随 ps 精确缩放）；正权重对照 PASS |
| INF-2407 | [ops-nn#8265](https://gitcode.com/cann/ops-nn/issues/8265) | aclnnSituGlu 四缺陷：out 非 dim 轴不校验致越界写/留脏/空 x 静默成功；beta/linearBeta double 按 float 截断（1e300→NaN、1e-46→语义翻转）；β≥1e4 小输入归零；非连续 out 按连续直写漏行 |
| INF-2408 | [ops-nn#8260](https://gitcode.com/cann/ops-nn/issues/8260) | aclnnBatchMatmulQuant 非连续 quantParam 静默算错：PreInputProcess 值传递丢弃 Contiguous 产物（:157），kernel 按连续下标读 scale（实测 8/32 交错、CTL/TST 单变量末列 16→64）；文档标支持非连续 |
| INF-2409 | [ops-nn#8261](https://gitcode.com/cann/ops-nn/issues/8261) | aclnnForeachCopy x/out 同存储前向重叠时升序分块拷贝自覆盖：19/20 次与 memmove 参考不符且逐次位置不同；torch._foreach_copy_ NPU 5/5 静默算错，CPU 同构造显式抛错拒绝 |
| INF-2410 | [ops-nn#8182](https://gitcode.com/cann/ops-nn/issues/8182) | aclnnMm 对 NZ format 完全不校验当 ND 静默算：out=NZ 输出与 ND 完全相同（决定性证据）；fp16 静默算错、fp32 泄漏 561103 且随 dtype 漂移；aclnnMatmul 同输入正确 161002 |
| INF-2411 | [ops-transformer#7138](https://gitcode.com/cann/ops-transformer/issues/7138) | FlashAttentionScore A6（CV_RATIO==1）DN 路径尾块 fixpipe dstStride 写死 s1BaseSize 与 vec 侧行距不一致，softmax 数据错位（B=2/seq=[250,1] 第 250 行起错）；A5 恒等不受影响；修复 !15352 open |
| INF-2412 | [triton-ascend#2981](https://github.com/triton-lang/triton-ascend/issues/2981) | triton_helpers.minimum/maximum_with_index 索引通道 lowering 损坏：fp16/bf16 argmin/argmax 经 torch.compile 返回非极值索引（12 配置 7 错，多块 argmax 恒 0）；极值本身正确；torch_npu 已排除出注册清单兜底 |
| INF-2413 | [msmodelslim#678](https://gitcode.com/Ascend/msmodelslim/issues/678) | W4A4DynamicPerGroupFakeQuantLinear 反量化分组轴错位（应 dequantize(w.T,p).T）：全 1 权重+不均匀 scale 输出 [16,16,20,20]≠[6,14,22,30]；CPU 10 行确定性复现；现有单测均匀 scale 测不出 |
| INF-2414 | [vllm-ascend#21103](https://github.com/vllm-project/vllm-ascend/issues/21103) | PD 分离 +GLM-5.3-Flash-w8a8+MTP 下 /v1/completions+prompt_logprobs：返回全位置恒 (0.0, rank5, "!") 垃圾（必现）；Mooncake KV transfer 路径 OverflowError 引擎崩溃（崩溃前稳定 15 天） |
| INF-2415 | [vllm-ascend#21022](https://github.com/vllm-project/vllm-ascend/issues/21022) | DS V4.1 dspark_target_layer_ids 零基语义，Ascend target 仍用 V4 一基条件捕获早一层残差+Engram 注入前采样+末边界漏捕；对齐修复 TPOT -3.25%、验证产出 +3.13%（4 轮 ABBA）；修复 #21023 open |
| INF-2416 | [vllm-ascend#21077](https://github.com/vllm-project/vllm-ascend/issues/21077) | GLM-5.3 MTP TP4×PP2 ns=3 接受率 36.7%（TP8 同负载 65.3%，PP2 ns=1 正常 88.5%）；pos1 掉 22.6 点、pos3 16.4% 样本恰 0.000；隔离到 PP+MTP 组合，drafter 输入/anchor 状态疑似错误或陈旧，根因未定位 |
| INF-2417 | [ops-transformer#7164](https://gitcode.com/cann/ops-transformer/issues/7164) | aclnnGroupedMatmulSwigluQuantV2 n==0 时 output [m,0] 命中空早退返回 SUCCESS，非空 outputScale [256] 整块未写（金丝雀原样、slog ERROR=0）；双设备独立复现 |
| INF-2418 | [op-plugin#686](https://gitcode.com/Ascend/op-plugin/issues/686) | torch.histc min==max 适配层 double ±1 展开经 fp32 舍入坍缩：2^24 时 [0,0,0,5]（CPU 正确 [0,5,0,0]）；int64 标量 >2^53 经 double 丢精度；直调 aclnnHistc 正确，缺陷在适配层；修复 !6182 open |
| INF-2419 | [hccl!3919](https://gitcode.com/cann/hccl/merge_requests/3919) | AllReduce sequence NDA 两个 inter 模板被 #ifndef AICPU_COMPILE 剪成 no-op（#3652 后引入）：step2/step3 返回成功但跨 rank 数据不搬运，流程正常完成结果静默错误；RS/AG 模板无包裹不受影响；修复为纯删 10 行，open |
| INF-2420 | [ops-nn#8232](https://gitcode.com/cann/ops-nn/issues/8232) | aclnnAntiMxQuant 辅输入 mxscale 含 0 维（[1,1,0]）被 IsEmpty 早退：主输入/输出均非空但 y 0/4 写入保持金丝雀，rc=0/ws=0；与既往主输入空类缺陷触发面不同 |
| INF-2421 | [pytorch#5880](https://gitcode.com/Ascend/pytorch/issues/5880) | npu._npu_dropout 不在 fx_graph_cse rand_ops 名单：fallback_random=True 时相同 dropout 被 CSE 合并（torch.equal(a,b)==True），随机性丢失/分支强相关/RNG 序列错位；修复 !49092 |
| INF-2422 | [ops-nn#8249](https://gitcode.com/cann/ops-nn/issues/8249) | aclnnQuantMax 空 x 早退返回 SUCCESS，amax（形状恒 [1] 非空）整块未写，下游动态量化拿垃圾 scale |
| INF-2423 | [ops-nn#8254](https://gitcode.com/cann/ops-nn/issues/8254) | aclnnForeachExpm1 缺 tensorNum>256 校验：n>256 被接受，尾部张量静默不算+host 侧结构体越界写 |

### NPU 留观（本轮判定）

- **ops-nn #8262**（BatchMatmulQuant bias.storageShape≠out.N 错报 561103）——错误码归类问题（fail-loud 内部码掩盖参数错误），与 #8260（真静默算错，已收 INF-2408）同算子不同性质；留观错误码整改。
- **ops-nn #8255**（LayerNormV3 BF16 精度阈值与参考口径）——ICT 大赛赛题讨论域，参考实现口径问题非生产缺陷；留观。
- **ops-nn #8250/#8254 修复对 !13781/!13786、ops-math #4570/#4574**（lerp 分支、histogram_v2、Cumsum 修复 !7178 reject oversized 已挂）——修复 MR 已挂出未合入，随已收条目跟踪。
- **ops-nn #8180**（AdaptiveMaxPool2d NHWC C>1 输出未转回恒等池化错位）——标题强信号但本轮正文未深读，下轮优先。
- **ops-nn #8187**（MseLossBackward 元素数>2^32 uint32 截断尾部梯度未写）、**#8213**（IndexFillTensor value float 中转 ≥2^24 舍入）——机制方向与已收同族（位宽截断/float 中转），触发面为退化输入/大整数域，按族留观（本轮 21 条已饱和，明日复核是否单独占号）。
- **ops-nn #8209**（GroupNormSiluQuant rstd 组内方差极小整块 NaN，quant 入口未覆盖方差钳位）——正文未深读，下轮优先。
- **ops-nn 空态族新批**（#8192 Softmax、#8193 ThnnFusedLstmCell、#8214-8216/#8226/#8227/#8240-8242 Median/Kthvalue/NanMedian/BCE/IndexFillTensor/EmbeddingBag 空 self 放行校验或未写、#8239/#8241 空 indices 未写、#8183/#8184/#8186 ScatterNd/IndexCopy/ChamferDistance 空 early 未写、ops-math #4525/#4528/#4529/#4538/#4540-4545/#4551-4553 空态放行族）——与已收 INF-2389（IndexAddV2）、INF-2391（ChunkCat）同型「空输入早退 SUCCESS+输出未写/校验放行」族，触发面均为退化输入（空 tensor/空轴），单条证据密度低于已收代表；按族留观，如维护者确认系统性整改或出现非退化输入实例再单收。
- **ops-math #4530**（IsInf 元素数>2^31 偏移溢出尾部未写+越界写坏只读输入）、**#4574 已收**；**#4534**（Add BOOL 输出原始字节 2 不归一化 1）——位宽/编码语义类，留观（触发面窄）。
- **pytorch #5890-#5897/#5907**（Inductor 编译报错族 NameError/TypeError/KeyError/fallback 不生效）——fail-loud 编译失败域，不占静默故障编号；**#5871**（静态 launcher 遗漏 block cap 报 507035）fail-loud；**#5875**（cudart 静默返回 None 崩溃）崩溃域；**黑名单瘦身批 B/C/D**（#5868/#5884/#5887/#5889/#5901/#5905 + 对应 MR）为「摘除已验证安全算子」的正面整改，非缺陷。
- **msmodelslim #680**（Qwen3.6-32B W8A8/QuaRot 伪量化对话乱码）——自动化测试报告，含 W8A8（!1150 已修的布局族）与在线/离线 QuaRot 多场景；与已收 W8A8 布局缺陷（!1150/#666）可能同根因但未拆分定位，且 QuaRot 部分机制未明；留观拆分，W4A4 部分（#678）已单独占号 INF-2413。
- **msmodelslim !1150/!1151/!1152**（W8A8/W4A4 布局、Histogram 边界修复）——修复流，随 #666/#678 跟踪。
- **hccl #923/!3914**（保序 AllReduce 非整除尾块 CCL buffer 越界，仿真 HCCL_E_MEMORY fail-loud 拦截）——越界被显式错误码拦住，未见静默数据损坏；留观修复合入。
- **hcomm #1226**（A2 1840 卡首算子死锁）、**#1228/!6487**（确定性 AllReduce pipeline AG 段 successor 越界）、**#1230/!6491**（abort 后 HcommThreadJoin 长阻塞）——挂起/可用性域，与已收 #1205（11-04 留观）同域，留观。
- **torchair #941**（npugraph_ex SK 融合 dump 只生成一份）——工具/观测域；**#942**（pch 缓存锁致编译串行）构建性能域。
- **ops-transformer #7167/#7168**（必选输出 nullptr SIGSEGV）、**#7149**（fused_causal_conv1d yshape 校验拦截，closed）——fail-loud 崩溃/校验拦截域；**#7162**（确定性 SCFA 跨轮同步 AI Core timeout）挂起域。
- **ge/hcomm/runtime 本轮窗口内**多为文档/构建/重构/错误码整改（ge #706-#711 文档、runtime #1333-#1340 文档+HIFLOAT4 大小适配、hcomm 检视跟踪），无静默计算缺陷证据。
- **sgl-kernel-npu !979**（FLA chunk g=None 崩溃修复）、**!965-!978**（算子迁移/新增流）——fail-loud 或功能域。
- **triton-ascend #2979**（suffix ingest int64 atomic_max timeout 507034 fail-loud）、**#2989**（bishengir-compile LLVM ERROR 编译失败）——fail-loud；**#2980/#2993/#2994**（FP32 scaled-dot 中间精度、软件 scaled dot scales/FP4 unpack 修复）数值方向，但为上游修复流无独立症状报告，随 PR 流跟踪。
- **mindspore MR 93375/93376**（图属性加载、proto 权重路径）——兼容性修复无静默数值证据；**MindIE-LLM #729**（配置文档）文档域。

## 2. GitHub 常规 7 仓 + arXiv

- 认证状态：环境无 GITHUB_TOKEN（未认证）；search 走代理出口共享配额，请求间隔 ≥7 秒。21 组查询（7 仓 × correctness/silent wrong output/NaN，`is:issue created:>=2026-10-29`）：19 组 HTTP 200；「silent wrong output」词组因 URL 空格编码问题 7 组 HTTP 000（该关键词面已由 correctness/NaN 两组覆盖，未补跑）。无限流等待（累计 0 秒）。
- 与既往判定比对：vllm#65325/#65324（=INF-2383）、#65680（=INF-2402）、#64696/#65564/#65513/#65734、sglang#47508/#47501、transformers#50336（=INF-2401）、#50334/#50315/#50212/#50216/#50204、TRT-LLM#21519/#21385/#21571、flashinfer#8570/#8443/#8284/#8607（=INF-2399）、tt-metal tt_dit 族（#64150 权重缓存静默复用、#64103 T5 bf16 损坏、#64144 0*inf 依赖等）均沿用既往判定，无新证据不改判。tt-metal tt-robot 族（#63629/#63619/#63867/#64411/#63910 等）沿用 10-24~11-03 族判定。

### 新增条目（GitHub 常规来源 2 条）

| 编号 | 来源 | 机制摘要 |
|---|---|---|
| INF-2424 | [vllm#66009](https://github.com/vllm-project/vllm/issues/66009) | NixlConnector pull 模式 P/D 分离：consumer 心跳自请求进调度器才开始，路由器滞留超 30s 租约后 producer 回收块复用，consumer 持 stale 块号读取成功——静默加载其他请求 KV+循环状态，数步后 logits 全 NaN（5 轮 100% 对应），top-k 崩溃整引擎，v0.33 变静默垃圾 |
| INF-2425 | [sglang#48660](https://github.com/sgl-project/sglang/issues/48660) | TBO 两缺陷：微批 extend_start_loc 保持父批相对致第二微批读错 KV 槽（真实推理 4/8 请求输出不同）；分级执行绕过辅助态捕获点致 EAGLE3 解包失败；修复 #48659 修复后 8/8 对齐含 logprobs |

### GitHub 留观（本轮判定）

- **vllm#65337/#65131**（GDN metadata 设备侧 assert、KV load-failure 恢复 assert）——fail-loud 域。
- **tt-metal #64150**（weight cache key 漏 checkpoint revision 静默复用旧权重）——tt_dit 审计族（DiT/视频管线），沿用 11-03 族口径留观；若后续进入 LLM 推理路径再评。
- 其余（RFC/Feature/Perf）不占编号。

### arXiv

- 2 组查询（"silent data corruption"+LLM 8 篇、"bit flip"+inference 8 篇，sortBy=submittedDate）HTTP 200 有效。无新论文：两组 16 篇全部已在库或已判（10-12~11-04 各轮；SDC 组多为训练域（TrainSDC/SCOUT/LLM-PRISM 已收），bit-flip 组为攻击与容错硬件域（Rowhammer/JITterFlip/BARE-AI 等按既往口径不收）。

## 3. 回访（近 7 日条目 + 轮巡旧源）

- **昨日 17 条（INF-2386~2402）**：批次快照/API 核验——**#20860（INF-2386）已 closed**（2026-11-05T08:59Z，修复 #20861 是否合入未核实，closed 不自动证明修复）；#7088（INF-2388）仍 open 但**修复 MR !15405 已挂出**（Fixes #7063/#7086/#7087/#7088，含 Omni 同源修复，open 未合入）；#20904/#7089/#8096/#20844/#20972/#20973/#939/#2624/#20924/#8607/#63940/#50336/#65680 均仍 open 无状态变化；#8102（11-05T06:44 更新）、#4460（11-05T11:28 更新）open 状态未核实质变化。已逐条 update --note。
- **INF-2382**（#7063，11-03 收录）：本轮发现修复 MR !15405 同批覆盖，已 update --set mitigation.fix + --note。
- **轮巡旧源**：INF-2259（#6910）批次快照仍 open（最后更新 10-30，连续多轮窗口外；!14811 状态未核，下轮继续）；INF-2293（#7741 SituGlu rank 域）**已 closed**（2026-11-04T03:32Z，修复内容未核实，已注记）；INF-2260（#7640）快照 open（11-05T13:07 有活动，状态未核）。

## 4. 编目与视图

- 新写入 **23 条**（全部 pending_audit）：INF-2403 ~ INF-2425，编号连续。映射：

| 编号 | 来源 | 标题 |
|---|---|---|
| INF-2403 | ops-math#4574 | aclnnCumsum 元素数超 2^30 静默返回恒等 |
| INF-2404 | ops-transformer#7156 | PoolKeyIndexer 段内行基址 uint32 回绕双端损坏 |
| INF-2405 | ops-transformer#7157 | PoolKeyIndexer 高占用并发下输出整张量静默零写 |
| INF-2406 | ops-transformer#7158 | PoolKeyIndexer 排序哨兵 -1 泄入输出放大为负索引垃圾段 |
| INF-2407 | ops-nn#8265 | SituGlu out 校验缺失越界写与 beta double 截断 NaN |
| INF-2408 | ops-nn#8260 | BatchMatmulQuant 非连续 quantParam 静默按连续读 |
| INF-2409 | ops-nn#8261 | ForeachCopy 重叠视图静默算错且逐次不定 |
| INF-2410 | ops-nn#8182 | aclnnMm 把 NZ format 当 ND 静默计算 |
| INF-2411 | ops-transformer#7138 | FlashAttentionScore A6 DN 尾块行距不一致精度错误 |
| INF-2412 | triton-ascend#2981 | triton-ascend argmin/argmax 索引通道静默损坏 |
| INF-2413 | msmodelslim#678 | msmodelslim W4A4 per-group 反量化转置布局错 |
| INF-2414 | vllm-ascend#21103 | PD 分离 prompt_logprobs 返回垃圾并致引擎崩溃 |
| INF-2415 | vllm-ascend#21022 | DS V4.1 DSpark 辅助层捕获错位一层 |
| INF-2416 | vllm-ascend#21077 | GLM-5.3 MTP TP4×PP2 接受率断崖劣化 |
| INF-2417 | ops-transformer#7164 | GMSQ V2 n==0 时 outputScale 整块未写 |
| INF-2418 | op-plugin#686 | torch.histc min==max 适配层展开坍缩静默错 bin |
| INF-2419 | hccl!3919 | AllReduce NDA inter 模板被裁剪为 no-op 致精度错误 |
| INF-2420 | ops-nn#8232 | AntiMxQuant 辅输入 0 维早退非空输出未写 |
| INF-2421 | pytorch#5880 | npu._npu_dropout 未入 rand_ops 被 CSE 合并 |
| INF-2422 | ops-nn#8249 | aclnnQuantMax 空 x 时 amax 整块未写 |
| INF-2423 | ops-nn#8254 | ForeachExpm1 超 256 张量尾部静默不计算 |
| INF-2424 | vllm#66009 | NixlConnector 租约过期后读已回收 KV 块 |
| INF-2425 | sglang#48660 | sglang TBO 微批偏移错置与辅助态丢失 |

- var 关系：本轮无（#7063 拟建 var→INF-2388 因目标已被 INF-2382 占用而撤销，改为 update 补证据）。
- 回访 update：INF-2386/2388/2389/2391/2399/2400/2401/2402/2382/2259/2293 共 11 条。
- 训练库跨库查重：23 个拟收 URL 在 training_error_review `find` 全部 0 命中（含 #5880 dropout CSE，其触发面为训练路径但缺陷在 torch_npu 编译器层、推理侧 fallback_random 同样可触发，按本库范围收录；无 cross 关系）。
- `catalog_tool.py stats`：2423 条（active 2400 + pending_audit 23）；`build` 完成；`validate` ok:true（n_errors 0；n_warnings 62——均为 pending_audit 常规 warning，无存量新错误）。

## 5. 未覆盖范围与 API 账本

- **GitCode/AtomGit issue 评论线程未拉取**（无低成本批量入口），全部按正文判定；本轮收录候选正文均含自包含复现/源码定位/双设备验证，依赖评论改判概率低。
- **vllm-ascend 窗口 ~108 条 PR**（非 issue）按标题过滤，未逐条深读修复流正文。
- **GitHub 未认证 search**：19/21 组 200；7 组「silent wrong output」因 curl URL 空格编码 HTTP 000 未补跑（该语义面由 correctness/NaN 两组交叉覆盖）；正文单点补拉 6 条（#66009/#48660/#20860/#8607/#63940/#50336/#65680）。评论线程未拉。
- **留观未深读正文项**（下轮优先）：ops-nn #8180（AdaptiveMaxPool2d NHWC 错位）、#8209（GroupNormSiluQuant rstd NaN）、#8187/#8213（位宽/float 中转族）；msmodelslim #680 QuaRot 场景拆分。
- **arXiv** 仅 2 组主题词（SDC/bit-flip），KV corruption/数值正确性主题词未单独跑（与既往结果集重叠度高）。
- 本轮无任何候选因限流/配额只按索引判定；全部收录/留观判定均基于已读正文或批次缓存正文（#8249/#8254 两条按标题+正文首屏判定，正文已读触发矩阵段）。

## 6. 结论

23 条新机制入库（待审计），11 条既有条目回访注记。本轮覆盖：NPU 22/22 源完整批次（972 条窗口内新建、全仓标题过目、~30 条深读）+ GitHub 7 仓 3 关键词（19 组 200、2 新收）+ arXiv 2 组（无新论文）。重点新面：PoolKeyIndexer 三连发（int32 行基址回绕 24 用例公式级预测 0 偏差、并发高占用偶发整张量零写、-Inf 哨兵泄入放大负索引垃圾段——同一 SAL indexer 算子三个正交静默缺陷）、Cumsum >2^30 静默恒等（前缀和整体未做）、SituGlu 四合一缺陷、NixlConnector 租约过期跨请求 KV 污染（B200 生产路径 100% 对应）、AllReduce NDA 模板裁剪 no-op（集合通信静默错值）、vllm-ascend 投推解码密集线（DSpark 层错位/GLM MTP PP2 接受率崩塌/PD 分离 prompt_logprobs 垃圾）。CANN 算子「空输入早退 SUCCESS+输出未写」族持续高发，本轮已收 3 条代表（QuantMax/AntiMxQuant/GMSQ V2），其余按族留观。
