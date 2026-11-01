# 每日扫描日报 2026-10-31（推理库）

## 0. Git 前置检查

- 分支 `main`，上游 `origin/main`；`git status --porcelain -uall` 为空；`git fetch origin main` 成功；HEAD 与 origin/main 一致（fb7d5f4032733e479e2d63ddc8705a0e3d22ef46）。通过，开始扫描。

## 1. NPU 新增来源（共享批次）

- 批次：`/root/.hermes/profiles/third/cache/fault_scan_sources/batch-20261031T230043195834-8uomr3g8`，采集窗口 2026-10-30T22:00:43Z → 2026-10-31T23:00:43Z（registry_hash 2f6422d0…，22 源全 complete、failures 空）。
- inference scope 候选 22135 条（updated 滚动窗口）；窗口内新建（created_at ≥ since）686 条（ops-nn 145、vllm-ascend 138、ops-math 105、ops-transformer 103、pytorch 63、hcomm 29、runtime 19、triton-ascend 16、hccl 14、ge 13、torchair 12、msmodelslim 12、op-plugin 7、sgl-kernel-npu 4、msprobe 3、mindformers 2、mssanitizer 1）；标题强信号 108 条、正文关键词补命中 63 条。
- 深读正文约 30 条（本地 body 缓存）；拟收候选完成近 14 天日报完整 URL 查重 + `catalog_tool.py find`/grep 编目查重 + 训练库跨库查重（仅 ops-nn#7163=TRN-0374 命中，建 cross 关系）。工具查重另拦下 9 条已在库来源（详见第 2 节纠错记录），本轮 grep 初筛时曾误判，以工具回显为准。

### 新增条目（NPU 来源 19 条；下表编号已按 YAML evidence 核对）

| 编号 | 来源 | 机制摘要 |
|---|---|---|
| INF-2323 | [pytorch#5785](https://gitcode.com/Ascend/pytorch/issues/5785) | 310P3 fp32 matmul/addmm/linear/conv2d/einsum 静默按「fp16 输入+fp32 累加」计算（误差 2~4e-4 超 4~6 倍容差），allow_hf32 三态开关失效无法恢复；A3 正常 |
| INF-2324 | [pytorch#5786](https://gitcode.com/Ascend/pytorch/issues/5786) | 950PR index_put_/scatter_ 越界写 8/8 档全静默，哨兵缓冲被逐位写穿（几何实证：线性偏移×4B−1MB−512B 精确预言落点），复机重跑逐位复现 |
| INF-2337 | [ge#698](https://gitcode.com/cann/ge/issues/698) | ATC 静默把 ONNX Resize half_pixel 语义替换为 asymmetric（解析层未映射 half_pixel_centers），分割模型 S-measure 掉 0.02，align_corners 正常 |
| INF-2325 | [ops-math#4356](https://gitcode.com/cann/ops-math/issues/4356) | SignBitsUnpack out<256 元素恒越界写 32 字节（8 个 -1.0f），40 组扫描零反例；同机 Abs 全零越界对照 |
| INF-2326 | [msmodelslim#664](https://gitcode.com/Ascend/msmodelslim/issues/664) | FP8 E4M3 伪量化只截断不舍入（1.1→1.1 而非 1.125），精度评估与真实 FP8 系统性不一致 |
| INF-2333 | [msmodelslim#666](https://gitcode.com/Ascend/msmodelslim/issues/666) | W8A8 per-group 反量化缺转置，伪量化输出 [16,16,20,20] vs 正确 [6,14,22,30] |
| INF-2327 | [msprobe#665](https://gitcode.com/Ascend/msprobe/issues/665) | Cosine 比对大数溢出误报 NaN、微小非零向量判全零返回 1.0，精度定位结论反向 |
| INF-2328 | [ops-math#4364](https://gitcode.com/cann/ops-math/issues/4364) | SilentCheck stepRef=0 跳过全部阈值判定（首步 inf/nan/超阈全放行）+ sfdaRef 被抹平；V1 资料无此门控 |
| INF-2329 | [ops-math#4367](https://gitcode.com/cann/ops-math/issues/4367) | SilentCheck val=0.0 精确判零早退，sfdaRef/stepRef 状态冻结不更新（1e-45 次正规都正常） |
| INF-2332 | [ops-math#4366](https://gitcode.com/cann/ops-math/issues/4366) | ReduceLogSum 空 tensor 输出填 0 而非 log(0)=-inf（按索引判定，正文未逐段深读） |
| INF-2334 | [ops-nn#7822](https://gitcode.com/cann/ops-nn/issues/7822) | EmbeddingBag includeLastOffset=true 聚合到 indices 末尾（62.11→110.28），133 条用例全 FAIL |
| INF-2338 | [ops-nn#7830](https://gitcode.com/cann/ops-nn/issues/7830) | Index 省略尾维时分离索引误判连续（边界判断读第 5 项 mask），静默错数 DYN_GOLD 31.25% |
| INF-2321 | [ops-nn#7848](https://gitcode.com/cann/ops-nn/issues/7848) | GatherV3 fp64 rank≥5 起错、rank≥6 输出整块全 0，三次重复一致（确定性），fp32 全 rank 正确 |
| INF-2330 | [ops-nn#7853](https://gitcode.com/cann/ops-nn/issues/7853) | aclnnIndex 整型索引不校验 out 形状：偏小静默截断丢数据、偏大尾部未初始化，全部返回 0 |
| INF-2331 | [ops-nn#7858](https://gitcode.com/cann/ops-nn/issues/7858) | ScatterNdUpdate 不校验 updates/indices 约束，越界读值静默写入 varRef |
| INF-2339 | [ops-nn#7867](https://gitcode.com/cann/ops-nn/issues/7867) | Unique/Unique2 不校验 valueOut dtype，位模式重解释脏数据（UniqueConsecutive 正确报 161002） |
| INF-2335 | [ops-nn#7869](https://gitcode.com/cann/ops-nn/issues/7869) | NormalizeBBox FP16 大高宽溢出输出全 0 + 旧 GPU 参考同步犯错致 cross_check 误判 PASS（双重缺陷） |
| INF-2340 | [ops-nn#7806](https://gitcode.com/cann/ops-nn/issues/7806) | MaxPool3DGrad FP16 小 C 轴 tiling 覆盖下发越界 cOutputInner，AIVEC 263（fail-loud 但 host 静默下发非法参数） |
| INF-2348 | [ops-transformer#6975](https://gitcode.com/cann/ops-transformer/issues/6975) | FA/FD minCheckValue 默认值符号错（+FLT_MAX vs 哨兵 -FLT_MAX），全掩码行 LSE 错值；修复 !15023 已 merged |
| INF-2342 | [ops-nn!13251](https://gitcode.com/cann/ops-nn/merge_requests/13251)（merged） | Logit eps double→float 窄化丢次正规语义（clamp 翻转/x=0 输出 inf）；cross→TRN-0374 |
| INF-2341 | [ops-nn#7835](https://gitcode.com/cann/ops-nn/issues/7835) | INTrainingReduceV2 AR 折叠丢低位（±FLT_MAX 对消 sum=NaN）+ ARA 结构失配动态 oom；修复 !13239 open |
| INF-2349 | [vllm-ascend#20510](https://github.com/vllm-project/vllm-ascend/issues/20510) | GLM-5.3-Flash MRV2 PCP 混合批四处缺陷（SFA/KPool 行数失配、sharding 标志、packed store 错位、block table 不足）致静默错答 |

### NPU 补充证据（update，不占新号）

- **INF-2205**（InplaceRemainder 大数塌 0）+ 修复 !6937（open）：A2 unsafe-lane IEEE-754 位级重算，真机大数逐元素精确通过。
- **INF-2316**（compressor 梯度全错）+ 修复 !15041（open）：新增公开入口判据恒 False 与 arch35/92 tilingKey 恒 1（推理态白写 536MB）两个证据面。

### NPU 拒收/留观（窗口内代表）

- **校验缺失/错误码契约大族增量**（ops-math#4354/#4355/#4357/#4359/#4361/#4363/#4377/#4381、ops-nn#7810/#7854/#7857/#7872/#7873/#7874、hcomm#1184/#1186、ge#699、hccl#892 等）— 与 10-22~10-30 已判大族同型（文档 dtype 表/错误码表/命名不一致类），按族留观维持；其中 hcomm#1182（CommInitRootInfo hcclGroupMap 无锁访问，修复 !6394 open）为通信域初始化竞态，机制清晰但触发面在建链阶段（fail-loud 或初始化失败域），未证实静默计算影响，留观。
- **ops-math#4365**（SilentCheck npuAsdDetect=3 命中 L1 时 AICPU 异常 507018 打断任务）— 与 #4364 同算子族但表现为 fail-loud 异常（打挂任务而非静默放行），域外值 0/-1/99 未校验属参数校验族；留观。**#4369**（Topk -inf 报 507015）、**#4370**（PrecisionCompare 有差异时同步 507018 把 out 写 503）— fail-loud 域，但 #4370 使「比对工具差异路径不可用」，监控覆盖影响未实测，留观下轮。**#4362**（SilentCheck result 空指针 SIGSEGV）、**#4358**（LeftShift 混 dtype 位宽基准文档）、**#4366** 已收（INF-2332）。
- **ops-math#4352**（融合 pass 桩 registrar 泄漏+静默门控）— 测试桩/UT 基建域，非设备计算路径，不占号。**#4350**（OneHot numClasses=0 golden 抛异常）— golden/测试资产域。**#4388**（IsClose golden BF16 DTypePromotionError+极值溢出误判 True）— golden 资产域，与已收 INF-2310（IsClose 算子本体）不同对象，测试判定类留观。**#4349/#4387/#4378/#4374**（Tan/MulNoNan/ReduceLogSum 校验汇总帖）— 汇总锚点帖，具体项已含在大族判定。
- **ops-nn#7839**（foreach_pow_scalar 混 dtype 列表越界写+负底数符号错、bn_infer_grad epsilon 校验、single_layer_lstm initH 校验过松，修复 PR !13276 open）— 一帖多缺陷混合 P1/P2，foreach_pow 负底数奇指符号错误（powf 丢负号）为静默数值缺陷但属 `_foreach_pow` API 且证据在汇总帖内未独立复现矩阵，留观下轮拆分补查。**#7840**（AddLayerNorm dstStride 无效赋值）— 代码走读类无行为变化。**#7831**（BN3DTrainingReduce 平方和 1ULP/8 块累加）— 与 INF-2308 同算子族新证（块参数 8→2 后 PASS），证据是修复过程记录，不另开号（族内补充）。
- **pytorch#5784**（external event RFC 语义裁决）— RFC 非缺陷。**torchair#926**（低版本 torch.cond 报错指向误导）、**!3933**（converter 拒收修复）— fail-loud/报错信息域。**triton-ascend !2903** 已收 GitHub 侧（INF-2346）。**sgl-kernel-npu/msprobe/mssanitizer/mindspore/mindformers/MindSpeed** 窗口内无新静默数值候选。
- **PR 修复流**（ops-nn !13209/!13216/!13222/!13223/!13242/!13275、ops-math !6913/!6914/!6921/!6924/!6927/!6931/!6933/!6934/!6936/!6953/!6956/!6959/!6970/!6971/!6973、ops-transformer !15015/!15022/!15044/!15087、ge !5702/!5703/!5710、hcomm !6380/!6382/!6388/!6397）— 白盒修复流按惯例不逐条开号（已收的 !13251/!15041/!6937 除外）；!15015（lightning_indexer eps 类型 NaN）、!6934（range 整型截断）正文薄未深读，下轮补查。

## 2. GitHub 常规 7 仓 + arXiv

- 认证状态：环境无 GITHUB_TOKEN（未认证）；search 走代理出口共享配额，请求间隔 ≥7 秒。21 组查询（7 仓 × correctness/silent wrong output/NaN，created:>=2026-10-24）全部 HTTP 200、无 403/限流等待。窗口内合计命中：vllm 34/10/5、sglang 18/2/1、transformers 1/2/1、TRT-LLM 1/0/1、triton 2/1/0、tt-metal 4/12/13、flashinfer 31/5/6。
- 深读 12 条（core API 单点）；既往判定核对：vllm#64758/#63620/#63776/#64087/#63811/#64005、sglang#46314/#46572/#46476、triton#12506、TRT-LLM#20991 等沿用已收录/已判；sglang#46314=INF-2253 等映射见工具回显。

### 新增条目（GitHub 侧 10 条，其中 6 条与 NPU/修复流来源重叠）

| 编号 | 来源 | 机制摘要 |
|---|---|---|
| INF-2322 | [vllm#64696](https://github.com/vllm-project/vllm/issues/64696) | npcache 源 .bin 更新后静默服务旧权重（全元素 +1.0 后 npcache 仍读旧 NumPy 数组，sum 378 vs 447；同目录 auto 读新值） |
| INF-2336 | [flashinfer#8119](https://github.com/flashinfer-ai/flashinfer/issues/8119) | B12xMoEWrapper 单 wrapper 逐层图捕获后 eager run() 静默全零（_folded_w1_alpha 键绑最后捕获层；main #5168 已加守卫、0.7.x 待 backport） |
| INF-2343 | [vllm#64704](https://github.com/vllm-project/vllm/issues/64704) | .bin 加载 tied 配置下静默丢弃数值独立 lm_head（argmax 63 vs 0 双进程对照；与 INF-1127 VLM config 层级不同机制） |
| INF-2344 | [vllm#64855](https://github.com/vllm-project/vllm/issues/64855) | sparse-indexer ragged spec-decode：weights 未随打包 q 重排（读错 token 权重）+ 因果界用 batch max（短请求丢自身位置 key）→ top-k KV 选错 |
| INF-2345 | [triton#12555](https://github.com/triton-lang/triton/issues/12555) | Triton 3.3.1 sm_100 num_warps=4 误编译循环内条件 tl.store（i_t==0 全零 tile 存出非零；3.5.0+ 正确；下游 fla#1594） |
| INF-2346 | [triton-ascend#2903](https://github.com/triton-lang/triton-ascend/pull/2903) | release/3.2.2 workspace program 总数 32 位回绕（grid 乘积 2^32→0 字节申请/越界；main-dev #2875 已修未覆盖 3.2.2） |
| INF-2347 | [transformers#50195](https://github.com/huggingface/transformers/issues/50195) | Mamba-1 多 token 前向遇 primed cache 静默从零重启 selective scan（Mamba-2 已修 #46084/#46741，Mamba-1 未覆盖） |
| INF-2349 | [vllm-ascend#20510](https://github.com/vllm-project/vllm-ascend/issues/20510) | （见 NPU 节） |
| INF-2342 | ops-nn!13251 | （见 NPU 节，cross→TRN-0374） |
| INF-2348 | ops-transformer#6975 | （见 NPU 节） |

**查重纠错记录**：sglang#47245、vllm#64106、sglang#46259、flashinfer#7179、tt-metal#63257、tt-metal#62901、transformers#50051、vllm#64326 八条首轮拟新收，`catalog_tool.py new` 查重回显确认已在库（INF-2264/2237/2179/2166/2267/2238/2269/2270，10-24~10-29 收录），全部撤回新开号、沿用原判定；本轮 grep 编目查重时曾误判为未收录，以工具回显为准。sglang#46572（Inkling KV slot 0 NaN）=INF-2234、#46476=INF-2211 亦为既往收录。

### GitHub 留观（本轮判定）

- **vllm#64326**（DelegatingParser 忽略 token ids，纯文本标记变 tool call）= INF-2270 已收（10-28），回访仍 open 无新证据。
- **tt-metal 窗口内 quasar 族**（#62465/#63277/#62605/#62613/#62436）与 #62506/#62628/#62383/#62557/#62421/#62517 均沿用 10-24~10-29 判定（已收录或留观维持）；#63259（tt_all_gather 守卫 any-axis 错判）= INF-2281 已收。#63472（sub-device L1 buffer 静默腐蚀）为窗口外老条目未回访。
- **flashinfer#7980**（H100 mxfp8/fp4_quantize 不支持架构返回全零/错字节而非报错）— 与 INF-2166（#7179 SM120 fused MoE）同「不支持的架构静默错果」主题，但对象是量化 API 且诉求为 raise 改造（fail-loud 化），未见静默任务影响实测，留观下轮；**#7598**（PDL 契约 RFC）、**#7637/#7636**（CAKE 功能）非缺陷。
- **transformers#50204**（resume 漏加载根级 adapter）训练侧 resume 域；**TRT-LLM#21124**（Attention-DP KV metrics 拷 rank0 值）指标域，静默影响仅监控读数，与已判 #20991 同期留观维持。

### arXiv

- 2 组查询（"silent data corruption"+LLM/inference、"bit flip"+inference，sortBy=submittedDate）HTTP 200 有效（首轮裸查询词被拆 OR 后以引号短语重发成功）。无新论文：2610.18557/2609.19743/2608.30769/2608.11034/2605.22437/2604.00726/2604.10390/2610.27621/2610.08739/2610.06816/2609.06781/2608.29745/2610.30798/2610.33475 均已在库或已判（10-13~10-30 各轮）。

## 3. 回访（近 7 日条目 + 轮巡旧源）

- **INF-2300**（MoeGatingTopK 1ULP）：源 issue #6944 本批次快照已 closed（10-31），关闭原因未核实，注记（closed≠已修复发布）。
- **INF-2308**（BN3DTrainingReduce FP16）：源 #7762 已 closed（10-30，修复 !13131 此前 merged），是否随版本发布未核实，注记。
- **INF-2309**（FakeQuantPerChannel）：源 #7791 已 closed（10-31），关闭原因未核实，注记；同型新源 ops-nn#7869 已另收（INF-2341，参考同步误判族）。
- **INF-2320**（ApplyAdamWQuant NaN）：源 #7736 已 closed（10-30，修复 !13012 已 merged），是否发布未核实，注记。
- **INF-2275**（DynamicBlockMxQuant）：#7717 保持 closed，无新评论，维持原判。
- 其余近 7 日源（#4272/#4325/#4306/#7741/#7739/#4298/#4301/#4303/#7747/!14989/#7756/#7772/#4344/#4314/#7770/#6939/#6953/#6962/#7640/#6910/#6872/!6791）批次快照均 open 无状态变化；#4298 的修复 !6953（ReduceSum 负轴归一+StdMeanCorrection 错误码）open，关联 INF-2295 待合入。
- 轮巡（更早入库）：INF-2260 源 #7640 open；INF-2259 源 #6910 open；INF-2254 源 #6872 open；INF-2160 修复 !6791 open 无进展。

## 4. 编目与视图

- 新写入 **29 条**（全部 pending_audit）：INF-2321 ~ INF-2349，编号连续。映射（以 catalog YAML 为准）：

| 编号 | 来源 | 标题 |
|---|---|---|
| INF-2321 | ops-nn#7848 | GatherV3 fp64 高秩输出全 0 |
| INF-2322 | vllm#64696 | npcache 静默服务旧权重 |
| INF-2323 | pytorch#5785 | 310P3 fp32 GEMM 静默 fp16 精度 |
| INF-2324 | pytorch#5786 | 950PR 越界写静默腐蚀邻接内存 |
| INF-2325 | ops-math#4356 | SignBitsUnpack 小 out 越界写 32B |
| INF-2326 | msmodelslim#664 | FP8 伪量化只截断不舍入 |
| INF-2327 | msprobe#665 | Cosine 溢出/微小值误判 |
| INF-2328 | ops-math#4364 | SilentCheck stepRef=0 跳过判定 |
| INF-2329 | ops-math#4367 | SilentCheck val=0 状态冻结 |
| INF-2330 | ops-nn#7853 | Index 不校验 out 形状 |
| INF-2331 | ops-nn#7858 | ScatterNdUpdate 约束不校验 |
| INF-2332 | ops-math#4366 | ReduceLogSum 空输入 0 非 -inf |
| INF-2333 | msmodelslim#666 | W8A8 per-group 反量化缺转置 |
| INF-2334 | ops-nn#7822 | EmbeddingBag includeLastOffset 边界 |
| INF-2335 | ops-nn#7869 | NormalizeBBox FP16 溢出+参考误判 |
| INF-2336 | flashinfer#8119 | B12x 图捕获后 eager 全零 |
| INF-2337 | ge#698 | ATC Resize half_pixel 语义替换 |
| INF-2338 | ops-nn#7830 | Index 尾维省略误判连续 |
| INF-2339 | ops-nn#7867 | Unique valueOut dtype 不校验 |
| INF-2340 | ops-nn#7806 | MaxPool3DGrad tiling 越界 AIVEC |
| INF-2341 | ops-nn#7835 | INTrainingReduceV2 AR 折叠 NaN |
| INF-2342 | ops-nn!13251 | Logit eps 窄化次正规丢失（cross→TRN-0374） |
| INF-2343 | vllm#64704 | tied 配置丢弃独立 lm_head |
| INF-2344 | vllm#64855 | indexer ragged 权重/因果界错位 |
| INF-2345 | triton#12555 | sm_100 num_warps=4 误编译 |
| INF-2346 | triton-ascend#2903 | workspace 32 位回绕 |
| INF-2347 | transformers#50195 | Mamba-1 primed 多 token 重启 |
| INF-2348 | ops-transformer#6975 | FA/FD minCheckValue 符号错 |
| INF-2349 | vllm-ascend#20510 | GLM-5.3-Flash PCP 混合批失配 |

（上表已按 `catalog/entries/*.yaml` 的 evidence 字段逐条核对生成，编号↔来源映射以 YAML 为准。）

- 更新 7 条：INF-2205（+!6937 修复证据与注记）、INF-2316（+!15041 两个新证据面）、INF-2300/2308/2309/2320/2275（回访注记）。
- cross 关系回显核对：INF-2342 → TRN-0374「LogitGrad double eps 被窄化 float32 边界判错」✓。
- `catalog_tool.py stats`：2349 条（active 2320 + pending_audit 29）。`build` 完成（CATALOG.zh.md、fault_catalog.csv、xlsx）；`validate` ok:true（n_errors 0，warnings 0）。
- 训练库跨库查重：NPU/GitHub 拟收 URL 在 training_error_review 命中 2 条（ops-math#4006=TRN-0501、ops-nn#7163=TRN-0374）；本轮仅对 INF-2342（!13251↔#7163）新建 cross 关系，ops-math#4006 的既有交叉状态未在本轮变更。

## 5. 未覆盖范围与 API 账本

- GitHub：21 组 search 全 200，无限流等待（累计等待 0，上限 5 分钟未触及）；core API 单点 15 次。新候选中 flashinfer#7980（量化 API 静默零张量）、ops-nn#7839（foreach_pow 多缺陷汇总）、!15015/!6934（修复流）未深读正文，按标题+既往族判定留观，下轮补查。
- NPU：GitCode/AtomGit issue 评论线程未拉取（无低成本批量入口），全部按正文判定——若评论中出现维护者定位/驳回，判定可能需改判；INF-2332（ReduceLogSum 空输入）仅按索引判定（标题+关键词），正文复现矩阵未逐段深读已在条目内注明；hcomm#1182（hcclGroupMap 竞态）留观。
- 强信号 108+63 条中约 30 条深读正文，其余按标题+关键词+既往大族判定留观（校验缺失/错误码契约/文档一致性族为主）。
- arXiv 仅 2 组主题词（SDC/bit-flip 短语），KV corruption/数值正确性主题词未单独跑（与既往结果集重叠度高，缺口小）。

## 6. 结论

29 条新机制入库（待审计），7 条既有条目回访/补证/关联。本轮覆盖：NPU 22 源完整批次窗口（686 条窗口内新建、108+63 强信号、约 30 深读）+ GitHub 7 仓 3 关键词（21 组全 200、12 深读）+ arXiv 2 组；工具查重纠正 8 条初筛误判（已在库沿用）；缺口集中在 GitCode 评论线程、flashinfer#7980/ops-nn#7839 修复流正文与 INF-2332 正文矩阵。
