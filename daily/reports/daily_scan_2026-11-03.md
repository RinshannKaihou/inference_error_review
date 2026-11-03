# 每日扫描日报 2026-11-03（推理库）

## 0. Git 前置检查

- 分支 `main`，上游 `origin/main`；`git status --porcelain -uall` 为空；`git fetch origin main` 成功；HEAD 与 origin/main 一致（37870d9ce31ce8a33c8c41614c15975afc67c5dc）。通过，开始扫描。
- Cron 前置脚本 NPU 批次完整：**22/22 源全部 complete**，无 failures（昨日 vllm-ascend issues 分页失败已恢复，本批该仓窗口为 2 日 2026-11-01T22:00:43Z → 2026-11-03T23:00:43Z 以覆盖缺口；其余 21 源窗口 2026-11-02T23:30:43Z → 2026-11-03T23:00:43Z）。registry_hash 2f6422d0…（22 源）。

## 1. NPU 新增来源（共享批次）

- 批次：`/root/.hermes/profiles/third/cache/fault_scan_sources/batch-20261103T230043152117-2bgahlwt`，generated 2026-11-03T23:00:43Z。inference scope 候选 24161 条（updated 滚动窗口全量）。
- 窗口内新建（非 vllm-ascend，created ≥ 2026-11-02T23:30:43Z）642 条：ops-nn 184（63 issue+121 PR）、ops-transformer 110（41+69）、pytorch 89（25+64）、ops-math 53（17+36）、ge 29 PR、triton-ascend 28（1+27）、runtime 34（10+24）、hccl 22（4+18）、sgl-kernel-npu 16 PR、hcomm 18（4+14）、mssanitizer 14（2+12）、op-plugin 10（3+7）、torchair 14（6+8）、msmodelslim 6（1+5）、mindspore 9（5+4）、msprobe 5（2+3）、mindformers 1 PR。
- vllm-ascend（created ≥ 2026-11-01T22:00:43Z，2 日窗口）266 条（41 issue + 225 PR），覆盖昨日分页失败后的回补。
- 筛选流程：正则强信号（静默/错值/精度/越界/NaN/残留/非确定等）命中 46 条 → 逐条读正文深读约 20 条；全部 184+41 条窗口内 issue 标题人工过目；PR 按标题+关键正文抽查。

### 新增条目（NPU/GitHub-NPU 来源 10 条）

| 编号 | 来源 | 机制摘要 |
|---|---|---|
| INF-2373 | [vllm-ascend#20827](https://github.com/vllm-project/vllm-ascend/issues/20827) | block_table append_row/add_row/move_row 残留上一租户块号，消费方按请求长度索引行解引用残留 id；hybrid dense 布局下同页被 bf16 KV 与 fp32 ssm 状态混读，前缀命中请求 12/12 确定性乱码（两台 910B 位级同签名）；410 份 FIA dump 逐单元归因；修复 #20828 open |
| INF-2374 | [vllm-ascend#20829](https://github.com/vllm-project/vllm-ascend/issues/20829) | 检查点声明 QuaRot 全局旋转但 optional/quarot.safetensors 缺失时静默回退单位阵（不校验声明路径存在性），语言流畅但知识/算术塌缩（3*4=9,216）、prompt_logprobs 近随机；修复 #20830 open（fail-fast） |
| INF-2375 | [vllm-ascend#20725](https://github.com/vllm-project/vllm-ascend/issues/20725) | BalanceScheduler 构造 SchedulerOutput 漏置 has_sync_kv_loads（默认 False），同步外部 KV 装载推迟到 forward 后，层等待提前返回读未恢复 KV；HBM 清空仅外部命中场景输出损坏；插桩时序+connector workaround 双向验证 |
| INF-2376 | [ops-nn#8006](https://gitcode.com/cann/ops-nn/issues/8006) | ForeachRoundOffNumber/V2 roundMode=6 返回未初始化 UB 缓冲残值：同输入每次不同、全 0 输入返非 0、随 shape 变化；roundMode 1~5 正确；两段返回码 0 |
| INF-2377 | [ops-nn#8007](https://gitcode.com/cann/ops-nn/issues/8007) | ForeachSin fp32 通路 |x|>=1e10 角度归约中间量溢出，sin 返回 5.18e21/±inf 越界值（违反有界性）；源码定位 foreach_sin.cpp tiling 分派；fp16/bf16 正确；精度线 28/775 全 FAIL |
| INF-2378 | [ops-nn#8012](https://gitcode.com/cann/ops-nn/issues/8012) | ForeachNorm 只实现 p∈{1,2}，p=1.5 静默返 L1、p=3/0.5/±inf/nan 返 L2（p=0.5 差万倍）；同机 torch._foreach_norm 全对，两段返回码 0 |
| INF-2379 | [ops-nn#8013](https://gitcode.com/cann/ops-nn/issues/8013) | foreach 幂族非负底数 NaN 指数当 0 次幂返 1.0（IEEE 应 NaN）；边界矩阵证明仅 exp=nan 列错；同机 torch.pow/普通 pow 全对，缺陷限定 PowerAdapter 适配链 |
| INF-2380 | [ops-transformer#7051](https://gitcode.com/cann/ops-transformer/issues/7051) | block_sparse_attention 超大 blockSparseMask 展平偏移超 uint32 回绕，kvseqlen=10M+blockShape=128 精度失败；修复 !15196 已 merged（发布未核实） |
| INF-2381 | [ops-transformer#7063](https://gitcode.com/cann/ops-transformer/issues/7063) | ScatterPaCache 场景二第三段从 key[0] 起写而非 compressSeqOffset+compressLens 起，keyCache 写入重复历史 token、注意力静默错值；自包含 C++ 复现 |
| INF-2382 | [mindspore#29710](https://atomgit.com/mindspore/mindspore/issues/29710) | 字符串枚举跨枚举静默映射：REG_STRING_TO_ENUM_COMMON 的 AddValues 空 key 把所有 COMMON 枚举 merge 成一张表，norm='none' 按 ortho 算、reduction='ortho' 按 none 算；完整逐值复现表+源码定位 |

（更正：INF-2377 来源是 ops-nn#8007，上表第一列笔误指向 #8006，以本行与编目 YAML 为准。）

### NPU 留观（窗口内代表）

- **#20825**（Biasless AscendRMSNorm 带 residual 时调 AddRmsNormBias beta=None，CANN 9.1 无该算子报 RuntimeError）——fail-loud 崩溃面，最小复现+源码定位清晰，但无静默错值证据；#17447（route unbiased RMSNorm to CANN）在途。如后续证实静默路径再评。
- **#20690**（padded MTP partial prefill fallback token 用 N-1 而非 C+Q）——CPU probe 验证+修复 #20696 open（190 passed）；属投机解码 draft 输入构造，影响面为 partial-prefill 边界行的 token 配对，机制真实，留观修复合入与任务影响实测。
- **#20628/#20743**（MooncakeConnectorV1/HybridConnector kv_both 只启动接收线程）——引擎正常起但 sender 永不创建，KV 只进不出；功能不可用域而非静默错值，正文含完整根因分析，留观。
- **#20748**（结构化输出并发下整型字段无限数字流至 max_tokens）——正文仅两个附件链接无环境/复现细节，证据不足，留观待补充。
- **ops-nn foreach 探针族**（#8004 LerpList view shape 校验、#8008 RoundOffNumber roundMode=7 文档、#8009 错误码不一致、#8014 PowScalarAndTensor INT32 白名单矛盾、#8017 PowScalarList out 小于 x 越界写风险、#8039/8040 性能与容量校验）——与 11-01/11-02 已判「foreach 契约族」同型；#8017 越界写风险未证实实际写坏（TTK PASS 即复现但无损坏证据），按族留观。**本轮已例外收录 #8006/#8007/#8012/#8013**（正文含真机自包含复现+源码定位+同机反证，超出族内纯探针）。
- **ops-nn 修复流**（!13477 conv2d_v2 pad>=kernel 精度全错修复、!13520 wqbmmv2 无尾块 baseN 置 0 core dump 修复（源 #8032）、!13451 deep_norm 精度（源 #8022 closed）、!13491 RotateQuant 越界返回 161002）——修复 PR 正文多为模板无细节，对应源 issue 多为 UT/精度线触发（#8055 正文仅三行无日志），fail-loud 或证据薄，留观。
- **pytorch Inductor 批**：#5816（静态完整归约选归约轴做 grid split，归一化 epilogue 复用输入缓冲竞争）——确定性复现+修复 !48919+NPU 九次数值对照，机制真实；但触发面为特定归一化模式编译路径，留观修复合入后评估。#5817（归约单例输入缺广播致非法 reshape）编译失败 fail-loud；#5818（21 libdevice 算子 extern 回退 CPU+erf/expm1 dtype 传播不一致）——回退有警告日志且 dtype 不一致影响面待评估；#5819（精度自检未接线）工具域。#5828（getCurrentNPUStream 未初始化默认流）、#5830/5832 内存/构建域非本库范围。
- **torchair #929**（AMP 图混合精度 Conv2d fp32 bias 被 Python 层 fp16 常量覆盖+2.8x 慢）——数值语义变化+性能，正文待深读，留观。
- **hcomm #1199**（AIV+UBMEM GetRemoteMems 成功但 remoteMem.type=INVALID）——返回值字段错误属 API 契约域，无计算污染证据，留观；#1198（pc 指针多 mission）描述过简。
- **msprobe #667/!1135/!1136**（ACLGraph nan_check 缺 FP8/FP4 兜底，采集可能中断）——观测工具自身缺陷域（监测≠计算故障），!1135 已 merged；按 NPU_SCAN 第 5 条归观测工具记录，不入故障编号。
- **ops-transformer #7070**（Flash Attention 反向 deterministic=True+TND 特定 shape 结果有误）——复现脚本完整、A3+CANN 9.1.0+序列>1024 复现；FA 反向主属训练域，推理库范围内前向不受影响，留观（训练侧扫描如已收以训练库为准）；**#7076**（indexer_quant_cache MX-FP4 scale floor vs ceil 与网络不一致）为需求建议域（精度对齐改进，非缺陷报告），留观；#7072/#7073（StemOamPrepPagedKv 除零/空指针崩溃）fail-loud；#7057/!15199（qliv2 golden scale 生成器精度，已修 merged）测试资产域。
- **hccl !3856**（AIV 归约跨卡写→读）昨日已留观，本批无新结论。
- **msmodelslim/mssanitizer/triton-ascend/ge/runtime**：本窗口新增均为 feature/文档/CI/构建域或 RFC（msmodelslim #671 CI 过滤、mssanitizer #244 peeraccess RFC、triton-ascend 27 PR 全为 feature/refactor、ge !5740 JIT 切片图编译选项、runtime #1319 DVPP 跨设备重复注册 107000 fail-loud），无静默计算故障证据。

## 2. GitHub 常规 7 仓 + arXiv

- 认证状态：环境无 GITHUB_TOKEN（未认证）；search 走代理出口共享配额，请求间隔 ≥8 秒。21 组查询（7 仓 × correctness/silent wrong output/NaN，`is:issue created:>=2026-10-27`）全部 HTTP 200、无限流等待（累计等待 0）。候选正文经 core API 单点补拉（6+5 条）。
- 与既往判定比对：vllm#64704/#64855/#64106/#64326/#64517、sglang#47245/#47508、transformers#50195/#50204/#50212/#50216/#50224/#50051、triton#12506/#12555、flashinfer#7687/#7980/#8119、TRT-LLM#21385/#21124、tt-metal quasar/tt-robot 族（#63619/#63622/#63627/#63628/#63631/#63642/#63672/#63682/#63684、#62662=INF-2268、#62628、#62901=INF-2238、#63257、#63866、#63867=INF-2369 等）均沿用 10-24~11-02 判定，无新证据不改判。

### 新增条目（GitHub 常规来源 3 条）

| 编号 | 来源 | 机制摘要 |
|---|---|---|
| INF-2383 | [vllm#65325](https://github.com/vllm-project/vllm/issues/65325) | level-2 sleep 期间排队的请求在 /wake_up（无 tags）后立即执行、先于 reload_weights，用丢弃权重算出与 prompt 无关输出并 HTTP 200；分步唤醒可规避（#65324 同内容） |
| INF-2384 | [tt-metal#64021](https://github.com/tenstorrent/tt-metal/issues/64021) | SFPU expm1_cw_clamped k=128 时 two_k=+inf，r<0 得 inf+(-inf)=NaN：ttnn.celu 负 alpha 且 x/alpha∈(88.376,88.723) 带 fp32 返 NaN/bf16 返 +inf（torch 有限值或 -inf）；源码级+逐位对照 |
| INF-2385 | [vllm#65385](https://github.com/vllm-project/vllm/issues/65385) | GDN Triton/FLA chunk_scaled_dot_kkt_fwd 的 BK<K+num_warps>=4 配置（12/27）非确定错误，autotune 在 H20+triton 3.6 恰选中；prompt logprobs 每次差 1-2.3 nat/token，RL rollout 偶发 EOS/token0/NaN |

### GitHub 留观（本轮判定）

- **sglang#48230**（PD+Mooncake prefill/decode DSA 后端不同致 item_len 失配，每页错位错偏移写，GSM8K=0 或 transfer 报错）——机制清晰（源码定位+两端字节布局推演），但「GSM8K 0 分」端到端实测细节与失败比例在正文压缩段未完整展开；且混合后端配置属部署契约边界，留观一轮看维护者归类（配置校验缺失 vs 传输缺陷）。
- **flashinfer#8570**（b12x fused MoE nvfp4 oracle vs deep_gemm greedy 6-8/20 prompt 改写级分歧）——报告者自述「不声称是 bug，可能是两条 FP4 数值路径的预期差异，问题在契约」；按既往 #8284 同判留观。
- **flashinfer#8443**（packed MLA/KDA 核 0*NaN 掩蔽致 NaN 跨序列扩散）——机制方向重要（单序列 NaN 污染批内其他序列），正文含两核独立脚本+服务级佐证；但 trtllm-gen MLA decode 档仅有服务级证据无独立脚本，且与已收 vllm#64758（FlashKDA 单 NaN 扩散 !! 流）相邻。本轮先留观比对两者关系，下轮决定是否并证或新收。
- **flashinfer#8119**（B12xMoEWrapper 逐层 capture 后 eager run 返全 0，release 0.7.2.post1 无守卫）——main 已有 #5168 fail-fast 守卫，属「发布分支未携带修复」域（构建/发布类），对 0.7.2.post1 用户是静默全 0；留观 0.7.x patch 发布。
- **tt-metal#63940**（permute 2^32 tile-padded 元素 uint32 溢出全错）、**#63965**（DeepSeekMLP TP/EP 分片返回 per-chip partial，源码审读未上机）——机制证据强（#63940 n=2047/2048 位级对照；#63965 源码路径完整但 not reproduced on device yet）。tt-metal 近期高产出且已收多条同族，本轮先收数值面最直接的 #64021，这两条下轮择重（#63940 候选优先）。
- **tt-metal tt-robot 新批**（#63910 sharded softmax 挂起+错值、#63924 transpose 分片错值、#63683 slice_write RM 分片错值、#62655 精确倒数大参数退化、#62654 SFPU Float16 dest 读垃圾）——按 11-01 已判「tt-robot 批量族」政策留观，下轮择重；#62655 的 88,242 ULP 门控问题与 #64021 同属 SFPU 数值精度域，可并读。
- **transformers#50315**（ZeroShotObjectDetection zip 截断批图像+字符串标签逐字符分配）、**#50328**（AudioClassification 回归/多标签默认 softmax 毁预测值）——均为 pipeline 层 API 语义缺陷、四架构复现/真实检查点实测，机制确凿；但属 CV/音频管线而非 LLM 推理核心路径，本库范围边缘，留观（如用户确认多模态 pipeline 入库范围再收）。
- **vllm#65324** 与 #65325 同内容（已并入 INF-2383）；#64239 沿用 10-28 判定（正文无故障细节）。
- **TRT-LLM#21519**（bench 多轮统计只用最后一轮）评测工具域；**#21571** 性能征集帖。

### arXiv

- 2 组查询（"silent data corruption"+LLM 8 entries、"bit flip"+inference 8 entries，sortBy=submittedDate）HTTP 200 有效。无新论文：SDC 组 8 篇（2610.33475/2610.18557/2609.19743/2608.30769/2608.11034/2604.10390/2604.00726/2502.12340）与 bit-flip 组前两条（2610.37033/2610.30798）均已在库或已判（10-12~11-02 各轮）；bit-flip 组其余 6 篇（p-DCIM/Rowhammer 攻击/Rad-hard NN/BARE-AI/结构剪枝容错/BLINK/JITterFlip）为攻击与容错硬件域，非推理计算故障，按既往口径不收。

## 3. 回访（近 7 日条目 + 轮巡旧源）

- **INF-2366**（block-FP8 TP 分片）：修复 #20653 仍 open（11-03 05:38 更新），源 #20652 仍 open——未合入，已注记。
- **INF-2367**（defer_block_free MTP）：源 #20677 仍 open（11-03 12:42 更新），无修复动向，已注记。
- **INF-2372**（310P QK 溢出）：修复 #20595 仍 open（11-02 后无更新），已注记。
- **INF-2361**（retention_interval）：修复 #20546 仍 open（11-03 07:09 更新），源 #20537 open，已注记。
- **INF-2353**（block-size 失配）：源 #20550 仍 open（11-03 16:06 更新），未见修复合入，已注记。
- **INF-2354**（SwigluGroupQuant 越界写）：#20539/#20540 均 closed（11-01），关闭原因未核实，维持注记不再推进。
- **INF-2331**（ScatterNdUpdate）：修复 !13327 仍 open，已注记。
- 其余近 7 日源（#20582/#20600/#20609/#20622 closed 由重开帖接替/#20673/#20674/#20628/#20692/#7917/#7954/#8022 等）本批快照状态无实质变化（#20609 11-03 12:09 仍 open 无根因；MRV2 parity 帖维持）；**ops-math#4412（StdMeanCorrection 除零）11-03 closed**——closed≠已修复，未见修复 MR 关联，留观下轮核实关闭原因。
- 轮巡（更早入库）：INF-2259 源 ops-math#6910、INF-2254 源 #6872 连续第四轮在滚动窗口外，状态未核（已注记下轮按 API 单点核验优先）；INF-2260 源 #7640 本批 open 无变化。

## 4. 编目与视图

- 新写入 **13 条**（全部 pending_audit）：INF-2373 ~ INF-2385，编号连续。映射：

| 编号 | 来源 | 标题 |
|---|---|---|
| INF-2373 | vllm-ascend#20827 | block_table 残留旧块号致前缀命中请求乱码 |
| INF-2374 | vllm-ascend#20829 | QuaRot 旋转文件缺失静默用单位阵知识塌缩 |
| INF-2375 | vllm-ascend#20725 | BalanceScheduler 漏置同步 KV 装载标志 |
| INF-2376 | ops-nn#8006 | RoundOffNumber roundMode=6 未初始化残值 |
| INF-2377 | ops-nn#8007 | ForeachSin fp32 大参数 inf 越界值 |
| INF-2378 | ops-nn#8012 | ForeachNorm 非 1/2 阶静默返 L1/L2 |
| INF-2379 | ops-nn#8013 | foreach 幂族 NaN 指数返 1.0 |
| INF-2380 | ops-transformer#7051 | BSA 偏移 uint32 回绕精度失败 |
| INF-2381 | ops-transformer#7063 | ScatterPaCache 场景二第三段错偏移 |
| INF-2382 | mindspore#29710 | 字符串枚举跨枚举静默映射 |
| INF-2383 | vllm#65325 | level-2 sleep 排队请求错误输出 |
| INF-2384 | tt-metal#64021 | celu 负 alpha NaN 带 |
| INF-2385 | vllm#65385 | GDN FLA 预填充非确定错误 |

- 回访 update：INF-2366/2367/2372/2361/2353/2354/2331/2259 共 8 条注记（见第 3 节）。
- 训练库跨库查重：13 个拟收 URL 在 training_error_review `find` 全部 0 命中，本轮无 cross 关系（#7070 FA 反向确定性问题训练侧如收录以训练库为准）。
- `catalog_tool.py stats`：2385 条（active 2372 + pending_audit 13）；`build` 完成；`validate` ok:true（n_errors 0；n_warnings 62——均为 pending_audit 常规 warning，无存量新错误）。

## 5. 未覆盖范围与 API 账本

- **GitCode/AtomGit issue 评论线程未拉取**（无低成本批量入口），全部按正文判定；若评论中出现维护者定位/驳回可能需改判（本轮收录的 10 条 NPU 候选正文均已含自复现或源码定位，依赖评论改判的概率低）。
- **vllm-ascend PR 增量**（2 日窗口 225 条 PR）按标题过滤+重点正文抽查（#20828/#20830/#20696/#20810 已读），未逐条深读全部修复流正文；#20704/#20759 等 EPLB/调度域标题级留观。
- **GitHub 未认证 search**：21 组全 200 无限流（累计等待 0 秒，未动用 5 分钟等待预算）；候选正文按 search item+单点 core API 补拉判定，未拉评论线程。tt-metal tt-robot 新批 5 条（#63910/#63924/#63683/#62655/#62654）正文未逐条深读。
- **arXiv** 仅 2 组主题词（SDC/bit-flip），KV corruption/数值正确性主题词未单独跑（与既往结果集重叠度高）。
- **回访轮巡缺口**：ops-math#6910/#6872 连续四轮窗口外未核状态（GitCode 无免认证单点入口预算），下轮优先；#4412 closed 原因未核实。
- 本轮无任何候选因限流/配额只按索引判定；全部收录/留观判定均基于已读正文。

## 6. 结论

13 条新机制入库（待审计），8 条既有条目回访注记。本轮覆盖：NPU 22/22 源完整批次（908 条窗口内新建、强信号 46 全筛、约 20 条深读）+ GitHub 7 仓 3 关键词（21 组全 200、约 90 unique、3 新收）+ arXiv 2 组（无新论文）；昨日 vllm-ascend 分页缺口已由 2 日窗口批次回补。重点新面：block_table 残留块号跨页混读（INF-2373，hybrid KV/状态池共用回收栈下静默错值最重）、QuaRot 声明文件缺失静默单位阵（INF-2374，量化域部署态静默塌缩）、BalanceScheduler 同步 KV 装载时序（INF-2375）、foreach 族四连真机复现错值（INF-2376~2379）、GDN FLA 预填充非确定（INF-2385，RL rollout 直接受害者）。
