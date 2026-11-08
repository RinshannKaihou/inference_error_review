# 每日扫描日报 2026-11-08（推理库）

## 0. Git 前置检查

- 分支 `main`，上游 `origin/main`；`git status --porcelain -uall` 为空；`git fetch origin main` 成功；HEAD 与 origin/main 一致（d776fd28663052217a1a2b8a8205393f8118e264）。通过，开始扫描。
- Cron 前置脚本 NPU 批次完整：**22/22 源全部 complete**，无 failures。窗口 2026-11-07T22:01:00Z → 2026-11-08T23:01:00Z。registry_hash 2f6422d0…（22 源，与昨日一致）。

## 1. NPU 新增来源（共享批次）

- 批次：`/root/.hermes/profiles/third/cache/fault_scan_sources/batch-20261108T230100790164-srjxsok2`，generated 2026-11-08T23:01:00Z。inference scope 候选 27450 条（updated 滚动窗口全量），覆盖 19 个 inference 适用仓。
- **本轮新项判定窗口**：created ≥ 2026-11-07T22:01:00Z（昨日批次截止），共 **129 条**（全部条目标题人工过目，含全部 PR）。重点候选全文深读 ~22 条（含窗口外漏项 2 条补判）。
- 近 14 天日报完整 URL 查重：本轮拟收 11 个来源 URL 全部 0 既往判定命中；编目 `find` 查重 0 命中；训练库跨库查重 14 URL 全部 0 命中（无 cross 关系）。
- 机制级比对：#4619/#4620 与已收 INF-2296（负除数乘积溢出）、INF-2205（大数塌 0）、INF-2057（int32 差一）同为 Remainder/FloorDivide 域但缺陷机制不同（真商贴近整数时先舍入成整数再取 floor 的分支选择错误，fp32 入口、张量/标量全覆盖），分立编号；#5954 与昨日 INF-2450/#5938 同属 torch_npu 编译器归约掩码缺陷密集期但缺陷位置不同（Welford epilogue 无掩码 vs 分核尾块 mask 删除），分立；#8390 与 INF-1659（IndexAdd arch22 索引校验）同算子不同机制（出参版对连续 self 的别名写）；sglang#49185/#49198 与 INF-2457（vllm#66211 行并行 scale 切分错配）同为量化权重加载错配但机制不同（router 构建时量化配置未传递导致权重跳过/无 scale）。

### 新增条目（NPU/GitHub-NPU 来源 5 条）

| 编号 | 来源 | 机制摘要 |
|---|---|---|
| INF-2459 | [ops-math#4620](https://gitcode.com/cann/ops-math/issues/4620) | aclnnFloorDivide fp32 真商贴近整数（相对差 ~1e-7）时 vdiv→floor 链路先把商舍入成整数再取 floor，结果偏差 1（0.5/0.1f→5 应 4）；张量/标量全入口（初版「标量正确」系除数位型未控制变量）；资料 :403 推荐的 FLOAT32 精度路径本身错误；A2/CANN 9.3.0 IEEE-754 位型实测 |
| INF-2460 | [pytorch#5954](https://gitcode.com/Ascend/pytorch/issues/5954) | triton_experimental 后端 Welford epilogue 无归约掩码：R0_BLOCK=64>内层 extent=8 时 weight/bias 读偏移 119 越界、lane 8..63 散写进行首相邻行，test_var_mean 86.5% 元素失配（max diff 23074）；R0_BLOCK=8 时 0 失配、default 后端正常；R0_BLOCK_HINT 被当候选非硬约束；修复方向已给未实施 |
| INF-2461 | [ops-nn#8390](https://gitcode.com/cann/ops-nn/issues/8390) | aclnnIndexAdd 出参版（签名含 out）在 self 连续时把结果原地写回输入 self 再拷 out，非连续时落在 Contiguous 副本上不改写——「连续改、非连续不改」证明是实现分支；资料声明 self 为输入；torch.index_add 对照不改写；字节级判据 |
| INF-2462 | [ops-math#4619](https://gitcode.com/cann/ops-math/issues/4619) | aclnnRemainderTensorTensor fp32 真商贴近整数时 floor 少取 1，余数由 other-ε 变 ≈0，绝对误差=|other|（100%）；IEEE-754 位型精确构造实测（真商 -6.000000157）；超出资料 fp32 容差口径 max(1e-2,32ULP) |
| INF-2466 | [msmodelslim#667](https://gitcode.com/Ascend/msmodelslim/issues/667) | KV cache per-channel 校准 ActPerChannelMinmax.forward 每次 reset 观察器再 update，跨批幅值统计被清零，deploy 固化末批 scale（幅值 100→1 两批 scale 从 0.787 变 0.0079）；官方 dynamic_cache 配置即 per_channel，部署后超末批幅值的输入即饱和截断；**10-31 批次漏项本轮补判**（同批 #664/#666 已收 INF-2326/2333） |

### NPU 留观（本轮判定）

- **「空输入早退+非空输出未写」族新批 +5**：ops-nn #8387（SoftmaxCrossEntropyWithLogits C=0 早退 loss 未写）、#8389（EmbeddingBag 空 indices output 未写）、ops-transformer #7237（MoeFusedTopk expertNum=0 短路全部约束校验且 y/indices 未写）、#7238（LightningIndexerKLLoss S=0 早退标量 loss 未写）、#7236（MoeTokenPermuteWithRoutingMapGrad 空 hidden 早退 probsGradOut 未写）——与已收代表（INF-2246/1794/2126/2189/1657/2391）及 11-05/11-07 族留观同型异算子，正文均含 canary 实测+源码定位，按族留观。
- **「输出 shape 不校验→越界写声明区外」族 +6**：ops-transformer #7239/#7240/#7241/#7242（MoeTokenUnpermute 四姊妹：InferShape 无条件覆写 out hidden 维，多写 8~16 字节越界、canary 被覆盖、两段返回 0）、#7244（NormRopeConcat 第一段纯透传无任何校验）、#7243（错误码错位 561103 vs 资料 161002）——与 10-21 #7100（TurboQuant，已 closed）、11-04 #8121/#8122/#8123 infershape 整改族同型，按族留观。**族证据密度持续走高（同日 5 连发自述同族），若维护者确认系统性整改或出现非退化输入实例，下轮可将族代表（#7239，canary+源码双证）单收**。
- **错误码域**：#7245（FlashAttentionScoreV4 softmaxOutLayout 非法枚举静默按默认执行——含静默接受面但当前两合法值行为一致、无数值差异证据，随错误码族留观，若后续版本增加第三合法值即成静默错值）、#7235（MoeTokenPermuteWithRoutingMap 错误码 561103 vs 161002）——错误码契约域按既往口径不占号。
- **pytorch #5950**（Inductor detect_flattened_dims 未注册轴遗留 size_only 条目，CPU 索引分析回归，3/5 失败）——主机侧分析函数缺陷，未见 NPU 设备端静默错值实测，随修复 PR !49261 留观。
- **pytorch #5951**（make_graphed_callables backward 返回 b.detach() 共享静态梯度缓冲，连续微批梯度覆盖 w.grad 8 vs 10）——CPU replay 替身验证（NPU capture 未跑），机制清晰但设备端未证实，随修复 PR !49262 留观（含 RUN_NPU_GRAPH_TESTS 显式集成用例未运行）。
- **pytorch #5952**（graphed callables 常量输出未 mark_non_differentiable，bridge 执行无数据贡献反向）——同 #5951 批次 CPU 替身验证，训练侧梯度域为主，随 !49263 留观。
- **pytorch #5953**（A3 Host pin_memory 207001/EL0018 64B/2MiB 分配失败链，DSpark 服务关联）——资源耗尽域 fail-loud（torch.OutOfMemoryError 响亮抛出），正文为分配链取证+源码确认非每请求新分配，无静默计算损坏；留观（与 #5900 上游讨论关联）。
- **pytorch #5955**（TE 后端不产生 autotune 缓存工件）——性能/缓存契约域（二次编译全 miss），结果正确；留观。
- **hcomm #1247**（PR#6511 ABI v5 升级将 v4 描述符 sqDepth/scqDepth/ubAttr 队列深度静默丢弃，1 CRITICAL 检视意见跟踪）——ABI 兼容域静态检视，未同步缺陷实测后果；与 #1243/#1244/#1245 检视跟踪域留观。
- **ops-math #4625**（CANNJudge StridedSliceGrad 隐藏点恒 Runtime Error、可见点 5/5 Pass）——判题平台咨询帖，隐藏用例不可见无法定位；留观。**#4626**（Lgamma/Digamma 资料公式绝对值位置写反）文档域。
- **ops-math !7216**（pow/add tiling 除法/溢出保护、dtype 契约、图推导静默放行 15 项修复）——白盒加固修复流，含「图推导静默放行」修复项但正文为修复清单无独立复现矩阵；随修复流留观，如修复说明披露具体静默放行后果再评。**ops-nn !13909**（NonZero -0.0 判零错位修复）、**!13900**（Unique valueOut dtype 拒绝）、**!13901**（2d conv pass 修复 merged）——修复流，#8386（Unique valueOut COMPLEX64 放行）对应 !13900，按防御校验域留观。
- **vllm-ascend #21270**（_dspark_swa_indices_kernel AI Core Error 95 invalid GM address abort）——fail-loud 设备错误域，修复 PR #21275 同日挂出（guard host tensors）；留观。**#21273**（csrc 构建随机失败+陈旧 opc 锁毒化重试，构建域）+ 修复 #21274——构建/发布域不占号。
- **torchair #950**（RNG handler 查询 NameError，pylint 压制诊断）fail-loud；**#951**（GE attention pse 输入被无条件 real_shift=None 丢弃——静默丢弃偏移输入但验证仅到图输入连线层面，无数值实测）——mock 验证域留观，待 NPU 数值证据。
- **文档/构建/共建域**（不占编号）：ops-math #4621/#4623（AcosGrad/ApproximateEqual 质量加固汇总）、#4622/#4624（build 文档）、!7212-!7230（docs 大批）；ops-nn #8388（JSON 重写残留字节）、#8391/#8392（build 文档）、!13895-!13915（docs/加固批）；ops-transformer !15498-!15504（docs+校验修复）；hccl !3947（stub）；hcomm #1248/!6523（build 文档）；mindspore #93379/#93380（**两笔 PTA 对齐修复 PR**：mint.all uint8 输出 dtype（bool 应 uint8）、mint.addcdiv 混合 dtype 输出与梯度 dtype——均带最小复现+PTA/MS 对照表，属推理域可感知的 dtype 语义缺陷，但以修复 PR 形态提交且 MindSpore 侧影响面未量化，随 PR 留观，若单独 issue 化再评）；pytorch #5949（硬件解耦屏蔽任务）、#5956（性能）、#5957（版本咨询）；vllm-ascend #21263-#21284（Doc/Feature/Perf/CI 流）。

## 2. GitHub 常规 7 仓 + arXiv

- 认证状态：环境无 GITHUB_TOKEN（未认证）；search 请求间隔 ≥8 秒。21 组查询（7 仓 × correctness/silent wrong output/NaN，`is:issue created:>=2026-11-01`）：**18/21 组 HTTP 200**（vllm 19/11/4、sglang 7/1/4、transformers 7/2/3、TRT-LLM 3/1/0、triton 0/0/0、flashinfer 9/2/5）。
- **tt-metal 3 组全部失败——仓库全域不可达**：`repos/tenstorrent/tt-metal` API 404、web 404、`search/repositories?q=tt+org:tenstorrent` 无结果、org 公共仓列表空（昨日 11-07 尚正常 21/21）。判定为仓库删除/转私有/改名的真实覆盖缺口，非限流（search 配额 10/10 未耗尽）；org 存在（HTTP 200）但无公共仓。**tt-metal 今日未覆盖，待仓库状态明朗后补查**。
- 与既往判定比对：vllm#65324/#65325/#65680/#65564/#65513/#65734/#65337/#65131/#65385/#66074（=INF-2455）/#66144（=INF-2458）/#66211（=INF-2457）/#66246、sglang#48660（=INF-2425）/#48230、transformers#50336/#50315/#50334/#50212/#50216/#50224、TRT-LLM#21519/#21385/#21571、flashinfer#8570（=INF-2399）/#8443/#8284/#8607/#8790（=INF-2108 补证）均沿用既往判定，无新证据不改判。

### 新增条目（GitHub 常规来源 3 条）

| 编号 | 来源 | 机制摘要 |
|---|---|---|
| INF-2463 | [sglang#48967](https://github.com/sgl-project/sglang/issues/48967)（11-06 留观本轮深读转收） | AMD gfx1151 MLA decode（Lk576/Lv512）ragged 批 ≥3 条序列跑满 64-split 时读未初始化/竞态内存：NaN 或错行（误差 ~4 vs bf16 1e-2），坏行逐 run 变化、8 seed 全坏；uniform 批与 GQA shape 全干净；双服务器 greedy 仅 4/64 一致（GQA 24-30/64）；竞态点未定位 |
| INF-2464 | [sglang#49185](https://github.com/sgl-project/sglang/issues/49185) + [#49198](https://github.com/sgl-project/sglang/issues/49198) | 量化 MoE router 权重静默不加载/无 scale 加载：Qwen3-MoE 仅 ModelSlim 传 quant_config、Mixtral/Llama4 全不传——compressed-tensors 量化 router 张量无参数可落被静默跳过（router 全 NaN、GSM8K 0.0%，修复后 85.7%）；FP8 codes 无 scale 拷入（router logits 相对误差 301~7394）；8 个公开 checkpoint 中招；#49198 补充 238 checkpoint 扫描与「按实际存储决定」修复约束；修复 PR #49187 open、维护者已认领 |
| INF-2465 | [transformers#50440](https://github.com/huggingface/transformers/issues/50440) | Step-3.7 need_fp32_gate=true 被忽略：官方实现 router matmul 前 cast FP32，transformers Step3p7TopKRouter 按 weight dtype（BF16）计算仅事后 upcast——随机权重 1024 token 中 38 个 top-8 专家集合与 FP32 参考不同；无 checkpoint 依赖 CPU 复现；#50002 明确保留 BF16 gate 未改 |

### GitHub 留观（本轮判定）

- **vllm#66290**（PP 等价性测试 RFC）测试基建域；**#66311/#65813**（Feature/RFC）；**#65112**（测试覆盖缺口）——不占编号。
- **transformers#50424**（GradientCheckpointingLayer 包 FSDP2 hooks、use_reentrant=True 静默丢梯度）——训练域，训练库范围；**#50476**（agents 工具调用边界检查 Idea）特性讨论。
- **flashinfer#8946/#8757/#8897/#8170**（Feature/Experimental/RFC 域）沿用既往判定。

### arXiv

- 3 组查询（silent data corruption+LLM、bit flip+inference、**KV cache+corruption 本轮新增**）全部 HTTP 200 有效。
- **新论文 1 篇转收**：KVGuard（[2611.06138](https://arxiv.org/abs/2611.06138)，2026-11-04）→ **INF-2467**（方法类）：PagedAttention 块式 KV 管理的内存安全缺陷类别（double-free/UAF/越界/未初始化读/权限违规，静默损坏不崩溃）+ 首个块语义级检查器；与本库已收 KV 域条目（INF-2424/2452/1430）方向吻合；检出效果本库未实测。
- 其余无新：SDC 组 TrainSDC/SCOUT/LLM-PRISM/Cauchy-Schwarz/Syndrome Decoding 已收或既往已判；bit-flip 组 Rowhammer/JITterFlip/BARE-AI/辐射容错为攻击与硬件容错域按既往口径不收；KV 组其余（Contiguity repair、Latent Agents Lie、fuzzing 等）为系统方法/安全域非本库故障编目对象。

## 3. 回访（近 7 日条目 + 轮巡旧源）

- **昨日 11 条（INF-2448~2458）批次快照核验**：GitCode/GitHub-NPU 来源全部仍 open 无实质变化（#691/#4607/#5938/#8366/#5943/#8372/#4616/#4617）；**#5946（INF-2453）已 closed**（修复 !49255 含 11-07 完整验证，合入未核实）——已 update --note；vllm#66144（INF-2458）仍 open、修复 PR #66146 无变化——已注记；vllm-ascend#21251（INF-2452）仍 open、修复 #21252 open——已注记。
- **轮巡旧源 3 条**：flashinfer#8284（11-02 accuracy 观察，open 无评论变化，随机解码混杂解释未排除维持留观）；vllm#66074（INF-2455）open；vllm#66211（INF-2457）open（1 评论，未读正文——search 快照确认无 state 翻转）。
- **11-06 承诺「下轮优先」执行情况**：sglang#48967 本轮深读转收（INF-2463）✓；pytorch#5948（aclgraph wait_event 自锁）批次快照无更新仍留观；tt-metal #64678 本轮无法核验（仓库不可达）；ops-nn #8218 昨日已勘误作废；op-plugin #688 仍 open 无评论（11-07 判定维持）。
- **10-31 漏项补判**：msmodelslim #665/#667 在 10-31 批次窗口内创建但当日日报未逐条判定（仅收同批 #664/#666）。本轮补读：**#667 转收 INF-2466**；#665（FlatQuant 裁剪比例进 IR 后二次 sigmoid，macOS CPU 源码提取复现、NPU 端到端未验证）按「未在设备复现的候选」留观（与 11-01 AI 探针族同口径）。

## 4. 编目与视图

- 新写入 **9 条**（全部 pending_audit）：INF-2459 ~ INF-2467，编号连续。映射：

| 编号 | 来源 | 标题 |
|---|---|---|
| INF-2459 | ops-math#4620 | FloorDivide fp32 近整商结果偏差 1 |
| INF-2460 | pytorch#5954 | TE 后端 Welford epilogue 无掩码 mean/var 损坏 |
| INF-2461 | ops-nn#8390 | IndexAdd 出参版原地改写输入 self |
| INF-2462 | ops-math#4619 | Remainder fp32 真商贴近整数时余数差一个除数 |
| INF-2463 | sglang#48967 | AMD MLA decode ragged 64-split NaN/错行 |
| INF-2464 | sglang#49185+#49198 | 量化 router 权重静默不加载或无 scale 加载 |
| INF-2465 | transformers#50440 | Step-3.7 need_fp32_gate 被忽略路由错 |
| INF-2466 | msmodelslim#667 | KV per-channel 校准观察器每批重置 |
| INF-2467 | arXiv 2611.06138 | KVGuard：PagedAttention KV 块内存安全检查器（方法） |

- var/cross 关系：本轮无（同机制判定均为分立：不同算子/不同缺陷位置/不同栈）。
- 回访 update：INF-2452/2453/2458 共 3 条注记。
- 训练库跨库查重：14 个拟收 URL 在 training_error_review `find` 全部 0 命中，无 cross 关系。
- `catalog_tool.py stats`：**2467 条**；`build` 完成；`validate` 见下。

## 5. 未覆盖范围与 API 账本

- **tt-metal 全仓未覆盖**（仓库 404，删除/私有/改名待确认）——昨日已判定的 tt_dit/tt-robot 族今日无法复核状态；仓库恢复后需补查 2026-11-01 以来新建 issue。
- **GitCode/AtomGit issue 评论线程未拉取**（无低成本批量入口），全部按正文判定；本轮收录候选正文均含自包含复现/源码定位/对照实验。
- **GitHub 未认证 search**：18/21 组 200（tt-metal 3 组 422 资源不可搜）；正文单点补拉 5 条（#49185/#49198 detail+comments、#48967、#50440、#8284/#66211 状态）；评论线程仅拉 #49185/#49198。
- **窗口内 129 条全部标题过目**；PR 修复流（!7212-!7230、!13895-!13915、!15498-!15504 等大批 docs/加固）按标题+关键正文抽查，未逐条深读。
- **留观未深读正文项**（下轮优先）：pytorch #5951/#5952（CPU 替身验证的梯度域双胞胎，待 NPU capture 用例）、torchair #951（pse 丢弃的数值面）、msmodelslim #665（FlatQuant sigmoid，待设备复现）、ops-math #4625（CANNJudge 隐藏用例）。
- 本轮无任何候选因限流/配额只按索引判定；全部收录/留观判定均基于已读正文或批次缓存正文。无限流等待（累计 0 秒）。

## 6. 结论

9 条新条目入库（待审计），3 条既有条目回访注记，1 条 11-06 留观转收（sglang#48967 AMD MLA ragged NaN），1 条 10-31 漏项补判转收（msmodelslim#667 KV 校准观察器重置），1 篇新方法论文（KVGuard）。覆盖：NPU 22/22 源完整批次（129 条窗口内新建全过目、~22 条深读）+ GitHub 6/7 仓 3 关键词（18/21 组 200、3 新收）+ arXiv 3 组（1 新论文）。**tt-metal 仓库全域 404 为本轮最大缺口**（昨尚正常），tt_dit/tt-robot 族状态无法复核，待仓库明朗补查。重点新面：**SGLang 量化 MoE router 加载三连**（compressed-tensors router 静默跳过 GSM8K 0%、FP8 无 scale 路由错 301~7394 倍、修复需按 checkpoint 实际存储）；**CANN fp32 整除/取模近整商族**（FloorDivide 偏差 1 + Remainder 差一整个除数，同日双报、位型级实测）；**torch_npu 编译器静默错值第四日**（Welford epilogue 无掩码 86.5% 失配）；Step-3.7 need_fp32_gate 被忽略路由错。「空输入早退」族 +5、「输出 shape 越界写」族 +6（同日 5 连发），按族留观，族代表 #7239 下轮可单收。
