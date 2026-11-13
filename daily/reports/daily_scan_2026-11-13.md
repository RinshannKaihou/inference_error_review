# 每日扫描日报 2026-11-13（推理库）

## 0. Git 前置检查

- 分支 `main`，上游 `origin/main`；`git status --porcelain -uall` 为空；`git fetch origin main` 成功；HEAD 与 origin/main 一致（`f76bc196189153109fef62ba34fe07fdb5723e00`）。通过，开始扫描。
- Cron 前置脚本 NPU 批次完整：**22/22 源全部 complete，failures 为空**。窗口 `2026-11-11T22:00:12Z → 2026-11-12T23:00:12Z`。registry_hash `2f6422d00a3d93e14cefe3f2ae3eb409684dc34184969178ccdff1f581d8d94d`（22 源）。批次 generated 2026-11-12T23:00:12Z，本轮 01:43 UTC 使用，处于 3 小时复用窗内；实际采集止于 11-12T23:00，不声称覆盖到本次运行时刻。

## 1. NPU 新增来源（共享批次）

- 批次：`/root/.hermes/profiles/third/cache/fault_scan_sources/batch-20261112T230012798508-dpkqr9fa`。
- inference scope 滚动候选索引 **31,689 条**（19 个推理适用仓库，issues 10,647 + PRs 21,042；含历史滚动缓存）。逐仓账本 `fetched` 合计 **3,368** 条（issues+pulls 全量检索）；其中 `created_at` 落在本轮窗口内的新候选 **898 条**（issue 272 + PR 626），为本轮判定主体。
- 逐仓覆盖（均 complete）：Ascend/pytorch 414（issues 73 + pulls 341）、op-plugin 44、torchair 89、cann/ops-transformer 565（205+360）、cann/ops-nn 640（271+369）、cann/ops-math 260（66+194）、cann/ge 67、cann/runtime 90、cann/hccl 138、cann/hcomm 185、mindspore 26、mindformers 3、msprobe 27、mssanitizer 12、MindSpeed 15、MindSpeed-LLM 15、MindSpeed-RL 0、MindIE-LLM 0、msmodelslim 20、vllm-ascend 671、sgl-kernel-npu 17、triton-ascend 70。MindSpeed/MindSpeed-LLM/MindSpeed-RL 为训练域，不进入 inference 候选视图。
- 判定方法：898 条新项**全部标题过目**；523 条强关键词标题命中（精度/静默/错值/溢出/NaN/竞态/量化/KV/wrong 等），逐条读批次缓存正文或摘要，深读约 45 条正文；未为 GitCode/AtomGit 额外拉 API。

### 新增条目（NPU/GitHub-NPU 来源 14 条，待审计）

| 编号 | 来源 | 机制摘要 |
|---|---|---|
| INF-2521 | [ops-nn#8663](https://gitcode.com/cann/ops-nn/issues/8663)、[#8674](https://gitcode.com/cann/ops-nn/issues/8674)、[#8685](https://gitcode.com/cann/ops-nn/issues/8685)、[#8688](https://gitcode.com/cann/ops-nn/issues/8688)、[#8702](https://gitcode.com/cann/ops-nn/issues/8702)、[#8736](https://gitcode.com/cann/ops-nn/issues/8736)、[#8746](https://gitcode.com/cann/ops-nn/issues/8746) | 无 `op_host/op_api`、由 op def 自动生成的 aclnn 接口回写时忽略出参 stride：非连续出参两段返回 0/0，结果按连续布局写 storage 前缀，一半位置留哨兵；资料标 √ 的静默错值，标 × 的该拒未拒（不返回 161002）。`AddRmsNormCast` 显式 `.AutoContiguous()` 也不生效；`TopKTopPSample` 64 个有效值只剩 32。跨训练库同机制 TRN-0755（cross 回显已核对标题）。 |
| INF-2522 | [ops-nn#8729](https://gitcode.com/cann/ops-nn/issues/8729)、[#8743](https://gitcode.com/cann/ops-nn/issues/8743) | AddRmsNorm 系空 tensor 路径返回成功但不写输出：DynamicQuantV2 在 y2Out 为空时第一段早退、不看 outputMask，mask 路由的有效 xOut/scale/y1Out 全部保持 canary；AddRmsNorm 在 x1 非空 + gamma 为空时非 regbase 校验缺失、numCol=0，yOut/xOut/rstdOut 32768 元素全未写。跨训练库 TRN-0670（cross 回显已核对）。 |
| INF-2523 | [ops-nn#8656](https://gitcode.com/cann/ops-nn/issues/8656) | MatMulV3 DAV_3510 fixpipe 模式按输入 dtype 选择：fp16/bf16→fp32 输出仍走 `ND_FIXPIPE_1_1` 且 splitM 未置位，未走 1V2/M 轴切分；Ascend950 bound 场景精度实测 74.78%。修复 PR !14100（issue closed，合入/发布未核实）按输出 dtype 选择并补 UT。 |
| INF-2524 | [ops-nn#8713](https://gitcode.com/cann/ops-nn/issues/8713) | RuntimeBankManager::Query 在 `rwlock_.rdlock()` 下命中 `tuning_cache_` 后直接 `erase`，多个 Query 读者可同时对 `unordered_map` 读写；并发命中可能返回错误缓存 tiling、查找异常或崩溃。修复 PR !14327（open）读锁只探测、升级写锁重查后擦除。 |
| INF-2525 | [ops-nn#8673](https://gitcode.com/cann/ops-nn/issues/8673) | ScatterMul/Div/Max/Min A5 归约：浮点折叠在 f32 域连乘，中间积溢出/冲洗后靠 `RescueNonFiniteSeg` 重算；救援只在最终值落 dtype 边界带时触发，中间冲洗而最终回摆的元素不救（L1_736 实测 43 个元素相对误差 0.3%~90%），宽切片 3.16s vs 搬运下界 134µs。var→INF-1978（共享 scatter_reduce_common_sort.h 折叠/救援路径）。 |
| INF-2526 | [ops-nn#8738](https://gitcode.com/cann/ops-nn/issues/8738) | FastHadamardQuant 分组参数形状未校验：host 用 group_size 算 groupsPerRow，却不校验 scale/offset 的 rank、组数、batch 维；x [1,128]、group_scales [1,1]、group_size=64 时 kernel 读 scales[1] 越界，量化输出用越界 scale。修复 PR !14357（open）。 |
| INF-2527 | [ops-transformer#7469](https://gitcode.com/cann/ops-transformer/issues/7469) | MoeInitRoutingV3/V4 大 H 分块路径取 smooth scale 漏 `expertIdx*H` 偏移，所有专家读第 0 行 scale，expandedX 精度失败；V3/V4 共用 kernel。修复 PR !15813（merged 2026-11-12，199/199 回归）。var→INF-1588（同属量化 scale 按专家/行取址错位）。 |
| INF-2528 | [msmodelslim#694](https://gitcode.com/Ascend/msmodelslim/issues/694) | FlatQuantOnlineWrapper 开启裁剪后，用展平 [B*S,H] 算出的 h_min/h_max（[B*S,1]）直接 clamp 三维 hidden_states：[2,3,4] 抛 RuntimeError，[2,1,4] 广播成 [2,2,4]，静默把序列维 1→2。修复 PR !1173（open）保留激活 shape。 |
| INF-2529 | [hcomm#1278](https://gitcode.com/cann/hcomm/issues/1278) | 单机 16 卡 910C、`HCCL_OP_EXPANSION_MODE=AIV`、等分 AllToAllV：仅 rank6/7 完成，14 rank 的 AIV kernel 不退出、AICore 100%，600s 后 EI0006/ERR00100；同 CANN so 换 driver 26.1/FW 9.0 正常，锁定 driver 26.2.rc1/FW 9.2.0.2.200。 |
| INF-2534 | [ops-transformer#7470](https://gitcode.com/cann/ops-transformer/issues/7470) | RoPE/RoPE Grad 的 int64 `mode` 在 tiling 按 char/uint32/int32 窄类型读取，高位丢失：mode=2^32 被当 mode=0 放行执行，两段返回 0，语义静默错误。修复 PR !15834（open）统一 int64 读取并做范围校验。 |
| INF-2535 | [sgl-kernel-npu#1025](https://github.com/sgl-project/sgl-kernel-npu/pull/1025) | DSpark `build_out_tokens_triton` 只覆盖 bonus 位，拒绝的 draft 后缀未清零；下游 `compute_spec_logprobs` 不按 correct_len 裁剪，把每个输出位置当词表索引，残留越界 token（如词表 100 时 101）触发 gather 越界或错 logprob。修复 PR #1025（open），上游 SGLang #37472 已 merged `c4b6338f`。 |
| INF-2536 | [torchair#969](https://gitcode.com/Ascend/torchair/issues/969) | inductor_npu_ext 的 ascendc（AF/AscIR）后端 rand/randn 不消费 seed/offset：AscIR Rand/Randn 固定 `philoxKey{0,0}`/`philoxCounter{0,0,0,0}`，`asc_ops.load_seed` 返回常量 0，seed 成死边；每次 kernel 得到同一随机序列，`torch.manual_seed` 不生效，dropout/randn 失去随机性。修复 PR #4005（open）。 |
| INF-2537 | [ops-math#4726](https://gitcode.com/cann/ops-math/issues/4726) | aclnnRoll 的 `l0op::Roll` 在 `RollAiCore` 返回 null（加算子失败）后仍返回已分配的 rollOut，`aclnnRollGetWorkspaceSize` 只判空：workspace 准备成功但 Roll 未下发，两段接口返回成功、输出保持原值/未初始化。 |
| INF-2538 | [hcomm#1279](https://gitcode.com/cann/hcomm/issues/1279) | `HcclRankGraphGetLayers` 的 netLayers 取本 rank 层集合、netLayerNum 却取整个通信域非空层总数；非对称组网下 netLayerNum 大于列表元素数，cluster_monitor/my_rank/coll_comm 按数量遍历读越界、得到垃圾层号。issue closed，修复取 levels.size()；同步 PR !6618 open。 |

### NPU 留观/不收（本轮判定）

- **3 次重交上限转留观**：ops-nn **#14313**（DynamicQuantV2 group_index 未覆盖全部 rows 时 GM 越界读：`GetSmoothIndex` 兜底返回入参 smoothIndex、调用点传 smoothIndex+1，smoothOffset 可达 groupNum 并逐行自增，SmoothCopyIn 从 smooth_scales 尾部外读 12B、groupIndexGm 越界；950PR 957b + CANN 9.3.0 + mssanitizer 单例 369 条 illegal read，修复兜底返回 groupNum-1 后 369→0。条目 JSON 因 cause.behavior 长度校验差 4 字连续 3 次重交未过，证据链完整记录于此，下轮可重提）。
- **留观（机制明确、影响未实测/触发窄）**：ops-nn **#8655**（MapIndex infershape 在非 arch35 芯片误用 24000/256 上限，x 长度 400~24000 可放行后越界；修复 !14271 open）、**#8710**（binary_cross_entropy/smooth_l1_loss 缺出参最大 8 维校验，仅潜在 UB）、**#8732**（AddRmsNormDynamicQuantV2 INT4 奇数尾轴未拒绝，但输出完整写出、无错值证据）、**#8676/#8719/#8734**（LayerNormV3 CANN Judge 用例失败/阈值口径，判题资产）、**#8659**（公共 InferShape UT 把 StorageShape 当 Tensor 越界读，测试资产）；ops-transformer **#7482**（KvRmsNormRopeCache InferShape rank 前置校验缺失、`GetDim(-1)` 哨兵可能写成 c_kv 尾维，端到端后果未证实）、**#7481**（lightning_indexer_v2 PA_BBND block_table 顺序步进 + 可选输入缺席触发 AIV 越界、整卡挂死，属 fail-loud/挂死）、**#7450**（FIA TND 缺失/空 seq len 表 Host tiling UB，修复 !15789，fail-loud）、**#7462**（GroupedMatmulFinalizeRoutingV2 开确定性后 AI Core 507015，fail-loud）、**#7456**（moe_init_routing_quant_v2 Ascend350 UB 越界，执行报 507035）；ops-math **#4725**（整型 alpha 超值域按补码截断，极端参数、非 LLM 主路径）、**#4720**（split_v 大 num_split 无符号回绕/UB 越界，图模式 num_split≥31745 可达性窄）、**#4721**（TopKV2 arch35 索引 Adds 越界与 S→V/S→MTE2 同步缺失，输出影响未实测）、**#4710**（BroadcastUbSch 末维 65536 除零 UB，未真机验证）；hcomm **#1283**（NDA TypicalQp 未初始化，可能非预期 UDP 源端口/建链失败，频率环境依赖）；hccl **#955/#957/#958**、hcomm **#1280/#1282/#1284-1289**（PR 合入后遗留检视意见跟踪，未确认运行时故障）；vllm-ascend **#21667**（SparseKVOffload 端点硬编码 8500+DP index，同 host 两服务端口冲突，源级确认但两服务启动失败未复现）、**#21684**（DSpark target head BF16 logits 致 tie 翻转、接受长度 1→0，精度口径待定）、**#21742**（Qwen3.6 batched MTP 非 batch-invariant，同批相同 prompt 第 10-22 token 分叉，根因未定位）、**#21730**（Qwen3.8 Mamba state/PLE/conv 输入三项修复 PR，仅摘要，无失效量级）。
- **训练域（移交训练扫描/已有训练条目）**：ops-nn **#8650**（ForeachDiv arch35 FTZ 次正规除数冲零→±Inf，训练库已收 TRN-0746，推理面不重复）、#8648（InplaceApplyAddSign 质量加固）；Ascend/pytorch **#6059**（put_ 负索引静默不写、take 梯度全零，训练库已收 TRN-0752）、#6073（DVM BatchNorm 反向梯度不一致）；ops-math **#4706**（InvGrad 广播/INT8/特殊值多缺陷）；msmodelslim #696（QAT 数据集 RFC）。
- **fail-loud/契约/文档/测试资产，不占故障编号**：ops-nn #8651/#8658（conv2d load3d/groups，closed，非 LLM 主路径）、#8670（头文件同名）、#8695/#8696/#8697/#8728/#8730/#8733/#8744（AddRmsNorm 系错误码/文档/该拒未拒）、#8709（BNTrainingReduce golden）、#8712/#8717/#8721/#8727/#8733（文档/输出名/公式）、#8715（lp_loss 失败返回传播，已合入）、#8718/#8739/#8741（需求/三方参考）、#8649/#8653/#8654（文档/性能）；ops-transformer #7446（UT 平台状态污染）、#7448/#7451/#7459/#7463/#7466/#7475/#7479/#7485/#7487（需求/文档/性能/Proto 合同）；cann/ge #718/#719（文档）；runtime #1363/#1365/#1367（防御性校验/文档）；op-plugin #706（npu_dynamic_block_mx_quant 动态 shape 追图被阻断，编译期 fail-loud）；Ascend/pytorch #6053/#6057/#6058/#6064/#6066/#6071/#6072/#6074/#6076（配置/特性/文档/工具/失败）；torchair #971（PyTorch 2.15 测试失败）；vllm-ascend #21641/#21644/#21743/#21744/#21756/#21760 及 #21640/#21665/#21719/#21729/#21732/#21749/#21753（RFC/Feature/CI/性能/启动 fail-loud；#21744 的 cudagraph_mm_encoder assert 被吞成 hang，修复 #21762 open）；triton-ascend #3130（RFC）；msprobe #690（asan container-overflow，工具域）、#692/#693（功能/文档）。
- **同族归并**：ops-nn 非连续出参族 #8663/#8669/#8674/#8675/#8677/#8679/#8680/#8681/#8682/#8683/#8684/#8685/#8686/#8687/#8688/#8689/#8691/#8693/#8698/#8699/#8701/#8702/#8704/#8706/#8736/#8737/#8740/#8746 归入 INF-2521（#8691 已撤回误报，仅作排除证据）；AddRmsNorm 空 tensor 两单归入 INF-2522；#8694/#8731/#8745 归入 INF-2390（epsilon 同一窄化机制，已补 evidence/note）。

## 2. GitHub 常规 7 仓 + arXiv

- 认证状态：环境无 `GITHUB_TOKEN`（未认证）；search 间隔 8 秒，全部带 `created:>=2026-11-06`。
- **API 账本**：search 7 仓 × 3 关键词组 = **21/21 HTTP 200**，`per_page=100` 单页，total 均 <100，无分页未覆盖。计数（correctness / silent wrong output / NaN）：vllm 15/14/5，sglang 16/4/9，transformers 4/3/2，TRT-LLM 0/0/0，triton 0/0/0，tt-metal 10/9/5，flashinfer 2/4/5。**core API 额度 60/60 已耗尽（403）**，8 个 issue 正文改用 GitHub HTML 读取（#67313/#50037/#50669/#9396/#9372/#9316/#64740/#64840），另以 HTML 核对 #65325/#50599/#9068 状态；评论线程未拉。
- 与既往判定比对（维持，无新证据不改判）：vllm #67065=INF-2513、#66886=INF-2498、#66144=INF-2458、#66801=INF-2500、#66610=INF-2487、#66437=INF-2486、#66074=INF-2455、#66211=INF-2457、#66573 留观、#66435 fail-loud、#67334/#67019/#66942/#66782/#66794 RFC/Tracking、#67183 启动失败、#67313 新收 INF-2533；sglang #48964=INF-2447、#49657=INF-2503、#49185/#49198=INF-2464、#49881=INF-2518、#50037 新收 INF-2532、#50010/#49944/#49536/#49336/#49324/#49327/#49257/#48766 Feature/Perf；transformers #50440=INF-2465、#50484=INF-2470、#50599=INF-2520（本轮 closed）、#50612/#50545 训练域、#50622 训练域、#50669 留观（并发 chat template 丢 assistant mask，SFT 预处理面）；flashinfer #9333=INF-2514、#9343=INF-2519（closed）、#9068=INF-2456（本轮 closed）、#9279/#8946/#8897/#9372 新收 INF-2531、#9396 新收 INF-2530、#9316=INF-2501；tt-metal #64834=INF-2480、#64811=INF-2481、#64411=INF-2482、#64678=INF-2483、#64827=INF-2484、#64826=INF-2485、#63940=INF-2400、#64740=INF-2505、#64840=INF-2502、#64851/#64845/#64648/#64637/#64600 既有 _bw/审计族；TRT-LLM、triton 0 命中。
- **新增条目（GitHub 常规来源 4 条）**：INF-2530 [flashinfer#9396](https://github.com/flashinfer-ai/flashinfer/issues/9396)（JIT silu_and_mul 只校验末维偶数与输出 shape，不校验行列 stride；行 stride 视图按连续行起点读写，错位/写 padding）；INF-2531 [flashinfer#9372](https://github.com/flashinfer-ai/flashinfer/issues/9372)（GDN prefill packed varlen 最后不完整 chunk 用乘 0 mask 屏蔽下一序列首行，NaN/Inf×0 仍 NaN，k/v 的 NaN 污染前一序列输出与 final state）；INF-2532 [sglang#50037](https://github.com/sgl-project/sglang/issues/50037)（CuTe DSL GDN 逆矩阵 Newton–Schulz 每次 MMA 转 BF16，病态 I+A 下 cond×2^-8 误差放大，65536 token 第 18 层 NaN 并沿 chunk 传播）；INF-2533 [vllm#67313](https://github.com/vllm-project/vllm/issues/67313)（tower/connector LoRA 每步只建一份映射，每个 kernel 都从 index 0 读 mapping[:token_nums]，单步多次 encoder 调用拿错 adapter/越界索引；var→INF-2500）。
- **本轮 GitHub 留观**：#50669（并发 render_jinja_template 共享 AssistantTracker 丢 generation mask/ValueError，SFT 预处理面，未测推理服务并发）、#50037 已收、#64740/#64840 已在前日收录、#65437 OpenVLA 赏金（动作正确性，非 LLM 推理主路径）、#64698 Fabric 测试基础设施。
- **arXiv**：2 组短语查询（"silent data corruption"+LLM、bit flip+inference，sortBy=submittedDate）HTTP 200 有效；返回 8 + 20 篇，全部 ID（2610.33475/2610.18557/2609.19743/2610.37033/2610.30798/2610.27621/2609.06781/2608.29745 等）均见于既往日报，**无新论文**。

## 3. 回访（近 7 日条目 + 轮巡旧源）

- **INF-2390**（RmsNormDynamicMxQuant epsilon 静默归零）：补 evidence #8694（AddRmsNormDynamicQuant）、#8731（V2）、#8745（AddRmsNorm）+ note。三单同为 epsilon 声明 double、proto 注册 Float、执行前窄化 fp32：#8694/#8731 的 eps=1e-50 在执行段 561103；#8745 在 A2 上 1e-50 与 0.0 输出逐字节一致、全零行 NaN，-1e-50 因非 regbase 首段无 epsilon 校验被放行（-1.0 报 561103 而非 161002）。
- **INF-2520**（transformers #50599 Inkling BF16 router tie）：issue 已 closed，stateReason=COMPLETED（关闭/合并日期未取得）；修复内容与发布版本未核实。已 note。
- **INF-2456**（flashinfer #9068 Rope neox head_size 静默错值）：issue 已 closed，stateReason=COMPLETED；修复 PR 与发布版本未核实。已 note。
- **INF-2383**（vllm #65325 level-2 sleep 排队请求）：仍 open（GitHub HTML 2026-11-13 核对）；core API 403，评论/关闭说明未读。已 note。
- 另核对近期来源：ops-nn #8390（INF-2461）仍 open（batch updated 11-11T06:44）、runtime #1357（INF-2496）closed、ops-nn #8254（INF-2422）closed，与昨日注记一致，无新证据不改判。

## 4. 未覆盖范围与缺口

- GitCode/AtomGit issue/PR **评论线程未拉取**（无低成本批量入口）：#8663 拆单后的各接口修复进展、#8674 AutoContiguous 讨论、#7469 修复评审、#1278 现场补充、#8713/#8738 修复 PR 审查均只读正文/摘要。
- GitHub **core API 60/60 耗尽**，8 个正文靠 HTML 读取；评论线程未读，#65325 关闭说明仍缺；search 三词组之外关键词面（量化/KV/logprob 等）依赖既往轮次与 NPU 批次，未逐日重扫。
- NPU 898 条新项全部标题过目；523 条强关键词命中读标题/正文筛选，深读约 45 条；其余 375 条弱命中（test/docs/CI/feature/perf/重构）按标题归档，未逐条深读。626 条 PR 中 363 条关键词命中仅看标题/摘要，未逐条读 diff。
- 部分候选仅按索引/摘要判定：#8655 MapIndex、#8710、#8732、#7482、#4720/#4721/#4710、#21667/#21684/#21742、#21730 等，均在上文标注“留观”及原因，未声称已验证。
- ops-nn **#14313**（DynamicQuantV2 group_index 越界读）因条目 JSON `cause.behavior` 长度校验连续 3 次未过转留观，未入编目；证据链完整记录于 §1，为明确缺口，下轮优先重提。
- 训练库同日已收 TRN-0746（#8650）、TRN-0752（#6059）、TRN-0755（#8663 族）、TRN-0670（#8729/#8743）；本库按推理面单独立目/不重复，cross 回显均已核对标题。

## 5. 小结

18 条新条目入库（待审计）：NPU/GitHub-NPU 14 条（INF-2521~2529、2534~2538）+ GitHub 常规 4 条（INF-2530~2533），其中 3 条 var 关系（INF-2525→INF-1978、INF-2527→INF-1588、INF-2533→INF-2500）、2 条 cross 关系（INF-2521→TRN-0755、INF-2522→TRN-0670），回显目标标题均已核对；4 条既有条目回访更新（INF-2390 补 3 条同机制证据，INF-2520/2456/2383 状态注记）。重点新面：**自动生成 aclnn 接口出参 stride 静默半写/该拒不拒**（跨推理算子族）、**AddRmsNorm 空 tensor 早退全不写**、**MatMulV3 fp16→fp32 fixpipe 精度 74.78%**、**Scatter A5 折叠救援漏中间下溢（相对误差 0.3%~90%）**、**MoeInitRouting 大 H 漏专家 scale 偏移**、**FlatQuant 三维裁剪静默扩序列维**、**910C AIV AllToAllV 16 卡挂死**、**DSpark 拒绝草稿 token 未清零致 logprob gather 越界**、**AscendC rand/randn 忽略 seed**、**aclnnRoll 启动失败当成功**、**GetNetLayers 非对称组网越界读**、**FlashInfer silu_and_mul 行 stride 错值**、**GDN prefill NaN 跨序列污染**、**SGLang CuTe GDN 逆矩阵 BF16 截断长文 NaN**、**Tower LoRA 单步多次编码映射错配**。留观重点：#14313（DynamicQuantV2 越界读，3 次重交上限转留观）、#8655/#8710/#8732/#7482/#4720/#4721/#4710/#21667/#21684/#21742/#21730/#50669。

## 6. 编目与校验

- `catalog_tool.py stats`：**2,538 条**（active 2,520，pending_audit 18）；故障 2,521 / 结论 8 / 方法 8 / 指针 1。
- `build`：ok，生成 `views/CATALOG.zh.md`、`views/fault_catalog.csv`、`views/fault_catalog.xlsx`。
- `validate`：`ok: true`，errors 0，warnings 65（既有），pending_audit 18。
