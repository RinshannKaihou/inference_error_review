# 每日扫描日报 2026-11-10（推理库）

## 0. Git 前置检查

- 分支 `main`，上游 `origin/main`；`git status --porcelain -uall` 为空；`git fetch origin main` 成功；HEAD 与 origin/main 一致（190af410bae8db782d1efa569b0c85b5ecdc9c21）。通过，开始扫描。
- Cron 前置脚本 NPU 批次完整：**22/22 源全部 complete，failures 为空**。窗口 2026-11-09T22:00:12Z → 2026-11-10T23:00:12Z。registry_hash 2f6422d0…（22 源，与 11-08/11-09 一致）。

## 1. NPU 新增来源（共享批次）

- 批次：`/root/.hermes/profiles/third/cache/fault_scan_sources/batch-20261110T230012796601-prufvjop`，generated 2026-11-10T23:00:12Z。inference scope 候选索引 29,865 条（滚动缓存全量：issue 10,040 + PR 19,825）。
- **本轮新项判定窗口**：created ≥ 2026-11-09T22:00:12Z，共 **925 条**（issue 245 + PR 680；GitCode 800 + GitHub-NPU 125），**全部条目标题过目**（强关键词标题命中 73 条全文筛读，重点候选深读 ~28 条正文，全部使用批次缓存正文，未额外拉 API）。
- 查重：本轮拟收/拟拒 NPU URL 全部执行编目 `find` 与近 14 天日报完整 URL 检索（早期裸数字 grep 命中 3076/1164/4658 为 2610.30798、#21164、#46584 等误匹配，已用完整 URL 复核排除）；训练库跨库查重 10 URL 全部 0 命中（无 cross 关系）。hccl !3914 命中 11-05 留观判定（见 §3）。
- 机制级比对：triton-ascend #3076 与已收 INF-2413（#2981 argmax/argmax 索引通道损坏）同为 torch_npu TE codegen+triton-ascend 编译路径静默数值错误类，但 4 机制（int64 截断/bool 翻转/hypot 溢出/pow 降精度）与 argmin/max 索引通道不同、分层定位待协查，分立编号；ops-nn !14059 与 INF-2422（#8254 ForeachExpm1 >256 尾部静默不计算）同机制（MAX_TENSOR_CONT 截断），为公共 tiling 路径补齐修复，走 update 补证据不另立；lightning_indexer 家族 8 连单（#7318~#7325）互链但机制互异（见下）。

### 新增条目（NPU/GitHub-NPU 来源 9 条）

| 编号 | 来源 | 机制摘要 |
|---|---|---|
| INF-2492 | [vllm-ascend#21462](https://github.com/vllm-project/vllm-ascend/issues/21462) | Qwen3.6-35B-A3B（Qwen3_5Moe）W8A8 compressed-tensors 在 vllm-ascend 0.30.0rc2（910B、CANN 9.2.0）temperature=0/13-token/eager/TP=1/2 全部输出乱码，0.18 同 checkpoint 正常；0.30 新 llmcompressor W8A8 融合（norm_quant/act_quant）与 LBNHC KV 布局为候选面但根因未定位，issue open |
| INF-2489 | [triton-ascend#3076](https://github.com/triton-lang/triton-ascend/issues/3076) | torch.compile(triton_experimental) 编译路径 4 dtype 组合静默错值：abs/int64 按 32 位截断（8589934599→7）、add/bool 25 元素翻 9、hypot/f32 无溢出保护注 inf/NaN、pow/f64 底数降 f32；eager 零差异排除 aclnn 层；逐样本探针复验，4 独立机制合并单 |
| INF-2493 | [ops-transformer#7324](https://gitcode.com/cann/ops-transformer/issues/7324) | aclnnLightningIndexer（910B3 arch22、v9.1.0）跨 S2 基块（actS2>512）且 actS1>4 且 K∈{16,32,64}：MergeSort 等分候选跨块归并不稳定，后块 0 分候选顶掉前块，top-K 索引集合错（23/1000 用例、530,843 发散索引全 ≥512）；4 种子 45-50/128 行失配、S2=512 对照全 PASS；prefill/长上下文常态命中，decode 不受影响；比较器细节为推断需内核确认 |
| INF-2494 | [ops-transformer#7319](https://gitcode.com/cann/ops-transformer/issues/7319) | LightningIndexerV2Metadata GQA 头数配比 nhq<nhk 时 groupSize 整除截断为 0 → mBaseSize_=0 除零 → isKvSeqAllZero 恒真走"全零 KV 序列"捷径，静默输出与全零长度输入逐 int 一致的空负载调度（0xDEADBEEF 哨兵 957/1024 未写）；接口全 0 返回；910B 分支无校验、文档无配比约束 |
| INF-2490 | [ops-nn#8561](https://gitcode.com/cann/ops-nn/issues/8561) | aclnnGather 越界 index 静默返回错误值（idx=4 读到第 0 元素 100、idx=99 返回 0.0）：host 只查 dtype/shape，kernel CheckIdxValue assert 生产版被编译掉；文档契约应报 161002；C++ 两段接口复现 |
| INF-2495 | [pytorch#5990](https://gitcode.com/Ascend/pytorch/issues/5990) | torch.compile 同图两个归约轴长度不同的归约：第二个归约走 rsplit partial 写 workspace 但 combo codegen（_codegen_standalone_kernel）无 combine 环节，out_ptr 从未写、输出为未初始化显存（maxdiff 178.66）；npu_rsplit_partial 四处赋值只有一处接 combine；对应 UT 用例 FAIL |
| INF-2496 | [runtime#1357](https://gitcode.com/cann/runtime/issues/1357) | aicpu_sched 守护进程 worker 初始化失败后 threadStatusList_ 不回退，CreateWorker 假成功：该 AICPU 核事件无人消费、任务挂起、host 同步超时，业务静默卡死无错误上报；修复 PR !6157（合入未核实） |
| INF-2497 | [ops-math#4658](https://gitcode.com/cann/ops-math/issues/4658) | WeightQuantPreprocess PerGroup（Ascend960 A16 C064）weight+scale 双转置时 host 从改写后 weight shape 取 G/N 而非 weightScale 逻辑 [G,N]，scale 预处理错、下游量化 matmul 按错误布局取 scale；修复已归档（closed，发布未核实） |
| INF-2491 | [msmodelslim !1164](https://gitcode.com/Ascend/msmodelslim/merge_requests/1164)（根 issue #680，!1163 同内容 closed） | 按文档 transformers==5.2.0 跑 Qwen3.6-27B w8a8 伪量化 eval 输出乱码，实际依赖 5.14.1；维护者确认 quarot 场景 2/3 为非问题；修复为纯文档版本更正（!1164 open），一键量化流程仍 5.2.0 |

（9 条 NPU + 8 条 GitHub 常规 = 17 条，编号 INF-2489~INF-2505。）

### NPU 留观/不收（本轮判定，理由）

- **机制明确但无失效实例/影响未实测**：ops-transformer **#7313**（mixed_quant_flash_attn_meta PA_NZ 5-D 布局下 v.size(2)/v.size(3) 非 D，fake output head dim 推导错误——源码级定位但正文无运行时失效实例与下游影响实测，随 #7319/#7324 族跟踪）；**#7305**+!15597（QSFA&SFA realkeyOffset 溢出，issue 正文仅一行标题、修复 PR 已 merged 但机制/症状均未展开）；**#7320**（V2Metadata 尺寸 v9.1.0 1024 / master 2048 跨版本契约不一致，同版本内自洽、混合部署才触发，无实测）；**#7323**（maxSeqlenQ/K 取值规则跨版本断裂，master 推荐 -1 被 9.1.0 GetWorkspaceSize 拒绝，fail-loud）；**#7318**（可选序列张量 null 时 AICPU 崩溃，fail-loud）。ops-math **#4663**（Multinomial bf16 CDF 平台 + u 上溢越界回退与旧 BM golden 的 tie 口径分歧——分析确认为 golden/算子边界口径问题而非算子错值，closed）；**#4668**（aclnnDot ttk vecdot_0024 用例精度失败，仅 CI 日志无机制分析）。ops-nn **#8468**（QuantBatchMatmulV3 mix 模板 A960 UB 空间分配溢出风险，未见静默数值损坏证据，closed）。
- **训练域（移交训练扫描）**：ops-transformer **#7325**（LightningIndexerGrad mode-3 死行 dweights 未初始化，0.63~1.18e37 脏值、非确定，29/50 精度失败用例——证据链完整但为反向算子，训练库范围）；ops-nn #8534（FusedMulApplyKerasMomentum 校验/精度）、#8545（ApplyCamePart1 归约精度+UB 规划）、#8547+!14093（SoftmaxFocalLoss pad 列 NaN 污染+大数归约容差）、#8502+!14040（PSROIPoolingGradV2D golden 坐标）、#8562+!14151（L2NormalizeGrad 1-ULP 级归约精度+性能，微小数值差异不自动收）、!14144（threshold_grad 判零精确比较）、!14119（AdvanceStep 越界防护）、!14143（foreach_div 非规格数精度）——优化器/损失/梯度域。Ascend/pytorch **!49365**（inductor expm1 精度，正文空模板仅标题）。
- **观测/工具域（不转述为计算故障）**：msprobe #682~#685（GradNorm NaN 案例 Dump 范围文档、nan_check 目录创建/交叉配置一致性/资料问题——观测工具行为域）；ops-transformer !15609（mhc_pre_sinkhorn 性能工程 PR 内含 4 项正确性修复（rf>1 系数布局仅 rf=1 正确、isLastK 整除挂死、尾块越读、SyncAll 剪除），一行级描述、独立证据不足，跟踪其拆分单）。
- **沿用既往判定**：hccl **!3968**（保序 AllReduce 非整除尾块 CCL buffer 越界）沿用 2026-11-05 #3914 留观判定（fail-loud 拦截、未见静默损坏）；本 PR 为 9.2.0 分支同步、open，修复归档推进中。msmodelslim #687、torchair #952~#957、ge/hccl/hcomm 其余按标题归档（fail-loud/特性/perf/文档/测试域）。

## 2. GitHub 常规 7 仓 + arXiv

- 认证状态：环境无 GITHUB_TOKEN（未认证）；search 请求间隔 ≥8 秒。
- **API 账本**：本轮实际发出 search 请求 35 组——初次 4 组单词 AND 语义（vllm/sglang/transformers/TRT-LLM silent wrong corruption incorrect）HTTP 200 但 0/0/0/1 结果，判定覆盖不足；7 组 OR 短语改写 HTTP **422**（旧版 issues search 不支持 OR 运算符，无结果返回，已弃用）；按既往口径重发 **21/21 组 HTTP 200**（7 仓 × correctness / silent wrong output / NaN，`is:issue created:>=2026-11-03`，per_page=50 单页均覆盖 total）。正文单点补拉 issue 10 条 + PR 1 条（core API 全 200）。评论线程未拉（除 #21387/#21348/#21164 状态核验 3 条）。
- 与既往判定比对（维持，无新证据不改判）：vllm #65680(=INF-2457)/#65564/#65513/#65734(=INF-2424 族)/#66074(=INF-2455)/#66144(=INF-2458)/#66211(=INF-2457)/#66246/#66610(=INF-2487)/#66437(=INF-2486)/#66009(=INF-2424)/#65324(=INF-2383 重复单)/#65325(=INF-2383)；sglang #48660(=INF-2425)/#48964(=INF-2447)/#48230（11-03 留观，本轮已 closed，见 §3）；transformers #50424（训练域 FSDP2 梯度丢弃）；TRT-LLM #21519/#21571；triton 0/0/0；tt-metal #64834(=INF-2480)/#64811(=INF-2481)/#64411(=INF-2482)/#64678(=INF-2483)/#64827(=INF-2484)/#64826(=INF-2485)/#63940(=INF-2400)/#64507/#64876/#63965（11-09 深读已判）与 tt_dit 审计族（#64103/#64104/#64105/#64106/#64111/#64113/#64123/#64125/#64127/#64131/#64140/#64144/#64150/#64151/#64187 沿用 11-07 族留观：模型 bring-up 审计清单域）；flashinfer #8570(=INF-2399)/#9068(=INF-2456，仍 open)/#8790/#8946/#8443/#8284；sglang #49657 之外 AMD/RFC/feature 项不收。
- **本轮新留观**：vllm **#66806**（DeepSeek V4.1 Flash P/D：PIECEWISE CUDA graph 与 eager producer prefill 输出差异，含 keep-vs-trim 对照（首 token logprob -1.6559/-1.7447）与 146-token 零 padding 仍分歧，但实验源基于 foraxe fork 快照、未在未修改 upstream 隔离，报告者自请归类指导——跟踪等维护者结论）；vllm **#66573**（Qwen3_5 embed_tokens 不传 quant_config：compressed-tensors 分支为推断的硬失败（fail-loud，未本地复现），GGUF+plugin PR#87 下 packed 路径已激活、静默回退未观察到——根因真实但无静默失效实例，关联 #54304/#45994 跟踪）；vllm #66435（DFlash drafter 复用 target 侧 LoRA metadata → CUDA illegal memory access，fail-loud 崩溃）；#65592（/generate max_tokens null 绕过默认值，API 契约域）。
- arXiv：2 组短语查询（"silent data corruption"+LLM、"bit flip"+inference，sortBy=submittedDate）HTTP 200 有效；16 篇全部与 11-02/11-03 已判定集重合（SDC 组 8 篇 + bit-flip 组 8 篇含攻击/容错硬件域），**无新论文**。

### 新增条目（GitHub 常规来源 8 条）

| 编号 | 来源 | 机制摘要 |
|---|---|---|
| INF-2498 | [vllm#66886](https://github.com/vllm-project/vllm/issues/66886) | MultiModuleMTPSpeculator（多模块 MTP、S>1）每槽缓存三连缺陷：add_request 空操作无重置 + 短窗只覆写尾部 + 拒绝后重灌不查有效数量——请求 B 的 draft 模型跑在请求 A 的 token ids/hidden states/MM embeds 上并写入 B 的后续模块 draft KV（TRITON_INTERPRET POC）；v0.31.0 |
| INF-2499 | [vllm#65906](https://github.com/vllm-project/vllm/issues/65906) | PR #61343（FP8 per-token-group 量化除法→乘倒数，含融合 RMSNorm+quant）后 Qwen3.5-122B-A10B-FP8 B200 GSM8K 前 10% 119/132→114/132 三轮稳定，GB200 换回旧 wheel 116→119；PR 自身对 397B 也报方向性下降；评论未读、维护者未确认 |
| INF-2500 | [vllm#66801](https://github.com/vllm-project/vllm/issues/66801) | Gemma 4 tower/connector LoRA：激活映射按 per-item 有效行数、投影消费 padded stride 展平张量，不等长 clip 批（25/50 行）中长 clip 尾 25 有效行收不到 LoRA（RMS 2.18 vs base 0.00026），greedy 首 token 改变（分组修复实验 96077→24286） |
| INF-2503 | [sglang#49657](https://github.com/sgl-project/sglang/issues/49657) | gfx1201（RDNA4 R9700）PCIe 双卡 HIP 定制 all-reduce 1-stage：start 旗标 __ATOMIC_RELAXED 发布/松弛轮询，旗标先于输入离开 L2 可见，对端读 stale——2000 次归约 ~2.5M 错误元素、两 rank 互不一致，full sync+barrier 不救；gfx12 release 语义修复已在报告者环境运行（main 未改） |
| INF-2501 | [flashinfer#9316](https://github.com/flashinfer-ai/flashinfer/issues/9316) | RadixTopKKernel_Unified 行循环缺 __syncthreads：下一行 LoadToSharedOrdered 覆写 shared_ordered 时慢 warp 仍在读，写出错误索引；bf16 B=192 V=65536 k=4096 8/300 种子坏行（各一个错 tie 元素），racecheck 1-2 hazards；一行 barrier 修复已验证；deterministic=True 不受影响；top_k_ragged_transform 共享该 kernel |
| INF-2504 | [flashinfer#8831](https://github.com/flashinfer-ai/flashinfer/issues/8831) | trtllm-gen MoE 小批 finalize（batch<74 且 4<topK≤64，top-6 kSplit=2）经共享内存合并部分和，与大批顺序累加的 FP 求和序不同：同 token MoE 输出依赖 batch 大小，decode/prefill logprob KL 3.5e-4~1.6e-3（0.7.2 为 0，useSmallBatch=false 归零），破坏 SGLang --enable-deterministic-inference 并阻塞升级（sglang#48468） |
| INF-2505 | [tt-metal#64740](https://github.com/tenstorrent/tt-metal/issues/64740) | 归一化工厂三缺陷：post_all_gather 在 2D 宽分片网格误开 two_stage_reduce（绑不存在 STATS shard、关 enable_sqrt、坏运行时参数→静默错值）；interleaved layernorm/rms_norm 显式 dtype 被硬编码忽略；GroupNorm 三工厂 NOC1 swap 错变量（非矩形 core group 组播挂死）+ COL_MAJOR 校验查错变量 |
| INF-2502 | [tt-metal#64840](https://github.com/tenstorrent/tt-metal/issues/64840) | scale_causal_mask_hw_dims_softmax_in_place（hw-dims 路径）不校验掩码宽高与输入/shard 几何，游标按模前进仅行尾回绕：窄掩码越界读（无回绕无报错）或错行掩码被应用，softmax 输出错值；#64681 明确不改本路径 |

## 3. 回访（近 7 日条目 + 轮巡旧源）

- **INF-2422**（ops-nn#8254）：修复面扩大——公共 tiling 修复 PR [!14059](https://gitcode.com/cann/ops-nn/merge_requests/14059) 已补齐 A2/A3 全部 foreach 公共路径（>256 报 561002），与 #8254 的 910C 证据同源，A5/arch35 已修；PR open。已 add-evidence + note。
- **INF-2473**（ops-transformer#7287 FIA LSE 竞态）：issue 已于 2026-11-10 关闭；修复 !15558 合入/发布未核实。已 note。
- **INF-2479**（vllm-ascend PR#21387 Kimi MLA FLOAT gate 修复）：PR 已 **merged**（2026-11-10T09:35Z）；随版发布未核实。已 note。
- **INF-2443**（vllm-ascend#21164 AscendStore PP 键）：issue 与修复 PR #21348 均仍 open，与 11-09 无变化。已 note。
- 轮巡旧源：**sglang#48230**（11-03 留观：PD+Mooncake 混合 DSA 后端 item_len 失配错位写）本轮发现已 **closed**——关闭原因/修复归属未读（正文压缩段与评论未拉取），维持留观待下轮读关闭说明；**vllm#65324**（INF-2383 level-2 sleep 重复单）已 closed，主单 #65325 仍 open；**ops-nn#8390**（INF-2461，11-09 承诺优先回访）批次索引显示 updated_at 2026-11-09T12:44Z（早于昨日采集水位，无新活动）、仍 open，GitCode 评论线程仍无低成本入口，无新证据不改判。
- 近 7 日新增条目来源状态批量核验（批次索引零 API）：昨日 21 条中 #21285/#21387 变化如上，#5960/#5961/#5962/#5972/#3057/#3058/#3043/#7287/#8403/#4637 本轮窗口内无再更新（updated_at 旧于水位或无状态翻转）；GitHub 侧 INF-2480~2486 来源 issue 今日全部仍 open。

## 4. 未覆盖范围与缺口

- GitCode/AtomGit issue 评论线程未拉取（无低成本批量入口），全部按正文判定；#48230 关闭原因、#65906 的 2 条维护者评论未读。
- GitHub search 单页 per_page=50 覆盖各组 total（最大 total 25），无分页缺口；但 correctness/silent/NaN 三词组之外的关键词面（量化/KV/logprob 等）依赖既往各轮查询与 NPU 批次，未逐日重扫。
- 本轮窗口内 NPU 925 新项全部标题过目+73 强关键词全文筛读；弱命中（纯 perf/特性/文档标题）按标题归档未逐条深读，与既往口径一致。
- 35 组 search 中 4 组 AND 语义与 7 组 422 OR 尝试无有效结果，有效覆盖由 21 组标准查询构成（已在上文声明）。

## 5. 小结

17 条新条目入库（待审计）：NPU/GitHub-NPU 9 条 + GitHub 常规 8 条；4 条既有条目回访更新（INF-2422 补修复证据、INF-2473/2479/2443 状态注记）。重点新面：**vllm-ascend 0.30 量化 MoE 乱码**（根因未定位）、**triton-ascend 编译路径 4 dtype 静默错值**、**NSA/lightning_indexer 双缺陷**（跨基块 top-K 等分丢失 + GQA 配比静默空负载）、**torch.compile 组合核归约输出未初始化**、**vLLM MTP speculator 跨请求槽位泄漏**、**flashinfer top-k 行间竞态与 MoE 小批输出随批变**、**gfx12 定制 all-reduce 松弛可见性**、**FP8 组量化倒数乘法精度回退**。留观重点：#66806 graph/replay 分歧（等上游归类）、#66573 量化 embedding 契约（等失效实例）、#48230 关闭原因待读、tt_dit 审计族与防御校验族延续。
