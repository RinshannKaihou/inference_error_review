# 每日扫描日报 2026-11-11（推理库）

## 0. Git 前置检查

- 分支 `main`，上游 `origin/main`；`git status --porcelain -uall` 为空；`git fetch origin main` 成功；HEAD 与 origin/main 一致（57747fd34d1a266fcabc8f453ff6de1a27ab1c7e）。通过，开始扫描。
- Cron 前置脚本 NPU 批次完整：**22/22 源全部 complete，failures 为空**。窗口 2026-11-10T22:00:12Z → 2026-11-11T23:00:12Z。registry_hash 2f6422d0…（22 源，与 11-08/09/10 一致）。

## 1. NPU 新增来源（共享批次）

- 批次：`/root/.hermes/profiles/third/cache/fault_scan_sources/batch-20261111T230012899450-ybbpf1q9`，generated 2026-11-11T23:00:12Z。inference scope 候选索引 30,757 条（滚动缓存全量）。
- **本轮新项判定窗口**：created ≥ 2026-11-10T22:00:12Z，共 **859 条**（issue 296 + PR 563），**全部条目标题过目**（强关键词标题命中 195 条全文筛读，重点候选深读约 30 条正文，全部使用批次缓存正文，未为 GitCode/AtomGit 额外拉 API）。
- 查重：本轮拟收/拟拒 URL 全部执行编目 `find` 与近 14 天日报完整 URL 检索（10-29/11-07 的 grep 命中为 ops-transformer#6910/#6915、op-plugin#691 误匹配，已用完整 URL 复核排除）。已存在走 update 不另立：ops-nn#8265→INF-2407、msmodelslim#667→INF-2466、vllm-ascend#21157→INF-2438、vllm-ascend#21285→INF-2478；GitHub 侧 #20853（SWA 主线修复 PR）此前未编目，本轮以其 v0.30.0 backport !21556 为源新立。
- 机制级比对：**AddRmsNormQuant 三连单**（同日同报告人）——#8643/#8644 同为 host 侧参数校验缺失→rc=0 静默放行（输出未写/越界读错值）合并一单，#8642 为 kernel 侧 SingleN 模板 UB 记账漏项（崩溃为主+≤256 静默越界区）独立一单；#21529 自父单 #21285（=INF-2478 量化 KV 索引读损坏）分立，报告者明确声明 MTP 接受率回退独立于 KV layout 问题；!21556（SWA 块 ID 残留发布）与 INF-2425（租约过期回收）、INF-2478（远端索引读损坏）同为 KV 生命周期错配但机制互异；!21551（M-RoPE PCP 位置错位）与既有位置编码族无重叠；#6025（erf dtype 回转）与 INF-2489（triton-ascend 编译路径 4 dtype 组合）分层不同（host 分解 vs kernel/codegen）。

### 新增条目（NPU/GitHub-NPU 来源 10 条）

| 编号 | 来源 | 机制摘要 |
|---|---|---|
| INF-2510 | [ops-nn#8643](https://gitcode.com/cann/ops-nn/issues/8643)+[#8644](https://gitcode.com/cann/ops-nn/issues/8644) | aclnnAddRmsNormQuant（A2/910B3、CANN 9.2.0、ops-nn master c090cedd）host 双校验缺失：zp2-only 组合资料口径应 161002 却 rc=0 放行且 y2Out 完全未写（32768 canary 全保留，下游消费残留）；scales1 声明 [1] 实存 [64] 放行、kernel 按 numCol 消费声明范围外 scales→252/256 错值、max_abs_error=25、并发摘要漂移（越界读 UB）；纯 aclnn 直调复现，修复 !14265/!14266 open |
| INF-2515 | [ops-nn#8642](https://gitcode.com/cann/ops-nn/issues/8642) | 同算子 SingleN 模板（numRow≤40）initSingleNUbSize 漏记 hasScales2 必配 inQueueBeta(4F)：UB 总分配 191KB+4F，fp16 组合 numCol>256 同步 507035 崩溃、≤256 越界写落尾部 1KB 预留区静默存活、全 bf16 任意 numCol 即崩；边界矩阵 7 崩 6 过全按根因预测；msSanitizer 抓到现行；修复 !14264 open |
| INF-2516 | [ops-transformer#7369](https://gitcode.com/cann/ops-transformer/issues/7369) | FIA MLA decode（arch35）CalcAIVMlaAttentionOutOffset 只实现 gSize==128/≤32 两档折叠分解，51/72 头（32<gSize<128）仍按旧口径算转置基准偏移、与写回端口径不一致，head>32 输出静默清零；修复 !15724 已 merged（commit 4374ff420，发布未核实），issue closed |
| INF-2517 | [ops-nn#8593](https://gitcode.com/cann/ops-nn/issues/8593) | LayerNorm 图融合 pass 经 add0/add1 输入探测 epsilon 而非子图边界第 3 输入，Add 操作数随 config 分支交换时探测落 mean 端口：layernorm_inference_fusion_pass 静默回退默认 1e-7（融合后 epsilon 与原图不一致），另一 pass 放弃融合；issue closed、修复 PR 14106 合入未核实 |
| INF-2511 | [Ascend/pytorch#6025](https://gitcode.com/Ascend/pytorch/issues/6025) | torch_npu Inductor 的 aten.erf 自定义分解对非 fp32 输入升 fp32 计算后回转 orig_dtype：bool/int64 的 INT_TO_FLOAT 提升结果被转回整型，编译路径输出 dtype 与 eager/CPU/meta（均 float32）自相矛盾、bool 结果压回 0/1；910B4 实卡+最小复现；修复 !49444 open |
| INF-2507 | [msmodelslim#691](https://gitcode.com/Ascend/msmodelslim/issues/691) | DeepSeek-V4.1-Flash 混合权重转换：linears 指定部分层 FLOAT、其余保持 MXFP4 的配置未完整应用，未设置的 MXFP4 层也被默认反量化，描述文件全标 FLOAT、MXFP4 层消失；转换成功无报错、必现（附完整 yaml/命令/截图） |
| INF-2506 | [vllm-ascend#21529](https://github.com/vllm-project/vllm-ascend/issues/21529) | 950DT、Qwen3.8-27B w8a8c8-mxfp8、mxfp8 KV、MTP：仅换镜像 v0.30.0-a5→v0.32.0rc-a5（同 CLI/checkpoint）接受率 ~49%→41.6%、吞吐 ~7.4k→4933 tok/s；根因未定位（draft 权重/采样 top-k/量化融合为候选面），输出正确性影响未证实；自 #21285 分立 |
| INF-2508 | [vllm-ascend#21558](https://github.com/vllm-project/vllm-ascend/issues/21558) | Mooncake layerwise PUSH + prefill PP>1：完成回调与失败清理被 last_layer_idx=-1 门控在最终 stage、接收侧计数无 PP quorum——早 stage 写阻塞/写失败时请求仍 finished_recving、无 invalid blocks，解码消费不完整前缀 KV；CPU 组件级受控复现（PP=2 三案例），端到端 NPU 未验证；issue 已 closed（关闭说明未读） |
| INF-2509 | [vllm-ascend !21551](https://github.com/vllm-project/vllm-ascend/pull/21551) | M-RoPE 模型 + ModelRunner V2 PCP>1：prepare_positions 的 1:1 rank 本地→全局映射假设被非连续碎片破坏，3D 位置按错误偏移重建、图像 H/W 与视觉 token 错位；Qwen3-VL-8B MMMU 58→≈34.33（无修复）/58.2（修复）；PR closed 未合入（关闭原因未读） |
| INF-2512 | [vllm-ascend !21556](https://github.com/vllm-project/vllm-ascend/pull/21556) | v0.30.0 SWA/RSWA + chunked prefill：退休 SWA 块换 null block 后 tracker 残留旧物理块 ID、元数据构建前不刷新组 ID，远端键缺失时按历史键发布该槽当前内容（可能已属其他请求）、bulk PUT 不滤非正块 ID；主线修复 #20853、v0.30/v0.32rc backport open，完整 NPU 验证未做 |

### NPU 留观/不收（本轮判定，理由）

- **3 次重交上限转留观**：vllm-ascend **!21554**（Kimi K3.1 Pro KDA conv weight：level-2 sleep→reload_weights 层重建无 hook、派生参数 ascend_conv1d_weight 不重打包→stale 权重→NaN logits（logprobs 请求 JSON nan 报错、普通请求退化）；8x950DT 4 层裁剪 checkpoint 三轮复现、修复后 bit-exact；PR open 含 CPU UT。条目 JSON 因 cause.behavior 长度校验差 4 字连续 3 次重交未过，按 daily_scan 规范转留观，证据链完整记录于此，下轮可重提）。
- **机制明确但无失效实例/影响未展开**：ops-nn **#8600**（GatherNd 动态索引深度 InferShape 把 -1 当下标遍历、kInvalidDimValue 追加进输出 shape——报告者自限 InferShape 层未推断设备执行后果，修复 !14213）；ops-math **#4685**（ConcatV2 同形输入共享 Shape 对象时指针比较误判基底、axis 维漏加——条件性触发未实测，修复 !7301/!7333）；ops-transformer **#7357**（swiglu_quant_v2 走 ReSplitBaseN 切非 64 对齐 baseN 致精度问题——正文一行无失效实例，960dt）；ge **!5909**（FIFO 增长 IO 地址表越界 #599——host 桩 ASan 复现、用户面症状未展开）；sgl-kernel-npu **!1016**（attn_res_mix_fused hidden=8193 multibuffer UB overflow——症状未描述、autotune 修复已验）；Ascend/pytorch **#6042**（社区 expm1 用例精度不达标——空模板仅标题）。
- **根因未定位问题单**：ops-nn **#8647**（同一冻结权重/输入跨日贪心解码从第 10 token 分叉、draft 命中率 78.7%→67.6%、端到端吞吐 -18%——独立程序复测 FIA/MatMul 跨进程逐位一致、算子不稳定假设被报告者自行排除，询问 tiling/算法选择锁定手段；Question 单、机制未定位留观跟踪）；vllm-ascend **!21527**（VLLM_BATCH_INVARIANT=1 在 torch_npu 2.10.0.post4+CANN 9.2.0 下 SparseFlashMla 单算子 replay 非确定→相同请求 logprobs 不一致——PR 单句自述无数值，post7+set_deterministic_level(3) 后 5 请求逐位一致；版本契约面待数值证据，与 #8647 主题呼应）。
- **fail-loud/契约/性能/文档/测试资产域**：ops-transformer #7351（S8S4 布局 561103）、#7372/#7373（文档口径 161001）、#7375/#7370（MoeFinalizeRoutingV2 信息库不一致）、#7352（kv_compress_epilog ST 用例 slot_mapping 范围过窄→重复 slot 并发写非确定——测试资产域；顺带确认"重复 slot 写序未定义"为算子既有契约而非缺陷）、#7353/#7442（CGDR 口径/UT）、#7350（离线编译）；vllm-ascend #21566（Engram 注册 rc=207001）、#21590（hybrid offloading crash，修复 !21592）、#21598（MemCache 4× KV 载荷浪费）、#21608+!21621（EPLB draft MoE 注册）、#21515/#21538/#21593（nightly）、#21532（w8a8 示例 dtype 文档，已随 !21559/60/61 closed）、#21528/#21534/#21536 等 RFC；Ascend/pytorch #6049（flex attention workspace UB overflow 编译失败——fail-loud，关联 triton-ascend !3129 xfail 复现与 !49499 修复，跟踪不改判）、#6027/#6030（AOTI RTLD_LOCAL 崩溃）、#6033（noinline 无法生成）、#6041/#6046（fallback/fake 实现缺失）；op-plugin #700（rounding 族整型/bool 报错——fail-loud，!6206 恒等短路修复中）；hcomm #1266（GetNetLayers 不对称组网 init 失败——fail-loud，!6593 修复越界读）、#1270（多线程初始化报错）；hccl #949（校验误打 ERROR 日志）；ops-nn #8588/#8589/#8591/#8604/#8648（校验加固类）、#8586（文档）、#8601（golden 假红已 closed）；ops-math #4676（expint 边界对齐——非 LLM 推理面）、#4677（cholesky tiling 拦截）、#4686（文档）。
- **训练域（移交训练扫描）**：ops-transformer #7355（sparse_lightning_indexer_kl_loss_grad 溢出精度异常）、!15737/!15741（sli/dli klg int64 截断）；ops-nn #8641（LstmBackward 950 精度）、#8578（MseLossGrad 误用 IsRegbase）、#8579/#8580/#8581/#8583/#8584/#8590/#8599/#8602/#8603（grad/optimizer/loss 域）；Ascend/pytorch #6048/#6051/#6052（inductor 黑名单摘除类）。tt-metal #64851/#64845（sqrt_bw/erfinv_bw NaN 吞噬——_bw 梯度域，tt-robot 审计族口径沿用）。
- **观测/工具域（不转述为计算故障）**：mssanitizer #256（FlashAttnFunc_SWA_V3/V4 报错）；msprobe #687（vllm-ascend PD kvcache 数值监控 v2 引擎适配——能力需求非实测覆盖）、#686/#688/#689（UT/审计/资料）。

## 2. GitHub 常规 7 仓 + arXiv

- 认证状态：环境无 GITHUB_TOKEN（未认证）；search 请求间隔 ≥8 秒。
- **API 账本**：search 请求 **21/21 组 HTTP 200**（7 仓 × correctness / silent wrong output / NaN，`is:issue created:>=2026-11-04`，per_page=50 单页均覆盖 total，最大 total 18：vllm correctness）。正文单点补拉 core API 7 条（#67065/#49881/#50599/#9333/#9343 detail 全 200；#9343 首拉正文被环境清单占据、二次取尾部描述段；#48230/#65325 状态核验——#65325 两次 HTTP 403 未核验成）。评论线程未拉。
- 与既往判定比对（维持，无新证据不改判）：vllm #65680/#65564/#65513/#65734（=INF-2424 族）/#66074（=INF-2455）/#66144（=INF-2458）/#66211（=INF-2457）/#66246/#66610（=INF-2487）/#66437（=INF-2486）/#66009（=INF-2425）/#65906（=INF-2499）/#66801（=INF-2500）/#66886（=INF-2498）/#66806/#66573（留观）/#66435（fail-loud）/#65592（契约域）；sglang #48660（=INF-2425 关联族）/#48964（=INF-2447）/#49657（=INF-2503）/#49185+#49198（=INF-2464）/#48967（11-09 判）；transformers #50424（训练域）/#50440（=INF-2465）/#50336（=INF-2401）/#50334/#50484（=INF-2470）/#50603（7~8 月已判）/#50545/#50612（训练域）；flashinfer #8570/#8607（=INF-2399）/#8790/#8946/#8897/#8443/#8284/#9068（=INF-2456，仍 open）；tt-metal #64834（=INF-2480）/#64811（=INF-2481）/#64411（=INF-2482）/#64678（=INF-2483）/#64827（=INF-2484）/#64826（=INF-2485）/#63940（=INF-2400）/#64507/#64876/#63965/#64187（tt_dit 审计族）/#64648/#64637/#64600（11-06/07 已判）；TRT-LLM 0/0/0；triton 0/0/0。
- **本轮新留观（GitHub 轮）**：#64575/#64647（sharded softmax 悬赏审计族）、#64933（upsample BLOCK_SHARED 悬赏）、#64327（Quasar LLK pack_untilize DEST 读取域）、#9279（JIT -z now 加载失败 fail-loud）、#66942/#66794/#66782/#66782/#65813/#66290/#66782 等 RFC/Tracking 不收；#49796（Qwen-VL 预处理 OOM 2.87×——资源域）、#49336（SM90 decode perf 回退——纯性能）。
- arXiv：2 组短语查询（"silent data corruption"+LLM、bit flip+inference，sortBy=submittedDate）HTTP 200 有效；18 篇全部与既往判定集重合（2610.33475/2610.37033/2610.30798/2610.18557 等均见于 10-28~11-10 日报），**无新论文**。

### 新增条目（GitHub 常规来源 5 条）

| 编号 | 来源 | 机制摘要 |
|---|---|---|
| INF-2513 | [vllm#67065](https://github.com/vllm-project/vllm/issues/67065) | NixlConnector pull 模式 n>1：n 个子请求共享 remote_request_id 各自发读通知，_get_new_notifs 按 req_id 计数不区分发送者、达 expected_consumers（TP 拓扑而非 n）即释放——首子读完即释放，晚到子读到被下个 prefill 复用重写的块，HTTP 200 返回无关文本（复现 16 子中 8/9 丢密钥，~5.4s 传输差窗口）；v0.9.0~v0.33.0 皆然；曾以 GHSA-grjf-r4cj-w3ch 私报；var→INF-2425（共享"消费完成前 KV 块被提前释放/回收、后到者读复用块"，触发差异：租约过期 vs 通知计数不含 n） |
| INF-2518 | [sglang#49881](https://github.com/sgl-project/sglang/issues/49881) | GLM-5.3-Flash+EAGLE+decode CUDA graphs：批大小不在捕获表（默认跳过 9/11/13/15）被 pad 到下一桶后 logits 变 NaN 并永久锁存——此后所有请求 token id 0、/flush_cache 无效、无日志、200 返回；变体矩阵锁定需 EAGLE+图+pad 三者同时；规避=逐桶全捕获；根因疑 padding 行（request-pool 行 0+零填 Mamba 索引）读写状态行留 NaN（未确认）；同症状类 #48964(=INF-2447)/#49274/#43514 不同触发 |
| INF-2514 | [flashinfer#9333](https://github.com/flashinfer-ai/flashinfer/issues/9333) | trtllm_fp4_block_scale_routed_moe per-token NVFP4 路径 hidden_size 为 128 倍数但非 256 倍数（2688=Nemotron-3 Nano）时无有效配置也不拒绝：平均绝对误差 ~5× 对齐值（ReLU2 diff 102.5~132.8、max 1896）、SwiGLU 256-token NaN、单 token 正确、无错误；SM100/SM107 一致；vLLM 靠 pad 到 256/512 幸免 |
| INF-2519 | [flashinfer#9343](https://github.com/flashinfer-ai/flashinfer/issues/9343) | gated_residual_rmsnorm Cake PDL 程序（rows≤128）在 griddepcontrol.wait 前 prologue 加载 branch（>128 行程序正确在 wait 后）：生产者触发依赖后仍覆写 branch，消费者读覆写前数据；vLLM Muse Spark DSpark draft 辅助隐状态路径（DP+EP TP=1、批 8-48 行）draft 收 NaN——greedy 全拒（接受长度 1.0）、温度采样全收输出垃圾（AIME25 0% vs 100%）；规避 enable_pdl=False/backend='cute'；var→INF-2399（共享 PDL 依赖未就绪即读生产者输出，差异：MoE scratch vs norm branch、rows≤128 变体） |
| INF-2520 | [transformers#50599](https://github.com/huggingface/transformers/issues/50599) | InklingTopkRouter 在 BF16 下做 sigmoid+bias（加载时 e_score_correction_bias/gate.global_scale 亦舍入 BF16）：不同分数 Top-K 前并列，专家集 CPU {5,6}/CUDA {0,1} vs FP32 {6,7}；三层小模型 19/48 token 专家集变而 loss 仅差 ~4.8e-7（合成，全模型质量未测）；与 INF-2465（need_fp32_gate 被忽略）同域不同机制 |

## 3. 回访（近 7 日条目 + 轮巡旧源）

- **INF-2407**（ops-nn#8265 SituGlu）：修复 PR [!14239](https://gitcode.com/cann/ops-nn/merge_requests/14239)（WIP open）四类缺陷整改（out 非切分维校验、非连续视图拒绝、beta float32 域校验、TanhScale 级数分支修复 β≥1e4 小输入归零）+910B3 自验。已 add-evidence + note。
- **INF-2466**（msmodelslim#667 KV per-channel 校准丢失）：修复 PR [!1168](https://gitcode.com/Ascend/msmodelslim/merge_requests/1168)（open）跨批累积统计、9 例 CPU 回归（修前 6 败）。已 add-evidence + note。
- **INF-2438**（vllm-ascend#21157 FP8 静默 BF16 回退）：修复 PR [!21516](https://github.com/vllm-project/vllm-ascend/pull/21516)（open）scheme 构造期拒绝 + A2 实机验证。已 add-evidence + note。
- **INF-2478**（#21285）：子单 #21529 分立编目为 INF-2506，已 note 提示关联避免重复计数。
- **INF-2496**（runtime#1357）：issue 已于 11-11 closed；修复 !6157 合入未核实。已 note。
- **INF-2491**（msmodelslim !1164 文档版本更正）：PR 已 merged（2026-11-11）。已 note。
- **INF-2422**（ops-nn#8254 ForeachExpm1）：issue 已 closed、修复 PR !14059 已 merged（11-11）；发布未核实。已 note。
- **INF-2461**（ops-nn#8390）：11-11T06:44 有新活动、仍 open，GitCode 评论无低成本入口未读，无新证据不改判。已 note。
- 轮巡旧源：**sglang#48230**（11-03 留观 PD+Mooncake item_len 错位写）API 核验 closed=completed（2026-11-06T06:51）、**0 条评论、无关闭说明**——关闭原因已无可读内容，修复归属未核实，维持留观；**vllm#65325**（INF-2383 主单）两次 detail 请求 HTTP 403 未核验成（昨日记录仍 open），列入缺口；昨日 17 条（INF-2489~2505）来源经批次索引与今日搜索快照零/低 API 核验：#7324/#8561/#8346/#8390 窗口内有活动仍 open，#21462/#3076/#7319/#5990 无翻转，GitHub 侧 INF-2498~2502 来源 issue 今日仍 open，#7287/#21387 状态与昨日注记一致。

## 4. 未覆盖范围与缺口

- GitCode/AtomGit issue/PR 评论线程未拉取（无低成本批量入口）：#8643/#8644/#8642 修复 PR 审查进展、#21558 关闭说明、!21551 关闭原因、#8390 今日新活动内容均未读。
- GitHub：#49881/#9333/#50599 各 1 条评论未读；#67065 0 评论；vllm#65325 状态核验 403 未完成（未认证 core 限额）；search 三词组之外关键词面（量化/KV/logprob 等）依赖既往各轮与 NPU 批次，未逐日重扫。
- vllm-ascend **!21554**（KDA reload NaN）因 new 命令 3 次重交上限转留观（长度校验 4 字之差），未入编目——证据链完整记录于本报告 §1，为明确缺口，下轮优先重提。
- NPU 859 新项全部标题过目、195 强关键词筛读；弱命中（纯 perf/特性/文档/重构标题）按标题归档未逐条深读，与既往口径一致。
- arXiv 检索为 2 组关键词短语，覆盖面依赖既有查询口径。

## 5. 小结

15 条新条目入库（待审计）：NPU/GitHub-NPU 10 条 + GitHub 常规 5 条（INF-2506~2520，其中 2 条 var 关系：INF-2513→INF-2425、INF-2519→INF-2399，回显目标标题均已核对）；8 条既有条目回访更新（INF-2407/2466/2438 补修复证据，INF-2478/2496/2491/2422/2461 状态注记）。重点新面：**AddRmsNormQuant host 校验双缺口静默放行**（y2Out 未写留脏 + scales 越界读错值）与 **SingleN 模板 UB 记账漏项**（>256 崩溃/≤256 静默越界区）、**FIA MLA gSize 折叠区间输出清零**、**LayerNorm 融合 pass epsilon 静默回退 1e-7**、**NixlConnector n>1 子请求读复用 KV 块**、**sglang EAGLE 图 padding NaN 永久锁存**、**flashinfer NVFP4 隐维不对齐静默错值**、**PDL norm 提前读 branch 致 draft NaN（AIME25 0%）**、**Mooncake PUSH 提前报完成**、**M-RoPE PCP 位置错位（MMMU -40%）**、**msmodelslim 混合量化配置静默失效**。留观重点：!21554 KDA reload NaN（3 次重交上限转留观，下轮重提）、#8647 跨日推理漂移与 !21527 batch-invariant 版本契约（同属确定性锁定主题）、#21529 根因定位、#66806/#66573/#48230 延续。
