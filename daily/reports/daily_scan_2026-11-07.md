# 每日扫描日报 2026-11-07（推理库）

## 0. Git 前置检查

- 分支 `main`，上游 `origin/main`；`git status --porcelain -uall` 为空；`git fetch origin main` 成功；HEAD 与 origin/main 一致（d1229d66b6a275b43e79687c1cfded7b5baefcbe）。通过，开始扫描。
- Cron 前置脚本 NPU 批次完整：**22/22 源全部 complete**，无 failures。窗口 2026-11-06T22:01:00Z → 2026-11-07T23:01:00Z。registry_hash 2f6422d0…（22 源，与昨日一致）。

## 1. NPU 新增来源（共享批次）

- 批次：`/root/.hermes/profiles/third/cache/fault_scan_sources/batch-20261107T230100846920-r61o8x1m`，generated 2026-11-07T23:01:00Z。inference scope 候选 27317 条（updated 滚动窗口全量），覆盖 19 个 inference 适用仓。
- **本轮新项判定窗口**：created ≥ 2026-11-06T23:01:00Z（昨日批次截止），共 223 条（71 issue + 152 PR）。71 条 issue 标题全部人工过目；~28 条重点候选全文深读（含 6 条修复 PR 正文）。
- 近 14 天日报完整 URL 查重：本轮拟收 11 个来源 URL 全部 0 既往判定命中；编目 `find` 查重 0 命中；训练库跨库查重 16 URL 全部 0 命中（无 cross 关系）。
- 机制级比对：#5938/#5946 与昨日 INF-2426（dynamic 归约长度固化）、INF-2430（static launcher 重复执行）同属 torch_npu 编译器静默错值密集期但缺陷位置与机制均不同（分核归约尾块 mask 删除 / slice_scatter 前缀丢失），分立编号；#8366（MaxPoolGradWithArgmax NHWC 块内 C 解码）与 INF-2440（AdaptiveMaxPool2d NHWC 缺反向 transpose）同为池化 NHWC 域但机制不同（tiling 切 C 除数错误 vs 转置缺失）；#691（bool add 语义）为新机制；#21251 滑窗 KV 陈旧块与 INF-2439（ScatterPaKvCache 越界写）同为 KV 域但机制不同。

### 新增条目（NPU/GitHub-NPU 来源 7 条）

| 编号 | 来源 | 机制摘要 |
|---|---|---|
| INF-2448 | [op-plugin#691](https://gitcode.com/Ascend/op-plugin/issues/691) | NPU bool 张量 torch.add 走算术 aclnn add 把整数和直接写 bool 字节：True+True 得 2（CPU 语义逻辑或=1）；广播/标量/add_/alpha 全形态受影响，(5,1)+(5,) 广播 25 元素 12 个失配；修复 !6187 open（路由 aclnnLogicalOr） |
| INF-2449 | [ops-math#4607](https://gitcode.com/cann/ops-math/issues/4607) | arch35 abs complex64 指数因子归一化 vB1=(B-2^k)+1.0≠B/2^k，仅 A∈[1,2)（k=0）正确：A=3,B=1 得 4.0（真实 3.162，误差 26.5%）；跨 binade/量级差大输入静默错模长；修复 !7201 open（改 vA·sqrt(1+(vB/vA)²)，二级冒烟过） |
| INF-2450 | [pytorch#5938](https://gitcode.com/Ascend/pytorch/issues/5938) | triton_experimental 分核归约：全局 numel 整除 block 但 ceil(numel/核数) 不整除时尾块 mask 被当恒真删除且分核改写无法恢复，跨分片重复读取+末片越界：sum(1048576×1.0) 得 3625399（相对差 2.46）；多轴 partial-body 失配仍生成 workspace combine；修复 !49242 open |
| INF-2451 | [ops-nn#8366](https://gitcode.com/cann/ops-nn/issues/8366) | MaxPoolGradWithArgmax NHWC 小 C：Argmax 全局 (h*W+w)*C+c 编码但 DoMulCNhwc 用块内 C 作除数，Tiling 无条件收缩 cOutputInner（C=2/3/15 切到 1/2）致解码 h/w 错、梯度散错位或被 Range Check 丢弃，4 用例全 FAIL；修复 !13770 open（Tiling 侧 BigC 阈值规避，未上板） |
| INF-2452 | [vllm-ascend#21251](https://github.com/vllm-project/vllm-ascend/issues/21251) | 投机解码滑窗 KV 陈旧块跨请求污染：Ascend 重写 initialize_kv_cache 未传 sliding_window_reach，MASK_SLIDING_WINDOW 永不激活，滑窗下已释放重分配块 ID 未置空；并发换血下 GSM8K 4/6 探针由对转错、重复块 ID 仅限滑窗 draft 层（step 316 首现）、权重逐字节一致排除乱写；修复 #21252 open |
| INF-2453 | [pytorch#5946](https://gitcode.com/Ascend/pytorch/issues/5946)（已 closed） | triton_experimental slice_scatter 融合 masked load/select 负偏移钳制后丢失未选中前缀：x[1:5]=src 重复修改误改 x[0]（0.5→0），无报错；修复 !49255 closed（含 11-07 完整验证 20 方法/24 实例全 PASS，合入未核实） |
| INF-2454 | [pytorch#5943](https://gitcode.com/Ascend/pytorch/issues/5943) | 两个独立根因：pad-lane 抽象格缺除法族传播（other=0.0 使 1/0=inf 误判 finite，守卫被删、0*inf=NaN 进累加器，R0_BLOCK=1 正确而 2/4/8/16/32 全 NaN，跨进程 50% 复现）；legacy group_fn 返 per-dim 违反 canonical 契约致 default 后端融合期 SympifyError（3/3 确定性）；修复 !49252 open |

### NPU 留观（本轮判定）

- **pytorch #5948**（aclgraph 多流捕获 wait_event 写在 with 内/外结果不同；task_queue 排空条件 `is_capture_context()` 检查 TLS 当前流而非目标流的自锁逻辑，源码级分析详尽）——正文字根因分析充分但未见静默错值实测（后果是图池路由错误/内存池归属，属分配正确性域），且修复 !49259/!49260 仍 WIP closed；随 #5933（多流 replay 挂死）域留观，下轮看修复落地。
- **ops-transformer #7226**（A5 FAG dropout seed/offset>2^31 精度问题）——正文一句话+截图，细节在修复 PR !15486（preserve int64 dropout seed and offset）；dropout 路径推理域默认关闭，证据密度暂不足；随 !15486 留观。
- **pytorch #5936/#5937**（triton_experimental 高阶算子/invoke_subgraph 专用 lowering 被批量 fallback 覆盖，AttributeError fail-loud）——fail-loud 编译崩溃域；#5937 已并入 #5936，修复 !49240 open。不占编号。
- **pytorch #5945**（persistent/multi-kernel 归约配置未落实到执行）——区域配置不生效（性能/策略面，结果仍正确）；修复 !49254。留观。
- **pytorch #5944**（triton_experimental 缺 C++/AOT wrapper 路径，AOT 测试失败 fail-loud）——功能缺口；修复 !49253 closed。留观。
- **「空输入早退+非空输出未写」族新批**：ops-math #4617（TriangularSolve (3,0) 早退不写 mOut[3,3]，canary 0/9 改写，修复 !7205 open）、ops-nn #8374（GroupNormBackward N=0 不写 gradGamma/Beta，哨兵 123456 残留）、ops-nn #8385（FLCELossGrad BT=0 不写 weightGradOut，0/2048 字节改写）——与已收代表（INF-2246/1794/2126/2189/1657/2391）同型异算子，正文均含 canary 实测+源码定位（d1efef7c/dd851356 复核），按族留观；#4617 若下轮仍无维护者响应可单独占号。
- **错误码域**：ops-nn #8379（AddRmsNormQuant A2 放行 FP32 首段 0）、#8380/#8381/#8382（DequantSitu/DequantSwiglu 错误码 561002/561103 vs 资料 161001/161002）、ops-transformer #7227/#7228（ScatterPaKvCache/NsaCompressWithCache 空指针 561103）、#7231（pageBlockSize 0/256 放行两段 0）、#7234（compressStride=0 口径空白）——错误码/约束校验域，非静默错值；按既往口径不占编号。
- **ops-math #4616**（SearchSorted values/sequence dtype 不一致时按 sequence dtype 读 values 缓冲，含越界读）——英文静态分析单，无复现脚本/实测数值；INT64 sequence+INT32 values 属 API 误用面，留观待实测。
- **ops-math #4612/#4613/#4614/#4615、ops-nn #8384**（DOUBLE 首段 0 执行段 507018）——**报告者已撤回**（实测为 device 2/3 AICPU 内核包损坏的环境问题，其他卡正常），维持撤回判定不收录；记录为环境排查案例。
- **hcomm #1243**（rtsq_a5 baseSqHead_ 跨线程无原子/屏障 + 环满等待锚点无刷新，2 MEDIUM 检视意见，作者暂不处理）——静态检视报告，未同步缺陷的实测后果；与 #1244/#1245 检视跟踪同域留观。
- **ops-nn #8372**（MaxPoolGradFusionPass 注册在 kAfterInferShape 替换节点缺 shape/dtype 推导）——影响图构建/融合生效，待黄区验证；留观（若证实精度结果影响再评）。
- **ops-transformer #7229**（Arch92 MXFP8 Tiling 下沉无法匹配 Tiling Key，正文仅标题一句）、**#7230**（QFA 静态编译 v_tail 紧凑 stride {1} 误判非连续致 tiling 失败，修复 !15489 open）——tiling/编译面 fail-loud 或信息不足；留观。
- **msprobe #676/#677**（nan_check is_nan 落盘 pt 文件命名错乱：调用次数从 0 重编、output 少参数序号）——观测工具落盘命名缺陷，误导 NaN 定位但不改计算结果；观测域记录不占编号。
- **op-plugin #688**（atb FA bf16 alibi compress 用例 FAIL 1e-05）——维持 11-06 留观判定（单测失败无根因），本轮窗口内无新评论。
- **文档/需求/构建域**（不占编号）：ops-math #4603（cast golden e8m0）/#4604（tile/expand infershape 加固）/#4605（流水线检查）/#4606（TriInvColSweep 新算子）/#4608（自定义算子接口兼容）/#4610（CANNJudge 判题咨询）/#4611（sin/cos 5162）；ops-nn #8367（acts_ulq README 掩码口径）/#8368（FatreluMul 文档，closed）/#8369（ForeachMulList 文档）/#8370（golden tf/torch 隔离）/#8371（softshrink 资料）/#8373（UniqueSortFusionPass 960）/#8375-#8378（Celu/NPUAllocFloatStatus/CeluV2 质量加固）/#8383（LayerNormV3 判题咨询）；runtime #1343-#1345；op-plugin #689/#690（Feature）；pytorch #5935（CRCR 异常类型迁移）/#5939（max_unpool indices INT8 契约）/#5940/#5941/#5942/#5947（FlexAttention 测试/性能）；torchair #948（BMM 转置性能）；hcomm #1244/#1245/#1246（检视/重构）。
- **修复 PR 流**（窗口内 152 条 PR 按标题过滤）：!6187（bool add→!691）、!7201（→#4607）、!7205（→#4617）、!7209（ReplicationPad 拒 DOUBLE→INF-2431）、!7210/#7211（→#4599/#4601）、!49240-!49255（triton_experimental 五笔修复组）、!21252（→#21251）、!15486（→#7226）、!15489（→#7230）、!13770（→#8366）、!15488（fused_infer_attention_score TND 零 batch needInit 过度置位→#6734）等已随对应 issue 判定；其余为适配/性能/clean code 流未逐条深读。

## 2. GitHub 常规 7 仓 + arXiv

- 认证状态：环境无 GITHUB_TOKEN（未认证）；search 走共享配额，请求间隔 ≥8 秒。21 组查询（7 仓 × correctness/silent wrong output/NaN，`is:issue created:>=2026-10-31`）：**21/21 组 HTTP 200**（tt-metal 1/2 两组首轮 504，重试后 200）。无限流等待（累计 0 秒）。
- 与既往判定比对：vllm#65324/#65325/#65680/#65564/#65513/#65734/#65337/#65131/#65385、sglang#48660（=INF-2425）、#48230、#48964（=INF-2447）、transformers#50336/#50315/#50334/#50212/#50216/#50195（=INF-2347）、TRT-LLM#21519/#21385/#21571、flashinfer#8570（=INF-2399）/#8443/#8284/#8607/#8790（=INF-2108 补证）、tt-metal tt_dit 族（#64150/#64151/#64103/#64113/#64144 等）与 tt-robot 族（#63910/#63924/#63619/#63627-#63684 等）均沿用既往判定，无新证据不改判。本轮 tt-metal 新增 #64740/#64780/#64811/#64507/#64411/#64600/#63866/#64647/#64575 为 tt-robot 审计流/赏金任务批，随既往族留观。

### 新增条目（GitHub 常规来源 4 条）

| 编号 | 来源 | 机制摘要 |
|---|---|---|
| INF-2455 | [vllm#66074](https://github.com/vllm-project/vllm/issues/66074) | Hopper SM90 默认 Humming MXFP4 kernel 以 FP8 激活开 fused E8M0 路径：每张量 code 钳到 [max-11,max] 并重编码 E2M1，距最大 scale 12-15 binade 块粗化（15 时仅 4/6 存活全变 8）、≥16 binade 权重全解码 0；fused 投影邻接部件与 N padding=1.0 抬窗口；无任何警告，Marlin 同层精确；规避 VLLM_DISABLED_KERNELS=HummingMxFp4LinearKernel |
| INF-2456 | [flashinfer#9068](https://github.com/flashinfer-ai/flashinfer/issues/9068) | apply_rope_with_cos_sin_cache_inplace(is_neox=True) head_size 16/48/80/96 走 RopeQuantize（vec=16）按整向量取旋转伙伴/符号，rotary_dim/2 非 16 倍数时跨 cos/sin 半边界+加载越界：误差至 1e37 且同输入两次调用不同；96 未被该机制解释仍实测错；0.6.18-0.7.3 在 |
| INF-2457 | [vllm#66211](https://github.com/vllm-project/vllm/issues/66211) | compressed-tensors NVFP4/MXFP4 行并行层 TP 切输入维非组大小整数倍时：codes 按元素偏移切、scales 按组偏移切，rank>0 codes 乘错组 scale、rank0 尾列无 scale；Gemma-4-26B-A4B-NVFP4 TP=8 dense 68.7%/expert 70.0% 非零 code 列错 scale，GSM8K 24.3%→1.44%（McNemar p=2.4e-76）；TP=4 全对；同族 ModelOpt MoE 已报 #49949/PR #49951 |
| INF-2458 | [vllm#66144](https://github.com/vllm-project/vllm/issues/66144)（昨日留观转收） | --enable-fault-tolerance 下 async EPLB worker 异常被吞：step() 不抛、引擎 HEALTHY，EPLB 永久静默停摆；作者 follow-up 源码定位 FT reset 4 种陈旧泄漏（stale error 复写/check-then-write 竞态/信号被新 epoch 重标跑旧失败组/per-rank 判定不一致致集合通信错配）；修复 PR #66146 open（最小修复，4 种泄漏已记录未修） |

### GitHub 留观（本轮判定）

- **vllm#66246**（单 token 基准探针继承主负载 generation-length 覆盖）——基准方法学域，不影响生产推理正确性；留观。
- **flashinfer#8946**（CAKE relaxed contracts 非确定性可接受声明）、**#8757/#8170/#8526/#8527**（Feature/实验跟踪）——特性域。
- 其余 RFC/Feature/Perf/CI/文档不占编号。

### arXiv

- 2 组查询（"silent data corruption"+LLM 8 篇、"bit flip"+inference 8 篇，sortBy=submittedDate）HTTP 200 有效。无新论文：SDC 组 TrainSDC/SCOUT/LLM-PRISM/2604.00726/2502.12340 已收或既往已判，Cauchy-Schwarz（2610.18557）与 Syndrome Decoding（2609.19743）为既往已判推理域论文无新增证据，2610.33475 为架构综述非故障域；bit-flip 组 Rowhammer（2610.30798）/JITterFlip/BARE-AI/辐射容错/剪影容错为攻击与硬件容错域，按既往口径不收。

## 3. 回访（近 7 日条目 + 轮巡旧源）

- **昨日 22 条（INF-2426~2447）批次快照核验**：20 条 GitCode/GitHub-NPU 来源全部仍 open 无状态变化；#8346（INF-2432）11-07T00:19 有更新仍 open（已 update --note）；#5946（今日 INF-2453）与 #7172/#7188（INF-2445/2442）issue 已 closed（修复验证完成，合入未核实——#7172/#7188 昨日已注记）。vllm#66141（INF-2446）、sglang#48964（INF-2447）搜索窗口内仍 open 无变化（已注记）。
- **轮巡旧源 3+ 条**：flashinfer#6907（INF-2108）11-05 closed（此前已注记）、残留 #8790 仍 open 无变化；INF-2259（#6910）、INF-2382（#7063）批次窗口内无新活动，状态沿用；**vllm#66144（昨日留观）读作者 11-06 follow-up 评论发现 4 种 FT reset 陈旧泄漏源码定位 + 修复 PR #66146（open，11-07 更新），新证据足，转收 INF-2458**。
- **昨日日报勘误**：昨日「ops-nn #8218（BlockSparseAttention 全 0 mask 未写输出）」判定有误——今日读正文实为 **gmm MXFP4 K 分组需求单**（closed），非 BlockSparseAttention 空态故障；该候选编号与内容对不上，昨日留观记录作废。BlockSparseAttention 全 0 mask 空态族继续按 #8216/#7218 族留观（等待正确编号的来源出现）。
- ops-math #4612-#4615、ops-nn #8384 五条「DOUBLE 507018」单 11-07 已由报告者撤回（AICPU 内核包环境问题），昨日同类未收录，无需勘误编目。

## 4. 编目与视图

- 新写入 **11 条**（全部 pending_audit）：INF-2448 ~ INF-2458，编号连续。映射：

| 编号 | 来源 | 标题 |
|---|---|---|
| INF-2448 | op-plugin#691 | bool add 算术和写入 bool 字节语义错 |
| INF-2449 | ops-math#4607 | abs complex64 指数因子归一化仅 A∈[1,2) 正确 |
| INF-2450 | pytorch#5938 | 分核归约尾块 mask 丢失求和错 3.5 倍 |
| INF-2451 | ops-nn#8366 | MaxPoolGradWithArgmax NHWC 小 C 解码错位梯度散错 |
| INF-2452 | vllm-ascend#21251 | 滑窗 KV 陈旧块跨请求污染输出 |
| INF-2453 | pytorch#5946 | slice_scatter 误改切片外数据 |
| INF-2454 | pytorch#5943 | Inductor pad-lane 除法 NaN 与 group_fn 违约 |
| INF-2455 | vllm#66074 | Humming MXFP4 E8M0 窗口重编码静默清零权重 |
| INF-2456 | flashinfer#9068 | Rope neox 特定 head_size 静默错值 |
| INF-2457 | vllm#66211 | NVFP4/MXFP4 行并行加载错配 block scale |
| INF-2458 | vllm#66144 | EPLB worker 异常静默停摆与 reset 泄漏 |

- var/cross 关系：本轮无（同机制判定均为分立：不同算子/不同缺陷位置）。
- 回访 update：INF-2432/2446/2447 共 3 条注记。
- 训练库跨库查重：16 个拟收 URL 在 training_error_review `find` 全部 0 命中，无 cross 关系。
- `catalog_tool.py stats`：**2458 条**（active 2447 + pending_audit 11）；`build` 完成；`validate` ok:true（n_errors 0；n_warnings 62——均为常规 warning，无存量新错误）。

## 5. 未覆盖范围与 API 账本

- **GitCode/AtomGit issue 评论线程未拉取**（无低成本批量入口），全部按正文判定；本轮收录候选正文均含自包含复现/源码定位/对照实验，依赖评论改判概率低。
- **窗口内 152 条 PR** 按标题过滤，修复流正文仅深读 6 条与收录候选直接相关者；其余适配/性能/clean code 流未逐条深读。
- **GitHub 未认证 search**：21/21 组 200（tt-metal 2 组首试 504 后重试成功）；正文单点补拉 6 条（#66211/#66074/#9068/#66144 评论/#66146 PR/#6907）。评论线程仅拉 #66144。
- **留观未深读正文项**（下轮优先）：pytorch #5948（aclgraph wait_event 自锁，等修复落地）、ops-math #4617（TriangularSolve 空态，族代表候选）、ops-math #4616（SearchSorted dtype 混用越界读，待实测）。
- **arXiv** 仅 2 组主题词（SDC/bit-flip），KV corruption/数值正确性主题词未单独跑（与既往结果集重叠度高）。
- 本轮无任何候选因限流/配额只按索引判定；全部收录/留观判定均基于已读正文或批次缓存正文。

## 6. 结论

11 条新机制入库（待审计），3 条既有条目回访注记，1 条昨日留观转收（vllm#66144 EPLB 静默停摆，作者 follow-up 提供源码级 4 种 reset 泄漏定位），1 处昨日日报勘误（#8218 实为 gmm MXFP4 需求单非 BlockSparseAttention 故障）。本轮覆盖：NPU 22/22 源完整批次（223 条窗口内新建、71 条 issue 标题全过目、~28 条深读）+ GitHub 7 仓 3 关键词（21/21 组 200、4 新收）+ arXiv 2 组（无新论文）。重点新面：**vLLM 量化加载三连**（NVFP4/MXFP4 行并行 scale 组错配 GSM8K 崩至 1.4%、Humming E8M0 窗口重编码清零权重、RopeQuantize neox 越界——全部静默无警）；**torch_npu 编译器静默错值持续第三日**（分核归约尾块 mask 丢失 sum 错 3.5 倍、slice_scatter 改坏片外数据、pad-lane 除法 NaN）；**KV/状态域两条**（vllm-ascend 滑窗陈旧块跨请求污染、EPLB 静默停摆）；CANN 侧 abs complex64 归一化公式错误（26.5% 相对误差）与 MaxPoolGrad NHWC 解码错位。「空输入早退+输出未写」族继续高发（+3 例），错误码域 +6 例，按族留观。
