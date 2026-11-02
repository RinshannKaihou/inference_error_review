# 每日扫描日报 2026-11-02（推理库）

## 0. Git 前置检查

- 分支 `main`，上游 `origin/main`；`git status --porcelain -uall` 为空；`git fetch origin main` 成功；HEAD 与 origin/main 一致（0cdf3df248216070298927c56cd887f13530fe29）。通过，开始扫描。
- **Cron 前置脚本（fault_scan_npu_entry.py）退出码 1**：采集器对 `vllm-project/vllm-ascend` issues 第 5 页报 FetchError（`duplicate original number across pages: 19608`，GitHub 分页把同一 issue 重复返回），该仓标记 incomplete，水位未推进（仍 2026-11-01T23:00:43Z）。**其余 21/22 源全部 complete**，批次数据完整落盘。Git 前置检查在本侧独立复核通过，不构成 BLOCKED；按 NPU_SCAN.md「NPU API 部分失败保留可用证据继续」执行，缺口见第 5 节。

## 1. NPU 新增来源（共享批次）

- 批次：`/root/.hermes/profiles/third/cache/fault_scan_sources/batch-20261102T230043102753-pzqtceri`，采集窗口 2026-11-01T22:00:43Z → 2026-11-02T23:00:43Z（registry_hash 2f6422d0…，22 源中 21 complete、vllm-ascend issues 分页失败，generated 23:00:43Z）。
- inference scope 候选 23246 条（updated 滚动窗口）；窗口内新建（created_at ≥ since）827 条，其中 vllm-ascend 123 条（23 issue + 100 PR）。非 vllm-ascend 新建 704 条按仓：ops-nn 216（94 issue+122 PR）、ops-transformer 120（34+86）、ops-math 89（31+58）、pytorch 79（18+61）、hcomm 35（9+26）、hccl 36（7+29）、ge 25（2+23）、runtime 25（3+22）、op-plugin 18（6+12）、triton-ascend 26 PR、sgl-kernel-npu 17（1+16）、torchair 6（1+5）、mindspore 4、msmodelslim 6、mssanitizer 2、MindIE-LLM 0。
- vllm-ascend 缺口定量：issues 端点仅取回 4/5+ 页（updated ≥ 2026-10-07T06:57 的 626 条），第 5 页起校验失败。**窗口内新建 issue（23 条）updated 均在窗口内、确定已全部在批**；实际缺口是 updated 在 10-07~11-01 之间的旧 issue 状态回访面（约 1 页 100 条未入快照），PR 端点完整（validated 4/4）。昨日批次同因失败，该仓连续两轮不完整。

### 新增条目（NPU/GitHub-NPU 来源 5 条）

| 编号 | 来源 | 机制摘要 |
|---|---|---|
| INF-2366 | [vllm-ascend#20652](https://github.com/vllm-project/vllm-ascend/issues/20652) | resolve_block_scales 假定 TP 分片从全局块边界开始，每 rank 行数非 block_n 倍数时约半数行配错 scale，CPU 复现 66.6% 元素静默错值（max diff 1.8）；修复 #20653 open（offset 参数+加载期校验拒绝） |
| INF-2367 | [vllm-ascend#20677](https://github.com/vllm-project/vllm-ascend/issues/20677) | hybrid GDN(align)+MTP+async 下被 defer_block_free 暂停的请求丢 accepted 计数（只存 batch 行号临时量），下次 forward 读过期 recurrent 状态槽，复读环/乱码静默污染，GPQA 83–88→64–72；内核级因果探针+A/B 修复验证（3298→0） |
| INF-2370 | [ops-nn#7917](https://gitcode.com/cann/ops-nn/issues/7917) | MaxPool2dWithMask 310P(M200) 5 处实现 indices 尾部清零用 3 参 DataCopy 按元素计数，非 32B 倍数被硬件向下取整，尾部最多 7 个 int32 残留脏数据；官方样例 5x5=25 即触发；910B 路径有正确对照写法 |
| INF-2371 | [ops-nn#7954](https://gitcode.com/cann/ops-nn/issues/7954) | aclnnForeachDivScalarV2 x=fp16+scalar=DOUBLE（文档允许）时标量降 fp16，70000→inf 输出全 0；fp32/bf16 对照正确；同族统一修复 !13432 open |
| INF-2372 | [vllm-ascend PR#20595](https://github.com/vllm-project/vllm-ascend/pull/20595) | 310P Distill Qwen FP16 未掩码 QK 达 32.2 万（>65504）在 attention scale 前溢出，生成错误答案（eager/ACL Graph 一致错）；修复 Q 预乘 1/8，32960 token 与 CPU FP32 精确一致；open 未发布 |

### NPU 留观（窗口内代表）

- **#20609**（Qwen3.6-35B-A3B batched MTP 非 batch 不变，同批相同 prompt 从 token 13 分叉、logprob 超 tolerance 2.8x）——单序列对照通过、identical-prompt 批复现，定位到 batch>1 投机解码但**根因未定位**（报告自述「应作为共享问题追踪」）；与已收 INF-2365（sgl compressor 小 batch 累加序漂移）机制方向相邻但平台/组件不同，留观待根因。
- **#20622/#20666/#20673/#20674**（DeepSeek V3.2 MRV2 精度非确定性 parity 帖）——测试 harness 域，closed 的 #20622/#20666 已由 #20630/#20675 重开，无新机制证据，维持既往留观。
- **#20582**（投机接受率归零「负优化」+ 长时间 token 停滞）、**#20600**（TP8 910B vector core timeout 507034）、**#20607/#20608**（GDN metadata 重建/ACLNN workspace 不复用，性能域）、**#20610**（dsv4-flash 性能劣化）——可用性/性能域或不占静默故障号，留观。
- **修复 PR 增量**（标题级留观未深读正文，下轮择重）：#20653 修复（已随 INF-2366 收）、#20617（V2 EAGLE MLA drafts 按模型 tag 识别）、#20632/#20654（static kernel 黑名单）、#20658/#20660/#20661/#20662（QSFA static-kernel tiling 访存，A2/A3/A5 四连）、#20645（KVOffload 仅 offload prefix-cacheable 组）、#20681（HBM/KV pool 前缀保留）、#20682/#20683/#20689（MLA Prolog MXFP8/INT8 NZ 转换推迟）、#20664/#20665（revert KV save fence #17555）、#20634（MooncakeConnectorV1 kv_both）、#20593/#20599（KV cache views 共享 data_ptr 单独拷贝）、#20653 之外 **#20652/#20677 的修复动向**下轮回访。
- **ops-nn 校验/文档族增量**（#7925/#7929/#7945/#7946/#7967 Foreach 错误码表、#7924/#7961 QuantMatmul 报错、#7933 A8W4 WeightNZ 适配、#7037/#15178 MC2 废弃错误码、#7979 卷积反向 tiling 误报、#7981~#7991 ErrorCodes UT 批）——与 10-22~11-01 已判「校验缺失/错误码契约族」同型，按族留观维持。
- **ops-nn AI 探针/修复流**（#7965 dynamic_quant arch35 1ULP 除法+修复 !13392 已 merged、#7968 LpNormReduce FP32 分组归约+!13437、#7973 BNInference 中间精度损失+!13441、#7975 L2Normalize 极值+!13443、#7977 ForeachLog1p |x|<ulp(1)/2 返回 0、#7958 apply_came_part2 越界与未初始化读、#7938 dlqbmm 尾轮跨核 GM 重复写（同值 WAW，精度暂不受影响）+wqbmmv2 用例形状笔误）——机制方向真实，与 11-01 已判「AI 辅助源码探针族」同型，修复流多数已开，待 native ST/发布后评估；**#7917、#7954 已例外收录**（正文含真机复现/位模式证据，超出族内纯源码探针）。
- **ops-math**：#4412（StdMeanCorrection N≤correction 除零静默 +Inf/NaN 文档未声明）——文档契约域，与已收 INF-2356（aclnnInverse 静默降精度）同「文档允许组合静默错值」主题但机制不同（除零边界 vs dtype 降级），数值面重，留观下轮深读；#4421/!7036（space_to_batch_nd padding 越界拷贝）、#4422（RaggedBincount 长加权精度+GE splits 校验）、#4395（sparse_bincount 精度提升，closed）——长尾算子加固域留观；!7007（reduce_var unbiased=TRUE 且元素数 1 返回 NaN）——NaN 契约修复流 open 留观。
- **pytorch**：#5809（静态 Triton launcher 错 kernel 名首次加载报错）fail-loud 不占号；#5811（aclop lazy init 下 GE 拓扑未初始化影响 world_size 查询，closed）适配域；!48860（fractional_max_pool3d NPU lowering 起点精度+随机采样索引修复，merged）机制真实但正文为 PR 模板无细节，源 issue #5522 待下轮读；!48902（同步 2.13.0）、!48869/!48874（RootInfo 默认化）非故障域。
- **op-plugin**：#678/#679/!6156（npu_grouped_matmul_swiglu_quant_v2 文档/类型约束修正）——文档域；与 INF-2354（SwigluGroupQuant MXFP 越界写，11-01 收）相邻组件不同缺陷。
- **hccl/hcomm**：!3856/!3857（AIV 归约前跨卡写→读精度归因实验臂）——精度问题归因实验，非缺陷报告，关注后续结论；!6420（设备级访问器越界/口径统一）静态修复流留观；#1194/!6418（hccl_rootinfo.json 路径宏）非故障域。
- **msmodelslim #670/!1140**（量化问答机器人/评测集 feature）非故障域；**mssanitizer**、**triton-ascend 26 PR**、**ge !5713/!5720**（符号化元素数/TracingRecorder 越界修复，merged）编译器域修复流按惯例不逐条开号。

## 2. GitHub 常规 7 仓 + arXiv

- 认证状态：环境无 GITHUB_TOKEN（未认证）；search 走代理出口共享配额，请求间隔 ≥8 秒。21 组查询（7 仓 × correctness/silent wrong output/NaN，`is:issue created:>=2026-10-26`）全部 HTTP 200、无限流等待（累计等待 0）。
- 去重后 unique issue 132 条；与既往判定比对：vllm#64106/#64326/#64517/#64696/#64704/#64758/#63811/#64005、sglang#46572/#47245、transformers#50195/#50204/#50212/#50216、triton#12506/#12555、flashinfer#7687/#7980/#8119/#7962、TRT-LLM#20991/#21124、tt-metal quasar/tt-robot 族（#63619~#63684 批、#62506/#62557/#62605 等）均沿用 10-24~11-01 判定（INF-2237/2271/2238/2264/2166/2267/2234/2211/2266/2358/2359/2360 等）。

### 新增条目（GitHub 常规来源 2 条）

| 编号 | 来源 | 机制摘要 |
|---|---|---|
| INF-2368 | [TRT-LLM#21385](https://github.com/NVIDIA/TensorRT-LLM/issues/21385) | enable_swa_scratch_reuse=True 的 SWA per-layer KV 页表路径复用持久 host staging 做 non_blocking 异步拷贝，overlap scheduler 下一迭代覆写 host 源后 GPU 收到错误页表、attention 静默读错物理 KV 页；5/5 竞态复现全失配；#16088 已修同机制默认路径并点名为遗留 follow-up |
| INF-2369 | [tt-metal#63867](https://github.com/tenstorrent/tt-metal/issues/63867) | Quasar SFPU float→int typecast 经 SFPCAST mode 4 只保留 8 位小数再舍入，(0.5,0.5+1/256) 偶数上文被差一（2.503→2）；2^32 全域统计 Int8 327,552 错、Int16 大值 wrap；RTL 仿真逐输入证实；修复方案已给（单条 SFP_STOCH_RND 替换，更快且精确） |

### GitHub 留观（本轮判定）

- **vllm#65131**（v0.29/v0.32 sync KV load-failure 恢复触发 num_output_placeholders 断言）——fail-loud assert 域，不占静默故障号；#64855 沿用 10-31 判定。
- **transformers#50273**（QuantizedCache.crop() FP8/Quanto/HQQ 三后端全 IndexError、断 speculative 回滚）——**fail-loud 崩溃面**（明确 IndexError 非静默错值），且为 API 层缓存实现域；如后续证实静默路径（如 crop 部分成功）再评。
- **transformers#50224**（`_attn_implementation` setter 不达 deepcopy 构建的 encoder/decoder 栈）——closed（11-02），配置传播域，静态行为可观察非数值错值，留观。
- **tt-metal#63766**（Quasar softmax(dim=-1) DataflowBuffer 死锁 hang）、**#63765**（测试期望旧 TT_FATAL）——可用性/测试域留观；#63866（NaN PCC 通过的 packer L1 累加）标题级关注，下轮深读。
- **sglang#47508**（ROCm Kimi-K3 fused KDA 权重重载换 buffer 地址，graph replay 可能读 stale storage）——代码级地址替换已确认，但 AMD GPU 输出复现未跑、修复已含在 #43675，留观待 AMD 验证；**#47501** 沿用（性能域）。
- **flashinfer#8284**（0.7.3rc1 vs 0.7.2.post1 GSM8K 132 题单题翻转 130→129）——报告自述「kernel 缺陷、确定性复现、统计显著均未建立，随机解码可能是解释」，单题差异不构成机制证据，留观；#7903/#7962 性能域沿用。
- **vllm#65172**（Holemiss RFC：KV 部分丢失选择性恢复）与 vllm-ascend#20692（同名 RFC）——feature 提案域不占号。

### arXiv

- 2 组查询（"silent data corruption"+LLM 8 entries、"bit flip"+inference 8 entries，sortBy=submittedDate）HTTP 200 有效。无新论文：2610.33475/2610.18557/2609.19743/2608.30769/2608.11034/2604.10390/2604.00726/2502.12340 与 2608.29745/2609.06781/2610.03117/2610.06816/2610.08739/2610.27621/2610.30798=INF-2247/2610.37033 均已在库或已判（10-12~10-31 各轮）。

## 3. 回访（近 7 日条目 + 轮巡旧源）

- **INF-2361**（retention_interval 读死 env）：修复 #20546 仍 open（11-02 12:57 有更新），源 #20537 仍 open——修复未合入，已注记。
- **INF-2331**（ScatterNdUpdate）：修复 !13327 仍 open（11-02 03:19 更新），已注记。
- **INF-2354**（SwigluGroupQuant MXFP 越界写）：backport #20540 closed（11-01）；主修复 #20539 不在批次（updated 窗口外，此前已 merged），关闭原因未核实，已注记。
- **INF-2349/2338/2348**（源 #20510/#7830/#6975 均此前已 closed）：本批快照状态无变化，维持注记（closed≠已修复发布）。
- 近 7 日其余源（#20550/#20537/#20532/#7004/!15092/#922/#5787/#5788/#5789/#5791/#7880/#7888/#7889/#7901）批次快照均 open 无状态变化；INF-2320 修复 !13012 仍 merged、源 #7736 closed 维持。
- 轮巡（更早入库）：INF-2259 源 ops-math#6910、INF-2254 源 #6872 连续第三轮在滚动窗口外（updated 久于 10-07），状态未核；下轮按 core API 单点核验。INF-2260 源 #7640 本批亦未入快照。

## 4. 编目与视图

- 新写入 **7 条**（全部 pending_audit）：INF-2366 ~ INF-2372，编号连续。映射（以 catalog YAML 为准）：

| 编号 | 来源 | 标题 |
|---|---|---|
| INF-2366 | vllm-ascend#20652 | block-FP8 TP 分片反量化静默错值 |
| INF-2367 | vllm-ascend#20677 | defer_block_free 丢 MTP 计数读过期状态槽 |
| INF-2368 | TRT-LLM#21385 | SWA KV 页表 staging 竞态读覆写数据 |
| INF-2369 | tt-metal#63867 | Quasar SFPCAST typecast 尾数截断错值 |
| INF-2370 | ops-nn#7917 | MaxPool2dWithMask M200 尾部脏 indices |
| INF-2371 | ops-nn#7954 | ForeachDivScalarV2 DOUBLE 标量降 fp16 全 0 |
| INF-2372 | vllm-ascend PR#20595 | 310P Qwen FP16 QK 溢出错答案 |

- 回访 update：INF-2361、INF-2331、INF-2354 三条注记（见第 3 节）。
- 训练库跨库查重：7 个拟收 URL 在 training_error_review `find` 全部 0 命中，本轮无 cross 关系。
- `catalog_tool.py stats`：2372 条（active 2365 + pending_audit 7）；`build` 完成；`validate` ok:true（n_errors 0；n_warnings 62——较昨日 55 增 7，均为本轮新条目 pending_audit 常规 warning，非存量新引入错误）。

## 5. 未覆盖范围与 API 账本

- **NPU vllm-ascend 缺口**：issues 端点第 5 页 FetchError（GitHub 分页重复返回 #19608），仅 4 页 validated；快照含 updated ≥ 2026-10-07 的 626 条 issue。窗口内新建 23 条 issue 已全覆盖；**未覆盖**：updated 在 10-07~11-01 间且排序落在第 5 页之后约 100+ 条旧 issue 的状态回访面。连续两轮同因失败，建议采集器对 GitHub 端点加去重重试或改按 created 排序补页（属采集器代码改进，本轮未动）。
- GitCode/AtomGit issue 评论线程未拉取（无低成本批量入口），全部按正文判定；ops-nn AI 探针族（#7938/#7965/#7968/#7973/#7975/#7977/#7958）自述非 native NPU 执行或修复流已开，按源码级/修复流留观。
- vllm-ascend 修复 PR 增量约 20 条（#20617/#20632/#20645/#20658~#20662/#20664/#20681~#20689 等）标题级留观未深读正文；pytorch !48860 源 issue #5522 未读。
- GitHub：21 组 search 全 200 无限流（累计等待 0）；新候选按 search item 自带 body 判定，未拉评论线程——若评论中出现维护者定位/驳回可能需改判。未认证配额未耗尽。
- arXiv 仅 2 组主题词（SDC/bit-flip），KV corruption/数值正确性主题词未单独跑（与既往结果集重叠度高，10-17 已跑过 6 组）。
- 回访轮巡：ops-math#6910/#6872 连续三轮窗口外未核状态；#7640 本轮未入快照。

## 6. 结论

7 条新机制入库（待审计），3 条既有条目回访注记。本轮覆盖：NPU 21/22 源完整批次（827 条窗口内新建、强信号全深读）+ GitHub 7 仓 3 关键词（21 组全 200、132 unique、2 新收）+ arXiv 2 组（无新论文）；缺口集中在 vllm-ascend issues 分页失败（连续两轮，同因 GitHub 分页重复条目）、GitCode 评论线程与修复 PR 正文。重点新面：block-FP8 TP 分片反量化静默错值（INF-2366，量化域最重）、defer_block_free 丢 accepted 计数污染 recurrent 状态（INF-2367，静默错值+任务质量实测 83→64）、TRT-LLM SWA KV 页表 staging 竞态（INF-2368，KV 域竞态）。
