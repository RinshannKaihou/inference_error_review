# 每日扫描日报 2026-11-01（推理库）

## 0. Git 前置检查

- 分支 `main`，上游 `origin/main`；`git status --porcelain -uall` 为空；`git fetch origin main` 成功；HEAD 与 origin/main 一致（7e36238b87f927339894b99437c5f662cd6adf80）。通过，开始扫描。

## 1. NPU 新增来源（共享批次）

- 批次：`/root/.hermes/profiles/third/cache/fault_scan_sources/batch-20261101T230043052835-0urobqq6`，采集窗口 2026-10-31T22:00:43Z → 2026-11-01T23:00:43Z（registry_hash 2f6422d0…，22 源全 complete、failures 空，generated 23:00:43Z）。
- inference scope 候选 22325 条（updated 滚动窗口）；窗口内新建（created_at ≥ since）167 条：ops-nn 72（30 issue+42 PR）、vllm-ascend 36（7+29）、ops-transformer 16（6+10）、pytorch 14（11+3）、ops-math 15（6+9）、triton-ascend 4、hcomm 3、sgl-kernel-npu 3、op-plugin 2、runtime 1、mindspore 1；hccl/ge/torchair/msprobe/mssanitizer/mindformers/msmodelslim/MindSpeed 系窗口内无新建。
- 标题强信号 12 条全部深读正文（本地 body 缓存），另深读 GitHub 侧 5 条正文；拟收 13 条完成近 14 天日报完整 URL 查重 + `catalog_tool.py find`（本库与训练库）查重，全部无撞号。478 条窗口外更新仅用于回访。

### 新增条目（NPU/GitHub-NPU 来源 11 条；编号已按 YAML evidence 核对）

| 编号 | 来源 | 机制摘要 |
|---|---|---|
| INF-2352 | [pytorch#5789](https://gitcode.com/Ascend/pytorch/issues/5789) | 310P3 一次 INNER-ABORT（fp64/bicubic 等 8 格）后同进程所有 NPU 算子静默返回未初始化垃圾（fp32 matmul rel 8.06e+17，全有限非 NaN），A3 干净报错不毒化；#5788 为同机制第二来源 |
| INF-2350 | [pytorch#5787](https://gitcode.com/Ascend/pytorch/issues/5787) | 310P3 kthvalue 值路径 fp16 内部量化（k=1/5/32 vs fp16 舍入差恰 0），返回不存在于输入的数；indices keepdim 缺维为四芯共性；var→INF-2323 |
| INF-2357 | [pytorch#5791](https://gitcode.com/Ascend/pytorch/issues/5791) | max_pool2d(return_indices) indices 契约四芯四态（310P3 float32+值面失真、A3/910B3 int8+形状破裂、950PR int32），无一符合 CPU 契约 |
| INF-2353 | [vllm-ascend#20550](https://github.com/vllm-project/vllm-ascend/issues/20550) | 省略 --block-size 时 refresh_block_size(128) 与 worker 后端偏好(16)失配，DeepSeek V4.1 Flash 全端点输出乱码（reasoning 复述错误校验码/content null），显式 128 正常 |
| INF-2361 | [vllm-ascend#20537](https://github.com/vllm-project/vllm-ascend/issues/20537) | AscendHybridKVCacheCoordinator 读已删除 env 恒 None+不调 super+不转发，retention_interval 完全失效，非法值 4097 静默通过校验；修复 #20546 open |
| INF-2351 | [vllm-ascend#20532](https://github.com/vllm-project/vllm-ascend/pull/20532) | group_cache_specs 用 zip() 转置 SWA 层，39 层骨干静默丢层 36/37/38 缓存；修复改 zip_longest open |
| INF-2354 | [vllm-ascend#20539](https://github.com/vllm-project/vllm-ascend/pull/20539)（merged，backport #20540） | SwigluGroupQuant MXFP 尾部四连存独立计数器越行覆写 UB，Qwen3.8-Next W4A8（中间维 640）量化激活错误/不稳定；GPQA 90.91% 验证 |
| INF-2362 | [ops-transformer#7004](https://gitcode.com/cann/ops-transformer/issues/7004) | lightning_indexer arch22 top-k 回读缺 MTE3→MTE2 依赖，MTE2 读陈旧 workspace 污染 top-k 索引/值（行号级定位）；修复 !15092 open |
| INF-2363 | [ops-nn#7889](https://gitcode.com/cann/ops-nn/issues/7889) | aclnnAdvanceStep numQueries>40（AIV 核数）时下标≥40 序列四输出全不更新（60/100 序列 seqLens 未 +1），返回 0/0 |
| INF-2355 | [ops-nn#7901](https://gitcode.com/cann/ops-nn/issues/7901) | aclnnApplyAdamWQuant mRef 恒 0（预填 127 仍写 0），下一步 0 号索引反量化致 absmaxMRef 系统性偏小（0.2013 vs 0.5268） |
| INF-2364 | [ops-nn#7888](https://gitcode.com/cann/ops-nn/issues/7888) | aclnnApplyAdamWQuant kernel 固定按 4096 元素读写，numel=256 越界读写（mssanitizer 稳定 14 条）；+8KB slack 吸收=普通分配下静默腐蚀邻接内存 |
| INF-2356 | [ops-nn#7880](https://gitcode.com/cann/ops-nn/issues/7880) | aclnnInverse out/self dtype 不一致（fp32→fp16/bf16）返回 0 静默降精度写出（1/3→0.33398），只查 cast 可行性不查一致性 |
| INF-2365 | [sgl-kernel-npu#922](https://github.com/sgl-project/sgl-kernel-npu/pull/922) | arch22 compressor 小 batch 换 D tile 宽度改变 K 块累加序，prefill vs verify 同列 0.56640625 vs 0.5625、状态漂移；修复按缺陷处理（open） |

（另 3 条 GitHub 常规来源见第 2 节表。）

### NPU 留观（窗口内代表）

- **校验缺失/错误码契约族增量**（ops-math#4390/#4391/#4392/#4394、ops-nn#7877/#7878/#7879/#7882/#7883/#7884/#7885/#7886/#7887/#7890/#7903、ops-nn#7894、hcomm#1188、mindspore#29708、op-plugin#673 等）— 与 10-22~10-31 已判大族同型（文档 dtype/错误码/边界校验类），按族留观维持。个别说明：#4391（SVD 空输入 U/V 未写返回成功）、#7892（AdaptiveAvgPool2d H=0 返回 SUCCESS 输出未写）数值面有脏数据风险，但属空输入边界族且修复流（!6976/!13324 等）已开，维持族内留观；**hcomm#1188**（RtsqA5 stale sqTail_ 回退硬件 SQ 尾、双提交者竞态，修复 !6402 open）机制清晰（三本账不同步），但表现为硬件 errorType:9 fail-loud，未证实静默计算影响，留观；**mindspore#29708**（MindSpeed MSAdapter all_to_all_forward 对 list 调 .tolist() 抛 AttributeError + input_split_sizes 误判 output）为 fail-loud 适配层缺陷（240 次 HCCL 对照，60 次预期异常），非静默，留观。
- **AI 辅助源码探针族**（ops-math#4389 TriangularSolve unitriangular NaN/Inf 对角保留 + 修复 !6974、ops-nn#7876 Addmv 小 alpha/beta float-epsilon 路由误零 + 修复 !13323、ops-nn#7904 CTCLoss 长度聚合溢出）— 机制方向可信（NaN 传播、非零系数被 FillZeroOutput 替换属真数值缺陷），但证据均为「源码提取+CPU doubles 执行」，自述非 native SDK/NPU 执行，且修复 PR 已开（WIP）；按「未在设备复现的候选修复流」留观，待 native ST 结果再评估。
- **ops-nn ApplyAdamW 修复流**（!13291 docs、#7891 bias correction t=step+1 相对差 1.14e-2、#7885/#7890 V2 校验）— #7891 数值偏差明确但属优化器域（训练侧为主），与已收 INF-2320 族同接口，留观下轮（训练库如需可另收）。
- **pytorch#5796**（950DT tensor.to(float64) 静默不生效无提示）、**#5794**（HF32 默认开启对 Conv2d 精度影响未文档化）— 静默降级主题真实，但 #5796 报告者自述「现场未保留完整 dtype 输出、按脚本复现确认」＝无实测输出凭证，#5794 属文档/提示缺失（与已收 INF-2323 HF32 面相邻），均留观待复现凭证；**#5790**（A3 bucketize SIGSEGV/bincount f64 FuncError）fail-loud 崩溃面，不占号；**#5787 已收**（INF-2350）；#5792/#5793/#5795/#5797 为 Inductor 编译报错/CI/feature 域。
- **vllm-ascend 增量 PR**：#20538（sparse-indexer 懒 store init）、#20543（多组 bulk load 失败按请求隔离）、#20545（spec-decode 组元数据 buffer 全容量传递）、#20555（GLM KPool ND/NZ 范围）、#20558（Qwen4Exp MTP QSA PP 下选错投影——有明确错径机制但堆叠在未合入的 #20311 PP 工作上）、#20559（native KV offload 交错 kernel block）、#20561（KV-pool RPC deadline）等未深读正文，按标题+机制方向留观下轮补查；#20547/#20549/#20563 为 feature/RFC/recipes 不占号；#20530 nightly CI 帖、#20531 已 closed 的文档帖。
- **ops-transformer 白盒源码帖**：#7006（MhcPreSinkhornPremix h_pre 梯度被丢，premix=None 链 x/phi/alpha/bias 梯度错）、#7007（空 sparse band tiling 除零）、#7002（CausalConv1d golden 短 prefill 报错）、#7003/#7005（950 性能重构）— #7006 为真梯度丢弃机制但属训练侧路径（sinkhorn premix 反向），本库留观（训练库候选）；#7007 fail-loud 域；#7002 golden 资产域；修复流 !15096/!15097/!15089 已开。
- **triton-ascend**（#2910~#2913 TLA/TTIR feature 流）、**sgl-kernel-npu #921/#923**（构建域）、**ops-nn !13292~!13321**（910_93 实验算子批量）、**ops-math !6975~!6982、ops-nn !13325/!13326/!13286~!13290**（对应上述校验族修复流）、**runtime !6025**（docs）、**hcomm !6403**（建链收编）— 按惯例不逐条开号。

## 2. GitHub 常规 7 仓 + arXiv

- 认证状态：环境无 GITHUB_TOKEN（未认证）；search 走代理出口共享配额，请求间隔 ≥7 秒。21 组查询（7 仓 × correctness/silent wrong output/NaN，created:>=2026-10-25）全部 HTTP 200、无限流等待（累计等待 0）。窗口内合计命中：vllm 34/9/5、sglang 15/2/1、transformers 2/3/2、TRT-LLM 2/1/1、triton 1/1/0、tt-metal 6/14/18、flashinfer 33/4/6。
- 去重后 unique issue 142 条（created≥10-25）；与既往判定比对：vllm#64758/#64696/#64704/#64855/#64106/#64326/#63811、sglang#47245/#46572/#46476、transformers#50195/#50204/#50212、tt-metal quasar 族（#62436/#62465/#62605/#62613/#63257/#63259/#63277 等）、flashinfer#8119/#7980/#7687、triton#12555/#12506 等均已收录或沿用 10-24~10-31 判定（INF-2264/2237/2179/2166/2267/2238/2269/2270/2336/2343/2344/2345/2347 等）。
- transformers#50212（Trainer 模型并行下 floor-divide 有效 token 数）训练侧 resume/优化器域，昨日已判留观维持；**#50216 深读后新收**（见下表）。

### 新增条目（GitHub 常规来源 3 条）

| 编号 | 来源 | 机制摘要 |
|---|---|---|
| INF-2358 | [tt-metal#63629](https://github.com/tenstorrent/tt-metal/issues/63629) | ttnn.batch_norm bf16 输入+fp32 参数+fp32_dest_acc_en=False：2048 值 1856 错（最大偏 2.5、1728 恰 0），随机数据 87% 输出偏差>5%，文档列该组合为支持且不提该开关 |
| INF-2359 | [tt-metal#63683](https://github.com/tenstorrent/tt-metal/issues/63683) | ttnn.experimental.slice_write RM BLOCK 分片行非 16B 倍数时恰好半数输出静默错值（8~36B 行 3/3 进程复现，16/32B 精确）；offset 核网格另致 3/3 挂起 |
| INF-2360 | [transformers#50216](https://github.com/huggingface/transformers/issues/50216) | ModernBERT mean pooling fp16 全 token 求和溢出 65504 输出 NaN logits（mmBERT-small 2048 token 起、四 checkpoint 8192 内必溢出；CUDA autocast 理论不触发） |

### GitHub 留观（本轮判定）

- **tt-metal tt-robot 新批**（#63619 非 32x32 tile 按 32x32 读、#63622 update_cache position≥S 落他行、#63627 untilize ND 分片不完成、#63628 单设备 op 按输入 layout 填输出、#63631 matmul/roll bad optional、#63642 CCL 忽略输出 config、#63672 full_like bf16 截断、#63682 sdpa share_cache 越界写、#63684 moreh 不校验预分配输出）— tt-robot 自动报告批，与 10-17~10-31 已判「quasar/LLK 大家族」同模式（多为静默错值+守卫缺失），其中 #63682（share_cache=True 越过 1-user 输出写）数值面重，但正文未逐条深读，按 tt-robot 批量族留观，下轮择重深读；#63627（untilize 挂起）为可用性域留观。
- **flashinfer#7687** 沿用 10-29/10-30 留观（CI 聚合帖）；#7209 沿用 10-29 留观。**vllm#64517**=INF-2271 已收回访仍 open。

### arXiv

- 2 组查询（"silent data corruption"+LLM（http export，8 entries）、"bit flip"+inference（10 entries））HTTP 200 有效。仅 1 篇窗口内新：2610.37033（Rowhammer 放大 LLM 推理延迟/成本，2026-10-27）— 可用性/成本攻击域而非静默正确性，留观不入库；其余（2610.33475/2609.19743/2608.11034/2604.00726/2610.27621/2610.06816/2609.06781/2608.23774）均已在库或已判。

## 3. 回访（近 7 日条目 + 轮巡旧源）

- **INF-2349**（GLM-5.3-Flash PCP 混合批）：源 vllm-ascend#20510 已 closed（11-01 09:08Z），关闭原因未核实（closed≠已修复发布），已注记。
- **INF-2338**（Index 尾维省略误判连续）：源 ops-nn#7830 已 closed（10-31），关闭原因未核实，已注记。
- **INF-2348**（FA/FD minCheckValue）：源 ops-transformer#6975 已 closed，修复 !15023 此前 merged，是否随版本发布未核实，已注记。
- **INF-2323**（310P3 fp32 GEMM fp16 降级）：补证据 #5787（kthvalue 值面同源量化，见 INF-2350），并修正「仅 Cube 路径」边界表述，已 update。
- **INF-2320**（ApplyAdamWQuant NaN）：同接口两新案 #7901/#7888 已另立 INF-2355/INF-2364；源 #7736 closed、修复 !13012 merged 发布未核实，已注记。
- **INF-2331**（ScatterNdUpdate）：修复 MR !13327 已开（11-01 18:36Z 更新）仍 open，已注记。
- 其余近 7 日源（#7848/#5785/#5786/#4356/#664/#666/#665/#4364/#4367/#4366/#7822/#7853/#7858/#7867/#7869/#7806/#7835/!13251、triton-ascend#2903）批次快照均 open 无状态变化（#7858 11-01 有更新仍 open）；已收修复 PR（!6937/?/!15041/!13239）均 open 无进展。
- 轮巡（更早入库）：INF-2260 源 #7640 open 无变化；INF-2259 源 #6910、INF-2254 源 #6872 本批滚动窗口外（fetched 0/未入快照），状态未核；INF-2346 源 triton-ascend#2903 open。ops-math#6910/#6872 连续两轮不可核，下轮考虑 core API 单点。

## 4. 编目与视图

- 新写入 **16 条**（全部 pending_audit）：INF-2350 ~ INF-2365，编号连续。映射（以 catalog YAML 为准）：

| 编号 | 来源 | 标题 |
|---|---|---|
| INF-2350 | pytorch#5787 | kthvalue 310P3 值面 fp16 量化（var→INF-2323） |
| INF-2351 | vllm-ascend#20532 | SWA zip() 静默丢尾层缓存 |
| INF-2352 | pytorch#5789 | INNER-ABORT 后进程级毒化静默垃圾 |
| INF-2353 | vllm-ascend#20550 | 省略 --block-size 元数据失配乱码 |
| INF-2354 | vllm-ascend#20539 | SwigluGroupQuant MXFP 尾部越界写 |
| INF-2355 | ops-nn#7901 | ApplyAdamWQuant mRef 恒 0 |
| INF-2356 | ops-nn#7880 | aclnnInverse dtype 不一致静默降精度 |
| INF-2357 | pytorch#5791 | max_pool2d indices 四芯契约分裂 |
| INF-2358 | tt-metal#63629 | batch_norm fp32 参数 16 位 Dest 错值 |
| INF-2359 | tt-metal#63683 | slice_write 分片行非对齐半数错值 |
| INF-2360 | transformers#50216 | ModernBERT mean pooling fp16 溢出 NaN |
| INF-2361 | vllm-ascend#20537 | retention_interval 读死 env 失效 |
| INF-2362 | ops-transformer#7004 | lightning_indexer MTE3→MTE2 竞态 |
| INF-2363 | ops-nn#7889 | AdvanceStep numQueries>40 不更新 |
| INF-2364 | ops-nn#7888 | ApplyAdamWQuant 固定 4096 越界读写 |
| INF-2365 | sgl-kernel-npu#922 | compressor 小 batch 累加序漂移 |

- var 关系回显核对：INF-2350 → INF-2323「310P3 fp32 GEMM/conv 静默按 fp16 输入精度计算」✓。
- 更新 6 条：INF-2349/2338/2348/2320/2331（回访注记）、INF-2323（补证+边界修正）。
- 训练库跨库查重：13 个拟收 URL + tt-metal/transformers 新源在 training_error_review `find` 全部 0 命中，本轮无 cross 关系。
- `catalog_tool.py stats`：2365 条（active 2349 + pending_audit 16）；`build` 完成；`validate` ok:true（n_errors 0；n_warnings 55 均为存量遗留——旧 var 目标 slug 无法解析与模式占位符，非本轮引入）。

## 5. 未覆盖范围与 API 账本

- GitHub：21 组 search 全 200 无限流（累计等待 0）；新候选按 search item 自带 body 判定，未拉评论线程——若评论中出现维护者定位/驳回可能需改判。
- NPU：GitCode/AtomGit issue 评论线程未拉取（无低成本批量入口），全部按正文判定；ops-math#4389/#7876/#7904 等 AI 辅助探针族自述非 native NPU 执行，按源码级证据留观；vllm-ascend 修复 PR 增量（#20538/#20543/#20545/#20555/#20558/#20559/#20561）标题级留观未深读正文。
- tt-metal tt-robot 新批 9 条未逐条深读正文，按家族留观，下轮择重（#63682/#63619 候选）。
- arXiv 仅 2 组主题词（SDC/bit-flip），KV corruption/数值正确性主题词未单独跑（与既往结果集重叠度高）。
- 回访轮巡：ops-math#6910/#6872 连续在滚动窗口外，未核状态。

## 6. 结论

16 条新机制入库（待审计），6 条既有条目回访/补证。本轮覆盖：NPU 22 源完整批次窗口（167 条窗口内新建、强信号 12 全深读）+ GitHub 7 仓 3 关键词（21 组全 200、142 unique、3 新收）+ arXiv 2 组（1 新论文留观）；缺口集中在 GitCode 评论线程、tt-robot 批量正文与 vllm-ascend 修复 PR 正文。重点新面：310P3 INNER-ABORT 进程级毒化（INF-2352，静默错值最重形态）、vLLM-Ascend block-size 失配乱码（INF-2353）、MXFP 量化尾部越界写已修复合入（INF-2354）。
