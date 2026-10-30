# 每日扫描日报 2026-10-30（推理库）

## 0. Git 前置检查

- 分支 `main`，上游 `origin/main`；`git status --porcelain -uall` 为空；`git fetch origin main` 成功；HEAD 与 origin/main 一致（ae18767703049d3c1c3fea1b48022cbabe51ba38）。通过，开始扫描。

## 1. NPU 新增来源（共享批次）

- 批次：`/root/.hermes/profiles/third/cache/fault_scan_sources/batch-20261030T230043149429-j_rii0a4`，采集窗口 2026-10-29T22:00:43Z → 2026-10-30T23:00:43Z（registry_hash 2f6422d0…，22 源全 complete、failures 空）。
- inference scope 候选 21402 条；窗口内新建（created_at ≥ since）835 条（ops-nn 170、ops-math 141、ops-transformer 136、vllm-ascend 105、pytorch 95、hcomm 35、hccl 32、runtime 31、msmodelslim 17、torchair 14、ge 14、op-plugin 13、triton-ascend 10、sgl-kernel-npu 9、msprobe 7、mindspore 5、MindIE-LLM 1）；标题强信号命中 80 条、正文关键词补命中 159 条。
- 深读正文约 28 条（本地 body 缓存）；全部收录候选完成近 14 天日报完整 URL 查重（26 个拟收 URL 0 既往判定）+ `catalog_tool.py find` 查重（0 命中）+ 训练库跨库查重（0 命中，无 cross 关系）。

### 新增条目（NPU 来源，22 条）

| 编号 | 来源 | 机制摘要 |
|---|---|---|
| INF-2290 | [ops-math#4272](https://gitcode.com/cann/ops-math/issues/4272) | LinalgCross fp64 FMA 残差：cross(x,x)≠0（e-18）、溢出区 ±inf 而非 NaN，残差随量级至 6.29e291 |
| INF-2291 | [ops-math#4325](https://gitcode.com/cann/ops-math/issues/4325) | Gcd INT16 A3 混合取值错值（gcd(1,16384)=6383=16384−10001），元素≥3 且取值不一致触发，错误率至 30.8% |
| INF-2292 | [ops-math#4306](https://gitcode.com/cann/ops-math/issues/4306) | Signbit(-0.0)=False（资料要求 True），-NaN 同错，四 dtype 一致；torch NPU 同暴露 |
| INF-2293 | [ops-nn#7741](https://gitcode.com/cann/ops-nn/issues/7741) | SituGlu rank≥26 两段 0/0、ws=0、输出金丝雀未写；rank 9–25 超声明却正常（var→INF-2260） |
| INF-2294 | [ops-nn#7739](https://gitcode.com/cann/ops-nn/issues/7739) | Shrink lambd/bias INT64/BF16 标量放行并静默截断（WIP !13146；Softshrink #7737 同型） |
| INF-2295 | [ops-math#4298](https://gitcode.com/cann/ops-math/issues/4298) | ReduceSum bool→INT64 负轴 size_t 越界致 promoteType 误选 fp32，和 16785409→16785408（源码定位+判别矩阵） |
| INF-2296 | [ops-math#4301](https://gitcode.com/cann/ops-math/issues/4301) | Remainder fp32 负除数中间乘积溢出返 -inf/0.0（35 组 19 命中，全族三入口复现，fp64 正确） |
| INF-2297 | [ops-math#4303](https://gitcode.com/cann/ops-math/issues/4303) | RightShift 无符号 shiftBits≥bitWidth 钳到 bitWidth-1 输出 1（资料要求 0；有符号分支正确） |
| INF-2298 | [ops-nn#7747](https://gitcode.com/cann/ops-nn/issues/7747) | SCEWithLogits labels 形状不一致两段 0，msSanitizer 72 处 illegal_write（最大 564B）越界写 |
| INF-2299 | [ops-transformer!14989](https://gitcode.com/cann/ops-transformer/merge_requests/14989) | PromptFA fp16 HighPerformance 分子 fp16 累加，Q/K 全零时 (i+1)*V>65504 输出连续 inf；修复切 fp32 累加 |
| INF-2300 | [ops-transformer#6944](https://gitcode.com/cann/ops-transformer/issues/6944) | MoeGatingTopK sigmoid 1 ULP 差在 top-10 边界打破平局，expert 122→472 翻转，grad_norm 整网放大 |
| INF-2301 | [vllm-ascend#20367](https://github.com/vllm-project/vllm-ascend/issues/20367) | PD stash 覆盖：decode TP<prefill TP 时 GQA draft KV 组被丢，未 reformat 布局被消费，仅推测接受长度坍缩 |
| INF-2302 | [vllm-ascend#20391](https://github.com/vllm-project/vllm-ascend/issues/20391) | Mooncake register_buffer 非零返回码仅记日志，带病起服务后 KV 传输延迟失败 |
| INF-2311 | [ops-nn#7770](https://gitcode.com/cann/ops-nn/issues/7770) | SwiGluQuant 动态量化 smoothScales=[G]（资料声明支持）kernel 按 [G,N] 寻址越界读 120/128B，digest 随相邻内存变 |
| INF-2306 | [ops-nn#7772](https://gitcode.com/cann/ops-nn/issues/7772) | SwiGluQuant groupIndex 不校验取值/覆盖，[1,1] 未覆盖行整段不乘 smooth 输出偏离 |
| INF-2312 | [ops-transformer#6939](https://gitcode.com/cann/ops-transformer/issues/6939)+[!14970](https://gitcode.com/cann/ops-transformer/merge_requests/14970) | FA TND 空 KV batch 清零分段按 coreNum*2 切分而 A6 只有 coreNum AIV，一半行残留脏数据+OOB 清零写 |
| INF-2313 | [ops-transformer#6953](https://gitcode.com/cann/ops-transformer/issues/6953) | attention_update tiling N 窄化 uint32 无检查（静态分析+Host 模型，未实机；实际可达性低） |
| INF-2316 | [ops-transformer#6962](https://gitcode.com/cann/ops-transformer/issues/6962) | compressor 公开入口 autograd forward 内 no_grad 判据恒 False，中间量未写、backward 梯度 rel 1.1~2.8/nan |
| INF-2305 | [ops-nn#7756](https://gitcode.com/cann/ops-nn/issues/7756)/[#7757](https://gitcode.com/cann/ops-nn/issues/7757) | SquaredRelu 偏小 out 越界写 16/24 元素、非连续 out 按连续写 4/8 逻辑元素错位 |
| INF-2307 | [ops-math#4344](https://gitcode.com/cann/ops-math/issues/4344) | AmpUpdateScale fp16 scale≥32768 增长分支无溢出保护输出 inf（资料要求保持；WIP !6903） |
| INF-2308 | [ops-nn#7762](https://gitcode.com/cann/ops-nn/issues/7762)+[!13131](https://gitcode.com/cann/ops-nn/pull/13131) | BN3DTrainingReduce 950 FP16 短行直加+分核丢补偿量（相消输入 -1 vs -0.998）；修复已 merged |
| INF-2309 | [ops-nn#7791](https://gitcode.com/cann/ops-nn/issues/7791) | FakeQuantPerChannel 倒置通道 clamp 序与 TF 相反、相等通道 0 vs NaN，22 例 L0 失败（matched_ratio≈P(min<max)） |
| INF-2320 | [ops-nn#7736](https://gitcode.com/cann/ops-nn/issues/7736)+[!13012](https://gitcode.com/cann/ops-nn/pull/13012) | ApplyAdamWQuant arch35 零 block 0/0=NaN 码值错（gnormScale=0 组 13 FAIL→修复 159/169 PASS） |

### NPU 拒收/留观（窗口内代表）

- **校验缺失/错误码契约大族增量**（ops-math#4334/#4331/#4327/#4312/#4310/#4308/#4285/#4298/#4294、ops-nn#7795/#7787/#7781/#7744/#7789/#7765/#7749、ops-transformer#6924/#6925 等）— 与 10-22~10-29 已判大族同型（非法入参静默接受/错误码语义/文档声明不一致），按族留观维持。其中 #4334（Std 全归约跳 shape 校验）、#4331（KlDiv 空输入 out 未写）与已判大族同机制但正文证据完整，若后续维护者确认可升级；本轮按族处理未单列。
- **错误码/文档类**：ops-math#4312/#4310/#4308/#4285/#4284/#4287/#4294、ops-nn#7753/#7750/#7751、ops-transformer#6924/#6965（logging %s）、runtime#1302/#1296（代码风格）— 文档/错误码契约类不占号。
- **ops-math#4305**（HistogramFixedWidth golden 整数溢出）— 缺陷位于 CPU golden/输入准备（测试资产），非设备 kernel；按「测试判定」域不占故障号，留观。**MR !6898**（scale PRIVATE 未初始化读取修复）、**!6901**（silent_check_v2 null tensor）— 修复流正文较薄，无独立数值后果证据，留观。
- **pytorch#5744**（_lazy_call 并发丢延迟回调）— 机制清晰（无锁检查+队列追加竞态致 manual_seed/RNG 状态静默丢失），但影响面是 RNG 种子设置丢失，属「设置丢失」非直接计算错值，且报告者仅以持锁模拟演示、无自然触发频率证据；留观下轮（如获自然触发证据再收）。**#5754**（save_cache_hook 漏传参崩溃）fail-loud 不占号；**#5764/#5765**（profiler 相关）、**#5750/#5763**（inductor 编译失败/LLVM ERROR）fail-loud。
- **vllm-ascend#20363**（DSV4.1-Flash A3 GatherV2 SMMU fault+poison 0xa5a5a5a5）— 图捕获 dspark decode 首次真实流量崩溃，fail-loud 崩溃（507035），毒化标记与 #20139 系同签名但根因未定位，留观；**#20349**（MiniMax M3.1 部署）、**#20329**（>200 token 死锁）、**#20353**（Qwen3.8 Flash-Next 环形缓存外部位命中归零，集成构建报告且版本线非公开 release）、**#20342**（fabric mem）— 部署/hang/集成构建域，机制未定位或证据在私有构建，留观。
- **msmodelslim#655**（GLM5.3 W4A4C8 minmax 精度低+mse_round 报错）—「精度低」为量化方案选择问题（用户明确以 ceilx/mseround 提精度为预期路径），mse_round 报错细节未附 traceback 之外的机制证据，与 10-29 !1126 判定一致域，留观；**!1134**（主配置切 mse_round/ceil_x）为方案切换非缺陷修复。
- **triton-ascend/torchair**：torchair#916（盘古 celoss+muon 入图精度 FAIL autofused add_div_linalg_vector_norm 系列）— 有精度 FAIL 证据但正文仅报错片段，无输入/机制定位，CANN 版本为商发 9.2.0，留观下轮补查；triton-ascend MR 均为功能/构建流。
- **PR 修复流**（ops-nn !13110~!13198 批量加固、ops-math !6854~!6911、ops-transformer !14932~!15014、op-plugin/pytorch/hccl/hcomm 例行）— 白盒修复流按惯例不逐条开号（已收录的 !14989/!14970/!13131/!13012 除外）；hccl !3811（静态扫描除零守卫）正文空模板无独立证据。
- **MindIE-LLM/msprobe/mssanitizer/mindspore/mindformers** 窗口内无静默数值候选（msprobe#661 为 RFC 设计帖、#663 文档测试汇总）。

## 2. GitHub 常规 7 仓 + arXiv

- 认证状态：环境无 GITHUB_TOKEN（未认证）；search 走代理出口共享配额，请求间隔 ≥7~8 秒。21 组查询（7 仓 × correctness/silent wrong output/NaN，created:>=2026-10-23）全部 HTTP 200（3 组带空格查询首轮管道解析失败后以引号编码重发 200；无 403）。窗口内合计：vllm 181/6/48、sglang 109/4、transformers 5、TRT-LLM 191/26、triton 6/0、tt-metal 100/0、flashinfer 343/61（correctness/swo/NaN）。
- 深读 10 条（core API 单点 6：#64758/#63620/#63776/#64087/#47521/#7847；search item body 补充 4）；既往判定沿用核对：sglang#46314=INF-2253（10-28 已收）、triton#12506=INF-2266、triton#12486 纯性能（10-29 判定沿用）、vllm#64005 沿用 10-27 留观（回访仍 open）、TRT-LLM#20991 fail-loud（10-29 判定沿用）。

### 新增条目（GitHub，9 条）

| 编号 | 来源 | 机制摘要 |
|---|---|---|
| INF-2314 | [vllm#64758](https://github.com/vllm-project/vllm/issues/64758) | Kimi-K3 FlashKDA 尾块 0×NaN 跨请求扩散成 token 0 洪流（AIME25 0.883→0.700）；首 NaN 疑 FlashMLA PDL 竞态；修复 FlashKDA#16/FlashMLA#41 open |
| INF-2315 | [vllm#63620](https://github.com/vllm-project/vllm/issues/63620) | 混合模型图回放 padding 行指向共享 null 页 block 0，XPU state kernel 写脏 13 视图致整批 NaN |
| INF-2318 | [vllm#63776](https://github.com/vllm-project/vllm/issues/63776) | DP dummy run 保留活请求 block ID，Mamba 状态写进他人 KV 块（RL 热更新 pause 期 2–56 步/次） |
| INF-2319 | [vllm#64087](https://github.com/vllm-project/vllm/pull/64087) | reshape_and_cache 超 512 chunk 丢写 KV head，ROCM_ATTN 读旧值（32×256 bf16 误差 0.19） |
| INF-2303 | [sglang#47521](https://github.com/sgl-project/sglang/pull/47521) | retract/rebootstrap 只带 token ids，多模态/embeds 请求错误 KV 静默续算；已流出 token 重启拼接错乱 |
| INF-2304 | [flashinfer#7847](https://github.com/flashinfer-ai/flashinfer/pull/7847) | MXFP4 E8M0 scale 253/254 先乘 scale 物化 inf 权重，零激活也应有限；安全顺序重算修复（Draft open） |
| INF-2301 | vllm-ascend#20367 | （见 NPU 节）PD stash 覆盖 |
| INF-2302 | vllm-ascend#20391 | （见 NPU 节）Mooncake 注册吞错 |
| INF-2317 | [vllm-ascend#20362](https://github.com/vllm-project/vllm-ascend/issues/20362) | MRv2 EPLB default policy 门控恒 armed 致重排 no-op（400 prompts 0 cycle vs stair 3） |

### GitHub 拒收/留观

- **TRT-LLM#21312**（overlap scheduler 变宽 beam search 读未写 KV 位置，修复=加拒绝）— 修复 PR 本身是 fail-loud 守卫化，原缺陷证据只在 PR 描述无用户实测，留观；**#21279**（NCCL vocab-parallel）、**#20958**（FP8 KV cross-attention）、**#21324**（spec replay 私有 KV buffer）为修复流，正文未深读，留观下轮补查。
- **vllm#64252/#64806**（XPU attn_sink/SM100 indexer 修复流）、**#64011**（量化前缀正则重命名映射）、**#64476**（DSpark Markov 采样修复集）— 修复流未深读；其中 #64476 涉采样分布族，下轮补查评论与用例。
- **flashinfer#7574**（MLA V 行 zero-fill）、**#7191**（fa2 softmax FMA 收缩）— 修复 PR open，正文未深读，留观下轮；**#7687** 沿用 10-29 留观（CI 聚合帖）。
- **sglang#46319**=INF-2253 的修复 PR，回访 open；**#46584/#46582**（NPU Mooncake L3/PP spec replay）NPU 功能域，无独立静默数值证据，留观。
- **tt-metal** 窗口内 correctness 命中均为 perf/feature/cleanup 流（#63019/#63473/#63463 为性能与回退），无新静默数值候选。
- **transformers#50112**（fairseq 转换 ast.literal_eval）closed 修复流，转换脚本域；无候选。

### arXiv

- 2 组查询（"silent data corruption"+LLM、"bit flip"+inference，sortBy=submittedDate）HTTP 200 有效（首轮裸 URL 302 空响应后 -L 跟随成功）。无新论文：2610.18557/BARE-AI/2609.19743/2610.30798=INF-2247 均已在库或已判；2610.33475（云架构讨论）、2610.37033（SRAM Ising 机）范围外；2610.27621（辐射加固容错设计）与 2609.06781（BLINK BN 完整性检查点，训练侧检测方法）为容错/检测设计类非本库故障编目对象（与 10-24 Radiation-Hardened 同判）。

## 3. 回访（近 7 日条目 + 轮巡旧源）

- **INF-2260**（GeluBackwardV2 rank26）：同型新证 ops-nn#7741（SituGlu，INF-2293）建 var 关联并注记；#7640 仍 open。
- **INF-2259**（MoeGatingTopKSoftmax 310P）：#6910 批次快照仍 open；修复 !14811 窗口内无更新。
- **INF-2254**（AttentionUpdate LSE）：#6872 仍 open，无修复进展。
- **INF-2275**（DynamicBlockMxQuant）：#7717 批次快照已 closed（10-29）；closed 不自动证明修复发布，关闭原因未核实，注记。
- **INF-2240**（GLM-5.3 视觉）：vllm-ascend#20115 仍 open。
- **INF-2245**（Gumbel NaN→token 1023）：vllm#63811 保持 closed（10-28），关闭原因与修复 commit 仍未核实（core API 快照）。
- **INF-2262**（KV 块拷贝 data_ptr 去重）：vllm-ascend#20157 仍 open。
- **INF-2160**（Addcmul NaN-as-1）：修复 !6791 仍 open，无合入进展。
- 轮巡（更早入库）：vllm#64005（MRV2 Qwen3-32B token 0）沿用 10-27 留观，core API 快照仍 open；vllm-ascend#20146（INF-2250 源）仍 open；msmodelslim/mindformers 系窗口内无遗留需改判。

## 4. 编目与视图

- 新写入 **31 条**（全部 pending_audit）：INF-2290 ~ INF-2320，编号连续。映射（以 catalog YAML 为准）：

| 编号 | 来源 | 标题 |
|---|---|---|
| INF-2290 | ops-math#4272 | LinalgCross fp64 FMA 残差非零/溢出 inf |
| INF-2291 | ops-math#4325 | Gcd INT16 混合取值错值 |
| INF-2292 | ops-math#4306 | Signbit(-0.0) 返回 False |
| INF-2293 | ops-nn#7741 | SituGlu rank≥26 输出未写（var→INF-2260） |
| INF-2294 | ops-nn#7739 | Shrink 标量 dtype 静默截断 |
| INF-2295 | ops-math#4298 | ReduceSum 负轴误选 fp32 丢末位 |
| INF-2296 | ops-math#4301 | Remainder 负除数乘积溢出 |
| INF-2297 | ops-math#4303 | RightShift 越界钳位输出 1 |
| INF-2298 | ops-nn#7747 | SCEWithLogits 越界写 |
| INF-2299 | ops-transformer!14989 | PromptFA fp16 分子溢出 inf |
| INF-2300 | ops-transformer#6944 | MoeGatingTopK 1ULP 翻转路由 |
| INF-2301 | vllm-ascend#20367 | PD stash 覆盖 GQA draft KV |
| INF-2302 | vllm-ascend#20391 | Mooncake 注册吞错延迟失败 |
| INF-2303 | sglang#47521 | retract rebootstrap 错误 KV |
| INF-2304 | flashinfer#7847 | MXFP4 极值 scale inf 权重 |
| INF-2305 | ops-nn#7756/#7757 | SquaredRelu out 契约错写/越界 |
| INF-2306 | ops-nn#7772 | SwiGluQuant groupIndex 不校验 |
| INF-2307 | ops-math#4344 | AmpUpdateScale fp16 增长 inf |
| INF-2308 | ops-nn#7762 | BN3D FP16 归约丢补偿量 |
| INF-2309 | ops-nn#7791 | FakeQuantPerChannel 语义异于 TF |
| INF-2310 | ops-math#4314 | IsClose 大整数 fp32 判等失效 |
| INF-2311 | ops-nn#7770 | SwiGluQuant [G] smooth 越界读 |
| INF-2312 | ops-transformer#6939 | FA TND 空 KV 脏输出 |
| INF-2313 | ops-transformer#6953 | attention_update N 截断 uint32 |
| INF-2314 | vllm#64758 | Kimi-K3 FlashKDA NaN 扩散 |
| INF-2315 | vllm#63620 | 图回放 padding 污染 null 页 |
| INF-2316 | ops-transformer#6962 | compressor autograd 梯度全错 |
| INF-2317 | vllm-ascend#20362 | MRv2 EPLB default no-op |
| INF-2318 | vllm#63776 | DP dummy run 写活请求状态 |
| INF-2319 | vllm#64087 | reshape_and_cache 丢写 KV head |
| INF-2320 | ops-nn#7736 | ApplyAdamWQuant 零 block NaN |

- 更新 7 条：INF-2293（var→INF-2260 + coverage=var）、INF-2260（同型注记）、INF-2259/INF-2254/INF-2240/INF-2275/INF-2245/INF-2262/INF-2160（回访注记）。
- var 关系回显核对：INF-2293→INF-2260（「GeluBackwardV2 rank26-30 返回成功但输出未写」✓）。
- `catalog_tool.py stats`：2320 条（active 2289 + pending_audit 31）。`build` 完成（CATALOG.zh.md、fault_catalog.csv、xlsx）；`validate` ok:true（n_errors 0，warnings 54 为既有存量+新增半受控 stack 告警）。
- 训练库跨库查重：36 个候选 URL 在 training_error_review 0 命中，无 cross 关系写入。

## 5. 未覆盖范围与 API 账本

- GitHub：21 组 search 全 200，无限流等待（累计等待 0，上限 5 分钟未触及）；core API 单点 9 次（64758/63620/63776/64087/47521/7847/63811/64005/20146）。新候选中 TRT-LLM#21279/#20958/#21324、flashinfer#7574/#7191、vllm#64476 未深读正文（修复流），判定按标题+既往族经验留观，下轮补查。
- NPU：GitCode/AtomGit issue 评论线程未拉取（无低成本批量入口），全部按正文判定——若评论中出现维护者定位/驳回，判定可能需改判；torchair#916（盘古入图精度 FAIL）证据仅报错片段留观下轮补查；pytorch#5744（_lazy_call 竞态）机制清晰但无自然触发证据留观。
- 强信号 80+159 条中约 28 条深读正文，其余按标题+关键词+既往大族判定留观（校验缺失/错误码契约族为主）。
- arXiv 仅 2 组主题词（SDC/bit-flip），KV corruption/数值正确性主题词未单独跑（与既往结果集重叠度高，缺口小）。

## 6. 结论

31 条新机制入库（待审计），9 条既有条目回访/补证/关联。本轮覆盖：NPU 22 源完整批次窗口（835 条窗口内新建、80+159 强信号、约 28 深读）+ GitHub 7 仓 3 关键词（~1300 条窗口内命中、10 深读）+ arXiv 2 组；缺口集中在 GitHub 评论线程、部分修复流正文与 torchair#916 补查。
