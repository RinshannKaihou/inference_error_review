# 每日扫描日报 2026-10-21（推理库）

- **扫描窗口**：GitHub `created:>=2026-10-14`（cutoff=2026-10-14）；重点判定 10-20 报告之后的新增。NPU 共享批次 `batch-20261021T230043103472-yzy4ui3l`（22/22 仓 complete、0 failures、registry_hash 2f6422d0…，采集窗口 2026-10-20T22:00:43Z → 2026-10-21T23:00:43Z updated_at 增量，candidates=14217，normalized 14414，inference 视图 19 源）。
- **目标仓库**：原 GitHub 7 仓 + arXiv + NPU 19 推理源。
- `GITHUB_TOKEN` 未配置，按未认证限额执行（Search ≥9s 间隔）。GitHub API 经 `curl --noproxy '*'` 直连全部有效；本轮 16 组 Search + 2 页补页（q02/q10 第 2 页）+ 10 次单号 REST 正文全部 200、无截断（q02 126 命中取满 2 页、q10 166 命中取满 2 页）。累计限流等待 0 秒（无配额等待）。
- Git 前置检查：main 干净、fetch 成功、HEAD == origin/main（5ee424b4）。
- arXiv：ax1（SDC+LLM）5 组有效、ax2（inference correctness）有效、ax3（KV cache corruption）有效 0 命中（真实结果）、ax5（numerical correctness cs.LG）有效、ax4（bit flip+inference）首次 2 次超时后分段等待重试成功；**ax6（wrong results+inference+GPU）两次超时未跑成**，缺口记录。窗口内新论文 2 篇候选：Cauchy-Schwarz SDC Detection（2026-10-13，ax1）、Shared KV Caching for Replicated 27B Inference（2026-09-14，ax2，摘要已读：packed-page patch 后 raw-pointer fallback 漏 CUDA stream 依赖，受控字节测试在施加延迟下失败）——均在上轮窗口或方法论相邻，本轮不新建条目，Shared KV Caching 留观下轮深读。

## 一、原 7 仓新增条目（8 条，pending_audit）

1. **INF-2114** [NEW] [vllm#62811](https://github.com/vllm-project/vllm/issues/62811)（c10-21）— DSV4.1 chunked prefill 末块走 PIECEWISE CUDA graph 时贪心输出全错：8202-token prompt/mnbt=8192（末块 10）从首个生成 token 起稳定错（[2997,2550,...] vs [3817,2242,...]）；enforce_eager/FULL/NONE 精确、PIECEWISE 1/6 错；逐步 trace 唯一差异是末块派发（10 padded 16 vs eager/FULL）。H100/H200、0.32.0 与 main。documented。
2. **INF-2117** [NEW] [vllm#62945](https://github.com/vllm-project/vllm/issues/62945)（c10-21）— nvidia/GLM-5.3-NVFP4 EP+TP>1+PCP>1 双重 all-reduce 静默放大 tp_size 倍：模型层 fused_allreduce_rms_norm 与 runner 侧归约都执行；TP4×PCP2 GSM8K 0.00（word salad）、TP2×PCP2 0.60、TP1×PCP4 0.90；同 0.32.0 节点仅改一行 skip_final_all_reduce=True 即 0.85（同节点 A/B）。B200+SM120 双架构、0.31.0~main 逐字节相同。documented。
3. **INF-2113** [NEW] [vllm#62828](https://github.com/vllm-project/vllm/issues/62828)（c10-21）— ROCM_ATTN 滑窗 Triton kernel 只 mask score 不 mask V：窗外 KV 块被复用后含 NaN/Inf 时 0×NaN 污染整行；DFlash2 drafter 8k 并发 16 接受长度 3.2→1.2、64k 并发 1 直接 run failed；修复任务 #62415 open。MI355X 实测。documented。
4. **INF-2108** [NEW] [flashinfer#6907](https://github.com/flashinfer-ai/flashinfer/issues/6907)（c10-21）— FA2 ragged prefill 早期主导 key 使 running max 跨 tile 不变时重缩放分支出 NaN：合成最小复现全部 128 输出 NaN、LSE -inf，FP64 参考为精确 1；disable_split_kv=True 单 state。A100/JIT main 23f3e8f。documented。
5. **INF-2115** [NEW] [tt-metal#61540](https://github.com/tenstorrent/tt-metal/issues/61540)（c10-20）— ttnn.experimental.paged_fused_update_cache ROW_MAJOR 路径三种校验放行组合写坏 KV cache：FP32 输入默认 dest_acc 更新行全坏（含非有限值，最多 851,961/4,194,304 元素不同）；FP32 cache 同 tile 其余行被舍到 bf16/tf32；bh16 输入开 dest_acc 更新行错。BH p100a 与 WH n150 数字一致。documented。
6. **INF-2109** [NEW] [tt-metal#61919](https://github.com/tenstorrent/tt-metal/issues/61919)（c10-21）— fabric ERISC 路由器把精确字节数传给 eth_send_packet_bytes_unsafe()，TX 队列只搬整 16B 字：非 16B 倍包尾部字丢弃、接收端转发陈旧字节；fast dispatch 完成队列写转发远端，宿主读远端 chip 数据静默损坏。4×N300 test_prefetcher 4 测全 Data mismatch。documented。
7. **INF-2112** [NEW] [tt-metal#61942](https://github.com/tenstorrent/tt-metal/issues/61942)（c10-21）— Kimi K3 MoE 共享专家不重叠 dispatch 时（overlap=False）第二次前向起 shared_output 损坏（PCC 0.9996→0.48~0.87 逐次不同）：reduce_scatter 输出分配在已释放 gate_out 之上+跳过起始 barrier；纯 bf16 eager 两次前向即复现，默认 overlap prefill/K2.7/GLM-5.3 不受影响。8x4 BH Galaxy。documented。
8. **INF-2116** [NEW] [vllm-ascend#19357](https://github.com/vllm-project/vllm-ascend/issues/19357) + 修复 [PR #19401](https://github.com/vllm-project/vllm-ascend/pull/19401)（均 open）— npu_hc_pre_v3 Sinkhorn 列归一化 epsilon 加错元素位置（hcMult=4 按 packed 计数漏行尾有效位），comb 输出依赖行位置；A3 算子级隔离 12 组全复现；修复 PR open 未合。documented。

### 拒绝 / 留观（原 7 仓，主要项）

- **vLLM**：#62851（Qwen3.6 GDN DFlash2 批依赖数值+抢占重算路径分叉——报告方自证非 DSpark 接受违规、含独立 gate-indexing 缺陷线索，正文长报告已读大半，**候选强**，本轮合并量控制留观下轮优先）；#62820（FlashInfer fp4_gemm autotune 改变温度 0 token——autotune 非确定性选核域，正文已读，机制未闭合留观）；#62793（MRV2 自适应验证 draft-prefill 图 padding 错配降低接受率——推测解码接受率域，留观）；#62865（Compile mode 2 Transformers 后端错 kernel——编译域候选，未深读留观）；#62283/#62931/#61768/#61905（KV Connector 三连+tiered 域维持既有判定范围）；#62417 维持 10-19 已收录；#62606/#62697/#62626/#62664/#62225/#62122 已收录维持；#62554（Packed NCCL 权重同步复用接收缓冲——权重损坏候选，机制未读透留观下轮优先）；#61673 closed（Cascade attention CPUOffload 跨请求 KV——10-14 已判留观维持）。
- **SGLang**：#45525（DSV4-Flash 推测 vs 普通解码贪心分叉——同 target logits 路径分叉，操作算子未定位，留观）；#45481（Qwen3.5-35B TP1/PP2+NEXTN 并发下贪心分叉——deterministic 开启下仍分叉，PP/spec 域留观）；#44999/#45000/#44998/#44758 已收录维持；#45622（context length 从 config 静默截断——配置域留观）；#44997（NGRAM+overlap 污染 radix KV——10-18 已收）。
- **transformers**：#49810（Gemma 2 softcap 在 sdpa 默认与无 softcap flash kernel 被静默丢弃——**候选强**（含警告被 #40744 移除的历史线索），closed 但 closed 不证明修复，本轮合并量控制留观下轮优先）；#49745 维持 10-20 留观；#49700（prefill_chunked 丢图像输入——多模态输入域留观）。
- **TensorRT-LLM**：#20376（grouped MLA cubin 注册为 Dense 但按 Causal 命名选择——kernelMetaInfo 错配，正文未深读留观）；#20603（CSA2 动态 token-sparse MLA 间歇 illegal address——fail-loud 域）；#20365 closed 维持 10-16。
- **tt-metal**：#61901/#61903/#61904/#61907/#61894（log_sigmoid/min/max/silu NaN 语义边界族——SFPU 边界值族候选，与 INF-1783 相邻，本轮合并量控制留观）；#61791（transpose height-sharded 偏移网格错值——与 INF-1779 同域不同 op 留观）；#61938（Quasar SrcB MXFP4_2x 清除不撤销——状态残留候选留观）；#61942 已收录；#61540 已收录。
- **triton 主仓**：#12394 closed（batched 3-D tl.dot N==1 累加器重叠——**closed 但修复内容未核实**，下轮回访确认后决定）；#12427（builtin float→fp32 常量先丢 fp64 精度——语言语义域留观）；#12368（persistent matmul 接受未 padded ragged-K——校验域，10-19 已判留观维持）。
- **flashinfer**：#6860（CUTLASS MLA padding NaN 传播——与 10-16 #6741 相邻但独立缺陷，**候选强**，正文已读，本轮合并量控制留观下轮优先）；#6861（CuTe DSL monolithic MLA 同类 padding NaN——同上留观）；#6866（b12x tiny-decode 忽略 swiglu_limit——量化 MoE 激活钳制缺失，DSV4.1 解码域候选强，留观下轮优先）；#6907 已收录。

## 二、NPU 来源扫描（batch-20261021T230043103472-yzy4ui3l）

- 22/22 仓 complete、0 failures、registry_hash 2f6422d0…（22 源全选）；本轮增量判定 = updated_at > 2026-10-20T23:00:43Z 的 **3,139 项**（issue+PR），其中 10-20 报告后新建 1,077 项；标题关键词分桶 + 强信号全列表过题（文档/CI/README/门禁/alt 文本/标点整改族占新建大头，CANN 自动化整改浪潮特征明显）；重点候选正文深读 30+ 篇（本地 bodies，零 API 成本）。

### 新增条目（12 条，pending_audit）

1. **INF-2103** [NEW] [ops-transformer#6417](https://gitcode.com/cann/ops-transformer/issues/6417) — PromptFlashAttention tiling 全局 BENCHMARK_TILING_KEY 跨调用残留：同进程先走 SH+sparse_mode=20 的 baseApi 路径后再调普通 PFA，kernel 实收 tilingKey 被污染（…01012→…401012）直接 AICore 越界，910B2 3/3 复现，cold/ctrl 组正常。documented。
2. **INF-2098** [NEW] [ops-nn#7006](https://gitcode.com/cann/ops-nn/issues/7006) — AddRmsNormQuant TU 级全局 optionalUbNum_0/_1 无 else 复位：同进程先调带 beta 的调用后，同输入 tiling 模板 singleN/12224→splitD/6880 静默漂移（输出当前逐 bit 不变、耗时差<1.5%，tiling 决定性破坏实测）。documented。
3. **INF-2107** [NEW] [ops-math#3709](https://gitcode.com/cann/ops-math/issues/3709) + [#3710](https://gitcode.com/cann/ops-math/issues/3710)（tril/triu 镜像同型并条）— experimental tril/triu tiling 状态存 TU 级全局、matrix_Num 唯一写点累乘不复位：同进程第二次调用（两次入参完全相同）matrixNum 2→4、lastTileLength 32→48 实测；kernel 循环越界为代码推演；仅 --experimental 构建受影响。documented。
4. **INF-2104** [NEW] [ops-nn#7026](https://gitcode.com/cann/ops-nn/issues/7026) — ThresholdBackward aclnn 层全局 thresholdVal_ 多线程互覆：A 线程读到 B 的 threshold 走错 kernel 模板（ReluGrad vs ThresholdGradV2D），输出 0 掩码整片错位静默错算；4 线程×32 次第一段连调构造最大化窗口。documented。
5. **INF-2101** [NEW] [op-plugin#616](https://gitcode.com/Ascend/op-plugin/issues/616) + 修复 [!6074](https://gitcode.com/Ascend/op-plugin/merge_requests/6074)（open）— gelu.out/gelu_backward 默认走 V1 aclnnGelu 丢弃 approximate：tanh 被算成 erf 精确版静默精度错误；默认环境变量未设即触发，op-plugin 26.0.0+/master 含派发版本。documented。
6. **INF-2105** [NEW] [ops-nn#7121](https://gitcode.com/cann/ops-nn/issues/7121) — aclnnScaledMaskedSoftmax 用 -10000 常量屏蔽：行内有效位 scale*x 低于 -10000 时概率全落屏蔽位（实测 [-60000,-60001,-100,-100]+mask 后 4 位期望 [0.73,0.27,0,0] 实测 [0,0,0.5,0.5]），三 dtype 腿均复现无告警。documented。
7. **INF-2106** [NEW] [ops-nn#7122](https://gitcode.com/cann/ops-nn/issues/7122) — 同算子 fp16 腿半精域乘 scale：x=60000、scale=2 时 fp16 整行 NaN（fp32/bf16 先 Cast fp32 正确），三 dtype 腿行为不一致，接口返回 0。documented。
8. **INF-2102** [NEW] [ops-nn#7037](https://gitcode.com/cann/ops-nn/issues/7037) — aclnnDeepNormGrad cutD 尾块（≤1024 元素）dgamma 尾段静默错值：fp32 D=6144 尾段 40 元素 max 0.63、fp16/bf16 D=5120 尾段 683/681 元素 max 0.12；错误区间起点恒为单次切 D 容量；常见 hidden 维度命中。documented。
9. **INF-2099** [NEW] [ops-math#3747](https://gitcode.com/cann/ops-math/issues/3747) — aclnnTril（已发布路径）uint32/int64 返回 0 但约半输出未写（恒 ceil(n/2)，预填哨兵原样保留）且已写部分掩码错；int32/uint64/fp32 同条件全对，排除宽度搬运解释。documented。
10. **INF-2100** [NEW] [ops-math#3796](https://gitcode.com/cann/ops-math/issues/3796) — aclnnSinkhorn fp32 分支忽略非连续输出 p 的 strides 静默写错位置（fp16/bf16 分支先临时张量+ViewCopy 正确）；文档声明 p 不支持非连续但不拒绝；自洽两读法判别实锤。documented。
11. **INF-2110** [NEW] [ops-nn#7143](https://gitcode.com/cann/ops-nn/issues/7143) — Ascend 950 Median/NanMedian 多核 radix 路径把负 subnormal 的 radix key 误计为 NaN（key 0x7ffffffe 按 FP32 解读即 NaN）：输入不含 NaN/Inf 时中位数值与索引双错，修复 PR 附零容差 CSV 用例（合入状态未核实）。950DT/9.3.0/TTK 实测。documented。
12. **INF-2111** [NEW] [ops-transformer#6454](https://gitcode.com/cann/ops-transformer/issues/6454) — aclnnRotaryPositionEmbedding A2 上 mode=2/3（quarter/interleave-half，950 专属）不拒绝：输出与 mode=0 逐字节相同（探针保证模式间可区分），静默回落 half；同代码 A2 与 950 语义不同都返回成功。documented。

### NPU 留观与拒收（主要项）

- **留观（下轮优先）**：ops-nn#7146（aclnnBatchNorm 4D+ND+training SIGSEGV + saveInvstd 回填批方差而非逆标准差——**双缺陷候选强**，崩溃组合窄但语义错输出影响面待评估，正文已读大半）；ops-nn#7093（BatchNormElemtBackward ReduceSum(counter) 当除数，多元素 counter 梯度整体偏小 C 倍——**候选强**已读主要段）；ops-nn#7100（TurboQuant y/scale shape 不校验声明缩小仍返回 0 并越界写——canary 实证已读前段）；ops-nn#7111（DequantSituQuant round-half-even vs 资料 trunc 系统性 1 LSB——量化契约域）；ops-nn#7035（BatchNormBackward 把 saveInvstd 当方差——契约/实现口径，训练反传域可降级）；ops-nn#7030（Softmax 空 tensor 跳过校验）；ops-nn#7087 closed（Logit eps<=0 被 tiling 改写 1e-6，x=1 输出 13.8 而非 inf——**修复 MR !12472 佐证+closed**，bit-exact 证据，下轮按修复落地改判收录）；ops-nn#7118（softmax 族混合精度低精度中转截断，修复 !12477 在办）；ops-transformer#6453（GMM SWiGLU Quant A8W8 四参数约束不校验静默 SUCCESS）；ops-transformer#6421/#6422（block_sparse_attention mask 行数>16384 / kv 块数>65536 kernel 无报错挂死且污染设备状态——**挂死域候选强**，2^14/2^16 边界定位完整）；Ascend/op-plugin#618/#619/#620（torch_npu 2.7.1 语义不一致三连：div floor ±inf 错值+空张量 max 静默 0+gather/scatter/embedding 越界静默接受——**候选强**，本轮合并量控制留观下轮优先，#620 越界读/写内存安全域最重）；ops-math#3794/#3795/#3797（Sinkhorn 挂死/错误码契约族）；cann/runtime#1188+!5715（PyACL 字典标量静默窄化——已有修复 PR open）；hcomm#1078+!6165（GetStream 隐含绑定 dispatcherCtx——已有修复）；vllm-ascend#19391（长 prompt ≥4K 永不被调度 capacity 幻影——调度域候选已读前段）；#19317/#19318（GBSA 黑名单拼写错致不兼容静态 kernel 图捕获失败——fail-loud 但暴露 #18869 排除机制失效面，留观）；#19396/#19397/#19398/#19399（KV engine ID 多节点 PCP 同步族——握手拒绝 fail-loud 域，机制完整留观）；#19400（GDN recurrent state indices 边界请求 abort EngineCore——fail-loud）；Ascend/pytorch#5575/#5577/#5590/#5591（mix-order reduction/inductor matmul backward 族——训练/编译域多数转训练库视角留观）。
- **拒收/不占号**：ops-math#3702~#3727 与 #3728+（README 文档族批量，含"仅含 aclnn 接口"声明纠错——文档域）；runtime#1184/#1186/#1189~#1244（PyACL/文档/头文件注释批量族——文档域为主，其中 #1215 CMO 长度截短、#1209 LabelList 回滚、#1234 SQ lock 回滚为防御域修复已带 PR）；ge#5489~#5536（codecheck 消解+文档修复批量族——承接单）；ops-nn#6991~#6999/#7003/#7008 等（README dtype/支持表文档族）；ops-nn#7005/#7011/#7012/#7021（需求建议/功能恢复类）；ops-nn#6995/#6996（ScatterElements/IndexToAddr 越界防御已带修复 PR——防御域修复在办不占号）；ops-transformer#6400~#6406/#6413~#6415（Moe 族 Proto/README 约束文档纠错——文档域，其中 #6402 MoeInitRoutingQuantV2 动态量化公式修正是实缺陷但属文档公式域）；ops-nn 压测提单族 #7006~#7020 中 #7013~#7019（AddLayerNormQuantV2 aivec UB 越界压测族——**机制标题完整但为压测自动提单**，与 #7006 同算子同根源不同层，随 #7006 修复观察）；triton-ascend#2662~#2688（RFC/feature 族为主）；#2648 维持 10-20 留观（open 无进展）；msmodelslim#627（九天 236B npu 量化不可用——正文疑似环境域，留观）；mindspore#93308（MindIR 图属性缓存作用域修复——训练图域修复单）。
- **沿用判定**：ops-transformer#6344/#6353/#6359、ops-nn#6961、hcomm!6144、Ascend/pytorch#5558/#5567、triton-ascend#2531、vllm-ascend#18754、ops-math#3677 等近 7 日条目经本轮批次快照核对状态无实质变化（见回访）。

## 三、回访已收录来源（update，共 6 条 note）

- **INF-1789**（FA out_dtype 修复 MR !14243）：仍 open（u2026-10-21T08:49 有更新），未合入未发布 → note。
- **INF-1787**（EPLB record 修复 PR #19288）：仍 open（u2026-10-21T10:37 更新）→ note。
- **INF-1793**（BN3D 溢出 #6955）：仍 open（u2026-10-21T09:16 更新，标题并入图路径+golden 完善），发布未核实 → note。
- **INF-1375**（triton-ascend#2531）、**INF-1379**（vllm-ascend#18754）、**INF-1428**（ops-transformer#6119）：轮巡仍 open 无进展 → note（经共享批次快照零 API 成本）。
- 其余近 7 日 GitHub 收录条目经本轮 Search 索引核对状态：#62606/#62697/#62626 open 维持、#62651 closed（10-20 已记）、#61738 closed（u10-20，宽推测 EOS 域，closed 原因未读，下轮回访）、#62466/#61766 closed 维持已判。

## 四、arXiv

ax1/ax2/ax3/ax4/ax5 有效（ax4 分段等待后成功）；**ax6（"wrong results"+inference+GPU）两次超时未跑成**，记为未覆盖。窗口内新论文 2 篇：Cauchy-Schwarz SDC Detection in LLM Inference（10-13，方法论相邻）、Shared KV Caching for Replicated 27B Inference（09-14，含 raw-pointer fallback 漏 stream 依赖的受控复现）——前者在上轮窗口内应已被覆盖，后者留观下轮深读。

## 五、API 账本与未覆盖范围

| 组 | 来源 | 状态 |
|---|---|---|
| 1-16 | GitHub Search（7 仓 × 2 类查询 + sglang/vllm 补充组，created:>=2026-10-14，per_page=100，≥9s 间隔） | 16 组全部 200 有效（q01 56、q02 126[2 页取全]、q03 36、q04 61、q05 6、q06 24、q07 4、q08 33、q09 16、q10 166[2 页取全]、q11 38、q12 14、q13 45、q02p2 100、q10p2 66），无截断 |
| 17-26 | GitHub core REST 单号正文 10 次（vllm#62811/#62945/#62828/#62851、fi#6907/#6860/#6861/#6866、sg#45525/#45481、tf#49810、tt#61540/#61919/#61942） | 全部 200 有效 |
| 27-31 | arXiv 5 组有效（ax1/2/3/4/5）；ax4 分段等待后成功 | ax6 两次超时未跑成（缺口） |
| 32 | NPU 共享批次 | 22/22 complete、0 failures、candidates=14217 |

**未覆盖/限制**：(1) **仅按索引/标题判定的候选**：ops-nn#7146/#7093/#7100（正文读了前段未逐段读完，列为下轮优先）；triton#12394（closed 修复内容未核实）；vllm#62554、flashinfer#6860/#6861/#6866、transformers#49810、op-plugin#618/#619/#620（留观，正文已读但合并量控制）。(2) NPU 增量 3,139 项中约 2,100 项未命中强信号关键词、按标题分类（文档/CI/Proto/README/alt 文本/标点整改族占新建大头），未逐条读正文——标题与正文严重不符的候选可能漏判。(3) GitHub 评论级深查未做（无 GITHUB_TOKEN）：vllm#61738（closed 原因）、#62820、#62417 的评论未读。(4) ax6 一组未跑成（见四）。(5) triton-ascend#2648 维持留观未深读新评论。

## 六、编目变更汇总

- 新增 20 条（全部 pending_audit）：**INF-2098 ~ INF-2117**（无断号）
  - NPU：INF-2098（AddRmsNormQuant 残留）、INF-2099（Tril dtype 半写）、INF-2100（Sinkhorn strides）、INF-2101（gelu approximate）、INF-2102（DeepNormGrad 尾块）、INF-2103（PFA tilingKey）、INF-2104（ThresholdBackward 竞争）、INF-2105（SMS -10000）、INF-2106（SMS fp16 溢出）、INF-2107（tril/triu 累乘）、INF-2110（Median subnormal）、INF-2111（RoPE mode23）
  - GitHub 主仓：INF-2108（FA2 running max NaN）、INF-2109（tt fabric 尾字）、INF-2112（tt Kimi K3 共享专家）、INF-2113（ROCM_ATTN 滑窗 NaN）、INF-2114（vLLM PIECEWISE 末块）、INF-2115（tt paged update KV）、INF-2116（hc_pre epsilon 错位）、INF-2117（GLM NVFP4 双归约）
- update：INF-1789、INF-1787、INF-1793（回访）、INF-1375、INF-1379、INF-1428（轮巡），共 6 条 note。
- 跨库查重：20 个新条目主 URL 已在训练库查重（26 个 URL 含并条来源），0 命中，无 cross 关系新增；无 var 关系新增（各新条目与既有条目机制不同，本轮同类「TU 级全局状态残留」家族三例分立收录不强行族化，待审计判定）。
- `catalog_tool.py build` + `validate`：通过（**ok: true，0 errors**，44 warnings 为存量模式待定项；2117 条，pending_audit 20）。
- 编目总数 **2097 → 2117**（active 2097 + pending_audit 20）。
