# 每日扫描日报 2026-10-22（推理库）

- **扫描窗口**：GitHub `created:>=2026-10-15`（cutoff=2026-10-15）；重点判定 10-21 报告之后的新增。NPU 共享批次 `batch-20261022T230043146418-ydw8tzkg`（22/22 仓 complete、0 failures、registry_hash 2f6422d0…，采集窗口 2026-10-21T22:00:43Z → 2026-10-22T23:00:43Z updated_at 增量，candidates=15295，normalized 15498，inference 视图 19 源）。
- **目标仓库**：原 GitHub 7 仓 + arXiv + NPU 19 推理源。
- `GITHUB_TOKEN` 未配置，按未认证限额执行（Search ≥9s 间隔）。GitHub API 经 `curl --noproxy '*'` 直连全部有效。arXiv 限流较重（多组首次超时），分段等待重试后 6/6 组跑成。累计限流等待约 4.5 分钟（arXiv 重试为主），未超 5 分钟上限。
- Git 前置检查：main 干净、fetch 成功、HEAD == origin/main（6b70f65d）。

## 一、原 GitHub 7 仓新增条目（16 条，pending_audit）

1. **INF-2118** [NEW] [vllm#63168](https://github.com/vllm-project/vllm/issues/63168)（c10-22）— GLM-5.3-Flash `_kpool_compress_insert` 假定 pool 对齐 chunk 起点：chunk 起点 mod 4 ≠ 0 时跨界 pool 被行号掩码丢弃（首行→槽位永不写出；前面有其他请求→池入他请求 token 的 key/gate）。CPU 逐字节复现 + 下游服务 observe-only 实测 217 个 index-K 写异常（186 未写/31 他人 token），DFlash 草稿槽预留把长 prompt 切成 2045-token chunk 恒触发。documented。
2. **INF-2121** [NEW] [vllm#63170](https://github.com/vllm-project/vllm/issues/63170)（c10-22）— GLM-5.x DSA 在 PP>1 时只向下一 stage 传 hidden/residual，indexer 层的 top-k 索引留在本 GPU；stage 首个复用层读自身 buffer 旧值（上一 batch 或未初始化内存），attend 错误 KV 行；输出仍连贯故静默，PP8 默认切分 13/78 层受影响。documented。
3. **INF-2130** [NEW] [sglang#45864](https://github.com/sgl-project/sglang/issues/45864)（c10-22）— PD decode 回退恢复后 QSA indexer 状态来自其他请求：QSATokenToKVPool 未覆写 cpu 备份（沿用只存 KV+Mamba 的版本），压缩索引键与 pending 环从不保存；live server 23/23 恢复错（max diff 3.75），修复后 0/33。documented。
4. **INF-2135** [NEW] [sglang#45866](https://github.com/sgl-project/sglang/issues/45866)（c10-22）— FP4 KV（fp4_mx_block16）PD 传输与 cpu_tensor 回退都不带块缩放因子：decode 用自己的（零/他人）scale 反量化，8 prompt 全部输出垃圾（0/8 与非 PD 一致；bf16 8/8、加修复 8/8）。documented。
5. **INF-2141** [NEW] [sglang#45755](https://github.com/sgl-project/sglang/issues/45755)（c10-22）— SM90 flashinfer_mxfp4 后端把 Kimi-K3 的 SiTU 常量按 GPT-OSS SwiGLU 语义（g·σ(4g)·(u+1) 钳 25）解释，硬编码 ActivationType.Swiglu 不读 activation；+1 项与 σ(4g) 数学上排除任何参数下等价，全程静默。documented。
6. **INF-2122** [NEW] [tt-metal#62016](https://github.com/tenstorrent/tt-metal/issues/62016)（c10-22）— ttnn.where 整型标量经 float32 舍入：int32/uint32 + |s|>2^24 返回 float32(s)（16777217→16777216），INT32_MAX/UINT32_MAX 合法值反被 TT_FATAL 拒绝；ttnn.add 同标量精确。documented。
7. **INF-2123** [NEW] [tt-metal#62019](https://github.com/tenstorrent/tt-metal/issues/62019)（c10-22）— ttnn.logit 预分配输出 dtype 与输入不同时 tmp0 按输出格式 pack、按输入格式 unpack：bf16→fp32 半数元素 -inf、99% 元素错位（奇位=错位的正确值）；fp32→bf16 近半非有限；同 dtype 与 60+ 其他算子正确。documented。
8. **INF-2131** [NEW] [tt-metal#62133](https://github.com/tenstorrent/tt-metal/issues/62133)（c10-22）— #56980 整数输入门控故意不含 INT8：一元算子对 int8 TILE 静默走 float SFPU kernel，abs 全 0、sign 正数也 -1、relu 不剪负；兄弟 dtype 正确 TT_FATAL。documented。
9. **INF-2124** [NEW] [tt-metal#62134](https://github.com/tenstorrent/tt-metal/issues/62134)（c10-22）— binary_ng 混合 fp32+bf16/bfloat8_b/bfloat4_b 操作数静默降精度：add 55% 元素错（145,060/262,144），比较也 13% 错；双 fp32 正确。documented。
10. **INF-2125** [NEW] [flashinfer#6994](https://github.com/flashinfer-ai/flashinfer/issues/6994)（c10-22）— 5 处块缩放量化站点守卫舍入前 vecMax 却消费舍入后 SFValue：块最大值 ≤2^-10 时 SF 舍 0、守卫通过，0×inf=NaN；全局 scale 顶格放置时即块最大值低于张量最大值 ~460000× 的块。#2441 已修同型一处，其余 5 站点未对齐。documented。
11. **INF-2132** [NEW] [flashinfer#7021](https://github.com/flashinfer-ai/flashinfer/issues/7021)（c10-22）— b12x fused_moe w4a8_mx deterministic_output=True 两缺陷：dynamic_tile_m=32 输出 cosine≈0；grouped 掩码路由（topk_ids=-1）不写 per-route 行、合并累加陈旧 scratch（EP 各 rank 恒命中）；规避 tile≥64+重映射。documented。
12. **INF-2133** [NEW] [vllm#63043](https://github.com/vllm-project/vllm/issues/63043)（c10-22）— native FlashInfer 在 FP8 K/V 下把 _q_scale_float 传给 kernel 即使 Q 是 BF16/FP16 未量化：非单位 q_scale 使注意力错误缩放，服务输出垃圾（Qwen3.8-27B FP8，0.29/0.32 复现、0.23 通过）；CPU 参数边界复现 passed 0.25→修复后 1.0。documented。
13. **INF-2134** [NEW] [transformers#49873](https://github.com/huggingface/transformers/issues/49873)（c10-22）— FineGrainedFP8 反量化以 rows//scale_rows 推块尺寸，DeepSeek 式 ceil 网格尾块：576 维 MLA 投影必 ValueError 拒载；维度整除时用错块尺寸（96 代 128）静默装错权重。documented。
14. **INF-2119** [NEW] [tt-metal#62124](https://github.com/tenstorrent/tt-metal/issues/62124)（c10-22）— Quasar topk SFPU 所有 SFPSWAP 传 imm12=0（int32 比较）：负浮点对排序序错、升降序互换；全非负正确（既有测试唯一覆盖）；阻塞 gpt-oss bringup。documented。
15. **INF-2129** [NEW] [tt-metal#62024](https://github.com/tenstorrent/tt-metal/issues/62024)（c10-22）— Wormhole remainder/fmod 边界：精确倍数符号相反返回除数（91%-13=-13）、inf 除数返回 NaN、fmod(inf,inf) 返 0；#61542 Blackhole 侧同族。documented。
16. **INF-2140** [NEW] [vllm#63128](https://github.com/vllm-project/vllm/issues/63128)（c10-22）— MTP draft-only 检查点 + PP>1 时 drafter 嵌入未初始化（共享仅 world_size==1；UnquantizedEmbeddingMethod 使未加载参数通过检查）：无错误、输出正确，但接受率 0.779→0.22（与零嵌入消融吻合），补嵌入后 0.55-0.85、吞吐 +49-53%。documented。

### 拒绝 / 留观（原 7 仓，主要项）

- **vLLM**：#63052（Breakable PIECEWISE 丢弃捕获上 warmup 激活缓存——已读正文，graph 捕获语义域，任务影响未证实留观）；#63037（KV Offload OffloadKey 复用丢失 BlockStored 元数据——事件流元数据域，延迟事件序已读，对实际 KV 正确性的影响链条未闭合留观）；#63000（长上下文偶发 mid-sentence `<|im_end|>` 提前停止——finish_reason=stop 且确实生成 im_end token，更像模型行为/采样域，频率 1/19~2/13，留观不占号）；#63026（Ascend SHM mailbox CPU 内存增长——资源泄漏域）；#63011/#63057 等 fail-loud/feature 不占号。
- **SGLang**：#45812（优先级抢占 victim 集合序跨 TP rank 不一致——set 迭代序按对象地址，可致 rank 间重排/abort 不同请求；调度一致性候选强，已读正文，机制清晰但静默错误面（vs 死锁/abort fail-loud 面）待权衡，留观下轮优先）；#45846（CI 数值分叉 decode vs prefill top-logprob 0.53 delta——确定性回归在 42 commits 内未二分，怀疑 #45631 logits 写路径，留观等定位）；#45781（GLM-5.3 FP8 挂起两天后 KV 满挂起域留观）；#45697（kimi_k3 tool-call 流式缓冲——输出解析域，正文未深读留观）。
- **tt-metal**：#62027（moreh layer_norm_backward 黑洞 odd 索引 mean/rstd 错——训练反传域为主，N150 正确，推理库视角降级）；#62021（moe_fused_swiglu 11x10 网格 ragged 尾块错值/挂起——默认 core_grid 未设时复现，候选强但 experimental deepseek prefill 域，留观下轮）；#62153（gather RM 分片未对齐宽度错值/挂死——数据搬运算子域，留观）；#62103（requires_grid_size 小网格静默 skip——测试基建域不占号）；#62068（ttnn.full uint32>INT32_MAX 填充错值——与 INF-2122 同"整型经 float"家族但独立算子路径，正文简短，留观下轮并条评估）；#62074（logical_not(-0.0) 返 0——body 仅 1 行，SFPU 边界族与 10-21 #61901 族相邻，留观）。
- **flashinfer**：#6995（groupwise fp8 GEMM CI 断言 1.6% mismatch——CI 单次失败、自动提单未独立验证，留观）；#7003/#6989/#6979（perf 域不占号）；#6977（b12x head_dim64 plan 与 kernel 矛盾——fail-loud 域，正文未读留观）。
- **transformers/triton**：窗口内无其他新判定候选（q05/q06 命中均弱）。
- **triton 主仓回访**：#12394（batched 3-D tl.dot N==1 累加器重叠）已 closed（10-21T04:04）但 closed 事件无 commit 关联、0 评论——closed 不证明修复，修复内容未核实，维持留观（10-21 判定沿用）。

## 二、NPU 来源扫描（batch-20261022T230043146418-ydw8tzkg）

- 22/22 仓 complete、failures=[]、registry_hash 2f6422d0…（22 源全选）；本轮增量判定 = updated_at > 2026-10-21T23:00:43Z 的 **3,360 项**（issue+PR），其中新建 882 项；标题关键词分桶（99 强信号新建全列 + 631 maybe 抽查）+ 30+ 篇正文深读（本地 bodies，零 API 成本）。CANN 自动化整改浪潮延续（文档/codecheck/bare-except 族占新建大头）。

### 新增条目（8 条，pending_audit）

1. **INF-2138** [NEW] [ops-nn#7225](https://gitcode.com/cann/ops-nn/issues/7225) + 修复 [!12598](https://gitcode.com/cann/ops-nn/merge_requests/12598)（open）— aclnnDequantSwigluQuantV2 不传 group_index 时 swiglu_mode 1/2/3（GPT-OSS 变体）静默按 mode0 计算：mode2 与 mode0 输出 100% 相同、golden 一致率 43.65%；无 group 链路完全无变体参数管道（硬编码标准 silu×up×smooth）。A2 910B3/CANN 9.3.0 源码级确认。documented。
2. **INF-2136** [NEW] [ops-nn#7223](https://gitcode.com/cann/ops-nn/issues/7223) + 修复 [!12603](https://gitcode.com/cann/ops-nn/merge_requests/12603)（open）— clipped_swiglu/clipped_swiglu_grad 传 group_index 时 sum(gi) 之外输出行为未初始化显存（kernel 只写 realBatchSize_ 行 + at::empty 分配）：同一输入有时全 0 有时残影，预热污染显存池后稳定复现；仓库自身 golden 表明正确语义是 0。documented。
3. **INF-2126** [NEW] [ops-nn#7229](https://gitcode.com/cann/ops-nn/issues/7229) — aclnnAdaptiveAvgPool2d 空 self + 非空 out 时早退返回 SUCCESS 一个元素不写：首次分配页常为 0 掩盖，池复用后读出脏值（污染实验证实）；同族 avg_pool2d/max_pool2d 正确报错。documented。
4. **INF-2120** [NEW] [op-plugin#625](https://gitcode.com/Ascend/op-plugin/issues/625) — torch.searchsorted 忽略 side='right' 恒返 lower_bound：s=[1,2,2,2,3] 查 2.0 应 [4] 实返 [1]，依赖上界的分桶/直方图索引系统性左移；bucketize(right=True) 正常，缺陷限定 searchsorted；910B3+A3 双平台复现。documented。
5. **INF-2137** [NEW] [ops-transformer#6524](https://gitcode.com/cann/ops-transformer/issues/6524) — aclnnMatmulReduceScatterV2 对 reduceOp/commTurn 无白名单校验：传 max/1 两段返回 0 且输出与 sum/0 逐元素一致（0/2048 不同），属性静默丢弃语义反转；V1 同输入正确 561002。documented。
6. **INF-2127** [NEW] [ops-math#3818](https://gitcode.com/cann/ops-math/issues/3818) + 修复 [!6604](https://gitcode.com/cann/ops-math/merge_requests/6604)（open）— Triu（已发布路径）tiling 状态进程级全局：matrix_Num 只初始化一次原位累乘，第二次调用起膨胀写进 tiling->matrixNum 使 kernel 越界读写；ub_Sharing_Num 同族不复位；var 关系指向 INF-2107（experimental tril/triu 同族，已回执核对标题）。documented。
7. **INF-2139** [NEW] [Ascend/pytorch#5613](https://gitcode.com/Ascend/pytorch/issues/5613) — npu_format_cast_(self, src) 两参重载 aclnn 分支误把 self 作转换源：src 数据静默丢弃、self 保持旧内容且 storage 指针被换（共享视图失效）；legacy 路径正确；Ascend950 恒触发、910B CANN≥9.1.0 可达；修复在报告者 fork 未上游。documented。
8. **INF-2128** [NEW] [Ascend/pytorch#5602](https://gitcode.com/Ascend/pytorch/issues/5602) + 修复 [!48370](https://gitcode.com/Ascend/pytorch/merge_requests/48370)（open）— TorchBench --accuracy 在 NPU 系统性误报 fail_accuracy：无 float64 内核致 fp64 golden 静默降级 float32（仅 W 告警）→ 伪 golden → fp64 RMSE 宽松回退被 dtype 门槛跳过 → 裸 allclose 误判合法浮点差异；同链 H100 全绿。kind=故障（测试判定域）。documented。

### NPU 留观与拒收（主要项）

- **留观（下轮优先）**：op-plugin#628/#626（整数 div/remainder 零除静默返 INT64_MAX 垃圾值/负指数幂静默 0——与 #625 同日报系列但 fail-silent 家族面广，正文已读，本轮合并量控制留观）；op-plugin#629（7 算子把 -0.0 规范化为 +0.0 含 signbit 错——IEEE 符号语义域）；op-plugin#630（normal 负 std/one_hot 越界静默——校验缺失族）；op-plugin#624（F.interpolate size=0 静默空张量——fail-silent 但属防御校验域，修复 !6084 已开）；ops-nn#7177/#7178（MaxUnpool2d/3dBackward indices 越界静默读显存外、同输入不可复现——越界读域候选强，正文已读）；ops-transformer#6533（MoeDistributeDispatch commAlg 非法值静默按默认执行——校验域）；ops-nn#7208（Ascend960 conv smallkernel 精度问题——正文仅一句话无环境无步骤，无法判定留观）；ops-math#3810+!6585（DropOutV3 int32 seed 精度——issue 正文空，修复 PR 已 merged，机制细节在 PR 描述之外留观）；ops-nn#7210（IndexPutWithSort syncSpaceGm 跨核 WAW racecheck 5640 条——竞态域候选，正文已读）；Ascend/pytorch#5601/#5609（triton_experimental BN 分解 2ulp 放大超门/select-lane 重写误改写——前者 fp64 对照已证良性数值重排非正确性错误，后者编译 fail-loud，均不占号但记录）；vllm-ascend#19498（Kimi-K3 dspark 采信率随上下文坍塌 33%→9%——现象实证强但正文短、根因未定位留观）；#19502（DeepSeek-V4.1 PD finegrained_tp+MultiConnector 确定性垃圾——配置组合窄、正文未读完留观）；#19485（Engram HOST_UVA 注册失败 rc=207001——启动失败域留观）。
- **拒收/不占号**：ops-transformer#6542/#6543/#6544（FinalizeRouting groupList 取负 AI Core 越界 507015——非法输入防御域，fail-loud 设备错误码）；#6548（BlockSparseAttention softmaxLseFlag 判空缺失 SIGSEGV——fail-loud）；#6534/#6535/#6536/#6537（SparseFlashMlaGrad 系列——反传/memcheck 域，训练库视角为主）；#6529（FA 空 KV 申请 100MB workspace——资源浪费非正确性）；#6531/#6532（MoeDistributeDispatch ep 声明与实际不一致挂死/未写——配置错误域）；#6510-#6517/#6525/#6541/#6546/#6547（错误码契约族：实测 561002 vs 资料 161002 等——资料一致性域批量，部分已有修复 PR）；ops-nn#7166/#7167/#7168/#7170/#7173/#7197（训练反传算子域：NLLLoss2d 越界返回 SUCCESS 全 0 属 fail-silent 候选但训练库视角，转训练扫描判定）；runtime#1255/#1259（log 系统竞态/挂起——观测基建域）；hccl#849/#850/#855/#856 与 hcomm#1102-#1105（需求建议/日志域）；msprobe#651（构建参数告警域）；ge#5543/#5552-#5555（ONNX 序列化/Pass 域，非推理正确性）；mindformers!8882（Muon 优化器 NaN 防护——训练域）；ops-math#6620（acosh 加固 PR：OI-18 域外 NaN 契约修复等——PR 型质量加固，其 issue#4 在外部个人仓，留观下轮按外部 issue 评估）；torchair#882-#891（静态内核重装/FX dump/cache key 族——工具链域）；vllm-ascend PR 族（#19494/#19521/#19530 等 BugFix PR——修复单本身按其 issue 判定，本轮无对应新 issue 命中强信号）；Ascend/pytorch#5616/#5619/#5621（Flight Recorder 观测工具域）；#5617（环境检查注释绕过——防御域）；#5611/#5622（构建/文档域）。
- **沿用判定**：ops-nn#7146/#7093/#7100/#7111、op-plugin#618/#620、runtime#1188、hcomm#1078、vllm-ascend#19391 等上轮留观项经本批次快照核对仍 open 无实质进展（ops-nn#7087 已 closed 且修复 !12472 佐证——10-21 已按"下轮按修复落地改判收录"记录，closed 不自动证明已发布，本日未见发布信息维持）；#19317-#19400 族无变化。

## 三、回访已收录来源（update，共 8 条 note + 1 条 note 附发现）

- **INF-1789**（FA out_dtype MR !14243）：仍 open（u10-22T10:50）→ note。
- **INF-1787**（EPLB PR #19288）：仍 open（u10-21T10:37）→ note。
- **INF-1793**（BN3D 溢出 #6955）：仍 open（u10-21T09:16）→ note。
- **INF-1375**（triton-ascend#2531）、**INF-1379**（vllm-ascend#18754）、**INF-1428**（ops-transformer#6119）：仍 open 无进展 → note（经共享批次快照零 API 成本）。
- **INF-2099**（ops-math#3709 tril）：仍 open；note 同时记录本日 #3818 Triu 同族发现（已 var 关联至 INF-2107）。
- **INF-2127**：新建时附 relations 回执核对 note。

## 四、arXiv

ax1/ax2/ax3/ax4/ax5/ax6 六组全部有效跑成（ax3「KV cache corruption」与 ax6「wrong results+inference+GPU」为真实 0 命中；ax1 唯一命中为 2026-10-13 Cauchy-Schwarz SDC 论文，10-21 已评估沿用；ax2 命中 2604.17249 Shared KV bit-flip（已编目 K4.9）与 CacheTrap（2025-11 旧文已覆盖）；ax4/ax5 无窗口内新故障域论文）。arXiv 限流较重：多组首次请求超时，按 30-55 秒分段等待重试后全部成功，累计等待约 4.5 分钟（在 5 分钟上限内）。

## 五、API 账本与未覆盖范围

| 组 | 来源 | 状态 |
|---|---|---|
| q01-q16 | GitHub Search（7 仓 × 2 类查询 + vllm/sglang PR 组，created:>=2026-10-15，per_page=100，≥9s 间隔） | 16 组全部 200 有效（q01 28、q02 128[2 页取全]、q03 39、q04 63、q05 1、q06 7、q07 6、q08 33、q09 9、q10 19、q11 74、q12 147[2 页取全]、q13 27、q14 38、q15 278[2 页 200]、q16 234[2 页 200]），无 incomplete |
| q02p2/q12p2/q15p2/q16p2 | 补页 | 全部 200 有效 |
| 17-43 | GitHub core REST 单号正文 27 次（vllm#63168/#63170/#63043/#63128/#63037/#63052/#63000、sg#45864/#45866/#45755/#45812/#45846/#45781、tf#49873、fi#6994/#6995/#7021、tt#62016/#62019/#62021/#62024/#62027/#62068/#62074/#62103/#62124/#62133/#62134/#62153） | 全部 200 有效 |
| 44-49 | arXiv 6 组 | 全部有效（分段等待后成功） |
| 50 | NPU 共享批次 | 22/22 complete、0 failures、candidates=15295 |

**未覆盖/限制**：(1) **q15/q16（vllm/sglang PR 搜索）总数 278/234 > 200**，第 2 页取满后余 78/34 条未取——PR 域覆盖不全，滚动下轮；(2) 仅按索引/标题判定的候选：sglang#45697（kimi_k3 tool-call 流式，正文未读）、fi#6977（b12x head_dim64，正文未读）、tt#62015（Gemma4 QB2 prefill 挂起，标题级）、tt#62149（测试内核域）等；(3) GitHub 评论级深查未做（无 GITHUB_TOKEN）：vllm#63170（4 评论）、sg#45755（1 评论）等新条目未读评论——机制已闭合故收录，评论级补充缺口记录；triton#12394 closed 无 commit 关联，修复内容未核实；(4) NPU 增量 3,360 项中约 2,400 项未命中强信号关键词、按标题+类型分桶处理（文档/CI/需求/适配族占大头），标题与正文严重不符的候选可能漏判；(5) ops-nn#7166-#7173/#7210/#7221 等训练反传域候选转训练扫描视角，本库不判定；(6) 上轮留观强候选（ops-nn#7146/#7093/#7100、op-plugin#618/#619/#620、vllm#62851、fi#6860/#6861/#6866、tf#49810、vllm#62554）本轮无新证据维持留观，未重读全文。

## 六、编目变更汇总

- 新增 24 条（全部 pending_audit）：**INF-2118 ~ INF-2141**（无断号）
  - GitHub 主仓：INF-2118（vLLM kpool 未对齐 chunk）、INF-2121（vLLM GLM DSA PP 陈旧 top-k）、INF-2130（sglang QSA 回退状态混淆）、INF-2135（sglang FP4 KV scale 丢失）、INF-2141（sglang Kimi-K3 SiTU→SwiGLU）、INF-2122（tt where 整型标量）、INF-2123（tt logit 混合 dtype）、INF-2131（tt INT8 门控）、INF-2124（tt binary_ng 混合精度）、INF-2125（fi 块缩放 NaN）、INF-2132（fi b12x 确定性 MoE）、INF-2133（vLLM BF16 Q q_scale）、INF-2134（transformers FP8 尾块）、INF-2119（tt Quasar topk 负值序）、INF-2129（tt remainder/fmod 边界）、INF-2140（vLLM MTP 嵌入未初始化）
  - NPU：INF-2138（DequantSwigluQuant mode 丢失）、INF-2136（ClippedSwiglu 未初始化行）、INF-2126（AdaptiveAvgPool2d 空输入）、INF-2120（searchsorted side）、INF-2137（MRSv2 属性忽略）、INF-2127（Triu tiling 全局态，var→INF-2107）、INF-2139（npu_format_cast_ 误用 self）、INF-2128（torchbench fp64 golden 降级）
- update：INF-1789、INF-1787、INF-1793、INF-1375、INF-1379、INF-1428（回访）、INF-2099（回访+同族记录）、INF-2127（relations 回执核对），共 8 条 note。
- 跨库查重：24 个新条目主 URL 已在训练库逐一查重，0 命中，无 cross 关系新增；var 关系 1 条（INF-2127→INF-2107，回执标题已核对）。
- `catalog_tool.py build` + `validate`：通过（**ok: true，0 errors，0 warnings**；2141 条，pending_audit 24）。
- 编目总数 **2117 → 2141**（active 2117 + pending_audit 24）。
