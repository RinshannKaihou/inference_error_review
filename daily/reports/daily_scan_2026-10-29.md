# 每日扫描日报 2026-10-29（推理库）

## 0. Git 前置检查

- 分支 `main`，上游 `origin/main`；`git status --porcelain -uall` 为空；`git fetch origin main` 成功；HEAD 与 origin/main 一致（7de9d301934f0906518b0a3930a46795dc5f1798）。通过，开始扫描。

## 1. NPU 新增来源（共享批次）

- 批次：`/root/.hermes/profiles/third/cache/fault_scan_sources/batch-20261029T230043149098-ql3d66uf`，采集窗口 2026-10-28T22:00:43Z → 2026-10-29T23:00:43Z（registry_hash 2f6422d0…，22 源全 complete、failures 空）。
- inference scope 候选 20518 条；窗口内新建（created_at ≥ since）1010 条（ops-nn 210、ops-transformer 188、ops-math 128、vllm-ascend 131、pytorch 115、hcomm 47、triton-ascend 41、op-plugin 26、runtime 32、ge 30、hccl 23、msmodelslim 9、mindspore 9、sgl-kernel-npu 5、torchair 10、mindformers 3、mssanitizer 2、msprobe 1）；标题强信号命中 245 条。
- 深读正文约 30 条（本地 body 缓存）；全部收录候选完成近 14 天日报完整 URL 查重 + `catalog_tool.py find` 查重 + 训练库跨库查重（9 个 NPU 候选 0 命中，无 cross 关系）。

### 新增条目（NPU 来源，23 条）

| 编号 | 来源 | 机制摘要 |
|---|---|---|
| INF-2259 | [ops-transformer#6910](https://gitcode.com/cann/ops-transformer/issues/6910) | MoeGatingTopKSoftmax 310P tiling Align() 越过真实行数→尾块 UB 溢出→MoE 路由错乱，vLLM GSM8K 21% 空输出；修复 !14811 open |
| INF-2254 | [ops-transformer#6872](https://gitcode.com/cann/ops-transformer/issues/6872) | AttentionUpdate 大 LSE 公共偏移丢失 log-sum 项→权重不归一输出翻倍（1.5→3），950PR 实测 5/7 失败全 rc=0 |
| INF-2258 | [ops-transformer#6870](https://gitcode.com/cann/ops-transformer/issues/6870) | DenseLightningIndexerSoftmaxLseV2 S2>16 输出错误 lse（S2=17~31 恒 128+log16；S2=32 偏差 895） |
| INF-2255 | [ops-nn#7721](https://gitcode.com/cann/ops-nn/issues/7721) | DynamicQuant 系 arch35 向量路径 bf16 极值极差溢出（var→INF-2242；A2 已修 !13078，arch35 单独立项） |
| INF-2256 | [ops-nn#7642](https://gitcode.com/cann/ops-nn/issues/7642) | FastGelu fp16 x<-6.5158 exp 中间量溢出返回 -0（理论/实测界差 0.54 ULP，45/45 点一致，bf16 正确） |
| INF-2257 | [ops-math#4216](https://gitcode.com/cann/ops-math/issues/4216) | NanToNum fp16 posinf/neginf 属性 NaN 被 vsel 饱和成 65504（fp32/bf16 正确） |
| INF-2260 | [ops-nn#7640](https://gitcode.com/cann/ops-nn/issues/7640) | GeluBackwardV2 rank26-30 两段 rc=0 但输出未写（金丝雀 32/32 残留；25 正常 31 报错） |
| INF-2261 | [ops-nn#7668](https://gitcode.com/cann/ops-nn/issues/7668)+[#7654](https://gitcode.com/cann/ops-nn/issues/7654)+[#7647](https://gitcode.com/cann/ops-nn/issues/7647) | GeluMul 非连续 out 按 stride 读回错值 / 偏小 out 越界写 128-240B；同型 FatreluMul arch22 缺口（memcheck 实测 illegal write） |
| INF-2262 | [vllm-ascend#20157](https://github.com/vllm-project/vllm-ascend/issues/20157) | KV 块拷贝按 data_ptr 去重跳过 overlay 视图：12 例 NPU 实测 6 败（mismatch 75%），上游 vllm#55112 对应 |
| INF-2265 | [vllm-ascend!20287](https://github.com/vllm-project/vllm-ascend/pull/20287) | MLA padded 头 Q_B 2-D 视图未物化→PROLOG_V3 Q NoPE cosine 0.036→DSpark 接受率 1.47%（修复后 69.83%） |
| INF-2263 | [hccl!3805](https://gitcode.com/cann/hccl/merge_requests/3805) | ReduceScatterV PROD/64位缺 device barrier：16 卡实测 exp:2147483647 act:-501334399；非 V 版 !154 早已修，V 版从未同步 |
| INF-2287 | [ops-transformer#6866](https://gitcode.com/cann/ops-transformer/issues/6866)/[#6868](https://gitcode.com/cann/ops-transformer/issues/6868) | FA arch92 optionalDn 漏算内部 PSE 与 dropout→精度失败（对照矩阵 4 组；修复 !14821 open；静态分析+实测反馈） |
| INF-2272 | [ops-transformer#6859](https://gitcode.com/cann/ops-transformer/issues/6859) | CompressorGrad TH 布局 cuSeqlens 首元素不校验→dX 整块全 0 且 rc=0（训练静默丢梯度） |
| INF-2275 | [ops-nn#7717](https://gitcode.com/cann/ops-nn/issues/7717)+[!12999](https://gitcode.com/cann/ops-nn/merge_requests/12999) | DynamicBlockMxQuant scale2 Interleave 源目标同址覆盖（时序性；修复已 merged，9/9 逐字节一致） |
| INF-2289 | [ops-nn!13086](https://gitcode.com/cann/ops-nn/merge_requests/13086) | FusedCEWithMaxSum V=0 静默错值/vocab 行数不校验越界/可选输入静默忽略（上板验证修复前 5 项全放行） |
| INF-2276 | [ops-transformer#6857](https://gitcode.com/cann/ops-transformer/issues/6857) | PartialRotaryMulGrad partialSlice 长度不校验：1 元素行为随堆内容（0/3984 改写两态） |
| INF-2279 | [ops-transformer#6905](https://gitcode.com/cann/ops-transformer/issues/6905)/[#6904](https://gitcode.com/cann/ops-transformer/issues/6904)/[#6900](https://gitcode.com/cann/ops-transformer/issues/6900) | GenericBlockSparseAttentionGrad 三边界错误：S1=1 dQuery 非 0、BSND lse 形状错、空块追加改梯度 |
| INF-2277 | [ops-math#4254](https://gitcode.com/cann/ops-math/issues/4254)/[#4247](https://gitcode.com/cann/ops-math/issues/4247) | Rsub(s) int64 alpha 窄化判等：2^32+1 被当成 1 静默错值 |
| INF-2280 | [ops-math#4207](https://gitcode.com/cann/ops-math/issues/4207) | InplaceReciprocal complex64 0/次正规输入返回 nan+nanj（complex128 正确） |
| INF-2284 | [ops-math#4258](https://gitcode.com/cann/ops-math/issues/4258)/[#4257](https://gitcode.com/cann/ops-math/issues/4257) | aclnnRange step 截断整数 + 多维 out 全 0 返回成功 |
| INF-2282 | [ops-math#4224](https://gitcode.com/cann/ops-math/issues/4224) | Histc INT8/UINT8 桶计数 8 位回绕出负计数 |
| INF-2285 | [ops-math#4246](https://gitcode.com/cann/ops-math/issues/4246) | StdMeanCorrection N≤correction 方差 0 时 0/0 未防护返回 +Inf（torch 为 NaN） |
| INF-2288 | [ops-transformer#6915](https://gitcode.com/cann/ops-transformer/issues/6915) | InplaceFusedCausalConv1dV2 2D 短序列派发死区恒 561002（260 用例 127 条不可执行；派发/能力判定双定位） |
| INF-2283 | [ops-math#4219](https://gitcode.com/cann/ops-math/issues/4219) | aclnnInplaceAtan fp32 大值饱和 51 ULP |
| INF-2286 | [ops-math#4221](https://gitcode.com/cann/ops-math/issues/4221)/[#4209](https://gitcode.com/cann/ops-math/issues/4209) | aclnnSinc/XLogY 非法 dtype 组合返回 0/0 静默丢虚部 |


### NPU 拒收/留观（窗口内代表）

- **校验缺失/错误码契约/文档声明大族增量**（ops-nn#7723/#7725/#7726/#7695/#7699/#7669/#7672/#7678/#7679/#7682/#7684、ops-math#4255/#4259/#4243/#4240/#4242/#4239/#4234/#4217/#4213/#4212/#4210、ops-transformer#6879/#6876/#6901/#6903/#6906/#6862/#6858/#6863/#6893/#6881、pytorch#5740/#5723/#5724/#5726/#5729、op-plugin#666 等）— 与 10-22~10-28 已判大族同型（非法入参静默接受/错误码语义/文档声明不一致），按族留观维持；数值面真实且实测完整的已单列（GeluMul 族、Rsub、Range、Histc 等）。
- **VECTOR_MASK 寄存器残留类**（ops-math#4259/#4223/#4201、ops-nn#7659）— msSanitizer 检出的寄存器残留，无端到端数值后果证据，留观。
- **ops-nn#7718**（Ascend960 MxA8W4 tiling 错误 AIC_ERROR）— fail-loud 崩溃无静默面，留观；**ops-math#6831**（TopkV2 偶现精度）PR 描述为空模板，证据不足留观；**ops-transformer#6884**（RacePoc WAW）— 自建 PoC 内核的 racecheck 报告非产品路径，留观；**ops-transformer#6889**（A6 smla 精度）需求帖无细节，留观。
- **vllm-ascend#20236**（4 节点 PD+Mooncake 双 Prefill 冻结）、**#20182**（Mooncake SSD 重计算开销）、**#20229**（OffloadingConnector 非连续 KV 失败）、**#20232**（MiniMax M3.1 两节点部署）、**#20172**（GLM-5.3-Flash Mooncake kv_connector）— hang/启动/性能/OOM 域为主，机制未定位，留观；#20203（DSV4.1 KV 初始化 ValueError）fail-loud 不占号。
- **msmodelslim !1126**（GLM-5.3 W4A4C8 量化方案修复）— 修复主体是新增实验配置 + mse_round 反量化崩溃修复（修复前 W8A8 校准崩溃，属 fail-loud），无静默数值缺陷证据，留观；**#646**（量化文档检查）文档类。
- **ops-transformer#6913**（mqfa PA kv 非连续支持）、**#6903/#6897**（SFA 性能优化需求）— 需求类。
- **PR 修复流**（ops-nn !13000/!13009/!13039/!13067/!13068/!13071/!13078/!13086/!13089/!13091/!13092/!13093/!13096/!13097、ops-transformer !14811/!14812/!14821/!14829/!14837/!14839/!14850/!14855、ops-math !6794/!6795/!6808/!6815/!6818/!6829/!6841、op-plugin !6123/!6129/!6130/!6132/!6133、pytorch !48685/!48688/!48689/!48707/!48720、hccl !3806、hcomm !6354 等）— 白盒修复流按惯例不逐条开号（已收录的 !14811/!13086/!12999/!20287/!20243 除外）；!13078 为 INF-2242 修复已补证据回访。

## 2. GitHub 常规 7 仓 + arXiv

- 认证状态：环境无 GITHUB_TOKEN（未认证）；search 走代理出口共享配额，请求间隔 ≥7 秒。21 组查询（7 仓 × correctness/silent wrong output/NaN，created:>=2026-10-22）全部 HTTP 200（首轮含空格编码问题的 3 组以 --data-urlencode 重发后 200；无 403 重试）。total：vllm 9/1(swo)、sglang 4、transformers 2、TRT-LLM 1/2、triton 2/1、tt-metal 4/15/15、flashinfer 29/9/3（correctness/NaN/swo），窗口内合计 96 条。
- 深读 12 条（search item 自带 body + 2 条 core API 单点详情）；既往判定沿用核对 8 条。

### 新增条目（GitHub，12 条）

| 编号 | 来源 | 机制摘要 |
|---|---|---|
| INF-2264 | [sglang#47245](https://github.com/sgl-project/sglang/issues/47245) | retract 后重 prefill 惩罚累加器清零：greedy 同输入输出不同、min_new_tokens 超长，KV 压力下自动触发 |
| INF-2266 | [triton#12506](https://github.com/triton-lang/triton/issues/12506) | MXFP4 _from_float tie-break 全局归约：同值编码随批内无关元素变化（1.5→1.0） |
| INF-2269 | [transformers#50051](https://github.com/huggingface/transformers/issues/50051) | prompt-lookup 把 draft token 喂给有状态 logits processor（CFG/SynthID），输出与不开 lookup 不同 |
| INF-2267 | [tt-metal#63257](https://github.com/tenstorrent/tt-metal/issues/63257) | BH 2D 组播 matmul DRAM 宽分片 in1 网格≠8 列静默错值（PCC 0.9999→0.002-0.07） |
| INF-2268 | [tt-metal#62662](https://github.com/tenstorrent/tt-metal/issues/62662) | ttnn.sum/mean ROW_MAJOR bf16 长宽归约输出精确减半（W=262144 sum=131072） |
| INF-2270 | [vllm#64326](https://github.com/vllm-project/vllm/issues/64326) | DelegatingParser 非流式忽略 token ids：普通文本 </think>/<tool_call> 被当结构边界，stream true/false 解析不同 |
| INF-2271 | [vllm#64517](https://github.com/vllm-project/vllm/issues/64517) | muse/minicpm5/poolside_v1 parser 对 Optional[str] 等 schema JSON 解码字符串参数（"1.10"→1.1；Rust 前端同 bug） |
| INF-2273 | [vllm-ascend!20243](https://github.com/vllm-project/vllm-ascend/pull/20243) | EC memcache 按模型 config 宽度 reinterpret 多模态缓存：Qwen3-Omni 视觉 8192/音频 2048 错 reshape 或静默丢弃 |
| INF-2274 | [tt-metal#63277](https://github.com/tenstorrent/tt-metal/issues/63277) | quasar slice RM 分片行 24B 非对齐静默错值（var→INF-2251 quasar 分片守卫族 item 21） |
| INF-2278 | [tt-metal#62654](https://github.com/tenstorrent/tt-metal/issues/62654) | BH SFPU 读 16-bit Dest 的 fp16 得垃圾：Float16 目标全 NaN、读值算子错 1e8-1e9 步（ULP 全量扫描） |
| INF-2281 | [tt-metal#63259](https://github.com/tenstorrent/tt-metal/issues/63259) | tt_all_gather 1xN mesh 错轴守卫：任意轴含 1 即跳过 gather，2×p150a TP=2 实测原样返回分片 |

### GitHub 拒收/留观

- **tt-metal#62383**（pow NaN/负数）标题与 10-24 已判一致未见新证据——经查近 14 天无既往判定记录，但为 ops 域已有大量同族条目且本条未见正文细节，留观；**#62436**（quasar DRAM 分片 matmul CB 别名，closed）已并入 quasar 族语境（var 关系已有 #62613/#62605 族），closed 未核实修复，不另开号；**#62013/#62024/#62052/#62134** 分别为 INF-2123 系（conv1d=INF-2156、binary_ng=INF-2124）已收录，回访仍 open。
- **flashinfer#7687**（cudnn prefill token_indptr CI 失败）— CI 基础设施帖（dlqa-triage 自动聚合），无独立机制细节，留观；**#7209**（mm_mxfp8 autotune 依赖进程）数值影响为「结果依赖 rank/restart」但未见静默错值判据，留观下轮。
- **triton#12486**（Prefetch 寄存器溢出）纯性能；**#12467**（Interpreter FP8 转换）调试器域——均不占号。
- **TRT-LLM#21124**（KV metrics 拷 rank0 值）指标正确性，观测域非计算故障，留观；**#20991**（Poolside parser 1e999 报错）fail-loud；**#20852**（draft-target 采样分布不保持，closed）与已收录推测解码分布族机制相近但 closed 无修复细节，留观。
- **vllm#64005**（MRV2 Qwen3-32B TP2 只吐 token 0）— 沿用 2026-10-27 留观判定（环境不完整、机制未定位），本轮回访仍 open，无新证据。
- **sglang#45755/#45864** — 10-22 窗口外（created 10-22 边界，近 14 天内未见既往判定）；#45755（flashinfer_mxfp4 静默算错 SwiGLU 类型）机制证据在正文，本轮未深读，留观下轮补查；#45864（QSA indexer 跨请求状态）同留观。

### arXiv

- 2 组查询（silent data corruption+inference、bit flip+LLM，sortBy=submittedDate）HTTP 200 有效。无新论文：2610.30798=INF-2247（10-28 已收）、2610.18557/BARE-AI/2609.19743 均已在库或已判；2610.33475/2610.37033 范围外（云架构讨论/SRAM Ising 机）。

## 3. 回访（近 7 日条目 + 轮巡旧源）

- **INF-2242**（DynamicQuantV2 bf16）：修复 PR [!13078](https://gitcode.com/cann/ops-nn/pull/13078) 已 closed（A2 验证矩阵完整）——已 --add-evidence（#7722 scatter 变体）+ 注记；arch35 变体新立 INF-2255（var）。
- **INF-2160**（Addcmul NaN-as-1）：修复 PR !6791 仍 open，无合入进展，注记。
- **INF-2237**（Kimi-K3 FP8 KV）：vllm#64106 仍 open（core API 单点核验），注记。
- **INF-2245**（Gumbel NaN→token 1023）：vllm#63811 已 closed（2026-10-28T22:29Z）——closed 不自动证明已修复，关闭原因与修复 commit 未核实，注记。
- **INF-2240**（GLM-5.3 视觉全错）：vllm-ascend#20115 仍 open（批次快照），注记。
- vllm-ascend#20146/#20113（INF-2250/INF-2252 源）批次快照仍 open。
- 轮巡（更早入库/长期未复核）：vllm#63276 维持 10-23/10-24 留观（机制未定位）；tt-metal#62052=INF-2156、#62134=INF-2124 源仍 open 无变化；triton#12397=INF-1766 系仍 open。NPU 侧 mindformers/msmodelslim/MindSpeed 系窗口内无遗留 open 旧源需改判。

## 4. 编目与视图

- 新写入 **36 条**（全部 pending_audit）：INF-2254 ~ INF-2289，编号连续，映射如下（以 catalog YAML 为准）。

| 编号 | 来源 | 标题 |
|---|---|---|
| INF-2254 | ops-transformer#6872 | AttentionUpdate 大 LSE 公共偏移权重失真输出翻倍 |
| INF-2255 | ops-nn#7721(+7722) | DynamicQuant arch35 向量路径 bf16 极差溢出（var→INF-2242） |
| INF-2256 | ops-nn#7642 | FastGelu fp16 负区间 exp 中间量溢出返回 -0 |
| INF-2257 | ops-math#4216 | NanToNum fp16 posinf 属性 NaN 被饱和成 65504 |
| INF-2258 | ops-transformer#6870 | IndexerSoftmaxLseV2 S2>16 输出错误 lse |
| INF-2259 | ops-transformer#6910 | MoeGatingTopKSoftmax 310P tiling UB 溢出 |
| INF-2260 | ops-nn#7640 | GeluBackwardV2 rank26-30 返回成功但输出未写 |
| INF-2261 | ops-nn#7668/#7654/#7647 | GeluMul/FatreluMul out 校验缺失错写/越界写 |
| INF-2262 | vllm-ascend#20157 | KV 块拷贝按 data_ptr 去重跳过重叠视图 |
| INF-2263 | hccl!3805 | ReduceScatterV PROD/64位缺 device barrier |
| INF-2264 | sglang#47245 | SGLang retract 清零采样惩罚历史 |
| INF-2265 | vllm-ascend!20287 | MLA padded 头 Q_B PROLOG_V3 投影错值 |
| INF-2266 | triton#12506 | Triton MXFP4 编码依赖同张量无关元素 |
| INF-2267 | tt-metal#63257 | BH 2D 组播 matmul DRAM 分片 in1 网格失配 |
| INF-2268 | tt-metal#62662 | ttnn.sum/mean ROW_MAJOR bf16 长宽归约减半 |
| INF-2269 | transformers#50051 | prompt-lookup 推进有状态 logits processor |
| INF-2270 | vllm#64326 | DelegatingParser 忽略 token ids |
| INF-2271 | vllm#64517 | 三工具解析器非纯 string schema JSON 解码 |
| INF-2272 | ops-transformer#6859 | CompressorGrad cuSeqlens 不校验输出全 0 |
| INF-2273 | vllm-ascend!20243 | EC memcache 按配置宽度重排多模态缓存 |
| INF-2274 | tt-metal#63277 | quasar slice RM 分片行未对齐（var→INF-2251） |
| INF-2275 | ops-nn#7717+!12999 | DynamicBlockMxQuant Interleave 源目标同址 |
| INF-2276 | ops-transformer#6857 | PartialRotaryMulGrad partialSlice 不校验 |
| INF-2277 | ops-math#4254/#4247 | Rsub(s) int64 alpha 窄化判等错值 |
| INF-2278 | tt-metal#62654 | BH SFPU 读 16-bit Dest fp16 得垃圾 |
| INF-2279 | ops-transformer#6905/#6904/#6900 | BlockSparseAttentionGrad 三边界数值错误 |
| INF-2280 | ops-math#4207 | InplaceReciprocal complex64 零输入 NaN |
| INF-2281 | tt-metal#63259 | tt_all_gather 1xN mesh 错轴守卫跳过聚合 |
| INF-2282 | ops-math#4224 | Histc INT8/UINT8 桶计数回绕出负数 |
| INF-2283 | ops-math#4219 | InplaceAtan fp32 大值区饱和 51 ULP |
| INF-2284 | ops-math#4258/#4257 | aclnnRange step 截断/多维 out 全 0 |
| INF-2285 | ops-math#4246 | StdMeanCorrection 小样本 0/0 返回 Inf |
| INF-2286 | ops-math#4221/#4209 | Sinc/XLogY 非法 dtype 组合 0/0 丢虚部 |
| INF-2287 | ops-transformer#6866/#6868 | FA arch92 optionalDn 漏算 PSE 与 dropout |
| INF-2288 | ops-transformer#6915 | FusedCausalConv1dV2 2D 短序列派发死区 |
| INF-2289 | ops-nn!13086 | FusedCEWithMaxSum V=0/行数不校验静默错值 |

- 更新 5 条：INF-2242（+证据 #7722 + 回访注记）、INF-2160（回访注记）、INF-2237/INF-2245/INF-2240（回访注记）。
- `catalog_tool.py stats`：2288 条（active 2253 + pending_audit 35）。`build` 完成（CATALOG.zh.md、fault_catalog.csv、xlsx）；`validate` ok:true（n_errors 0，warnings 52 为既有存量）。
- var 关系回显核对：INF-2255→INF-2242（「CANN DynamicQuantV2 bf16 极值量化全饱和」✓）、INF-2274→INF-2251（「ttnn quasar 分片布局守卫失配静默错值三连」✓）。
- 训练库跨库查重：9 个 NPU 候选在 training_error_review 0 命中，无 cross 关系写入。

## 5. 未覆盖范围与 API 账本

- GitHub core API 仅单点核验 4 个 vllm issue 状态（64106/63811/64326/64517）；新候选 12 条按 search item 自带 body 判定，未拉评论线程——若评论中出现维护者定位/驳回，判定可能需改判；已按 documented 收录。
- search 全部 200，无限流等待（累计等待 0，上限 5 分钟未触及）。
- NPU PR 修复流正文未逐条深读（~30 条）；仅收录涉及静默数值证据的 6 条 PR（!14811/!13086/!12999/!20287/!20243/!3805）。
- 强信号 245 条中约 30 条深读正文，其余按标题+关键词+既往大族判定留观；sglang#45755/#45864（10-22 创建，本窗口边缘）留观下轮补查。
- GitCode/AtomGit issue 评论线程未拉取（无低成本批量入口），仅按正文判定。
- arXiv 仅 2 组主题词（SDC/bit-flip），KV 损坏/数值正确性主题词未单独跑（结果集与既往重叠度高，缺口小）。

## 6. 结论

36 条新机制入库（待审计），5 条既有条目回访/补证。本轮覆盖：NPU 22 源完整批次窗口（1010 条窗口内新建、245 强信号、约 30 深读）+ GitHub 7 仓 3 关键词（96 条窗口内）+ arXiv 2 组；缺口集中在 GitHub 评论线程、NPU PR 修复流正文与 arXiv 补充主题词。
