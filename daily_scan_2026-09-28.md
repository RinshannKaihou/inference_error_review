# LLM 推理故障每日扫描报告 — 2026-09-28

- **扫描日期**：2026-09-28（UTC）
- **扫描窗口**：GitHub `created:>=2026-09-21`（7 日完整窗口，`cutoff=$(date -u -d '7 days ago' +%F)` 实际计算；与 09-27 扫描窗口 created:>=2026-09-20 全重叠——09-27 及更早判定条目全部按既有判定跳过，本日新判定集中于 09-27 扫描执行（约 09-27 23:18 UTC commit `315d33b` 前）之后创建的 47 个 issue/PR）；**arxiv 查询在连续两天 HTTP 406 后恢复**——6 组引号短语+AND 查询全部 200 有效，等效覆盖 09-19..09-28 失败窗口补查
- **目标仓库**：vllm-project/vllm、sgl-project/sglang、huggingface/transformers、NVIDIA/TensorRT-LLM、triton-lang/triton、tenstorrent/tt-metal、flashinfer-ai/flashinfer
- **主文档基线**：1229 条（2026-09-27 扫描后）；编号编目 1115 条 + K* 扩展 114 条

## Git 前置检查（第 3 节）

- 分支 `main`、上游 `origin/main`、`git status --porcelain -uall` 为空 ✅
- `git fetch origin main` 成功；`HEAD` = `origin/main` = `315d33b39f99ac4d4db588230526b2db14d15be0` ✅ → 开始扫描

## API 调用记录

| # | 类型 | 查询 | 结果 |
|---|------|------|------|
| 1 | GH Search | `repo:vllm-project/vllm is:issue created:>=2026-09-21 wrong OR incorrect OR corruption OR NaN` | 43 命中（单页取全） |
| 2 | GH Search | `repo:vllm-project/vllm is:issue label:bug created:>=2026-09-21` | 63 命中（单页取全） |
| 3 | GH Search | `repo:sgl-project/sglang is:issue created:>=2026-09-21 wrong OR incorrect OR corruption OR NaN` | 18 命中（单页取全） |
| 4 | GH Search | `repo:huggingface/transformers is:issue created:>=2026-09-21 wrong output OR incorrect OR NaN` | 7 命中（单页取全） |
| 5 | GH Search | `repo:NVIDIA/TensorRT-LLM is:issue created:>=2026-09-21 wrong OR incorrect OR corruption` | 1 命中（#19501 RFC，09-21 已判维持） |
| 6 | GH Search | `repo:triton-lang/triton is:issue created:>=2026-09-21 wrong OR miscompile OR NaN` | 2 命中（#11889 已判维持 + #11993 本日新判） |
| 7 | GH Search | `repo:tenstorrent/tt-metal is:issue created:>=2026-09-21 numerical OR wrong OR pcc OR silent` | 159 命中（**两页取全 100+59**） |
| 8 | GH Search | `repo:flashinfer-ai/flashinfer is:issue created:>=2026-09-21 wrong OR incorrect OR NaN` | 4 命中（#5389/#5393/#5438 已判维持 + #5637 本日新合并） |
| 9 | GH Search | `repo:vllm-project/vllm is:issue created:>=2026-09-21 quantization OR "KV cache" OR deterministic` | 95 命中（单页取全） |
| 10 | GH Search | `repo:vllm-project/vllm is:issue created:>=2026-09-21 silent OR "bit flip" OR "weight corruption"` | 58 命中（单页取全） |
| 11 | GH Search | `repo:sgl-project/sglang is:issue label:bug created:>=2026-09-21` | 0 命中 |
| 12 | arxiv | `"silent data corruption" AND LLM` | 200，6 命中全已编目（SProbe/TrainSDC/SCOUT/LLM-PRISM/2604.00726/2502.12340） |
| 13 | arxiv | `"bit flip" AND LLM AND inference` | 200，8 命中全已编目或已判（JITterFlip/UEP/RangeGuard/2603.10042/BitFlipScope 等） |
| 14 | arxiv | `"KV cache" AND corruption` | 200，15 命中（2609.25881 为 OR 噪声非 LLM 域、其余已编目或方法论文不收录） |
| 15 | arxiv | `"soft error" AND GPU AND inference` | 200，3 命中（2609.16742 CNN 边缘域已判不收录、2601.19912 已编目、2104.09455 非 LLM 域） |
| 16 | arxiv | `LLM AND inference AND "numerical correctness"` | 200，3 命中均为非故障域论文 |
| 17 | arxiv | `LLM AND inference AND "wrong output"` | 200，0 命中 |

GitHub Search API 11 组查询全部有效、**分页完整（fetched == total 逐组核对：43/63/18/7/1/2/159(100+59)/4/95/58/0）**、无限速 403、无 Validation Failed；查询间 ≥8s 间隔。去重后 350 个唯一 issue/PR（跨 11 组查询重叠），其中 **47 个为 09-27 扫描执行后新建需判定**、303 个重叠区按既有判定跳过。候选正文优先取自 Search API 缓存（响应自带 body）；core REST 详情/评论/PR/timeline **15 次调用全部成功**（`--http1.1`，未认证额度 60/小时内：12 次来源回访快照 + PR #58940 评论 + issue #58930 评论 + tt-metal #58046 评论与 timeline + PR #58180 状态）。**未认证运行（环境无 GITHUB_TOKEN）。**

## 已编目/已判条目（本次跳过，重叠区去重）

- vLLM：#58902（→9.30）、#58930（→10.88）、#58894（→6.259）、#58824（→10.87）、#58807（→13.40）、#58729（→7.99）及其余 09-21..09-27 判定项（#58616/#58575/#58599 等 300 余项按昨日报告判定跳过）
- transformers：窗口内 7 命中全为 09-27 及更早已判（#49134→13.41、#49126 CI triage 等）
- SGLang/tt-metal/flashinfer/triton：#41192（→5.46）；tt-metal #57988–#58046（09-26/09-27 已判，#58046→1.262 本日回访见下）；#5389/#5393/#5438（→7.96 等）已判维持；#11889（已判维持）

## 🆕 新发现（本次合并，共 8 条）

### A. 全新子签名（4 条）

1. **[新签名] tt-metal #58177 — matmul `packer_l1_acc=True`（非 fp32 输出默认）在核内恰好 2 个 K 块（无 fused bias）时被静默关闭**：`packer_l1_acc_en = packer_l1_acc && (bias&&num_blocks>1 || num_blocks>2)`（2D mcast / 1D in1-mcast / Reuse 三工厂同款），partials CB 格式绑定该 flag——关闭时部分和以**输出格式**落盘，block-float 输出把部分和舍入到 block float，**K 拆 2 块反而比 4 块更不准**（bf4_b PCC 0.96224 vs 0.97268；bf8_b 0.99983 vs 0.99989）；1D in0-mcast/DRAM-sharded 工厂已用 `num_blocks>1`（正确）。→ 归入第 1 类 **1.265**，Cov NEW（matmul partials 精度格式按块数谓词静默分支），verified。
2. **[新签名] SGLang #41494 — GLM-5.3-Flash（fp8）DSA k-pool indexer 16K 长上下文 NIAH 数字损坏**：模型逐字引用正确 needle 句子却输出错误数字（gold `2921859`→`7803`、`5418934`→`5906`），16K 准确率 55%（4K=100%/8K=95%，max_tokens 平坦 45-55% 排除截断）；**三引擎同输入/同权重/同 indexer 参数对照：transformers（fp32 打分）✓、vLLM 0.30.1（同款 fp8 deep_gemm sparse-MQA）✓、SGLang ✗**——把缺陷钉死在 SGLang k-pool indexer 实现层；`--dtype bfloat16` 仅部分恢复；v0.5.20 与 main 均中。→ 归入第 6 类 **6.260**，Cov NEW（GLM kpool 选择损坏族 Hopper 主路径首例，与 8.73/8.83/8.94 跨硬件对偶），documented。
3. **[新签名] vLLM #59090 — `/v1/chat/completions` 对 message `role` 只验类型不验取值，chat template 不认识时静默丢整条消息仍 200**：`usr`/`assistan`/`USER`/`bababoi`/空串全部蒸发该轮（prompt_tokens 39→23、prompt 塌到 3 token 后模型自由联想）；类型错（`7`/`null`）被 400 证明校验存在但只查类型；同请求 `/v1/messages` 正确 400。→ 归入第 10 类 **10.89**，Cov NEW（"校验存在但只查类型"形态），verified。
4. **[新签名] flashinfer #5637 — `chain_speculative_sampling` generator 托管路径 Philox offset 记账错位，跨调用重用随机坐标**：wrapper 预付 `B*(N+1)` 上取整（B=1,N=3 → 起点 4/8）而 kernel 实耗最多 **N+2** 字（拒绝时 j+1 前缀 + N-j 诊断重抽 + 1 恢复），首次拒绝的恢复抽取（坐标 8）被下一调用的首次接受判定复用；208 种子配对研究：托管 208/208 vs 显式不相交 97/208（p=9.2e-33）；main `85bfa5e4b5` 与测试 pin 字节一致。→ 归入第 11 类 **11.89**，Cov NEW（RNG 流消费量契约错位首例），verified。

### B. 已知类别新变体（4 条）

5. **[已知类别] tt-metal #58078 — 行主序 `binary_ng` host 只按操作数 A 的元素大小打包一个 reader stride，两个 CB 用同一字节偏移/长度/L1 行距**：混合浮点 dtype 时 fp32 操作数行按 bf16 节距打包——**fp32 行后半成为下一行前半**（2×32 实测 `[2..17,2..17]`）；行/列/标量/混合广播全失败、TILE 布局正确。→ 归入第 1 类 **1.263**，Cov var:1.221（binary_ng 混合 dtype 域——数据搬运单一 stride 维度），verified。
6. **[已知类别] tt-metal #58181 — `MatmulMultiCoreReuseProgramConfig` + A tile 高度 <16 + bfloat8_b B + 多 K 块输出错值（PCC 至 -0.0035/NaN，错值随运行变化指向陈旧数据）**：验证器**接受**该配置且它是 block-float B + 矮 tile 下唯一放行的工厂（mcast 被拒、Reuse 被留）——按验证器指引恰好落进坏路径。→ 归入第 1 类 **1.264**，Cov var:1.119（MatmulMultiCoreReuse 族"验证放行≠内核正确"第三点），verified。
7. **[已知类别] vLLM #58973 — V2 speculative prefill 的三个 fused RMSNorm 变体经 autotune 选出与 AR 不同的 reduction-block 配置，Qwen3-4B greedy 首 token 翻转（334→95456）**：分歧发生在初始 target prefill、任何 draft/rejection 决策之前；对齐 RMSNorm launch 设置后 36 层 prompt K/V、hidden state 与 prefill logits **bitwise 等于 AR**（干预级因果闭合）；decode 侧对照证明 rejection sampler 无罪。→ 归入第 4 类 **4.27**，Cov var:4.26（Inductor autotune 非确定族从"跨 rank"扩到"跨模式"），documented。
8. **[已知类别] SGLang #41569 — `_mimo_v2_overrides` 在 SM100 按 `quant_method=fp8` 标签恒选 `flashinfer_trtllm`，无视同 config 的 `store_dtype: mxfp4`（routed experts 为 packed FP4）**：启动即 kernel dtype check 崩（fail-loud）；`ModelConfig.is_fp4_experts` 已检测该布局但 override 不读；选择层免 GPU 最小复现确认。→ 归入第 7 类 **7.100**，Cov var:7.98（量化配置元数据与实际权重格式错配——按标签而非按结构分派域），verified。

### C. 评估后不收录（记录备查）

- **vLLM #59086**（`VLLM_BATCH_INVARIANT=1` 下 AWQ 模型 sm8x 默认模式批非不变——A 批入 B 后 greedy token 位 4 翻转、logprob gap 0.0135、3/3 轮；bf16/FP8 对照不变、`--enforce-eager` 复原、cudagraph NONE 不变——**inductor 引入**；prefix caching 冷/热全长度分歧）：证据扎实（对照矩阵完整），与 4.17/4.22 batch-invariant flag 失效族同域，但 0.27.1/0.29.0 旧栈 AWQ×compile 组合、无根因定位（哪个 AWQ kernel 被 compile 后失去批不变），留观待主线复现或根因后并入 4.17 语境。
- **tt-metal #58106**（UINT8 transpose/permute/concat/untilize 全零——16-bit Dest 硬件约束，tilize 已为 UINT8 开 32-bit Dest 而四个数据搬运 op 未跟、无 dtype 校验、无测试覆盖）：静默错值真实且机制清晰（kernel descriptor 表直接证实），但 UINT8 数据搬运非 LLM 推理数值主路径、Wormhole 状态待查，留观待修复 PR 或 LLM 域触点后编目。
- **tt-metal #58132**（#56292 删 legacy sqrt/rsqrt/reciprocal 兼容路径使四模型门禁连红三晚——ModernBERT PCC 0.9893<0.99、BGE-M3 multi-vector 0.456、Flux.1 schnell 27.64<28.0、SDXL group-norm perf 快出带；bisect 四腿钉死 #56292）：**"清理 PR 意外移动门禁数值"与 10.86（Gemma1 GELU 守卫移除）同谱系**，但报告本身是回归/重基线决策流程帖（问"是否预期"）、非静默故障新机制，留观维护者决策（若定性为未预期数值回归则按 10.86 谱系编目）。
- **tt-metal #58138**（gtz/lez 在 INT32_MIN 溢出翻转——`0 - INT32_MIN` 回绕为负）：精确单值边界（4096 扫描各恰 1 错）、与 1.167（INT32_MIN 残差校正整数族）同根因域（sfpi 整数比较无溢出保护），单点未及 1.167 的多算子面，作族语境不占号。
- **tt-metal #58142**（`ttnn.max/min/sum` 接受 UINT32 但走 FPU 路径返 fp32 位型——0xFFFFFFFF → 0x7F800000(+inf)）、**#58150**（UINT16 cumsum 全零——16-bit 加法 forced 32-bit Dest 读 LO16）：机制清晰、静默错值真实，但 UINT32 归约/UINT16 累积均非 LLM 推理主路径（#31575 放行系 VADv2 CI 解锁），与 1.265 同日收录的 matmul 主路径相比触发面窄，留观。
- **tt-metal #58082**（`test_softmax_with_3D` N150 Merge Gate 约 1% ND 竞态——2/199 失败、ATOL 1.0 级损坏指向部分 tile 归约、两假设（per-core 竞态/mask_padded_data）未定位）：竞态证据真实（连续提交对照、跨 SKU 统计），根因未定，留观定位后编目。
- **tt-metal #58061**（Quasar sfpi `vFloat !=` 在 `b == -a` 时误判相等——整数差 0x80000000 被按 FP32/SMAG32 读作 -0）：静态审计系列、codegen 级复现完整，但 in-tree 唯一调用点（对 0 比较）免疫、Quasar 预生产；同批 #56659（±0/NaN 位型语义）已留观先例。
- **tt-metal #58189**（`shrink_shard_for_rm_page_alignment` Lenient 模式对 HEIGHT/BLOCK sharded RM 输入原样返回未对齐 spec——三 caller 下游 L1 分配器或静默损坏）：防御性加固议题（`repeat` 已有正确模式可抄），实测失败以 assert（Max ULP Delta: inf）显形、当前测试 adaptive skip，留观。
- **triton #11993**（`tl.argmin/argmax` 单 NaN 输入返回 NaN lane 作 min/max 索引而值正确）：与 1.151（argmax 比较器 NaN/±0 语义 dtype 分裂）同域——NaN 输入下索引契约争议（PyTorch 语义 vs 当前行为），留观维护者定性。
- **vLLM #58960**（Qwen4Exp QSA key-cache 派生视图钉住 profiling KV cache 9.76 GiB 致 FULL graph capture OOM——closed，根因+修复+回归测试随 PR）：内存泄漏域 fail-loud（启动 OOM）、无输出影响，不占号。
- **vLLM #59027/#59045**（ROCm v0.30.0 GLM-5.3-Flash / DSV4.1-Flash 无法启动——#57252 `record_logical_topk_ready` 漏 cherry-pick（#58575 同族第二例）/ Engram pinned-host offload 在 `deepseek_v4_1→v41` 重命名中丢失 +50 GiB profile workspace OOM）：均 fail-loud 启动崩溃、根因与修复明确；#59045 的 offload 丢失（0.29→0.30 重命名丢配置接线）与 10.86 谱系相似但结果面 fail-loud，发布工程域。
- **vLLM #59064**（Kimi-K3 MXFP4 `flashinfer_trtllm` 选无 SiTu 内核的组合启动崩 "No kernel found ... mActType: 2"——modular/monolithic 双路径）：fail-loud；与 7.100 镜像（SGLang 读标签选错 runner vs vLLM 用户显式选择无内核组合不拒绝），选择期拒绝缺失语境记录于 7.100。
- **vLLM #59065**（FlashInfer MLA DCP>1 混合 spec/non-spec decode warmup 形状 ValueError——`causal_seqlens_kv_global must have shape (9,), got (2,)`）：fail-loud 启动失败，混合 batch 形状契约缺口。
- **vLLM #59043/#59089**（derender 特性两缺陷：SentencePiece 首 token 前导空格丢失 / 非流式尾不完整字节多出 U+FFFD——流式路径两者皆正确）：derender（token-in/token-out 无 GPU 重建）为新特性路径、非主服务路径，输出保真域留观待该特性 GA。
- **vLLM #59087**（`--tool-call-parser mistral` 非 Mistral tokenizer 启动通过后**所有** chat 请求 500——含无 tools 请求，`/health` 200 掩盖）：fail-loud 服务面（500）、启动期可判定（修复方向明确），不占静默号。
- **vLLM #58969**（bench serve 结果序列化五缺陷——e2el 指标全落 untracked、scrub 列表缺两数组等）：bench 工具域。
- **vLLM #58959/#59016/#59018/#59022/#59055/#59062/#59024**（RFC/feature/性能提案/hidden-state 跟踪）、**#59057**（官方镜像 import 崩 torchvision::nms——打包缺陷 fail-loud）、**#58980**（DCP FP8 sparse attention 处理他 rank 掩码槽——性能提案）；**tt-metal #58103/#58113/#58114/#58141/#58155**（subdevice 调度 feature/model bring-up 跟踪/测试基建/功率墙分析）、**#58090/#58091**（workflow/脚本重构）：feature/RFC/工具域，不收录。

## 来源回访（近 7 日新增条目引用来源 + 额度轮巡旧来源）

本日回访 12 个来源（core REST 快照全读，其中 4 个含评论级深查）：

- **1.262（tt-metal #58046）**：**1 条新评论（09-28 rmillerTT）——确认 2D 工厂同缺陷**（block-sharded 输出 + per_core_M==out_block_h==1 + out_block_w<per_core_N 同错值，同因 partials 跨 N 块原地共享输出 shard），修复 PR **#58180** 覆盖两工厂（本轮 core REST 核实 state=open/merged=False）。**条目来源行已补回访记**，判定不变。
- **10.88（vLLM #58930 + PR #58940）**：Kylinny 09-27 认领并在 PR #58940 对 legacy/新两种形态的嵌套 schema 施加同一 unsupported-feature 检查；PR 09-28 仍有评论更新（5 条 review 线）、**open 未合并**。条目不动。
- **修复 PR 状态核实（core REST）**：#55601（6.204/6.259 修复）open 未合并；#58890（9.30 修复）open 未合并；#59016 相关无 PR。三个"发布带病 + 修复 open"暴露面维持。
- **6.259（vLLM #58894）**：state/comments 不变（3 评论，最后 09-27）——无新证据，条目不动。
- **9.30（vLLM #58902）**、**10.87（vLLM #58824）**、**13.40（vLLM #58807）**：经 core REST 快照确认 state/comments 不变（0/3/1 评论），无状态翻转，条目不动。
- **6.260（SGLang #41494）**、**11.89（flashinfer #5637）**：本日新合并主来源，快照存档（均 open、0 评论、无 triage 响应）。
- **tt-metal #57865（ring_joint_sdpa 谱系，#58030 引用）**：state/comments 不变，语境引用无需改条目。
- 未检查完说明：近 7 日新增条目引用来源约 108 个（09-21..09-27 合并 103 条 + 本日 8 条），本日按 core 额度回访上述 12 个高价值来源（昨日 6 条新合并主来源全量 + 3 个修复 PR 状态 + 2 个本日主来源存档 + 1 个谱系引用）；其余约 96 个经本日 q01–q11 search 快照确认无 state 翻转/评论数突变（快照含全部窗口内活跃 issue 的 state/comments 字段），评论级深查缺口滚动至后续扫描。

## 合并动作摘要

- 主文档新增 **8 条**（4 个新子签名 + 4 个已知类别变体）：1.263、1.264、1.265、4.27、6.260、7.100、10.89、11.89
- 总条数 1229 → **1237**；编号编目 1115 → **1123**（K* 扩展 114 不变）
- 分类分布：第 1 类 261→264、第 4 类 26→27、第 6 类 259→260、第 7 类 99→100、第 10 类 88→89、第 11 类 87→88
- 轴分布更新（按解析器实测）：infra_off 1060→1068、config 267→270（+6⚠ 不变）、hw_specific 316→320、other_stack 234→235（+25⚠ 不变）、documented 706→708、verified 516→522、NEW 890→894、var:x 336→340
- 条目更正（回访补记，判定不变）：1.262（tt-metal #58046 补 rmillerTT 确认 2D 工厂同缺陷 + 修复 PR #58180 open）
- 观察：**4.27 把 4.26 的 Inductor autotune 非确定根因面从"跨 rank"扩到"跨模式"**——同一进程内 AR 与 spec prefill 对同源 RMSNorm 选不同 reduction-block 配置即翻首 greedy token，干预对齐后 bitwise 相等完成因果闭合；**1.264 与 1.262/1.119 使 MatmulMultiCoreReuse"验证放行≠内核正确"契约缺口成三点**（块数超核丢块/豁免谓词放行/矮 tile+block-float 下唯一放行工厂恰是坏路径）；**1.265 首录"中间精度格式按块数谓词静默分支"**（K 拆 2 块反而比 4 块更不准是可检测指纹）；**6.260 是 GLM kpool 选择损坏族的 Hopper 主路径首例**（三引擎对照把 fp8 方法本身排除）；**10.89 揭示"校验存在但只查类型"新形态**（任意 role 字符串静默蒸发整条消息）；**11.89 是 RNG 流消费量契约错位首例**（wrapper N+1 预付 vs kernel N+2 实耗）
- 未覆盖范围：q07（tt-metal 159 命中）与 q01/q02/q09 中 09-21..09-27 创建、既有判定条目未逐一重读正文（按既有判定跳过）；tt-metal 静态审计三批 meta-issue（#57360–#57367/#57510 附近/#57674–#57694）维持整批留观未逐项展开；约 96 个来源仅经 search 快照确认无状态翻转（无评论级深查）；**arxiv 09-19..09-26 失败窗口已随本日恢复补查**（6 组引号短语查询等效覆盖该窗口的 SDC/bit-flip/KV/soft-error/正确性关键词面，全部命中已编目或已判——除查询词面外的该窗口论文不在此补查范围内）
- 导出与校验：`python3 export_faults.py`（1237 行 CSV/XLSX）→ `python3 verify_catalog.py` **PASS 1237 条：正文汇总、CSV、XLSX、字段边界一致**
