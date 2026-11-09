# 每日扫描日报 2026-11-09（推理库）

## 0. Git 前置检查

- 分支 `main`，上游 `origin/main`；`git status --porcelain -uall` 为空；`git fetch origin main` 成功；HEAD 与 origin/main 一致（94187515920f0fa2a760c49c4868a5e6f9713b7f）。通过，开始扫描。
- Cron 前置脚本 NPU 批次完整：**22/22 源全部 complete，failures 为空**。窗口 2026-11-08T22:01:00Z → 2026-11-09T23:00:12Z。registry_hash 2f6422d0…（22 源，与昨日一致）。

## 1. NPU 新增来源（共享批次）

- 批次：`/root/.hermes/profiles/third/cache/fault_scan_sources/batch-20261109T230012939639-cx7wiv5d`，generated 2026-11-09T23:00:12Z。inference scope 候选索引 28285 条（滚动缓存全量）。
- **本轮新项判定窗口**：created ≥ 2026-11-08T22:01:00Z（本批次采集水位），共 **805 条**（230 issue + 575 PR；GitCode 664 + GitHub-NPU 141），**全部条目标题人工过目**（issue 全列、PR 先关键词筛 250 条再全量 325 条复看）。重点候选全文深读 ~37 条（全部使用批次缓存正文）。
- 近 14 天日报完整 URL 查重：本轮拟收 12 个 NPU 来源 URL 全部 0 既往判定命中（#21164 命中为 INF-2443 本身，走 update）；编目 `find` 查重 0 命中；训练库跨库查重 13 URL 全部 0 命中（无 cross 关系）。
- 机制级比对：#5972/#5960/#5961/#5962 与已收 torch_npu 编译器条目（INF-2460 Welford epilogue 无掩码、11-07 #5938 分核尾块 mask）均为不同缺陷位置（rsplit 切片改写锚点失配 / 模板 codegen 漏调上游 / 取模化简 / 父子轴等价漏判），分立编号；triton-ascend #3043 自述为 #2332（仿射前缀掩码已修）的残余形态，#2332/#2648 均未入库，无 var 目标；#21387 与 INF-2464（sglang router 量化配置不传）同为"量化描述与 checkpoint 不匹配静默错载"主题但机制与栈不同（融合 shard 分配 INT8 vs router 构建丢 quant_config），分立。

### 新增条目（NPU/GitHub-NPU 来源 11 条）

| 编号 | 来源 | 机制摘要 |
|---|---|---|
| INF-2478 | [vllm-ascend#21285](https://github.com/vllm-project/vllm-ascend/issues/21285) | mxfp8/fp8 KV cache 按槽位张量 gather 索引读返回位级损坏数据（950DT、v0.30.0-a5、Qwen3.8-27B-w8a8c8-mxfp8 实测）；同一缓存顺序分页读正确、bf16 cache 精确——缺陷仅量化 KV 索引访问路径；稀疏/LSA 式 top-k 读全坏；规避=bf16 镜像（+4GB/16 层） |
| INF-2479 | [vllm-ascend#21387](https://github.com/vllm-project/vllm-ascend/pull/21387)（修复 PR 转收） | Kimi MLA 融合 QKV/Gate 三 shard 的 ModelSlim 映射只描述两个：FLOAT gate 被按 INT8 分配、BF16 值 0.25 静默 cast 为 0、scale/offset 哨兵未填；nightly GPQA draft 接受率 22.26%→1.53%；真实 checkpoint 布局佐证（Eco-Tech/Kimi-K3-w4a8 layer 11） |
| INF-2475 | [pytorch#5972](https://gitcode.com/Ascend/pytorch/issues/5972) | torch.compile 多轴归约（含 dim0 且 ≥2 轴）结果恒为正确值 48 倍（=910B4 向量核数）：rsplit 部分核切片改写只匹配扁平 r 循环锚点，promoted r-tree 嵌套形态零匹配静默跳过，lo/hi 悬空 48 核各算全量再 combine；触发矩阵 5 形状全错；已排除 split_reductions/outer-rsplit/combo 三路径 |
| INF-2488 | [pytorch#5960](https://gitcode.com/Ascend/pytorch/issues/5960) | max_autotune Triton 模板 GEMM 融合 epilogue 缺 fp32→fp16/bf16→fp32 截断 cast（codegen_template 漏调上游 _compute_fusion_metadata，has_epilogue_fusion 恒 False），融合与未融合路径数值不一致；mm/addmm 均中；修复 MR !49099 已提交 |
| INF-2474 | [pytorch#5961](https://gitcode.com/Ascend/pytorch/issues/5961) | eliminate_modular 用单轴长代入判定完整表达式，(6*i)%3 被错化简为 6*i 生成错误 load 地址（记录在案 eager 24 vs 编译 22）；修复 !49132（dev !90 已合入），离线符号检查 560 表达式无不等价 |
| INF-2468 | [pytorch#5962](https://gitcode.com/Ascend/pytorch/issues/5962) | 父子轴替换只比结构漏判复合符号数值等价（s*u+s*z vs s*floor(…/s)），repeat 场景 load 换子轴 store 留父轴行映射错；修复 !49133（dev !91 合入），11-05 源码 A/B 两用例 FAIL→PASS |
| INF-2469 | [triton-ascend#3057](https://github.com/triton-lang/triton-ascend/issues/3057) | stride-2 行栅格 + 列轴单位步长 2D tl.load：奇数行 lane 静默返回 mask other 值（含默认 0.0），位级稳定跨 run，无警告无错误；affine-tap 重构可规避 |
| INF-2477 | [triton-ascend#3058](https://github.com/triton-lang/triton-ascend/issues/3058) | 非仿射寻址三连：负坐标进地址算术时 clamp/masked/where 三防护全错且错值位级相同；ptr+scalar+3D 与 gather 式重复地址两形态标量基址被静默丢弃读错地址段 |
| INF-2476 | [triton-ascend#3043](https://github.com/triton-lang/triton-ascend/issues/3043) | MaskState::parseDiv 硬编码禁用，除法派生索引掩码不可转访问范围 → load 无谓词化（掩码仅剩尾部 select），掩码外 lane 仍发 GM 事务：生产 vLLM 行 gather 尾部越界读至多 9KB，落未映射页即 vector core exception 507035 杀进程；带 IR 对比与最小复现；#2332 残余形态 |
| INF-2473 | [ops-transformer#7287](https://gitcode.com/cann/ops-transformer/issues/7287) | FIA arch22 split-KV combine LSE_MODE=OUT_ONLY：LSE DataCopyPad（MTE3）后不等完成即返回，tvUbTensor 被 non-split 任务 Brcb 覆写——MTE3 读/V 写缺同步，combine LSE 输出异常（竞争窗口相关非必现）；修复 !15558 补 MTE3_V(EVENT_ID5) 同步对 |
| INF-2472 | [ops-nn#8403](https://gitcode.com/cann/ops-nn/issues/8403) | ActsULQ fp16：clamp_min_mask 连续块写 0（应 1）、y/loss 同块异常、同输入 10 轮 MD5 跨轮不一致（max_abs_error=1.0，阈值 0.1）；fp32 无此现象；修复 !13928 补 fp16 BC 的 V_MTE2(WAR) 同步 |
| INF-2471 | [ops-math#4637](https://gitcode.com/cann/ops-math/issues/4637) | aclnnIsInScalarTensor RegBase（950）整型张量+浮点标量<10 元素分支推导 double 而非 float32：16777216.5 vs int32[9] 含 16777217 在 float32 舍入边界应 True 实测 False（torch.isin 为 True）；对应 PR !7139 |

（11 条 NPU + 10 条 GitHub 常规 = 21 条，编号连续 INF-2468~2488。）

### NPU 留观（本轮判定）

- **防御校验/审计族延续**（沿用 11-05/11-07 族口径）：ops-transformer #7250（MoeComputeExpertTokens out 长度校验漏 == numOfExpert，memcheck 实测越界写 size=numExperts×4−outBytes 四项算术全对、master 2c1938672 仍在——**族内证据密度最高的一条**，若维护者确认系统性整改可随族代表转收）、#7290（AttentionToFfnV2 context 长度校验缺失越界读，修复 !15571）、#7296（GroupedMatmul 950 split-K 早退成功漏 transpose 校验，UT 确认，修复 !15577）；ops-math #4633（Bincount minlength<0 静默放行但输出与合法调用一致——无数值损坏，修复 !7254）、#4650（TransData RT2 infershape 无条件覆写输出 shape，后果为格式转换失败未见静默错值，修复 !7275）；ops-nn #4630（Concat pure_copy blockCount uint16 截断**静默丢行**机制明确但白盒定位无上板实测，closed 随加固流；同报 0 维除零 SIGFPE 为 fail-loud）、#8415/#8447（closed 加固域：softsign_grad 越域静默错、ApplyCenteredRMSProp 方差项非负截断缺失——后者优化器域属训练）。
- **fail-loud/崩溃域**：pytorch #5967（torch_npu cast_weight 与 ComfyUI ops 命名冲突 AttributeError，推理失败响亮）、#5958（invoke_subgraph lowering 失败）、#5980（XGLM OOM）、#5982/#5983/#5984；vllm-ascend #21286（super_kernel+static_kernel 启动 AI Core Error 271）、#21296/#21298/#21299（HIXL/fill_/SDMA 错误域，均 closed）、#21347/#21292/#21312（MTP draft RoPE/position cache 长上下文欠尺寸 → OOB 507011/EngineDead，fail-loud，修复 PR 群 120 单测+A5 回归）、#21316（EPLB Gloo NZ cast 参数 29 vs 50 错配 copy 被拒，实测 A5）；op-plugin #693（complex ones aclnnInplaceOne 失败）。
- **训练域（移交训练扫描）**：op-plugin #694（950 SDPA dropout 反向 torch.compile 下 RNG 状态丢失：v1 schema int seed/offset 无法过 AOTAutograd、meta 写死 0，反向用 seed=0 重生成掩码，310k/311k 梯度元素超差 max_abs 0.68~4.75；前向逐位一致；根因链完整、修复 !6179 原型验证 4 组合 max_abs=0——训练库范围）、#692（无图融合 BN 反向精度/GELU 模式）；tt-metal #63965 见 §2。
- **性能/契约/工具域**：vllm-ascend #21287（--async-scheduling 在 Ascend 静默无效，吞吐无差、无警告——纯性能域，closed）；pytorch #5968/#5969（TE 组合核 triton_meta/inductor_meta 缺上游公共字段，enable_fp_fusion 精度影响未量化，随 #5373 根因族留观）、#5965/#5966（黑名单摘除）；runtime #1352（pyACL 字典无符号窄化静默截断，Python 值与 ACL 实收不一致、修复 !6199 加上界检查——host 绑定域任务影响未量化）、#1350（超时档位 get/set 差 1 档，修复 !6192）；hccl #937+!3963（AlltoAllV/VC V 格式 task cache 整网卡住，**临时禁用缓存规避、根因待现场验证**——留观跟踪）、#929/#930/#931（检视转跟踪）、#936（加固需求）；ops-transformer #7284+!15563/!15564（mc2 数据地址越界批量排查修复流）与 mssanitizer #251/#252（MC2 检测缺陷、fixpipe quantPre 位宽误报修复 !458——**监测工具域，不转述为计算故障**）；ops-math !7252/!7262/!7257（Scale PR #6933 紧急回退-重提链，!7257 正文空模板无法判定）；ops-transformer #7278（sparse_flash_attention_grad pre/next_tokens 默认 INT32_MAX vs INT64_MAX 契约不齐，无失效实例）、#7282（bsa mxfp8 PV 尾块对齐 softmax 精度+PV L1 RAW，closed、正文为用例 JSON 无量化叙述）；#7273/#7297/#7259/#7260/#7280/#7281/#7247/#7248/#7249（装包交付一致性/ABI 不符族，编译/链接/段错误 fail-loud）；msprobe #680/#681（KV Guard 设计、PD kvcache 数值监控 RFC——方法域跟踪）；torchair #952~#957、hccl/hcomm/ge/runtime 其余、mindspore #93381/#93382、mindformers #2626/#2627、msmodelslim #687、sgl-kernel-npu #999/#1001 按标题归档（fail-loud/特性/perf/文档/测试域）。

## 2. GitHub 常规 7 仓 + arXiv

- 认证状态：环境无 GITHUB_TOKEN（未认证）；search 请求间隔 ≥8 秒。21 组查询（7 仓 × correctness/silent wrong output/NaN，`is:issue created:>=2026-11-02`）：**21/21 组 HTTP 200**。**tt-metal 仓库恢复**（昨日 API/web 全 404，今日 repos API 200），昨日缺口（11-08 当天新建）已由本轮 7 日窗口覆盖闭合。
- tt-metal 积压深读 9 条（#64834/#64811/#64411/#64678/#64827/#64826/#64507/#64876/#63965 全文）；#64678 为 **11-07 承诺"下轮优先"项，本轮深读转收 INF-2483**。
- 与既往判定比对（维持，无新证据不改判）：vllm#65324/#65325/#65680/#65564/#65513/#65734/#65337/#65131/#65385（=INF-2424 相关族）/ #66074（=INF-2455）/#66144（=INF-2458）/#66211（=INF-2457）/#66246、sglang#48660（=INF-2425）/#48230/#48964（=INF-2447，仍 open）/#49185+#49198（=INF-2464）、transformers#50336/#50315/#50334/#50212/#50440（=INF-2465）/#50424/#50476、TRT-LLM#21519/#21385/#21571、flashinfer#8570/#8443/#8284/#8607/#8790/#8946/#8897、triton 0/0/0；flashinfer **#9068（RoPE neox 特定 head_size）= INF-2456（11-07 收），本轮复核仍 open、0.7.3 仍复现**，沿用。

### 新增条目（GitHub 常规来源 10 条）

| 编号 | 来源 | 机制摘要 |
|---|---|---|
| INF-2480 | [tt-metal#64834](https://github.com/tenstorrent/tt-metal/issues/64834) | ttnn.pad HEIGHT_SHARDED 优化工厂准入不查输入/输出 shard grid/朝向/shard 高度一致：合法重切分 pad 被派到只做核内本地拷贝的工厂，**静默返回损坏数据**；附 BF16 2 字节栈越界读、UINT16 pad 值装错半区恒补 0、ND_SHARDED 解引用崩溃三处 |
| INF-2481 | [tt-metal#64811](https://github.com/tenstorrent/tt-metal/issues/64811) | sdpa_decode 掩码按文档版式 [1,b,nh,s] 传入时 Bmask=1、%Bmask 恒 0：批量 decode 每个 user 都用 user 0 的掩码行（Llama 3.2 Vision 正是该版式）；校验只查 dim2/3，仓内 batch_2 用例可覆盖但 CI 从未运行 |
| INF-2482 | [tt-metal#64411](https://github.com/tenstorrent/tt-metal/issues/64411) | SDPA 更精确设置反出零行：exp_approx_mode=False 在行内分数跨度大 bf16 输入上 29/32 行精确 0（RMS 96.2%）、fp32_dest_acc_en=True 单开 1/64 行零（15.2%）；无 inf/nan；main 的 ttt_compat prefill 调用方正用 False 配置 |
| INF-2483 | [tt-metal#64678](https://github.com/tenstorrent/tt-metal/issues/64678)（11-07 留观本轮深读转收） | moe_fused_swiglu FUSE_BIAS：out_interm scratch 无 push/wait，bias drain unpack 与 down matmul pack 竞争；测试 PCC 跨 run 非确定（0.9787~0.9821）被 0.97 阈值掩盖；加同步即稳定；hybrid_routed_expert_ffn 同名 kernel 无测试覆盖 |
| INF-2484 | [tt-metal#64827](https://github.com/tenstorrent/tt-metal/issues/64827) | relu/where/multiply 带输出张量时 rank-6 高度分片最后 shard 的 tile 行数算错（忽略第六维），最后一个 tile 行从不写入、保留内存旧值（-99.0 填充实测 1024/2048 错）；rank-5/4 对照精确 |
| INF-2485 | [tt-metal#64826](https://github.com/tenstorrent/tt-metal/issues/64826) | TILE 输入带整 tile 级过量填充（公开 API 可构造）时五算子（matmul×3/interleaved_to_sharded/transpose/typecast/nlp_create_qkv_heads）按逻辑形状定输出、按填充形状计工作量：6/7 情形静默错值，1D matmul 越界改写相邻 guard 张量 8192 值 |
| INF-2486 | [vllm#66437](https://github.com/vllm-project/vllm/issues/66437) | HummingLinearKernel（W4A16 最低优先）组 scale 在 warp-tile 末强制 flush：group_size 与 128 K-tile 不可交换（24/48/96/112…）时 flush 落组中间用错 scale，rel L2 0.17~1.10、M=1 NaN；can_implement 不查 group size；G=16/32/64 及 128 倍数正确 |
| INF-2487 | [vllm#66610](https://github.com/vllm-project/vllm/issues/66610) | /detokenize 不校验 token ID：词表外 ID 静默丢弃返回 200（[151936]→空串 200），同一 ID /v1/completions 400；create_detokenize 漏调 validate_token_ids，零日志 |
| INF-2470 | [transformers#50484](https://github.com/huggingface/transformers/issues/50484) | KimiLinear（KDA）缓存前缀后续写段带零 padding：hidden states 置零但卷积窗口与递归衰减仍被 padding 推进，输出与后续 token 状态改变；无 checkpoint 层级复现（left/right padding 对照），影响 padded cache 续写/分块 prefill |
| INF-2469 | （triton-ascend#3057，见 §1 表） | — |

### GitHub 留观（本轮判定）

- **tt-metal #63965**（tt-train DeepSeekMLP TP/EP 下 forward 直调本地 fused swiglu 跳过 broadcast/all-reduce，每 chip 返回部分和、shape 吻合无报错——**训练域，移交训练扫描**；tt_dit 审计族 #64103/#64113/#64150 等沿用 11-05 族留观）。
- **#64507**（convert_to_hwc C<32 padding 通道持 L1 陈旧数据实测、NaN 进 +Inf 跨调用残留——CNN 实验算子域，零权卷积实测无影响，LLM 路径未涉及）；**#64876**（softmax in-place numeric_stable 默认 false 为已知 TODO 改进请求，#28531 已 closed）；#63766/#63765（quasar 用例回归）。
- **vllm#66435**（DFlash drafter 复用目标侧 LoRA 元数据 lora_expand 越界读 → CUDA illegal memory access 杀引擎——fail-loud，根因清晰、候选修复未提交，随修复跟进）；**#66573**（Qwen3.5 embedding 不传 quant_config：compressed-tensors 路径 loud ValueError、GGUF 静默回退未复现——#54304 广域实例留观）；#66451/#66467/#65172/#65056 等 perf/RFC 域；#66246（基准探针测试基建）沿用。
- transformers#50315/#50328（closed 已判）、#50476 沿用；flashinfer#9068=INF-2456 沿用（见上）。

### arXiv

- 3 组查询（silent data corruption+LLM、bit flip+inference、KV cache+corruption）全部 HTTP 200 有效。
- **无新收**：SDC 组 TrainSDC/SCOUT/Cauchy-Schwarz/Syndrome/LLM-PRISM 已收或既往已判，"LLMs in Cloud" 为架构综述；KV 组 KVGuard=INF-2467（昨日已收），其余（Learning-to-Memorize、TRIAGE、视频生成记忆、观测论文）非故障编目对象；bit-flip 组 Rowhammer/JITterFlip/BARE-AI/辐射容错按既往口径不收，新增 BLINK（2609.06781，BN 完整性检查点做硬件故障在位检测）为通用 DNN 容错方法、非 LLM 推理域，留观不收。

## 3. 回访（近 7 日条目 + 轮巡旧源）

- **昨日 11 条（INF-2459~2467）批次/搜索快照核验**：#4619/#4620（11-09 有更新仍 open）、#5954、#8390（11-09T12:44 有活动、仍 open、评论未读）、#667、sglang#48967/#49185/#49198、transformers#50440、vllm#66144 均仍 open 无状态翻转；无撤回/驳回。
- **11-07 条目（INF-2448~2458）**：#5946（INF-2453）closed 昨日已注记；其余无状态翻转。
- **INF-2443（vllm-ascend#21164）**：修复 PR [#21348](https://github.com/vllm-project/vllm-ascend/pull/21348) 本日开出（pp_rank 从组 metadata 推导、按字段解析 @pp_rank、rank 不符/重复键 fail fast + 单测）——已 `update --add-evidence + --note`；issue 仍 open，合入未核实。
- **轮巡旧源 3 条**：flashinfer#8284（open 无评论变化，随机解码混杂解释未排除维持留观）；vllm#65385（open）；tt-metal#63940（INF-2400，open——tt-metal 恢复后首次复核）。
- **11-06/11-07 承诺执行**：tt-metal#64678 深读转收 INF-2483 ✓；msmodelslim#665（FlatQuant sigmoid）批次无更新维持留观；pytorch#5948 无更新维持留观。
- **11-08 承诺执行**：「族代表 #7239 待维护者确认或非退化输入实例」——本轮新增族证据 #7250（memcheck 双证，见 §1 留观），触发条件（维护者确认/非退化输入）未满足，维持族留观。

## 4. 编目与视图

- 新写入 **21 条**（全部 pending_audit）：**INF-2468 ~ INF-2488**，编号连续。映射：

| 编号 | 来源 | 标题 |
|---|---|---|
| INF-2468 | pytorch#5962 | repeat 复合符号等价漏判致行映射错 |
| INF-2469 | triton-ascend#3057 | stride-2 行栅格列载静默错 |
| INF-2470 | transformers#50484 | KimiLinear padding 推进卷积与递归状态 |
| INF-2471 | ops-math#4637 | IsIn RegBase 推导 double 边界翻转 |
| INF-2472 | ops-nn#8403 | ActsULQ fp16 同步缺失输出不确定 |
| INF-2473 | ops-transformer#7287 | FIA split-KV LSE 搬运与 UB 复用竞态 |
| INF-2474 | pytorch#5961 | Inductor 取模消除错化简致 load 地址错 |
| INF-2475 | pytorch#5972 | 多轴归约编译结果为正确值 48 倍 |
| INF-2476 | triton-ascend#3043 | 除法索引掩码不谓词化掩码外越界读 |
| INF-2477 | triton-ascend#3058 | 非仿射寻址时标量基址静默丢弃 |
| INF-2478 | vllm-ascend#21285 | 量化 KV cache 任意槽位索引读返回损坏数据 |
| INF-2479 | vllm-ascend#21387 | Kimi 混合精度 FLOAT gate 静默按 INT8 分配 |
| INF-2480 | tt-metal#64834 | ttnn.pad 分片重切版图静默损坏 |
| INF-2481 | tt-metal#64811 | sdpa_decode 掩码批维索引错用 user0 |
| INF-2482 | tt-metal#64411 | SDPA 精确模式输出整行静默为零 |
| INF-2483 | tt-metal#64678 | moe_fused_swiglu FUSE_BIAS 读写竞态 |
| INF-2484 | tt-metal#64827 | rank-6 分片输出尾 tile 行不写 |
| INF-2485 | tt-metal#64826 | 过度填充 TILE 输入五算子误读 |
| INF-2486 | vllm#66437 | Humming W4A16 组尺寸不整除 K 块错值 |
| INF-2487 | vllm#66610 | /detokenize 词表外 ID 静默丢弃 |
| INF-2488 | pytorch#5960 | 模板 GEMM 融合 epilogue 缺 fp32 截断 cast |

- var/cross 关系：本轮无（同机制比对均为分立；跨库查重 22 个拟收 URL 在 training_error_review `find` 全部 0 命中）。
- 回访 update：INF-2443 add-evidence（PR #21348）+ 注记，共 1 条。
- `catalog_tool.py stats`：**2488 条**（active 2467 + pending_audit 21）；`build` 完成；`validate` **ok:true**（0 errors；62 warnings 为存量词表/模式提示）。

## 5. 未覆盖范围与 API 账本

- **GitCode/AtomGit issue 评论线程未拉取**（无低成本批量入口），全部按正文判定；#8390（INF-2461）11-09 有新活动但评论内容未读，下轮回访优先。
- **GitHub 未认证 search**：21/21 组 200（tt-metal 3 组昨日 422/404 今日恢复）；正文单点补拉 16 次（tt-metal 9 + vllm 4 + transformers 1 + flashinfer 1 + #66610 二次分段）全 200；评论线程本轮未拉。
- **窗口内 805 条全部标题过目**；PR 修复流大批（ops-nn !13916-!14023、ops-transformer !15505-!15583、ops-math !7231-!7277、pytorch !49264-!49344、runtime !6164-!6200 等含大量 docs/CI/责任田）按标题 + 关键正文抽查，未逐条深读；!7257（weightQuantPreporcess Scale 转置修复）正文为空模板未判，随 !7252/!7262 回退-重提链留观。
- 本轮无任何候选因限流/配额只按索引判定；全部收录/留观判定均基于已读正文（批次缓存或 GitHub API 正文）。无限流等待（累计 0 秒）。
- op-plugin#694、tt-metal#63965 为训练域候选，本轮已读正文并留档，由训练侧扫描收录（本库不占号）。

## 6. 结论

21 条新条目入库（待审计），1 条既有条目回访补修复证据（INF-2443←PR #21348），1 条 11-07 留观转收（tt-metal#64678）。覆盖：NPU 22/22 源完整批次（805 条窗口内全过目、~37 条深读）+ GitHub 7 仓 3 关键词 21/21 组 200（**tt-metal 恢复，昨日缺口闭合**，积压深读 9 条转收 6）+ arXiv 3 组（无新）。重点新面：**量化 KV cache 索引读损坏**（vllm-ascend#21285，稀疏/检索读全坏）；**torch_npu Inductor 编译器四连**（48 倍归约、模板 GEMM epilogue cast、取模化简、父子轴等价——全部静默错值且 3/4 已有修复 PR）；**triton-ascend 地址/掩码 lowering 三连**（stride-2 列载错、标量基址丢、除法掩码不谓词化越界读）；**两处 CANN kernel 同步竞态**（FIA LSE、ActsULQ fp16，输出不确定）；**量化权重两连**（Kimi FLOAT gate 静默 INT8、Humming W4A16 组尺寸错 scale）；**KimiLinear padding 推进线性注意力状态**；tt-metal pad 分片静默损坏/sdpa 掩码批维/精确模式零行族。留观重点：防御校验族证据密度持续走高（#7250 memcheck 双证）、AlltoAllV task cache 整网卡住根因未定位、Scale 回退-重提链正文缺失。
