# LLM 推理故障每日扫描报告 — 2026-09-25

- **扫描日期**：2026-09-25（UTC）
- **扫描窗口**：GitHub `created:>=2026-09-18`（7 日完整窗口，`cutoff=$(date -u -d '7 days ago' +%F)` 实际计算；与 09-24 扫描窗口 created:>=2026-09-17 全重叠——09-24 及更早判定条目全部按既有判定跳过，本日新判定集中于 09-24 扫描执行后新创建的 issue：vLLM ≥#58675、SGLang ≥#41192、tt-metal ≥#57827 等）；arxiv 按提交时间倒序取最近 8 条/查询
- **目标仓库**：vllm-project/vllm、sgl-project/sglang、huggingface/transformers、NVIDIA/TensorRT-LLM、triton-lang/triton、tenstorrent/tt-metal、flashinfer-ai/flashinfer
- **主文档基线**：1206 条（2026-09-24 扫描后）；编号编目 1092 条 + K* 扩展 114 条

## Git 前置检查（第 3 节）

- 分支 `main`、上游 `origin/main`、`git status --porcelain -uall` 为空 ✅
- `git fetch origin main` 成功；`HEAD` = `origin/main` = `63db0dd552da02bc90842b9c7f02cbd6bb6c80be` ✅ → 开始扫描

## API 调用记录

| # | 类型 | 查询 | 结果 |
|---|------|------|------|
| 1 | GH Search | `repo:vllm-project/vllm is:issue created:>=2026-09-18 wrong OR incorrect OR corruption OR NaN` | 50 命中（单页取全） |
| 2 | GH Search | `repo:vllm-project/vllm is:issue label:bug created:>=2026-09-18` | 61 命中（单页取全） |
| 3 | GH Search | `repo:sgl-project/sglang is:issue created:>=2026-09-18 wrong OR incorrect OR corruption OR NaN` | 22 命中 |
| 4 | GH Search | `repo:huggingface/transformers is:issue created:>=2026-09-18 wrong output OR incorrect OR NaN` | 6 命中 |
| 5 | GH Search | `repo:NVIDIA/TensorRT-LLM is:issue created:>=2026-09-18 wrong OR incorrect OR corruption` | 1 命中（#19501 RFC，09-24 已判） |
| 6 | GH Search | `repo:triton-lang/triton is:issue created:>=2026-09-18 wrong OR miscompile OR NaN` | 1 命中（#11889 功能询问，09-24 已判） |
| 7 | GH Search | `repo:tenstorrent/tt-metal is:issue created:>=2026-09-18 numerical OR wrong OR pcc OR silent` | 155 命中（**两页取全 100+55**） |
| 8 | GH Search | `repo:flashinfer-ai/flashinfer is:issue created:>=2026-09-18 wrong OR incorrect OR NaN` | 6 命中（#5334/#5339/#5389/#5393/#5318 已编目或已判；#5438→7.96） |
| 9 | GH Search | `repo:vllm-project/vllm is:issue created:>=2026-09-18 quantization OR "KV cache" OR deterministic` | 103 命中（**两页取全 100+3**） |
| 10 | GH Search | `repo:vllm-project/vllm is:issue created:>=2026-09-18 silent OR "bit flip" OR "weight corruption"` | 66 命中（单页取全） |
| 11 | GH Search | `repo:sgl-project/sglang is:issue label:bug created:>=2026-09-18` | 1 命中（#40155→11.88） |
| 12–17 | arxiv | 引号短语 + AND ×6：`"silent data corruption" AND LLM` / `"bit flip" AND LLM` / `"KV cache" AND corruption` / `"soft error" AND GPU AND inference` / `"inference" AND "numerical correctness"` / `inference AND "wrong output"` | 3/6/8/8/8/5 命中，全部有效（HTTP 200、无 429） |

GitHub Search API 11 组查询全部有效、**分页完整（fetched == total 逐组核对：50/61/22/6/1/1/155/6/103/66/1——q07 与 q09 超 100 均补取第 2 页）**、无限流 403、无 Validation Failed；查询间 ≥8s 间隔。core REST 详情/评论/状态 23 次调用全部成功（`--http1.1`；起始额度 60，剩余 8 时停止调用）。arxiv 6 组全部一次成功。**候选正文优先取自 Search API 缓存（响应自带 body），仅回访与新条目状态核对使用 core REST。**

## 已编目/已判条目（本次跳过，重叠区去重）

- vLLM（09-24 及更早已判/已编目）：#58627（→6.255）、#58636（→4.26，评论见回访）、#58470（→6.256）、#58532（→7.98）、#58485（→11.87，评论见回访）、#58719（09-24 MFU/MBU 观测指标域同族）、#58034（09-22 留观）、#57661（→6.243）、#57716（→8.91，评论见回访）、#57822（→6.248，评论见回访）、#57562（→6.242）、#57578（8.84 补充）、#57486（7.95 关联）、#57490（→10.72）、#57493（→4.24）、#57614（→8.90）、#57719（→6.245）、#57721/#57740/#57794/#57847/#57853/#58080/#58100/#58138/#58211/#58214/#58242/#58267/#58273/#58303/#58362/#58423/#58520/#58565/#58616（均已判）；其余 09-18..09-24 判定项
- SGLang：#41038（→6.256）、#40966（→12.43）、#40903（→7.97）、#40921（→5.45，评论见回访）、#40835（→12.42）、#40735（→5.42）、#40730（→5.43）、#40127（→12.40）、#40144（→11.82）、#40156（→11.81）、#40285（→12.41）、#40459（→6.249）、#40564/#40623/#40817/#40417/#40305/#40370/#40504（已判）
- transformers：#49051（→10.86）、#49066（→10.85）、#49036（留观）、#48947（→6.246）、#48954/#49026（CI triage）
- tt-metal（09-24 判定项全跳过）：#57633/#57634/#57643/#57714/#57732（→1.248/1.249/1.250/1.251/1.252）、#57461（→1.247）、#57701（→8.93）、#57408（→1.242）等；静态审计三批 meta-issue（#57360–#57367/#57474–#57510/#57674–#57694）整批留观维持
- flashinfer：#5438（→7.96）、#5334（→1.230）、#5393（留观）、#5389（09-21 已判 fail-closed）、#5318（发布工程）；triton/TRT-LLM 窗口无新增未判项

## 🆕 新发现（本次合并，共 12 条）

### A. 全新子签名（5 条）

1. **[新签名] vLLM #58675 — `--hf-overrides` 传 `rope_scaling` 时 transformers v5 整体替换 `rope_parameters` 丢 `rope_theta`，`get_rope` 静默回退 base 10000：重述 checkpoint 自身的 scaling（应为 no-op）使 Llama-3.2-3B GSM8K 379→279、mean NLL 1.128→3.660（与显式 base=10000 逐位一致）** — top300 模型普查 36%（64/180 适用路径）静默换 base（gpt-oss-20b/120b 150000→10000、Qwen3-30B-A3B 1e6→1e4 等）；config.json 通道无恙，分歧纯由注入通道决定；零警告零日志。→ 归入第 9 类 **9.29**，Cov var:9.18（transformers v5 rope_parameters 迁移族：save_pretrained 污染 vs 运行时注入丢字段），documented。
2. **[新签名] SGLang #41192（+ vLLM-Omni 对侧 #8135）— Qwen-Image-2.1 diffusion `--tp-size 2` 输出致密 chroma speckle、单卡干净；两个独立引擎的 TP 实现呈相同损坏，NCCL 专项清白（bf16 1 ULP）、rank 交换不变** — 假设：单流 DiT 的 AdaLN 调制/per-head QK-norm 状态被按普通 FFN 语义切分求和。→ 归入第 5 类 **5.46**，Cov NEW（DiT 不可切分状态按 FFN 切分子签名；跨双引擎交叉验证），documented。
3. **[新签名] tt-metal #57873 — Gemma4-31B（vLLM TT plugin、P150x8、chunked prefill 2048）同一步共享多个 prefill 请求时错答：12 请求错 6/5（普通/contract 类），单请求 split-prefill 与长 prefill 全对** — 日志实证 `batch=2` 混合 2807+1289 token 的 ragged chunk 边界；无 `chunk_start` 告警排除 #57531 链；两 run 错答有逐字重合（非 speculation）。→ 归入第 6 类 **6.257**，Cov NEW（共享调度 chunk ragged 偏移跨请求 KV 错配），documented。
4. **[新签名] vLLM #58729 — GLM-5.3-MXFP4 + 8×GB200（聚合 DP8+EP）GSM8K 塌到有效 0：1,319 请求 1,178 null content、2,638 槽位 2,356 空串、仅 8 个 `####` 答案** — nightly 2026-09-14 无精度下降（版本二分）；GLM5.2-mxfp4-fp8 同病；DP 聚合/分层 KV/MXFP4 位面未分离。→ 归入第 7 类 **7.99**，Cov NEW（MXFP4 MoE 服务整链精度坍塌），documented，族 ambiguous（原因未定位）。
5. **[新签名] tt-metal #57889 — SFPU `v_if` 序列中非末 block 的 `v_and` 编译输出反逻辑：汇编自证 `SFPCOMPC` 取反插入位置错误（block 末尾而非开头），后续 `SFPLOADI`/`SFPENCC` 在已取反状态上执行** — 4 行最小重现 + 反汇编逐行标注。→ 归入第 1 类 **1.253**，Cov NEW（SFPU 谓词状态机 lowering 时序子签名），documented。

### B. 已知类别新变体（7 条）

6. **[已知类别] tt-metal #57861 — 分布式 layernorm `post_all_gather` 默认 bf16 统计按 E[x²]−E[x]² 计算方差：mean≫std 激活（post-residual 常态）灾难性相消，var 塌 0（rstd≈1e6）或为负（NaN 行）；σ≲μ/23 完全塌缩** — `fp32_dest_acc_en=True` 时 stats 输入仍 bf16 舍入、同输入 var=−576→NaN（两种配置各败其道）；RMSNorm 分布式版构造上免疫；#54016 Welford bounty 文本自认现象。→ 归入第 1 类 **1.254**，Cov var:1.231（分布式归约统计管线族），documented（源码级+内核算术逐级模拟）。
7. **[已知类别] tt-metal #57852 — `ttnn.requantize` 任一 scale/zero_point 拼写为 Tensor 即走 composite 回退，dequantize 中间量默认 bf16：int32 域 94.2% 元素偏、max 80 LSB；全标量路径 fp32 精确** — torch 导出量化模型恰以 Tensor 携带 scale——真实导入路径必中；op 自带 golden 中间量 fp32，属对注册参考的背离。→ 归入第 1 类 **1.255**，Cov var:7.83（bf16 中转双舍入族），documented。
8. **[已知类别] tt-metal #57869 — `ttnn.moreh_norm` 无 max 重缩放：`|x|^p` 落入 subnormal 被 DAZ/FTZ 清零，范数返回精确 0（torch fp32 返回正常数）** — p=2、[1e-20,3e-20]→0；p=16、[0.002,0.003]→0；与 #51861 hypot、#52614 var/std 同类互不覆盖。→ 归入第 1 类 **1.256**，Cov var:1.183（subnormal flush 族），documented。
9. **[已知类别] tt-metal #57847 + #57848 — `ttnn.expm1(-0.0)`/`ttnn.log1p(-0.0)` 返 +0.0（IEEE/torch 为 −0.0）：expm1 重建门 `j!=0` 跳过、log1p 首句 `a+1.0` 吞符号** — WH/BH 双副本同行号；±0.0 族第四、五例（继 #57538、#57714 后）。→ 合并归入第 1 类 **1.257**，Cov var:1.251，documented（IEEE replay）。
10. **[已知类别] tt-metal #57867 — `ttnn.addcmul_bw` 先乘 `grad×tensorN` 再乘标量 `value`：中间超 FLT_MAX 变 inf 而真值有限；姊妹 `addcdiv_bw` 与前向 LLK（#53282）均已修、唯 backward 漏修** — torch.autograd 对照判决。→ 归入第 1 类 **1.258**，Cov var:1.171（反向中间量域损失族），documented。
11. **[已知类别] SGLang #40155 — spec-v2 七 worker 的 `forward_batch_generation` 签名分歧：调度器恒传 `pp_proxy_tensors`，dspark/multi_layer_eagle/frozen_kv_mtp/uno 四 worker 缺参首步即 TypeError；`BaseSpecWorker` ABC 不声明该方法，接口漂移零强制** — 含七 worker 审计矩阵；frozen_kv_mtp/uno 为未报告新点位。→ 归入第 11 类 **11.88**，Cov NEW（接口契约缺失致实现族静默分歧；fail-loud 但结构模式典型），verified（源码定位因果闭合）。
12. **[已知类别] vLLM #58807 — MTP predictor 从 target 继承 quant config：compressed-tensors pack-quantized target + 密集 BF16 MTP 分片（量化器未写 ignore）时按 packed 名找参数、加载期 `ValueError: no module named 'fc.weight'`** — 文件内已有 modelopt_fp4 与 GPTQ dynamic 两条同族 workaround、第三种格式缺口。→ 归入第 13 类 **13.40**，Cov var:12.41（dense-draft-on-quantized-target 族），documented。

### C. 评估后不收录（记录备查）

- **vLLM #58727**（Anthropic /v1/messages 内联 system 恒走 merge 路径：`--chat-template` CLI 值 None 短路探测、模型自带模板者也被合并提升——中段 system 语义位置改变 + 前缀缓存全miss）：语义位置+性能复合面，语义变化属模板契约域（#58647 拆分项 1/3），无静默错值证据。留观。
- **vLLM #58692**（Dynamic SD + MTP V2 `SpeculatorCudaGraphManager._init_candidates` ZeroDivisionError）：fail-loud 启动 crash。不收录。
- **vLLM #58726**（GB10 safetensors mmap 逐张量 H2D 慢）：性能。不收录。
- **vLLM #58742/#58719**（MFU/MBU 把 spec 步算 prefill / GPTQ·AWQ·gpt-oss 等丢 attention+FFN）：观测指标域。不收录。
- **vLLM #58804**（Tiered Offloading 性能）：性能。不收录。
- **vLLM #58751/#58774**（kernel autotuning RFC / span pooling RFC）：RFC。不收录。
- **SGLang #40155/#41192**：已编目（11.88/5.46）。
- **tt-metal #57827**（MiniMax-M3 pipeline-prefill 性能追踪矩阵）：性能追踪。不收录。
- **tt-metal #57835**（grid_sample nearest 半整数坐标 tie-break 差一像素、golden 用 round-half-to-even）：视觉重采样 tie-break 契约、单像素级；非推理数值链路。留观。
- **tt-metal #57853**（Gemma4 dFlash 类 warmup 捕获 prefill trace 后从不回放）：纯浪费 warmup 工作/trace 区资源，数值面无影响。不收录。
- **tt-metal #57856**（LLK fast_tilize unit_dim 非法值静默 no-op——FWASSERT 为 no-op 宏）：当前无活调用路径（报告者自证），防御缺失。留观。
- **tt-metal #57899**（GLM 5.2/5.3 MTP 多轮 token 分片缺口描述）：设计说明无失败证据。不收录。
- **tt-metal #57902**（Quasar eltwise binary tiny-tile 数学与 unpack 工作项）：PR 工作项清单。不收录。
- **tt-metal #57920**（SDPA windowed 模式请求 is_causal=True 支持）：功能请求。不收录。
- **tt-metal #57923**（MiniMax-H3 ring SDPA chunk 表 keyed 在未 bucketed 长度上、bucketed 全落 heuristic）：性能表失配、报告者自证 mostly harmless。不收录。
- **tt-metal #57929**（Gemma4 prefill 开 1D line fabric 使 Ring 集体静默降级 Linear）：纯性能（-16%）。不收录。
- **tt-metal #57948**（dispatch/combine_fabric2d 背靠背 launch hazard——combine 的 arrival counter 清零竞态 ring 级 hang + 转发区覆写 wrong bytes）：hang 域 + 数据损坏面真实（24 overlapped launch byte-exact、半闲 draw 1-3 launch 即腐），但当前模型路径被"现实流量全流 + combine 每-launch barrier"双重掩盖、combine 尚未投产。留观（转 MoE prefill 路径后升级）。
- **tt-metal #57955**（combine_fabric2d worker 放置 fallback 忽略 NOC 代价 + 疑似 link ordinal 当 link index）：无功能失败观察（报告者自证测试全过）。留观。
- **tt-metal #57950**（Quasar packer remapper teardown 早于 tensix_sync：craq-sim `posted=64 acked=65`、A0 上仅 sticky counter bit）：模拟器层证据、硅上损害未证。留观。
- **tt-metal #57954**（MatmulMultiCoreReuse `per_core_M < Mt` 且块数超核时输出 tile 永不写——batch 64 时 50% 输出为 stale buffer）：与 1.119（#54798 过订阅静默丢块）**同 op 同族第二形态**（1.119 为块数>核数、本条 per_core_M<Mt 同样致块数>核数触发面重合）；不另占编号，作 1.119 语境补充跟踪。
- **tt-metal #57965**（分布式 RMSNorm/LayerNorm FLOAT32 stats 列 16/48 未定义被 post_all_gather 求和、pre+post 偏差至 0.117）：#57861（→1.254）相邻数值面、同内核族（layer norm 均值 100 已含 inf 相消）；待设备级数据后并入 1.254 关联，不另占编号。
- **flashinfer #5318**（v0.7.0 分支漏 cherry-pick #5097 sparse-MLA L2 guard-rail）：发布工程。不收录。
- **arxiv**：6 组查询全部有效；命中全部已编目或已判（SProbe/TrainSDC/SCOUT/LLM-PRISM/2502.12340/JITterFlip/GBFA/ROBBIN/Decision-Level/UEP/RangeGuard/2604.17249/2606.28958/2609.17983/2609.16742/Nexus/Moment-KV/2606.07571 等）；**2608.23663**（Confidently Wrong, Silently So，2026-08-24，v2——端侧部署模型可靠性审计：task-asymmetric miscalibration、false-premise confabulation 69%、自信正确/错误输出表面不可区分 AUROC 0.55、黑盒 consistency wrapper 恢复）为能力/对齐审计论文**不收录**（与第 15 类"检测器已验证信号"语境关联：其"15 个用户可见特征不可分"结论与本项目"post-forward 结构性盲点"互证，已在此记录）；**2609.25881**（虚拟试穿非对称信息动力学）非 LLM 域。无新编目论文。

## 来源回访（近 7 日新增条目引用来源 + 额度轮巡旧来源）

本日回访 17 个来源（完整评论 6 个、状态+评论数快照 11 个）：

- **vLLM #58485（11.87）**：5 条评论（09-24/09-25）——hclsys 源码级独立复核"16 处 force_index 全为单元素"且"比报告更强"；njhill 指示 MRV2；hclsys 实测 V2 干净（GB10 驱动 apply_thinking_budget + 3-token end + 3-position spec 窗）、指出该 repro config 默认 MRV2、V1 仅在显式 `VLLM_USE_V2_MODEL_RUNNER=0` 或 V1-only spec 特性（ngram/draft_model/medusa）下可达；amittell 询问 V1 修复政策；hclsys 更正致谢（repro config 归属 amittell）。**11.87 判定与触发面表述已含此边界，无需修改**。
- **vLLM #58636（4.26）**：2 条评论——hclsys 在 torch **2.12.1** 复现 `TORCHINDUCTOR_DETERMINISTIC=1` 失效（原条目归因 pytorch#198563 的 2.13 范围扩大至 2.12.1）；Bizuayeu 确认（2.12.1+cpu repro：首编译帧后 deterministic 翻 False、强制入口保持 True）并补充到 PyTorch issue。**4.26 的 workaround 失效范围比条目记载更宽**（非 2.13-only）——已在判定层面知悉，条目不另改（版本细节属 PyTorch 上游跟踪）。
- **vLLM #57822（6.248）**：4 条评论（09-21..09-24）——报告者 09-23 大更新：环境表 torch 版本系误读 v0.24 容器；**v0.29.0 同机同 checkpoint 同 flags 生产 2 天 9 小时 80,000 请求 0 崩溃 0 Xid**（对照 v0.28 3 小时 3 崩 580 Xid；并行 v0.24 实例同期 78,000 请求 0 错）；v0.28 上按三次崩溃实时记录的 dump_input 形状合成重放 29,404 请求**无法复现**（触发面窄于记录形状）；关联 #55571。gau-nernst 09-24 问"可假定 0.29 已修？"。**→ 6.248 来源行补勘误性补记**（判定不变：v0.28 版本窗口回归证据加强；新增 v0.29 稳定与形状重放不可复现两个边界事实）。
- **vLLM #58627（6.255）**：状态 open、评论 2（昨日已读：报告者撤回服务影响演示）。无变化。
- **tt-metal #57461（1.247）**：评论 1（09-23 VirdhatchaniKN 相邻三缺陷——已并入昨日条目）。#57463 修复 PR 状态未单独核（core 额度限）。
- **tt-metal #57732（1.252）**：评论 1（09-25 Repo Assist bot）：**已开修复 PR**——首 DRAM-bank 读钳制到 `per_core_N`（与循环内后续读取同款 clamp、decode 路径工厂已有等价守卫）。修复在途、issue open。条目不需改（未合并前维持）。
- **tt-metal #57742（09-24 留观：#57180 校准余波）**：3 条评论（09-24 mtairum）——修复 PR 三件（SDXL #57721 owner、BGE-M3 #57743、Qwen2.5-VL #57744）；**CI bisect 双重确认**：SDXL VAE 与 BGE-M3 单元 job 独立 bisect 到同一 commit（父 97326f34706 绿、90272bb3ffe 红），#57180 是窗口内唯一移动这些测试的变更。修复方向获实证支持；内核侧 follow-up（streaming 近似 #56710 等）维持留观。
- **vLLM #57716（8.91）**：3 条评论——标签更正（rocm→cuda，hongxiayang 确认）；无机制新证据。open 不变。
- **SGLang #40921（5.45）**：2 条评论（09-23）——PansaLegrand 询问是否有维护者调查（#32272/#34917 是否覆盖双侧 trtllm_mha）；akhilg-nv（NVIDIA）确认"无人调查，#32272 只做 perf 不假设 correctness"。**5.45 根因待维护者响应状态不变**。
- **vLLM #58031（5.44）**：状态 open、评论 9（昨日已全读）。无新变化。
- **vLLM #58267（09-23 留观）**：状态 open、评论 2（昨日已读：main 双半缺陷确认 + 修复 PR #58268 在途）。无变化。
- **vLLM #58616（09-24 留观）**：状态 open、评论 2。快照无实质变化。
- **vLLM #58470（6.256）/#58532（7.98）/#57661（6.243）/#58362（6.254）/flashinfer #5438（7.96）**：状态 open、评论 0/0/0/0/0（#58470/#58532/#57661/#58362/#5438 均无新评论）。无状态翻转。
- 未检查完说明：近 7 日新增条目引用来源约 85 个（09-19..09-24 合并 83 条 + 勘误），本日按 core 额度（60，实际消耗 23 次详情/评论+1 次 rate_limit 探测后于剩余 8 时停止）回访 17 个高价值来源（昨日 12 条新合并的主来源 + 6.248 大更新 + 5.45 维护者响应 + 1.252 修复 PR 核实）；其余约 68 个经本日 search 快照确认无 closed/reopened 翻转（q01–q11 覆盖窗口内全部活跃 issue），评论级深查缺口滚动至后续扫描。

## 合并动作摘要

- 主文档新增 **12 条**（5 个新子签名 + 7 个已知类别变体）：1.253–1.258、5.46、6.257、7.99、9.29、11.88、13.40
- 总条数 1206 → **1218**；编号编目 1092 → **1104**（K* 扩展 114 不变）
- 分类分布：第 1 类 251→257、第 5 类 45→46、第 6 类 256→257、第 7 类 98→99、第 9 类 28→29、第 11 类 86→87、第 13 类 39→40
- 轴分布更新（按解析器实测）：infra_off 1039→1049、ambiguous 92→94、hw_specific 304→311、other_stack 232→234（其中 ⚠ 25 不变）、config 258→261（其中 ⚠ 6 不变）、documented 690→701、verified 509→510、NEW 883→888、var:x 320→327
- 条目更正：6.248（vLLM #57822）回访补记（torch 版本误读勘误；v0.29.0 80k 请求 0 崩 0 Xid 负对照；v0.28 记录形状重放不可复现；判定不变）
- 观察：**9.29 是首条"照厂商文档操作即中招"的 RoPE 故障**（重述 checkpoint 自身 rope_scaling 应为 no-op 却静默换 base 10000；top300 模型 36% 受影响；GSM8K 379→279 且输出保持流畅——静默面极大）；**5.46 是 diffusion 输出域首个跨双引擎交叉验证的 TP 数值损坏**（SGLang 主路径 + vLLM-Omni 独立实现同构、NCCL 清白五重排除）；1.254 把"教科书统计公式进窄尾数域"落到分布式归一化默认路径（两种精度开关各败其道、RMSNorm 构造免疫的对照干净）；1.255 首次把触发面定到**参数拼写**（scalar vs Tensor——最小测试恒绿、torch 导出必中）；11.88/13.40 补全"fail-loud 但结构模式典型"的接口/配置契约缺口（ABC 零强制、量化配置继承第三种格式缺口）
- 未覆盖范围：q07（tt-metal 155 命中）与 q09（vLLM 103 命中）中 09-18..09-23 创建、昨日已判定条目未逐一重读正文（按昨日判定跳过）；tt-metal 静态审计三批 meta-issue 维持整批留观未逐项展开；core REST 额度耗尽前未回访的约 68 个来源仅经 search 快照确认无状态翻转（无评论级深查）；#57463（1.247 修复 PR）合并状态未核
- 导出与校验：`python3 export_faults.py`（1218 行 CSV/XLSX）→ `python3 verify_catalog.py` **PASS 1218 条：正文汇总、CSV、XLSX、字段边界一致**
