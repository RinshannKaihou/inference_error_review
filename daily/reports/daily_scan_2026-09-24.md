# LLM 推理故障每日扫描报告 — 2026-09-24

- **扫描日期**：2026-09-24（UTC）
- **扫描窗口**：GitHub `created:>=2026-09-17`（7 日完整窗口，`cutoff=$(date -u -d '7 days ago' +%F)` 实际计算；与 09-23 扫描窗口 created:>=2026-09-21 重叠三日，已逐条去重——重叠区间条目昨日已判定者全部按既有判定跳过）；arxiv 按提交时间倒序取最近 8 条/查询
- **目标仓库**：vllm-project/vllm、sgl-project/sglang、huggingface/transformers、NVIDIA/TensorRT-LLM、triton-lang/triton、tenstorrent/tt-metal、flashinfer-ai/flashinfer
- **主文档基线**：1194 条（2026-09-24 勘误去重后）；编号编目 1080 条 + K* 扩展 114 条

## Git 前置检查（第 3 节）

- 分支 `main`、上游 `origin/main`、`git status --porcelain -uall` 为空 ✅
- `git fetch origin main` 成功；`HEAD` = `origin/main` = `c3c5d55c8cc7adb294712afce537f4b83c98b858` ✅ → 开始扫描

## API 调用记录

| # | 类型 | 查询 | 结果 |
|---|------|------|------|
| 1 | GH Search | `repo:vllm-project/vllm is:issue created:>=2026-09-17 wrong OR incorrect OR corruption OR NaN` | 50 命中（全取） |
| 2 | GH Search | `repo:vllm-project/vllm is:issue label:bug created:>=2026-09-17` | 67 命中（per_page=100 单页取全） |
| 3 | GH Search | `repo:sgl-project/sglang is:issue created:>=2026-09-17 wrong OR incorrect OR corruption OR NaN` | 22 命中 |
| 4 | GH Search | `repo:huggingface/transformers is:issue created:>=2026-09-17 wrong output OR incorrect OR NaN` | 7 命中 |
| 5 | GH Search | `repo:NVIDIA/TensorRT-LLM is:issue created:>=2026-09-17 wrong OR incorrect OR corruption` | 2 命中（均已判：#19319 09-17 已判、#19501 RFC） |
| 6 | GH Search | `repo:triton-lang/triton is:issue created:>=2026-09-17 wrong OR miscompile OR NaN` | 2 命中（#11836 已编目 1.229、#11889 功能询问） |
| 7 | GH Search | `repo:tenstorrent/tt-metal is:issue created:>=2026-09-17 numerical OR wrong OR pcc OR silent` | 147 命中（per_page=100 两页取全） |
| 8 | GH Search | `repo:flashinfer-ai/flashinfer is:issue created:>=2026-09-17 wrong OR incorrect OR NaN` | 6 命中（#5438 已编目 7.96、#5393 已留观、其余 CI/feature） |
| 9 | GH Search | `repo:vllm-project/vllm is:issue created:>=2026-09-17 quantization OR "KV cache" OR deterministic` | 106 命中（两页取全） |
| 10 | GH Search | `repo:vllm-project/vllm is:issue created:>=2026-09-17 silent OR "bit flip" OR "weight corruption"` | 63 命中 |
| 11–15 | arxiv（首版） | 无引号多词查询 ×5 | **判失效**（查询串空格被 API 拆解为 OR，返回天文/物理等不相关最新提交——与 09-17 记录的失效模式相同） |
| 11–15 | arxiv（重发） | 引号短语 + AND：`"silent data corruption" AND LLM` / `"bit flip" AND "LLM"` / `"KV cache" AND corruption` / `"inference" AND "numerical correctness"` / `"soft error" AND GPU AND inference` | 6/8/8/7/3 命中，全部有效 |

GitHub Search API 10 组查询全部有效、**分页完整（fetched == total 逐组核对：50/67/22/7/2/2/147/6/106/63）**、无限流 403、无 Validation Failed；查询间 ≥8s 间隔。**执行过程记录**：扫描脚本首版 `run_q` 传参错位（name 被当作 per_page）致 q01–q08 报 ValueError 后未发请求，修复后全部重跑成功——首版失败不产生任何 API 调用，不影响覆盖。core REST 详情+评论 24 次调用全部成功（`--http1.1`，剩余额度 36）。arxiv 首版 5 组查询语法失效（已列明），同日以修正语法重发 5 组全部成功、无 429。

## 已编目/已判条目（本次跳过，重叠区去重）

- vLLM（09-23 已判/已编目）：#58138（→10.83，评论见回访）、#58105（→10.82，评论见回访）、#58080（→11.85）、#58362（→6.254）、#58394（→6.253）、#58391（→8.92）、#58390（→11.86）、#58422（留观，评论见回访）、#58421/#58406/#58384/#58353/#58303/#58273/#58267（留观——评论见回访，修复 PR #58268 在途）/#58241/#58228/#58222/#58280/#58441/#58290/#58324/#58233/#58236/#58246/#58211/#58134/#58135/#58127（→10.84 补充）/其余 09-22/09-21 判定项
- SGLang：#40921（→5.45）、#40903（→7.97）、#40966（→12.43）、#40835（→12.42）、#40735（→5.42）、#40730（→5.43，回访确认仍 open 无新评论）、#40926/#40817（09-23 已判）
- transformers：#49051（→10.86）、#49066（→10.85）、#49036（留观）、#49026（CI triage，不收录）
- tt-metal：#57608（→1.245）、#57538（→1.246）、#57408（→1.242）、#57413（→1.244）、#57410（→1.241）、#57382（→1.243）、#57461（昨日限额留观→**本次补录 1.247**）、#57394/#57370/#57403（09-22 已判）、静态审计第一/二批 #57352–#57367/#57474–#57510（整批留观维持）
- flashinfer：#5438（→7.96，回访确认仍 open 无新评论）、#5393（留观）；TensorRT-LLM/triton 窗口无新增未判项

## 🆕 新发现（本次合并，共 12 条）

### A. 全新子签名（6 条）

1. **[新签名] tt-metal #57461 — `moe_expert_token_remap` writer 组计数器双重推进：每组丢一页元数据 + `reduce_idx` 越界覆写下一 core 页** — wormhole_b0 实测 906/4096–1346/4096 错元素；全部访问在已分配张量内、零 assert；`output_mapping` 路径不受影响。→ 归入第 1 类 **1.247**，Cov NEW。**（昨日因 core 限额留观，本次评论核实修复 #57463 open 未合并后补录；评论另载三个相邻缺陷：部分尾组丢失、`reduction_size` 无主机校验、真实消费者测试被 #28376 跳过）**
2. **[新签名] tt-metal #57732（+同 commit 独立报告 #57630）— 2D mcast matmul DRAM width-sharded in1 首读未钳制：`per_core_N < ceil(N/banks)` 时宽网格静默 garbage** — 预测规则 22/22+288/288 配置全中；修复后 PCC≥0.99988、212 既有测试不变、Llama-3.1-8B logits bit-identical。→ 归入第 1 类 **1.252**，Cov NEW（错误同时把 tt_transformers P150 prefill 锁死 ≤8 列网格，修复后 13 列快 14–21%）。
3. **[新签名] vLLM #58627 — DSV4.1 sparse MQA：DeepSelect NaN 行只写首槽哨兵，其余槽保留陈旧 col_indices 被当真实位置重映射——静默陈旧 top-k 选择** — #58215 修 OOB 后暴露的后续缺陷；报告者评论已撤回"演示的服务影响"（29 空回复 0 重复尾巴），按机制+源码定位收录。→ 归入第 6 类 **6.255**，Cov NEW，documented。
4. **[新签名] vLLM #58470 + SGLang #41038 — PD 分离 MLA target × GQA draft（DFlash）非对称 TP 下 draft KV 误传输** — vLLM：模型级 `use_mla` 复制假设套到 head-sharded draft 区域（正确几何被 assert 拒、错几何被接受、7/8 rank 读 draft head 0）；SGLang：P 侧 item length 被当 D 侧 stride（B300 实测 28/256 传输失败）。同目标双栈同机制合并编目（判例同 11.73）。→ 归入第 6 类 **6.256**，Cov NEW，verified。
5. **[新签名] vLLM #58532 — Triton fused MoE：`per_act_token_quant` 单标志同时选激活/权重 scale 索引布局，Quark 静态 INT8（per-tensor 激活 × per-channel 权重）组合按错误布局读权重 scale——扰动专家 3 全部 W13 scale ×4 输出无变化** — #57092 修的是相反组合、本组合测试仍跳过。→ 归入第 7 类 **7.98**，Cov NEW，documented（组件级证据）。
6. **[新签名] tt-metal #57701 — Gemma4 精确窗口 bounded sliding ring 上 speculative verify 覆写仍在因果窗内的位置：>1019 token 上下文自首 token 静默损坏，服务路径只打一条启动 warning** — ring 内代码注释自证（"K=5 corrupts from the first token"）；CI 与 tt-inference-server 四个生产配置全部落在自动开启区。→ 归入第 8 类 **8.93**，Cov NEW，verified。

### B. 已知类别新变体（6 条）

7. **[已知类别] tt-metal #57633 — `ttnn.softmax` float32 默认用近似 exp：ones 行恒 −2.9%、N(0,1) 4096 宽逐元素最大 7.45e-2、行和 0.96–0.97；`numeric_stable`/HiFi4/fp32_dest 一位不改，仅 `math_approx_mode=False` 有效** — 与 `ttnn.exp` 文档默认相反。→ 归入第 1 类 **1.248**，Cov var:1.137（融合算子默认近似无逃生族），verified（BH 实测）。
8. **[已知类别] tt-metal #57634 — `ttnn.prod` fp32 按 ~10-bit 形成乘积：4096 元素 99% 误差、72,192 值乘积返 3.89e-31（真值 98.9）** — SrcA TF32 unpack + DEST→SrcA 折返双机制；`(1+2⁻¹⁰)` 因子整行消失是判决对照。→ 归入第 1 类 **1.249**，Cov var:4.2（TF32 10-bit 截断族在 tt-metal 维度首例），verified（BH 实测）。
9. **[已知类别] tt-metal #57643 — typecast uint32→fp32 双重舍入：bit31 置位输入 16.47% 返 1 ULP 邻值（全域 2³¹ 穷举）** — golden 位精确比较下现有测试必绕开上半域。→ 归入第 1 类 **1.250**，Cov var:1.190（typecast 整数域舍入族），documented（无设备穷举 replay + Repo Assist 根因确认）。
10. **[已知类别] tt-metal #57714 — fmod/remainder 精确零丢被除数符号（+0.0 vs torch −0.0）** — 两个 fix-up 循环全在零处失效；`+0.0 == −0.0` 使既有测试结构性失明。→ 归入第 1 类 **1.251**，Cov var:1.246（±0.0 位域语义族），documented（IEEE replay）。
11. **[已知类别] vLLM #58636 — GLM-5.x 复制 indexer key norm 逐 rank Inductor autotune：XBLOCK=1 归约顺序不同 → TP rank 间 top-512 候选集分歧 → temp=0 greedy 随启动分叉（16 次启动三态、钉配置干预闭合因果）** — `TORCHINDUCTOR_DETERMINISTIC=1` 因 pytorch#198563 失效。→ 归入第 4 类 **4.26**，Cov var:4.12（near-tie 放大族，根因新面：per-rank autotune），documented。
12. **[已知类别] vLLM #58485 — V1 thinking budget × spec decoding × 多 token end 序列：draft 匹配即放行但被拒后游标不回退，end token 被跳过永不发出（17/32 损坏、V2 全净、#52677 全净 Fisher p=8.2e-7）** — 全部 16 处 force_index 为"至多一个位置"。→ 归入第 11 类 **11.87**，Cov var:11.86（thinking-budget 强制结束状态机姊妹半边），documented。

### C. 评估后不收录（记录备查）

- **vLLM #58616**（L4 单卡权重加载后确定性 segfault，仅 `VLLM_TRACE_FUNCTION=1` 回避——提示竞态但为 fail-loud crash，无静默错值证据）：不收录。
- **vLLM #58565**（CPU bge-m3 首个 /v1/embeddings 请求 EngineCore "RuntimeError: cancelled" 永久死亡）：fail-loud 崩溃。不收录。
- **vLLM #58492**（verbose_json 对 EOS-only Whisper chunk IndexError）：fail-loud。不收录。
- **vLLM #58575**（ROCm v0.30.0 漏 cherry-pick #57252 致 GLM-5.3 无法启动）：发布工程 fail-loud。不收录。
- **vLLM #58599**（MiniMax fused allreduce+RMSNorm Lamport workspace 在 cuda-bindings 13.4 AttributeError 被 try/except 吞、静默回退 eager——纯性能回退，数值正确；`cudaIpcMemHandle_t.reserved` 被移除）：性能域。不收录。
- **vLLM #58624/#58324/#58233**（MoE decode 15% 回归 / 实时会话 O(n) / 并发性能求助）：性能域。不收录。
- **vLLM #58597**（MFU/MBU 把线性注意力层计为全注意力、报出超硬件峰值的带宽）：观测指标域。不收录。
- **vLLM #58523**（显式 off-grid max_cudagraph_capture_size 被静默截断、最大 decode batch 走 eager——已 closed，性能面）：性能域。不收录。
- **vLLM #58388**（XPU MRV2 大首请求 eager all-reduce 后引擎死）：fail-loud。不收录。
- **vLLM #58520/#58537/#58562/#58329/#58487/#58580/#58544/#58263/#58240**（Mooncake Store 发布粒度 / KV Context Editing / Rust connector metrics / KVPP / FP8 KV 校准 / Gemma4 multimodal / Helm probe / batch-invariant 绑定 / KV offload 共享——全部 RFC/Feature）：不占编号。
- **tt-metal #57800**（KDA PCC drop 调查桩，无根因数据）：留观。
- **tt-metal #57742**（#57180 SDPA accurate exponential 修复的正确校准余波：SDXL VAE PCC 0.9289<0.93、BGE-M3、Qwen2.5-VL 单 ulp——修复 PR #57721/#57743/#57744 已出；内核侧 follow-up：streaming kernel 仍硬编码近似 #56710、bf16 截断 rescale 尺度不一致、init hoisting）：校准/阈值议题留观，修复本身正确。
- **tt-metal #57695**（Galaxy GCB 重建依赖未验证的 trace 捕获地址——当前因分配器确定性大概率正确，契约加固议题）：留观。
- **tt-metal #57724**（nlp_concat_heads_decode varargs 越读 → watcher assert 停机）：fail-loud hang。不收录。
- **tt-metal #57752**（tt-train 优化器状态非 per-parameter）：训练域。不收录。
- **tt-metal #57733/#57734/#57626/#57627/#57629/#57722**（conv2d 权重预处理重跑 / l1_small_size=0 诊断不点名 / minimal_matmul 形状效率 / fuse_swiglu / rms_norm 带宽 / head-major remap——性能与易用性）：不收录。
- **tt-metal #57652/#57698/#57760/#57771/#57750**（CI/阈值/迁移运维）：不收录。
- **tt-metal 静态审计第三批 #57674–#57694**（eltwise/normalization/ccl/deepseek/quasar/hw/impl/fabric/llrt/api 等约 11 份 meta-issue，延续 #57360–#57367 与 #57474–#57510 判例）：静态审计产物无逐项运行时证据，**整批留观**，待单项缺陷获得运行时证据后逐条编目。
- **arxiv**：首版 5 组无引号查询判失效后重发；有效命中全部已编目或已判（SProbe/TrainSDC/SCOUT/LLM-PRISM/2502.12340/2604.00726/JITterFlip/GBFA/ROBBIN/Decision-Level/UEP/RangeGuard/2604.17249/2606.28958/2609.17983/Nexus/Moment-KV/2606.07571 等）；**2609.16742**（Carry-Through Checksum，CNN 边缘推理轻量故障检测，2026-09-15）为非 LLM 方法论文不收录。无新编目论文。

## 来源回访（近 7 日新增条目引用来源的后续更新）

按额度回访 14 个来源（完整评论 8 个、状态快照 6 个）：

- **tt-metal #57461**（昨日"首选下一条目"）：评论（VirdhatchaniKN，修复 #57463 作者）记录三个相邻缺陷——writer 仍丢部分尾组（与 #51290 联动暴露）、`reduction_size` 无校验、真实消费者测试被跳过；**#57463 open 未合并** → 本次补录为 **1.247**。
- **vLLM #58031（5.44）**：补读全部 9 条评论——第二台 GB300 + MiMo-V2.6-Pro-RL 复现同类病理性 autotune；autotuner 实有 tqdm 进度（`\r` 写 stderr 可被行采集吞）。**条目勘误**：「无进度日志」修正为捕获口径问题；根因未定判定不变。
- **vLLM #58485**：4 条评论——hclsys 独立源码复核（16 处 force_index 全为单元素，"比报告更强"）；njhill 指示 MRV2；hclsys 实测 V2 干净、指出 V1-only spec 特性仍可达。编入 **11.87** 时已纳入。
- **vLLM #58627**：2 条评论——报告者撤回"演示的服务影响"（29 空回复 0 重复尾巴）。**6.255** 按机制收录、影响声明从简。
- **vLLM #58422**（09-23 留观）：1 条评论（hclsys）——#53130 是不同失效（调度拒纳 vs 本例 sync 等待）、建议分谱系。维持留观。
- **vLLM #58267**（09-23 留观）：2 条评论——hclsys 确认两半缺陷在 main 成立（3-D gap 为单 if/else 两行不对称）；修复 PR #58268 已提。第二重（假 bias 垃圾转录）静默面待运行时确认，维持留观。
- **vLLM #58105（→10.82）**：3 条评论——he-yufeng 独立确认（installed `vllm.parser.qwen3.Qwen3Parser` 路径、dedicated/ordinary 对照输出）；yuchenwang3 给 docs-first 方向（XGrammar token 元素）。条目无需修改。
- **vLLM #58138（→10.83）**：1 条评论——修复 PR #58143 已提。
- **vLLM #58119（→6.252 语境）**：1 条评论（viz 转交）。状态 open 不变。
- **vLLM #58390（→11.86）/#58080（→11.85）/SGLang #40730（→5.43）/flashinfer #5438（→7.96）**：状态 open、评论计数无实质变化（REST 快照）。
- 未检查完说明：近 7 日新增条目引用来源约 73 个（09-18..09-23 合并 71 条 + 勘误），本日按额度回访 14 个（优先昨日限额缺口 5 个 + 昨日新合并 12 条的高价值来源 + #57461 补录核实），其余约 59 个经本日 search 快照确认无状态翻转（无 closed/reopened），评论级深查缺口滚动至后续扫描。

## 合并动作摘要

- 主文档新增 **12 条**（6 个新子签名 + 6 个已知类别变体）：1.247–1.252、4.26、6.255–6.256、7.98、8.93、11.87
- 总条数 1194 → **1206**；编号编目 1080 → **1092**（K* 扩展 114 不变）
- 分类分布：第 1 类 245→251、第 4 类 25→26、第 6 类 254→256、第 7 类 97→98、第 8 类 92→93、第 11 类 85→86
- 轴分布更新：infra_off 1027→1039、config 252+6⚠、other_stack 207+25⚠、hw_specific 299→304、yes 121→122、documented 683→690、verified 504→509、NEW 877→883、var:x 314→320
- 条目更正：5.44（vLLM #58031）评论补读勘误（tqdm 进度实存按捕获口径；第二台 GB300 复现；根因未定不变）
- 观察：**6.256 首次按"同目标双栈同机制"合并 vLLM Nixl 与 SGLang Mooncake**（MLA 复制假设套到 head-sharded draft 区域）；**4.26 确立"复制计算 × 离散选择需要 bit-identical 或单一写者"水律**（per-rank autotune 为新根因面、干预级因果闭合）；1.249 是 TF32 截断族在 tt-metal SrcA 维度首例（与 DEST→SrcA 折返双机制复合）；11.87 与 11.86 构成 thinking-budget 强制结束状态机的两半；1.252 显示静默数值错误可同时掩盖性能上限（修复后快 14–21%）
- 导出与校验：`python3 export_faults.py`（1206 行 CSV/XLSX）→ `python3 verify_catalog.py` **PASS 1206 条：正文汇总、CSV、XLSX、字段边界一致**
