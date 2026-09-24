# LLM 推理故障每日扫描报告 — 2026-09-23

- **扫描日期**：2026-09-23（UTC）
- **扫描窗口**：GitHub `created:>=2026-09-21`（与 09-22 扫描窗口重叠两日，已逐条去重）；arxiv 按提交时间倒序取最近 5 条/查询
- **目标仓库**：vllm-project/vllm、sgl-project/sglang、huggingface/transformers、NVIDIA/TensorRT-LLM、triton-lang/triton、tenstorrent/tt-metal、flashinfer-ai/flashinfer
- **主文档基线**：1184 条（2026-09-22 更新后）；编号编目 1070 条 + K* 扩展 114 条

## API 调用记录

| # | 类型 | 查询 | 结果 |
|---|------|------|------|
| 1 | GH Search | `repo:vllm-project/vllm is:issue created:>=2026-09-21 wrong OR incorrect OR corruption OR NaN` | 18 命中 |
| 2 | GH Search | `repo:vllm-project/vllm is:issue label:bug created:>=2026-09-21` | 34 命中（30+4 两页补全） |
| 3 | GH Search | `repo:sgl-project/sglang is:issue created:>=2026-09-21 wrong OR incorrect OR corruption OR NaN` | 11 命中 |
| 4 | GH Search | `repo:huggingface/transformers is:issue created:>=2026-09-21 wrong output OR incorrect OR NaN` | 4 命中 |
| 5 | GH Search | `repo:NVIDIA/TensorRT-LLM is:issue created:>=2026-09-21 wrong OR incorrect OR corruption` | 1 命中（RFC，已判） |
| 6 | GH Search | `repo:triton-lang/triton is:issue created:>=2026-09-21 wrong OR miscompile OR NaN` | 0 命中 |
| 7 | GH Search | `repo:tenstorrent/tt-metal is:issue created:>=2026-09-21 numerical OR wrong OR pcc OR silent` | 65 命中（首 30 + per_page=100 单请求补全） |
| 8 | GH Search | `repo:flashinfer-ai/flashinfer is:issue created:>=2026-09-21 wrong OR incorrect OR NaN` | 3 命中（全部已编目） |
| 9 | GH Search | `repo:vllm-project/vllm is:issue created:>=2026-09-21 quantization OR "KV cache" OR deterministic` | 43 命中（两页补全） |
| 10 | GH Search | `repo:vllm-project/vllm is:issue created:>=2026-09-21 silent OR "bit flip" OR "weight corruption"` | 29 命中 |
| 11–15 | arxiv | `"silent data corruption" AND LLM` / `"bit flip" AND "LLM weights"` / `"KV cache" AND corruption` / `"LLM inference correctness"` / `"GPU soft error" AND "machine learning"` | 5/1/5/1/0 命中 |

GitHub Search API 10 组查询全部有效、无限流 403、无 Validation Failed。**执行过程勘误**：扫描脚本首版 `run_q` 传参错误致 q 参数丢失（Validation Failed `field: q, code: missing`），修复后全部重跑成功——首版失败不计入覆盖。core REST 详情+评论调用 33 次中 28 次成功（`--http1.1`）；5 次因未认证 core 限额（60/h）耗尽 403 失败（#58105/#58138/#58119/#58080/#58031 的评论正文），这些目标的状态与评论计数已由当日 search 快照覆盖（全部 open），**评论正文缺口在下一限额窗口补查**。arxiv 5 组查询全部成功（1 组 0 命中、无 429）。

## 已编目/已判条目（本次跳过）

- vLLM：#58138（→10.83）、#58105（→10.82）、#58080（→11.85）、#58031（→5.44）、#58029（→10.78）、#58020（→6.250）、#57999/#57998/#57997/#57996（09-21 已判）、#57986（→11.84）、#57950（→10.79）、#57938（→6.251）、#57935（→10.80）、#57929（→10.81）、#57944（已修）、#58134/#58211（CI）、#58034（09-21 留观）、#58035/#58214/#58135/#58191/#58155/#58145/#58144/#58087/#58063/#58178（09-22 已判）
- SGLang：#40735（→5.42）、#40730（→5.43）、#40623（09-22 留观）、#40564（09-21 留观）、#40504（性能）
- tt-metal：#57413（→1.244）、#57410（→1.241）、#57408（→1.242，见回访）、#57382（→1.243）、#57394/#57403/#57370（09-22 已判）、#57360–#57367（静态审计批留观）、#57248（→1.237）、#57205（→1.240）、#57163（→1.238）、#57149（→1.239）、#57212/#57211（09-21 已判）
- flashinfer：#5438（→7.96）、#5393（09-22 留观）、#5389（09-21 留观）；TRT-LLM #19501（RFC）；triton 窗口 0 命中

## 🆕 新发现（本次合并，共 12 条）

### A. 全新子签名（5 条）

1. **[新签名] tt-metal #57608 — SDPA 隐式 tile padding 内容到达输出：非 32 倍数序列长 + K/V padding 大值时 PCC −0.003、max err 5.5e36，静默** — 非 causal 无 mask 路径内核应自行掩掉越界 key 位置却未做到；K padding 单独污染 PCC 0.973、V 单独无影响、matmul→softmax→matmul 同数据全对（缺陷在 SDPA 边界处理）。→ 归入第 1 类 **1.245**，Cov NEW。
2. **[新签名] SGLang #40921 — v0.5.20 在 H200/SM90 接受 `trtllm_mha` 双侧（prefill+decode）后静默返回错误补全；v0.5.17 曾明确拒绝；单侧各自正确** — 拒绝门消失使 fail-loud 启动错退化为 fail-silent 错服务（gpt-oss-120b 四组合探针实测）。→ 归入第 5 类 **5.45**，Cov NEW（与 7.70 门缺失互为镜像：本条是门被移除）。
3. **[新签名] vLLM #58362 — Mooncake connector 把 min() 塌缩后的逻辑 block_size 读作物理传输块尺寸，hybrid 混合块尺寸模型 PD 首次传输即断** — `core.py` 把 cache_config.block_size 塌缩为各 group 最小值（1024→128），worker assert 只覆盖 Eagle 方向，Ascend 910B 上首笔 P2P `remote_block_size(128) not divisible by kernel_size(1024)` 杀 decode EngineCore。→ 归入第 6 类 **6.254**，Cov var:6.211（逻辑/物理粒度双权威源族）。
4. **[新签名] SGLang #40903 — DeepSeek chunked-prefix prefill 把 base-2 LSE 按自然对数公式合并：每次 prefix-cache 命中缓存前缀被 2^(0.4427·Δ) 指数过权，静默精度损失** — Blackwell 上 trtllm_mla 自动选中该路径、**默认配置即触发**；grep 证实无 log2/ln 转换；0.5.15.post1/0.5.19 运行时复现；修复 #40927。→ 归入第 7 类 **7.97**，Cov NEW（分块注意力 LSE 底数契约分裂首例）。
5. **[已知类别] transformers #49051 — Gemma 1 checkpoint 自 v4.48.0 起静默改用 exact erf GELU（#35235 删 hidden_activation 守卫），训练用 tanh 近似，无警告** — 版本 tag bisect 精确到 v4.47.1→v4.48.0；candle-mi Rust 实现交叉验证；维护者确认修复方向（config post-init 恢复映射）。→ 归入第 10 类 **10.86**，Cov var:12.39（版本 pin 摩擦谱系——unpublished 契约跨版本丢失；原文该条列于新签名组，编号后核实归为 12.39 变体）。
6. **[新签名] vLLM #58390 — `thinking_token_budget` 生成在预算耗尽点精确坍塌为重复循环：forced end 序列全量入尾后 prefix-match 重对齐为 0，永久重复首 token** — 生产实测确定性复现；根因已定位（end_prefix_len 重算 0），修复 PR #58402 附 GPU 回归测试。→ 归入第 11 类 **11.86**，Cov NEW（预算强制状态机边界重对齐子签名）。
7. **[新签名] SGLang #40966 — Gemma-4-31B（attention_k_eq_v）LoRA 加载三重失败：顶层 config 读不存在字段、per-layer buffer 维度按全局 head 数错位、vision-tower 张量碰撞** — safetensors header 证实 50 个 sliding 层投影齐全 + 10 个 full_attention 层无 v_proj 的异构投影集，统一七投影假设全断。→ 归入第 12 类 **12.43**，Cov NEW。

### B. 已知类别新变体（7 条）

8. **[已知类别] tt-metal #57538 — `ttnn.log/log2/log10`（bf16 accurate mode）x=−0.0 返 +inf（应 −inf）：指数读于 −0.0 规范化前、尾数读于规范化后** — fp32 路径恰正确（同函数两 dtype 分歧）；#55931 backward 修复后的 forward 姊妹缺陷；修复 #57554 仅需交换两条语句顺序。→ 归入第 1 类 **1.246**，Cov var:1.193（log 族 ±0.0 极点符号族前向补全）。
9. **[已知类别] vLLM #58394 — 部分 prefix-cache 条目提升（promotion）事件引用同批刚移除的 parent hash，外部消费者链不可重建** — CPU-only 单元级复现真实 BlockPool；#58383 不覆盖（本形态被既有测试 pin）；修复方向 #58396（重锚定幸存块对齐祖先）。→ 归入第 6 类 **6.253**，Cov var:6.252（KV 事件契约族第三形态：parent 从未上报 → parent 同批撤回）。
10. **[已知类别] vLLM #58391 — DSV4.1-Flash nightly 间 teacher-forced logprob 在结构化文本上位移（tool 转录 top-1 翻转 1.77%→12.66%），perplexity 不变掩盖** — 81-commit 窗口 7 候选，社区源码核查指向 #56893（V4.1 KV 记录全 512 维 MXFP8 化：584→528 B/token）；报告者带恒等 hook caveat（未做无 hook 复现）。→ 归入第 8 类 **8.92**，Cov var:8.53（KV 存储格式变更数值位移族），置信度 documented。
11. **[已知类别] transformers #49066 — `Qwen2Tokenizer.__init__` 用模块级常量覆盖 tokenizer.json 的 pre_tokenizer：Qwen3.6/3.8 refined 正则（新增 `\p{M}`）被静默丢弃，全族 token IDs 错** — `"नमस्ते दुनिया"` 发布 6 token vs 加载 8 token；4.57.6 正确、5.17.0 错。→ 归入第 10 类 **10.85**，Cov var:10.56（tokenizer 双源漂移族——硬编码常量 vs 发布文件）。
12. **[已知类别] SGLang #40835 — LoRA 加载器无视 `use_rslora`，恒按 `lora_alpha/r` 计算 scaling：rsLoRA 效果静默削弱 √r 倍（r=64 时 8×）** — CPU 纯 Python 复现（PEFT scaling 16.0 vs SGLang 2.0）；与 12.32（vLLM pack_moe 重算丢 √r）构成跨栈姊妹对。→ 归入第 12 类 **12.42**，Cov var:12.32。

### C. 评估后不收录（记录备查）

- **vLLM #58422**（TP=1 SM120 GDN-hybrid MoE 静默 wedge：generation counter 冻结、running>0/waiting=0、health 200、GPU 100% util 但 26–138W 空转）：观察面真实且三起一致，但根因未定（社区评论区分 #53130（调度拒纳）与本例（sync 等待永不落地）、并指出 `--kv-cache-dtype nvfp4` 的 `_validate_cache_dtype` 只 log 不查 arch/backend）；同窗口 16 次 IMA fail-loud 崩溃是独立形态。留观，待根因或可复现最小配置。
- **vLLM #58441**（Qwen4Exp PLE prefetch 读 graph-pool 内存被后续 segment 复用覆写，greedy 冷暖不复现 2/8、持久 buffer 后 8/8）：side-stream 竞态实测真实，但 PinnedHost 本体未能运行（128GB 统一内存装不下 pinned 表）、非 PinnedHost 复现；候选留观，待 PinnedHost 生产路径证据。
- **vLLM #58384**（Harmony parser `Unknown channel: comment` ValueError 关 SSE 流）：fail-loud 流中断（客户端可见错误），错误报告质量域。不收录。
- **vLLM #58353**（Qwen2-VL 单帧视频 round 后 0 帧除零）：fail-loud ZeroDivisionError；复现者正在修复。不收录。
- **vLLM #58303**（Mamba+EAGLE dense 默认交织长对话 0% 前缀复用）：性能/调度策略域（dense 与 sparse 两端均可 0%），非正确性。不收录。
- **vLLM #58273**（Whisper `LLM.generate` 无 HF `generate_with_fallback` 温度回退：续写循环/空 EOT）：解码策略缺失（模型能力域），vLLM Python API 语义设计议题。不收录。
- **vLLM #58267**（CPU W4A16 Whisper：`cpu_gemm_wna16` 拒 3-D 激活 + 假 k_proj bias 合成错（WER 100））：第一重 fail-loud IMA（参照 #57941 先例）；第二重（bias 合成错→垃圾转录）在第一重修复后才能到达，修复 PR #58268 在途。整条留观，待运行时确认第二重静默面。
- **vLLM #58241**（`cutlass_scaled_mm_supports_block_fp8` 排除 SM89）：社区核实为 CUTLASS 库层无 c2x blockwise 实现——回退 Marlin 是**正确行为**（非静默错值）。不收录。
- **vLLM #58228**（`finish_reason:"tool_calls"` 遮蔽 max_tokens 截断）：API 语义/协议域（客户端可由 usage 推断），非推理正确性。不收录。
- **vLLM #58222**（NIXL lease reaper 心跳头阻塞后续收割）：源码检查级发现（standalone 复现未量化运行时影响）、资源回收治理。留观。
- **vLLM #58280**（DSV4 chat template tools 合成 system 消息）：main 已修、报告者自行关闭（released 0.28.0 限定）。不收录。
- **vLLM #58406**（mm_processor_kwargs flat/scoped 合并语义）：数据丢失半已修（#58408 deep-merge）、precedence 契约半留观（#56372 完整修复在途）。留观。
- **vLLM #58421**（Responses `truncation:"auto"` token 级左截断丢 instructions/mid-item）：API 规范兼容域（与 OpenAI spec 分歧），非静默损坏。不收录。
- **vLLM #58290**（多节点 TP8 NCCL 启动挂）、#58324（实时会话 O(n) 调度）、#58233（并发性能求助）、#58236（FlashInfer 0.6.18 编译失败）、#58246（SDK 兼容测试）、#58263/#58240（feature）、#58242（RPC 治理）、#58423（文档）、#58100（bench 工具）、#57981（性能）、#57971（性能，closed）、#57974（资源治理）、#57936（容量建议工具）、#57932（fail-loud 后端拒绝）、#57847（bench connector 重复填充）、#57853（sleep 中权重更新 segfault——fail-loud 生命周期）、#57878（fail-loud 启动 crash）、#57895/#58329（RFC）、#57927（09-22 已判单例留观）、#57956/#58211/#58134（CI）：环境/性能/fail-loud/feature/CI 域，均不占故障编号。
- **SGLang #40926**（HiCache hybrid `cudaHostRegister` 失败杀全实例 + 错误路径 TypeError 吞 rc）：fail-loud 启动崩溃（加速缺失被升级为全 outage 属可靠性/治理），诊断质量半（TypeError 掩盖真实 rc）不占号；三个修复 PR 在途。留观。
- **SGLang #40817**（未知 tool call 解析期丢弃杀 agent 循环）：feature 议题（转发语义设计）。不收录。
- **tt-metal #57510/#57509/#57506/#57504/#57503/#57502/#57499/#57498/#57496/#57478/#57477/#57476/#57475/#57474**（静态审计 meta-issue 第二批 ~90 项，impl/hw/fabric/ccl/eltwise/moreh/data_movement/mmfusedreduce/convolutions 域）：延续 #57360–#57367 判例——静态审计产物无逐项运行时证据，**整批留观**，待单项缺陷获得运行时证据后逐条编目。
- **tt-metal #57461**（`moe_expert_token_remap` writer 双计数组边界漂移：每 reduction 组丢尾页 + reduce_idx 越界覆写下一 core 页，Wormhole 实测 906/4096–1346/4096 错元素）：**运行时证据充分**，因 core 限额耗尽未及拉取评论核实修复状态——**首选下一条目**，明日优先补录（避免在未完成核实时占号）。
- **tt-metal #57515**（repeat all-ones 快路径返回未扩维张量，丢 rank/memory_config）：已 closed（当日修复）；静默错形状面窄（repeat 向量长于 rank 的 all-ones 拼写）。留观。
- **tt-metal #57458**（`all_gather_minimal_matmul_async` 无持久 buffer 时对端写 lagging 设备 L1，Qwen3.6 TP-4 16/20 runs corrupted）：静默损坏真实且严重，但触发需实验性 API 的无防护调用（docstring 要求 persistent buffer 或 barrier、barrier 被接受但从未使用是独立缺陷）。留观，待生产路径证据。
- **tt-metal #57455**（`relu6_bw` NaN 输入返 6.0，ULP 距 1.09e9）：NaN 输入语义边界（golden/torch 返 0）；backward 域、触发需上游 NaN。留观（与 1.188 sign(NaN) 族机制相近，待上游 NaN 来源案例再判归并）。
- **tt-metal #57512**（`reduction_size` 主机端无校验（0/非整除））：防御性加固（当前 split 掩盖、无静默面）。不收录。
- **tt-metal #57480**（MemoryConfig 复挂载不复验）、#57603（重构）、#57586（线程池性能）、#57540（bring-up hang）、#57452/#57271/#57155（CI 机器人）、#57450（性能）、#57343/#57335（feature）、#57308（支持项）、#57236（bring-up）、#57230（profiler）、#57178（性能）、#57164（#57163 重复贴，closed）：非正确性域。不收录。
- **transformers #49036**（`Cache.crop` 不回滚 GDN recurrent state：draft-verify 块拒绝后状态含被拒 token，后续 logits 依赖未发出 token）：机制上与 6.246（DynamicCache offloading GDN 往返损坏）同族且严重（spec/块验证非无损），但维护者标记"核心已知、等 Cyrilvallez 方案"、triage 要求更清晰复现。留观，待修复方案落地或运行时证据补强后并入 6.246 谱系。
- **transformers #49026**（serge 集成分拣）：CI 集成域。不收录。
- **arxiv**：5 组查询命中全部已编目或已判（SProbe/TrainSDC/SCOUT/LLM-PRISM/2604.00726/BitSifter/2609.17983/2608.07408/2608.05863 已编目；Nexus/ReaLM 方法论文不收录；2609.25881 高保真虚拟试穿非 LLM 域、OR 拆解噪声）。无新编目论文。

## 来源回访（近 7 日新增条目的后续更新）

按额度分批检查 15 个已编目来源（完整正文+评论 4 个、状态/计数快照 11 个）：

- **1.242 / tt-metal #57408**：**已关闭（completed）但仅部分修复**——维护者合并 #57421 只修 `calculate_square`；报告者复核实测其余五处存储点（rad2deg/deg2rad/softsign/hardsigmoid/prelu）仍截断（`rad2deg(-4.5)` 仍 −256.0），持有六内核全量补丁（ttsim census 1024/1024 bit-exact）待问剩余站点是否在 #57172 跟踪。主文档 1.242 已补"部分修复"状态注记（机制判定不变）。
- **1.244 / tt-metal #57413**：修复候选 #57414 验证有效——host 侧 load-time 精确 tanh 门控使 e2e BERTScore F1 达 **0.7555**（高于 #57179 前父 0.702 与窗口前 0.722，越过 0.69 门禁）；6-entry 表对小 gate 值的精度问题对其他 `fast_and_approximate_mode=True` 用户仍然开放。
- **1.241 / tt-metal #57410**：社区补丁就绪（convert_u32_to_bfp + 设备 parity 回归：bfloat8_b 2983/35840→0、bfloat4_b 435/1024→4/1024，余 4 例为 BFP8 溢出路径）；bounty 流程沟通中。未合并。
- **1.243 / tt-metal #57382**：维护者认领（row-major stride offset bug in slice_op.cpp），复现测试搭建中。
- vLLM #58105/#58138/#58119/#58080/#58031：core 限额耗尽未能拉取新评论正文；search 快照确认全部仍 open、评论计数 3/1/1/2/9（#58031 的 9 条评论含 09-23 撤回后的后续讨论，**下一限额窗口优先补查**）。SGLang #40730、flashinfer #5438：仍 open、无新评论。
- 未检查完说明：近 7 日新增条目引用来源共约 61 个（09-17..09-22 合并 71 条），本日按额度完整回访 15 个（优先 09-22 新增 13 条全部来源 + 高评论数条目），其余 46 个经 search 快照确认无状态翻转（无 closed/reopened），评论级深查缺口约 46 个，滚动至后续扫描。

## 合并动作摘要

- 主文档新增 **12 条**（5 个新子签名 + 7 个已知类别变体）：1.245、5.45、6.253–6.254、7.97、10.85–10.86、11.86、12.42–12.43、1.246、8.92、10.86
- 总条数 1184 → **1196**；编号编目 1070 → **1082**（K* 扩展 114 不变）
- 分类分布：第 1 类 244→246、第 5 类 44→45、第 6 类 252→254、第 7 类 96→97、第 8 类 91→92、第 10 类 84→86、第 11 类 85→86、第 12 类 41→43
- 轴分布更新：infra_off 1017→1029、config 243→247+6⚠、other_stack 202→207+25⚠、hw_specific 297→300、documented 682→685、verified 495→504、NEW 874→879、var:x 307→314
- 条目更正：1.242 补部分修复状态注记（#57421 只修 1/6 内核、issue closed 但五站点残留）
- 观察：7.97 是分块注意力 **LSE 底数契约分裂**首例（每次前缀命中静默指数过权、Blackwell 默认路径即触发）；6.253 使 KV 事件契约族成三形态（parent 从未上报/载荷不可解码/parent 同批撤回）；12.42 与 12.32 构成 rsLoRA √r 丢失**跨栈姊妹对**；10.85 与 10.56 构成 tokenizer 双源漂移对；5.45 是首条"拒绝门被移除→静默错服务"方向条目（与 7.70 门缺失互为镜像）；10.85/10.86 同日把 transformers 侧两条 v4→v5 版本回归（常量覆盖 pre_tokenizer、守卫删除激活漂移）编入。
- 执行勘误（不改历史记录）：扫描脚本首版 run_q 传参缺陷致 GitHub 查询首访全败（q 参数未附上），修复重跑后当日全部查询有效；core REST 5 次限额失败已记录，缺口与补查窗口见上文。
- 导出与校验：`python3 export_faults.py`（1196 行 CSV/XLSX）+ `python3 verify_catalog.py` → **PASS 1196 条：正文汇总、CSV、XLSX、字段边界一致**
