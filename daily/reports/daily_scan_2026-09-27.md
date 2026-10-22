# LLM 推理故障每日扫描报告 — 2026-09-27

- **扫描日期**：2026-09-27（UTC）
- **扫描窗口**：GitHub `created:>=2026-09-20`（7 日完整窗口，`cutoff=$(date -u -d '7 days ago' +%F)` 实际计算；与 09-26 扫描窗口 created:>=2026-09-19 全重叠——09-26 及更早判定条目全部按既有判定跳过，本日新判定集中于 09-26 扫描执行（约 09-26 23:39 UTC 前）之后创建的 issue：vLLM #58864/#58882/#58886/#58893/#58894/#58899/#58902/#58909/#58910/#58912/#58913/#58922/#58930/#58931/#58943、transformers #49134、SGLang #41466/#41467/#41471、tt-metal #58030/#58031/#58044/#58046（均 09-27 新建））；arxiv 查询当日持续全部失败（见未覆盖范围）
- **目标仓库**：vllm-project/vllm、sgl-project/sglang、huggingface/transformers、NVIDIA/TensorRT-LLM、triton-lang/triton、tenstorrent/tt-metal、flashinfer-ai/flashinfer
- **主文档基线**：1224 条（2026-09-26 扫描后）；编号编目 1110 条 + K* 扩展 114 条

## Git 前置检查（第 3 节）

- 分支 `main`、上游 `origin/main`、`git status --porcelain -uall` 为空 ✅
- `git fetch origin main` 成功；`HEAD` = `origin/main` = `d716abe230b42c2928dbb9c3998d25401dbd3296` ✅ → 开始扫描

## API 调用记录

| # | 类型 | 查询 | 结果 |
|---|------|------|------|
| 1 | GH Search | `repo:vllm-project/vllm is:issue created:>=2026-09-20 wrong OR incorrect OR corruption OR NaN` | 42 命中（单页取全） |
| 2 | GH Search | `repo:vllm-project/vllm is:issue label:bug created:>=2026-09-20` | 56 命中（单页取全） |
| 3 | GH Search | `repo:sgl-project/sglang is:issue created:>=2026-09-20 wrong OR incorrect OR corruption OR NaN` | 18 命中（单页取全） |
| 4 | GH Search | `repo:huggingface/transformers is:issue created:>=2026-09-20 wrong output OR incorrect OR NaN` | 6 命中（单页取全） |
| 5 | GH Search | `repo:NVIDIA/TensorRT-LLM is:issue created:>=2026-09-20 wrong OR incorrect OR corruption` | 1 命中（#19501 RFC，09-24 已判） |
| 6 | GH Search | `repo:triton-lang/triton is:issue created:>=2026-09-20 wrong OR miscompile OR NaN` | 1 命中（#11889，09-24 已判） |
| 7 | GH Search | `repo:tenstorrent/tt-metal is:issue created:>=2026-09-20 numerical OR wrong OR pcc OR silent` | 143 命中（**两页取全 100+43**） |
| 8 | GH Search | `repo:flashinfer-ai/flashinfer is:issue created:>=2026-09-20 wrong OR incorrect OR NaN` | 3 命中（#5389/#5393/#5438 均已编目或已判） |
| 9 | GH Search | `repo:vllm-project/vllm is:issue created:>=2026-09-20 quantization OR "KV cache" OR deterministic` | 89 命中（单页取全） |
| 10 | GH Search | `repo:vllm-project/vllm is:issue created:>=2026-09-20 silent OR "bit flip" OR "weight corruption"` | 64 命中（单页取全） |
| 11 | GH Search | `repo:sgl-project/sglang is:issue label:bug created:>=2026-09-20` | 0 命中 |
| 12–17 | arxiv | 引号短语 + AND ×6：`"silent data corruption" AND LLM` / `"bit flip" AND LLM` / `"KV cache" AND corruption` / `"soft error" AND GPU AND inference` / `inference AND "numerical correctness"` / `inference AND "wrong output"` | **6 组全部 HTTP 406 失败**（连续第二天；诊断见未覆盖范围） |

GitHub Search API 11 组查询全部有效、**分页完整（fetched == total 逐组核对：42/56/18/6/1/1/143(100+43)/3/89/64/0）**、无限速 403、无 Validation Failed；查询间 ≥8s 间隔。core REST 详情/评论/PR 状态 29 次调用全部成功（`--http1.1`，未认证额度 60/小时，本轮消耗 29 次）。候选正文优先取自 Search API 缓存（响应自带 body），新条目与回访评论核对使用 core REST。**未认证运行（环境变量 TOKEN 经核验为 401 Bad credentials 无效凭据，未使用）。**

## 已编目/已判条目（本次跳过，重叠区去重）

- vLLM（09-26 及更早已判/已编目）：#58824（→10.87，评论见回访）、#58849（性能）、#58850（Xid 留观）、#58858（→8.94）、#58807（→13.40，评论见回访）、#58729（→7.99，评论见回访）、#58640/#58677/#58680/#58688/#58692/#58697/#58733（09-26 已判）及其余 09-20..09-25 判定项
- transformers：#49126（serge CI triage，closed）、#49066（→10.85）、#49051（→10.86，closed）、#49036（留观）、#49026（CI triage）
- SGLang/tt-metal/flashinfer/triton/TRT-LLM：#41192（→5.46，0 评论无变化）；tt-metal #57988–#58001（09-26 已判：#57988 留观、#57989→1.261、#58000→1.259、#58001→1.260）；#5389（closed 后无进一步变化）/#5393/#5438（→7.96）已判维持；triton/TRT-LLM 窗口无新增未判项

## 🆕 新发现（本次合并，共 5 条）

### A. 全新子签名（1 条）

1. **[新签名] vLLM #58902 — Qwen3-Omni `get_mrope_input_positions` 把 `PlaceholderRange.offset` 当模态起始索引，每个媒体项多发一个 bos 槽位：M-RoPE 位置比 token 跨度长一格——stock chat template 下多余槽被尾部文本静默吸收，无报错但首个 pad token 起**全部多模态请求（图/音/视频）位置错一位**；模板以媒体结尾才 raise 长度失配** — 免 GPU/免权重合成输入单测（6 token → 7 位置）；修复 PR #58890（open 未合并）。→ 归入第 9 类 **9.30**，Cov NEW（vLLM 位置装配层 offset 契约误解，与 9.16/9.17 SGLang fused kernel 数值错同任务域但机制独立），verified。

### B. 已知类别新变体（4 条）

2. **[已知类别] tt-metal #58046 — `MatmulMultiCoreReuseMultiCast1DProgramConfig`（mcast_in0）+ width-sharded L1 输出在 `out_block_w < per_core_N` 时静默错值（PCC 0.10–0.63）：验证器因 `out_block_h == 1` 放行（`out_block_w == per_core_N || out_block_h == 1`），但该豁免对 sharded 输出不成立；INTERLEAVED 同参与 DRAM 输出全对** — 生产形状 `1×2048×39576`（per_core_N=20）out_block_w=2 时 PCC 0.10；报告者附 repro 脚本与 5 组形状矩阵。→ 归入第 1 类 **1.262**，Cov var:1.119（MatmulMultiCoreReuse 族：验证豁免谓词过度放行 vs 块数超核丢块），verified。
3. **[已知类别] vLLM #58894 — DFlash2 spec + Qwen3.5 hybrid GDN + prefix caching：首个前缀命中瞬间 acceptance 塌到精确 0.0% 且 server 生命期内永不恢复（drafter 照常 ~28-29 tok/s、Accepted 0.00）；x86 形态同场景 12,928-token 命中直接 IMA——twu3202 定位为 6.204 同因（mamba state index 以被降低的 block_size 64 播种），#55601 修补后两硬件均恢复（Orin 报告者确认"thinks a lot faster now"）；修复 PR #55601 本轮核实仍 open 未合并、v0.30.0 发布带病** — 双独立重启复现、v0.29.0 正常（跨版本回归）。→ 归入第 6 类 **6.259**，Cov var:6.204（显形谱系第三态：Xid 崩/静默错 state/spec 全拒），documented。
4. **[已知类别] vLLM #58930 — `validate_xgrammar_grammar()` 对 structural tag 内嵌套 JSON schema 跳过 unsupported-feature 检查：plain json 会被拒并回退 guidance/outlines 的 schema（如 `multipleOf`）在三条路径下静默留在 xgrammar——(1) 用户 structural_tag（新/legacy 形态）、(2) strict tool 参数 schema（required/auto/named）、(3) Harmony（gpt-oss）把一切 response_format 包进 structural tag；`{"multipleOf":0.5}` 编译出的 grammar 接受 `0.3`** — main `2b9b55c7f1` + xgrammar 0.2.7 复现；维护者认领并已开修复 PR #58940（open）。→ 归入第 10 类 **10.88**，Cov var:10.40（约束静默失效族第五个独立机制：校验准入的嵌套盲区），verified。
5. **[已知类别] transformers #49134 — UMT5 decoder self-attention 不做因果掩码：`UMT5LayerSelfAttention` 不传 `is_causal=config.is_decoder`（T5/MT5 均传），SDPA 下该 flag 即掩码——teacher-forced 前向位置 0 的 logits 随后续 token 追加而变（drift 脚本 UMT5 `LEAKING` / T5 `CAUSAL`）；v5.14.1 正确、v5.15.0 起回归（#47014 SDPA 迁移漏传），v5.15/5.16/5.17/main 全中** — Hub 直跑复现、回归 PR 定位精确。→ 归入第 13 类 **13.41**，Cov var:13.27（掩码参数静默丢弃族：SDPA 迁移回归形态），verified。

### C. 评估后不收录（记录备查）

- **vLLM #58912**（XPU `logits_soft_cap` 在 `flash_attn_varlen_func` 被静默丢弃——签名接受 `softcap` 参数但转发调用处被注释掉，且 `get_flash_attn_version()` 在 XPU 短路 v2 使 softcap/ALiBi 回退逻辑永不触发；Gemma-2/3/4 在 Intel GPU 无 soft cap 运行，无警告；内核侧 plumbing 已具备、纯 vLLM 侧缺陷）：**静默数值面真实且机制定位清晰**，但 XPU 非默认栈、报告者未附任务质量实测值（"plausible-but-wrong logits"为推断），按"未定位根因者留观/无任务质量实测"口径留观，待质量证据或修复 PR 出现后并入。与 8.90（gfx1151 ALiBi 静默吞）构成跨后端"参数被接受但语义被丢弃"对偶的候选。
- **vLLM #58913**（XPU `get_attn_backend_cls` 从不调用 `validate_configuration()`——CUDA/ROCm 平台各 2 处、XPU 0 处，全部注意力后端能力门（head size/KV dtype/block size/mm_prefix/batch invariance）在 XPU 为死代码；FlashAttentionBackend 能力 flag 与 xpu.py 自述限制矛盾（supports_batch_invariance()→True vs 文档说 XPU 非 batch-invariant））：结构缺陷真实（静态可核），但无具体静默错值实例（报告以"若干其他 XPU 注意力 bug 因此呈静默形态"为语境）；与 5.45（门被移除→静默错服务）互为镜像的"门从未接线"形态。留观。
- **vLLM #58931**（MooncakeStoreConnector 在 `reset_prefix_cache(reset_running_requests=True)` 抢占 + 下一步重接纳时序下：V2 runner assert / V1 TypeError 杀 EngineCore（fail-loud 为主）；async load 形态 load spec 被丢、请求永久 `WAITING_FOR_REMOTE_KVS` 挂起）：挂起面真实但主形态 fail-loud；CPU-only 调度器单元级最小复现完整。6.139（KV load 拒绝后部分回滚状态分歧）offload 族语境，留观待上游分类（RL weight-update 工作流预期路径）。
- **vLLM #58864**（GLM-5.3-MXFP4 TP1/DP8 无 EP：FlashInfer autotune 32K dummy run 中 sparse MLA logical→physical top-k index 转换 IMA，8 DP worker 全挂——fail-loud 启动崩溃，main 复现）：不收录（无静默错值证据）。
- **vLLM #58886**（ROCm AITER MLA FP8 prefill async 调度下 persistent PS metadata 竞态 GPU memory fault——修复 #58887 已开（pinned host staging + 当前流拷贝排序），ROCm/aiter#5879 跟踪）：fail-loud 崩溃域，不收录。
- **vLLM #58893**（AITER v0.1.23 bump：gfx950 a16w4 MXFP4 MoE Gluon backend `AttributeError`——依赖 triton#11470（09-08 合并）晚于 vLLM ROCm 镜像 pin 的 triton `669b31a`）：依赖版本失配 fail-loud，发布/依赖锁定议题，不收录。
- **vLLM #58910**（DSV4.1-Flash SM120 4×RTX PRO 6000 长 prefill（>35K）Xid 31 MMU fault：graph 模式 0/3 全崩、eager ~25% flaky，70+ 受控实验排除 spec/prefix/guided 单一请求触发；FlashInfer 0.7.0 不修；跨投 flashinfer#5608 内核层）：fail-loud 崩溃域，与 6.248/#58850 同 Xid 谱系但触发面为长 prefill，未定位前留观。
- **vLLM #58882**（ROCM_AITER_UNIFIED_ATTN LBNHC/NHD 布局特性请求）、**#58909**（Reasoning governor RFC）、**#58922**（rocm_unquantized_gemm 忽略 tensor device 在 CPU tensor 上 crash——CI 测试面 fail-loud）、**#58943**（MiniCPM-V-4.6 官方 GPTQ/AWQ checkpoint vision tower 被量化构建加载失败——fail-loud，量化 ignore 列表缺口与 13.40 同族语境）：不收录。
- **SGLang #41466/#41467**（native `/generate` 无字段类型校验：单请求错误类型字段在 scheduler/detokenizer 抛 TypeError → SIGQUIT 整服务器死亡；637 请求扫描 25 次击倒、7 类成因；OpenAI 兼容端点同值干净 400）、**#41471**（两并发请求 DisallowedTokensLogitsProcessor 不同 token_ids 同批 → assert 杀引擎）：fail-loud SIGQUIT 域，6.192（单畸形请求引擎自杀×重试巡回击穿）族语境补充，留观不占号。
- **tt-metal #58030**（ring_joint_sdpa 短 Q slab 占用/越界 K chunk 掩码浪费/packed K/V 需双 slice 拷贝——性能提案，但附 per-head KV PCC 0.913 与 K-split 后 0.943 的数值界定）、**#58031**（gemma4_d_p chunk 开销性能）、**#58044**（GPT-OSS-120B BH Galaxy all_to_all_dispatch_metadata 内核不编译——bring-up fail-loud）：性能/bring-up 域，#58030 的 PCC 界定与 #57965/#57865 谱系关联已记录，不收录。
- **transformers #49126**（serge 集成 CI triage，closed）：CI 噪声，不收录。
- **arxiv**：6 组查询当日全部失败（HTTP 406），无命中可判——见未覆盖范围。

## 来源回访（近 7 日新增条目引用来源 + 额度轮巡旧来源）

本日回访 12 个来源（core REST 详情+评论全读）：

- **10.87（vLLM #58824）**：3 条新评论（09-26）——neevmodh 开 scoped 修复 #58829（latch 形态）；he-yufeng 逐读两份竞争修复（#58826 只修中途 MalformedJSON 分支、完成态仍丢），判定 **#58829 latch 是正确形状**、指出两缺口（latch 翻转不复位——同流 plain-JSON 先行+真调用会被吞；两 PR 均缺回归测试）；neevmodh 随即补三场景测试（plain-JSON-first/malformed interim/real-tool-call-after）并把 latch 限制写入代码注释。PR open 未合并。**条目来源行补回访记**，判定不变。
- **13.40（vLLM #58807）**：1 条新评论（09-27 he-yufeng）——认领修复：按 safetensors index 实际 tensor 名（`mtp.*` 是否带量化后缀）决定 predictor 量化与否，镜像 GPTQ dynamic/modelopt_fp4 先例；单文件 checkpoint 保持现行为。**条目来源行补回访记**，判定不变。
- **7.99（vLLM #58729）**：2 条评论（09-26 baonudesifeizhai 复现+根因收窄已录入；09-27 michaelmanly）——michaelmanly 待报告者提供确切 vllm serve 命令以映射 Badgr Serve 做真实验收。无新机制证据，条目不动。
- **6.259 根因链（vLLM #58894 评论）**：twu3202（x86 IMA 定位 #55601 同因 + 12,928-token 命中修复验证 acceptance 33–46%）、he-yufeng（启动警告为 `_warn_if_unannotated_eagle_mamba` 规则缺口、仅卫生问题；collapse 签名匹配 boundary-misregistration 族）、valacroix（#55601 在 Orin 验证有效）——已全部写入 6.259 条目。
- **修复 PR 状态核实（core REST）**：#55601（6.204/6.259 修复）open 未合并；#58940（10.88 修复，09-27 新开）open；#58890（9.30 修复）open；#58931 相关无 PR。
- **#54360/#57032（6.219/11.79 谱系，#58894 引用）**：经 core REST 快照确认 state/comments 不变（各 6 评论），语境引用无需改条目。
- 未检查完说明：近 7 日新增条目引用来源约 103 个（09-20..09-26 合并 97 条 + 本日 5 条 + 更正），本日按 core 额度回访上述 12 个高价值来源（本日 6 条新合并主来源全量 + 3 个修复 PR 状态 + 根因收窄链 + 2 个谱系引用）；其余约 91 个经本日 q01–q11 search 快照确认无 state 翻转/评论数突变（快照含全部窗口内活跃 issue 的 state/comments 字段），评论级深查缺口滚动至后续扫描。

## 合并动作摘要

- 主文档新增 **5 条**（1 个新子签名 + 4 个已知类别变体）：1.262、6.259、9.30、10.88、13.41
- 总条数 1224 → **1229**；编号编目 1110 → **1115**（K* 扩展 114 不变）
- 分类分布：第 1 类 260→261、第 6 类 258→259、第 9 类 29→30、第 10 类 87→88、第 13 类 40→41
- 轴分布更新（按解析器实测）：infra_off 1055→1060、config 257→261（+6⚠ 不变）、hw_specific 315→316、documented 705→706、verified 512→516、NEW 889→890、var:x 332→336
- 条目更正（回访补记，判定均不变）：10.87（#58824 修复 PR #58829 latch 方向获社区审查确认+测试补齐）、13.40（#58807 修复认领）
- 观察：**9.30 与 9.16/9.17 构成 Qwen M-RoPE 跨栈对偶**——SGLang fused kernel 内部数值错 vs vLLM 位置装配 offset 契约误解，两栈各自独立破坏"位置 id 精确"，且 9.30 在 stock 模板下全多模态请求静默中招；**13.41 是"SDPA 迁移回归"形态的掩码丢失**（单 flag 语义下沉后子类失传，与 10.86 Gemma1 GELU 守卫移除同谱系）；**6.259 把 6.204 显形谱系补全为三态**（Xid 崩引擎/静默错 state/spec acceptance 永久 0%），同一未合并修复 #55601 跨 Orin/x86 双硬件验证——"修复 open 未合并 + 发布带病"是该族当前最大暴露面；**10.88 使约束静默失效族横跨第五个独立机制**（10.40 整体丢弃/10.47 子集丢弃/10.64 allOf 不强制/10.73 门控不施加/本条准入检查被嵌套绕过），gpt-oss Harmony 全系默认路径命中
- 未覆盖范围：**arxiv 6 组查询全部失败（连续第二天）**——export.arxiv.org 对一切非单 token 查询持续 HTTP 406（诊断：单 token `all:electron` 200 正常；双词裸串/AND/引号短语/括号/任意字段前缀组合一律 406；`curl -fsSL` 跟随重定向、`%22`/`%20`/`+` 三种编码、多 UA 均同）——服务端查询解析层异常，**补查窗口 2026-09-19..09-27 顺延至下次扫描**；q07（tt-metal 143 命中）与 q02/q09 中 09-20..09-25 创建、既有判定条目未逐一重读正文（按既有判定跳过）；tt-metal 静态审计三批 meta-issue 维持整批留观未逐项展开；约 91 个来源仅经 search 快照确认无状态翻转（无评论级深查）
- 导出与校验：`python3 export_faults.py`（1229 行 CSV/XLSX）→ `python3 verify_catalog.py` **PASS 1229 条：正文汇总、CSV、XLSX、字段边界一致**
