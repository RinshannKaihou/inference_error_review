# LLM 推理故障每日扫描报告 — 2026-10-13

- **扫描日期**：2026-10-13（UTC）
- **扫描窗口**：GitHub `created:>=2026-10-06`（`cutoff=$(date -u -d '7 days ago' +%F)` 实际计算）；与 10-12 两轮扫描（早轮 created:>=2026-10-05、晚间轮 created:>=2026-10-12）的重叠区间按既有判定跳过，本日新判定集中于 10-12 晚间轮执行（23:01 UTC commit）后创建的 **31 个新 issue**；arxiv 6 组查询全部 200 有效
- **目标仓库**：vllm-project/vllm、sgl-project/sglang、huggingface/transformers、NVIDIA/TensorRT-LLM、triton-lang/triton、tenstorrent/tt-metal、flashinfer-ai/flashinfer
- **编目基线**：1362 条（2026-10-12 晚间轮后）；结构化编目 `catalog/entries/`（INF-0001–INF-1362）

## Git 前置检查（第 3 节）

- 分支 `main`、上游 `origin/main`、`git status --porcelain -uall` 为空 ✅
- `git fetch origin` 成功；`HEAD` = `origin/main` = `93fccca450a3fe469f83bbd7812500517d864df9` ✅ → 开始扫描

## API 调用记录

| # | 类型 | 查询 | 结果 |
|---|------|------|------|
| 1 | GH Search | `repo:vllm-project/vllm is:issue created:>=2026-10-06 correctness OR "silent wrong output" OR "wrong output"` | 30 命中（单页取全） |
| 2 | GH Search | `repo:vllm-project/vllm is:issue created:>=2026-10-06 KV OR quantization OR "kernel numerics" OR SDC` | 114 命中（两页取全 100+14） |
| 3 | GH Search | `repo:sgl-project/sglang is:issue created:>=2026-10-06 correctness OR "wrong output" OR silent OR garbage` | 43 命中（单页取全） |
| 4 | GH Search | `repo:huggingface/transformers is:issue created:>=2026-10-06 correctness OR "wrong output" OR numerics` | 3 命中（单页取全） |
| 5 | GH Search | `repo:NVIDIA/TensorRT-LLM is:issue created:>=2026-10-06 correctness OR "wrong result" OR silent OR numerics` | 11 命中（单页取全） |
| 6 | GH Search | `repo:triton-lang/triton is:issue created:>=2026-10-06 miscompile OR "wrong result" OR numerics OR correctness` | 3 命中（单页取全） |
| 7 | GH Search | `repo:tenstorrent/tt-metal is:issue created:>=2026-10-06 "Bad Outputs" OR correctness OR "wrong value"` | 43 命中（单页取全） |
| 8 | GH Search | `repo:flashinfer-ai/flashinfer is:issue created:>=2026-10-06 correctness OR "wrong output" OR numerics` | 18 命中（单页取全） |
| s1 | GH Search（补漏） | `repo:vllm-project/vllm is:issue label:bug created:>=2026-10-12` | 23 命中 |
| s2 | GH Search（补漏） | `repo:sgl-project/sglang is:issue created:>=2026-10-12 accurate OR accuracy OR degraded OR quality` | 10 命中 |
| s3 | GH Search（补漏） | `repo:tenstorrent/tt-metal is:issue label:bug created:>=2026-10-13` | 4 命中 |
| 12 | arxiv | `all:"silent data corruption" AND all:inference` | 200，5 命中全已编目（SProbe=2609.19743 等） |
| 13 | arxiv | `all:"bit flip" AND all:LLM` | 200，近月命中全已编目或已判（BARE-AI=2610.08739 即 2605.11202 v2、2609.38477 防御方法论文 10-12 已判不收录） |
| 14 | arxiv | `all:"KV cache" AND all:corruption` | 200，命中全已编目或方法论文不收录（Contiguity 2609.17983、Virtual Try-On 2609.25881 非推理域） |
| 15 | arxiv | `all:"soft error" AND all:GPU` | 200，3 命中已判（Carry-Through Checksum=2609.16742 CNN 边缘域、2601.19912 已编目） |
| 16 | arxiv | `all:"numerical correctness"` | 200，2610.06251（HQQ 量化早停）10-12 已判留观维持，2610.08675（Citation Swaps）非故障域 |
| 17 | arxiv | `all:"wrong output" AND all:LLM` | 200，0 命中 |

GitHub Search API 8 组主查询 + 3 组补漏全部有效、**分页完整（fetched==total 逐组核对：30/114(100+14)/43/3/11/3/43/18 + 23/10/4，`incomplete_results=false` 逐组核验）**、查询间 ≥7s 间隔、无限速 403。去重后 242 个唯一 issue；其中 31 个为上次扫描执行（10-12 23:01 UTC）后创建、全部逐一正文判定；补漏查询 s1–s3 额外覆盖 10-12 晚间轮查询词面之外的 9 个新 issue（#61490/#61372/#61359/#61341/#61337/#61315、sglang #44084/#44025/#43880/#43875/#43949/#43956、tt-metal #60484/#60398——其中 #61359/#60398 判定为新条目）。core REST 触发一次 403 限速（60/h 耗尽，~23:20 UTC），前台 sleep 70s 等重置后恢复，全部计划内深读完成（19 次 issue 详情 + 7 组评论 + 3 次 rate_limit 查询）；#61374/#61319/#43914 正文超过 Search 快照 2600 字符截断，经 GitHub 页面 HTML 内嵌 JSON 补全（合规路径，与 10-12 同款）。

## 🆕 新发现（本次编目，共 8 条：6 NEW + 2 var）

### vLLM

1. **INF-1363**（缺号无——注：本条为 sglang 来源，列于下方 SGLang 段）——见 3。
2. **INF-1365** [var:INF-1349] [vllm-project/vllm#61486](https://github.com/vllm-project/vllm/issues/61486) — Qwen4Exp AMD 路径 PLE n-gram ID 生成留在 PIECEWISE 图 piece 内：捕获期 dummy 请求数固化读取长度，重放读到未刷新的 query_start_loc/ngram_context 陈旧尾部并按其分配请求行；并发突发后 10 探针 0/10 一致（修复后 10/10），CPU 模型 46 步 94 处 ID 错。verified。与 6.281 共享 PIECEWISE 捕获契约缺口机制（地址固化 vs 缓冲长度/内容固化）。
3. **INF-1366** [var:INF-0328（4.28）] [vllm-project/vllm#61319](https://github.com/vllm-project/vllm/issues/61319) — LFM2-8B-A1B（hybrid conv+SSM+full-attention）temperature=0 两次 generate() 暖轮间输出分岔：长前缀+单 token 分岔+再对齐的非确定归约签名；禁 CUDA graph 仍在、dense-attention 对照稳定；内核未定位故 documented。与 4.28 共享 hybrid 前向归约非确定机制（批内多序列 vs 暖轮时间序，触发维度不同）。
4. **INF-1367** [NEW] [vllm-project/vllm#61359](https://github.com/vllm-project/vllm/issues/61359) — 请求带显式 truncation_side（即使与默认同值）时截断作用于加特殊 token 后的序列：BOS/[CLS]/[SEP]/EOS 被静默切掉（CPU 级单变量复现 [1,...] vs [278,...]），Cohere /v2/embed truncate=START 同路径。verified。

### SGLang

3. **INF-1363** [NEW] [sgl-project/sglang#43997](https://github.com/sgl-project/sglang/issues/43997)（closed/completed）— v0.5.21 HiCache host 层只建 KV+MAMBA 池不备份 DSA indexer 键，L2→L1 回载后 DSA 在不完整索引上计算：verbatim-recall host 腿 18/20 错 vs 冷/device 0/20；六臂对照（+INDEXER sidecar 0/20）归因闭合；main 已由 #40913/#40915 修复、v0.5.21 backport PR #43998 open。verified。
4. **INF-1364** [NEW] [sgl-project/sglang#44070](https://github.com/sgl-project/sglang/issues/44070) — PD prefill 早发缓存前缀（默认开）在 HiCache H2D 回载仍在写时把页发往 decode 侧且不再重发：同批共享 host-only 前缀的两请求 39/40（direct）29/30（kernel）损坏，开 checksum 全部被标记；关早发 0/84。verified。

### tt-metal

7. **INF-1368** [NEW] [tenstorrent/tt-metal#60393](https://github.com/tenstorrent/tt-metal/issues/60393) — chunk_gated_delta_rule chunk_size=64（Ct==2）时 decay_all 的 pack_tile 按调用顺序而非索引写 CB，dl 与 decayfac_1 错位：final state PCC 0.003；现有 chunk-64 测试只比设备路径互相之间、全路径共享缺陷恒绿；Qwen GDN 走 chunk 32 无已知生产调用方。documented。
8. **INF-1369** [NEW] [tenstorrent/tt-metal#60398](https://github.com/tenstorrent/tt-metal/issues/60398) — ttnn.gather rank>4 且 dim<rank-3 时合并-逆转置不匹配：输出形状正确内容乱序、无报错；codegen/native/forced native 三路同错（缺陷在共享准备层）。documented。
9. **INF-1370** [NEW]（补录）[tenstorrent/tt-metal#59828](https://github.com/tenstorrent/tt-metal/issues/59828)（closed 10-09）— reuse/DRAM-sharded/batched 三个 matmul 工厂 fp32 部分和经 SrcA 重载被 TF32 截断：误差随 K block 增长（2.47e-2 vs mcast 工厂 8.3e-5），系统性负偏置，含 LLM decode 形状（32x4096x1280）；mcast 工厂 #50250 已修、这三个漏改。**编目勘误**：10-09 日报 vLLM 拒收列表中的"#59828（CPU 采样 buffer 别名）"指 vLLM 侧同号 issue，本条 tt-metal#59828 此前从未被单独判定（当日 tt-metal 查询 q07 213 命中未逐一重读正文的既有未覆盖项），本轮补录。documented。

## 拒绝 / 留观（本日主要候选）

- **vLLM**：#61374（Thor SM110 MXFP4 MoE 后端选择回归 → Triton 编译 CompilationError 启动失败——fail-loud 崩溃域，wtdcode 已认领查看；关联史：#29317/#29339 曾把 SM110 从 Triton 路径排除、#58877 重新引入）、#61330（DFlash+gpt-oss FA3 illegal memory access——崩溃域）、#61337（DFlash drafter FLEX_ATTENTION 捕获 AssertionError——fail-loud）、#61487（HiSparse+DBO dbo_register_recv_hook NoneType——崩溃域；其中"workspace 只按 ubatch slot 0 预留"的隐患面留观）、#61373（TieringOffloadingSpec fs-tier lookup 成功但 promotion 永不触发——**性能/冗余重算域**，无错值证据，留观）、#61485（DiffusionGemma MI300/MI355 AITER 性能——纯性能）、#61389（MoRIIO P 侧 FINISHED_STOPPED 不导出 prefill 块——PD 编排边界，decode 侧失败形态未证实为静默错值，留观）、#61443（DFlash2 SWA draft 组漏标 is_eagle_group → block-drop 静默禁用+保守回退——性能优化静默不生效域，留观；与 60560 关联）、#61490（reset_prefix_cache 重置进行中 prefill 丢 prompt logprobs——非流式 200 静默 null 数组**是静默面**，但触发要求管理端点与 in-flight prefill 时序相交，主影响为可观测性损失，留观下轮优先评估）、#61318（return_sampling_mask 前置条件逐个报错——诊断质量域）、#61316（fp8 kernel selection 丢拒绝原因——诊断质量域）、#61372（Rust DeepSeek renderer 拒绝 latest_reminder 角色——400 fail-loud，双前端契约分歧留观）、#61317（jit_monitor_mode=error 杀引擎——fail-loud）、#61313/#61314/#61315（启动失败 fail-loud；61315 MoE max_lora_rank=320 断言在 MoE-LoRA 后端容量 vs config 允许值不调和，fail-loud）、#61341（垃圾帖 closed）、#61319 已编目 INF-1366、#61386 无此号、#60536（GLM-5.3-Flash JIT DeepGEMM tf32_hc_prenorm_gemm——性能域）、RFC/Roadmap/Feature（#60366/#60523/#60904/#60929/#61305/#61396/#61406/#61423/#61478）。
- **SGLang**：#43914（NVFP4 W4A4 GraniteMoE GB200 FlashInfer CUTLASS fused MoE 质量劣化 40–44/48 vs vLLM 45–46——机制未定位（MoE 路径嫌疑、dense FP4 已排除、无法试 TRT-LLM 后端因 routing_method_type 断言），作为 INF-0883（8.99 族）姊妹证据并入不占号，closed）、#44099（QSA MTP draft_tokens/topk 越界晚崩——fail-loud，但"topk==1 时静默改写 draft_tokens"语义留观）、#44025（NPU GPT-OSS FIA sink attention BSND 路径 prefill nextToken 断言失败——NPU 域 fail-loud；报告含 FP64 对照与 TND 修复方向，NPU 非主路径留观）、#43958（NPU Qwen3-ASR graph capture 失败——fail-loud）、#44083（ComfyUI server-mode 7 bug 批——插件域整批留观，修复 #44082 在途）、#43875（diffusion HTTP API 26 bug 批——同上留观）、#43880（W8A16 GEMV feature）、#43949（Monokernel RFC）、#43956（Qwen3.8-27B 分数随批形状变化 P(positive) 0.500→0.622——**batch shape 敏感数值漂移**，报告者自证非默认 batch-invariant 承诺、含 BF16 投影最小复现，与 4.28/INF-1366 族相邻；决策翻转证据真实但属"文档数值限制 vs 决策面"边界，留观下轮评估是否入族）。
- **tt-metal**：#60365（Gemma4/LLAMA-QB2 vLLM device sampling 静默钳 top_k≤32——模型类未声明 max_device_top_k 能力键，插件 #140 后已有正确路由、缺陷在模型侧声明缺失，vllm-tt-plugin 集成域**强候选**，证据充分（钳制代码行号+契约文档+修复先例），本日合并量控制留观下轮优先）、#60452（tt-train 3-tier scheduler_type 从不生效——训练侧 LR 恒定，训练编目域不占推理号）、#60484（n300 core placement 同 chip 上 CoreRangeSet overlap——CI 单 runner 硬件相关挂起域）、#60247（matmul bias h∈(1,M) padding 行入输出——10-12 已判留观维持，PR #59819 在途）、#60294（concat NOC burst 挂起——10-12 已判维持）、#60342 已编目 INF-1361（本轮回访 closed）。
- **flashinfer**：#6419（FP8 ragged prefill trtllm-native vs prims-ts refcheck 失配——两后端互相不匹配但未确定哪个错（rtol=0.2 宽容度下）、合成 benchmark 域，无 e2e 错值证据，留观）、#6348（CI 加固 closed——基建）。
- **TensorRT-LLM**：#20182（trtllm-bench 过滤 TRT-only 选项——bench 工具域）、#20236（beam search 每 beam 重物化 output_token_ids——性能）。
- **transformers/triton**：窗口内无新判定候选（10-12 判定维持）。
- **arxiv**：无新编目（见 API 表）。

## 来源回访（近 7 日新增条目引用来源 48 个逐一快照 + 活跃来源评论深查）

本日回访 48 个来源（10-12 两轮合并的 45 条引用来源 + 3 个旧 open 长期来源），全部经 Search API 单号查询取得 state/comments/updated 快照；其中 7 个活跃来源（评论增长或状态翻转）进一步做了评论级深查（core REST）：

- **状态翻转（2）**：tt-metal#60342（INF-1361，10.113）→ closed/completed 10-13T20:10Z，0 评论，已 update 记录（closed≠已发布修复）；tt-metal#59824（INF-1344 的 var 母体 1.296 关联）无变化。sgl#43331（10-12 已知 closed）维持。
- **活跃新证据（5）**：vllm#61225（INF-1353/8.100）两项独立复现（RTX 5090 op 级往返 K relerr 1.486；作者侧 RTX PRO 5000 写穿证明 scale/data 别名字节 512）→ update 补证据；vllm#60174（INF-0667/6.274）FP8 target 同现损坏 + TRITON_ATTN workaround 确认 + 修复 PR #55601 merged 未发布 → update；vllm#61100（INF-0672/6.279）第三方完整 checkpoint 无法复现 + 受控 A/B 提议 → update（复现性存疑、判定不变）；vllm#61229（INF-1350/6.282）维护者质疑人为触发 vs 报告者澄清心跳饥饿现实路径 → update（判定不变）；tt-metal#59828 bounty 邀请（jberkowitzTT 10-13）→ 并入 INF-1370。
- **评论级复查清欠（3）**：10-09/10-12 两轮遗留的旧 open 长期来源——vllm#58824（INF-1003/10.87，3 评论止于 09-29）、vllm#58973（INF-0343/4.27，1 评论止于 09-28）、sglang#41617（INF-0654/6.261，1 评论止于 09-29，修复 PR #41767 仍 open）——全部无新进展，已 update 记录。**该未了项本轮清零**。
- 其余 40 个来源快照确认无 state 翻转/评论数突变（10-12 合并条目引用来源全量覆盖）。

## 编目操作与校验

- 新增 8 条（`catalog_tool.py new`，全部 `pending_audit`）：INF-1363、INF-1364、INF-1365（var:INF-1349）、INF-1366（var:INF-0328）、INF-1367、INF-1368、INF-1369、INF-1370
- 更新 9 条（`update --note/--add-evidence`）：INF-1361（closed 记录）、INF-1353（独立复现证据）、INF-0667（workaround+修复状态）、INF-0672（复现性质疑）、INF-1350（维护者讨论）、INF-0654/INF-1003/INF-0343（评论级复查清欠）、INF-0883（#43914 姊妹证据）、INF-1351/INF-1355/INF-1356（快照确认）
- **操作事故记录（已修复）**：本轮一度把 #61225/#60174 的回访注误写入 INF-1349（6.281），2 分钟内发现并手工回滚该条目的错误 note 与 evidence（git diff 核对仅剩 migration.updates 残迹亦清除），证据已改写到正确条目 INF-1353/INF-0667；validate 通过。
- `python3 tools/catalog_tool.py build` → views/ 三文件再生成；`python3 tools/catalog_tool.py validate` → **ok:true，1370 条，0 errors**（43 warnings 为既有半受控 stack 提示，非本轮引入）
- 编目总数 **1362 → 1370**（active 1362 + pending_audit 8）

## 观察

1. **HiCache/PD 分离路径的"恢复未完成即消费"三连发**：INF-1363（host 回载缺 indexer 键）、INF-1364（H2D 写入途中早发）、10-12 的 6.277（host pool 元素/字节混淆）——三种独立机制都落在"KV 数据从一层搬往另一层时完整性契约缺失"，且全部静默。
2. **PIECEWISE 捕获契约缺口扩族**：INF-1365 与 6.281 构成"图按捕获期假设重放、运行期真实值无人刷新"的第二站（地址→缓冲内容）；NVIDIA 路径刻意把请求计数依赖放图外的对照设计值得写进修复方向。
3. **测试"路径互比"盲区再现**：INF-1368（chunk-64 测试只比设备路径互相之间、全路径共享缺陷恒绿）与 9.33/1.282 同族——"校验存在但校验错对象"主线在 tt-metal 连续第三周出现。
4. 10-09 空窗期 tt-metal 侧漏判的 #59828（同号 issue 跨仓库混淆所致）本轮补录，暴露"日报拒收列表按裸 #号记录"的歧义风险，后续日报记录拒收时带全仓库名。

## Git 提交

- 暂存：`catalog/entries/INF-1363..INF-1370.yaml`（新增 8）、`catalog/id_registry.tsv`、被 update 的 9 个条目 YAML、`views/CATALOG.zh.md`、`views/fault_catalog.csv`、`views/fault_catalog.xlsx`、`daily_scan_2026-10-13.md`
- 提交与推送结果见最终摘要（Git 操作在摘要前完成）。
