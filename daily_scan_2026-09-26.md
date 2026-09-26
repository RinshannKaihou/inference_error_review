# LLM 推理故障每日扫描报告 — 2026-09-26

- **扫描日期**：2026-09-26（UTC）
- **扫描窗口**：GitHub `created:>=2026-09-19`（7 日完整窗口，`cutoff=$(date -u -d '7 days ago' +%F)` 实际计算；与 09-25 扫描窗口 created:>=2026-09-18 全重叠——09-25 及更早判定条目全部按既有判定跳过，本日新判定集中于 09-25 扫描执行（约 09-25 22:30 UTC 前）之后创建的 issue：vLLM #58824/#58849/#58850/#58858（09-26 新建）及 09-25 后半未判定项 #58638/#58640/#58647/#58653/#58677/#58680/#58688/#58697/#58733、tt-metal #57988/#57989/#58000/#58001（09-26 新建）等）；arxiv 查询当日全部失败（见下）
- **目标仓库**：vllm-project/vllm、sgl-project/sglang、huggingface/transformers、NVIDIA/TensorRT-LLM、triton-lang/triton、tenstorrent/tt-metal、flashinfer-ai/flashinfer
- **主文档基线**：1218 条（2026-09-25 扫描后）；编号编目 1104 条 + K* 扩展 114 条

## Git 前置检查（第 3 节）

- 分支 `main`、上游 `origin/main`、`git status --porcelain -uall` 为空 ✅
- `git fetch origin main` 成功；`HEAD` = `origin/main` = `54b5edfef3b4d28a8761216ffe0f0b28290c9016` ✅ → 开始扫描

## API 调用记录

| # | 类型 | 查询 | 结果 |
|---|------|------|------|
| 1 | GH Search | `repo:vllm-project/vllm is:issue created:>=2026-09-19 wrong OR incorrect OR corruption OR NaN` | 41 命中（单页取全） |
| 2 | GH Search | `repo:vllm-project/vllm is:issue label:bug created:>=2026-09-19` | 58 命中（单页取全） |
| 3 | GH Search | `repo:sgl-project/sglang is:issue created:>=2026-09-19 wrong OR incorrect OR corruption OR NaN` | 18 命中 |
| 4 | GH Search | `repo:huggingface/transformers is:issue created:>=2026-09-19 wrong output OR incorrect OR NaN` | 6 命中 |
| 5 | GH Search | `repo:NVIDIA/TensorRT-LLM is:issue created:>=2026-09-19 wrong OR incorrect OR corruption` | 1 命中（#19501 RFC，09-24 已判） |
| 6 | GH Search | `repo:triton-lang/triton is:issue created:>=2026-09-19 wrong OR miscompile OR NaN` | 1 命中（#11889，09-24 已判） |
| 7 | GH Search | `repo:tenstorrent/tt-metal is:issue created:>=2026-09-19 numerical OR wrong OR pcc OR silent` | 143 命中（**两页取全 100+43**） |
| 8 | GH Search | `repo:flashinfer-ai/flashinfer is:issue created:>=2026-09-19 wrong OR incorrect OR NaN` | 4 命中（#5339/#5389/#5393/#5438 均已编目或已判） |
| 9 | GH Search | `repo:vllm-project/vllm is:issue created:>=2026-09-19 quantization OR "KV cache" OR deterministic` | 89 命中（单页取全） |
| 10 | GH Search | `repo:vllm-project/vllm is:issue created:>=2026-09-19 silent OR "bit flip" OR "weight corruption"` | 63 命中（单页取全） |
| 11 | GH Search | `repo:sgl-project/sglang is:issue label:bug created:>=2026-09-19` | 0 命中 |
| 12–17 | arxiv | 引号短语 + AND ×6：`"silent data corruption" AND LLM` / `"bit flip" AND LLM` / `"KV cache" AND corruption` / `" soft error" AND GPU AND inference` / `inference AND "numerical correctness"` / `inference AND "wrong output"` | **6 组全部 HTTP 406 失败**（见未覆盖范围） |

GitHub Search API 11 组查询全部有效、**分页完整（fetched == total 逐组核对：41/58/18/6/1/1/143(100+43)/4/89/63/0）**、无限流 403、无 Validation Failed；查询间 ≥8s 间隔。core REST 详情/评论/PR 状态 47 次调用全部成功（`--http1.1`，未认证额度 60/小时，本轮消耗 47+1 次 rate_limit 探测）。候选正文优先取自 Search API 缓存（响应自带 body），回访与新条目评论核对使用 core REST。**未认证运行（环境变量 TOKEN 经核验为 401 Bad credentials 无效凭据，未使用）。**

## 已编目/已判条目（本次跳过，重叠区去重）

- vLLM（09-25 及更早已判/已编目）：#58675（→9.29）、#58729（→7.99，评论见回访）、#58807（→13.40，评论见回访）、#58727（09-25 留观）、#58692（09-25 已判 fail-loud）、#58726/#58804（性能）、#58719/#58742（观测指标域）、#58751/#58774（RFC）、#58485（→11.87，评论见回访）、#58636（→4.26）、#58470（→6.256）、#57822（→6.248）、#58031（→5.44）、#58267/#58616（留观维持）及其余 09-19..09-24 判定项
- SGLang：#41192（→5.46，0 评论无变化）、#40155（→11.88，不在本窗口命中内）、#41038（→6.256）、#40966（→12.43）等 09-19..09-24 判定项全跳过
- transformers：#49051（→10.86，**已 closed** 见回访）、#49066（→10.85，维护者响应见回访）、#49036（留观）、#48947（→6.246）、#49026/#48954（CI triage）
- tt-metal（09-25 判定项全跳过）：#57847/#57848（→1.257）、#57852（→1.255）、#57861（→1.254）、#57867（→1.258）、#57869（→1.256）、#57873（→6.257）、#57889（→1.253）、#57827/#57835/#57853/#57856/#57899/#57902/#57920/#57923/#57929/#57948/#57950/#57954/#57955/#57965（09-25 已判维持）；静态审计三批 meta-issue（#57352–#57367/#57474–#57510/#57674–#57694）整批留观维持；#57088（→1.232，**已 closed** 见回访）
- flashinfer/triton/TRT-LLM：#5438（→7.96，0 评论无变化）、#5389/#5393 已判维持；triton/TRT-LLM 窗口无新增未判项

## 🆕 新发现（本次合并，共 6 条）

### A. 全新子签名（3 条）

1. **[新签名] tt-metal #57989 — Gemma4 dFlash drafter `_shard_argmax` 在 rows==1 时 `ttnn.reshape` 返回与 `sel_u` 共享设备 buffer 的视图，随后 `sel_u.deallocate(True)` 强制释放该 buffer——返回的 draft token id 从已释放内存读出、后续分配覆写** — `buffer_address()` 相等性证明 + 免模型最小重现（4 个 32×32 分配后 ids 被 7 覆盖）；rows≥4 时 reshape 拷贝故无害；触发 `GEMMA4_DFLASH_SHARD_ARGMAX=1` + TP>1 + 单行 drafter 批，默认配置不中。→ 归入第 1 类 **1.261**，Cov NEW（reshape 视图别名 × 强制 deallocate 生命周期断裂），documented。
2. **[新签名] vLLM #58858 — GLM-5.3-Flash kpool indexer 在 ROCm 上：存储块 128 不整除 kernel 块 640 时 builder 的 `// factor` 表转换被完全跳过，640-token 表被 kpool writer/gather 按 32-pool 页号解释——gfx1030 实测 74 页仅 3 页正确（全零页+错 keys 页遍布），页粒度修正表 74/74 全对** — NVIDIA 侧 kernel 块 64 整除分支命中免疫；gfx942/gfx950 共享路径疑似（无硬件确认，报告者请求社区核实）。→ 归入第 8 类 **8.94**，Cov var:8.83（GLM-5.3-Flash indexer 寻址族第三形态：非整除 else 分支无人处理），documented。
3. **[新签名] vLLM #58824 — `Llama3JsonToolParser.extract_tool_calls_streaming` 仅凭首字符 `startswith("{")` 判定 tool call 且永不回退：以 `{` 开头的普通 JSON 回答在流式路径整段消失（content=None、零调用），非流式正常返回；空参数调用流式 `arguments=''`；调用前文字流式全量当 content 零调用** — CPU + tokenizer 纯 parser 复现；修复 #58829（latch）已开源未合并，社区审查指出 latch 不复位缺口。→ 归入第 10 类 **10.87**，Cov var:10.42（parser 首标记假设族——流式/非流式判定算法分裂维度），verified（parser-only 复现因果闭合）。

### B. 已知类别新变体（3 条）

4. **[已知类别] tt-metal #58000 — `ttnn.floor_div(INT32 tensor, scalar)` 复合路径无整数实现：截断 float 倒数乘法使 `|s|>1` 全元素归零 + `ttnn.floor` 无校验接受 INT32 对原始整数 bits 跑 float SFPU（正数 denormal flush 0、负数 NaN 穿过）——s=2/3/7/−2 全域 100% 错、s=1 仍 ~50% 错；同输入 `rounding_mode="floor"` 0% 错（正确实现就在旁边未被复用）** — 判决例 `[7,-7,100,INT32_MAX,...]÷2` → 全 0。→ 归入第 1 类 **1.259**，Cov var:1.123（floor_div 族：整除边界差一 → 全域错误），verified（实测矩阵+源码双机制）。
5. **[已知类别] tt-metal #58001 — `ttnn.typecast` 整数源（uint16/int32/uint32）→ bf16 经硬件 rounder tie **away from zero**，同 op 的 fp32→bf16 臂却是软件 RNE——同一算子按源 dtype 施加两种舍入规则；[256,65535] 穷举 1024 tie 中 512 背离 torch，[256,512) 内每个奇数都是 tie、该区间 1/4 整数错** — fp32 臂（#46231 软件 RNE）自证正确做法存在于同文件。→ 归入第 1 类 **1.260**，Cov var:1.250（typecast 舍入族——tie-break 方向分裂维度），documented。
6. **[已知类别] vLLM #58653 — hybrid GDN + `VLLM_USE_SIMPLE_KV_OFFLOAD=1` + 显式不对齐 `--block-size` 使 SimpleCPU 卸载退化为只写不读：CPU RSS 按写路径增长、`external_kv_transfer` 恒 0、逐出后 re-ask T3/T1=1.006–1.46（≈全冷）；省略 flag 自动对齐即完全正常；无错误无警告** — 两独立生产部署同 wheel 同病；启动日志 `Setting attention block size to 896...` 自证对齐重算。→ 归入第 6 类 **6.258**，Cov var:6.227（offload 配置静默失效族——半边失效维度），documented。

### C. 评估后不收录（记录备查）

- **vLLM #58677**（draft compile cache key 随启动史在两值间翻转）：根因已被复现者定位——v0.27.1 `CompilationConfig.enabled/disabled_custom_ops` 计数器入哈希、全编译启动时 backbone fusion pass 会 bump 而暖启动不 bump；v0.29.0 经 #53378 已把两计数器加入 ignored_factors（1×A100 复现验证）。纯启动编译开销（~196s/次重编译），无输出错误。不收录。
- **vLLM #58688**（ROCm Qwen3.8-Flash-Next-FP8 `ngram_embedding.weight_scale` 加载 ValueError）：fail-loud 启动失败；#54722/#54882 只修 NVIDIA 路径、AMD `ple_layer.py` 无 FP8 支持——发布/移植缺口。不收录。
- **vLLM #58733**（DeepSeek-V4-Flash + DFlash drafter 启动双态失败：fp8_ds_mla 无 backend / hidden 16384 vs 4096 不匹配）：fail-loud 启动 crash（`--load-format dummy` 可复现）。不收录。
- **vLLM #58638**（KV cache grouping for hybrid + drafters RFC）：RFC 议案（#58207/#58380 性能与容量设计）。不收录。
- **vLLM #58697**（Shared JSON Schema structure RFC）：RFC；其引用的 #58694/#58695（Guidance gate 误判 schema 关键字）不在本窗口搜索命中内，未逐项核。留观。
- **vLLM #58849**（WSL2 pin memory 使 V2 runner 每步快 ~12%）：纯性能。不收录。
- **vLLM #58850**（pynccl all_reduce CUDA graph 捕获期间歇 Xid 31 MMU fault，~1/3 启动率，成功后稳定服务数天）：fail-loud 启动崩溃域（SM120 6×RTX PRO 6000 + Ray PP3/TP2 GLM-5.2-NVFP4）；无静默错值证据。与 6.248（生产流量 Xid 三连崩）同 Xid 谱系但触发面不同（启动捕获期 vs 生产流量），未定位前留观。
- **vLLM #58640**（V4.1 DSML 前导空格破坏 V4 parser：状态机不识别标签 + 参数正则不匹配 + arg delta 过滤吞增量——tool call 参数无法流式返回）：真实流式丢失面，但 willweimike 已认领在修、修复方向明确（V4.1 专属 parser 子类），10.55/10.75 DSML parser 族语境补充，待修复 PR 落地后并入族条目不另占编号。留观。
- **vLLM #58680**（Rust frontend top_logprobs 重复采样 token + prompt+max_tokens 超限静默接受截断）：API 契约域双缺陷——logprobs 元数据错误真实（采样 token 计入 top-k 使备选少一个）+ 静默截断输出预算，Python 前端为对照；非生成内容错值。留观（Rust 前端默认关闭，实验特性）。
- **vLLM #58647**（Anthropic /v1/messages hardening RFC）：09-25 已判（#58727 拆分母体），维持。
- **tt-metal #57988**（slice step≤0：负 step 静默空张量 S1 + step=0 host SIGFPE S2）：S1 静默空结果真实且 ttsim 已复现（报告者跟评：ttnn 0.79.0 release wheel + Wormhole ttsim，`x[..., ::-1]` 返 (1,1,32,0) 无错、`::0` 杀进程）；属"防御缺失→静默空结果"域，1.243 slice 族语境补充。留观（待上游修复方向明确后定）。
- **SGLang #41192**：已编目 5.46；0 评论、无状态变化。
- **arxiv**：6 组查询当日全部失败（HTTP 406），无命中可判——见未覆盖范围。

## 来源回访（近 7 日新增条目引用来源 + 额度轮巡旧来源）

本日回访 23 个来源（core REST 详情+评论全读 13 个、仅状态/快照 10 个）：

- **1.232（tt-metal #57088）**：状态 **closed/completed（2026-09-20）**、0 评论。closed 事实不证明评分界已按模型深度解析——**条目来源行补回访记**（判定与触发面不变，剩余状态待后续 PR 核实）。
- **10.86（transformers #49051）**：状态 **closed/completed（2026-09-25）**，6 条评论全读——维护者 vasqu 确认修复方向（`GemmaConfig.__post_init__` 恢复 `gelu→gelu_pytorch_tanh` 映射并 warn，覆盖 Hub/from_dict 全人群）；报告者以四树 ×CPU/GPU ×F32/BF16 十二腿 verifier 实测两个竞速 PR；Cyrilvallez 关闭两个 AI bot PR、由报告者提 PR。**closed+修复方向确认≠已发布**，含修复 release 待查。条目来源行补回访记。
- **10.85（transformers #49066）**：2 条评论（09-24/09-25）——独立复核者以 Qwen2.5-0.5B + 补丁 `\p{M}` 模式证明无需 gated 模型即可复现，且 ID 分歧 vocab 相关（阿拉伯语捕获、泰文同错、天城文该 vocab 不发散）、5.16.1 同病（v5 全系）；**维护者 itazap 响应：常量覆盖是有意设计（v5 以 tokenization_<model>.py 为 ground truth），修复走 tokenizer 映射表使该族不再映射到 TokenizersBackend，PR 在途**。判定不变（走 TokenizersBackend 路径仍静默错），条目来源行补回访记。
- **11.87（vLLM #58485）**：6 条评论（09-24..09-26）——amittell 09-26 澄清：#52677 只修 V2（V1 选择现经 `_get_v1_model_runner_unsupported_features` 显式 raise）、**bug 在 V1 仍存在**（`thinking_budget_state.py` 自 2026-08-15 未变）；MRV1 仍可经 7 类 spec 方法（ngram/draft_model/medusa 等）+ EAGLE parallel drafting + `mamba_cache_mode='all'` + ROCm DSA（DeepSeek-V3.2/V4/GLM-MoE 默认 V1）到达；njhill 待决 won't-fix。**11.87 触发面表述已含 V1 边界，判定不变**。
- **7.99（vLLM #58729）**：1 条新评论（09-26 baonudesifeizhai）——**B300 复现**（非 GB200）：Cutlass MoE 下 2+2 答成 "What is2+"重复循环；仅 MoE 路径回退 Marlin 即正确（"4\n"）；逐层对照：layers 0–2 逐位一致、**首个分歧恰在 model.layers.3.mlp——first_k_dense_replace=3 后第一个 MoE 层**，router 前输入一致、专家输出分歧、无 NaN/Inf；评论者给根因假设 `CutlassExpertsMxfp4 only allow ep_size==1`。根因显著收窄（量化 MoE 内核位面），判定维持 documented/ambiguous 待维护者确认。
- **13.40（vLLM #58807）**：1 条新评论（09-26 he-yufeng）——认领修复：检测按 checkpoint 实际 tensor 名（safetensors index 里 mtp.* 是否带量化后缀）决定 predictor 整体量化与否，镜像 GPTQ dynamic/modelopt_fp4 先例；单文件 checkpoint 无 index 保持现行为；PR 即将附红绿测试。修复在途，条目不动。
- **1.244（tt-metal #57413）**：1 条新评论（09-26 Repo Assist）——确认 6-entry tanh LUT 为 bfloat16 ULP 最优拟合的真数值回归而非测试抖动（表注释自证以最小化 bf16 ULP 误差为目标重拟合，fp32 max-abs 0.018962→0.018840 略优而四样本 BERTScore 门恰在阈值附近被移过 0.69）；修复方向（重调 LUT vs 回退）留维护者。判定不变。
- **1.247（tt-metal #57461）**：修复 PR **#57463 仍 open 未合并**（昨日遗留项本轮经 core REST 核实：state=open、merged=False）。条目不动。
- **1.252（tt-metal #57732）**：Repo Assist 09-25 已开修复 PR（首 DRAM-bank 读钳制到 per_core_N + 回归测试，CI 在跑）；issue open。未合并前维持。
- **1.257b（tt-metal #57848）**：Repo Assist 确认根因（`u = a + 1.0f` 吞 −0.0 符号）并开修复（照 #57554 log 族模式，三 arch + copysign 回归测试）；因 protected file 走 fallback review issue #57946。在途。
- **1.258（tt-metal #57867）**：Repo Assist 确认（addcdiv_bw 已先折叠 value、addcmul_bw 后乘）并开修复 PR（乘法重排 + overflow 回归测试）。在途。
- **6.257（tt-metal #57873）**：issue 正文 09-26 有编辑（补充机理段落：指向 batched eager prefill 的 per-request sliding-tail stash 与 continuation chunk 的 bounded ring 写位置两个候选位点）+ 0 评论。与条目"根因区间已收窄、位点未定位"的记载一致，无需改。
- **6.248/#57822、4.26/#58636、5.45/#40921、5.44/#58031、6.256/#58470、9.29/#58675**：经 search 快照确认无 closed/reopened 翻转、评论数不变（#57822 4、#58636 2、#40921 2、#58031 9、#58470 0、#58675 0），无新机制证据。
- **5.46（SGLang #41192）**：0 评论、open 不变（编目后无进展）。
- 未检查完说明：近 7 日新增条目引用来源约 97 个（09-19..09-25 合并 83 条 + 09-26 新增 6 条 + 勘误），本日按 core 额度（未认证 60/小时，本轮消耗 47 次 + 1 探测）回访上述 23 个高价值来源（昨日 12 条新合并主来源全量 + 3 个 closed 翻转 + 4 个修复 PR 状态 + 根因收窄项）；其余约 74 个经本日 search 快照确认无状态翻转（q01–q11 覆盖窗口内全部活跃 issue 的 state/comments 字段），评论级深查缺口滚动至后续扫描。

## 合并动作摘要

- 主文档新增 **6 条**（3 个新子签名 + 3 个已知类别变体）：1.259、1.260、1.261、6.258、8.94、10.87
- 总条数 1218 → **1224**；编号编目 1104 → **1110**（K* 扩展 114 不变）
- 分类分布：第 1 类 257→260、第 6 类 257→258、第 8 类 93→94、第 10 类 86→87
- 轴分布更新（按解析器实测）：infra_off 1049→1055、hw_specific 304→315 → 311→315（第 1 类 +4 hw_specific、第 8 类 +1）、config 263（含 6⚠ 不变）、documented 701→705、verified 509→512 → 510→512、NEW 888→889、var:x 327→332
- 条目更正（回访补记，判定均不变）：1.232（#57088 closed/completed）、10.86（#49051 closed/completed + 修复方向确认）、10.85（#49066 维护者响应：常量覆盖有意、修复走映射表）
- 观察：**1.259/1.260 与 1.190/1.250 构成"整数 bit 型被浮点内核重解释"四连**（正溢出饱和 / 双舍入 bit31 / 截断倒数全零 / tie 方向分裂）——tt-metal eltwise 整数域的 dtype 契约松弛是系统性面；**8.94 是 GLM-5.3-Flash kpool 寻址族第三形态**（8.73 top-k 宽溢出 → 8.83 粒度断裂 → 8.94 非整除守卫 else 分支无人处理），"整除才转换"的守卫在非整除世界 silently no-op；**10.87 首次记录同一 parser 流式/非流式判定算法分裂**（首字符启发式 vs 完整 JSON 解析）；1.261 是 spec draft 路径首个 UAF 形态（强制 deallocate 绕过共享引用检测）
- 未覆盖范围：**arxiv 6 组查询全部失败**——export.arxiv.org 对一切含引号短语/AND 的复合查询持续返回 HTTP 406（诊断：单 token 查询 `all:electron` 200 正常、任何引号/多词/AND 形态 406；curl/urllib/http.client × 多 UA × `%22`/`%20`/`+` 编码 × 20-30s 退避重试均同；昨日同组查询成功，疑服务端查询解析层变更或代理路径差异），**补查窗口 2026-09-19..09-26 顺延至下次扫描**；q07（tt-metal 143 命中）与 q02/q09 中 09-19..09-24 创建、既有判定条目未逐一重读正文（按既有判定跳过）；tt-metal 静态审计三批 meta-issue 维持整批留观未逐项展开；约 74 个来源仅经 search 快照确认无状态翻转（无评论级深查）；#58697 RFC 引用的 #58694/#58695 不在窗口命中内未核
- 导出与校验：`python3 export_faults.py`（1224 行 CSV/XLSX）→ `python3 verify_catalog.py` **PASS 1224 条：正文汇总、CSV、XLSX、字段边界一致**
